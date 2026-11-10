"""Pinned model weights (`vendor/weights.lock.json`): install once, verify offline.

The pool runs the engine inside a hardened runtime with no general egress, so
stages never download. Weights are installed ahead of time at pinned
revisions (`install`), and every run checks them (`verify`) against the lock:
presence and size always, content hashes when `full` is set or when a cached
hash still matches the file's size and mtime.

A file whose lock entry has no sha256 yet (DINOv3 is gated, and rembg pins its
ONNX files by md5) verifies as `unpinned`, with the hash observed on disk
reported so it can be recorded. A runtime with any unpinned file is not
`pinned`, and the pool must not attest it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "vendor" / "weights.lock.json"
ROOT_NAMES = ("trellis_models", "unirig", "u2net")
_CHUNK = 8 * 1024 * 1024


class WeightsError(RuntimeError):
    """The lock is malformed, or an install could not produce a verified file."""


@dataclass(frozen=True)
class WeightFile:
    dest: str
    size: int
    sha256: str | None
    md5: str | None = None
    path: str | None = None  # path inside a Hugging Face repository
    url: str | None = None


@dataclass(frozen=True)
class WeightSet:
    id: str
    source: str
    root: str
    licence: str
    stage: str
    gated: bool
    files: tuple[WeightFile, ...]
    repository: str | None = None
    revision: str | None = None


@dataclass
class FileCheck:
    weight_id: str
    dest: str
    status: str  # ok | missing | size_mismatch | hash_mismatch | unverified | unpinned
    observed_sha256: str | None = None


@dataclass
class VerificationReport:
    checks: list[FileCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every file is present and no check failed (unpinned files allowed)."""
        return all(c.status in ("ok", "unpinned", "unverified") for c in self.checks)

    @property
    def pinned(self) -> bool:
        """Every file matched a locked hash. Required before the pool attests a runtime."""
        return bool(self.checks) and all(c.status == "ok" for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "pinned": self.pinned,
            "files": [c.__dict__ for c in self.checks],
        }


def load_weights(lock_path: Path = LOCK_PATH) -> dict[str, WeightSet]:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise WeightsError(f"Unsupported weights lock schema: {data.get('schema_version')!r}")
    sets: dict[str, WeightSet] = {}
    for entry in data["weights"]:
        if entry["root"] not in ROOT_NAMES:
            raise WeightsError(f"{entry['id']}: unknown root {entry['root']!r}")
        if entry["source"] == "huggingface" and len(entry.get("revision") or "") != 40:
            raise WeightsError(f"{entry['id']}: a Hugging Face source needs a full commit revision")
        files = tuple(
            WeightFile(
                dest=f["dest"], size=int(f["size"]), sha256=f.get("sha256"), md5=f.get("md5"),
                path=f.get("path"), url=f.get("url"),
            )
            for f in entry["files"]
        )
        for weight in files:
            if Path(weight.dest).is_absolute() or ".." in Path(weight.dest).parts:
                raise WeightsError(f"{entry['id']}: unsafe destination {weight.dest!r}")
        sets[entry["id"]] = WeightSet(
            id=entry["id"], source=entry["source"], root=entry["root"], licence=entry["licence"],
            stage=entry["stage"], gated=bool(entry.get("gated")), files=files,
            repository=entry.get("repository"), revision=entry.get("revision"),
        )
    return sets


def _digests(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def _load_cache(cache_path: Path | None) -> dict:
    if cache_path and cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def verify(
    roots: dict[str, Path],
    ids: list[str] | None = None,
    *,
    full: bool = False,
    cache_path: Path | None = None,
    lock_path: Path = LOCK_PATH,
) -> VerificationReport:
    """Check installed weights against the lock without touching the network."""
    sets = load_weights(lock_path)
    selected = [sets[i] for i in ids] if ids else list(sets.values())
    cache = _load_cache(cache_path)
    report = VerificationReport()
    for weight_set in selected:
        base = roots[weight_set.root]
        for weight in weight_set.files:
            target = base / weight.dest
            if not target.is_file():
                report.checks.append(FileCheck(weight_set.id, weight.dest, "missing"))
                continue
            stat = target.stat()
            if stat.st_size != weight.size:
                report.checks.append(FileCheck(weight_set.id, weight.dest, "size_mismatch"))
                continue
            key = str(target.resolve())
            cached = cache.get(key)
            # `full` always rehashes: a cache keyed on size and mtime cannot see a
            # file rewritten in place with its old mtime restored.
            if not full and cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
                sha, md5 = cached["sha256"], cached["md5"]
            elif full:
                sha, md5 = _digests(target)
                cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha, "md5": md5}
            else:
                report.checks.append(FileCheck(weight_set.id, weight.dest, "unverified"))
                continue
            if weight.sha256:
                status = "ok" if sha == weight.sha256 else "hash_mismatch"
            elif weight.md5 and md5 != weight.md5:
                status = "hash_mismatch"
            else:
                status = "unpinned"
            report.checks.append(FileCheck(weight_set.id, weight.dest, status, observed_sha256=sha))
    if cache_path and full:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return report


def _download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "aismith-engine"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK)


def install(
    roots: dict[str, Path],
    ids: list[str],
    *,
    token: str | None = None,
    cache_path: Path | None = None,
    lock_path: Path = LOCK_PATH,
) -> VerificationReport:
    """Fetch weights at their pinned revisions, then verify them in full.

    Network access happens only here, never in a stage. A downloaded file that
    fails its locked hash is removed and the install fails.
    """
    sets = load_weights(lock_path)
    for weight_id in ids:
        weight_set = sets[weight_id]
        base = roots[weight_set.root]
        for weight in weight_set.files:
            target = base / weight.dest
            if target.is_file() and target.stat().st_size == weight.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
                if weight_set.source == "huggingface":
                    from huggingface_hub import hf_hub_download

                    fetched = Path(hf_hub_download(
                        repo_id=weight_set.repository, filename=weight.path,
                        revision=weight_set.revision, local_dir=scratch, token=token,
                    ))
                elif weight_set.source == "url":
                    fetched = Path(scratch) / "download"
                    _download_url(weight.url, fetched)
                else:
                    raise WeightsError(f"{weight_id}: unknown source {weight_set.source!r}")
                os.replace(fetched, target)
    report = verify(roots, ids, full=True, cache_path=cache_path, lock_path=lock_path)
    bad = [c for c in report.checks if c.status not in ("ok", "unpinned")]
    for check in bad:
        (roots[sets[check.weight_id].root] / check.dest).unlink(missing_ok=True)
    if bad:
        raise WeightsError("Installed weights failed verification: " + ", ".join(f"{c.dest} ({c.status})" for c in bad))
    return report


def default_cache_path(roots: dict[str, Path]) -> Path:
    """Where verified hashes are cached, so per-job checks need not rehash."""
    return roots["trellis_models"] / ".weights-verified.json"


def offline_environment(roots: dict[str, Path]) -> dict[str, str]:
    """Environment for stage subprocesses: no hub access, weights from `roots`."""
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "U2NET_HOME": str(roots["u2net"]),
    }


def _parse_roots(values: list[str]) -> dict[str, Path]:
    from backend.app import native

    roots = native.engine_weight_roots()
    for value in values:
        name, _, path = value.partition("=")
        if name not in ROOT_NAMES or not path:
            raise SystemExit(f"--root expects one of {', '.join(ROOT_NAMES)} as name=path, got {value!r}")
        roots[name] = Path(path).expanduser().resolve()
    return roots


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m backend.engine.weights", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "verify"):
        command = sub.add_parser(name)
        command.add_argument("ids", nargs="*", help="weight set ids (default: all)")
        command.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
        if name == "verify":
            command.add_argument("--full", action="store_true", help="hash every file, not only cached ones")
    args = parser.parse_args(argv)
    roots = _parse_roots(args.root)
    ids = args.ids or list(load_weights())
    if args.command == "install":
        report = install(roots, ids, token=os.getenv("HF_TOKEN") or None, cache_path=default_cache_path(roots))
    else:
        report = verify(roots, ids, full=args.full, cache_path=default_cache_path(roots))
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
