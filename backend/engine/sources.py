"""Pinned third-party sources (`vendor/sources.lock.json`).

The pool pins an engine release and attests what it ran, so every source the
engine builds or loads has to be reproducible from a commit: the vendored
submodules (whose gitlink must equal the lock) and the repositories cloned at
install time. Loading is standard-library only, because the isolated refiner
runtime's installer reads the same file.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "vendor" / "sources.lock.json"


class SourceError(RuntimeError):
    """A pinned source is missing, unpinned or checked out at the wrong commit."""


@dataclass(frozen=True)
class PinnedSource:
    name: str
    repository: str
    commit: str
    licence: str
    used_by: str
    vendor_path: str | None = None
    ref: str | None = None


def load_sources(lock_path: Path = LOCK_PATH) -> dict[str, PinnedSource]:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SourceError(f"Unsupported sources lock schema: {data.get('schema_version')!r}")
    sources: dict[str, PinnedSource] = {}
    for entry in data["sources"]:
        source = PinnedSource(
            name=entry["name"],
            repository=entry["repository"],
            commit=entry["commit"],
            licence=entry["licence"],
            used_by=entry["used_by"],
            vendor_path=entry.get("vendor_path"),
            ref=entry.get("ref"),
        )
        if len(source.commit) != 40 or any(c not in "0123456789abcdef" for c in source.commit):
            raise SourceError(f"{source.name}: commit must be a full 40-character SHA")
        if source.name in sources:
            raise SourceError(f"Duplicate source {source.name!r}")
        sources[source.name] = source
    return sources


def source(name: str, lock_path: Path = LOCK_PATH) -> PinnedSource:
    try:
        return load_sources(lock_path)[name]
    except KeyError:
        raise SourceError(f"{name!r} is not in {lock_path.name}") from None


def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise SourceError(f"git {' '.join(args)} failed: {(completed.stdout + completed.stderr)[-800:]}")
    return completed.stdout.strip()


def require_vendored(name: str, root: Path = ROOT, lock_path: Path = LOCK_PATH) -> Path:
    """Return a vendored submodule's directory, checked against the lock.

    An empty directory means the submodule was never initialised. A checkout
    with its own Git metadata must sit at exactly the pinned commit.
    """
    pinned = source(name, lock_path)
    if not pinned.vendor_path:
        raise SourceError(f"{name} is not vendored")
    directory = root / pinned.vendor_path
    if not directory.is_dir() or not any(directory.iterdir()):
        raise SourceError(
            f"{pinned.vendor_path} is empty. Run: git submodule update --init {pinned.vendor_path}"
        )
    if (directory / ".git").exists():
        head = _git(["rev-parse", "HEAD"], directory)
        if head != pinned.commit:
            raise SourceError(
                f"{pinned.vendor_path} is at {head[:12]}, but the lock pins {pinned.commit[:12]}. "
                f"Run: git submodule update --init {pinned.vendor_path}"
            )
    return directory


def clone_pinned(name: str, destination: Path, lock_path: Path = LOCK_PATH) -> PinnedSource:
    """Clone a pinned repository and check it out detached at the locked commit."""
    pinned = source(name, lock_path)
    if destination.exists():
        raise SourceError(f"{destination} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", "--filter=blob:none", "--no-checkout", pinned.repository, str(destination)], destination.parent)
    _git(["checkout", "--detach", pinned.commit], destination)
    head = _git(["rev-parse", "HEAD"], destination)
    if head != pinned.commit:
        raise SourceError(f"{name}: checked out {head[:12]}, expected {pinned.commit[:12]}")
    return pinned
