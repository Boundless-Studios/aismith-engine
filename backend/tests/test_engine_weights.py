"""The weights lock: pinned revisions and hashes, verified without the network."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app import native
from backend.engine import weights


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_repository_lock_is_complete():
    sets = weights.load_weights()
    assert {"trellis2-4b-fp8", "dinov3-vitl16", "trellis-image-large-ss-decoder", "unirig-skeleton-ar256",
            "birefnet-general", "birefnet-general-lite"} <= set(sets)
    for weight_set in sets.values():
        assert weight_set.licence and weight_set.files
        for weight in weight_set.files:
            assert weight.size > 0
            # Only gated files (no public hash) and rembg's md5-pinned files lack a sha256.
            if not weight.sha256:
                assert weight_set.gated or weight.md5, (weight_set.id, weight.dest)


def test_lock_covers_every_file_the_refiner_requires(tmp_path):
    roots = {"trellis_models": tmp_path / "models", "unirig": tmp_path / "unirig", "u2net": tmp_path / "u2net"}
    locked = {roots[s.root] / f.dest for s in weights.load_weights().values() for f in s.files}
    with patch.object(native, "trellis_refiner_dir", return_value=tmp_path):
        required = [
            tmp_path / "models" / "visualbruno" / "TRELLIS.2-4B-FP8" / "pipeline_fp8.json",
            tmp_path / "models" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m" / "model.safetensors",
            tmp_path / "models" / "microsoft" / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16.safetensors",
        ]
    assert set(required) <= locked


def _lock(tmp_path: Path, files: list[dict], *, source="url", gated=False) -> Path:
    lock = tmp_path / "weights.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "weights": [{
        "id": "w", "source": source, "root": "u2net", "licence": "MIT", "stage": "mesh", "gated": gated,
        "files": files, **({"repository": "org/repo", "revision": "a" * 40} if source == "huggingface" else {}),
    }]}))
    return lock


def test_verify_statuses_and_cache(tmp_path):
    root = tmp_path / "u2net"
    root.mkdir()
    good, wrong = b"weights", b"WEIGHTS"
    (root / "ok.bin").write_bytes(good)
    (root / "bad.bin").write_bytes(wrong)
    (root / "short.bin").write_bytes(b"x")
    lock = _lock(tmp_path, [
        {"dest": "ok.bin", "size": len(good), "sha256": _sha(good)},
        {"dest": "bad.bin", "size": len(wrong), "sha256": _sha(good)},
        {"dest": "short.bin", "size": 5, "sha256": _sha(good)},
        {"dest": "gone.bin", "size": 1, "sha256": _sha(good)},
    ])
    roots = {"u2net": root}
    cache = tmp_path / "cache.json"

    quick = weights.verify(roots, cache_path=cache, lock_path=lock)
    assert [c.status for c in quick.checks] == ["unverified", "unverified", "size_mismatch", "missing"]

    full = weights.verify(roots, full=True, cache_path=cache, lock_path=lock)
    assert [c.status for c in full.checks] == ["ok", "hash_mismatch", "size_mismatch", "missing"]
    assert not full.ok and not full.pinned

    # A cached hash still matching size and mtime counts without rehashing.
    with patch.object(weights, "_digests", side_effect=AssertionError("rehashed")):
        again = weights.verify(roots, cache_path=cache, lock_path=lock)
    assert [c.status for c in again.checks][:2] == ["ok", "hash_mismatch"]


def test_md5_only_and_gated_files_verify_as_unpinned(tmp_path):
    root = tmp_path / "u2net"
    root.mkdir()
    data = b"onnx"
    (root / "m.onnx").write_bytes(data)
    lock = _lock(tmp_path, [{"dest": "m.onnx", "size": len(data), "md5": hashlib.md5(data).hexdigest(), "sha256": None}])
    report = weights.verify({"u2net": root}, full=True, lock_path=lock)
    assert report.checks[0].status == "unpinned"
    assert report.checks[0].observed_sha256 == _sha(data)
    assert report.ok and not report.pinned

    lock = _lock(tmp_path, [{"dest": "m.onnx", "size": len(data), "md5": "0" * 32, "sha256": None}])
    assert weights.verify({"u2net": root}, full=True, lock_path=lock).checks[0].status == "hash_mismatch"


def test_lock_rejects_unpinned_revisions_and_unsafe_paths(tmp_path):
    lock = tmp_path / "weights.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "weights": [{
        "id": "w", "source": "huggingface", "repository": "org/repo", "revision": "main", "root": "u2net",
        "licence": "MIT", "stage": "mesh", "files": [{"dest": "a", "path": "a", "size": 1, "sha256": None}]}]}))
    with pytest.raises(weights.WeightsError, match="full commit"):
        weights.load_weights(lock)
    lock = _lock(tmp_path, [{"dest": "../escape", "size": 1, "sha256": None}])
    with pytest.raises(weights.WeightsError, match="unsafe"):
        weights.load_weights(lock)


def test_install_from_url_verifies_and_removes_a_bad_download(tmp_path):
    payload = b"model bytes"
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(payload)
    root = tmp_path / "u2net"
    lock = _lock(tmp_path, [{"dest": "m.bin", "url": upstream.as_uri(), "size": len(payload), "sha256": _sha(payload)}])
    report = weights.install({"u2net": root}, ["w"], lock_path=lock)
    assert report.pinned and (root / "m.bin").read_bytes() == payload

    (root / "m.bin").unlink()
    lock = _lock(tmp_path, [{"dest": "m.bin", "url": upstream.as_uri(), "size": len(payload), "sha256": "0" * 64}])
    with pytest.raises(weights.WeightsError, match="hash_mismatch"):
        weights.install({"u2net": root}, ["w"], lock_path=lock)
    assert not (root / "m.bin").exists()


def test_install_from_hugging_face_passes_the_pinned_revision(tmp_path):
    payload = b"ckpt"
    lock = _lock(tmp_path, [{"dest": "d/model.ckpt", "path": "skeleton/model.ckpt", "size": len(payload),
                             "sha256": _sha(payload)}], source="huggingface")
    calls = []

    def fake_download(*, repo_id, filename, revision, local_dir, token):
        calls.append((repo_id, filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.hf_hub_download = fake_download
    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        report = weights.install({"u2net": tmp_path / "u2net"}, ["w"], lock_path=lock)
    assert calls == [("org/repo", "skeleton/model.ckpt", "a" * 40)]
    assert report.pinned and (tmp_path / "u2net" / "d" / "model.ckpt").read_bytes() == payload


def test_offline_mode_reaches_stage_subprocesses(tmp_path, monkeypatch):
    monkeypatch.setenv("AISMITH_OFFLINE", "1")
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "u2net"))
    output = native.run_process([sys.executable, "-c",
        "import os; print(os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'], os.environ['U2NET_HOME'])"],
        cwd=tmp_path, timeout=60)
    assert f"1 1 {tmp_path / 'u2net'}" in output


def test_refiner_install_refuses_to_download_in_offline_mode(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setenv("AISMITH_OFFLINE", "1")
    state = main.DownloadState(id="offline-refiner", model="trellis-refiner")
    with patch.object(main, "DOWNLOADS_DIR", tmp_path), \
         patch.object(main.native, "trellis_refiner_dir", return_value=tmp_path / "runtime"):
        main._run_trellis_refiner_download(state)
    assert state.status == "failed"
    assert "AISMITH_OFFLINE" in (state.error or "")


def test_full_verification_ignores_the_cache(tmp_path):
    root = tmp_path / "u2net"
    root.mkdir()
    good = b"weights"
    target = root / "ok.bin"
    target.write_bytes(good)
    lock = _lock(tmp_path, [{"dest": "ok.bin", "size": len(good), "sha256": _sha(good)}])
    cache = tmp_path / "cache.json"
    assert weights.verify({"u2net": root}, full=True, cache_path=cache, lock_path=lock).pinned
    # Same size, mtime restored: only a rehash can see the change.
    stat = target.stat()
    target.write_bytes(b"WEIGHTS")
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert weights.verify({"u2net": root}, cache_path=cache, lock_path=lock).checks[0].status == "ok"
    full = weights.verify({"u2net": root}, full=True, cache_path=cache, lock_path=lock)
    assert full.checks[0].status == "hash_mismatch"


def test_a_worker_environment_cannot_switch_a_stage_back_online(tmp_path, monkeypatch):
    monkeypatch.setenv("AISMITH_OFFLINE", "1")
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "u2net"))
    output = native.run_process(
        [sys.executable, "-c", "import os; print(os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'])"],
        cwd=tmp_path, timeout=60, env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
    assert "1 1" in output


def test_gated_weights_record_where_their_licence_was_accepted():
    lock = json.loads(weights.LOCK_PATH.read_text(encoding="utf-8"))
    gated = [w for w in lock["weights"] if w.get("gated")]
    assert gated
    for entry in gated:
        acceptance = entry.get("licence_acceptance")
        assert isinstance(acceptance, dict), entry["id"]
        assert acceptance["licensee"] and acceptance["recorded_at"]
        # A public record in this repository that names the gated repository
        # and the pinned revision, and says each user must accept for themselves.
        record = weights.LOCK_PATH.parents[1] / acceptance["recorded_in"]
        assert record.is_file(), acceptance["recorded_in"]
        text = record.read_text(encoding="utf-8")
        assert entry["repository"] in text and entry["revision"] in text, entry["id"]
        assert acceptance["licensee"] in text
        # A gated file is pinned only once its licence is accepted.
        assert all(f.get("sha256") for f in entry["files"]), entry["id"]
