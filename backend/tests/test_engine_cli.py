"""`aismith-engine run`: spec validation, statuses, offline mode and stage wiring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import trimesh
from PIL import Image

from backend.app import native
from backend.engine import cli, inspect, stages


def _png(path: Path) -> Path:
    Image.new("RGB", (8, 8), (200, 40, 40)).save(path)
    return path


def _glb(path: Path) -> Path:
    trimesh.creation.box().export(path)
    return path


def _spec(tmp_path: Path, stage: str, inputs: dict, settings: dict | None = None, **extra) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "stage": stage, "inputs": {k: str(v) for k, v in inputs.items()},
                                "settings": settings or {}, **extra}))
    return spec


def _run(tmp_path: Path, stage: str, spec: Path) -> tuple[int, dict]:
    code = cli.main(["run", stage, "--spec", str(spec), "--workdir", str(tmp_path / "work")])
    return code, json.loads((tmp_path / "work" / "result.json").read_text())


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in ("AISMITH_OFFLINE", "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.mark.parametrize("make_spec, message", [
    (lambda t: _spec(t, "mesh", {"front": _png(t / "a.png")}, schema_version=2), "schema_version"),
    (lambda t: _spec(t, "clean", {"mesh": _glb(t / "m.glb")}), "not 'mesh'"),
    (lambda t: _spec(t, "mesh", {"front": t / "missing.png"}), "not a file"),
    (lambda t: _spec(t, "mesh", {"back": _png(t / "a.png")}), "front (required)"),
    (lambda t: _spec(t, "mesh", {"front": _glb(t / "m.glb")}), "not an image"),
    (lambda t: _spec(t, "mesh", {"front": _png(t / "a.png")}, {"steps": 20}), "unsupported mesh settings"),
    (lambda t: _spec(t, "mesh", {"front": _png(t / "a.png")}, {"resolution": 2048}), "resolution"),
    (lambda t: _spec(t, "mesh", {"front": _png(t / "a.png")}, {"seed": "7"}), "integer"),
])
def test_invalid_mesh_specs_are_rejected(tmp_path, make_spec, message):
    with patch.object(native, "run_trellis_fp8", side_effect=AssertionError("must not run")):
        code, result = _run(tmp_path, "mesh", make_spec(tmp_path))
    assert code == 2 and result["status"] == "rejected"
    assert message in result["error"]
    assert result["engine"]["version"]


@pytest.mark.parametrize("make_spec, message", [
    (lambda t: _spec(t, "clean", {"mesh": _png(t / "a.png")}), "GLB"),
    (lambda t: _spec(t, "clean", {"mesh": _glb(t / "m.glb")}, {"quality_tier": "giant"}), "giant"),
    (lambda t: _spec(t, "clean", {"mesh": _glb(t / "m.glb")}, {"method": "instant-meshes"}), "unsupported clean settings"),
])
def test_invalid_clean_specs_are_rejected(tmp_path, make_spec, message):
    with patch.object(native, "run_canonicalize", side_effect=AssertionError("must not run")):
        code, result = _run(tmp_path, "clean", make_spec(tmp_path))
    assert code == 2 and result["status"] == "rejected" and message in result["error"]


def test_unreadable_spec_is_rejected(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{not json")
    code, result = _run(tmp_path, "mesh", spec)
    assert code == 2 and "unreadable spec" in result["error"]


def test_mesh_runs_offline_with_the_requested_seed(tmp_path):
    front, back = _png(tmp_path / "front.png"), _png(tmp_path / "back.png")
    seen = {}

    def fake_generate(image, output, resolution, progress, views, *, seed):
        seen.update(offline=os.environ.get("AISMITH_OFFLINE"),
                    wait=os.environ.get("AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS"), views=views, image=image)
        _glb(output)
        return {"model": "visualbruno/TRELLIS.2-4B-FP8", "resolution": 1024, "seed": seed, "textured": True}

    with patch.object(native, "run_trellis_fp8", side_effect=fake_generate), \
         patch.object(stages, "check_stage_weights", return_value=None):
        code, result = _run(tmp_path, "mesh", _spec(tmp_path, "mesh", {"front": front, "back": back}, {"seed": 7}))
    assert code == 0 and result["status"] == "ok"
    assert seen["offline"] == "1" and seen["wait"] == "0"
    assert seen["image"] == front and set(seen["views"]) == {"front", "back"}
    assert result["effective_inputs"] == {"resolution": 1024, "seed": 7, "views": ["back", "front"],
                                          "model": "visualbruno/TRELLIS.2-4B-FP8"}
    output = Path(result["output"]["path"])
    assert result["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["output"]["content_type"] == "model/gltf-binary"
    assert result["measured"]["faces"] == 12


def test_mesh_fails_before_the_gpu_when_weights_are_missing(tmp_path):
    roots = {"trellis_models": tmp_path / "m", "unirig": tmp_path / "u", "u2net": tmp_path / "n"}
    with patch.object(native, "engine_weight_roots", return_value=roots), \
         patch.object(native, "run_trellis_fp8", side_effect=AssertionError("must not run")):
        code, result = _run(tmp_path, "mesh", _spec(tmp_path, "mesh", {"front": _png(tmp_path / "a.png")}))
    assert code == 1 and result["status"] == "failed"
    assert "pipeline_fp8.json missing" in result["error"]


def test_a_stage_crash_is_reported_as_failed(tmp_path):
    with patch.object(native, "run_trellis_fp8", side_effect=RuntimeError("CUDA error: an illegal memory access")), \
         patch.object(stages, "check_stage_weights", return_value=None):
        code, result = _run(tmp_path, "mesh", _spec(tmp_path, "mesh", {"front": _png(tmp_path / "a.png")}))
    assert code == 1 and result["status"] == "failed"
    assert "illegal memory access" in result["error"] and "Traceback" in result["diagnostics"]["traceback"]


def _canonicalize_metadata(components: int) -> dict:
    return {
        "attempts": [{"components": components, "components_unwelded": components, "watertight": True,
                      "non_manifold_edges": 0, "faces": 30_000}],
        "similarity": {"s1_s2_image_similarity_score": 0.97, "s1_s2_detail_ratio": 0.9},
        "triangle_budget_band": [20_000, 40_000],
        "quality_tier": "standard",
    }


@pytest.mark.parametrize("components, passed", [(1, True), (3, False)])
def test_clean_reports_the_shared_s2_gate(tmp_path, components, passed):
    calls = {}

    def fake_canonicalize(mesh, output, target_vertices, views, size, progress, *, method, settings, view_images):
        calls.update(method=method, settings=settings, view_images=view_images)
        _glb(output)
        return _canonicalize_metadata(components)

    with patch.object(native, "run_canonicalize", side_effect=fake_canonicalize):
        code, result = _run(tmp_path, "clean", _spec(tmp_path, "clean", {"mesh": _glb(tmp_path / "raw.glb")},
                                                     {"quality_tier": "standard"}))
    # A failed gate is still a produced output: status ok, gate.passed false.
    assert code == 0 and result["status"] == "ok"
    assert result["gate"]["name"] == "s2-canonicalize" and result["gate"]["passed"] is passed
    assert calls == {"method": "voxel", "settings": {"quality_tier": "standard"}, "view_images": None}
    assert result["effective_inputs"]["quality_tier"] == "standard"
    assert result["effective_inputs"]["triangle_band"] == [20_000, 40_000]


def test_trellis_worker_receives_the_seed(tmp_path):
    captured = {}

    def capture(command, **_kwargs):
        captured.update(json.loads(command[-1]))
        _glb(Path(captured["output_glb"]))
        return ""

    with patch.object(native, "trellis_refiner_ready", return_value=True), \
         patch.object(native, "trellis_refiner_python", return_value=Path("/usr/bin/python3")), \
         patch.object(native, "wait_for_gpu_preflight", return_value=type("G", (), {"available": True})()), \
         patch.object(native, "resolve_trellis_low_vram", return_value=False), \
         patch.object(native, "_glb_has_texture", return_value=True), \
         patch.object(native, "run_process", side_effect=capture):
        result = native.run_trellis_fp8(_png(tmp_path / "a.png"), tmp_path / "o.glb", 512, seed=42)
    assert captured["seed"] == 42 and result["seed"] == 42


def test_inspect_identifies_the_engine_and_its_locks():
    report = inspect.identity()
    assert report["version"] and len(report["sources_lock_sha256"]) == 64 and len(report["weights_lock_sha256"]) == 64


def test_inspect_reads_the_comfyui_version_and_finds_wsl_nvidia_smi(tmp_path):
    version_file = tmp_path / "comfyui_version.py"
    version_file.write_text('# generated\n__version__ = "0.33.0"\n')
    assert inspect._comfyui_version(version_file) == "0.33.0"
    assert inspect._comfyui_version(tmp_path / "missing.py") is None
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text("#!/bin/sh\necho 'NVIDIA GeForce RTX 4090, 8.9, 24564, 610.88'\n")
    fake_smi.chmod(0o755)
    with patch.object(inspect.shutil, "which", return_value=None), patch.object(inspect, "WSL_NVIDIA_SMI", fake_smi):
        assert inspect._gpus() == [{"name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9",
                                    "memory_mb": 24564, "driver": "610.88"}]


def _glb_bytes(json_chunk: bytes, *, version: int = 2, extra: bytes = b"", declared: int | None = None) -> bytes:
    import struct

    json_chunk += b" " * (-len(json_chunk) % 4)
    body = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk + extra
    total = 12 + len(body)
    return b"glTF" + struct.pack("<II", version, declared if declared is not None else total) + body


@pytest.mark.parametrize("payload, message", [
    (b"glTF", "not a binary glTF"),
    (_glb_bytes(b'{"meshes":[{}]}', version=1), "version 1"),
    (_glb_bytes(b'{"meshes":[{}]}', declared=10_000), "truncated or padded"),
    (_glb_bytes(b"{not json"), "unreadable JSON chunk"),
    (_glb_bytes(b'{"asset":{"version":"2.0"}}'), "no meshes"),
    (_glb_bytes(b'{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],'
                b'"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],'
                b'"bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]}'),
     "could not be loaded"),
])
def test_corrupt_glbs_are_rejected_before_blender_starts(tmp_path, payload, message):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(payload)
    with patch.object(native, "run_canonicalize", side_effect=AssertionError("must not run")):
        code, result = _run(tmp_path, "clean", _spec(tmp_path, "clean", {"mesh": bad}))
    assert code == 2 and result["status"] == "rejected"
    assert message in result["error"]
