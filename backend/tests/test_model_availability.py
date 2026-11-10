"""Which local workflows the studio app offers, and what each one needs first.

The legacy trellis.cpp download is gone: Generate runs on the TRELLIS.2 FP8
runtime, or on the trellis2cpp GGUF fallback when its own models are present.
Rebuild-from-views needs the optional Pixal3D scripts, which not every checkout
carries.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app import main

client = TestClient(main.app)

NOTHING_READY = {
    "trellis_gguf_ready": False,
    "trellis_refiner_ready": False,
    "instant_meshes_ready": False,
    "kimodo_ready": False,
    "unirig_ready": False,
    "rebuild_views_ready": False,
}


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    directory = tmp_path / "jobs"
    directory.mkdir()
    monkeypatch.setattr(main, "JOBS_DIR", directory)
    return directory


def test_the_legacy_trellis_cpp_package_is_not_offered():
    response = client.post("/api/models/trellis-cpp/download")
    assert response.status_code == 404


def test_readiness_reads_the_gguf_fallback_from_its_own_model_files():
    with (
        patch.object(main.native, "trellis_gguf_ready", return_value=True),
        patch.object(main.native, "trellis_refiner_ready", return_value=False),
        patch.object(main.native, "instant_meshes_exe", return_value=None),
        patch.object(main.native, "kimodo_executable", return_value=None),
        patch.object(main.native, "unirig_ready", return_value=False),
        patch.object(main.native, "rebuild_views_available", return_value=False),
    ):
        readiness = main._model_readiness()
    assert readiness["trellis_gguf_ready"] is True
    assert readiness["rebuild_views_ready"] is False
    assert "trellis_cpp_ready" not in readiness


def test_rebuild_views_is_offered_only_with_its_scripts(tmp_path):
    script = tmp_path / "approach9.sh"
    with patch.object(main.native, "REBUILD_VIEWS_SCRIPT", script):
        assert main.native.rebuild_views_available() is False
        script.write_text("#!/usr/bin/env bash\n")
        assert main.native.rebuild_views_available() is True


def test_generate_is_refused_until_a_geometry_workflow_is_installed(jobs_dir, tmp_path):
    image = main.Artifact(
        id="front-1", name="front.png", kind="image", path=str(tmp_path / "front.png"),
        size_bytes=1, created_at=0,
    )
    with (
        patch.object(main, "read_artifact", return_value=image),
        patch.object(main, "_model_readiness", return_value=dict(NOTHING_READY)),
    ):
        response = client.post("/api/jobs", json={
            "kind": "generate", "submission_id": "generate-without-models", "source_artifact_id": image.id,
        })
    assert response.status_code == 409
    assert "No geometry workflow" in response.json()["detail"]


def test_rebuild_views_is_refused_without_its_scripts(jobs_dir, tmp_path):
    mesh = main.Artifact(
        id="mesh-1", name="mesh.glb", kind="glb", path=str(tmp_path / "mesh.glb"),
        size_bytes=1, created_at=0,
    )
    with (
        patch.object(main, "read_artifact", return_value=mesh),
        patch.object(main, "_model_readiness", return_value=dict(NOTHING_READY)),
        patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
    ):
        response = client.post("/api/jobs", json={
            "kind": "rebuild-views", "submission_id": "rebuild-without-scripts", "source_artifact_id": mesh.id,
        })
    assert response.status_code == 409
    assert "Pixal3D" in response.json()["detail"]
