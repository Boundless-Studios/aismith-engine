import unittest
import io
import asyncio
import json
import os
import subprocess
import struct
import sys
import uuid
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import trimesh

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app import main
from backend.app import native
from backend.app import pipeline_policy
from backend.app.run_store import ReviewState, RunStore


client = TestClient(app)


class ApiTests(unittest.TestCase):
    def test_corrupt_joint_recipe_history_is_treated_as_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            store = RunStore(root / "runs")
            run = store.create("Mary", "clean-mesh-1", "glb")
            invalid = root / "invalid.json"
            main._write_json(invalid, {"id": "metadata-instead-of-recipe"})
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "output_directory", return_value=artifacts),
                ):
                    artifact = main.save_artifact(invalid, name="joint-recipe.json", kind="joint-recipe")
                    attempt = store.start_attempt(run.id, "joint-recipe", "clean-mesh-1", {})
                    store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id=artifact.id, promote_output=False)

                    response = client.get(f"/api/runs/{run.id}/joint-recipe")

                self.assertEqual(response.status_code, 404, response.text)
            finally:
                main.run_store = previous_store

    def test_json_artifact_payload_does_not_collide_with_its_metadata_sidecar(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = root / "joint-recipe.json"
            payload = {"schema_version": 1, "clean_mesh_artifact_id": "clean-mesh-1"}
            main._write_json(source, payload)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="joint-recipe.json", kind="joint-recipe")

                self.assertEqual(main._read_json(main._artifact_path(artifact.id)), payload)
                self.assertNotEqual(main._artifact_path(artifact.id), artifacts / f"{artifact.id}.json")

    def test_joint_recipe_is_versioned_without_promoting_it_as_the_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            outputs = root / "outputs"
            artifacts.mkdir()
            outputs.mkdir()
            store = RunStore(root / "runs")
            run = store.create("Mary", "clean-mesh-1", "glb")
            recipe = {
                "schema_version": 1,
                "clean_mesh_artifact_id": "clean-mesh-1",
                "skeleton_template": "Human",
                "bone_naming": "mesh2motion-human-v1",
                "mirror_enabled": True,
                "transforms": [
                    {
                        "name": "hips",
                        "position": [0.0, 1.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
                "editor_version": "motion-lab-v1",
            }
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "output_directory", return_value=outputs),
                ):
                    first = client.put(f"/api/runs/{run.id}/joint-recipe", json=recipe)
                    second = client.put(f"/api/runs/{run.id}/joint-recipe", json=recipe)
                    second_metadata = main._read_json(artifacts / f"{second.json()['artifact_id']}.json")
                    self.assertEqual(Path(second_metadata["path"]).resolve().parent, outputs.resolve())
                    self.assertEqual(
                        main._artifact_path(second.json()["artifact_id"]).parent,
                        outputs.resolve(),
                    )
                    loaded = client.get(f"/api/runs/{run.id}/joint-recipe")

                self.assertEqual(first.status_code, 200)
                self.assertEqual(second.status_code, 200)
                self.assertEqual(loaded.status_code, 200, loaded.text)
                self.assertNotEqual(first.json()["artifact_id"], second.json()["artifact_id"])
                self.assertEqual(loaded.json()["artifact_id"], second.json()["artifact_id"])
                self.assertEqual(loaded.json()["recipe"], recipe)
                persisted = store.get(run.id)
                self.assertEqual(persisted.current_artifact_id, "clean-mesh-1")
                self.assertEqual([item.stage for item in persisted.attempts], ["joint-recipe", "joint-recipe"])
                self.assertTrue(all(item.status == "completed" for item in persisted.attempts))
                self.assertTrue(all(not item.promoted for item in persisted.attempts))
            finally:
                main.run_store = previous_store

    def test_instant_meshes_download_matches_host_platform(self):
        with patch.object(main.sys, "platform", "linux"):
            self.assertEqual(
                main._instant_meshes_download_spec(),
                (
                    "https://instant-meshes.s3.eu-central-1.amazonaws.com/instant-meshes-linux.zip",
                    "instant-meshes-linux.zip",
                    "Linux",
                ),
            )

    def test_cors_defaults_cover_supported_vite_development_ports(self):
        with patch.dict(os.environ, {}, clear=False):
            origins = main._cors_origins()

        self.assertIn("http://localhost:5174", origins)
        self.assertIn("http://127.0.0.1:5174", origins)

    def test_prepare_image_retains_profiled_candidate_without_promoting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "mary.png"
            Image.new("RGB", (32, 32), (120, 60, 40)).save(source_path)
            source = main.Artifact(
                id="mary-portrait",
                name="mary.png",
                kind="image",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Mary preparation", source.id, source.kind)
            job = main.JobState(id="prepare-image-job", kind="prepare-image", run_id=run.id)
            attempt = store.reserve_attempt(
                run.id,
                job.id,
                "prepare-image",
                source.id,
                {"adapter": "upload", "profile": "humanoid-a-pose-sword-v1"},
            )
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="prepare-image",
                submission_id="prepare-mary-1",
                source_artifact_id=source.id,
                run_id=run.id,
                seed=42,
                settings={
                    "adapter": "upload",
                    "profile": "humanoid-a-pose-sword-v1",
                },
            )
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                self.assertEqual(store.get(run.id).current_artifact_id, source.id)
                output = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertEqual(output.kind, "image")
                self.assertEqual(output.metadata["preparation_profile"]["id"], "humanoid-a-pose-sword-v1")
                self.assertEqual(output.metadata["adapter"], "upload")
                self.assertEqual(Path(output.path).read_bytes(), source_path.read_bytes())
            finally:
                main.run_store = previous_store

    def test_prepare_image_uses_comfyui_and_retains_generation_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "mary.png"
            Image.new("RGB", (32, 32), (120, 60, 40)).save(source_path)
            source = main.Artifact(
                id="mary-portrait", name="mary.png", kind="image",
                path=str(source_path), size_bytes=source_path.stat().st_size, created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Mary preparation", source.id, source.kind)
            job = main.JobState(id="prepare-comfy-job", kind="prepare-image", run_id=run.id)
            settings = {"adapter": "comfyui", "profile": "humanoid-a-pose-sword-v1"}
            attempt = store.reserve_attempt(run.id, job.id, "prepare-image", source.id, settings)
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="prepare-image", submission_id="prepare-mary-comfy-1",
                source_artifact_id=source.id, run_id=run.id, seed=42, settings=settings,
            )
            previous_store = main.run_store

            def generate(_source, output, **_kwargs):
                Image.new("RGB", (64, 96), (30, 90, 120)).save(output)
                return {
                    "prompt_id": "prompt-42",
                    "workflow_version": "flux2-klein-image-edit-v1",
                    "model": "flux-2-klein-base-4b-fp8.safetensors",
                }

            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "run_comfyui_image_edit", side_effect=generate),
                    patch.object(main.model_manager, "acquire", new_callable=AsyncMock) as acquire,
                    patch.object(main.model_manager, "release", new_callable=AsyncMock) as release,
                    patch.dict(os.environ, {"AISMITH_COMFYUI_URL": "http://comfy.local:8188"}),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                output = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                with Image.open(output.path) as prepared:
                    self.assertEqual(prepared.size, (64, 96))
                self.assertEqual(output.metadata["adapter"], "comfyui")
                self.assertEqual(output.metadata["prompt_id"], "prompt-42")
                self.assertFalse(output.metadata["native"])
                self.assertFalse(output.metadata["worker_isolated"])
                self.assertEqual(output.metadata["execution"], "external-comfyui")
                self.assertEqual(store.get(run.id).current_artifact_id, source.id)
                acquire.assert_awaited_once_with("flux2-klein", job.id)
                release.assert_awaited_once()
            finally:
                main.run_store = previous_store

    def test_completed_prepared_image_can_be_promoted_but_arbitrary_image_cannot(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "runs")
            run = store.create("Mary preparation", "source-image", "image")
            attempt = store.reserve_attempt(
                run.id, "prepare-job", "prepare-image", "source-image", {"adapter": "comfyui"}
            )
            store.finish_attempt(
                run.id, attempt.id, "completed",
                output_artifact_id="prepared-image", promote_output=False,
            )
            prepared = main.Artifact(
                id="prepared-image", name="prepared.png", kind="image",
                path=str(Path(directory) / "prepared.png"), size_bytes=1, created_at=0,
            )
            arbitrary = prepared.model_copy(update={"id": "arbitrary-image"})
            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "read_artifact", return_value=prepared):
                    promoted = asyncio.run(main.promote_run_artifact(
                        run.id, prepared.id,
                        main.ArtifactPromotionRequest(reason="Approved pose and identity"),
                    ))
                self.assertEqual(promoted.current_artifact_id, prepared.id)

                with patch.object(main, "read_artifact", return_value=arbitrary):
                    with self.assertRaises(main.HTTPException) as rejected:
                        asyncio.run(main.promote_run_artifact(
                            run.id, arbitrary.id,
                            main.ArtifactPromotionRequest(reason="Not a run candidate"),
                        ))
                self.assertEqual(rejected.exception.status_code, 409)
            finally:
                main.run_store = previous_store
    def test_deformation_review_frames_are_persisted_as_downloadable_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            frame = root / "review-mid.png"
            frame.write_bytes(b"\x89PNG\r\n\x1a\nreview")
            report = {
                "frames": [{"frame": index} for index in range(5)],
                "maximum_p99_edge_stretch": 5.0,
                "_review_files": [{
                    "label": "attack-mid",
                    "ratio": 0.5,
                    "rest_pose": False,
                    "focus_bone": "right-hand",
                    "path": str(frame),
                }],
            }

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "SETTINGS_PATH", root / "settings.json"),
            ):
                persisted = main.persist_deformation_review_artifacts(report)
                reference = persisted["review_artifacts"][0]
                artifact = main.read_artifact(reference["artifact_id"])

            self.assertEqual(artifact.kind, "image")
            self.assertEqual(artifact.metadata["stage"], "deformation-review")
            self.assertEqual(artifact.metadata["focus_bone"], "right-hand")
            self.assertEqual(reference["label"], "attack-mid")
            self.assertEqual(reference["focus_bone"], "right-hand")
            self.assertFalse(frame.exists())

    def test_kimodo_uses_isolated_runtime_and_durable_model_cache(self):
        with TemporaryDirectory() as directory:
            data_root = Path(directory)
            runtime = data_root / "tools" / "kimodo"
            executable = runtime / ".venv" / "bin" / "kimodo_gen"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            output = data_root / "attack.bvh"

            def generate(command, **_kwargs):
                if command[0] == str(executable.resolve()):
                    output.write_text("HIERARCHY\n" + "x" * 64, encoding="utf-8")
                else:
                    spec = json.loads(command[-1])
                    Path(spec["output"]).write_text("HIERARCHY\n" + "z" * 64, encoding="utf-8")
                return "generated"

            with (
                patch.dict(os.environ, {
                    "AISMITH_DATA_DIR": str(data_root),
                    "HF_TOKEN_PATH": "/secure/huggingface/token",
                }),
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "run_process", side_effect=generate) as run,
            ):
                result = native.run_kimodo_motion(
                    "A female sword fighter performs one decisive slash and recovers",
                    output,
                    duration=2.0,
                    seed=7,
                )

            command = run.call_args_list[0].args[0]
            self.assertEqual(command[0], str(executable.resolve()))
            self.assertIn("Kimodo-SOMA-RP-v1.1", command)
            self.assertIn("--bvh_standard_tpose", command)
            self.assertEqual(
                run.call_args_list[0].kwargs["env"]["HF_HOME"],
                str((data_root / "models" / "huggingface").resolve()),
            )
            self.assertEqual(
                run.call_args_list[0].kwargs["env"]["HF_TOKEN_PATH"],
                "/secure/huggingface/token",
            )
            self.assertEqual(result["model"], "Kimodo-SOMA-RP-v1.1")
            self.assertEqual(result["axes"], "blender-z-up")
            self.assertEqual(result["axes_normalized_from"], "y-up")

    def test_kimodo_preserves_token_from_incoming_hf_home(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "tools" / "kimodo" / ".venv" / "bin" / "kimodo_gen"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            incoming_home = root / "logged-in-huggingface"
            incoming_home.mkdir()
            token = incoming_home / "token"
            token.write_text("not-a-real-token", encoding="utf-8")
            output = root / "attack.bvh"

            def generate(command, **_kwargs):
                if command[0] == str(executable.resolve()):
                    output.write_text("HIERARCHY\n" + "x" * 64, encoding="utf-8")
                else:
                    spec = json.loads(command[-1])
                    Path(spec["output"]).write_text("HIERARCHY\n" + "z" * 64, encoding="utf-8")
                return "generated"

            with (
                patch.dict(os.environ, {"AISMITH_DATA_DIR": str(root), "HF_HOME": str(incoming_home), "HF_TOKEN_PATH": ""}),
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "run_process", side_effect=generate) as run,
            ):
                native.run_kimodo_motion("Sword attack", output, duration=2.0, seed=7)

            self.assertEqual(run.call_args_list[0].kwargs["env"]["HF_TOKEN_PATH"], str(token))

    def test_external_kimodo_executable_runs_from_its_existing_install_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "external-kimodo" / "bin" / "kimodo_gen"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            output = root / "attack.bvh"

            def generate(command, **_kwargs):
                if command[0] == str(executable.resolve()):
                    output.write_text("HIERARCHY\n" + "x" * 64, encoding="utf-8")
                else:
                    spec = json.loads(command[-1])
                    Path(spec["output"]).write_text("HIERARCHY\n" + "z" * 64, encoding="utf-8")
                return "generated"

            with (
                patch.dict(os.environ, {
                    "AISMITH_DATA_DIR": str(root / "missing-managed-data"),
                    "KIMODO_EXECUTABLE": str(executable),
                }),
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "run_process", side_effect=generate) as run,
            ):
                native.run_kimodo_motion("sword slash", output, duration=2.0, seed=0)

            self.assertEqual(run.call_args_list[0].kwargs["cwd"], executable.parent.resolve())

    def test_kimodo_reports_the_separate_meta_encoder_license_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "kimodo" / ".venv" / "bin" / "kimodo_gen"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")

            with (
                patch.dict(os.environ, {
                    "AISMITH_DATA_DIR": str(root),
                    "KIMODO_EXECUTABLE": str(executable),
                }),
                patch.object(
                    native,
                    "run_process",
                    side_effect=RuntimeError(
                        "403 Cannot access gated repo meta-llama/Meta-Llama-3-8B-Instruct; not in the authorized list"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
                ) as raised:
                    native.run_kimodo_motion(
                        "sword slash",
                        root / "attack.bvh",
                        duration=2.0,
                        seed=23,
                    )

            self.assertNotIn("authorized list", str(raised.exception))

    def test_kimodo_preserves_unrelated_meta_encoder_403_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "kimodo" / ".venv" / "bin" / "kimodo_gen"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            detail = (
                "403 CDN request failed while fetching "
                "meta-llama/Meta-Llama-3-8B-Instruct"
            )

            with (
                patch.dict(os.environ, {
                    "AISMITH_DATA_DIR": str(root),
                    "KIMODO_EXECUTABLE": str(executable),
                }),
                patch.object(native, "run_process", side_effect=RuntimeError(detail)),
            ):
                with self.assertRaisesRegex(RuntimeError, "CDN request failed"):
                    native.run_kimodo_motion(
                        "sword slash",
                        root / "attack.bvh",
                        duration=2.0,
                        seed=23,
                    )

    def test_generate_motion_retains_bvh_without_replacing_current_character(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "mary-rigged.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="mary-rigged",
                name="mary-rigged.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                metadata={"quality_decisions": {
                    "rig": {"status": "ready"},
                    "rig-certificate": {"status": "ready", "policy_version": 2},
                    "animation-review": {"status": "ready"},
                    "deformation-review": {"status": "ready"},
                }},
            )
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind)
            job = main.JobState(id="job-motion", kind="generate-motion", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "generate-motion", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="generate-motion",
                submission_id="motion-1",
                source_artifact_id=source.id,
                run_id=run.id,
                seed=12,
                settings={"prompt": "one decisive sword slash", "duration_seconds": 2.0},
            )

            def generate(prompt, output, *, duration, seed, progress):
                self.assertEqual(prompt, "one decisive sword slash")
                self.assertEqual(duration, 2.0)
                self.assertEqual(seed, 12)
                output.write_text("HIERARCHY\n" + "x" * 64, encoding="utf-8")
                return {"model": "Kimodo-SOMA-RP-v1.1"}

            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "output_directory", return_value=artifacts),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "run_kimodo_motion", side_effect=generate),
                    patch.object(main.model_manager, "acquire", new=AsyncMock()) as acquire,
                    patch.object(main.model_manager, "release", new=AsyncMock()) as release,
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                self.assertEqual(store.get(run.id).current_artifact_id, source.id)
                motion = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertEqual(motion.kind, "bvh")
                self.assertEqual(motion.metadata["source_artifact_id"], source.id)
                acquire.assert_awaited_once_with("kimodo", job.id)
                release.assert_awaited_once()
            finally:
                main.run_store = previous_store

    def test_generate_motion_dispatches_slither_for_chain_certified_rig(self):
        """Task F3: a chain rig's `generate-motion` request bakes a
        procedural Slither cycle with Blender instead of asking NVIDIA
        Kimodo for a humanoid SOMA clip, and goes through the same
        rig/animation/deformation gates as retarget-motion."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "serpent-rig.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="serpent-rig",
                name="serpent-rig.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                metadata={"quality_decisions": {
                    "rig-certificate": {
                        "status": "ready",
                        "policy_version": 2,
                        "profile": "chain",
                        "resolved_roles": {"spine": "spine_00", "head": "head"},
                    },
                }},
            )
            store = RunStore(root / "runs")
            run = store.create("Serpent", source.id, source.kind)
            job = main.JobState(id="job-slither", kind="generate-motion", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "generate-motion", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="generate-motion",
                submission_id="slither-1",
                source_artifact_id=source.id,
                run_id=run.id,
                settings={"frames": 48, "wavelength": 1.0, "amplitude": 0.12, "speed": 1.0, "head_stabilize": 0.5},
            )

            def bake(input_glb, output, animation, settings, *, progress):
                self.assertEqual(Path(input_glb), source_path)
                self.assertEqual(animation, "Slither")
                self.assertEqual(settings["frames"], 48)
                trimesh.creation.box().export(output)
                return {"backend": "blender-slither", "animation": "Slither", "settings": dict(settings)}

            # 14 spine bones + head: clears the chain certificate's own
            # floor (>= 8 spine_ bones) with margin, so this fixture drives
            # a real "ready" animation-review verdict rather than papering
            # over the chain-profile gate with a minimal/degenerate rig
            # (review F3-5).
            chain_joint_names = [f"spine_{i:02d}" for i in range(14)] + ["head"]
            rig_report = {
                "joint_names": chain_joint_names,
                "profile": "chain",
                "skins": 1,
                "joints": len(chain_joint_names),
                "skinned_vertices": 1000,
                "animations": [{
                    "name": "Slither",
                    "duration_seconds": 2.0,
                    "animated_joint_coverage": 1.0,
                    "animated_joint_names": chain_joint_names,
                    "invalid_targets": 0,
                }],
            }
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "output_directory", return_value=artifacts),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "analyze_rig_path", return_value=rig_report),
                    patch.object(
                        main,
                        "persist_deformation_review_artifacts",
                        side_effect=lambda report: report,
                    ),
                    patch.object(
                        main.native,
                        "run_deformation_diagnostics",
                        return_value={"per_frame": [], "review_artifacts": []},
                    ),
                    patch.object(main.native, "run_slither_bake", side_effect=bake) as slither_bake,
                    patch.object(main.native, "run_kimodo_motion") as kimodo,
                    patch.object(main.model_manager, "acquire", new=AsyncMock()) as acquire,
                    patch.object(main.model_manager, "release", new=AsyncMock()),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                slither_bake.assert_called_once()
                kimodo.assert_not_called()
                acquire.assert_awaited_once_with("blender", job.id)
                artifact = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertEqual(artifact.kind, "glb")
                self.assertEqual(artifact.metadata["backend"], "blender-slither")
                self.assertEqual(artifact.metadata["animation"], "Slither")
                self.assertEqual(
                    artifact.metadata["quality_decisions"]["rig-certificate"]["profile"],
                    "chain",
                )
                self.assertEqual(
                    artifact.metadata["quality_decisions"]["rig-certificate"]["status"],
                    "ready",
                )
                animation_review = artifact.metadata["quality_decisions"]["animation-review"]
                self.assertEqual(
                    animation_review["status"],
                    "ready",
                    f"expected a clean animation-review verdict for a rig that clears "
                    f"the chain minimum, got: {animation_review}",
                )
                self.assertIn("deformation-review", artifact.metadata["quality_decisions"])
            finally:
                main.run_store = previous_store

    def test_retarget_motion_rejects_chain_rig_before_queueing(self):
        """Task F3: chain-rigged creatures never go through SOMA
        retarget-motion -- 422 with a clear message pointing at
        generate-motion's Slither path instead."""
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "chain"},
                }
            }
            serpent = main.Artifact(
                id="serpent-1", name="serpent.glb", kind="glb", path="serpent.glb",
                size_bytes=1, created_at=0, metadata=certificate,
            )
            motion = main.Artifact(id="motion-1", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            run = main.run_store.create("Serpent", serpent.id, serpent.kind)
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {serpent.id: serpent, motion.id: motion}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "retarget-motion",
                        "submission_id": "retarget-chain-rejected",
                        "source_artifact_id": serpent.id,
                        "run_id": run.id,
                        "settings": {"motion_artifact_id": motion.id},
                    })

                self.assertEqual(response.status_code, 422)
                self.assertIn("Chain-rigged", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
                self.assertEqual(main.jobs, {})
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_generate_motion_for_chain_rig_does_not_require_kimodo(self):
        """A chain rig's generate-motion request must not be blocked by a
        missing Kimodo install -- it never calls Kimodo at all."""
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "chain"},
                }
            }
            serpent = main.Artifact(
                id="serpent-2", name="serpent.glb", kind="glb", path="serpent.glb",
                size_bytes=1, created_at=0, metadata=certificate,
            )
            run = main.run_store.create("Serpent", serpent.id, serpent.kind)
            try:
                with (
                    patch.object(main, "read_artifact", return_value=serpent),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": False}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "generate-motion",
                        "submission_id": "slither-no-kimodo",
                        "source_artifact_id": serpent.id,
                        "run_id": run.id,
                        "settings": {},
                    })

                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(main.run_store.get(run.id).attempts), 1)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_bvh_retarget_invokes_blender_with_character_and_motion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            character = root / "mary.glb"
            motion = root / "attack.bvh"
            output = root / "animated.glb"
            character.write_bytes(b"glTF")
            motion.write_text("HIERARCHY", encoding="utf-8")
            blender = root / "blender"
            blender.write_text("", encoding="utf-8")

            def bake(command, **_kwargs):
                output.write_bytes(b"glTF" + b"x" * 64)
                return "baked"

            with (
                patch.object(native, "find_blender", return_value=blender),
                patch.object(native, "run_process", side_effect=bake) as run,
            ):
                result = native.run_bvh_retarget(
                    character,
                    motion,
                    output,
                    target_bones={"pelvis": "Hips", "spine": "Spine"},
                )

            spec = json.loads(run.call_args.args[0][-1])
            self.assertEqual(spec["op"], "retarget-bvh")
            self.assertEqual(spec["motion"], str(motion))
            self.assertEqual(spec["target_bones"]["pelvis"], "Hips")
            self.assertEqual(result["animation"], "Sword_Attack")

    def test_bvh_retarget_names_the_baked_action_after_the_requested_clip(self):
        # Regression for the curated-clip picker: the retarget worker used to
        # hardcode "Sword_Attack" as the baked output's own action name no
        # matter which clip was actually attached (Dance Simple would play
        # correctly but show as "Sword Attack" in the viewport). The
        # `animation` kwarg must reach both the worker spec and the returned
        # metadata.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            character = root / "mary.glb"
            motion = root / "dance.bvh"
            output = root / "animated.glb"
            character.write_bytes(b"glTF")
            motion.write_text("HIERARCHY", encoding="utf-8")
            blender = root / "blender"
            blender.write_text("", encoding="utf-8")

            def bake(command, **_kwargs):
                output.write_bytes(b"glTF" + b"x" * 64)
                return "baked"

            with (
                patch.object(native, "find_blender", return_value=blender),
                patch.object(native, "run_process", side_effect=bake) as run,
            ):
                result = native.run_bvh_retarget(
                    character,
                    motion,
                    output,
                    target_bones={"pelvis": "Hips"},
                    animation="Dance_Simple",
                )

            spec = json.loads(run.call_args.args[0][-1])
            self.assertEqual(spec["animation"], "Dance_Simple")
            self.assertEqual(result["animation"], "Dance_Simple")

    def test_bvh_retarget_rejects_a_worker_that_did_not_export_a_glb(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            character = root / "mary.glb"
            motion = root / "attack.bvh"
            output = root / "missing.glb"
            blender = root / "blender"
            character.write_bytes(b"glTF")
            motion.write_text("HIERARCHY", encoding="utf-8")
            blender.write_text("", encoding="utf-8")

            with (
                patch.object(native, "find_blender", return_value=blender),
                patch.object(native, "run_process", return_value="traceback"),
            ):
                with self.assertRaisesRegex(RuntimeError, "valid animated GLB"):
                    native.run_bvh_retarget(
                        character,
                        motion,
                        output,
                        target_bones={"pelvis": "Hips"},
                    )

    def test_refiner_selects_an_interpreter_that_has_torch(self):
        with TemporaryDirectory() as directory:
            runtime = Path(directory)
            stale = runtime / ".venv" / "bin" / "python"
            working = runtime / ".venv-cu128" / "bin" / "python"
            stale.parent.mkdir(parents=True)
            working.parent.mkdir(parents=True)
            stale.write_text("", encoding="utf-8")
            working.write_text("", encoding="utf-8")
            resolved_working = working.resolve()

            def probe(command, **_kwargs):
                return Mock(returncode=0 if Path(command[0]) == resolved_working else 1)

            with (
                patch.object(native, "trellis_refiner_dir", return_value=runtime),
                patch.object(native.subprocess, "run", side_effect=probe),
                patch.dict(os.environ, {"TRELLIS_REFINER_PYTHON": str(stale)}),
            ):
                selected = native.trellis_refiner_python()

            self.assertEqual(selected, resolved_working)

    def test_native_model_defaults_follow_the_durable_data_root(self):
        with TemporaryDirectory() as directory:
            data_root = Path(directory).resolve()
            with patch.dict(
                os.environ,
                {
                    "AISMITH_DATA_DIR": str(data_root),
                    "TRELLIS_GGUF_MODEL_DIR": "",
                    "TRELLIS_REFINER_DIR": "",
                    "INSTANT_MESHES_DIR": "",
                },
            ):
                self.assertEqual(native.trellis_gguf_model_dir(), data_root / "models" / "Trellis2")
                self.assertEqual(native.trellis_refiner_dir(), data_root / "tools" / "trellis2-refiner")
                self.assertEqual(native.instant_meshes_dir(), data_root / "tools" / "instant-meshes")

    def test_get_artifact_returns_durable_metadata(self):
        artifact = main.Artifact(
            id="mesh-1",
            name="mary.glb",
            kind="glb",
            path="mary.glb",
            size_bytes=1,
            created_at=0,
            metadata={"quality_decisions": {"rig": {"status": "blocked"}}},
        )
        with patch.object(main, "read_artifact", return_value=artifact):
            response = client.get(f"/api/assets/{artifact.id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"]["quality_decisions"]["rig"]["status"], "blocked")

    def test_generated_job_automatically_attaches_mesh_diagnostics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            image_path = root / "mary.png"
            Image.new("RGBA", (8, 8), "white").save(image_path)
            source = main.Artifact(
                id="source-1",
                name="mary.png",
                kind="image",
                path=str(image_path),
                size_bytes=image_path.stat().st_size,
                created_at=0,
            )
            job = main.JobState(id="job-1", kind="generate")
            request = main.JobRequest(
                kind="generate",
                submission_id="submission-1",
                source_artifact_id=source.id,
            )

            def generate(_source, output, _resolution, _progress, **_kwargs):
                trimesh.creation.box().export(output)
                return {"generator": "test"}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "JOBS_DIR", jobs_dir),
                patch.object(main, "output_directory", return_value=artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "trellis_refiner_ready", return_value=True),
                patch.object(main.native, "run_trellis_fp8", side_effect=generate),
                patch.object(main.model_manager, "acquire", new=AsyncMock()),
                patch.object(main.model_manager, "release", new=AsyncMock()),
            ):
                asyncio.run(main._execute_job(job, request))
                artifact = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )

            self.assertEqual(job.status, "completed")
            self.assertEqual(artifact.metadata["mesh_quality"]["components"], 1)
            self.assertEqual(artifact.metadata["mesh_quality"]["boundary_edges"], 0)
            self.assertEqual(artifact.metadata["quality_decisions"]["rig"]["status"], "blocked")

    def test_gguf_fallback_ignores_source_views_and_warns(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            front_path = root / "front.png"
            Image.new("RGBA", (8, 8), "white").save(front_path)
            front = main.Artifact(
                id="front-1", name="front.png", kind="image",
                path=str(front_path), size_bytes=front_path.stat().st_size, created_at=0,
            )
            back = main.Artifact(id="back-1", name="back.png", kind="image", path="back.png", size_bytes=1, created_at=0)
            artifacts_by_id = {front.id: front, back.id: back}
            job = main.JobState(id="job-gguf", kind="generate")
            request = main.JobRequest(
                kind="generate",
                submission_id="submission-gguf",
                source_artifact_id=front.id,
                source_views={"front": front.id, "back": back.id},
            )

            def generate_gguf(source, output, _resolution, _progress):
                self.assertEqual(source, front_path)
                trimesh.creation.box().export(output)
                return {"generator": "gguf-test"}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "JOBS_DIR", jobs_dir),
                patch.object(main, "output_directory", return_value=artifacts),
                patch.object(main, "read_artifact", side_effect=lambda artifact_id: artifacts_by_id[artifact_id]),
                patch.object(main.native, "trellis_refiner_ready", return_value=False),
                patch.object(main.native, "run_trellis_gguf", side_effect=generate_gguf),
                patch.object(main.model_manager, "acquire", new=AsyncMock()),
                patch.object(main.model_manager, "release", new=AsyncMock()),
            ):
                asyncio.run(main._execute_job(job, request))
                artifact = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )

            self.assertEqual(job.status, "completed")
            # The GGUF fallback path never conditions on the extra views, so
            # provenance for them must not be stamped -- it would be a lie.
            self.assertNotIn("source_views", artifact.metadata)
            self.assertTrue(
                any("GGUF fallback" in message and "does not support multi-view" in message for message in job.logs)
            )

    def test_regressed_mesh_is_retained_without_replacing_the_current_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "refined.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="refined-1",
                name="refined.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                metadata={
                    "mesh_quality": {
                        "faces": 48974,
                        "largest_component_face_ratio": 0.996263,
                        "non_manifold_edges": 496,
                        "degenerate_faces": 0,
                        "finite": True,
                    }
                },
            )
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind)
            job = main.JobState(id="job-regression", kind="retopo", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "retopo", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="retopo",
                submission_id="regression-1",
                source_artifact_id=source.id,
                run_id=run.id,
            )

            def fragment(_source, output, _target, _repair, _progress):
                pieces = [trimesh.creation.box(transform=trimesh.transformations.translation_matrix([index * 2, 0, 0])) for index in range(100)]
                trimesh.util.concatenate(pieces).export(output)
                return {"backend": "test-fragmenter"}

            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "run_instant_meshes", side_effect=fragment),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                loaded = store.get(run.id)
                self.assertEqual(loaded.current_artifact_id, source.id)
                self.assertTrue(loaded.attempts[-1].output_artifact_id)
                self.assertFalse(loaded.attempts[-1].promoted)
                self.assertIn("primary component coverage regressed", loaded.attempts[-1].warnings[0])
            finally:
                main.run_store = previous_store

    def test_weight_repair_candidate_requires_a_passing_rig_certificate_for_promotion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "bad-rig.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="bad-rig",
                name="mary-rig.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                metadata={"mesh_quality": {"faces": 12}},
            )
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind)
            job = main.JobState(id="job-weights", kind="repair-weights", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "repair-weights", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="repair-weights",
                submission_id="weights-1",
                source_artifact_id=source.id,
                run_id=run.id,
            )

            def repair(_source, output, _progress):
                trimesh.creation.box().export(output)
                return {"backend": "test-weight-repair"}

            blocked_report = {
                "skins": 1,
                "joints": 28,
                "joint_names": ["pelvis"],
                "skinned_vertices": 1000,
                "unweighted_vertices": 0,
                "non_normalized_vertices": 0,
                "invalid_joint_references": 0,
                "unused_joint_names": ["pelvis"],
                "joint_weight_coverage": [{"joint": "head", "total_weight": 1000}],
                "animations": [],
            }
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "run_repair_weights", side_effect=repair),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 5.0,
                    }),
                    patch.object(main, "analyze_rig_path", return_value=blocked_report),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                loaded = store.get(run.id)
                self.assertEqual(loaded.current_artifact_id, source.id)
                self.assertFalse(loaded.attempts[-1].promoted)
                candidate = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertEqual(
                    candidate.metadata["quality_decisions"]["rig-certificate"]["status"],
                    "blocked",
                )
            finally:
                main.run_store = previous_store

    def test_auto_rig_persists_a_skeleton_contact_sheet_alongside_the_rig(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "clean.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="clean-mesh", name="clean.glb", kind="glb",
                path=str(source_path), size_bytes=source_path.stat().st_size, created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind)
            job = main.JobState(id="job-auto-rig", kind="auto-rig", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "auto-rig", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="auto-rig",
                submission_id="auto-rig-sheet-1",
                source_artifact_id=source.id,
                run_id=run.id,
            )

            def auto_rig(_source, output, **_kwargs):
                trimesh.creation.box().export(output)
                return {"backend": "test-unirig"}

            rig_report = {
                "skins": 1, "joints": 28, "joint_names": ["pelvis"],
                "skinned_vertices": 1000, "unweighted_vertices": 0,
                "non_normalized_vertices": 0, "invalid_joint_references": 0,
                "unused_joint_names": ["pelvis"],
                "joint_weight_coverage": [{"joint": "head", "total_weight": 1000}],
                "animations": [],
            }
            sheet_png = root / "sheet.png"
            Image.new("RGB", (16, 16), (100, 100, 100)).save(sheet_png)

            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "unirig_ready", return_value=True),
                    patch.object(main.native, "run_unirig_auto_rig", side_effect=auto_rig),
                    patch.object(main, "analyze_rig_path", return_value=rig_report),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 5.0,
                    }),
                    patch.object(main.native, "run_skeleton_sheet", return_value={
                        "_review_files": [{"label": "skeleton-sheet", "path": str(sheet_png)}],
                    }),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                candidate = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                review = candidate.metadata["skeleton_review"]["review_artifacts"]
                self.assertEqual(len(review), 1)
                self.assertEqual(review[0]["label"], "skeleton-sheet")
                self.assertTrue((artifacts / f"{review[0]['artifact_id']}.json").exists())
                # The persisted temp render is deleted once it is copied into
                # the artifact store -- nothing durable is left behind at
                # the render's own throwaway path.
                self.assertFalse(sheet_png.exists())
            finally:
                main.run_store = previous_store

    def test_auto_rig_completes_with_a_warning_when_the_skeleton_sheet_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "clean.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="clean-mesh", name="clean.glb", kind="glb",
                path=str(source_path), size_bytes=source_path.stat().st_size, created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind)
            job = main.JobState(id="job-auto-rig-2", kind="auto-rig", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "auto-rig", source.id, {})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="auto-rig",
                submission_id="auto-rig-sheet-2",
                source_artifact_id=source.id,
                run_id=run.id,
            )

            def auto_rig(_source, output, **_kwargs):
                trimesh.creation.box().export(output)
                return {"backend": "test-unirig"}

            rig_report = {
                "skins": 1, "joints": 28, "joint_names": ["pelvis"],
                "skinned_vertices": 1000, "unweighted_vertices": 0,
                "non_normalized_vertices": 0, "invalid_joint_references": 0,
                "unused_joint_names": ["pelvis"],
                "joint_weight_coverage": [{"joint": "head", "total_weight": 1000}],
                "animations": [],
            }

            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "unirig_ready", return_value=True),
                    patch.object(main.native, "run_unirig_auto_rig", side_effect=auto_rig),
                    patch.object(main, "analyze_rig_path", return_value=rig_report),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 5.0,
                    }),
                    patch.object(
                        main.native, "run_skeleton_sheet",
                        side_effect=RuntimeError("Blender 4.2+ is required"),
                    ),
                ):
                    asyncio.run(main._execute_job(job, request))

                # A Blender-side rendering failure is advisory evidence lost,
                # not a rig defect -- the job that produced a valid rig must
                # still complete and promote normally.
                self.assertEqual(job.status, "completed")
                loaded = store.get(run.id)
                self.assertTrue(
                    any("Skeleton contact sheet" in warning for warning in loaded.attempts[-1].warnings)
                )
                candidate = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertNotIn("skeleton_review", candidate.metadata)
            finally:
                main.run_store = previous_store

    def test_rig_transfer_uses_persisted_donor_and_retains_visually_unreviewed_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "refined.glb"
            donor_path = root / "certified-rig.glb"
            trimesh.creation.box().export(source_path)
            trimesh.creation.box().export(donor_path)
            source = main.Artifact(id="refined", name="refined.glb", kind="glb", path=str(source_path), size_bytes=1, created_at=0)
            donor = main.Artifact(id="donor", name="certified.glb", kind="glb", path=str(donor_path), size_bytes=1, created_at=0)
            store = RunStore(root / "runs")
            run = store.create("Mary refined", source.id, source.kind)
            job = main.JobState(id="job-rig-transfer", kind="rig-transfer", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "rig-transfer", source.id, {"donor_artifact_id": donor.id})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="rig-transfer",
                submission_id="rig-transfer-1",
                source_artifact_id=source.id,
                run_id=run.id,
                settings={"donor_artifact_id": donor.id},
            )

            def transfer(_source, actual_donor, output, _progress):
                self.assertEqual(actual_donor, donor_path)
                trimesh.creation.box().export(output)
                return {"backend": "test-rig-transfer"}

            roles = [
                "pelvis", "spine", "head", "upperarm_l", "lowerarm_l", "hand_l",
                "upperarm_r", "lowerarm_r", "hand_r", "thigh_l", "calf_l", "foot_l",
                "thigh_r", "calf_r", "foot_r",
            ]
            ready_report = {
                "skins": 1,
                "joints": len(roles),
                "joint_names": roles,
                "skinned_vertices": 1000,
                "unweighted_vertices": 0,
                "non_normalized_vertices": 0,
                "invalid_joint_references": 0,
                "rigid_vertex_ratio": 0.5,
                "unused_joint_names": [],
                "joint_weight_coverage": [
                    {"joint": name, "vertices": 100, "total_weight": 1000 / len(roles)}
                    for name in roles
                ],
                "animations": [{"name": "Sword_Attack", "duration_seconds": 1.0, "animated_joint_coverage": 1.0, "invalid_targets": 0}],
            }
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, donor.id: donor}[artifact_id]),
                    patch.object(main.native, "run_rig_transfer", side_effect=transfer),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 5.0,
                    }),
                    patch.object(main, "analyze_rig_path", return_value=ready_report),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                self.assertEqual(store.get(run.id).current_artifact_id, source.id)
                output = main.Artifact.model_validate(main._read_json(artifacts / f"{job.output_artifact_id}.json"))
                self.assertEqual(output.metadata["donor_artifact_id"], donor.id)
                self.assertEqual(output.metadata["quality_decisions"]["rig-certificate"]["status"], "ready")
                self.assertEqual(output.metadata["quality_decisions"]["deformation-review"]["status"], "needs_review")
            finally:
                main.run_store = previous_store

    def test_retarget_motion_bakes_and_gates_on_the_selected_clip_not_sword_attack(self):
        # Regression: settings.animation ("Dance_Simple") must reach both
        # native.run_bvh_retarget's own `animation` kwarg (the baked output
        # action name) AND the animation-review gate's required_animation --
        # the gate used to hardcode BATCH_ONE_ATTACK ("Sword Attack"), which
        # would have flagged every non-sword-attack clip as missing its
        # required animation and permanently blocked promotion.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "rigged-mary.glb"
            trimesh.creation.box().export(source_path)
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {
                        "status": "ready", "policy_version": 2, "profile": "humanoid",
                        "resolved_roles": {"pelvis": "pelvis"},
                    },
                },
            }
            source = main.Artifact(id="rigged-mary", name="rigged-mary.glb", kind="glb", path=str(source_path), size_bytes=1, created_at=0, metadata=certificate)
            motion = main.Artifact(id="dance-bvh", name="dance.bvh", kind="bvh", path="dance.bvh", size_bytes=1, created_at=0)
            store = RunStore(root / "runs")
            run = store.create("Mary", source.id, source.kind, subject_kind="biped")
            job = main.JobState(id="job-retarget-dance", kind="retarget-motion", run_id=run.id)
            attempt = store.reserve_attempt(run.id, job.id, "retarget-motion", source.id, {"motion_artifact_id": motion.id, "animation": "Dance_Simple"})
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="retarget-motion",
                submission_id="retarget-dance-1",
                source_artifact_id=source.id,
                run_id=run.id,
                settings={"motion_artifact_id": motion.id, "animation": "Dance_Simple"},
            )

            captured_animation: list[str] = []

            def retarget(_character, _motion, output, *, target_bones, animation, progress=None):
                captured_animation.append(animation)
                trimesh.creation.box().export(output)
                return {"backend": "test-bvh-retarget", "animation": animation}

            roles = [
                "pelvis", "spine", "head", "upperarm_l", "lowerarm_l", "hand_l",
                "upperarm_r", "lowerarm_r", "hand_r", "thigh_l", "calf_l", "foot_l",
                "thigh_r", "calf_r", "foot_r",
            ]
            ready_report = {
                "skins": 1,
                "joints": len(roles),
                "joint_names": roles,
                "skinned_vertices": 1000,
                "unweighted_vertices": 0,
                "non_normalized_vertices": 0,
                "invalid_joint_references": 0,
                "rigid_vertex_ratio": 0.5,
                "unused_joint_names": [],
                "joint_weight_coverage": [
                    {"joint": name, "vertices": 100, "total_weight": 1000 / len(roles)}
                    for name in roles
                ],
                "animations": [{"name": "Dance_Simple", "duration_seconds": 1.0, "animated_joint_coverage": 1.0, "invalid_targets": 0}],
            }
            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, motion.id: motion}[artifact_id]),
                    patch.object(main.native, "run_bvh_retarget", side_effect=retarget),
                    patch.object(main, "analyze_rig_path", return_value=ready_report),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 5.0,
                    }),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(captured_animation, ["Dance_Simple"])
                output = main.Artifact.model_validate(main._read_json(artifacts / f"{job.output_artifact_id}.json"))
                animation_review = output.metadata["quality_decisions"]["animation-review"]
                self.assertNotIn(
                    "animation.missing",
                    [finding["code"] for finding in animation_review["findings"]],
                )
            finally:
                main.run_store = previous_store

    def test_curated_sword_rig_uses_bundled_donor_and_persists_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "refined.glb"
            curated_path = root / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb"
            trimesh.creation.box().export(source_path)
            curated_path.parent.mkdir(parents=True)
            curated_path.write_bytes(b"bundled-donor")
            source = main.Artifact(
                id="refined",
                name="refined.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Mary refined", source.id, source.kind)
            job = main.JobState(id="job-curated-sword", kind="target-proxy-rig", run_id=run.id)
            attempt = store.reserve_attempt(
                run.id,
                job.id,
                "target-proxy-rig",
                source.id,
                {"curated_motion": "grounded-sword-v1"},
            )
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="target-proxy-rig",
                submission_id="curated-sword-1",
                source_artifact_id=source.id,
                run_id=run.id,
                settings={
                    "curated_motion": "grounded-sword-v1",
                    "action_name": "Sword_Attack_Air_Vertical",
                    "deformation_surface": "voxel-proxy",
                },
            )
            curated = native.CuratedSwordDonor(
                path=curated_path,
                sha256="curated-donor-sha256",
                action_name="Sword_Regular_A",
            )
            roles = [
                "pelvis", "spine", "head", "upperarm_l", "lowerarm_l", "hand_l",
                "upperarm_r", "lowerarm_r", "hand_r", "thigh_l", "calf_l", "foot_l",
                "thigh_r", "calf_r", "foot_r",
            ]
            ready_report = {
                "skins": 1,
                "joints": len(roles),
                "joint_names": roles,
                "skinned_vertices": 1000,
                "unweighted_vertices": 0,
                "non_normalized_vertices": 0,
                "invalid_joint_references": 0,
                "rigid_vertex_ratio": 0.5,
                "unused_joint_names": [],
                "joint_weight_coverage": [
                    {"joint": name, "vertices": 100, "total_weight": 1000 / len(roles)}
                    for name in roles
                ],
                "animations": [{
                    "name": "Sword_Attack",
                    "duration_seconds": 1.0,
                    "animated_joint_coverage": 1.0,
                    "invalid_targets": 0,
                }],
            }
            calls = []

            def proxy_rig(actual_source, actual_donor, output, _progress, *, action_name, attach_sword, deformation_surface):
                calls.append((actual_source, actual_donor, output, action_name, attach_sword, deformation_surface))
                trimesh.creation.box().export(output)
                return {"backend": "test-target-proxy-rig"}

            previous_store = main.run_store
            previous_artifacts = main.ARTIFACTS
            previous_jobs_dir = main.JOBS_DIR
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "ROOT", root),
                    patch.object(main.native, "curated_sword_donor", return_value=curated),
                    patch.object(main.native, "run_target_proxy_rig", side_effect=proxy_rig),
                    patch.object(main, "analyze_path", return_value={
                        "faces": 1000,
                        "finite": True,
                        "largest_component_face_ratio": 1.0,
                        "non_manifold_edges": 0,
                        "degenerate_faces": 0,
                        "components": 1,
                        "watertight": True,
                    }),
                    patch.object(main, "analyze_rig_path", return_value=ready_report),
                    patch.object(main.native, "run_deformation_diagnostics", return_value={
                        "complete_action_range": True,
                        "frames": [{"frame": frame} for frame in range(5)],
                        "maximum_p99_edge_stretch": 1.0,
                    }),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "completed")
                self.assertFalse(job.promoted_output)
                self.assertEqual(store.get(run.id).current_artifact_id, source.id)
                self.assertEqual(
                    calls,
                    [(
                        source_path,
                        curated_path,
                        artifacts / f"job-{job.id}.glb",
                        "Sword_Regular_A",
                        True,
                        "voxel-proxy",
                    )],
                )
                output = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )
                self.assertEqual(output.metadata["motion_source"], "curated-grounded-sword-v1")
                self.assertEqual(output.metadata["donor_sha256"], curated.sha256)
                self.assertEqual(
                    output.metadata["donor_path"],
                    str(curated_path.relative_to(root)),
                )
            finally:
                main.run_store = previous_store
                main.ARTIFACTS = previous_artifacts
                main.JOBS_DIR = previous_jobs_dir

    def test_curated_sword_resolver_failure_does_not_launch_worker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = root / "source.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="source-resolver-failure",
                name="source.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            store = RunStore(root / "runs")
            run = store.create("Curated resolver failure", source.id, source.kind)
            job = main.JobState(
                id="job-curated-resolver-failure",
                kind="target-proxy-rig",
                run_id=run.id,
            )
            attempt = store.reserve_attempt(
                run.id,
                job.id,
                "target-proxy-rig",
                source.id,
                {"curated_motion": "grounded-sword-v1"},
            )
            job.attempt_id = attempt.id
            request = main.JobRequest(
                kind="target-proxy-rig",
                submission_id="curated-resolver-failure-1",
                source_artifact_id=source.id,
                run_id=run.id,
                settings={"curated_motion": "grounded-sword-v1"},
            )
            worker = Mock()
            previous_store = main.run_store
            previous_artifacts = main.ARTIFACTS
            previous_jobs_dir = main.JOBS_DIR
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(
                        main.native,
                        "curated_sword_donor",
                        side_effect=RuntimeError("bundled donor SHA-256 does not match expected value"),
                    ),
                    patch.object(main.native, "run_target_proxy_rig", worker),
                ):
                    asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "failed")
                self.assertIn("SHA-256", job.error or "")
                worker.assert_not_called()
                self.assertEqual(store.get(run.id).attempts[-1].status, "failed")
            finally:
                main.run_store = previous_store
                main.ARTIFACTS = previous_artifacts
                main.JOBS_DIR = previous_jobs_dir

    def test_target_proxy_rig_rejects_curated_and_explicit_donor_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            source_path = artifacts / "source.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="source-1",
                name="source.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            main._write_json(artifacts / f"{source.id}.json", source.model_dump())
            previous_store = main.run_store
            previous_artifacts = main.ARTIFACTS
            previous_jobs_dir = main.JOBS_DIR
            main.run_store = RunStore(root / "runs")
            main.ARTIFACTS = artifacts
            main.JOBS_DIR = jobs_dir
            try:
                response = client.post(
                    "/api/jobs",
                    json={
                        "kind": "target-proxy-rig",
                        "submission_id": "conflicting-sword-sources",
                        "source_artifact_id": source.id,
                        "settings": {
                            "curated_motion": "grounded-sword-v1",
                            "donor_artifact_id": "uploaded-donor",
                        },
                    },
                )

                self.assertEqual(response.status_code, 400, response.json())
                self.assertIn("both curated_motion and donor_artifact_id", response.json()["detail"])
            finally:
                main.run_store = previous_store
                main.ARTIFACTS = previous_artifacts
                main.JOBS_DIR = previous_jobs_dir

    def test_mesh_diagnostics_persist_on_artifact_and_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_artifacts = main.ARTIFACTS
            previous_store = main.run_store
            main.ARTIFACTS = root / "artifacts"
            main.ARTIFACTS.mkdir()
            main.run_store = RunStore(root / "runs")
            mesh_path = main.ARTIFACTS / "mesh.glb"
            trimesh.creation.box().export(mesh_path)
            artifact = main.Artifact(
                id="mesh-1",
                name="mary.glb",
                kind="glb",
                path=str(mesh_path),
                size_bytes=mesh_path.stat().st_size,
                created_at=0,
            )
            main._write_json(main.ARTIFACTS / "mesh-1.json", artifact.model_dump())
            run = main.run_store.create("Mary", artifact.id, artifact.kind)
            try:
                response = client.post(
                    f"/api/assets/{artifact.id}/diagnose",
                    json={"run_id": run.id},
                )

                self.assertEqual(response.status_code, 200, response.json())
                report = response.json()["metadata"]["mesh_quality"]
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["components"], 1)
                self.assertEqual(
                    response.json()["metadata"]["quality_decisions"]["rig"]["status"],
                    "blocked",
                )
                persisted = main.read_artifact(artifact.id)
                self.assertEqual(persisted.metadata["mesh_quality"], report)
                updated_run = main.run_store.get(run.id)
                self.assertEqual(updated_run.current_artifact_id, artifact.id)
                self.assertEqual(updated_run.attempts[-1].stage, "mesh-diagnostics")
                self.assertEqual(updated_run.attempts[-1].status, "completed")
            finally:
                main.ARTIFACTS = previous_artifacts
                main.run_store = previous_store

    def test_rig_diagnostics_persist_on_artifact_and_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_artifacts = main.ARTIFACTS
            previous_store = main.run_store
            main.ARTIFACTS = root / "artifacts"
            main.ARTIFACTS.mkdir()
            main.run_store = RunStore(root / "runs")
            mesh_path = main.ARTIFACTS / "model.glb"
            trimesh.creation.box().export(mesh_path)
            artifact = main.Artifact(
                id="model-1",
                name="mary.glb",
                kind="glb",
                path=str(mesh_path),
                size_bytes=mesh_path.stat().st_size,
                created_at=0,
            )
            main._write_json(main.ARTIFACTS / "model-1.json", artifact.model_dump())
            run = main.run_store.create("Mary", artifact.id, artifact.kind)
            try:
                response = client.post(
                    f"/api/assets/{artifact.id}/diagnose-rig",
                    json={"run_id": run.id},
                )

                self.assertEqual(response.status_code, 200, response.json())
                report = response.json()["metadata"]["rig_quality"]
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["skins"], 0)
                self.assertEqual(
                    response.json()["metadata"]["quality_decisions"]["animation-review"]["status"],
                    "blocked",
                )
                self.assertEqual(
                    response.json()["metadata"]["quality_decisions"]["rig-certificate"]["status"],
                    "blocked",
                )
                persisted = main.read_artifact(artifact.id)
                self.assertEqual(persisted.metadata["rig_quality"], report)
                updated_run = main.run_store.get(run.id)
                self.assertEqual(updated_run.attempts[-1].stage, "rig-diagnostics")
                self.assertEqual(updated_run.attempts[-1].status, "completed")
            finally:
                main.ARTIFACTS = previous_artifacts
                main.run_store = previous_store

    def test_deformation_diagnostics_can_review_an_existing_run_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_artifacts = main.ARTIFACTS
            previous_store = main.run_store
            main.ARTIFACTS = root / "artifacts"
            main.ARTIFACTS.mkdir()
            main.run_store = RunStore(root / "runs")
            mesh_path = main.ARTIFACTS / "model.glb"
            trimesh.creation.box().export(mesh_path)
            artifact = main.Artifact(
                id="model-1",
                name="mary.glb",
                kind="glb",
                path=str(mesh_path),
                size_bytes=mesh_path.stat().st_size,
                created_at=0,
            )
            main._write_json(main.ARTIFACTS / "model-1.json", artifact.model_dump())
            run = main.run_store.create("Mary", artifact.id, artifact.kind)
            report = {
                "schema_version": 1,
                "action": "Sword_Attack",
                "frames": [{"frame": 1, "p99_edge_stretch": 9.0}],
                "maximum_p99_edge_stretch": 9.0,
                "maximum_bounds_diagonal_ratio": 1.2,
            }
            try:
                with patch.object(main.native, "run_deformation_diagnostics", return_value=report):
                    response = client.post(
                        f"/api/assets/{artifact.id}/diagnose-deformation",
                        json={"run_id": run.id},
                    )

                self.assertEqual(response.status_code, 200, response.json())
                metadata = response.json()["metadata"]
                self.assertEqual(metadata["deformation_review"], report | {"review_artifacts": []})
                self.assertEqual(metadata["quality_decisions"]["deformation-review"]["status"], "blocked")
                persisted = main.read_artifact(artifact.id)
                self.assertEqual(persisted.metadata["deformation_review"]["action"], "Sword_Attack")
                updated_run = main.run_store.get(run.id)
                self.assertEqual(updated_run.current_artifact_id, artifact.id)
                self.assertEqual(updated_run.attempts[-1].stage, "deformation-review")
                self.assertEqual(updated_run.attempts[-1].status, "completed")
                self.assertFalse(updated_run.attempts[-1].promoted)
                evidence_id = updated_run.attempts[-1].output_artifact_id
                self.assertNotEqual(evidence_id, artifact.id)
                evidence = main.read_artifact(evidence_id)
                self.assertEqual(evidence.kind, "deformation-review")
                self.assertEqual(evidence.metadata["source_artifact_id"], artifact.id)
            finally:
                main.ARTIFACTS = previous_artifacts
                main.run_store = previous_store

    def test_restart_reconciliation_fails_interrupted_jobs_and_orphaned_attempts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "runs")
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            run = store.create("Mary", "source-1")
            attempt = store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})
            orphan = store.reserve_attempt(run.id, "job-missing", "retopo", "source-1", {})
            main._write_json(
                jobs_dir / "job-1.json",
                main.JobState(
                    id="job-1",
                    kind="generate",
                    submission_id="submission-1",
                    source_artifact_id="source-1",
                    run_id=run.id,
                    attempt_id=attempt.id,
                    status="running",
                ).as_dict(),
            )

            recovered = main.reconcile_interrupted_jobs(jobs_dir, store)

            self.assertEqual(recovered["job-1"].status, "failed")
            self.assertIn("restart", recovered["job-1"].error.lower())
            loaded = store.get(run.id)
            self.assertEqual(loaded.attempts[0].status, "failed")
            self.assertEqual(loaded.attempts[1].id, orphan.id)
            self.assertEqual(loaded.attempts[1].status, "failed")
            self.assertEqual(main._read_json(jobs_dir / "job-1.json")["status"], "failed")

    def test_restart_reconciliation_preserves_external_attempts_and_repairs_completed_jobs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "runs")
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            run = store.create("Mary", "source-1")
            external = store.start_attempt(run.id, "external", "source-1", {})
            completed_attempt = store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})
            main._write_json(
                jobs_dir / "job-1.json",
                main.JobState(
                    id="job-1",
                    kind="generate",
                    run_id=run.id,
                    attempt_id=completed_attempt.id,
                    status="completed",
                    output_artifact_id="mesh-1",
                ).as_dict(),
            )
            (jobs_dir / "broken.json").write_text("{", encoding="utf-8")

            main.reconcile_interrupted_jobs(jobs_dir, store)

            loaded = store.get(run.id)
            self.assertEqual(next(item for item in loaded.attempts if item.id == external.id).status, "running")
            repaired = next(item for item in loaded.attempts if item.id == completed_attempt.id)
            self.assertEqual(repaired.status, "completed")
            self.assertEqual(repaired.output_artifact_id, "mesh-1")
            self.assertFalse((jobs_dir / "broken.json").exists())
            self.assertEqual(len(list(jobs_dir.glob("broken.json.corrupt-*"))), 1)

    def test_job_submission_durably_creates_run_attempt_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="source-1", name="mary.png", kind="image", path="mary.png", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", source.id, source.kind)
            preparation = main.run_store.reserve_attempt(
                run.id, "prepare-job", "prepare-image", source.id, {"adapter": "upload"}
            )
            main.run_store.finish_attempt(
                run.id, preparation.id, "completed", output_artifact_id=source.id, promote_output=True,
            )
            payload = {
                "kind": "generate",
                "source_artifact_id": "source-1",
                "submission_id": "submission-1",
                "run_id": run.id,
            }
            try:
                with (
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    created = client.post("/api/jobs", json=payload)
                    duplicate = client.post("/api/jobs", json=payload)

                self.assertEqual(created.status_code, 200)
                self.assertEqual(duplicate.json()["id"], created.json()["id"])
                self.assertEqual(created.json()["submission_id"], "submission-1")
                self.assertTrue(created.json()["run_id"])
                self.assertTrue(created.json()["attempt_id"])
                run = main.run_store.get(created.json()["run_id"])
                self.assertEqual(len(run.attempts), 2)
                self.assertEqual(run.attempts[1].job_id, created.json()["id"])
                self.assertEqual(run.attempts[1].settings["resolution"], 768)
                self.assertEqual(run.attempts[1].settings["seed"], 0)
                self.assertEqual(run.attempts[1].settings["target_faces"], 10000)
                self.assertTrue((main.JOBS_DIR / f"{created.json()['id']}.json").exists())
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_source_views_rejects_unknown_view_key(self):
        payload = {
            "kind": "generate",
            "submission_id": "submission-bad-key",
            "source_views": {"front": "front-1", "top": "top-1"},
        }

        response = client.post("/api/jobs", json=payload)

        self.assertEqual(response.status_code, 422, response.json())

    def test_source_views_requires_front(self):
        payload = {
            "kind": "generate",
            "submission_id": "submission-no-front",
            "source_views": {"back": "back-1"},
        }

        response = client.post("/api/jobs", json=payload)

        self.assertEqual(response.status_code, 422, response.json())

    def test_source_views_rejects_duplicate_artifact_id_across_labels(self):
        payload = {
            "kind": "generate",
            "submission_id": "submission-duplicate-view",
            "source_views": {"front": "same-artifact", "back": "same-artifact"},
        }

        response = client.post("/api/jobs", json=payload)

        self.assertEqual(response.status_code, 422, response.json())

    def test_source_views_must_agree_with_source_artifact_id(self):
        payload = {
            "kind": "generate",
            "submission_id": "submission-disagreement",
            "source_artifact_id": "front-1",
            "source_views": {"front": "front-2"},
        }

        response = client.post("/api/jobs", json=payload)

        self.assertEqual(response.status_code, 422, response.json())

    def test_job_generate_with_source_views_stores_front_as_source_and_persists_views(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            front = main.Artifact(id="front-1", name="front.png", kind="image", path="front.png", size_bytes=1, created_at=0)
            back = main.Artifact(id="back-1", name="back.png", kind="image", path="back.png", size_bytes=1, created_at=0)
            artifacts = {front.id: front, back.id: back}
            payload = {
                "kind": "generate",
                "submission_id": "submission-mv-1",
                "source_views": {"front": "front-1", "back": "back-1"},
            }
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: artifacts[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    created = client.post("/api/jobs", json=payload)

                self.assertEqual(created.status_code, 200, created.json())
                self.assertEqual(created.json()["source_artifact_id"], "front-1")
                stored_payload = created.json()["request_payload"]
                self.assertEqual(stored_payload["source_views"], {"front": "front-1", "back": "back-1"})
                self.assertEqual(stored_payload["source_artifact_id"], "front-1")
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_source_views_rejects_non_image_artifact_as_a_view(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            front = main.Artifact(id="front-1", name="front.png", kind="image", path="front.png", size_bytes=1, created_at=0)
            # A GLB mesh is not a valid conditioning image for any view.
            left = main.Artifact(id="left-1", name="left.glb", kind="glb", path="left.glb", size_bytes=1, created_at=0)
            artifacts = {front.id: front, left.id: left}
            payload = {
                "kind": "generate",
                "submission_id": "submission-glb-view",
                "source_views": {"front": "front-1", "left": "left-1"},
            }
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: artifacts[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json=payload)

                self.assertEqual(response.status_code, 422, response.json())
                self.assertIn("left", response.json()["detail"])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_cancelling_waiting_job_does_not_terminate_active_job_process(self):
        with TemporaryDirectory() as directory:
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            previous_active_job_id = getattr(main.model_manager, "active_job_id", None)
            main.JOBS_DIR = Path(directory)
            active = main.JobState(id="active-job", kind="generate", submission_id="active", status="running")
            waiting = main.JobState(id="waiting-job", kind="generate-motion", submission_id="waiting", status="running")
            main.jobs.clear()
            main.jobs.update({active.id: active, waiting.id: waiting})
            main.model_manager.active_job_id = active.id
            try:
                with patch.object(main.native, "cancel_active_process") as cancel_process:
                    response = client.post(f"/api/jobs/{waiting.id}/cancel")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "cancelled")
                cancel_process.assert_not_called()
            finally:
                main.model_manager.active_job_id = previous_active_job_id
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
            main.jobs.update(previous_jobs)

    def test_cancelling_active_job_terminates_its_process_and_logs_it(self):
        """bead xckqr: cancelling the job actually holding the GPU worker
        must terminate it and record what happened in the job's own log --
        not just flip a status flag (the 2026-08-31 incident)."""
        with TemporaryDirectory() as directory:
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            previous_active_job_id = getattr(main.model_manager, "active_job_id", None)
            main.JOBS_DIR = Path(directory)
            active = main.JobState(id="active-job-2", kind="generate", submission_id="active", status="running")
            main.jobs.clear()
            main.jobs.update({active.id: active})
            main.model_manager.active_job_id = active.id
            try:
                with patch.object(main.native, "cancel_active_process", return_value=True) as cancel_process:
                    response = client.post(f"/api/jobs/{active.id}/cancel")

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["status"], "cancelled")
                cancel_process.assert_called_once()
                self.assertIn("Worker process confirmed terminated", body["logs"])
            finally:
                main.model_manager.active_job_id = previous_active_job_id
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
            main.jobs.update(previous_jobs)

    def test_cancelled_blender_job_cannot_publish_a_late_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source_path = root / "source.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="source",
                name="source.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            job = main.JobState(id="paint-job", kind="paint-bake")
            request = main.JobRequest(
                kind="paint-bake",
                submission_id="paint-cancel",
                source_artifact_id=source.id,
            )

            def finish_after_cancel(_source, output, _settings, _progress):
                job.status = "cancelled"
                trimesh.creation.box().export(output)
                return {"backend": "late-blender"}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_paint", side_effect=finish_after_cancel),
            ):
                asyncio.run(main._execute_job(job, request))

            self.assertEqual(job.status, "cancelled")
            self.assertIsNone(job.output_artifact_id)
            self.assertEqual(list(artifacts.iterdir()), [])

    def test_cancellation_during_diagnostics_cannot_publish_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source_path = root / "source.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(id="source", name="source.glb", kind="glb", path=str(source_path), size_bytes=1, created_at=0)
            job = main.JobState(id="paint-job", kind="paint-bake")
            request = main.JobRequest(kind="paint-bake", submission_id="paint-cancel-diagnostics", source_artifact_id=source.id)

            def paint(_source, output, _settings, _progress):
                trimesh.creation.box().export(output)
                return {"backend": "blender"}

            def diagnose(_output):
                job.status = "cancelled"
                return {"faces": 12, "finite": True}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_paint", side_effect=paint),
                patch.object(main, "analyze_path", side_effect=diagnose),
            ):
                asyncio.run(main._execute_job(job, request))

            self.assertEqual(job.status, "cancelled")
            self.assertIsNone(job.output_artifact_id)
            self.assertEqual(list(artifacts.iterdir()), [])

    def test_project_views_job_records_verdict_and_promotes_on_green_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source_path = root / "canonical.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="canonical-glb",
                name="canonical.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                # `_resolve_conditioning_view_paths` finds these directly on
                # the source artifact, so the lineage walk never needs to
                # climb further up (main.read_artifact is patched to a fixed
                # return value below, which is enough since every resolved
                # role id just needs to come back as SOME GLB-bearing path).
                metadata={
                    "source_views": {
                        "front": "view-front", "back": "view-back", "left": "view-left",
                    }
                },
            )
            job = main.JobState(id="project-views-job", kind="project-views")
            request = main.JobRequest(
                kind="project-views",
                submission_id="project-views-green",
                source_artifact_id=source.id,
            )
            report = {
                "schema_version": 1,
                "views_used": ["back", "front", "left"],
                "per_role_coverage": {"front": 0.5, "back": 0.4, "left": 0.3},
                "projected_texel_fraction": 0.72,
                "fallback_texel_fraction": 0.28,
                "occlusion_rejected_fraction": {"front": 0.05, "back": 0.02, "left": 0.01},
                "mean_color_delta_vs_bake": 0.03,
                "atlas_size": 2048,
                "frame_alignment": {},
                "caveats": [],
            }

            def run_project_views(_source, output, _views, _settings, _progress):
                trimesh.creation.box().export(output)
                return {**report, "backend": "blender-view-projection"}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_project_views", side_effect=run_project_views),
            ):
                asyncio.run(main._execute_job(job, request))

            self.assertEqual(job.status, "completed")
            self.assertTrue(job.promoted_output)
            self.assertIsNotNone(job.output_artifact_id)
            output = main.Artifact.model_validate(
                main._read_json(artifacts / f"{job.output_artifact_id}.json")
            )
            self.assertEqual(output.metadata["stage"], "project-views")
            self.assertEqual(output.metadata["conditioning_views_resolved"], ["back", "front", "left"])
            decision = output.metadata["quality_decisions"]["project_views"]
            self.assertEqual(decision["status"], "green")
            self.assertEqual(decision["policy_version"], "projection-v1")

    def test_project_views_job_requires_two_resolvable_views(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source_path = root / "canonical.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="canonical-glb-2",
                name="canonical.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
                metadata={"source_views": {"front": "view-front"}},
            )
            job = main.JobState(id="project-views-job-2", kind="project-views")
            request = main.JobRequest(
                kind="project-views",
                submission_id="project-views-insufficient",
                source_artifact_id=source.id,
            )

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_project_views") as run_project_views,
            ):
                asyncio.run(main._execute_job(job, request))

            run_project_views.assert_not_called()
            self.assertEqual(job.status, "failed")
            self.assertIsNone(job.output_artifact_id)

    @staticmethod
    def _rebuild_views_source(source_id: str, roles: list[str], source_path: Path) -> "main.Artifact":
        trimesh.creation.box().export(source_path)
        return main.Artifact(
            id=source_id,
            name="canonical.glb",
            kind="glb",
            path=str(source_path),
            size_bytes=source_path.stat().st_size,
            created_at=0,
            # Same shortcut the project-views tests take: read_artifact is
            # patched to a fixed return value, so every role resolves.
            metadata={"source_views": {role: f"view-{role}" for role in roles}},
        )

    def test_rebuild_views_job_passes_only_front_and_back_by_default(self):
        """THE RULE: left/right are three-quarter views in the T-pose set and
        repaint the face from the side, so they are never passed by default."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            final_glb = root / "final.glb"
            trimesh.creation.box().export(final_glb)
            source = self._rebuild_views_source(
                "rebuild-source", ["front", "back", "left", "right"], root / "canonical.glb"
            )
            job = main.JobState(id="rebuild-views-job", kind="rebuild-views")
            request = main.JobRequest(
                kind="rebuild-views",
                submission_id="rebuild-views-default",
                source_artifact_id=source.id,
            )
            captured: dict = {}

            def run_rebuild_views(_source, output, views, _settings, _progress):
                captured["views"] = dict(views)
                trimesh.creation.box().export(output)
                return {
                    "backend": "pixal3d-mv",
                    "mv_iou": {"mean_iou": 0.94, "front": {"iou": 0.95}},
                    "gate": {"front": {"masked_ssim": 0.544, "face_ssim": 0.32}},
                    "pixal_mv_glb": str(final_glb),
                    "workdir": str(root),
                }

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_rebuild_views", side_effect=run_rebuild_views),
            ):
                asyncio.run(main._execute_job(job, request))

            self.assertEqual(job.status, "completed")
            self.assertEqual(sorted(captured["views"]), ["back", "front"])
            output = main.Artifact.model_validate(
                main._read_json(artifacts / f"{job.output_artifact_id}.json")
            )
            self.assertEqual(output.metadata["stage"], "rebuild-views")
            self.assertEqual(output.metadata["reference_roles"], ["front", "back"])
            self.assertEqual(sorted(output.metadata["references"]), ["back", "front"])
            self.assertEqual(output.metadata["mv_iou"]["mean_iou"], 0.94)
            pixal_mv = main.Artifact.model_validate(
                main._read_json(artifacts / f"{output.metadata['pixal_mv_artifact_id']}.json")
            )
            self.assertEqual(pixal_mv.kind, "glb")
            self.assertEqual(pixal_mv.metadata["role"], "pixal-mv")
            self.assertEqual(pixal_mv.metadata["stage"], "rebuild-views")
            self.assertEqual(pixal_mv.metadata["for_job_id"], job.id)

    def test_rebuild_views_reference_roles_setting_overrides_the_default(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = self._rebuild_views_source(
                "rebuild-source-2", ["front", "back"], root / "canonical.glb"
            )
            job = main.JobState(id="rebuild-views-job-2", kind="rebuild-views")
            request = main.JobRequest(
                kind="rebuild-views",
                submission_id="rebuild-views-front-only",
                source_artifact_id=source.id,
                settings={"reference_roles": ["front"]},
            )
            captured: dict = {}

            def run_rebuild_views(_source, output, views, _settings, _progress):
                captured["views"] = dict(views)
                trimesh.creation.box().export(output)
                return {"backend": "pixal3d-mv", "mv_iou": {}, "gate": None,
                        "pixal_mv_glb": str(root / "missing.glb"), "workdir": str(root)}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_rebuild_views", side_effect=run_rebuild_views),
            ):
                asyncio.run(main._execute_job(job, request))

            self.assertEqual(job.status, "completed")
            self.assertEqual(list(captured["views"]), ["front"])
            output = main.Artifact.model_validate(
                main._read_json(artifacts / f"{job.output_artifact_id}.json")
            )
            self.assertEqual(output.metadata["reference_roles"], ["front"])
            self.assertNotIn("pixal_mv_artifact_id", output.metadata)

    def test_rebuild_views_job_fails_without_a_resolvable_front(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = self._rebuild_views_source(
                "rebuild-source-3", ["back", "left"], root / "canonical.glb"
            )
            job = main.JobState(id="rebuild-views-job-3", kind="rebuild-views")
            request = main.JobRequest(
                kind="rebuild-views",
                submission_id="rebuild-views-no-front",
                source_artifact_id=source.id,
            )

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main.native, "run_rebuild_views") as run_rebuild_views,
            ):
                asyncio.run(main._execute_job(job, request))

            run_rebuild_views.assert_not_called()
            self.assertEqual(job.status, "failed")
            self.assertIsNone(job.output_artifact_id)
            self.assertIn("front reference view", job.error)

    def test_retry_replays_a_persisted_backend_attempt_with_overrides(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="source-1", name="mary.png", kind="image", path="mary.png", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", source.id, source.kind)
            preparation = main.run_store.reserve_attempt(
                run.id, "prepare-job", "prepare-image", source.id, {"adapter": "upload"}
            )
            main.run_store.finish_attempt(
                run.id, preparation.id, "completed", output_artifact_id=source.id, promote_output=True,
            )
            original_request = main.JobRequest(
                kind="generate",
                submission_id="original-submission",
                source_artifact_id=source.id,
                run_id=run.id,
                resolution=768,
                seed=7,
                settings={"remove_background": True},
            )
            original_job = main.JobState(
                id="job-original",
                kind="generate",
                submission_id=original_request.submission_id,
                source_artifact_id=source.id,
                run_id=run.id,
                request_payload=original_request.model_dump(mode="json"),
                status="failed",
                error="bad silhouette",
            )
            original_attempt = main.run_store.reserve_attempt(
                run.id,
                original_job.id,
                "generate",
                source.id,
                {"resolution": 768, "seed": 7},
            )
            original_job.attempt_id = original_attempt.id
            main.run_store.finish_attempt(run.id, original_attempt.id, "failed", error=original_job.error)
            main._write_json(main.JOBS_DIR / f"{original_job.id}.json", original_job.as_dict())

            try:
                with (
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    payload = {
                        "submission_id": "retry-submission",
                        "seed": 9,
                        "settings": {"remove_background": False},
                    }
                    retried = client.post(f"/api/runs/{run.id}/attempts/{original_attempt.id}/retry", json=payload)
                    duplicate = client.post(f"/api/runs/{run.id}/attempts/{original_attempt.id}/retry", json=payload)

                self.assertEqual(retried.status_code, 200)
                self.assertEqual(duplicate.json()["id"], retried.json()["id"])
                self.assertEqual(retried.json()["retry_of_attempt_id"], original_attempt.id)
                self.assertEqual(retried.json()["request_payload"]["seed"], 9)
                self.assertEqual(retried.json()["request_payload"]["resolution"], 768)
                self.assertEqual(retried.json()["request_payload"]["settings"], {"remove_background": False})
                attempts = main.run_store.get(run.id).attempts
                self.assertEqual(len(attempts), 3)
                self.assertEqual(attempts[2].retry_of_attempt_id, original_attempt.id)
                self.assertEqual(attempts[2].input_artifact_id, source.id)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_quality_action_executes_reconstruction_with_retained_source_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            image = main.Artifact(id="image-1", name="mary.png", kind="image", path="mary.png", size_bytes=1, created_at=0)
            mesh = main.Artifact(
                id="mesh-1",
                name="mary.glb",
                kind="glb",
                path="mary.glb",
                size_bytes=1,
                created_at=0,
                metadata={
                    "quality_decisions": {
                        "rig": {
                            "status": "blocked",
                            "recommended_actions": ["refine-reconstruct", "repair-topology"],
                        }
                    }
                },
            )
            run = main.run_store.create("Mary", image.id, image.kind)
            generated = main.run_store.start_attempt(run.id, "generate", image.id, {})
            main.run_store.finish_attempt(run.id, generated.id, "completed", output_artifact_id=mesh.id)

            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {image.id: image, mesh.id: mesh}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    payload = {
                        "submission_id": "quality-refine-1",
                        "target_vertices": 50000,
                        "settings": {"repair_holes": False},
                    }
                    created = client.post(f"/api/runs/{run.id}/quality-actions/refine-reconstruct", json=payload)
                    duplicate = client.post(f"/api/runs/{run.id}/quality-actions/refine-reconstruct", json=payload)

                self.assertEqual(created.status_code, 200)
                self.assertEqual(duplicate.json()["id"], created.json()["id"])
                self.assertEqual(created.json()["kind"], "retopologize")
                self.assertEqual(created.json()["source_artifact_id"], mesh.id)
                self.assertEqual(created.json()["request_payload"]["target_vertices"], 50000)
                self.assertEqual(created.json()["request_payload"]["settings"]["refine_image_artifact_id"], image.id)
                attempt = main.run_store.get(run.id).attempts[-1]
                self.assertEqual(attempt.stage, "retopologize")
                self.assertEqual(attempt.input_artifact_id, mesh.id)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_quality_action_executes_blender_weight_repair_on_current_rig(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            rig = main.Artifact(
                id="rig-1",
                name="mary-rig.glb",
                kind="glb",
                path="mary-rig.glb",
                size_bytes=1,
                created_at=0,
                metadata={
                    "quality_decisions": {
                        "rig-certificate": {
                            "status": "blocked",
                            "recommended_actions": ["repair-weights"],
                        }
                    }
                },
            )
            run = main.run_store.create("Mary", rig.id, rig.kind)

            try:
                with (
                    patch.object(main, "read_artifact", return_value=rig),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    created = client.post(
                        f"/api/runs/{run.id}/quality-actions/repair-weights",
                        json={"submission_id": "quality-weights-1"},
                    )

                self.assertEqual(created.status_code, 200, created.json())
                self.assertEqual(created.json()["kind"], "repair-weights")
                self.assertEqual(created.json()["source_artifact_id"], rig.id)
                self.assertEqual(
                    created.json()["request_payload"]["settings"]["quality_action"],
                    "repair-weights",
                )
                attempt = main.run_store.get(run.id).attempts[-1]
                self.assertEqual(attempt.stage, "repair-weights")
                self.assertEqual(attempt.input_artifact_id, rig.id)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_quality_action_does_not_repeat_the_same_bind_fit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            mesh = main.Artifact(id="mesh-1", name="mary.glb", kind="glb", path="mesh.glb", size_bytes=1, created_at=0)
            donor = main.Artifact(id="donor-1", name="donor.glb", kind="glb", path="donor.glb", size_bytes=1, created_at=0)
            rig = main.Artifact(
                id="rig-1",
                name="mary-rig.glb",
                kind="glb",
                path="rig.glb",
                size_bytes=1,
                created_at=0,
                metadata={
                    "quality_decisions": {
                        "rig-certificate": {
                            "status": "blocked",
                            "recommended_actions": ["fit-bind-pose"],
                        }
                    }
                },
            )
            run = main.run_store.create("Mary", mesh.id, mesh.kind)
            original = main.run_store.start_attempt(
                run.id,
                "target-proxy-rig",
                mesh.id,
                {"donor_artifact_id": donor.id, "action_name": "Fighting Right Jab"},
            )
            main.run_store.finish_attempt(
                run.id,
                original.id,
                "completed",
                output_artifact_id=rig.id,
                promote_output=False,
            )

            try:
                artifacts = {mesh.id: mesh, donor.id: donor, rig.id: rig}
                with (
                    patch.object(main, "read_artifact", side_effect=artifacts.__getitem__),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    created = client.post(
                        f"/api/runs/{run.id}/quality-actions/fit-bind-pose",
                        json={"submission_id": "quality-fit-1"},
                    )

                self.assertEqual(created.status_code, 404, created.json())
                self.assertEqual(created.json()["detail"], "Unknown quality action")
                self.assertEqual(main.run_store.get(run.id).current_artifact_id, mesh.id)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_quality_action_rejects_an_unrecommended_or_manual_action(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            main.run_store = RunStore(root / "runs")
            image = main.Artifact(id="image-1", name="mary.png", kind="image", path="mary.png", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", image.id, image.kind)
            try:
                with patch.object(main, "read_artifact", return_value=image):
                    unrecommended = client.post(
                        f"/api/runs/{run.id}/quality-actions/repair-topology",
                        json={"submission_id": "quality-repair-1"},
                    )
                    manual = client.post(
                        f"/api/runs/{run.id}/quality-actions/inspect-weight-paint",
                        json={"submission_id": "quality-manual-1"},
                    )

                self.assertEqual(unrecommended.status_code, 409)
                self.assertIn("not recommended", unrecommended.json()["detail"])
                self.assertEqual(manual.status_code, 409)
                self.assertIn("manual", manual.json()["detail"])
            finally:
                main.run_store = previous_store

    def test_retained_candidate_can_be_promoted_through_the_api(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            main.run_store = RunStore(Path(directory) / "runs")
            run = main.run_store.create("Mary", "source-1", "glb")
            attempt = main.run_store.reserve_attempt(run.id, "job-1", "retopo", "source-1", {})
            main.run_store.finish_attempt(
                run.id,
                attempt.id,
                "completed",
                output_artifact_id="candidate-1",
                promote_output=False,
            )
            candidate = main.Artifact(id="candidate-1", name="candidate.glb", kind="glb", path="candidate.glb", size_bytes=1, created_at=0)
            try:
                with patch.object(main, "read_artifact", return_value=candidate):
                    response = client.post(
                        f"/api/runs/{run.id}/artifacts/{candidate.id}/promote",
                        json={"reason": "Human accepted visual quality"},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["current_artifact_id"], candidate.id)
                self.assertEqual(response.json()["attempts"][-1]["stage"], "promote-artifact")
            finally:
                main.run_store = previous_store

    def test_bvh_motion_child_cannot_replace_the_current_character(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            main.run_store = RunStore(Path(directory) / "runs")
            run = main.run_store.create("Mary", "character-1", "glb")
            attempt = main.run_store.reserve_attempt(run.id, "job-motion", "generate-motion", "character-1", {})
            main.run_store.finish_attempt(
                run.id,
                attempt.id,
                "completed",
                output_artifact_id="motion-1",
                promote_output=False,
            )
            motion = main.Artifact(id="motion-1", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            try:
                with patch.object(main, "read_artifact", return_value=motion):
                    response = client.post(
                        f"/api/runs/{run.id}/artifacts/{motion.id}/promote",
                        json={"reason": "Should remain a child motion"},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertEqual(main.run_store.get(run.id).current_artifact_id, "character-1")
            finally:
                main.run_store = previous_store

    def test_uploaded_stage_is_durably_attached_and_idempotent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="mesh-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0)
            output = main.Artifact(
                id="rigged-1",
                name="mary-rigged.glb",
                kind="glb",
                path="mary-rigged.glb",
                size_bytes=1,
                created_at=0,
                metadata={"quality_decisions": {
                    name: {"status": "ready"}
                    for name in (
                        "rig", "rig-certificate", "animation-review", "deformation-review",
                    )
                }},
            )
            blocked_output = output.model_copy(deep=True)
            blocked_output.id = "rigged-blocked"
            blocked_output.metadata["quality_decisions"]["rig"]["status"] = "blocked"
            payload = {
                "submission_id": "rig-submission-1",
                "stage": "rig",
                "input_artifact_id": source.id,
                "output_artifact_id": output.id,
                "run_name": "Mary rig",
                "settings": {"skeleton": "Human", "animations": ["Sword Attack"]},
            }
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {
                        source.id: source,
                        output.id: output,
                        blocked_output.id: blocked_output,
                    }[artifact_id]),
                    patch.object(main, "_attach_rig_intake_diagnostics", side_effect=lambda artifact: artifact) as diagnostics,
                ):
                    created = client.post("/api/runs/record-artifact", json=payload)
                    duplicate = client.post("/api/runs/record-artifact", json=payload)
                    blocked = client.post("/api/runs/record-artifact", json={
                        **payload,
                        "submission_id": "rig-submission-blocked",
                        "output_artifact_id": blocked_output.id,
                        "run_id": created.json()["id"],
                        "run_name": None,
                    })

                self.assertEqual(created.status_code, 200)
                self.assertEqual(duplicate.json()["id"], created.json()["id"])
                self.assertEqual(created.json()["source_artifact_kind"], "glb")
                self.assertEqual(created.json()["current_artifact_id"], output.id)
                self.assertEqual(len(created.json()["attempts"]), 1)
                self.assertEqual(created.json()["attempts"][0]["status"], "completed")
                self.assertEqual(blocked.status_code, 200, blocked.json())
                self.assertEqual(blocked.json()["current_artifact_id"], output.id)
                self.assertFalse(blocked.json()["attempts"][-1]["promoted"])
                self.assertIn("Rig export retained", blocked.json()["attempts"][-1]["warnings"][0])
                self.assertEqual(diagnostics.call_count, 2)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_record_submission_serializes_overlapping_diagnostics(self):
        request = main.ArtifactStageRequest(
            submission_id="same-rig-submission",
            stage="rig",
            input_artifact_id="mesh",
            output_artifact_id="rigged",
        )
        active = 0
        maximum_active = 0

        async def record(_request):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0)
            active -= 1
            return {"ok": True}

        async def exercise():
            with patch.object(main, "_record_artifact_stage", side_effect=record):
                await asyncio.gather(
                    main.record_artifact_stage(request),
                    main.record_artifact_stage(request),
                )

        asyncio.run(exercise())

        self.assertEqual(maximum_active, 1)

    def test_recorded_motion_child_does_not_replace_the_current_glb(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = main.Artifact(
                id="mary-rigged",
                name="mary-rigged.glb",
                kind="glb",
                path="mary-rigged.glb",
                size_bytes=1,
                created_at=0,
            )
            motion = main.Artifact(
                id="mary-attack",
                name="mary-attack.bvh",
                kind="bvh",
                path="mary-attack.bvh",
                size_bytes=1,
                created_at=0,
            )
            run = main.run_store.create("Mary", character.id, character.kind)
            payload = {
                "submission_id": "motion-submission-1",
                "stage": "generate-motion",
                "input_artifact_id": character.id,
                "output_artifact_id": motion.id,
                "run_id": run.id,
            }
            try:
                with patch.object(
                    main,
                    "read_artifact",
                    side_effect=lambda artifact_id: {
                        character.id: character,
                        motion.id: motion,
                    }[artifact_id],
                ):
                    recorded = client.post("/api/runs/record-artifact", json=payload)

                self.assertEqual(recorded.status_code, 200)
                self.assertEqual(recorded.json()["current_artifact_id"], character.id)
                self.assertFalse(recorded.json()["attempts"][0]["promoted"])
                self.assertEqual(recorded.json()["attempts"][0]["output_artifact_id"], motion.id)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_recorded_voxel_proxy_rig_is_retained_even_when_quality_is_ready(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(
                id="mary-mesh", name="mary.glb", kind="glb", path="mary.glb",
                size_bytes=1, created_at=0,
            )
            decisions = {
                name: {"status": "ready"}
                for name in ("rig", "rig-certificate", "animation-review", "deformation-review")
            }
            output = main.Artifact(
                id="mary-voxel-rig", name="mary-voxel.glb", kind="glb",
                path="mary-voxel.glb", size_bytes=1, created_at=0,
                metadata={"quality_decisions": decisions},
            )
            run = main.run_store.create("Mary", source.id, source.kind)
            try:
                with (
                    patch.object(
                        main,
                        "read_artifact",
                        side_effect=lambda artifact_id: {
                            source.id: source,
                            output.id: output,
                        }[artifact_id],
                    ),
                    patch.object(
                        main,
                        "_attach_serialized_rig_intake_diagnostics",
                        new=AsyncMock(return_value=output),
                    ),
                ):
                    response = client.post("/api/runs/record-artifact", json={
                        "submission_id": "record-voxel-proxy",
                        "stage": "target-proxy-rig",
                        "input_artifact_id": source.id,
                        "output_artifact_id": output.id,
                        "run_id": run.id,
                        "settings": {"deformation_surface": "voxel-proxy"},
                    })

                self.assertEqual(response.status_code, 200, response.json())
                self.assertEqual(response.json()["current_artifact_id"], source.id)
                self.assertFalse(response.json()["attempts"][0]["promoted"])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_uploaded_rig_diagnostics_run_outside_submission_file_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="mesh", name="mesh.glb", kind="glb", path="mesh.glb", size_bytes=1, created_at=0)
            output = main.Artifact(id="rig", name="rig.glb", kind="glb", path="rig.glb", size_bytes=1, created_at=0)
            lock_state = {"held": False}

            class TrackingLock:
                def __init__(self, _path):
                    pass

                def __enter__(self):
                    lock_state["held"] = True

                def __exit__(self, *_args):
                    lock_state["held"] = False

            def diagnose(artifact):
                self.assertFalse(lock_state["held"])
                return artifact

            try:
                with (
                    patch.object(main, "FileLock", TrackingLock),
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, output.id: output}[artifact_id]),
                    patch.object(main, "_attach_rig_intake_diagnostics", side_effect=diagnose),
                ):
                    response = client.post("/api/runs/record-artifact", json={
                        "submission_id": "outside-lock",
                        "stage": "rig",
                        "input_artifact_id": source.id,
                        "output_artifact_id": output.id,
                    })

                self.assertEqual(response.status_code, 200, response.json())
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_recorded_rig_validates_run_membership_before_diagnostics(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            run = main.run_store.create("Mary", "mary-source", "image")
            outsider = main.Artifact(
                id="zix-mesh", name="zix.glb", kind="glb", path="zix.glb",
                size_bytes=1, created_at=0,
            )
            output = main.Artifact(
                id="zix-rig", name="zix-rig.glb", kind="glb", path="zix-rig.glb",
                size_bytes=1, created_at=0,
            )
            diagnostics = AsyncMock(return_value=output)
            try:
                with (
                    patch.object(
                        main,
                        "read_artifact",
                        side_effect=lambda artifact_id: {
                            outsider.id: outsider,
                            output.id: output,
                        }[artifact_id],
                    ),
                    patch.object(
                        main,
                        "_attach_serialized_rig_intake_diagnostics",
                        new=diagnostics,
                    ),
                ):
                    response = client.post("/api/runs/record-artifact", json={
                        "submission_id": "wrong-run-rig",
                        "stage": "rig",
                        "input_artifact_id": outsider.id,
                        "output_artifact_id": output.id,
                        "run_id": run.id,
                    })

                self.assertEqual(response.status_code, 409, response.json())
                diagnostics.assert_not_awaited()
            finally:
                main.run_store = previous

    @staticmethod
    def _diagnose_rig_with(metadata, sheet):
        """Run `_attach_rig_diagnostics` on a GLB whose numeric checks all
        pass, with `sheet` standing in for `native.run_skeleton_sheet`."""
        artifact = main.Artifact(
            id="rigged-mary", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0,
            metadata=metadata,
        )
        with (
            patch.object(main, "_artifact_path", return_value=Path("mary.glb")),
            patch.object(main, "analyze_rig_path", return_value={"animations": []}),
            patch.object(main, "evaluate_for_rigging", return_value={"status": "ready"}),
            patch.object(main, "evaluate_rig_certificate", return_value={"status": "ready"}),
            patch.object(main, "evaluate_for_animation", return_value={"status": "ready"}),
            patch.object(main, "_write_json"),
            patch.object(main.native, "run_skeleton_sheet", sheet),
        ):
            return main._attach_rig_diagnostics(artifact)

    def test_rig_diagnostics_survive_a_skeleton_sheet_that_cannot_render(self):
        # No Blender on this box: the contact sheet is advisory evidence,
        # exactly as on the job path -- diagnose-rig must still answer with
        # the numeric report rather than fail outright.
        diagnosed = self._diagnose_rig_with(
            {"stage": "auto-rig", "mesh_quality": {}},
            Mock(side_effect=RuntimeError("Blender 4.2+ is required")),
        )

        self.assertEqual(diagnosed.metadata["quality_decisions"]["rig-certificate"]["status"], "ready")
        self.assertNotIn("skeleton_review", diagnosed.metadata)
        self.assertIn("Blender 4.2+", diagnosed.metadata["skeleton_review_error"])

    def test_rig_diagnostics_do_not_rerender_an_existing_skeleton_sheet(self):
        # Re-diagnosing must not re-render and orphan the previous sheet
        # artifacts: a `skeleton_review` already on the artifact is kept.
        existing = {"review_artifacts": [{"artifact_id": "sheet-1", "label": "skeleton-sheet"}]}
        sheet = Mock()
        diagnosed = self._diagnose_rig_with(
            {"stage": "auto-rig", "mesh_quality": {}, "skeleton_review": existing}, sheet,
        )

        sheet.assert_not_called()
        self.assertEqual(diagnosed.metadata["skeleton_review"], existing)

    def test_rig_intake_generates_fresh_deformation_review_evidence(self):
        def deformation(_path, output_path, _progress):
            review_path = output_path.with_name("deformation-rest.png")
            review_path.write_bytes(b"png")
            return {
                "complete_action_range": True,
                "frames": [{"frame": frame} for frame in range(5)],
                "maximum_p99_edge_stretch": 9.0,
                "review_files": [{"path": str(review_path)}],
            }

        def persist(report):
            self.assertTrue(Path(report["review_files"][0]["path"]).exists())
            return report

        artifact = main.Artifact(
            id="rigged-mary",
            name="mary.glb",
            kind="glb",
            path="mary.glb",
            size_bytes=1,
            created_at=0,
            metadata={
                "mesh_quality": {},
                "quality_decisions": {
                    "deformation-review": {"status": "needs_review"},
                },
            },
        )
        with (
            patch.object(main, "_artifact_path", return_value=Path("mary.glb")),
            patch.object(main, "analyze_rig_path", return_value={"animations": []}),
            patch.object(main, "evaluate_for_rigging", return_value={"status": "ready"}),
            patch.object(main, "evaluate_rig_certificate", return_value={"status": "ready"}),
            patch.object(main, "evaluate_for_animation", return_value={"status": "ready"}),
            patch.object(main.native, "run_deformation_diagnostics", side_effect=deformation),
            patch.object(main, "persist_deformation_review_artifacts", side_effect=persist),
            patch.object(main, "_write_json"),
        ):
            diagnosed = main._attach_rig_intake_diagnostics(artifact)

        self.assertEqual(
            diagnosed.metadata["quality_decisions"]["deformation-review"]["status"],
            "blocked",
        )

    def test_uploaded_stage_retry_resumes_from_a_persisted_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="mesh-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0)
            output = main.Artifact(id="rigged-1", name="rigged.glb", kind="glb", path="rigged.glb", size_bytes=1, created_at=0)
            payload = {
                "submission_id": "rig-submission-1",
                "stage": "rig",
                "input_artifact_id": source.id,
                "output_artifact_id": output.id,
            }
            original_create = main.run_store.create
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, output.id: output}[artifact_id]),
                    patch.object(main, "_attach_rig_intake_diagnostics", side_effect=lambda artifact: artifact),
                    patch.object(main.run_store, "create", side_effect=RuntimeError("simulated crash")),
                ):
                    failed = TestClient(app, raise_server_exceptions=False).post("/api/runs/record-artifact", json=payload)
                self.assertEqual(failed.status_code, 500)
                self.assertEqual(len(list(main.JOBS_DIR.glob("*.json"))), 1)

                main.run_store.create = original_create
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, output.id: output}[artifact_id]),
                    patch.object(main, "_attach_rig_intake_diagnostics", side_effect=lambda artifact: artifact),
                ):
                    retried = client.post("/api/runs/record-artifact", json=payload)

                self.assertEqual(retried.status_code, 200)
                self.assertEqual(len(main.run_store.list()), 1)
                self.assertEqual(len(retried.json()["attempts"]), 1)
            finally:
                main.run_store.create = original_create
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_non_glb_retry_restores_retained_promotion_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(id="rigged-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0)
            output = main.Artifact(id="motion-1", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", source.id, source.kind)
            payload = {
                "submission_id": "motion-receipt-retry-1",
                "stage": "generate-motion",
                "input_artifact_id": source.id,
                "output_artifact_id": output.id,
                "run_id": run.id,
            }
            real_write_json = main._write_json
            writes = 0

            def fail_final_receipt(path, value):
                nonlocal writes
                writes += 1
                if writes == 3:
                    raise RuntimeError("simulated receipt write failure")
                real_write_json(path, value)

            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, output.id: output}[artifact_id]),
                    patch.object(main, "_write_json", side_effect=fail_final_receipt),
                ):
                    failed = TestClient(app, raise_server_exceptions=False).post("/api/runs/record-artifact", json=payload)
                self.assertEqual(failed.status_code, 500)

                main.jobs.clear()
                with patch.object(main, "read_artifact", side_effect=lambda artifact_id: {source.id: source, output.id: output}[artifact_id]):
                    retried = client.post("/api/runs/record-artifact", json=payload)

                self.assertEqual(retried.status_code, 200)
                receipt = next(job for job in main.jobs.values() if job.submission_id == payload["submission_id"])
                self.assertFalse(receipt.promoted_output)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_run_history_create_list_get_and_review(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                source = main.Artifact(id="source-1", name="mary.png", kind="image", path="mary.png", size_bytes=1, created_at=0)
                with patch.object(main, "read_artifact", return_value=source):
                    created = client.post(
                        "/api/runs",
                        json={"name": "Mary A-pose", "source_artifact_id": "source-1"},
                    )
                self.assertEqual(created.status_code, 200)
                run_id = created.json()["id"]
                self.assertEqual(client.get("/api/runs").json()[0]["id"], run_id)
                self.assertEqual(client.get(f"/api/runs/{run_id}").json()["name"], "Mary A-pose")
                attempt = client.post(
                    f"/api/runs/{run_id}/attempts",
                    json={"stage": "generate", "input_artifact_id": "source-1", "settings": {"seed": 7}},
                )
                self.assertEqual(attempt.status_code, 200)
                self.assertEqual(attempt.json()["stage"], "generate")

                output = main.Artifact(id="rigged-1", name="rigged.glb", kind="glb", path="rigged.glb", size_bytes=1, created_at=0)
                with patch.object(main, "read_artifact", return_value=output):
                    finished = client.post(
                        f"/api/runs/{run_id}/attempts/{attempt.json()['id']}/finish",
                        json={
                            "status": "completed",
                            "output_artifact_id": "rigged-1",
                            "logs": ["Sword Attack embedded"],
                        },
                    )
                self.assertEqual(finished.status_code, 200)
                self.assertEqual(finished.json()["current_artifact_id"], "rigged-1")
                self.assertEqual(finished.json()["attempts"][0]["status"], "completed")
                rig_attempt = main.run_store.start_attempt(run_id, "target-proxy-rig", "rigged-1", {})
                main.run_store.finish_attempt(run_id, rig_attempt.id, "completed", output_artifact_id="rigged-2")
                motion_attempt = main.run_store.start_attempt(run_id, "retarget-motion", "rigged-2", {})
                main.run_store.finish_attempt(run_id, motion_attempt.id, "completed", output_artifact_id="animated-1", promote_output=False)

                reviewed = client.patch(
                    f"/api/runs/{run_id}",
                    json={"review_status": "rejected", "review_note": "garbled", "artifact_id": "animated-1"},
                )
                self.assertEqual(reviewed.status_code, 200)
                self.assertEqual(reviewed.json()["review_note"], "garbled")

                handed_back = client.post(
                    f"/api/runs/{run_id}/checkpoints/skeleton",
                    json={"note": "Right wrist marker corrected.", "artifact_id": "rigged-2"},
                )
                self.assertEqual(handed_back.status_code, 200)
                self.assertEqual(handed_back.json()["checkpoints"]["skeleton"]["note"], "Right wrist marker corrected.")
                self.assertEqual(handed_back.json()["checkpoints"]["skeleton"]["artifact_id"], "rigged-2")
            finally:
                main.run_store = previous

    def test_skeleton_checkpoint_approves_a_retained_rig_the_pointer_never_reached(self):
        """A fresh auto-rig whose certificate lands `needs_review` is retained,
        so it never becomes the run's current artifact. Approving it must
        succeed over HTTP and pin it as current (the operator's "I can't even
        approve when the auto-rig is successful")."""
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Lyra", "source-1", "image")
                mesh = main.run_store.start_attempt(run.id, "generate", "source-1", {})
                main.run_store.finish_attempt(run.id, mesh.id, "completed", output_artifact_id="mesh-1")
                rig = main.run_store.start_attempt(run.id, "auto-rig", "mesh-1", {})
                retained = main.run_store.finish_attempt(
                    run.id, rig.id, "completed", output_artifact_id="rig-1", promote_output=False
                )
                self.assertEqual(retained.current_artifact_id, "mesh-1")

                handed_back = client.post(
                    f"/api/runs/{run.id}/checkpoints/skeleton",
                    json={"note": "Skeleton reviewed.", "artifact_id": "rig-1"},
                )

                self.assertEqual(handed_back.status_code, 200)
                self.assertEqual(handed_back.json()["checkpoints"]["skeleton"]["artifact_id"], "rig-1")
                self.assertEqual(handed_back.json()["current_artifact_id"], "rig-1")
                self.assertEqual(handed_back.json()["attempts"][-1]["settings"]["reason"], "skeleton checkpoint pinned")

                rejected = client.post(
                    f"/api/runs/{run.id}/checkpoints/skeleton",
                    json={"note": "Not a rig.", "artifact_id": "mesh-1"},
                )
                self.assertEqual(rejected.status_code, 409)
                self.assertIn("completed rig artifact from this run", rejected.json()["detail"])
            finally:
                main.run_store = previous

    def test_final_acceptance_rejects_an_animated_artifact_with_a_blocked_quality_gate(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "source-1")
                rig = main.run_store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
                main.run_store.finish_attempt(
                    run.id,
                    rig.id,
                    "completed",
                    output_artifact_id="rig-1",
                )
                motion = main.run_store.start_attempt(run.id, "retarget-motion", "rig-1", {})
                main.run_store.finish_attempt(
                    run.id,
                    motion.id,
                    "completed",
                    output_artifact_id="animated-1",
                    promote_output=False,
                )
                artifact = main.Artifact(
                    id="animated-1",
                    name="walk.glb",
                    kind="glb",
                    path="walk.glb",
                    size_bytes=1,
                    created_at=0,
                    metadata={
                        "quality_decisions": {
                            "animation-review": {"status": "blocked"},
                            "deformation-review": {"status": "needs_review"},
                        },
                    },
                )

                with patch.object(main, "read_artifact", return_value=artifact):
                    response = client.patch(
                        f"/api/runs/{run.id}",
                        json={
                            "review_status": "accepted",
                            "review_note": "Looks fine.",
                            "artifact_id": "animated-1",
                        },
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn("animation-review", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).review_status.value, "unreviewed")
            finally:
                main.run_store = previous

    def test_failed_job_finishes_associated_run_attempt(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary A-pose", "source-1")
                attempt = main.run_store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})
                job = main.JobState(
                    id="job-1",
                    kind="generate",
                    submission_id="submission-1",
                    source_artifact_id="source-1",
                    run_id=run.id,
                    attempt_id=attempt.id,
                )
                request = main.JobRequest(
                    kind="generate",
                    submission_id="submission-1",
                    source_artifact_id="source-1",
                    run_id=run.id,
                    seed=7,
                )

                asyncio.run(main._execute_job(job, request))

                loaded = main.run_store.get(run.id)
                self.assertEqual(job.status, "failed")
                self.assertEqual(loaded.attempts[0].status, "failed")
                self.assertEqual(loaded.attempts[0].error, job.error)
            finally:
                main.run_store = previous

    def test_target_proxy_rig_requires_a_durable_donor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_artifacts = main.ARTIFACTS
            main.run_store = RunStore(root / "runs")
            main.ARTIFACTS = root / "artifacts"
            main.ARTIFACTS.mkdir()
            source_path = main.ARTIFACTS / "source.glb"
            trimesh.creation.box().export(source_path)
            source = main.Artifact(
                id="source-1",
                name="mary.glb",
                kind="glb",
                path=str(source_path),
                size_bytes=source_path.stat().st_size,
                created_at=0,
            )
            main._write_json(main.ARTIFACTS / "source-1.json", source.model_dump())
            run = main.run_store.create("Mary proxy rig", source.id, source.kind)
            attempt = main.run_store.reserve_attempt(run.id, "job-proxy", "target-proxy-rig", source.id, {})
            job = main.JobState(id="job-proxy", kind="target-proxy-rig", run_id=run.id, attempt_id=attempt.id)
            request = main.JobRequest(
                kind="target-proxy-rig",
                submission_id="proxy-rig-1",
                source_artifact_id=source.id,
                run_id=run.id,
            )
            try:
                asyncio.run(main._execute_job(job, request))

                self.assertEqual(job.status, "failed")
                self.assertIn("donor_artifact_id", job.error)
                self.assertEqual(main.run_store.get(run.id).attempts[-1].status, "failed")
            finally:
                main.run_store = previous_store
                main.ARTIFACTS = previous_artifacts

    def test_conflicting_attempt_completion_returns_conflict(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary A-pose", "source-1")
                attempt = main.run_store.start_attempt(run.id, "generate", "source-1", {})
                main.run_store.finish_attempt(
                    run.id,
                    attempt.id,
                    "completed",
                    output_artifact_id="mesh-1",
                )

                response = TestClient(app, raise_server_exceptions=False).post(
                    f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                    json={"status": "failed", "error": "late failure"},
                )

                self.assertEqual(response.status_code, 409)
            finally:
                main.run_store = previous

    def test_manual_completion_rejects_backend_owned_attempt_and_missing_output(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "source-1")
                backend_attempt = main.run_store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})
                owned = client.post(
                    f"/api/runs/{run.id}/attempts/{backend_attempt.id}/finish",
                    json={"status": "failed", "error": "client override"},
                )
                manual = main.run_store.start_attempt(run.id, "manual", "source-1", {})
                with patch.object(main, "read_artifact", side_effect=main.HTTPException(status_code=404, detail="Artifact not found")):
                    missing = client.post(
                        f"/api/runs/{run.id}/attempts/{manual.id}/finish",
                        json={"status": "completed", "output_artifact_id": "missing"},
                    )

                self.assertEqual(owned.status_code, 409)
                self.assertEqual(missing.status_code, 404)
                self.assertEqual(main.run_store.get(run.id).current_artifact_id, "source-1")
            finally:
                main.run_store = previous

    def test_manual_bvh_completion_keeps_current_character_glb(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "character-1", "glb")
                attempt = main.run_store.start_attempt(run.id, "generate-motion", "character-1", {})
                motion = main.Artifact(
                    id="motion-1",
                    name="attack.bvh",
                    kind="bvh",
                    path="attack.bvh",
                    size_bytes=1,
                    created_at=0,
                )
                with patch.object(main, "read_artifact", return_value=motion):
                    response = client.post(
                        f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                        json={"status": "completed", "output_artifact_id": motion.id},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["current_artifact_id"], "character-1")
                self.assertFalse(response.json()["attempts"][0]["promoted"])
            finally:
                main.run_store = previous

    def test_manual_rig_completion_honors_quality_gates(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "mesh-1", "glb")
                attempt = main.run_store.start_attempt(run.id, "rig", "mesh-1", {})
                rig = main.Artifact(
                    id="rig-blocked",
                    name="rig.glb",
                    kind="glb",
                    path="rig.glb",
                    size_bytes=1,
                    created_at=0,
                    metadata={"quality_decisions": {
                        "rig": {"status": "blocked"},
                        "rig-certificate": {"status": "ready"},
                        "animation-review": {"status": "ready"},
                        "deformation-review": {"status": "ready"},
                    }},
                )
                with (
                    patch.object(main, "read_artifact", return_value=rig),
                    patch.object(main, "_attach_rig_intake_diagnostics", return_value=rig),
                ):
                    response = client.post(
                        f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                        json={"status": "completed", "output_artifact_id": rig.id},
                    )

                self.assertEqual(response.status_code, 200, response.json())
                self.assertEqual(response.json()["current_artifact_id"], "mesh-1")
                self.assertFalse(response.json()["attempts"][0]["promoted"])
            finally:
                main.run_store = previous

    def test_manual_voxel_proxy_rig_is_retained_even_when_quality_is_ready(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "mesh-1", "glb")
                attempt = main.run_store.start_attempt(
                    run.id,
                    "target-proxy-rig",
                    "mesh-1",
                    {"deformation_surface": "voxel-proxy"},
                )
                decisions = {
                    name: {"status": "ready"}
                    for name in ("rig", "rig-certificate", "animation-review", "deformation-review")
                }
                rig = main.Artifact(
                    id="voxel-rig-ready", name="voxel-rig.glb", kind="glb",
                    path="voxel-rig.glb", size_bytes=1, created_at=0,
                    metadata={"quality_decisions": decisions},
                )
                with (
                    patch.object(main, "read_artifact", return_value=rig),
                    patch.object(
                        main,
                        "_attach_serialized_rig_intake_diagnostics",
                        new=AsyncMock(return_value=rig),
                    ),
                ):
                    response = client.post(
                        f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                        json={"status": "completed", "output_artifact_id": rig.id},
                    )

                self.assertEqual(response.status_code, 200, response.json())
                self.assertEqual(response.json()["current_artifact_id"], "mesh-1")
                self.assertFalse(response.json()["attempts"][0]["promoted"])
            finally:
                main.run_store = previous

    def test_overlapping_manual_rig_completion_runs_diagnostics_once(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "mesh-1", "glb")
                attempt = main.run_store.start_attempt(run.id, "rig", "mesh-1", {})
                decisions = {
                    name: {"status": "ready"}
                    for name in ("rig", "rig-certificate", "animation-review", "deformation-review")
                }
                decisions["rig-certificate"]["policy_version"] = 2
                rig = main.Artifact(
                    id="rig-ready",
                    name="rig.glb",
                    kind="glb",
                    path="rig.glb",
                    size_bytes=1,
                    created_at=0,
                    metadata={"quality_decisions": decisions},
                )

                async def diagnose(output):
                    await asyncio.sleep(0.01)
                    return output

                request = main.AttemptFinishRequest(
                    status="completed",
                    output_artifact_id=rig.id,
                )

                async def finish_twice():
                    return await asyncio.gather(
                        main.finish_run_attempt(run.id, attempt.id, request),
                        main.finish_run_attempt(run.id, attempt.id, request),
                    )

                with (
                    patch.object(main, "read_artifact", return_value=rig),
                    patch.object(
                        main,
                        "_attach_serialized_rig_intake_diagnostics",
                        new=AsyncMock(side_effect=diagnose),
                    ) as diagnostics,
                ):
                    first, second = asyncio.run(finish_twice())

                self.assertEqual(first.current_artifact_id, rig.id)
                self.assertEqual(second.current_artifact_id, rig.id)
                self.assertEqual(diagnostics.await_count, 1)
            finally:
                main.run_store = previous

    def test_failed_manual_rig_with_partial_output_never_promotes(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "mesh-1", "glb")
                attempt = main.run_store.start_attempt(run.id, "rig", "mesh-1", {})

                response = client.post(
                    f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                    json={
                        "status": "failed",
                        "output_artifact_id": "partial-or-missing",
                        "error": "manual export failed",
                    },
                )

                self.assertEqual(response.status_code, 200, response.json())
                self.assertEqual(response.json()["current_artifact_id"], "mesh-1")
                self.assertFalse(response.json()["attempts"][0]["promoted"])
            finally:
                main.run_store = previous

    def test_manual_rig_diagnostic_failure_finalizes_the_attempt(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "mesh-1", "glb")
                attempt = main.run_store.start_attempt(run.id, "rig", "mesh-1", {})
                rig = main.Artifact(
                    id="rig-partial",
                    name="rig.glb",
                    kind="glb",
                    path="rig.glb",
                    size_bytes=1,
                    created_at=0,
                )
                with (
                    patch.object(main, "read_artifact", return_value=rig),
                    patch.object(
                        main,
                        "_attach_serialized_rig_intake_diagnostics",
                        new=AsyncMock(side_effect=RuntimeError("Blender stopped")),
                    ),
                ):
                    response = client.post(
                        f"/api/runs/{run.id}/attempts/{attempt.id}/finish",
                        json={"status": "completed", "output_artifact_id": rig.id},
                    )

                self.assertEqual(response.status_code, 422, response.json())
                stored = main.run_store.get(run.id)
                self.assertEqual(stored.current_artifact_id, "mesh-1")
                self.assertEqual(stored.attempts[0].status, "failed")
                self.assertIn("Blender stopped", stored.attempts[0].error)
            finally:
                main.run_store = previous

    def test_job_rejects_source_outside_existing_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            try:
                run = main.run_store.create("Mary", "source-1")
                source = main.Artifact(id="source-2", name="zix.png", kind="image", path="zix.png", size_bytes=1, created_at=0)
                with (
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                ):
                    response = client.post(
                        "/api/jobs",
                        json={
                            "kind": "generate",
                            "submission_id": "submission-1",
                            "source_artifact_id": "source-2",
                            "run_id": run.id,
                        },
                    )

                self.assertEqual(response.status_code, 409)
                self.assertEqual(len(main.run_store.get(run.id).attempts), 0)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_retarget_job_rejects_missing_blender_before_queueing(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {"quality_decisions": {"rig-certificate": {"status": "ready", "policy_version": 2}}}
            character = main.Artifact(id="character-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0, metadata=certificate)
            motion = main.Artifact(id="motion-1", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", character.id, character.kind)
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character, motion.id: motion}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=None),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "retarget-motion",
                        "submission_id": "retarget-without-blender",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"motion_artifact_id": motion.id},
                    })

                self.assertEqual(response.status_code, 409)
                self.assertIn("Blender", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_target_proxy_rig_rejects_missing_blender_before_queueing(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = main.Artifact(id="character-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", character.id, character.kind)
            try:
                with (
                    patch.object(main, "read_artifact", return_value=character),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=None),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "target-proxy-rig",
                        "submission_id": "proxy-rig-without-blender",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"curated_motion": "grounded-sword-v1", "attach_sword": True},
                    })

                self.assertEqual(response.status_code, 409)
                self.assertIn("Blender", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
                self.assertEqual(main.jobs, {})
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_target_proxy_rig_rejects_unknown_deformation_surface_before_queueing(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = main.Artifact(id="character-1", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", character.id, character.kind)
            try:
                with (
                    patch.object(main, "read_artifact", return_value=character),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "target-proxy-rig",
                        "submission_id": "proxy-rig-unknown-surface",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"curated_motion": "grounded-sword-v1", "deformation_surface": "magic"},
                    })

                self.assertEqual(response.status_code, 400)
                self.assertIn("Unsupported deformation surface", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
                self.assertEqual(main.jobs, {})
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_target_proxy_rig_persists_normalized_default_deformation_surface(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = main.Artifact(
                id="character-normalized-surface",
                name="mary.glb",
                kind="glb",
                path="mary.glb",
                size_bytes=1,
                created_at=0,
            )
            run = main.run_store.create("Mary", character.id, character.kind)
            try:
                with (
                    patch.object(main, "read_artifact", return_value=character),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "target-proxy-rig",
                        "submission_id": "proxy-rig-normalized-surface",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"curated_motion": "grounded-sword-v1"},
                    })

                self.assertEqual(response.status_code, 200)
                job = main.JobState(**response.json())
                self.assertEqual(
                    job.request_payload["settings"]["deformation_surface"],
                    "visible-original",
                )
                attempt = main.run_store.get(run.id).attempts[0]
                self.assertEqual(
                    attempt.settings["deformation_surface"],
                    "visible-original",
                )
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_health_and_capabilities(self):
        self.assertEqual(client.get("/api/health").json()["status"], "ok")
        payload = client.get("/api/system/capabilities").json()
        self.assertIn("gpu", payload)
        self.assertEqual(payload["gpu"]["model_variant"], "C++ Generate + TRELLIS.2 FP8 Refine")

    def test_capabilities_gpu_block_includes_free_and_busy_external(self):
        payload = client.get("/api/system/capabilities").json()
        self.assertIn("vram_free_mb", payload["gpu"])
        self.assertIn("busy_external", payload["gpu"])

    def test_gpu_info_reports_free_and_busy_external_from_snapshot(self):
        busy_snapshot = native.GpuMemorySnapshot(
            available=True, used_mb=21000, total_mb=24564, free_mb=3564, busy_external=True
        )
        with patch.object(native, "gpu_memory_snapshot", return_value=busy_snapshot):
            payload = main._gpu_info()

        self.assertEqual(payload["vram_free_mb"], 3564)
        self.assertTrue(payload["busy_external"])

    def test_gpu_info_not_busy_external_when_gpu_snapshot_free(self):
        free_snapshot = native.GpuMemorySnapshot(
            available=True, used_mb=1000, total_mb=24564, free_mb=23564, busy_external=False
        )
        with patch.object(native, "gpu_memory_snapshot", return_value=free_snapshot):
            payload = main._gpu_info()

        self.assertEqual(payload["vram_free_mb"], 23564)
        self.assertFalse(payload["busy_external"])

    def test_run_index_omits_heavy_attempt_details_but_detail_retains_them(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "source-1", "image")
                attempt = main.run_store.start_attempt(
                    run.id,
                    "generate",
                    "source-1",
                    {"seed": 7, "expanded_prompt": "x" * 10_000},
                )
                main.run_store.finish_attempt(
                    run.id,
                    attempt.id,
                    "failed",
                    error="generation failed",
                    logs=["large worker log" * 10_000],
                    warnings=["review warning"],
                    promote_output=False,
                )

                index = client.get("/api/runs")
                detail = client.get(f"/api/runs/{run.id}")

                self.assertEqual(index.status_code, 200)
                indexed_attempt = index.json()[0]["attempts"][0]
                self.assertEqual(indexed_attempt["settings"], {})
                self.assertEqual(indexed_attempt["logs"], [])
                self.assertEqual(indexed_attempt["warnings"], [])
                self.assertIsNone(indexed_attempt["error"])
                self.assertEqual(detail.json()["attempts"][0]["settings"]["seed"], 7)
                self.assertTrue(detail.json()["attempts"][0]["logs"])
                compressed = client.get(
                    f"/api/runs/{run.id}",
                    headers={"Accept-Encoding": "gzip"},
                )
                self.assertEqual(compressed.headers.get("content-encoding"), "gzip")
            finally:
                main.run_store = previous

    def test_stale_rig_certificate_is_not_ready_for_motion(self):
        artifact = main.Artifact(
            id="rig-1",
            name="legacy.glb",
            kind="glb",
            path="legacy.glb",
            size_bytes=1,
            created_at=0,
            metadata={
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 1},
                }
            },
        )

        self.assertFalse(main._has_ready_rig_quality(artifact))

    def test_ready_certificate_does_not_bypass_another_blocked_rig_gate(self):
        decisions = {
            name: {"status": "ready"}
            for name in ("rig", "rig-certificate", "animation-review", "deformation-review")
        }
        decisions["rig-certificate"] = {"status": "ready", "policy_version": 2}
        decisions["deformation-review"] = {"status": "blocked"}
        artifact = main.Artifact(
            id="rig-blocked",
            name="blocked.glb",
            kind="glb",
            path="blocked.glb",
            size_bytes=1,
            created_at=0,
            metadata={"quality_decisions": decisions},
        )

        self.assertFalse(main._has_ready_rig_quality(artifact))

    def test_cors_origins_include_launcher_supplied_wsl_origin(self):
        with patch.dict(
            main.os.environ,
            {"AISMITH_CORS_ORIGINS": "http://192.0.2.10:5174,http://localhost:5174"},
        ):
            origins = main._cors_origins()

        self.assertIn("http://127.0.0.1:5173", origins)
        self.assertIn("http://192.0.2.10:5174", origins)
        self.assertIn("http://localhost:5174", origins)

    def test_data_root_accepts_launcher_supplied_shared_store(self):
        with patch.dict(main.os.environ, {"AISMITH_DATA_DIR": "/srv/aismith/studio-data"}):
            self.assertEqual(main._data_root(), Path("/srv/aismith/studio-data"))

    def test_gpu_info_uses_wsl_nvidia_smi_when_not_on_path(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 24564, 1024, 8.9\n",
            stderr="",
        )
        with (
            patch.object(main.shutil, "which", return_value=None),
            patch.object(main.platform, "release", return_value="6.18.0-microsoft-standard-WSL2"),
            patch.object(main.Path, "exists", return_value=True),
            patch.object(main.subprocess, "run", return_value=completed) as run,
            patch.dict(sys.modules, {"torch": None}),
        ):
            payload = main._gpu_info()

        self.assertEqual(run.call_args.args[0][0], "/usr/lib/wsl/lib/nvidia-smi")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["compute_capability"], "8.9")

    def test_ram_info_reads_linux_proc_meminfo(self):
        meminfo = "MemTotal:       33554432 kB\nMemAvailable:   8388608 kB\n"
        with (
            patch.object(main.platform, "system", return_value="Linux"),
            patch.object(main.Path, "read_text", return_value=meminfo),
        ):
            payload = main._ram_info()

        self.assertEqual(payload, {"total_mb": 32768, "used_mb": 24576})

    def test_linux_refiner_installs_gpu_native_runtime(self):
        with patch.object(main.os, "name", "posix"):
            packages = main._trellis_refiner_runtime_packages()
        for package in (
            "spconv-cu121==2.3.8",
            "rembg[cpu]==2.0.81",
            "nvidia-cuda-nvcc==13.0.48",
            "nvidia-nvvm==13.0.48",
            "nvidia-cuda-crt==13.0.48",
            "nvidia-cuda-cccl",
            "ninja",
        ):
            self.assertIn(package, packages)

    def test_linux_refiner_builds_native_extensions_for_the_current_gpu(self):
        command = main._trellis_native_install_command(Path("/runtime/python"), Path("/runtime"))
        self.assertEqual(command[0], "/runtime/python")
        self.assertEqual(command[-2:], ["--marker", "/runtime/native-contract.json"])

    def test_fp8_generation_requests_image_conditioned_texture(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mary.glb"

            def generate_textured(_command, **_kwargs):
                document = {
                    "asset": {"version": "2.0"},
                    "images": [{"bufferView": 0, "mimeType": "image/png"}],
                    "textures": [{"source": 0}],
                    "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
                    "meshes": [{"primitives": [{"material": 0}]}],
                }
                chunk = json.dumps(document).encode("utf-8")
                chunk += b" " * (-len(chunk) % 4)
                output.write_bytes(
                    struct.pack("<4sII", b"glTF", 2, 20 + len(chunk))
                    + struct.pack("<I4s", len(chunk), b"JSON")
                    + chunk
                )

            with (
                patch.object(native, "trellis_refiner_ready", return_value=True),
                patch.object(native, "trellis_refiner_dir", return_value=Path("/runtime")),
                patch.object(native, "trellis_refiner_python", return_value=Path("/runtime/python")),
                patch.object(native, "run_process", side_effect=generate_textured) as run_process,
            ):
                result = native.run_trellis_fp8(Path("mary.png"), output, 512)

        spec = __import__("json").loads(run_process.call_args.args[0][-1])
        self.assertTrue(spec["generate_texture"])
        self.assertEqual(spec["seed"], 1337)
        self.assertTrue(result["textured"])
        self.assertEqual(result["stage"], "textured-geometry")

    def test_fp8_generation_rejects_geometry_only_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mary.glb"

            def generate_clay(_command, **_kwargs):
                trimesh.creation.box().export(output)

            with (
                patch.object(native, "trellis_refiner_ready", return_value=True),
                patch.object(native, "trellis_refiner_dir", return_value=Path("/runtime")),
                patch.object(native, "trellis_refiner_python", return_value=Path("/runtime/python")),
                patch.object(native, "run_process", side_effect=generate_clay),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not contain an embedded PBR texture"):
                    native.run_trellis_fp8(Path("mary.png"), output, 512)

    def test_refine_policy_records_supported_runtime_contract(self):
        policy = native.trellis_refine_policy()
        self.assertEqual(policy["resolution"], 512)
        self.assertEqual(policy["conv_backend"], "spconv")
        self.assertEqual(policy["sparse_compute_dtype"], "float16")
        self.assertFalse(policy["use_tiled_decoder"])
        self.assertEqual(policy["downsampling"], 32)
        self.assertEqual(policy["reconstruction"], "preserve-decoded")

    def test_export_artifact_copies_into_the_configured_output_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            export_dir = root / "exports"
            artifacts.mkdir()
            export_dir.mkdir()
            source = root / "source.glb"
            source.write_bytes(b"glTF" + b"x" * 32)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="Mary Rig.glb", kind="glb")

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "configured_output_directory", return_value=export_dir),
            ):
                response = client.post(f"/api/assets/{artifact.id}/export")

            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            expected = export_dir / f"Mary Rig-{artifact.id[:8]}.glb"
            self.assertEqual(Path(body["path"]), expected)
            self.assertTrue(expected.exists())
            self.assertEqual(body["size_bytes"], expected.stat().st_size)

    def test_export_artifact_uses_a_caller_supplied_name_hint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            export_dir = root / "exports"
            artifacts.mkdir()
            export_dir.mkdir()
            source = root / "source.glb"
            source.write_bytes(b"glTF" + b"x" * 32)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="unrigged-mary.glb", kind="glb")

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "configured_output_directory", return_value=export_dir),
            ):
                response = client.post(
                    f"/api/assets/{artifact.id}/export",
                    json={"name_hint": "Mary / the Knight?"},
                )

            self.assertEqual(response.status_code, 200, response.text)
            expected = export_dir / f"Mary _ the Knight-{artifact.id[:8]}.glb"
            self.assertEqual(Path(response.json()["path"]), expected)
            self.assertTrue(expected.exists())

    def test_export_artifact_avoids_overwriting_an_existing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            export_dir = root / "exports"
            artifacts.mkdir()
            export_dir.mkdir()
            source = root / "source.glb"
            source.write_bytes(b"glTF" + b"y" * 32)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="Mary Rig.glb", kind="glb")

            colliding_name = f"Mary Rig-{artifact.id[:8]}.glb"
            (export_dir / colliding_name).write_bytes(b"existing")

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "configured_output_directory", return_value=export_dir),
            ):
                response = client.post(f"/api/assets/{artifact.id}/export")

            self.assertEqual(response.status_code, 200, response.text)
            expected = export_dir / f"Mary Rig-{artifact.id[:8]}-2.glb"
            self.assertEqual(Path(response.json()["path"]), expected)
            self.assertTrue(expected.exists())
            self.assertEqual((export_dir / colliding_name).read_bytes(), b"existing")

    def test_export_artifact_404s_for_an_unknown_artifact(self):
        response = client.post("/api/assets/does-not-exist/export")
        self.assertEqual(response.status_code, 404)

    def test_export_artifact_409s_when_no_output_directory_is_configured(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = root / "source.glb"
            source.write_bytes(b"glTF" + b"z" * 32)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="Mary Rig.glb", kind="glb")

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "configured_output_directory", return_value=None),
            ):
                response = client.post(f"/api/assets/{artifact.id}/export")

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("output folder", response.json()["detail"].lower())

    @unittest.skipIf(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        "chmod cannot restrict a root-owned process",
    )
    def test_export_artifact_409s_when_the_output_directory_is_not_writable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            export_dir = root / "exports"
            artifacts.mkdir()
            export_dir.mkdir()
            source = root / "source.glb"
            source.write_bytes(b"glTF" + b"w" * 32)

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "output_directory", return_value=artifacts),
            ):
                artifact = main.save_artifact(source, name="Mary Rig.glb", kind="glb")

            export_dir.chmod(0o500)
            try:
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "configured_output_directory", return_value=export_dir),
                ):
                    response = client.post(f"/api/assets/{artifact.id}/export")
            finally:
                export_dir.chmod(0o700)

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("writable", response.json()["detail"].lower())

    # -- subject_kind: general 2D->3D pipeline with optional biped rigging --

    def test_create_run_endpoint_persists_subject_kind_and_applicable_stages(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                source = main.Artifact(
                    id="chest-1", name="chest.png", kind="image",
                    path="chest.png", size_bytes=1, created_at=0,
                )
                with patch.object(main, "read_artifact", return_value=source):
                    created = client.post(
                        "/api/runs",
                        json={
                            "name": "Treasure Chest",
                            "source_artifact_id": "chest-1",
                            "subject_kind": "object",
                        },
                    )
                    default_run = client.post(
                        "/api/runs",
                        json={"name": "Mary", "source_artifact_id": "chest-1"},
                    )
                self.assertEqual(created.status_code, 200, created.text)
                self.assertEqual(created.json()["subject_kind"], "object")
                self.assertEqual(created.json()["applicable_stages"], ["prepare", "generate", "clean"])
                self.assertEqual(default_run.status_code, 200, default_run.text)
                self.assertEqual(default_run.json()["subject_kind"], "auto")
                self.assertEqual(
                    default_run.json()["applicable_stages"],
                    ["prepare", "generate", "clean", "skeleton", "animate", "review"],
                )
            finally:
                main.run_store = previous

    def test_pre_migration_run_lists_and_gets_with_default_subject_kind(self):
        """Idempotent migration surfaces through the API too: a manifest
        written before subject_kind existed reads back as "auto" with every
        stage applicable."""
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "source-1")
                path = Path(directory) / f"{run.id}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                del payload["subject_kind"]
                del payload["applicable_stages"]
                path.write_text(json.dumps(payload), encoding="utf-8")

                indexed = client.get("/api/runs").json()[0]
                detail = client.get(f"/api/runs/{run.id}").json()

                self.assertEqual(indexed["subject_kind"], "auto")
                self.assertEqual(detail["subject_kind"], "auto")
                self.assertEqual(
                    detail["applicable_stages"],
                    ["prepare", "generate", "clean", "skeleton", "animate", "review"],
                )
            finally:
                main.run_store = previous

    def test_patch_run_updates_subject_kind_independently_of_review(self):
        with TemporaryDirectory() as directory:
            previous = main.run_store
            main.run_store = RunStore(Path(directory))
            try:
                run = main.run_store.create("Mary", "source-1")

                response = client.patch(f"/api/runs/{run.id}", json={"subject_kind": "biped"})

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["subject_kind"], "biped")
                self.assertEqual(
                    response.json()["applicable_stages"],
                    ["prepare", "generate", "clean", "skeleton", "animate", "review"],
                )
                self.assertEqual(main.run_store.get(run.id).review_status.value, "unreviewed")
            finally:
                main.run_store = previous

    def test_patch_run_rejects_mixing_subject_kind_with_review_fields(self):
        response = client.patch(
            "/api/runs/does-not-matter",
            json={
                "subject_kind": "biped",
                "review_status": "accepted",
                "review_note": "x",
                "artifact_id": "a",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_patch_run_rejects_an_empty_body(self):
        response = client.patch("/api/runs/does-not-matter", json={})

        self.assertEqual(response.status_code, 422)

    def test_prepare_image_requires_biped_or_auto_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(
                id="prop-portrait", name="chest.png", kind="image",
                path="chest.png", size_bytes=1, created_at=0,
            )
            run = main.run_store.create("A Sword", source.id, source.kind, subject_kind="object")
            try:
                with patch.object(main, "read_artifact", return_value=source):
                    response = client.post("/api/jobs", json={
                        "kind": "prepare-image",
                        "submission_id": "prepare-object-rejected",
                        "source_artifact_id": source.id,
                        "run_id": run.id,
                        "settings": {"adapter": "upload"},
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("T-pose preparation applies to biped characters", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_synthesize_views_rejects_an_unknown_profile_family(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(
                id="raw-photo", name="photo.png", kind="image",
                path="photo.png", size_bytes=1, created_at=0,
            )
            run = main.run_store.create("Mary", source.id, source.kind, subject_kind="auto")
            try:
                with patch.object(main, "read_artifact", return_value=source):
                    response = client.post("/api/jobs", json={
                        "kind": "synthesize-views",
                        "submission_id": "synthesize-views-unknown-family",
                        "source_artifact_id": source.id,
                        "run_id": run.id,
                        "settings": {"profile_family": "nope"},
                    })

                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("Unknown synthesize-views profile_family", response.json()["detail"])
                self.assertIn("tpose", response.json()["detail"])
                self.assertIn("source-pose", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_synthesize_views_accepts_and_stamps_the_source_pose_family(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(
                id="raw-photo-2", name="photo.png", kind="image",
                path="photo.png", size_bytes=1, created_at=0,
            )
            run = main.run_store.create("Mary", source.id, source.kind, subject_kind="auto")
            try:
                with (
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "run_comfyui_image_edit", return_value={"prompt_id": "p1", "model": "test"}),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "synthesize-views",
                        "submission_id": "synthesize-views-source-pose",
                        "source_artifact_id": source.id,
                        "run_id": run.id,
                        "settings": {"profile_family": "source-pose", "views": ["back"]},
                    })

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json()["request_payload"]["settings"]["profile_family"], "source-pose",
                )
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_generate_on_object_run_succeeds_without_prepare(self):
        """Generate never requires prepare-image; an object run must be
        able to go straight from an uploaded image to geometry."""
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            source = main.Artifact(
                id="sword-image", name="sword.png", kind="image",
                path="sword.png", size_bytes=1, created_at=0,
            )
            run = main.run_store.create("A Sword", source.id, source.kind, subject_kind="object")
            try:
                with (
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "generate",
                        "submission_id": "object-generate-no-prepare",
                        "source_artifact_id": source.id,
                        "run_id": run.id,
                    })

                self.assertEqual(response.status_code, 200, response.text)
                attempts = main.run_store.get(run.id).attempts
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0].stage, "generate")
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_auto_rig_rejected_for_object_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            mesh = main.Artifact(
                id="chest-mesh", name="chest.glb", kind="glb",
                path="chest.glb", size_bytes=1, created_at=0,
            )
            run = main.run_store.create("A Sword", mesh.id, mesh.kind, subject_kind="object")
            try:
                with (
                    patch.object(main, "read_artifact", return_value=mesh),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "auto-rig",
                        "submission_id": "auto-rig-object-rejected",
                        "source_artifact_id": mesh.id,
                        "run_id": run.id,
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("Rigging is not available for objects", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_auto_rig_stamps_profile_hint_from_subject_kind(self):
        cases = [("biped", "humanoid"), ("creature", "chain"), ("auto", None)]
        for subject_kind, expected_hint in cases:
            with self.subTest(subject_kind=subject_kind):
                with TemporaryDirectory() as directory:
                    previous_store = main.run_store
                    previous_jobs_dir = main.JOBS_DIR
                    previous_jobs = main.jobs.copy()
                    main.run_store = RunStore(Path(directory) / "runs")
                    main.JOBS_DIR = Path(directory) / "jobs"
                    main.JOBS_DIR.mkdir()
                    main.jobs.clear()
                    mesh = main.Artifact(
                        id="mesh-1", name="mesh.glb", kind="glb",
                        path="mesh.glb", size_bytes=1, created_at=0,
                    )
                    run = main.run_store.create("Mary", mesh.id, mesh.kind, subject_kind=subject_kind)
                    try:
                        with (
                            patch.object(main, "read_artifact", return_value=mesh),
                            patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True}),
                            patch.object(main, "_execute_job", new=AsyncMock()),
                        ):
                            response = client.post("/api/jobs", json={
                                "kind": "auto-rig",
                                "submission_id": f"auto-rig-hint-{subject_kind}",
                                "source_artifact_id": mesh.id,
                                "run_id": run.id,
                            })

                        self.assertEqual(response.status_code, 200, response.text)
                        attempt = main.run_store.get(run.id).attempts[0]
                        self.assertEqual(attempt.settings.get("profile_hint"), expected_hint)
                    finally:
                        main.run_store = previous_store
                        main.JOBS_DIR = previous_jobs_dir
                        main.jobs.clear()
                        main.jobs.update(previous_jobs)

    def test_retarget_motion_requires_biped_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "humanoid"},
                }
            }
            character = main.Artifact(
                id="character-1", name="mary.glb", kind="glb",
                path="mary.glb", size_bytes=1, created_at=0, metadata=certificate,
            )
            motion = main.Artifact(id="motion-1", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            run = main.run_store.create("Mislabeled", character.id, character.kind, subject_kind="creature")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character, motion.id: motion}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "retarget-motion",
                        "submission_id": "retarget-creature-rejected",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"motion_artifact_id": motion.id},
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("biped", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_retarget_motion_succeeds_for_biped_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "humanoid"},
                }
            }
            character = main.Artifact(
                id="character-2", name="mary.glb", kind="glb",
                path="mary.glb", size_bytes=1, created_at=0, metadata=certificate,
            )
            motion = main.Artifact(id="motion-2", name="attack.bvh", kind="bvh", path="attack.bvh", size_bytes=1, created_at=0)
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character, motion.id: motion}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "retarget-motion",
                        "submission_id": "retarget-biped-ok",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"motion_artifact_id": motion.id},
                    })

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(main.run_store.get(run.id).attempts), 1)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_generate_motion_slither_requires_creature_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            certificate = {
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "chain"},
                }
            }
            serpent = main.Artifact(
                id="serpent-3", name="serpent.glb", kind="glb",
                path="serpent.glb", size_bytes=1, created_at=0, metadata=certificate,
            )
            run = main.run_store.create("Mislabeled Serpent", serpent.id, serpent.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", return_value=serpent),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": False}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "generate-motion",
                        "submission_id": "slither-biped-rejected",
                        "source_artifact_id": serpent.id,
                        "run_id": run.id,
                        "settings": {},
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("creature", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)


class CuratedAnimationsEndpointTests(unittest.TestCase):
    """GET /api/animations/curated backs the Animate panel's clip picker --
    the pure listing/labeling logic lives in curated_motion.py (see
    test_curated_motion.py); this only checks the route wires that through
    against the real bundled donor library."""

    def test_lists_every_donor_clip_with_the_exact_action_name_and_label(self):
        response = client.get("/api/animations/curated")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["donor"].endswith(
            "vendor/mesh2motion-app/static/animations/human-base-animations.glb"
        ))
        clips = payload["clips"]
        self.assertEqual(len(clips), 87)
        labels = [clip["label"] for clip in clips]
        self.assertEqual(labels, sorted(labels))

        sword_attack = next(clip for clip in clips if clip["label"] == "Sword Attack")
        # This must be the exact Blender action name POST
        # /api/animations/export-curated-bvh expects as `animation` -- see
        # curated_motion.py's module docstring for how that was confirmed.
        self.assertEqual(sword_attack["animation"], "Sword_Attack_Armature")
        self.assertGreater(sword_attack["frames"], 0)
        self.assertGreater(sword_attack["duration_seconds"], 0)


class RigAdjustSettingsTests(unittest.TestCase):
    """`_parse_rig_adjust_settings` is the pure settings-parsing contract
    behind the `rig-adjust` job kind: a rig-adjust job must carry hand-
    adjusted joint positions and/or joints to remove, and rejects a
    request carrying neither."""

    def test_parses_positions_only(self):
        positions, remove_joints = main._parse_rig_adjust_settings(
            {"positions": {"head": [0.0, 0.1, 1.5]}}
        )
        self.assertEqual(positions, {"head": [0.0, 0.1, 1.5]})
        self.assertEqual(remove_joints, [])

    def test_parses_remove_joints_only(self):
        positions, remove_joints = main._parse_rig_adjust_settings(
            {"remove_joints": ["spine_05", "spine_06"]}
        )
        self.assertEqual(positions, {})
        self.assertEqual(remove_joints, ["spine_05", "spine_06"])

    def test_parses_both(self):
        positions, remove_joints = main._parse_rig_adjust_settings(
            {"positions": {"head": [0.0, 0.0, 1.0]}, "remove_joints": ["spine_05"]}
        )
        self.assertEqual(positions, {"head": [0.0, 0.0, 1.0]})
        self.assertEqual(remove_joints, ["spine_05"])

    def test_rejects_neither(self):
        with self.assertRaises(ValueError):
            main._parse_rig_adjust_settings({})

    def test_rejects_empty_positions_and_empty_remove_joints(self):
        with self.assertRaises(ValueError):
            main._parse_rig_adjust_settings({"positions": {}, "remove_joints": []})

    def test_remove_joints_entries_are_coerced_to_str(self):
        _, remove_joints = main._parse_rig_adjust_settings({"remove_joints": [123]})
        self.assertEqual(remove_joints, ["123"])


class CanonicalizeSettingsTests(unittest.TestCase):
    """`main._parse_canonicalize_settings` -- job-kind settings validation,
    consistent with the `_parse_rig_adjust_settings` tests above."""

    @staticmethod
    def _request(**settings):
        return main.JobRequest(
            kind="canonicalize",
            submission_id="canonicalize-settings-test",
            source_artifact_id="source-1",
            settings=settings,
        )

    def test_defaults_to_the_native_target_and_all_four_views(self):
        target_vertices, views, method = main._parse_canonicalize_settings(self._request())

        self.assertEqual(target_vertices, native.DEFAULT_CANONICALIZE_TARGET_VERTICES)
        self.assertEqual(views, native.DEFAULT_CANONICALIZE_VIEWS)
        self.assertEqual(method, native.DEFAULT_CANONICALIZE_METHOD)

    def test_settings_target_vertices_overrides_the_default(self):
        target_vertices, _views, _method = main._parse_canonicalize_settings(self._request(target_vertices=8_000))

        self.assertEqual(target_vertices, 8_000)

    def test_top_level_target_vertices_is_honored_when_settings_omit_it(self):
        request = main.JobRequest(
            kind="canonicalize",
            submission_id="canonicalize-settings-test-2",
            source_artifact_id="source-1",
            target_vertices=12_345,
        )

        target_vertices, _views, _method = main._parse_canonicalize_settings(request)

        self.assertEqual(target_vertices, 12_345)

    def test_settings_target_vertices_takes_priority_over_top_level(self):
        request = main.JobRequest(
            kind="canonicalize",
            submission_id="canonicalize-settings-test-3",
            source_artifact_id="source-1",
            target_vertices=12_345,
            settings={"target_vertices": 6_000},
        )

        target_vertices, _views, _method = main._parse_canonicalize_settings(request)

        self.assertEqual(target_vertices, 6_000)

    def test_rejects_non_integer_target_vertices(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_vertices="a lot"))

    def test_rejects_target_vertices_out_of_range(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_vertices=100))
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_vertices=2_000_000))

    def test_rejects_target_vertices_the_schema_allows_but_the_gate_refuses(self):
        # 250,001..1,000,000 passes the pydantic bound yet resolves past
        # quality_tiers' CUSTOM_TARGET_TRIANGLES_MAXIMUM; it must fail here
        # (a 400) rather than after the job has been accepted.
        self.assertEqual(main.CANONICALIZE_TARGET_VERTICES_MAXIMUM, 250_000)
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_vertices=300_000))
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(main.JobRequest(
                kind="canonicalize",
                submission_id="canonicalize-settings-test",
                source_artifact_id="source-1",
                target_vertices=600_000,
            ))
        target_vertices, _views, _method = main._parse_canonicalize_settings(
            self._request(target_vertices=250_000)
        )
        self.assertEqual(target_vertices, 250_000)

    def test_out_of_range_target_vertices_is_a_400_at_submission(self):
        mesh = main.Artifact(
            id="mesh-canon", name="mary.glb", kind="glb", path="mary.glb", size_bytes=1, created_at=0,
        )
        previous_jobs = main.jobs.copy()
        main.jobs.clear()
        try:
            with (
                patch.object(main, "read_artifact", return_value=mesh),
                patch.object(main, "_model_readiness", return_value={
                    "trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True,
                }),
                patch.object(main, "_execute_job", new=AsyncMock()) as execute,
            ):
                response = client.post("/api/jobs", json={
                    "kind": "canonicalize",
                    "submission_id": f"canonicalize-range-{uuid.uuid4().hex}",
                    "source_artifact_id": mesh.id,
                    "target_vertices": 600_000,
                })
        finally:
            main.jobs.clear()
            main.jobs.update(previous_jobs)

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("target_vertices", response.json()["detail"])
        execute.assert_not_called()

    def test_custom_views_are_accepted(self):
        _target_vertices, views, _method = main._parse_canonicalize_settings(self._request(views=["front", "back"]))

        self.assertEqual(views, ("front", "back"))

    def test_rejects_unknown_view_name(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(views=["front", "top"]))

    def test_rejects_empty_views_list(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(views=[]))

    def test_rejects_duplicate_views(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(views=["front", "front"]))

    def test_defaults_to_voxel_method(self):
        _target_vertices, _views, method = main._parse_canonicalize_settings(self._request())

        self.assertEqual(method, "voxel")

    def test_settings_method_selects_instant_meshes(self):
        _target_vertices, _views, method = main._parse_canonicalize_settings(
            self._request(method="instant-meshes")
        )

        self.assertEqual(method, "instant-meshes")

    def test_settings_method_is_case_insensitive(self):
        _target_vertices, _views, method = main._parse_canonicalize_settings(
            self._request(method="INSTANT-MESHES")
        )

        self.assertEqual(method, "instant-meshes")

    def test_a_quality_tier_drives_the_legacy_vertex_target_too(self):
        # Both remesh backends follow the operator's density: the voxel path
        # via the triangle target, Instant Meshes via its `--vertices` knob.
        target_vertices, _views, _method = main._parse_canonicalize_settings(
            self._request(quality_tier="ultra")
        )

        self.assertEqual(target_vertices, 75_000)

    def test_an_unknown_quality_tier_is_rejected_before_the_job_starts(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(quality_tier="cinematic"))

    def test_a_custom_triangle_target_is_accepted_and_overrides_the_tier(self):
        target_vertices, _views, _method = main._parse_canonicalize_settings(
            self._request(quality_tier="draft", target_triangles=96_000)
        )

        self.assertEqual(target_vertices, 48_000)

    def test_an_out_of_range_custom_triangle_target_is_rejected(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_triangles=900_000))
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(target_triangles=10))

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            main._parse_canonicalize_settings(self._request(method="quad-remesh"))

    def test_job_request_accepts_canonicalize_kind(self):
        request = self._request()
        self.assertEqual(request.kind, "canonicalize")


class ApplyClipJobTests(unittest.TestCase):
    """N5: direct clip application job-kind validation
    (backend/app/main.py's `apply-clip` kind) -- mirrors the retarget-motion
    validation tests above, since apply-clip is retarget-motion's
    no-retarget-math sibling for template-conformant rigs."""

    def _conformant_character(self, artifact_id: str = "conformant-rig-1") -> "main.Artifact":
        return main.Artifact(
            id=artifact_id, name="mary.glb", kind="glb",
            path="mary.glb", size_bytes=1, created_at=0,
            metadata={
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "humanoid"},
                },
                "rig_quality": {"template_conformance": []},
            },
        )

    def test_apply_clip_rejects_rig_with_no_template_conformance_data(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            # Certified (rig-certificate ready, policy_version 2) but no
            # rig_quality.template_conformance data at all -- pre-N3, or a
            # non-humanoid rig. Must not be treated as conformant even
            # though it carries no conformance findings.
            character = main.Artifact(
                id="uncertified-conformance-1", name="mary.glb", kind="glb",
                path="mary.glb", size_bytes=1, created_at=0,
                metadata={
                    "quality_decisions": {
                        "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "humanoid"},
                    },
                },
            )
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-no-conformance-data",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"clip": "Sword_Attack_Armature"},
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("template conformance", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_apply_clip_rejects_unknown_clip_name(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = self._conformant_character("conformant-rig-2")
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-unknown-name",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"clip": "Not_A_Real_Clip_Armature"},
                    })

                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("clip store", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_apply_clip_requires_settings_clip(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = self._conformant_character("conformant-rig-3")
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-missing-clip-setting",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {},
                    })

                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("settings.clip", response.json()["detail"])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_apply_clip_requires_biped_subject_kind(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = self._conformant_character("conformant-rig-4")
            run = main.run_store.create("Mislabeled", character.id, character.kind, subject_kind="creature")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-creature-rejected",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"clip": "Sword_Attack_Armature"},
                    })

                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("biped", response.json()["detail"])
                self.assertEqual(main.run_store.get(run.id).attempts, [])
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_apply_clip_succeeds_for_conformant_biped_rig(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = self._conformant_character("conformant-rig-5")
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-biped-ok",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"clip": "Sword_Attack_Armature"},
                    })

                self.assertEqual(response.status_code, 200, response.text)
                attempts = main.run_store.get(run.id).attempts
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0].settings.get("clip"), "Sword_Attack_Armature")
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

    def test_apply_clip_defaults_family_to_biped(self):
        with TemporaryDirectory() as directory:
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(Path(directory) / "runs")
            main.JOBS_DIR = Path(directory) / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            character = self._conformant_character("conformant-rig-6")
            run = main.run_store.create("Mary", character.id, character.kind, subject_kind="biped")
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: {character.id: character}[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={"trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True, "kimodo_ready": True}),
                    patch.object(main.native, "find_blender", return_value=Path("/tools/blender")),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    # No "family" in settings at all -- resolved against the
                    # biped manifest by default.
                    response = client.post("/api/jobs", json={
                        "kind": "apply-clip",
                        "submission_id": "apply-clip-family-default",
                        "source_artifact_id": character.id,
                        "run_id": run.id,
                        "settings": {"clip": "Roll_Armature"},
                    })

                self.assertEqual(response.status_code, 200, response.text)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)


class TemplateConformantRigFallbackTests(unittest.TestCase):
    """_is_template_conformant_rig (backend/app/main.py) falls back to the
    GLB's own extras.template_conformance when cached artifact metadata
    predates that field (e.g. an artifact bound before N3's rig_quality
    wiring landed, or re-exported by a worker that doesn't recompute
    rig_quality) -- verified live on Mary's rig-adjust output
    (eb2b922531b3499fb47ff7b4c13839ba), whose cached metadata.rig_quality
    has no template_conformance key at all even though the GLB it points at
    embeds a clean `[]` on its armature node."""

    def _certified(self, rig_quality=None) -> "main.Artifact":
        return main.Artifact(
            id="stale-metadata-rig", name="mary.glb", kind="glb",
            path="mary.glb", size_bytes=1, created_at=0,
            metadata={
                "quality_decisions": {
                    "rig-certificate": {"status": "ready", "policy_version": 2, "profile": "humanoid"},
                },
                **({"rig_quality": rig_quality} if rig_quality is not None else {}),
            },
        )

    def test_falls_back_to_clean_glb_extras_when_metadata_lacks_the_field(self):
        artifact = self._certified()  # no rig_quality at all in cached metadata
        with (
            patch.object(main, "_read_glb", return_value=({"nodes": [{"name": "Armature", "extras": {"template_conformance": []}}]}, b"")),
        ):
            self.assertTrue(main._is_template_conformant_rig(artifact))

    def test_falls_back_to_dirty_glb_extras_and_stays_false(self):
        artifact = self._certified()
        with (
            patch.object(main, "_read_glb", return_value=({"nodes": [{"name": "Armature", "extras": {"template_conformance": [{"code": "reparented_bone"}]}}]}, b"")),
        ):
            self.assertFalse(main._is_template_conformant_rig(artifact))

    def test_cached_metadata_wins_when_present_no_glb_read(self):
        artifact = self._certified(rig_quality={"template_conformance": []})
        with patch.object(main, "_read_glb", side_effect=AssertionError("should not read the GLB when metadata already has the field")):
            self.assertTrue(main._is_template_conformant_rig(artifact))

    def test_unreadable_glb_is_not_conformant(self):
        artifact = self._certified()
        with patch.object(main, "_read_glb", side_effect=ValueError("bad glb")):
            self.assertFalse(main._is_template_conformant_rig(artifact))


class PipelinePolicyWiringTests(unittest.TestCase):
    """PIPELINE_POLICY.md evaluation wired into the live job system
    (bead x8qhy): the `_execute_job` hook and `GET /api/runs/{id}/policy`.

    Metadata fixtures mirror the real shapes on the box -- Mary's canonical
    run (generate -> voxel canonicalize -> template rig -> direct clip
    attach) -- so the endpoint contract is tested against what the pipeline
    actually writes.
    """

    GENERATE_METADATA = {
        "stage": "textured-geometry",
        "backend": "trellis2-fp8",
        "model": "TRELLIS.2-4B-FP8",
        "resolution": 512,
        "seed": 1337,
        "textured": True,
        "source_artifact_id": "img-1",
        "source_image_artifact_id": "img-1",
        "mesh_quality": {"faces": 231576, "components": 1, "watertight": False},
    }
    CANONICALIZE_METADATA = {
        "stage": "canonicalize",
        "source_artifact_id": "gen-1",
        "similarity": {"s1_s2_image_similarity_score": 0.994851},
        "settings": {},
        "attempts": [{
            "components": 1, "components_unwelded": 1, "watertight": True,
            "non_manifold_edges": 0, "faces": 29992,
        }],
        "mesh_quality": {
            "faces": 29992, "components": 1, "watertight": True, "non_manifold_edges": 0,
            "bounds": {"minimum": [-0.5, 0.0, -0.11], "maximum": [0.5, 0.99, 0.11]},
        },
    }
    AUTO_RIG_METADATA = {
        "stage": "auto-rig",
        "source_artifact_id": "canon-1",
        "weighting": "blender-bone-heat",
        "rig_quality": {
            "joints": 21,
            "joint_names": ["head", "pelvis"],
            "maximum_influences": 4,
            "non_normalized_vertices": 0,
            "template_conformance": [],
            "animations": [{"name": "Rest", "animated_joints": 0, "invalid_targets": 0}],
        },
        "deformation_review": {"complete_action_range": True, "maximum_p99_edge_stretch": 1.0},
        "quality_decisions": {"rig-certificate": {"status": "ready", "profile": "humanoid"}},
    }
    APPLY_CLIP_METADATA = {
        "stage": "apply-clip",
        "source_artifact_id": "rig-1",
        "clip": "Sword_Attack_Armature",
        "family": "biped",
        "mechanism": "direct-attach",
        "rig_quality": {
            "joints": 21,
            "animations": [{
                "name": "Sword_Attack_Armature", "animated_joints": 18, "invalid_targets": 0,
            }],
            "template_conformance": None,
        },
        "deformation_review": {"complete_action_range": True, "maximum_p99_edge_stretch": 4.1403},
    }

    def _write_artifact(self, artifacts, artifact_id, kind, metadata):
        main._write_json(artifacts / f"{artifact_id}.json", {
            "id": artifact_id,
            "name": f"{artifact_id}.{kind}",
            "kind": kind,
            "path": str(artifacts / f"{artifact_id}.{kind}"),
            "size_bytes": 1,
            "created_at": 0.0,
            "metadata": metadata,
        })

    def _mary_run(self, root, *, through="clip-1"):
        artifacts = root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        store = RunStore(root / "runs")
        run = store.create("Mary canonical", "img-1", "image")
        self._write_artifact(artifacts, "img-1", "image", {})
        chain = [
            ("generate", "img-1", "gen-1", self.GENERATE_METADATA),
            ("canonicalize", "gen-1", "canon-1", self.CANONICALIZE_METADATA),
            ("auto-rig", "canon-1", "rig-1", self.AUTO_RIG_METADATA),
            ("apply-clip", "rig-1", "clip-1", self.APPLY_CLIP_METADATA),
        ]
        for stage, source_id, output_id, metadata in chain:
            self._write_artifact(artifacts, output_id, "glb", metadata)
            attempt = store.start_attempt(run.id, stage, source_id, {})
            store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id=output_id)
            if output_id == through:
                break
        return store, run, artifacts

    def test_run_policy_endpoint_evaluates_the_whole_canonical_chain(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)
            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "ARTIFACTS", artifacts):
                    response = client.get(f"/api/runs/{run.id}/policy")
            finally:
                main.run_store = previous_store

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["run_id"], run.id)
            self.assertEqual(payload["policy_version"], 1)
            self.assertEqual(payload["family"], "biped")
            self.assertEqual(
                [stage["stage"] for stage in payload["stages"]],
                ["S0_SOURCE", "S1_GENERATION", "S2_CANONICALIZATION", "S3_SKELETON_FIT",
                 "S4_SKINNING", "S5_ANIMATION", "S6_EXPORT"],
            )
            by_stage = {stage["stage"]: stage for stage in payload["stages"]}
            self.assertEqual(by_stage["S2_CANONICALIZATION"]["artifact_id"], "canon-1")
            self.assertEqual(by_stage["S5_ANIMATION"]["artifact_id"], "clip-1")
            # No stage is violated, and the un-measured checks are reported
            # as not-evaluable rather than silently passing.
            self.assertIsNone(payload["first_violated_stage"])
            for stage in payload["stages"]:
                self.assertEqual(stage["counts"]["error"], 0, stage)
            self.assertTrue(payload["not_evaluable"])
            self.assertIn(
                {"stage": "S2_CANONICALIZATION", "check": "s2.axis_convention",
                 "missing_metrics": ["up_axis", "facing_axis"]},
                payload["not_evaluable"],
            )
            # The operator accept is what is left, so that is the advice.
            self.assertEqual(payload["next_action"], {"kind": "advance", "stage": "S6_EXPORT"})
            self.assertEqual(payload["stages_not_run"], ["S6_EXPORT"])

    def test_a_run_that_skipped_canonicalize_is_told_to_canonicalize(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            store = RunStore(root / "runs")
            run = store.create("No canonicalize", "img-1", "image")
            self._write_artifact(artifacts, "img-1", "image", {})
            self._write_artifact(artifacts, "gen-1", "glb", self.GENERATE_METADATA)
            rig = json.loads(json.dumps(self.AUTO_RIG_METADATA))
            rig["source_artifact_id"] = "gen-1"
            self._write_artifact(artifacts, "rig-1", "glb", rig)
            for stage, source_id, output_id in (("generate", "img-1", "gen-1"),
                                                ("auto-rig", "gen-1", "rig-1")):
                attempt = store.start_attempt(run.id, stage, source_id, {})
                store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id=output_id)

            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "ARTIFACTS", artifacts):
                    payload = client.get(f"/api/runs/{run.id}/policy").json()
            finally:
                main.run_store = previous_store

            # S2 is mandatory and no canonicalize artifact exists anywhere in
            # this run's lineage, so the loop stops there rather than
            # reporting on the rig built from an uncanonicalized mesh.
            self.assertEqual(
                payload["next_action"], {"kind": "advance", "stage": "S2_CANONICALIZATION"}
            )
            self.assertIn("S2_CANONICALIZATION", payload["stages_not_run"])

    def test_a_violated_contract_reports_the_blame_table_owner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)
            broken = json.loads(json.dumps(self.CANONICALIZE_METADATA))
            # The blame table's "08-30 Mary hips" row: unwelded duplicate
            # geometry that a welded component count alone would miss.
            broken["attempts"][-1]["components_unwelded"] = 4
            self._write_artifact(artifacts, "canon-1", "glb", broken)

            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "ARTIFACTS", artifacts):
                    payload = client.get(f"/api/runs/{run.id}/policy").json()
            finally:
                main.run_store = previous_store

            self.assertEqual(payload["first_violated_stage"], "S2_CANONICALIZATION")
            self.assertEqual(payload["next_action"]["kind"], "stop-at-stage")
            self.assertEqual(payload["next_action"]["stage"], "S2_CANONICALIZATION")
            self.assertEqual(
                payload["next_action"]["blame"][0]["finding_code"],
                "s2.single_connected_component_unwelded",
            )

    def _policy(self, store, run_id, artifacts):
        previous_store = main.run_store
        try:
            main.run_store = store
            with patch.object(main, "ARTIFACTS", artifacts):
                return client.get(f"/api/runs/{run_id}/policy").json()
        finally:
            main.run_store = previous_store

    def test_a_rig_built_on_a_remediated_mesh_is_reported_stale_and_re_entered(self):
        # Run 070772d8 on the box: S2 was remediated into a new canonical
        # mesh while S3/S4 still held a rig of the mesh it replaced. The
        # endpoint used to advise "advance -> S5" -- straight over a rig of
        # geometry the run had already thrown away.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="rig-1")
            remediated = json.loads(json.dumps(self.CANONICALIZE_METADATA))
            remediated["settings"] = {
                pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: {
                    "initiator": "api", "mode": "remediation",
                    "stage": "S2_CANONICALIZATION", "source_artifact_id": "gen-1",
                }
            }
            self._write_artifact(artifacts, "canon-2", "glb", remediated)
            attempt = store.start_attempt(run.id, "canonicalize", "gen-1", {})
            store.finish_attempt(
                run.id, attempt.id, "completed", output_artifact_id="canon-2"
            )

            payload = self._policy(store, run.id, artifacts)

            self.assertEqual(
                payload["next_action"], {"kind": "advance", "stage": "S3_SKELETON_FIT"}
            )
            self.assertEqual(
                [entry["stage"] for entry in payload["stale_stages"]],
                ["S3_SKELETON_FIT", "S4_SKINNING"],
            )
            entry = payload["stale_stages"][0]
            self.assertEqual(entry["artifact_id"], "rig-1")
            self.assertEqual(
                entry["obsoleted_by"],
                {"stage": "S2_CANONICALIZATION", "artifact_id": "canon-2"},
            )
            # Reported, not deleted: the stale stage keeps its own verdict.
            by_stage = {stage["stage"]: stage for stage in payload["stages"]}
            self.assertTrue(by_stage["S3_SKELETON_FIT"]["stale"])
            self.assertEqual(by_stage["S3_SKELETON_FIT"]["artifact_id"], "rig-1")
            self.assertFalse(by_stage["S2_CANONICALIZATION"]["stale"])
            self.assertEqual(by_stage["S2_CANONICALIZATION"]["artifact_id"], "canon-2")

    def test_a_rig_inherited_from_the_runs_own_source_is_not_stale(self):
        # Run a459f7b0: the source upload was ALREADY rigged, a clip was
        # attached to it, and the run then produced its own rig. Two rigs,
        # nothing remediated -- calling the clip stale would re-run a stage
        # nothing invalidated.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            store = RunStore(root / "runs")
            inherited = json.loads(json.dumps(self.AUTO_RIG_METADATA))
            inherited["source_artifact_id"] = "canon-1"
            self._write_artifact(artifacts, "img-1", "image", {})
            self._write_artifact(artifacts, "gen-1", "glb", self.GENERATE_METADATA)
            self._write_artifact(artifacts, "canon-1", "glb", self.CANONICALIZE_METADATA)
            self._write_artifact(artifacts, "rig-0", "glb", inherited)
            run = store.create("Inherited rig", "rig-0", "glb")
            rerig = json.loads(json.dumps(self.AUTO_RIG_METADATA))
            rerig["source_artifact_id"] = "rig-0"
            self._write_artifact(artifacts, "clip-1", "glb", {
                **self.APPLY_CLIP_METADATA, "source_artifact_id": "rig-0",
            })
            self._write_artifact(artifacts, "rig-1", "glb", rerig)
            for stage, source_id, output_id in (
                ("apply-clip", "rig-0", "clip-1"), ("auto-rig", "rig-0", "rig-1")
            ):
                attempt = store.start_attempt(run.id, stage, source_id, {})
                store.finish_attempt(
                    run.id, attempt.id, "completed", output_artifact_id=output_id
                )

            payload = self._policy(store, run.id, artifacts)

            self.assertEqual(payload["stale_stages"], [])
            self.assertFalse(any(stage["stale"] for stage in payload["stages"]))
            self.assertEqual(
                payload["next_action"], {"kind": "advance", "stage": "S6_EXPORT"}
            )

    def test_an_accepted_run_is_done(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)
            # `RunStore.review` only accepts a retarget-motion output, so
            # stamp the review fields the S6 gate reads directly.
            manifest = store.get(run.id)
            manifest.review_status = ReviewState.ACCEPTED
            manifest.reviewed_artifact_id = "clip-1"
            store._save(manifest)

            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "ARTIFACTS", artifacts):
                    payload = client.get(f"/api/runs/{run.id}/policy").json()
            finally:
                main.run_store = previous_store

            s6 = next(s for s in payload["stages"] if s["stage"] == "S6_EXPORT")
            self.assertEqual(s6["status"], "green")
            self.assertEqual(payload["next_action"], {"kind": "done", "stage": None})

    def test_unknown_run_is_a_404(self):
        response = client.get("/api/runs/deadbeef/policy")
        self.assertEqual(response.status_code, 404, response.text)

    def test_job_execution_stamps_a_compact_policy_block_on_job_and_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="rig-1")
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            source = main.Artifact(
                id="rig-1", name="rig-1.glb", kind="glb",
                path=str(artifacts / "rig-1.glb"), size_bytes=1, created_at=0,
                metadata=self.AUTO_RIG_METADATA,
            )
            attempt = store.reserve_attempt(
                run.id, "job-policy-1", "apply-clip", "rig-1", {},
            )
            job = main.JobState(
                id="job-policy-1", kind="apply-clip", run_id=run.id, attempt_id=attempt.id,
            )
            request = main.JobRequest(
                kind="apply-clip",
                submission_id="submission-policy-1",
                source_artifact_id=source.id,
                settings={"clip": "Sword_Attack_Armature", "family": "biped"},
            )

            def apply_clip(_source, output, **_kwargs):
                trimesh.creation.box().export(output)
                return {"backend": "blender-direct-attach", "baked_keyframes": True}

            previous_store = main.run_store
            try:
                main.run_store = store
                with (
                    patch.object(main, "ARTIFACTS", artifacts),
                    patch.object(main, "JOBS_DIR", jobs_dir),
                    patch.object(main, "output_directory", return_value=artifacts),
                    patch.object(main, "read_artifact", return_value=source),
                    patch.object(main.native, "run_apply_clip", side_effect=apply_clip),
                    patch.object(main.native, "run_deformation_diagnostics",
                                 return_value={"complete_action_range": True,
                                               "maximum_p99_edge_stretch": 4.1403,
                                               "frames": []}),
                    patch.object(main.model_manager, "acquire", new=AsyncMock()),
                    patch.object(main.model_manager, "release", new=AsyncMock()),
                ):
                    asyncio.run(main._execute_job(job, request))
                    self.assertEqual(job.status, "completed", (job.error, job.logs))
                    artifact = main.Artifact.model_validate(
                        main._read_json(artifacts / f"{job.output_artifact_id}.json")
                    )
            finally:
                main.run_store = previous_store

            block = artifact.metadata["pipeline_policy"]
            self.assertEqual(block, job.pipeline_policy)
            self.assertEqual(block["policy_version"], 1)
            self.assertEqual(
                [stage["stage"] for stage in block["stages"]],
                ["S0_SOURCE", "S1_GENERATION", "S2_CANONICALIZATION", "S3_SKELETON_FIT",
                 "S4_SKINNING", "S5_ANIMATION", "S6_EXPORT"],
            )
            # The job's own not-yet-saved output is the S5 artifact.
            s5 = next(stage for stage in block["stages"] if stage["stage"] == "S5_ANIMATION")
            self.assertIsNone(s5["artifact_id"])
            self.assertEqual(s5["counts"]["error"], 0)
            self.assertTrue(block["not_evaluable"])
            self.assertEqual(block["next_action"]["kind"], "advance")
            # Compact form: findings carry no prose message in stored metadata.
            for stage in block["stages"]:
                for finding in stage["findings"]:
                    self.assertNotIn("message", finding)
            # Advisory only -- promotion is still decided by the quality gates.
            self.assertIn("quality_decisions", artifact.metadata)

    def test_a_policy_evaluation_failure_never_fails_the_job(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            jobs_dir = root / "jobs"
            artifacts.mkdir()
            jobs_dir.mkdir()
            image_path = root / "mary.png"
            Image.new("RGBA", (8, 8), "white").save(image_path)
            source = main.Artifact(
                id="source-1", name="mary.png", kind="image", path=str(image_path),
                size_bytes=image_path.stat().st_size, created_at=0,
            )
            job = main.JobState(id="job-policy-boom", kind="generate")
            request = main.JobRequest(
                kind="generate", submission_id="submission-policy-boom",
                source_artifact_id=source.id,
            )

            def generate(_source, output, _resolution, _progress, **_kwargs):
                trimesh.creation.box().export(output)
                return {"generator": "test"}

            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "JOBS_DIR", jobs_dir),
                patch.object(main, "output_directory", return_value=artifacts),
                patch.object(main, "read_artifact", return_value=source),
                patch.object(main, "_evaluate_job_policy", side_effect=RuntimeError("boom")),
                patch.object(main.native, "trellis_refiner_ready", return_value=True),
                patch.object(main.native, "run_trellis_fp8", side_effect=generate),
                patch.object(main.model_manager, "acquire", new=AsyncMock()),
                patch.object(main.model_manager, "release", new=AsyncMock()),
            ):
                asyncio.run(main._execute_job(job, request))
                artifact = main.Artifact.model_validate(
                    main._read_json(artifacts / f"{job.output_artifact_id}.json")
                )

            self.assertEqual(job.status, "completed")
            self.assertIsNone(job.pipeline_policy)
            self.assertNotIn("pipeline_policy", artifact.metadata)
            self.assertTrue(
                any("Pipeline policy evaluation skipped" in line for line in job.logs),
                job.logs,
            )


class PreGenerateViewConsistencyTests(unittest.TestCase):
    """Bead 8c9gf: a multi-view `generate` is checked against the front image
    it will ACTUALLY be conditioned on, before any GPU time is spent.

    `synthesize-views` only checks views it just drew, against the front it
    drew them from. This gate covers the cases that never went through it --
    an operator-uploaded view, a set synthesized from a different front, or
    a stale set left in the slots after the reference image changed.
    """

    BACKGROUND = (200, 200, 200)
    SUBJECT = (40, 40, 60)

    @classmethod
    def _figure(cls, path, *, skirt=False):
        """A standing figure: two separated legs, or one fused skirt hem."""
        size, hip, foot, centre = 400, 260, 380, 200
        image = Image.new("RGB", (size, size), cls.BACKGROUND)
        pixels = image.load()
        for y in range(20, hip):
            for x in range(centre - 40, centre + 40):
                pixels[x, y] = cls.SUBJECT
        for y in range(hip, foot):
            if skirt:
                flare = int(46 * (0.6 + 0.4 * (y - hip) / (foot - hip)))
                spans = [(centre - flare, centre + flare)]
            else:
                spans = [(centre - 40, centre - 12), (centre + 12, centre + 40)]
            for start, end in spans:
                for x in range(start, end):
                    pixels[x, y] = cls.SUBJECT
        image.save(path, format="PNG")
        return path

    def _submit(self, root, *, settings=None, skirt_left=True):
        """Post a front+left generate and return (response, jobs seen)."""
        front_path = self._figure(root / "front.png")
        left_path = self._figure(root / "left.png", skirt=skirt_left)
        front = main.Artifact(
            id="front-vc", name="front.png", kind="image", path=str(front_path),
            size_bytes=front_path.stat().st_size, created_at=0,
        )
        left = main.Artifact(
            id="left-vc", name="left.png", kind="image", path=str(left_path),
            size_bytes=left_path.stat().st_size, created_at=0,
        )
        artifacts = {front.id: front, left.id: left}
        payload = {
            "kind": "generate",
            "submission_id": f"submission-vc-{uuid.uuid4().hex}",
            "source_views": {"front": front.id, "left": left.id},
            "settings": settings or {},
        }
        previous_store = main.run_store
        previous_jobs_dir = main.JOBS_DIR
        previous_jobs = main.jobs.copy()
        main.run_store = RunStore(root / "runs")
        main.JOBS_DIR = root / "jobs"
        main.JOBS_DIR.mkdir(exist_ok=True)
        main.jobs.clear()
        try:
            with (
                patch.object(main, "read_artifact", side_effect=lambda artifact_id: artifacts[artifact_id]),
                patch.object(main, "_model_readiness", return_value={
                    "trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True,
                }),
                patch.object(main, "_execute_job", new=AsyncMock()),
            ):
                return client.post("/api/jobs", json=payload)
        finally:
            main.run_store = previous_store
            main.JOBS_DIR = previous_jobs_dir
            main.jobs.clear()
            main.jobs.update(previous_jobs)

    def test_a_left_view_that_contradicts_the_front_blocks_the_generate_by_name(self):
        with TemporaryDirectory() as directory:
            response = self._submit(Path(directory), skirt_left=True)

        self.assertEqual(response.status_code, 422, response.text)
        detail = response.json()["detail"]
        # The operator must learn WHICH tile to fix from the error alone.
        self.assertIn("LEFT view contradicts the front image", detail)
        self.assertIn("regenerate or remove it", detail)
        self.assertIn("skip_view_consistency", detail)

    def test_the_skip_flag_lets_a_contradicting_set_through(self):
        with TemporaryDirectory() as directory:
            response = self._submit(
                Path(directory), settings={"skip_view_consistency": True}, skirt_left=True,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["source_artifact_id"], "front-vc")

    def test_a_consistent_view_set_is_accepted_without_warnings(self):
        with TemporaryDirectory() as directory:
            response = self._submit(Path(directory), skirt_left=False)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["warnings"], [])

    def test_an_unreadable_view_warns_instead_of_blocking(self):
        # A gate that turns a missing file into a refused generate is a worse
        # failure than the one it guards against.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            front_path = self._figure(root / "front.png")
            front = main.Artifact(
                id="front-vc", name="front.png", kind="image", path=str(front_path),
                size_bytes=front_path.stat().st_size, created_at=0,
            )
            missing = main.Artifact(
                id="left-vc", name="left.png", kind="image",
                path=str(root / "does-not-exist.png"), size_bytes=1, created_at=0,
            )
            artifacts = {front.id: front, missing.id: missing}
            payload = {
                "kind": "generate",
                "submission_id": f"submission-vc-{uuid.uuid4().hex}",
                "source_views": {"front": front.id, "left": missing.id},
            }
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir(exist_ok=True)
            main.jobs.clear()
            try:
                with (
                    patch.object(main, "read_artifact", side_effect=lambda artifact_id: artifacts[artifact_id]),
                    patch.object(main, "_model_readiness", return_value={
                        "trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True,
                    }),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json=payload)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(
            any("could not be read" in warning for warning in response.json()["warnings"]),
            response.json()["warnings"],
        )

    def test_a_single_image_generate_never_runs_the_gate(self):
        # No source_views means no conditioning views to contradict anything.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            front_path = self._figure(root / "front.png")
            front = main.Artifact(
                id="front-vc", name="front.png", kind="image", path=str(front_path),
                size_bytes=front_path.stat().st_size, created_at=0,
            )
            payload = {
                "kind": "generate",
                "submission_id": f"submission-vc-{uuid.uuid4().hex}",
                "source_artifact_id": front.id,
            }
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir(exist_ok=True)
            main.jobs.clear()
            try:
                with (
                    patch.object(main, "read_artifact", return_value=front),
                    patch.object(main, "_model_readiness", return_value={
                        "trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True,
                    }),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                    patch.object(main, "_conditioning_view_consistency", side_effect=AssertionError("gate ran")),
                ):
                    response = client.post("/api/jobs", json=payload)
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

        self.assertEqual(response.status_code, 200, response.text)


class CanonicalTposeSetTests(unittest.TestCase):
    """Bead 4vquq: Prepare and Auto-generate-views collapse into ONE operator
    step that produces the canonical four-view T-pose set.

    The composite is not a convenience wrapper. It removes a whole class of
    ambiguity structurally: `synthesize-views` had to CHOOSE its reference at
    submission time (`select_view_synthesis_source`, and the live "witch"
    incident that rule exists to fix), while `create-tpose-set` draws the
    T-pose front itself and can only fan views off the front it just made.
    From then on the raw upload is reference-only -- it never conditions a
    biped generate again.
    """

    BACKGROUND = (200, 200, 200)
    SUBJECT = (40, 40, 60)

    @classmethod
    def _figure(cls, path, *, skirt=False):
        """A standing figure: two separated legs, or one fused skirt hem.

        Same generator as PreGenerateViewConsistencyTests -- these are the
        silhouettes `view_consistency` was calibrated against, so a `skirt`
        figure standing in for a generated T-pose front is exactly the
        "the model reinterpreted the character" failure the gate must catch.
        """
        size, hip, foot, centre = 400, 260, 380, 200
        image = Image.new("RGB", (size, size), cls.BACKGROUND)
        pixels = image.load()
        for y in range(20, hip):
            for x in range(centre - 40, centre + 40):
                pixels[x, y] = cls.SUBJECT
        for y in range(hip, foot):
            if skirt:
                flare = int(46 * (0.6 + 0.4 * (y - hip) / (foot - hip)))
                spans = [(centre - flare, centre + flare)]
            else:
                spans = [(centre - 40, centre - 12), (centre + 12, centre + 40)]
            for start, end in spans:
                for x in range(start, end):
                    pixels[x, y] = cls.SUBJECT
        image.save(path, format="PNG")
        return path

    @staticmethod
    def _register(artifacts_dir, artifact):
        """Write an artifact's real sidecar so `read_artifact` finds it.

        Deliberately not a `read_artifact` patch: the composite reads back
        its own freshly-saved front artifact in the shared job tail, so a
        stubbed reader would hide exactly the wiring under test.
        """
        main._write_json(artifacts_dir / f"{artifact.id}.json", artifact.model_dump())
        return artifact

    def _source(self, root, artifacts_dir, *, artifact_id="reference-image"):
        source_path = self._figure(root / "reference.png")
        return self._register(artifacts_dir, main.Artifact(
            id=artifact_id, name="reference.png", kind="image", path=str(source_path),
            size_bytes=source_path.stat().st_size, created_at=0,
        ))

    def _comfy(self, root, *, front_skirt=False, fail_view=None):
        """A ComfyUI stub that records every prompt it was asked for."""
        calls: list[str] = []

        def generate(_source, output, *, profile, seed, **_kwargs):
            calls.append(profile.id)
            if fail_view is not None and fail_view in profile.id:
                raise RuntimeError(f"ComfyUI rejected the {fail_view} prompt")
            is_front = profile.id.startswith("humanoid-")
            self._figure(output, skirt=front_skirt and is_front)
            return {"prompt_id": f"prompt-{len(calls)}", "model": "flux-2-klein-base-4b-fp8.safetensors"}

        return generate, calls

    def _run_composite(self, root, *, front_skirt=False, fail_view=None, views=None, cancel_after=None):
        """Execute one `create-tpose-set` job end to end against real files.

        `cancel_after` names the view profile fragment after whose prompt the
        operator hits cancel (the job flips to "cancelled" as ComfyUI returns
        that view), modelling a mid-fan cancel.
        """
        artifacts_dir = root / "artifacts"
        jobs_dir = root / "jobs"
        artifacts_dir.mkdir()
        jobs_dir.mkdir()
        source = self._source(root, artifacts_dir)
        store = RunStore(root / "runs")
        run = store.create("Punk T-pose set", source.id, source.kind, "biped")
        job = main.JobState(id="tpose-set-job", kind="create-tpose-set", run_id=run.id)
        settings = {
            "adapter": "comfyui",
            "profile": "humanoid-t-pose-rig-v1",
            "views": list(views or main.TPOSE_SET_VIEW_ROLES),
        }
        attempt = store.reserve_attempt(run.id, job.id, "create-tpose-set", source.id, settings)
        job.attempt_id = attempt.id
        request = main.JobRequest(
            kind="create-tpose-set",
            submission_id=f"tpose-set-{uuid.uuid4().hex}",
            source_artifact_id=source.id,
            run_id=run.id,
            seed=7,
            settings=settings,
        )
        generate, calls = self._comfy(root, front_skirt=front_skirt, fail_view=fail_view)
        if cancel_after is not None:
            inner_generate = generate

            def generate(_source, output, *, profile, seed, **kwargs):
                result = inner_generate(_source, output, profile=profile, seed=seed, **kwargs)
                if cancel_after in profile.id:
                    job.status = "cancelled"
                return result
        previous_store = main.run_store
        try:
            main.run_store = store
            with (
                patch.object(main, "ARTIFACTS", artifacts_dir),
                patch.object(main, "JOBS_DIR", jobs_dir),
                patch.object(main, "output_directory", return_value=artifacts_dir),
                patch.object(main, "run_comfyui_image_edit", side_effect=generate),
                patch.object(main.model_manager, "acquire", new_callable=AsyncMock),
                patch.object(main.model_manager, "release", new_callable=AsyncMock),
            ):
                asyncio.run(main._execute_job(job, request))
                artifact = (
                    main.Artifact.model_validate(
                        main._read_json(artifacts_dir / f"{job.output_artifact_id}.json")
                    )
                    if job.output_artifact_id
                    else None
                )
                members = {}
                if artifact is not None:
                    for role, member_id in artifact.metadata["tpose_set"]["members"].items():
                        members[role] = main.Artifact.model_validate(
                            main._read_json(artifacts_dir / f"{member_id}.json")
                        )
        finally:
            main.run_store = previous_store
        return job, artifact, members, calls, store.get(run.id), source

    def test_a_mid_fan_cancel_still_registers_the_partial_set_and_cleans_up(self):
        # Cancel lands as the back view returns: the front and back are
        # already saved artifacts, so they must carry a `tpose_set` block
        # (incomplete, naming the roles never drawn) instead of being
        # orphaned, and the working file must be unlinked like every other
        # cancelled job.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, artifact, _members, calls, run, _source = self._run_composite(
                root, cancel_after="view-synthesis-back",
            )
            artifacts_dir = root / "artifacts"
            saved = [
                main.Artifact.model_validate(main._read_json(path))
                for path in artifacts_dir.glob("*.json")
                if path.stem != "reference-image"
            ]
            leftover_pngs = [p for p in artifacts_dir.glob("*.png") if p.stem != "reference"]

        self.assertEqual(job.status, "cancelled")
        self.assertIsNone(artifact)  # a cancelled job publishes no output
        self.assertEqual(calls, ["humanoid-t-pose-rig-v1", "view-synthesis-back-v4"])
        by_role = {item.metadata["tpose_set"]["role"]: item for item in saved}
        self.assertEqual(sorted(by_role), ["back", "front"])
        membership = by_role["front"].metadata["tpose_set"]
        self.assertFalse(membership["complete"])
        self.assertEqual(sorted(membership["members"]), ["back", "front"])
        self.assertEqual(sorted(membership["failed_members"]), ["left", "right"])
        self.assertIn("cancelled", membership["failed_members"]["left"])
        self.assertEqual(by_role["back"].metadata["tpose_set"]["members"], membership["members"])
        # Only the registered members' files remain; the job's working
        # output was unlinked by the shared tail.
        self.assertEqual(
            sorted(Path(item.path).name for item in saved),
            sorted(p.name for p in leftover_pngs),
        )
        self.assertEqual(run.attempts[-1].status, "cancelled")

    def test_one_job_produces_the_gated_front_and_the_three_views_as_one_set(self):
        with TemporaryDirectory() as directory:
            job, front, members, calls, run, source = self._run_composite(Path(directory))

        self.assertEqual(job.status, "completed", job.error)
        # One front, then exactly the three views fanned off it.
        self.assertEqual(
            calls,
            [
                "humanoid-t-pose-rig-v1",
                "view-synthesis-back-v4",
                "view-synthesis-left-v7",
                "view-synthesis-right-v7",
            ],
        )
        # Staged progress an operator can read while it runs.
        self.assertTrue(any("Generating the T-pose front… 1/4" in line for line in job.logs), job.logs)
        self.assertTrue(any("Generating the back view… 2/4" in line for line in job.logs), job.logs)
        self.assertTrue(any("Generating the right view… 4/4" in line for line in job.logs), job.logs)
        # The front is the job's output, and the set is complete.
        membership = front.metadata["tpose_set"]
        self.assertEqual(membership["role"], "front")
        self.assertTrue(membership["complete"])
        self.assertEqual(membership["failed_members"], {})
        self.assertEqual(sorted(membership["members"]), ["back", "front", "left", "right"])
        self.assertEqual(membership["members"]["front"], front.id)
        self.assertEqual(membership["reference_artifact_id"], source.id)
        self.assertEqual(front.metadata["tpose_front_gate"]["status"], "passed")
        # Every member carries the SAME set id, its own role, and the shared
        # member map -- the grouping travels with the artifacts.
        for role, member in members.items():
            self.assertEqual(member.metadata["tpose_set"]["set_id"], membership["set_id"])
            self.assertEqual(member.metadata["tpose_set"]["role"], role)
            self.assertEqual(member.metadata["tpose_set"]["members"], membership["members"])
        # Each view keeps its existing per-view provenance, drawn from the
        # T-POSE FRONT -- never from the raw reference image.
        for role in main.TPOSE_SET_VIEW_ROLES:
            view = members[role]
            self.assertEqual(view.metadata["synthesized_view"], role)
            self.assertEqual(view.metadata["source_artifact_id"], front.id)
            self.assertEqual(view.metadata["view_synthesis_source"], "tpose-set-front")
            self.assertIn("view_consistency", view.metadata)
            self.assertIn("view_synthesis_profile", view.metadata)
        # A set is a candidate, not a promotion: it waits for the operator.
        self.assertFalse(job.promoted_output)
        self.assertEqual(run.current_artifact_id, source.id)

    def test_a_front_that_fails_its_own_gate_never_fans_the_three_views(self):
        # The front comes back as a floor-length garment -- the exact failure
        # `view_consistency` was written for. Fanning three redraws off it
        # would spend three more ComfyUI passes on a bad reference.
        with TemporaryDirectory() as directory:
            job, front, _members, calls, run, _source = self._run_composite(
                Path(directory), front_skirt=True,
            )

        self.assertEqual(job.status, "failed")
        self.assertIsNone(front)
        self.assertEqual(calls, ["humanoid-t-pose-rig-v1"])
        self.assertIn("did not pass its own check", job.error)
        self.assertIn("contradicts the reference image", job.error)
        self.assertEqual(run.current_artifact_id, run.source_artifact_id)

    def test_one_failed_view_leaves_the_rest_of_the_set_intact(self):
        # Losing an already-paid-for front and two good views to one bad
        # redraw is the worse outcome; per-tile regenerate is the remedy.
        with TemporaryDirectory() as directory:
            job, front, members, calls, _run, _source = self._run_composite(
                Path(directory), fail_view="left",
            )

        self.assertEqual(job.status, "completed", job.error)
        self.assertEqual(len(calls), 4)
        membership = front.metadata["tpose_set"]
        self.assertFalse(membership["complete"])
        self.assertEqual(sorted(membership["members"]), ["back", "front", "right"])
        self.assertIn("left", membership["failed_members"])
        self.assertNotIn("left", members)
        self.assertTrue(
            any("left view could not be generated" in warning for warning in job.warnings),
            job.warnings,
        )

    def test_the_gate_records_the_palette_distance_it_does_not_gate_on(self):
        # `view_consistency` publishes colour distance and deliberately never
        # gates it (the live robed-witch failure wore the same purple as the
        # jumpsuit it replaced). The composite keeps that rule and records it.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self._figure(root / "reference.png").read_bytes()
            front = self._figure(root / "front.png").read_bytes()

            result = main.evaluate_tpose_front(reference, front)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["front_measurable"])
        self.assertEqual(result["reasons"], [])
        self.assertIsNotNone(result["palette_distance"])

    def test_an_object_run_cannot_build_a_tpose_set(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            source = self._source(root, artifacts_dir)
            previous_store = main.run_store
            previous_jobs_dir = main.JOBS_DIR
            previous_jobs = main.jobs.copy()
            main.run_store = RunStore(root / "runs")
            main.JOBS_DIR = root / "jobs"
            main.JOBS_DIR.mkdir()
            main.jobs.clear()
            run = main.run_store.create("Crate", source.id, source.kind, "object")
            try:
                with (
                    patch.object(main, "ARTIFACTS", artifacts_dir),
                    patch.object(main, "_execute_job", new=AsyncMock()),
                ):
                    response = client.post("/api/jobs", json={
                        "kind": "create-tpose-set",
                        "submission_id": f"tpose-object-{uuid.uuid4().hex}",
                        "source_artifact_id": source.id,
                        "run_id": run.id,
                    })
            finally:
                main.run_store = previous_store
                main.JOBS_DIR = previous_jobs_dir
                main.jobs.clear()
                main.jobs.update(previous_jobs)

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("biped", response.json()["detail"])

    def _run_with_approved_set(self, root):
        """A biped run whose canonical set exists and has been approved."""
        artifacts_dir = root / "artifacts"
        artifacts_dir.mkdir()
        source = self._source(root, artifacts_dir)
        ids = {"front": "set-front", "back": "set-back", "left": "set-left", "right": "set-right"}
        membership = {
            "set_id": "set-1",
            "role": "front",
            "front_artifact_id": ids["front"],
            "reference_artifact_id": source.id,
            "job_id": "tpose-set-job",
            "members": dict(ids),
            "failed_members": {},
            "complete": True,
        }
        for role, artifact_id in ids.items():
            path = self._figure(root / f"{role}.png")
            self._register(artifacts_dir, main.Artifact(
                id=artifact_id, name=f"T-pose {role}.png", kind="image", path=str(path),
                size_bytes=path.stat().st_size, created_at=0,
                metadata={"stage": "create-tpose-set", "tpose_set": {**membership, "role": role}},
            ))
        store = RunStore(root / "runs")
        run = store.create("Punk", source.id, source.kind, "biped")
        attempt = store.reserve_attempt(run.id, "tpose-set-job", "create-tpose-set", source.id, {})
        store.finish_attempt(
            run.id, attempt.id, "completed",
            output_artifact_id=ids["front"], promote_output=True,
        )
        return store, store.get(run.id), source, ids

    def _post_generate(self, root, store, payload):
        previous_store = main.run_store
        previous_jobs_dir = main.JOBS_DIR
        previous_jobs = main.jobs.copy()
        main.run_store = store
        main.JOBS_DIR = root / "jobs"
        main.JOBS_DIR.mkdir(exist_ok=True)
        main.jobs.clear()
        try:
            with (
                patch.object(main, "ARTIFACTS", root / "artifacts"),
                patch.object(main, "_model_readiness", return_value={
                    "trellis_gguf_ready": True, "trellis_refiner_ready": True, "instant_meshes_ready": True,
                }),
                patch.object(main, "_execute_job", new=AsyncMock()),
            ):
                return client.post("/api/jobs", json=payload)
        finally:
            main.run_store = previous_store
            main.JOBS_DIR = previous_jobs_dir
            main.jobs.clear()
            main.jobs.update(previous_jobs)

    def _add_unreviewed_set(self, root, store, run, source):
        """A second `create-tpose-set` that completed but was never approved
        -- the state every fresh set is in until the operator promotes it."""
        newer_path = self._figure(root / "newer-front.png")
        self._register(root / "artifacts", main.Artifact(
            id="newer-front", name="T-pose front (newer).png", kind="image", path=str(newer_path),
            size_bytes=newer_path.stat().st_size, created_at=0,
            metadata={"stage": "create-tpose-set", "tpose_set": {
                "set_id": "set-2", "role": "front", "front_artifact_id": "newer-front",
                "reference_artifact_id": source.id, "job_id": "tpose-set-job-2",
                "members": {"front": "newer-front"}, "failed_members": {}, "complete": True,
            }},
        ))
        attempt = store.reserve_attempt(run.id, "tpose-set-job-2", "create-tpose-set", source.id, {})
        store.finish_attempt(
            run.id, attempt.id, "completed", output_artifact_id="newer-front", promote_output=False,
        )
        return store.get(run.id)

    def test_the_approved_set_stays_the_conditioning_when_a_newer_set_is_unreviewed(self):
        # Set A approved (current_artifact_id), then set B drawn and left
        # unreviewed. Generate must keep conditioning on A -- picking the
        # newest set would demand B be promoted and make A ungeneratable.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, source, ids = self._run_with_approved_set(root)
            run = self._add_unreviewed_set(root, store, run, source)
            self.assertEqual(run.current_artifact_id, ids["front"])
            self.assertEqual(main.latest_tpose_set_front(run), ids["front"])

            explicit = self._post_generate(root, store, {
                "kind": "generate",
                "submission_id": f"generate-approved-{uuid.uuid4().hex}",
                "source_artifact_id": ids["front"],
                "run_id": run.id,
            })
            implicit = self._post_generate(root, store, {
                "kind": "generate",
                "submission_id": f"generate-implicit-{uuid.uuid4().hex}",
                "source_artifact_id": source.id,
                "run_id": run.id,
            })

        for response in (explicit, implicit):
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["source_artifact_id"], ids["front"])
            self.assertEqual(body["request_payload"]["source_views"], ids)
            self.assertEqual(body["request_payload"]["settings"]["tpose_set_id"], "set-1")

    def test_with_nothing_approved_the_newest_set_is_the_conditioning(self):
        # No promotion anywhere: the fallback is still the newest set.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            source = self._source(root, artifacts_dir)
            store = RunStore(root / "runs")
            run = store.create("Punk", source.id, source.kind, "biped")
            for index in ("1", "2"):
                attempt = store.reserve_attempt(run.id, f"job-{index}", "create-tpose-set", source.id, {})
                store.finish_attempt(
                    run.id, attempt.id, "completed", output_artifact_id=f"front-{index}", promote_output=False,
                )
            self.assertEqual(main.latest_tpose_set_front(store.get(run.id)), "front-2")

    def test_generate_is_conditioned_on_the_set_not_the_raw_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, source, ids = self._run_with_approved_set(root)
            # The studio submits the run's raw reference as the front; the
            # backend must substitute the canonical set for it.
            response = self._post_generate(root, store, {
                "kind": "generate",
                "submission_id": f"generate-set-{uuid.uuid4().hex}",
                "source_artifact_id": source.id,
                "run_id": run.id,
            })

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["source_artifact_id"], ids["front"])
        self.assertEqual(body["request_payload"]["source_views"], ids)
        self.assertEqual(body["request_payload"]["settings"]["tpose_set_id"], "set-1")
        self.assertEqual(body["request_payload"]["settings"]["tpose_set_overrides"], {})
        self.assertTrue(
            any("canonical T-pose set" in warning for warning in body["warnings"]),
            body["warnings"],
        )

    def test_an_uploaded_view_replaces_its_set_member_and_is_recorded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, _source, ids = self._run_with_approved_set(root)
            replacement = root / "operator-left.png"
            self._figure(replacement)
            self._register(root / "artifacts", main.Artifact(
                id="operator-left", name="operator-left.png", kind="image",
                path=str(replacement), size_bytes=replacement.stat().st_size, created_at=0,
            ))
            response = self._post_generate(root, store, {
                "kind": "generate",
                "submission_id": f"generate-override-{uuid.uuid4().hex}",
                "source_views": {"front": ids["front"], "left": "operator-left"},
                "run_id": run.id,
            })

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()["request_payload"]
        self.assertEqual(payload["source_views"]["left"], "operator-left")
        # The roles the operator did NOT override still come from the set.
        self.assertEqual(payload["source_views"]["back"], ids["back"])
        self.assertEqual(payload["source_views"]["right"], ids["right"])
        self.assertEqual(payload["settings"]["tpose_set_overrides"], {"left": "operator-left"})

    def test_a_different_front_is_refused_unless_the_caller_opts_out(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, _source, _ids = self._run_with_approved_set(root)
            stray = root / "stray.png"
            self._figure(stray)
            self._register(root / "artifacts", main.Artifact(
                id="stray-front", name="stray.png", kind="image", path=str(stray),
                size_bytes=stray.stat().st_size, created_at=0,
            ))
            refused = self._post_generate(root, store, {
                "kind": "generate",
                "submission_id": f"generate-stray-{uuid.uuid4().hex}",
                "source_artifact_id": "stray-front",
                "run_id": run.id,
            })

        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertIn("canonical T-pose set", refused.json()["detail"])
        self.assertIn("skip_tpose_set", refused.json()["detail"])

    def test_the_composite_front_can_be_approved_like_a_prepared_candidate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            source = self._source(root, artifacts_dir)
            front_path = self._figure(root / "front.png")
            front = self._register(artifacts_dir, main.Artifact(
                id="set-front", name="T-pose front.png", kind="image", path=str(front_path),
                size_bytes=front_path.stat().st_size, created_at=0,
            ))
            store = RunStore(root / "runs")
            run = store.create("Punk", source.id, source.kind, "biped")
            attempt = store.reserve_attempt(run.id, "job-1", "create-tpose-set", source.id, {})
            store.finish_attempt(
                run.id, attempt.id, "completed",
                output_artifact_id=front.id, promote_output=False,
            )
            previous_store = main.run_store
            try:
                main.run_store = store
                with patch.object(main, "ARTIFACTS", artifacts_dir):
                    promoted = asyncio.run(main.promote_run_artifact(
                        run.id, front.id,
                        main.ArtifactPromotionRequest(reason="Approved the canonical T-pose set"),
                    ))
            finally:
                main.run_store = previous_store

        self.assertEqual(promoted.current_artifact_id, front.id)


class StampSourceFidelityTests(unittest.TestCase):
    """`main._stamp_source_fidelity`: the advisory generate/canonicalize/
    project-views hook that scores an output against its ROOT source photo."""

    def _report(self, side_by_side_path):
        return {
            "policy_version": "source-fidelity-v1",
            "silhouette_iou": 0.95,
            "masked_ssim": 0.85,
            "face_ssim": 0.8,
            "perceptual_distance": 0.05,
            "masked_luma_delta": 0.01,
            "masked_rgb_l1": 0.02,
            "face_pixels": 4096,
            "alignment": {"bbox_width_ratio": 1.0},
            "camera": {"fov_x_degrees": 40.0},
            "render": str(side_by_side_path.parent / "render.png"),
            "side_by_side": str(side_by_side_path),
        }

    def test_success_saves_the_side_by_side_as_an_image_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            photo = root / "photo.png"
            photo.write_bytes(b"\x89PNG\r\n\x1a\nphoto")
            output = root / "mesh.glb"
            output.write_bytes(b"glb-bytes")
            side_by_side = root / "side_by_side.png"
            side_by_side.write_bytes(b"\x89PNG\r\n\x1a\nside-by-side")
            report = self._report(side_by_side)

            job = main.JobState(id="job-fidelity", kind="generate")
            metadata: dict = {"quality_decisions": {}}

            with (
                patch.dict(os.environ, {"AISMITH_DATA_DIR": str(root)}),
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main.native, "resolve_source_photo", return_value=(photo, "photo-id", "ok")),
                patch.object(main.native, "measure_source_fidelity", return_value=report) as measure,
            ):
                main._stamp_source_fidelity(job, "source-id", output, metadata, {})
                measure.assert_called_once()
                artifact = main.read_artifact(metadata["source_fidelity"]["side_by_side_artifact_id"])

            self.assertEqual(artifact.kind, "image")
            self.assertEqual(artifact.metadata["stage"], "source-fidelity")
            self.assertEqual(artifact.metadata["for_job_id"], job.id)
            self.assertEqual(artifact.metadata["photo_artifact_id"], "photo-id")
            self.assertEqual(artifact.metadata["source_artifact_id"], "source-id")
            self.assertEqual(metadata["source_fidelity"]["photo_artifact_id"], "photo-id")
            self.assertEqual(
                metadata["quality_decisions"]["source_fidelity"]["status"], "green"
            )
            self.assertTrue(job.promoted_output)

    def test_a_raising_measurement_is_advisory_not_an_exception(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "photo.png"
            photo.write_bytes(b"\x89PNG\r\n\x1a\nphoto")
            output = root / "mesh.glb"
            output.write_bytes(b"glb-bytes")

            job = main.JobState(id="job-fidelity-fail", kind="canonicalize")
            metadata: dict = {"quality_decisions": {}}

            with (
                patch.object(main.native, "resolve_source_photo", return_value=(photo, "photo-id", "ok")),
                patch.object(main.native, "measure_source_fidelity", side_effect=RuntimeError("boom")),
            ):
                main._stamp_source_fidelity(job, "source-id", output, metadata, {})

            self.assertIn("skipped", metadata["source_fidelity"])
            self.assertIn("boom", metadata["source_fidelity"]["skipped"])
            self.assertEqual(
                metadata["quality_decisions"]["source_fidelity"]["status"], "not_evaluable"
            )
            self.assertTrue(any("source fidelity" in w.lower() for w in job.warnings))
            self.assertTrue(job.promoted_output)

    def test_skip_source_fidelity_setting_skips_without_resolving_a_photo(self):
        job = main.JobState(id="job-fidelity-skip", kind="project-views")
        metadata: dict = {"quality_decisions": {}}

        with patch.object(main.native, "resolve_source_photo") as resolve:
            main._stamp_source_fidelity(
                job, "source-id", Path("/nonexistent/mesh.glb"), metadata,
                {"skip_source_fidelity": True},
            )

        resolve.assert_not_called()
        self.assertEqual(
            metadata["source_fidelity"], {"skipped": "disabled by request settings"}
        )
        self.assertEqual(
            metadata["quality_decisions"]["source_fidelity"]["status"], "not_evaluable"
        )


