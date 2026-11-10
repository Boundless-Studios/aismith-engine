"""The bounded pipeline-loop driver: POST /api/runs/{id}/pipeline/step
(bead hlxij).

Every test here mocks job submission (`main.create_job`) -- the point of
this endpoint is that ONE call spends AT MOST ONE job, and proving that
must not itself require a GPU. What is asserted is therefore: which job
kind was chosen for which stage, what input artifact and settings it was
rebuilt from, the provenance stamped on it, and -- just as important --
every path where NOTHING is submitted (needs_input, capped, done, busy).

Metadata fixtures mirror the real shapes on the box (Mary's canonical run:
generate -> voxel canonicalize -> template rig -> direct clip attach), the
same fixtures `PipelinePolicyWiringTests` in test_api.py evaluates against.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import main
from backend.app import pipeline_policy
from backend.app.main import app
from backend.app.run_store import ReviewState, RunStore

client = TestClient(app)


class StageJobKindTests(unittest.TestCase):
    """The stage -> job kind table, and which stages have no job at all."""

    def test_every_automatable_stage_maps_to_the_kind_that_produces_it(self):
        self.assertEqual(
            pipeline_policy.STAGE_JOB_KINDS,
            {
                pipeline_policy.Stage.S2_CANONICALIZATION: "canonicalize",
                pipeline_policy.Stage.S3_SKELETON_FIT: "auto-rig",
                pipeline_policy.Stage.S4_SKINNING: "auto-rig",
                pipeline_policy.Stage.S5_ANIMATION: "apply-clip",
            },
        )

    def test_one_auto_rig_pass_covers_both_the_skeleton_and_the_skinning_stage(self):
        # The mirror image of ARTIFACT_STAGES mapping one auto-rig artifact
        # onto S3 and S4: one job, two stages, in both directions.
        self.assertEqual(
            pipeline_policy.stage_job_kind(pipeline_policy.Stage.S3_SKELETON_FIT),
            pipeline_policy.stage_job_kind(pipeline_policy.Stage.S4_SKINNING),
        )
        self.assertEqual(
            pipeline_policy.ARTIFACT_STAGES["auto-rig"],
            (pipeline_policy.Stage.S3_SKELETON_FIT, pipeline_policy.Stage.S4_SKINNING),
        )

    def test_source_generation_and_export_are_human_input_not_jobs(self):
        for stage, expected in (
            (pipeline_policy.Stage.S0_SOURCE, "source-image"),
            (pipeline_policy.Stage.S1_GENERATION, "generation-request"),
            (pipeline_policy.Stage.S6_EXPORT, "operator-accept"),
        ):
            job_kind, input_required = pipeline_policy.stage_step_kind(stage)
            self.assertIsNone(job_kind, stage)
            self.assertEqual(input_required, expected, stage)

    def test_animation_has_a_job_kind_and_still_wants_a_clip_named(self):
        job_kind, input_required = pipeline_policy.stage_step_kind(
            pipeline_policy.Stage.S5_ANIMATION
        )
        self.assertEqual(job_kind, "apply-clip")
        self.assertEqual(input_required, "clip")

    def test_every_stage_answers_with_a_job_or_an_input(self):
        for stage in pipeline_policy.STAGE_ORDER:
            job_kind, input_required = pipeline_policy.stage_step_kind(stage)
            self.assertTrue(
                job_kind is not None or input_required is not None,
                f"{stage} would leave a step with nothing to say",
            )


class RemediationCountingTests(unittest.TestCase):
    """The cap counts remediations off the run's own recorded settings."""

    def _remediation(self, stage, mode="remediation"):
        return {
            pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: {
                "initiator": "api", "mode": mode, "stage": stage.value,
            }
        }

    def test_counts_only_remediations_of_the_named_stage(self):
        settings = [
            {},
            {"seed": 0},
            self._remediation(pipeline_policy.Stage.S2_CANONICALIZATION),
            self._remediation(pipeline_policy.Stage.S3_SKELETON_FIT),
            self._remediation(pipeline_policy.Stage.S2_CANONICALIZATION),
        ]
        self.assertEqual(
            pipeline_policy.count_stage_remediations(
                settings, pipeline_policy.Stage.S2_CANONICALIZATION
            ),
            2,
        )

    def test_an_advance_step_is_not_a_remediation(self):
        settings = [self._remediation(pipeline_policy.Stage.S2_CANONICALIZATION, mode="advance")]
        self.assertEqual(
            pipeline_policy.count_stage_remediations(
                settings, pipeline_policy.Stage.S2_CANONICALIZATION
            ),
            0,
        )

    def test_malformed_settings_never_raise(self):
        self.assertEqual(
            pipeline_policy.count_stage_remediations(
                [None, "not a dict", {pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: "nope"}],
                pipeline_policy.Stage.S2_CANONICALIZATION,
            ),
            0,
        )


class PipelineStepEndpointTests(unittest.TestCase):
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
    # Exactly what `create_job` folds into every attempt's settings.
    RECORDED_JOB_FIELDS = {
        "resolution": 768, "seed": 0, "target_faces": 10000,
        "target_vertices": None, "quality": "game",
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

    def _mary_run(self, root, *, through="clip-1", overrides=None):
        """Mary's canonical run, materialised through `through`.

        Attempts are reserved with a job id and finished, exactly as a real
        API-submitted job records them, so the step endpoint can find the
        settings and input artifact each stage was actually produced from.
        """
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
            payload = json.loads(json.dumps(metadata))
            payload.update((overrides or {}).get(output_id, {}))
            self._write_artifact(artifacts, output_id, "glb", payload)
            settings = dict(self.RECORDED_JOB_FIELDS)
            if stage == "apply-clip":
                settings.update(clip="Sword_Attack_Armature", family="biped")
            attempt = store.reserve_attempt(
                run.id, f"job-{output_id}", stage, source_id, settings,
            )
            store.mark_attempt_running(run.id, attempt.id, f"job-{output_id}")
            store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id=output_id)
            if output_id == through:
                break
        return store, run, artifacts

    def _step(self, store, artifacts, run_id, body=None, submitted=None):
        """Call the step endpoint with job submission mocked out.

        `submitted` collects the JobRequest each call built, which is where
        the stage->kind choice, the reproduced settings, and the provenance
        block are asserted from.
        """
        submitted = submitted if submitted is not None else []

        async def fake_create_job(request, _background_tasks):
            submitted.append(request)
            job = main.JobState(
                id=f"job-step-{len(submitted)}",
                kind=request.kind,
                submission_id=request.submission_id,
                source_artifact_id=request.source_artifact_id,
                run_id=request.run_id,
                attempt_id=f"attempt-step-{len(submitted)}",
            )
            return job

        previous_store = main.run_store
        try:
            main.run_store = store
            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "create_job", fake_create_job),
            ):
                response = client.post(
                    f"/api/runs/{run_id}/pipeline/step", json=body if body is not None else {}
                )
        finally:
            main.run_store = previous_store
        return response, submitted

    # -- advance ----------------------------------------------------------

    def test_a_run_that_skipped_canonicalize_advances_with_one_canonicalize_job(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            store = RunStore(root / "runs")
            run = store.create("No canonicalize", "img-1", "image")
            self._write_artifact(artifacts, "img-1", "image", {})
            self._write_artifact(artifacts, "gen-1", "glb", self.GENERATE_METADATA)
            attempt = store.reserve_attempt(
                run.id, "job-gen-1", "generate", "img-1", dict(self.RECORDED_JOB_FIELDS),
            )
            store.mark_attempt_running(run.id, attempt.id, "job-gen-1")
            store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id="gen-1")

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["action"], "advanced")
            self.assertEqual(payload["stage"], "S2_CANONICALIZATION")
            self.assertEqual(payload["job_kind"], "canonicalize")
            self.assertEqual(payload["job_id"], "job-step-1")
            self.assertEqual(payload["policy"]["next_action"]["stage"], "S2_CANONICALIZATION")
            # Exactly one job, from the run's current artifact.
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0].kind, "canonicalize")
            self.assertEqual(submitted[0].source_artifact_id, "gen-1")
            self.assertEqual(submitted[0].run_id, run.id)

    def test_an_advance_uses_the_runs_current_artifact_not_the_stages_old_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="canon-1")

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["action"], "advanced")
            self.assertEqual(payload["stage"], "S3_SKELETON_FIT")
            self.assertEqual(submitted[0].kind, "auto-rig")
            self.assertEqual(submitted[0].source_artifact_id, "canon-1")

    # -- needs_input ------------------------------------------------------

    def test_the_animation_stage_asks_for_a_clip_instead_of_picking_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="rig-1")

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["action"], "needs_input")
            self.assertEqual(payload["stage"], "S5_ANIMATION")
            self.assertEqual(payload["input_required"], "clip")
            self.assertEqual(payload["family"], "biped")
            self.assertIsNone(payload["job_id"])
            # Nothing was spent, and the choice is handed back as options.
            self.assertEqual(submitted, [])
            self.assertTrue(payload["clips"], payload)
            self.assertIn(
                "Sword_Attack_Armature", [clip["clip"] for clip in payload["clips"]]
            )

    def test_a_named_clip_lets_the_same_step_submit_the_apply_clip_job(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="rig-1")

            response, submitted = self._step(
                store, artifacts, run.id, body={"clip": "Sword_Attack_Armature"}
            )

            payload = response.json()
            self.assertEqual(payload["action"], "advanced")
            self.assertEqual(payload["job_kind"], "apply-clip")
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0].settings["clip"], "Sword_Attack_Armature")
            self.assertEqual(submitted[0].settings["family"], "biped")
            self.assertEqual(submitted[0].source_artifact_id, "rig-1")

    def test_a_failed_clip_attempts_recorded_clip_is_never_silently_reused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="rig-1")
            # An apply-clip that failed leaves its clip in the run's recorded
            # settings but produces no S5 artifact -- the stage is still
            # "not run", and the driver must still refuse to pick.
            failed = store.reserve_attempt(
                run.id, "job-clip-failed", "apply-clip", "rig-1",
                {**self.RECORDED_JOB_FIELDS, "clip": "Sword_Attack_Armature", "family": "biped"},
            )
            store.mark_attempt_running(run.id, failed.id, "job-clip-failed")
            store.finish_attempt(run.id, failed.id, "failed", error="blender exploded")

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["action"], "needs_input")
            self.assertEqual(payload["input_required"], "clip")
            self.assertEqual(submitted, [])

    def test_an_unreviewed_but_otherwise_complete_run_asks_for_the_operator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["action"], "needs_input")
            self.assertEqual(payload["stage"], "S6_EXPORT")
            self.assertEqual(payload["input_required"], "operator-accept")
            self.assertEqual(submitted, [])

    # -- remediation ------------------------------------------------------

    def _broken_canonicalize_run(self, root):
        """Mary's run with the blame table's "08-30 Mary hips" symptom:
        unwelded duplicate geometry the welded component count would miss."""
        broken = json.loads(json.dumps(self.CANONICALIZE_METADATA))
        broken["attempts"][-1]["components_unwelded"] = 4
        return self._mary_run(root, overrides={"canon-1": broken})

    def test_a_blocked_stage_is_rerun_from_its_original_input_and_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["action"], "remediated")
            self.assertEqual(payload["stage"], "S2_CANONICALIZATION")
            self.assertEqual(payload["job_kind"], "canonicalize")
            self.assertEqual(payload["attempts_used"], 1)
            self.assertEqual(
                payload["attempt_cap"], pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )
            self.assertEqual(len(submitted), 1)
            request = submitted[0]
            # Determinism: the SAME input the failed canonicalize consumed,
            # and the seed/resolution/faces it was submitted with -- not the
            # run's newer current artifact and not today's defaults.
            self.assertEqual(request.source_artifact_id, "gen-1")
            self.assertEqual(request.seed, self.RECORDED_JOB_FIELDS["seed"])
            self.assertEqual(request.resolution, self.RECORDED_JOB_FIELDS["resolution"])
            self.assertEqual(request.target_faces, self.RECORDED_JOB_FIELDS["target_faces"])
            self.assertEqual(request.quality, self.RECORDED_JOB_FIELDS["quality"])
            # The five JobRequest fields are lifted back OUT of settings.
            for key in self.RECORDED_JOB_FIELDS:
                self.assertNotIn(key, request.settings)

    def test_a_remediation_stamps_step_provenance_on_the_job_it_submits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            provenance = submitted[0].settings[pipeline_policy.PIPELINE_STEP_SETTINGS_KEY]
            self.assertEqual(provenance, payload["pipeline_step"])
            self.assertEqual(provenance["initiator"], "api")
            self.assertEqual(provenance["mode"], "remediation")
            self.assertEqual(provenance["stage"], "S2_CANONICALIZATION")
            self.assertEqual(provenance["triggering_stage"], "S2_CANONICALIZATION")
            self.assertEqual(provenance["policy_action"], "stop-at-stage")
            self.assertEqual(provenance["source_artifact_id"], "gen-1")
            self.assertEqual(provenance["attempt"], 1)
            self.assertEqual(
                provenance["attempt_cap"], pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )
            self.assertIn(
                "s2.single_connected_component_unwelded", provenance["finding_codes"]
            )
            # The evaluation that ASKED for this job is nameable, and it is
            # the one this very response reports.
            self.assertEqual(
                provenance["policy_evaluated_at"], payload["policy_evaluated_at"]
            )
            # The attempt it remediates is named, so the pair is traceable.
            self.assertEqual(
                provenance["remediates_attempt_id"],
                next(a.id for a in store.get(run.id).attempts
                     if a.output_artifact_id == "canon-1"),
            )

    def test_an_advance_stamps_advance_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="canon-1")

            _, submitted = self._step(store, artifacts, run.id)

            provenance = submitted[0].settings[pipeline_policy.PIPELINE_STEP_SETTINGS_KEY]
            self.assertEqual(provenance["mode"], "advance")
            self.assertEqual(provenance["policy_action"], "advance")
            self.assertEqual(provenance["stage"], "S3_SKELETON_FIT")
            self.assertNotIn("triggering_stage", provenance)

    def test_a_step_never_inherits_an_earlier_steps_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            stale = {
                "initiator": "api", "mode": "remediation",
                "stage": "S2_CANONICALIZATION", "policy_evaluated_at": "1999-01-01T00:00:00+00:00",
            }
            manifest = store.get(run.id)
            canon = next(a for a in manifest.attempts if a.output_artifact_id == "canon-1")
            canon.settings[pipeline_policy.PIPELINE_STEP_SETTINGS_KEY] = stale
            store._save(manifest)

            _, submitted = self._step(store, artifacts, run.id)

            provenance = submitted[0].settings[pipeline_policy.PIPELINE_STEP_SETTINGS_KEY]
            self.assertNotEqual(provenance["policy_evaluated_at"], "1999-01-01T00:00:00+00:00")
            self.assertEqual(provenance["attempt"], 2, "the stale block is still counted")

    # -- the cap ----------------------------------------------------------

    def _record_remediations(self, store, run_id, count, stage="S2_CANONICALIZATION"):
        manifest = store.get(run_id)
        for index in range(count):
            attempt = store.reserve_attempt(
                run_id, f"job-remediation-{index}", "canonicalize",
                manifest.source_artifact_id,
                {
                    **self.RECORDED_JOB_FIELDS,
                    pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: {
                        "initiator": "api", "mode": "remediation", "stage": stage,
                    },
                },
            )
            store.mark_attempt_running(run_id, attempt.id, f"job-remediation-{index}")
            store.finish_attempt(run_id, attempt.id, "failed", error="still not watertight")

    def test_the_cap_stops_remediating_and_submits_nothing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            self._record_remediations(
                store, run.id, pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["action"], "capped")
            self.assertEqual(payload["stage"], "S2_CANONICALIZATION")
            self.assertEqual(payload["job_kind"], "canonicalize")
            self.assertEqual(
                payload["attempts_used"], pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )
            self.assertEqual(
                payload["attempt_cap"], pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )
            self.assertIsNone(payload["job_id"])
            self.assertEqual(submitted, [], "a capped step must spend nothing")
            # The findings are handed back so a human can take it from here.
            self.assertIn(
                "s2.single_connected_component_unwelded",
                [item["code"] for item in payload["findings"]],
            )

    def test_the_last_remediation_under_the_cap_still_runs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            self._record_remediations(
                store, run.id, pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP - 1
            )

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["action"], "remediated")
            self.assertEqual(
                payload["attempts_used"], pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP
            )
            self.assertEqual(len(submitted), 1)

    def test_a_tighter_cap_is_honoured(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            self._record_remediations(store, run.id, 1)

            response, submitted = self._step(
                store, artifacts, run.id, body={"attempt_cap": 1}
            )

            payload = response.json()
            self.assertEqual(payload["action"], "capped")
            self.assertEqual(payload["attempt_cap"], 1)
            self.assertEqual(submitted, [])

    def test_a_remediation_of_another_stage_does_not_consume_this_stages_cap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            self._record_remediations(
                store, run.id, pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP,
                stage="S4_SKINNING",
            )

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.json()["action"], "remediated")
            self.assertEqual(len(submitted), 1)

    # -- stale downstream work (bead 38nlk) --------------------------------

    def _remediated_canonicalize_run(self, root, *, then_clip=False):
        """Mary's run through the rig, then S2 remediated into a new mesh.

        This is run 070772d8 on the box: the rig is real, evaluates fine,
        and describes a mesh the run has replaced. `then_clip` adds a clip
        attached to that stale rig AFTER the remediation, which is what
        leaves the run's own `current_artifact_id` pointing at stale work.
        """
        store, run, artifacts = self._mary_run(root, through="rig-1")
        remediated = json.loads(json.dumps(self.CANONICALIZE_METADATA))
        remediated["settings"] = {
            pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: {
                "initiator": "api",
                "mode": "remediation",
                "stage": "S2_CANONICALIZATION",
                "source_artifact_id": "gen-1",
            }
        }
        self._write_artifact(artifacts, "canon-2", "glb", remediated)
        attempt = store.reserve_attempt(
            run.id, "job-canon-2", "canonicalize", "gen-1", dict(self.RECORDED_JOB_FIELDS),
        )
        store.mark_attempt_running(run.id, attempt.id, "job-canon-2")
        store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id="canon-2")
        if then_clip:
            self._write_artifact(artifacts, "clip-1", "glb", self.APPLY_CLIP_METADATA)
            clip = store.reserve_attempt(
                run.id, "job-clip-1", "apply-clip", "rig-1",
                {**self.RECORDED_JOB_FIELDS, "clip": "Sword_Attack_Armature", "family": "biped"},
            )
            store.mark_attempt_running(run.id, clip.id, "job-clip-1")
            store.finish_attempt(run.id, clip.id, "completed", output_artifact_id="clip-1")
        return store, run, artifacts

    def test_a_rig_of_the_pre_remediation_mesh_is_re_run_not_stepped_over(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._remediated_canonicalize_run(root)

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            # Before this bead the same run reported advance -> S5: it
            # counted a rig of the discarded mesh as S3/S4 done.
            self.assertEqual(payload["action"], "advanced")
            self.assertEqual(payload["stage"], "S3_SKELETON_FIT")
            self.assertEqual(payload["job_kind"], "auto-rig")
            self.assertEqual(
                [entry["stage"] for entry in payload["policy"]["stale_stages"]],
                ["S3_SKELETON_FIT", "S4_SKINNING"],
            )
            self.assertEqual(
                payload["policy"]["stale_stages"][0]["obsoleted_by"],
                {"stage": "S2_CANONICALIZATION", "artifact_id": "canon-2"},
            )
            self.assertIn("stale", payload["why"])
            # One job, and it consumes the REMEDIATED mesh -- re-running
            # the rigger from the old rig would rebuild what was invalidated.
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0].source_artifact_id, "canon-2")
            provenance = submitted[0].settings[pipeline_policy.PIPELINE_STEP_SETTINGS_KEY]
            self.assertEqual(provenance["mode"], "advance")
            self.assertEqual(provenance["stage"], "S3_SKELETON_FIT")
            self.assertEqual(provenance["source_artifact_id"], "canon-2")
            # The stage's tuning is still inherited from the rig it replaces.
            self.assertEqual(submitted[0].seed, self.RECORDED_JOB_FIELDS["seed"])
            self.assertEqual(
                submitted[0].target_faces, self.RECORDED_JOB_FIELDS["target_faces"]
            )

    def test_a_stale_current_artifact_is_never_the_input_to_the_re_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._remediated_canonicalize_run(
                root, then_clip=True
            )
            # The run's own pointer is at the clip, which was attached to
            # the stale rig: following it blindly would rig a rigged clip.
            self.assertEqual(store.get(run.id).current_artifact_id, "clip-1")

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["stage"], "S3_SKELETON_FIT")
            self.assertEqual(
                [entry["stage"] for entry in payload["policy"]["stale_stages"]],
                ["S3_SKELETON_FIT", "S4_SKINNING", "S5_ANIMATION"],
            )
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0].source_artifact_id, "canon-2")

    def test_a_run_with_no_remediation_has_nothing_stale_and_is_unchanged(self):
        # The false-positive guard, at the endpoint: Mary's untouched run
        # still advances to the operator gate and still spends nothing.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["policy"]["stale_stages"], [])
            self.assertEqual(payload["action"], "needs_input")
            self.assertEqual(payload["stage"], "S6_EXPORT")
            self.assertEqual(submitted, [])

    # -- safety -----------------------------------------------------------

    def test_a_run_with_a_live_job_is_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            attempt = store.reserve_attempt(
                run.id, "job-live", "canonicalize", "gen-1", dict(self.RECORDED_JOB_FIELDS),
            )
            store.mark_attempt_running(run.id, attempt.id, "job-live")

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("never runs alongside a live job", response.json()["detail"])
            self.assertEqual(submitted, [])

    def test_a_queued_attempt_also_refuses(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)
            store.reserve_attempt(
                run.id, "job-queued", "canonicalize", "gen-1", dict(self.RECORDED_JOB_FIELDS),
            )

            response, submitted = self._step(store, artifacts, run.id)

            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(submitted, [])

    def test_one_call_submits_at_most_one_job(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root, through="canon-1")

            _, submitted = self._step(store, artifacts, run.id)

            # The endpoint returns after one submission; it does not walk the
            # rest of the run. S4/S5/S6 remain untouched.
            self.assertEqual(len(submitted), 1)

    def test_an_accepted_run_is_done_and_submits_nothing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._mary_run(root)
            manifest = store.get(run.id)
            manifest.review_status = ReviewState.ACCEPTED
            manifest.reviewed_artifact_id = "clip-1"
            store._save(manifest)

            response, submitted = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["action"], "done")
            self.assertIsNone(payload["stage"])
            self.assertIsNone(payload["job_id"])
            self.assertEqual(submitted, [])
            self.assertEqual(payload["policy"]["next_action"]["kind"], "done")

    def test_unknown_run_is_a_404(self):
        response = client.post("/api/runs/deadbeef/pipeline/step", json={})
        self.assertEqual(response.status_code, 404, response.text)

    def test_every_response_carries_the_policy_summary_that_decided_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store, run, artifacts = self._broken_canonicalize_run(root)

            response, _ = self._step(store, artifacts, run.id)

            payload = response.json()
            self.assertEqual(payload["policy"]["policy_version"], 1)
            self.assertEqual(payload["policy"]["first_violated_stage"], "S2_CANONICALIZATION")
            self.assertEqual(payload["run_id"], run.id)
            self.assertTrue(payload["why"])


if __name__ == "__main__":
    unittest.main()
