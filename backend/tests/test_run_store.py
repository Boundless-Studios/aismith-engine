import json
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.run_store import ReviewState, RunManifest, RunStore, StageAttempt, select_view_synthesis_source


class RunStoreTests(unittest.TestCase):
    def test_candidate_output_can_be_retained_without_advancing_then_promoted_explicitly(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            attempt = store.reserve_attempt(run.id, "job-1", "retopo", "source-1", {})

            retained = store.finish_attempt(
                run.id,
                attempt.id,
                "completed",
                output_artifact_id="candidate-1",
                promote_output=False,
            )

            self.assertEqual(retained.current_artifact_id, "source-1")
            self.assertFalse(retained.attempts[0].promoted)

            promoted = store.promote_artifact(run.id, "candidate-1", "Human accepted visual quality")

            self.assertEqual(promoted.current_artifact_id, "candidate-1")
            self.assertTrue(promoted.attempts[0].promoted)
            self.assertEqual(promoted.attempts[-1].stage, "promote-artifact")
            self.assertEqual(promoted.attempts[-1].settings["reason"], "Human accepted visual quality")

            promoted.attempts[0].promoted = False
            store._save(promoted)
            promoted_again = store.promote_artifact(
                run.id,
                "candidate-1",
                "Human reconfirmed visual quality",
            )

            self.assertTrue(promoted_again.attempts[0].promoted)

    def test_concurrent_attempts_do_not_overwrite_each_other(self):
        class SlowRunStore(RunStore):
            def _save(self, run):
                if run.attempts:
                    time.sleep(0.05)
                super()._save(run)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = RunStore(root).create("Concurrent", "source-1")
            barrier = threading.Barrier(2)

            def start(stage: str):
                barrier.wait()
                return SlowRunStore(root).start_attempt(run.id, stage, "source-1", {})

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(start, ["generate", "refine"]))

            loaded = RunStore(root).get(run.id)
            self.assertEqual({attempt.stage for attempt in loaded.attempts}, {"generate", "refine"})

    def test_run_remains_running_until_all_attempts_finish(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Overlapping", "source-1")
            first = store.start_attempt(run.id, "generate", "source-1", {})
            second = store.start_attempt(run.id, "diagnostics", "source-1", {})

            after_first = store.finish_attempt(run.id, first.id, "completed", output_artifact_id="mesh-1")
            self.assertEqual(after_first.status, "running")

            after_second = store.finish_attempt(run.id, second.id, "failed", error="bad topology")
            self.assertEqual(after_second.status, "failed")

    def test_terminal_attempt_rejects_a_conflicting_late_callback(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Late callback", "source-1")
            attempt = store.start_attempt(run.id, "generate", "source-1", {})
            completed = store.finish_attempt(
                run.id,
                attempt.id,
                "completed",
                output_artifact_id="mesh-1",
                logs=["done"],
            )
            completed_at = completed.attempts[0].completed_at

            duplicate = store.finish_attempt(
                run.id,
                attempt.id,
                "completed",
                output_artifact_id="mesh-1",
                logs=["done"],
            )
            self.assertEqual(duplicate.attempts[0].completed_at, completed_at)

            with self.assertRaises(ValueError):
                store.finish_attempt(run.id, attempt.id, "failed", error="late failure")

            preserved = store.get(run.id)
            self.assertEqual(preserved.attempts[0].status, "completed")
            self.assertEqual(preserved.current_artifact_id, "mesh-1")

    def test_reservation_is_idempotent_and_rejects_unrelated_inputs(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")

            reserved = store.reserve_attempt(run.id, "job-1", "generate", "source-1", {"seed": 7})
            duplicate = store.reserve_attempt(run.id, "job-1", "generate", "source-1", {"seed": 7})

            self.assertEqual(duplicate.id, reserved.id)
            self.assertEqual(store.get(run.id).attempts[0].status, "queued")
            with self.assertRaises(ValueError):
                store.reserve_attempt(run.id, "job-2", "refine", "other-source", {})

    def test_reserved_attempt_transitions_to_running(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            attempt = store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})

            running = store.mark_attempt_running(run.id, attempt.id, "job-1")

            self.assertEqual(running.status, "running")

    def test_manual_attempt_rejects_an_artifact_outside_the_run(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")

            with self.assertRaises(ValueError):
                store.start_attempt(run.id, "rig", "unrelated-model", {})

    def test_reserving_another_attempt_preserves_running_aggregate_status(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            store.start_attempt(run.id, "manual", "source-1", {})

            store.reserve_attempt(run.id, "job-1", "generate", "source-1", {})

            self.assertEqual(store.get(run.id).status, "running")

    def test_run_round_trip_preserves_failed_attempt_and_retry(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary A-pose", "source-1")
            first = store.start_attempt(run.id, "generate", "source-1", {"seed": 7})
            store.finish_attempt(run.id, first.id, "failed", error="bad silhouette")
            retry = store.start_attempt(run.id, "generate", "source-1", {"seed": 8})
            store.finish_attempt(
                run.id,
                retry.id,
                "completed",
                output_artifact_id="mesh-2",
                logs=["generated"],
                warnings=["thin hair"],
            )

            loaded = RunStore(Path(directory)).get(run.id)

            self.assertEqual([item.status for item in loaded.attempts], ["failed", "completed"])
            self.assertEqual(loaded.current_artifact_id, "mesh-2")
            self.assertEqual(loaded.attempts[1].logs, ["generated"])
            self.assertEqual(loaded.attempts[1].warnings, ["thin hair"])

    def test_review_and_newest_first_listing_survive_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root)
            older = store.create("Older", "source-1")
            newer = store.create("Newer", "source-2")
            motion = store.start_attempt(older.id, "retarget-motion", "source-1", {})
            store.finish_attempt(older.id, motion.id, "completed", output_artifact_id="animated-1", promote_output=False)
            store.review(older.id, ReviewState.REJECTED, "exploded deformation", "animated-1")

            restarted = RunStore(root)

            self.assertEqual([run.id for run in restarted.list()], [newer.id, older.id])
            reviewed = restarted.get(older.id)
            self.assertEqual(reviewed.review_status, ReviewState.REJECTED)
            self.assertEqual(reviewed.review_note, "exploded deformation")

    def test_checkpoint_handoff_survives_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root)
            run = store.create("Mary", "source-1")
            attempt = store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
            store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id="rig-1")

            store.hand_back(run.id, "skeleton", "Moved the right wrist marker onto the wrist.", "rig-1")

            restarted = RunStore(root).get(run.id)
            self.assertEqual(restarted.checkpoints["skeleton"].note, "Moved the right wrist marker onto the wrist.")
            self.assertEqual(restarted.checkpoints["skeleton"].artifact_id, "rig-1")
            self.assertGreater(restarted.checkpoints["skeleton"].completed_at, 0)

    def test_checkpoint_rejects_a_non_rig_artifact(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            attempt = store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
            store.finish_attempt(run.id, attempt.id, "completed", output_artifact_id="rig-1")

            with self.assertRaisesRegex(ValueError, "completed rig artifact from this run"):
                store.hand_back(run.id, "skeleton", "Stale approval.", "source-1")

    def test_checkpoint_rejects_a_rig_belonging_to_another_run(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            mine = store.create("Mary", "source-1")
            theirs = store.create("Lyra", "source-2")
            mine_rig = store.start_attempt(mine.id, "auto-rig", "source-1", {})
            store.finish_attempt(mine.id, mine_rig.id, "completed", output_artifact_id="rig-mine")
            theirs_rig = store.start_attempt(theirs.id, "auto-rig", "source-2", {})
            store.finish_attempt(theirs.id, theirs_rig.id, "completed", output_artifact_id="rig-theirs")

            with self.assertRaisesRegex(ValueError, "completed rig artifact from this run"):
                store.hand_back(mine.id, "skeleton", "Wrong run.", "rig-theirs")
            self.assertEqual(store.get(mine.id).checkpoints, {})

    def test_checkpoint_approves_a_retained_rig_the_pointer_never_advanced_onto(self):
        """The operator's reported P1: a fresh auto-rig whose certificate lands
        `needs_review` is retained rather than promoted, so it never becomes the
        run's current artifact -- and approval must still bind to it."""
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Lyra", "source-1")
            mesh = store.start_attempt(run.id, "generate", "source-1", {})
            store.finish_attempt(run.id, mesh.id, "completed", output_artifact_id="mesh-1")
            rig = store.start_attempt(run.id, "auto-rig", "mesh-1", {})
            retained = store.finish_attempt(
                run.id, rig.id, "completed", output_artifact_id="rig-1", promote_output=False
            )
            self.assertEqual(retained.current_artifact_id, "mesh-1")

            approved = store.hand_back(run.id, "skeleton", "Skeleton reviewed.", "rig-1")

            self.assertEqual(approved.checkpoints["skeleton"].artifact_id, "rig-1")
            # the approval is what promotes the rig: the pointer follows the
            # human decision, and the manifest says why it moved
            self.assertEqual(approved.current_artifact_id, "rig-1")
            pin = approved.attempts[-1]
            self.assertEqual(pin.stage, "promote-artifact")
            self.assertEqual(pin.output_artifact_id, "rig-1")
            self.assertEqual(pin.settings["reason"], "skeleton checkpoint pinned")
            self.assertTrue(next(item for item in approved.attempts if item.id == rig.id).promoted)

    def test_checkpoint_approves_a_rig_the_pointer_has_advanced_past(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Lyra", "source-1")
            prepared = store.start_attempt(run.id, "prepare-image", "source-1", {})
            store.finish_attempt(run.id, prepared.id, "completed", output_artifact_id="prepared-1")
            rig = store.start_attempt(run.id, "auto-rig", "prepared-1", {})
            store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1", promote_output=False)
            # re-approving the prepared image moves the pointer off the rig
            moved = store.promote_artifact(run.id, "prepared-1", "operator re-approved the prepared image")
            self.assertEqual(moved.current_artifact_id, "prepared-1")

            approved = store.hand_back(run.id, "skeleton", "Skeleton reviewed.", "rig-1")

            self.assertEqual(approved.checkpoints["skeleton"].artifact_id, "rig-1")
            self.assertEqual(approved.current_artifact_id, "rig-1")

    def test_checkpoint_approves_the_current_rig_without_adding_a_pin(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            rig = store.start_attempt(run.id, "rig-adjust", "source-1", {})
            promoted = store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1")
            self.assertEqual(promoted.current_artifact_id, "rig-1")
            before = len(promoted.attempts)

            approved = store.hand_back(run.id, "skeleton", "Skeleton reviewed.", "rig-1")

            self.assertEqual(approved.checkpoints["skeleton"].artifact_id, "rig-1")
            self.assertEqual(approved.current_artifact_id, "rig-1")
            self.assertEqual(len(approved.attempts), before)

    def test_checkpoint_never_rewinds_a_run_past_its_own_animation(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            rig = store.start_attempt(run.id, "auto-rig", "source-1", {})
            store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1", promote_output=False)
            clip = store.start_attempt(run.id, "apply-clip", "rig-1", {})
            animated = store.finish_attempt(run.id, clip.id, "completed", output_artifact_id="animated-1")
            self.assertEqual(animated.current_artifact_id, "animated-1")

            approved = store.hand_back(run.id, "skeleton", "Skeleton reviewed.", "rig-1")

            self.assertEqual(approved.checkpoints["skeleton"].artifact_id, "rig-1")
            self.assertEqual(approved.current_artifact_id, "animated-1")
            self.assertEqual(approved.attempts[-1].stage, "apply-clip")

    def test_acceptance_promotes_and_is_bound_to_the_exact_animated_artifact(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            rig = store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
            store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1")
            motion = store.start_attempt(run.id, "retarget-motion", "rig-1", {})
            store.finish_attempt(
                run.id,
                motion.id,
                "completed",
                output_artifact_id="animated-1",
                promote_output=False,
            )

            store.review(run.id, ReviewState.ACCEPTED, "Looks good.", "animated-1")

            reviewed = RunStore(Path(directory)).get(run.id)
            self.assertEqual(reviewed.current_artifact_id, "animated-1")
            self.assertEqual(reviewed.reviewed_artifact_id, "animated-1")

    def test_review_rejects_a_stale_or_non_animated_artifact(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            rig = store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
            store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1")

            with self.assertRaisesRegex(ValueError, "completed animated artifact"):
                store.review(run.id, ReviewState.ACCEPTED, "Wrong artifact.", "rig-1")

    def test_rejecting_an_accepted_animation_restores_its_input_rig(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")
            rig = store.start_attempt(run.id, "target-proxy-rig", "source-1", {})
            store.finish_attempt(run.id, rig.id, "completed", output_artifact_id="rig-1")
            motion = store.start_attempt(run.id, "retarget-motion", "rig-1", {})
            store.finish_attempt(run.id, motion.id, "completed", output_artifact_id="animated-1", promote_output=False)
            store.review(run.id, ReviewState.ACCEPTED, "Premature approval.", "animated-1")

            store.review(run.id, ReviewState.REJECTED, "Knees reverse and hands deform.", "animated-1")

            reviewed = store.get(run.id)
            self.assertEqual(reviewed.review_status, ReviewState.REJECTED)
            self.assertEqual(reviewed.reviewed_artifact_id, "animated-1")
            self.assertEqual(reviewed.current_artifact_id, "rig-1")

    def test_run_defaults_to_auto_subject_kind_with_every_stage_applicable(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("Mary", "source-1")

            self.assertEqual(run.subject_kind, "auto")
            self.assertEqual(
                run.applicable_stages,
                ["prepare", "generate", "clean", "skeleton", "animate", "review"],
            )

    def test_object_run_excludes_skeleton_animate_and_review_stages(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            run = store.create("A Sword", "source-1", subject_kind="object")

            self.assertEqual(run.subject_kind, "object")
            self.assertEqual(run.applicable_stages, ["prepare", "generate", "clean"])

    def test_creature_and_biped_runs_keep_every_stage_applicable(self):
        with TemporaryDirectory() as directory:
            store = RunStore(Path(directory))
            creature = store.create("Serpent", "source-1", subject_kind="creature")
            biped = store.create("Mary", "source-2", subject_kind="biped")

            expected = ["prepare", "generate", "clean", "skeleton", "animate", "review"]
            self.assertEqual(creature.applicable_stages, expected)
            self.assertEqual(biped.applicable_stages, expected)

    def test_set_subject_kind_updates_applicable_stages_and_survives_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root)
            run = store.create("Mary", "source-1")

            updated = store.set_subject_kind(run.id, "object")

            self.assertEqual(updated.subject_kind, "object")
            self.assertEqual(updated.applicable_stages, ["prepare", "generate", "clean"])

            restarted = RunStore(root).get(run.id)
            self.assertEqual(restarted.subject_kind, "object")
            self.assertEqual(restarted.applicable_stages, ["prepare", "generate", "clean"])

    def test_pre_migration_manifest_without_subject_kind_defaults_to_auto(self):
        """Idempotent migration: manifests written before subject_kind
        existed load with the default "auto" kind and every stage applicable."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root)
            run = store.create("Mary", "source-1")
            path = root / f"{run.id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["subject_kind"]
            del payload["applicable_stages"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = RunStore(root).get(run.id)

            self.assertEqual(loaded.subject_kind, "auto")
            self.assertEqual(
                loaded.applicable_stages,
                ["prepare", "generate", "clean", "skeleton", "animate", "review"],
            )
            # Loading is read-only here (get() re-persists the same manifest
            # shape via _load, not a write), so re-saving explicitly proves
            # the migrated shape round-trips cleanly too.
            store._save(loaded)
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["subject_kind"], "auto")


class SelectViewSynthesisSourceTests(unittest.TestCase):
    """Pure-function coverage for the synthesize-views source preference
    (beads t0z17, 8c9gf): the newest approved prepare-image candidate beats
    the raw front image for as long as the run keeps it, an explicit settings
    override beats everything, and the raw front is the last resort."""

    @staticmethod
    def _run(current_artifact_id, attempts):
        return RunManifest(
            id="run-1",
            name="Mary",
            status="ready",
            source_artifact_id="front-1",
            source_artifact_kind="image",
            current_artifact_id=current_artifact_id,
            attempts=attempts,
        )

    @staticmethod
    def _prepare_attempt(output_artifact_id, *, status="completed", promoted=True):
        return StageAttempt(
            id=f"attempt-{output_artifact_id}",
            stage="prepare-image",
            status=status,
            input_artifact_id="front-1",
            settings={},
            output_artifact_id=output_artifact_id,
            promoted=promoted,
        )

    @staticmethod
    def _generate_attempt(output_artifact_id, input_artifact_id):
        return StageAttempt(
            id=f"attempt-{output_artifact_id}",
            stage="generate",
            status="completed",
            input_artifact_id=input_artifact_id,
            settings={},
            output_artifact_id=output_artifact_id,
            promoted=True,
        )

    def test_prefers_the_approved_t_pose_candidate_over_the_raw_front_image(self):
        run = self._run("tpose-1", [self._prepare_attempt("tpose-1")])

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "tpose-1")
        self.assertEqual(selected.label, "approved-t-pose-candidate")
        self.assertIn("tpose-1", selected.reason)

    def test_approved_candidate_still_wins_after_the_lineage_advanced_past_it(self):
        # The live "witch" incident (run 39247f85): the operator approved a
        # purple-jumpsuit T-pose candidate, ran Generate, and the run's
        # current artifact advanced to the resulting mesh. A later
        # synthesize-views must STILL redraw from the approved jumpsuit
        # candidate -- falling back to the raw upload (a robed witch in the
        # original art) is what produced views faithful to the wrong source.
        run = self._run(
            "mesh-1",
            [
                self._prepare_attempt("tpose-1"),
                self._generate_attempt("mesh-1", "tpose-1"),
            ],
        )

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "tpose-1")
        self.assertEqual(selected.label, "approved-t-pose-candidate")
        # The reason has to name the pointer that moved, or the choice cannot
        # be audited from artifact metadata alone.
        self.assertIn("mesh-1", selected.reason)
        self.assertIn("tpose-1", selected.reason)

    def test_the_newest_approved_candidate_wins_over_an_older_approved_one(self):
        run = self._run(
            "mesh-1",
            [
                self._prepare_attempt("tpose-1"),
                self._prepare_attempt("tpose-2"),
                self._generate_attempt("mesh-1", "tpose-2"),
            ],
        )

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "tpose-2")

    def test_explicit_settings_override_wins_even_with_an_approved_candidate(self):
        run = self._run("tpose-1", [self._prepare_attempt("tpose-1")])

        selected = select_view_synthesis_source(
            run, "front-1", {"source_artifact_id": "custom-ref-1"},
        )

        self.assertEqual(selected.artifact_id, "custom-ref-1")
        self.assertEqual(selected.label, "explicit-override")
        self.assertIn("custom-ref-1", selected.reason)

    def test_explicit_override_wins_after_the_lineage_advanced_too(self):
        run = self._run(
            "mesh-1",
            [
                self._prepare_attempt("tpose-1"),
                self._generate_attempt("mesh-1", "tpose-1"),
            ],
        )

        selected = select_view_synthesis_source(
            run, "front-1", {"source_artifact_id": "front-1"},
        )

        self.assertEqual(selected.artifact_id, "front-1")
        self.assertEqual(selected.label, "explicit-override")

    def test_falls_back_to_the_front_image_when_no_prepare_image_attempt_exists(self):
        run = self._run("front-1", [])

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "front-1")
        self.assertEqual(selected.label, "front-image")
        self.assertIn("no approved prepare-image candidate", selected.reason)

    def test_falls_back_to_the_front_image_when_the_candidate_was_never_promoted(self):
        # `promoted` on a prepare-image attempt IS the operator's approval:
        # the stage lands with promoted=False and only an explicit promote
        # flips it. An unapproved candidate is not a reference.
        run = self._run(
            "front-1",
            [self._prepare_attempt("tpose-1", promoted=False)],
        )

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "front-1")
        self.assertEqual(selected.label, "front-image")

    def test_ignores_a_still_running_prepare_image_attempt(self):
        run = self._run(
            "front-1",
            [self._prepare_attempt("tpose-1", status="running", promoted=False)],
        )

        selected = select_view_synthesis_source(run, "front-1", {})

        self.assertEqual(selected.artifact_id, "front-1")
        self.assertEqual(selected.label, "front-image")

    def test_blank_override_string_is_ignored_not_treated_as_explicit(self):
        run = self._run("tpose-1", [self._prepare_attempt("tpose-1")])

        selected = select_view_synthesis_source(
            run, "front-1", {"source_artifact_id": "   "},
        )

        self.assertEqual(selected.artifact_id, "tpose-1")
        self.assertEqual(selected.label, "approved-t-pose-candidate")


if __name__ == "__main__":
    unittest.main()
