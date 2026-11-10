"""Tests for workstream N5's clip_store module: the family clip-store
manifest lookup and the pure s5.exact_bone_name_match bone-coverage helper
that gates direct clip application (backend/app/main.py's `apply-clip` job
kind, backend/workers/blender_worker.py's `apply_clip` op).
"""
import unittest

from backend.app.clip_store import (
    ClipManifestEntry,
    classify_missing_bones,
    find_clip,
    load_manifest,
    missing_animated_bones,
)


class FindClipTests(unittest.TestCase):
    def test_finds_known_biped_clips_by_exact_name(self):
        for clip_name in ("Sword_Attack_Armature", "Roll_Armature", "Walk_Armature"):
            with self.subTest(clip_name=clip_name):
                entry = find_clip("biped", clip_name)
                self.assertIsInstance(entry, ClipManifestEntry)
                self.assertEqual(entry.clip_name, clip_name)
                self.assertGreater(entry.frame_count, 0)
                self.assertGreaterEqual(entry.duration_seconds, 0.0)

    def test_unknown_clip_name_returns_none(self):
        self.assertIsNone(find_clip("biped", "Not_A_Real_Clip_Armature"))

    def test_unknown_family_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            find_clip("quadruped", "Sword_Attack_Armature")

    def test_lookup_is_exact_not_case_insensitive_or_prefix(self):
        self.assertIsNone(find_clip("biped", "sword_attack_armature"))
        self.assertIsNone(find_clip("biped", "Sword_Attack"))

    def test_manifest_declares_the_87_clip_store(self):
        manifest = load_manifest("biped")
        self.assertEqual(manifest["family"], "biped")
        self.assertEqual(manifest["clip_count"], len(manifest["clips"]))


class MissingAnimatedBonesTests(unittest.TestCase):
    def test_empty_when_every_animated_bone_is_on_the_rig(self):
        self.assertEqual(
            missing_animated_bones(
                {"hips", "spine_01", "hand_l"},
                {"hips", "spine_01", "hand_l", "hand_r"},
            ),
            [],
        )

    def test_lists_animated_bones_absent_from_the_rig_sorted(self):
        self.assertEqual(
            missing_animated_bones(
                ["hips", "tail_01", "wing_l"],
                ["hips"],
            ),
            ["tail_01", "wing_l"],
        )

    def test_extra_rig_bones_the_clip_does_not_animate_are_not_a_mismatch(self):
        # Direct-attach only requires every ANIMATED bone to exist on the
        # rig -- a rig bone the clip never touches (e.g. an optional finger
        # segment) is not a coverage gap.
        self.assertEqual(
            missing_animated_bones({"hips"}, {"hips", "finger_01_l"}),
            [],
        )

    def test_no_animated_bones_is_never_a_mismatch(self):
        self.assertEqual(missing_animated_bones([], {"hips"}), [])
        self.assertEqual(missing_animated_bones([], []), [])


class ClassifyMissingBonesTests(unittest.TestCase):
    """Live-verification finding (N5): every clip in the biped store was
    baked on the full 66-bone canonical template, so it always carries
    channels for optional bones (fingers, toes, head_tip) a required-tier-
    only rig legitimately doesn't have (S3: an optional bone's absence is
    not a conformance error). classify_missing_bones must let those through
    and only hard-fail a genuinely missing REQUIRED bone."""

    def test_optional_template_bones_are_skipped_not_hard_failed(self):
        hard_fail, skipped = classify_missing_bones(["toes_l", "index_1_l"], "biped")
        self.assertEqual(hard_fail, [])
        self.assertEqual(skipped, ["index_1_l", "toes_l"])

    def test_missing_required_template_bone_is_a_hard_fail(self):
        hard_fail, skipped = classify_missing_bones(["pelvis"], "biped")
        self.assertEqual(hard_fail, ["pelvis"])
        self.assertEqual(skipped, [])

    def test_a_name_that_is_not_even_a_template_bone_is_a_hard_fail(self):
        hard_fail, skipped = classify_missing_bones(["not_a_real_bone"], "biped")
        self.assertEqual(hard_fail, ["not_a_real_bone"])
        self.assertEqual(skipped, [])

    def test_mixed_required_and_optional_gaps_partition_correctly(self):
        hard_fail, skipped = classify_missing_bones(["pelvis", "toes_l", "thumb_1_r"], "biped")
        self.assertEqual(hard_fail, ["pelvis"])
        self.assertEqual(skipped, ["thumb_1_r", "toes_l"])

    def test_empty_input_is_empty_output(self):
        self.assertEqual(classify_missing_bones([], "biped"), ([], []))


if __name__ == "__main__":
    unittest.main()
