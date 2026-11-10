import unittest

from backend.app.curated_motion import curated_action_name, curated_clip_label, curated_clips


class CuratedMotionLabelTests(unittest.TestCase):
    def test_strips_trailing_armature_suffix(self):
        self.assertEqual(curated_clip_label("Sword_Attack_Armature"), "Sword Attack")

    def test_replaces_underscores_and_title_cases(self):
        self.assertEqual(curated_clip_label("Idle_FoldArms_Armature"), "Idle Foldarms")

    def test_single_word_action_name(self):
        self.assertEqual(curated_clip_label("Sprint_Armature"), "Sprint")

    def test_no_armature_suffix_is_left_untouched(self):
        # Defensive: a name that never carried the suffix should still
        # prettify sensibly rather than erroring.
        self.assertEqual(curated_clip_label("Walk"), "Walk")

    def test_curated_action_name_appends_the_blender_import_suffix(self):
        self.assertEqual(curated_action_name("Sword_Attack"), "Sword_Attack_Armature")


class CuratedMotionRealDonorTests(unittest.TestCase):
    """Exercises the parser against the actual bundled donor library.

    backend/workers/unirig_bind_worker.py's export_soma_bvh imports this same
    GLB through Blender and raises a "no action named ...; available: [...]"
    error listing every action name when given an unknown one -- that error
    (triggered manually against the live curated-donor endpoint) confirmed
    every one of these clips' exact Blender action name is
    "<raw-glTF-name>_Armature", which is what this test pins.
    """

    def test_lists_every_donor_clip_sorted_by_label(self):
        result = curated_clips()
        self.assertTrue(result["donor"].endswith(
            "vendor/mesh2motion-app/static/animations/human-base-animations.glb"
        ))
        clips = result["clips"]
        self.assertEqual(len(clips), 87)
        labels = [clip["label"] for clip in clips]
        self.assertEqual(labels, sorted(labels))

    def test_sword_attack_clip_matches_the_known_exact_action_name(self):
        result = curated_clips()
        matches = [clip for clip in result["clips"] if clip["animation"] == "Sword_Attack_Armature"]
        self.assertEqual(len(matches), 1)
        clip = matches[0]
        self.assertEqual(clip["label"], "Sword Attack")
        self.assertGreater(clip["frames"], 0)
        self.assertGreater(clip["duration_seconds"], 0)

    def test_every_clip_carries_the_armature_suffix_and_a_positive_duration(self):
        result = curated_clips()
        for clip in result["clips"]:
            self.assertTrue(clip["animation"].endswith("_Armature"), clip["animation"])
            self.assertGreater(clip["frames"], 0, clip["animation"])
            self.assertGreater(clip["duration_seconds"], 0, clip["animation"])

    def test_result_is_cached_across_calls(self):
        first = curated_clips()
        second = curated_clips()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
