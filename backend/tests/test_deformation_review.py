import unittest

from backend.app.deformation_review import (
    all_animation_frames,
    deformation_review_frame_specs,
    resolve_review_focus_bone,
    select_review_frame,
)


class DeformationReviewTests(unittest.TestCase):
    def test_review_set_includes_a_mid_strike_weapon_hand_closeup(self):
        specs = deformation_review_frame_specs()

        self.assertIn(
            ("weapon-hand-50", 0.5, False, "right-hand"),
            specs,
        )

    def test_semantic_right_hand_focus_resolves_supported_rig_names(self):
        self.assertEqual(
            resolve_review_focus_bone(["Hips", "mixamorig:RightHand"], "right-hand"),
            "mixamorig:RightHand",
        )

    def test_literal_review_focus_is_preserved(self):
        self.assertEqual(
            resolve_review_focus_bone(["custom_focus"], "custom_focus"),
            "custom_focus",
        )

    def test_selects_repeatable_frames_across_an_animation(self):
        self.assertEqual(select_review_frame(1, 49, 0.0), 1)
        self.assertEqual(select_review_frame(1, 49, 0.5), 25)
        self.assertEqual(select_review_frame(1, 49, 1.0), 49)

    def test_clamps_review_ratio_to_animation_bounds(self):
        self.assertEqual(select_review_frame(10, 20, -1), 10)
        self.assertEqual(select_review_frame(10, 20, 2), 20)

    def test_rejects_an_invalid_frame_range(self):
        with self.assertRaisesRegex(ValueError, "frame_end"):
            select_review_frame(20, 10, 0.5)

    def test_reviews_every_integer_frame_in_the_action_range(self):
        self.assertEqual(list(all_animation_frames(3, 6)), [3, 4, 5, 6])

    def test_rejects_unreasonable_action_ranges(self):
        with self.assertRaisesRegex(ValueError, "limited to 600"):
            all_animation_frames(1, 10_000_000)


if __name__ == "__main__":
    unittest.main()
