import math
import unittest

import numpy as np

from backend.app.motion_metrics import (
    compute_motion_metrics,
    find_bone,
    foot_bone_names,
    resolve_vertical_axis,
    root_bone_name,
)


def _rotate_x(vector, degrees):
    angle = math.radians(degrees)
    x, y, z = vector
    return (
        x,
        y * math.cos(angle) - z * math.sin(angle),
        y * math.sin(angle) + z * math.cos(angle),
    )


# A four-bone stand-in for a biped: a head, a root, and two feet at slightly
# different heights so the "lowest foot is the stance foot" heuristic has a
# deterministic winner. Skeleton height (vertical extent at frame 0) is 1.8.
_REST = {
    "Head": ((0.0, 0.0, 1.6), (0.0, 0.0, 0.2)),
    "Hips": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.2)),
    "LeftFoot": ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
    "RightFoot": ((-0.1, 0.0, 0.05), (0.0, 0.1, 0.0)),
}
_HEIGHT = 1.8


def _clip(angles, *, head_offsets=None):
    """Build (frames, heads, tails) for a clip that rotates every bone.

    `angles` gives the per-frame rotation (degrees, about X) applied to each
    bone's rest direction; `head_offsets` optionally translates each bone's
    head per frame, which is how root travel and foot slide are exercised.
    """
    frames = list(range(len(angles)))
    heads: dict[str, list] = {name: [] for name in _REST}
    tails: dict[str, list] = {name: [] for name in _REST}
    for index, angle in enumerate(angles):
        for name, (head, direction) in _REST.items():
            offset = (head_offsets or {}).get(name, [(0.0, 0.0, 0.0)] * len(angles))[index]
            moved_head = tuple(a + b for a, b in zip(head, offset))
            heads[name].append(moved_head)
            tails[name].append(
                tuple(a + b for a, b in zip(moved_head, _rotate_x(direction, angle)))
            )
    return (
        frames,
        {name: np.asarray(values, dtype=np.float64) for name, values in heads.items()},
        {name: np.asarray(values, dtype=np.float64) for name, values in tails.items()},
    )


class BoneRoleResolutionTests(unittest.TestCase):
    def test_root_prefers_a_named_hips_over_a_generic_root_wrapper(self):
        self.assertEqual(root_bone_name(["root", "Hips", "Spine"]), "Hips")

    def test_root_falls_back_to_a_wrapper_when_no_hips_exists(self):
        self.assertEqual(root_bone_name(["root", "Spine"]), "root")

    def test_feet_include_ball_toe_and_leaf_joints(self):
        self.assertEqual(
            foot_bone_names(["LeftFoot", "ball_r", "ball_leaf_l", "Spine", "hand_l"]),
            ["LeftFoot", "ball_r", "ball_leaf_l"],
        )

    def test_find_bone_returns_none_when_nothing_matches(self):
        self.assertIsNone(find_bone(["Spine", "hand_l"], ("head",)))

    def test_vertical_axis_is_measured_not_assumed(self):
        # A Y-up skeleton: the head sits furthest from the feet along Y.
        self.assertEqual(
            resolve_vertical_axis(np.asarray([0.0, 1.6, 0.0]), [np.asarray([0.0, 0.0, 0.1])]),
            1,
        )

    def test_vertical_axis_defaults_to_blender_z_without_a_head_or_feet(self):
        self.assertEqual(resolve_vertical_axis(None, []), 2)


class JointMotionCoverageTests(unittest.TestCase):
    def test_a_fully_animated_clip_covers_every_bone(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertTrue(metrics["measurable"])
        self.assertEqual(metrics["joint_motion_coverage"], 1.0)
        self.assertEqual(metrics["measured_bones"], 4)
        self.assertEqual(metrics["moved_bones"], 4)

    def test_a_dead_clip_covers_no_bone(self):
        frames, heads, tails = _clip([0.0] * 10)

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["joint_motion_coverage"], 0.0)
        self.assertEqual(metrics["moved_bones"], 0)

    def test_sub_threshold_jitter_does_not_count_as_motion(self):
        frames, heads, tails = _clip([index * 0.4 for index in range(10)])

        metrics = compute_motion_metrics(frames, heads, tails)

        # 9 frames x 0.4 deg = 3.6 deg total, under the 5 deg threshold.
        self.assertEqual(metrics["joint_motion_coverage"], 0.0)

    def test_least_moved_bones_name_the_quietest_bones(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])
        # Freeze one bone's direction for the whole clip.
        tails["RightFoot"] = heads["RightFoot"] + np.asarray([0.0, 0.1, 0.0])

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["least_moved_bones"][0]["bone"], "RightFoot")
        self.assertEqual(metrics["least_moved_bones"][0]["max_direction_change_degrees"], 0.0)
        self.assertEqual(metrics["joint_motion_coverage"], 0.75)


class StaticTailTests(unittest.TestCase):
    def test_a_clip_that_moves_throughout_has_no_static_tail(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["static_tail_fraction"], 0.0)
        self.assertEqual(metrics["static_tail_frames"], 0)

    def test_a_clip_that_freezes_halfway_reports_the_dead_suffix(self):
        # This is the 2026-08-31 defect shape: the bake stops writing motion
        # partway through and every later frame repeats the last pose.
        frames, heads, tails = _clip([0.0, 11.25, 22.5, 33.75] + [45.0] * 6)

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["static_tail_frames"], 5)
        self.assertEqual(metrics["measured_transitions"], 9)
        self.assertAlmostEqual(metrics["static_tail_fraction"], 5 / 9, places=6)

    def test_a_frozen_clip_is_entirely_static_tail(self):
        frames, heads, tails = _clip([0.0] * 10)

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["static_tail_fraction"], 1.0)


class FootSlideTests(unittest.TestCase):
    def test_a_planted_stance_foot_reports_no_slide(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["foot_slide_max"], 0.0)
        self.assertEqual(metrics["foot_slide_mean"], 0.0)
        self.assertEqual(metrics["stance_switch_count"], 0)
        self.assertEqual(metrics["stance_frames_fraction"], 1.0)

    def test_a_dragging_stance_foot_reports_slide_as_a_fraction_of_height(self):
        drift = [(0.2 * index / 9.0, 0.0, 0.0) for index in range(10)]
        frames, heads, tails = _clip(
            [index * 5.0 for index in range(10)],
            head_offsets={"LeftFoot": drift},
        )

        metrics = compute_motion_metrics(frames, heads, tails)

        # LeftFoot is the lower foot all clip, so it is the stance foot for
        # every frame; it drifts 0.2 units against a 1.8-unit skeleton.
        self.assertEqual(metrics["foot_bones"], ["LeftFoot", "RightFoot"])
        self.assertAlmostEqual(metrics["foot_slide_max"], 0.2 / _HEIGHT, places=5)
        self.assertAlmostEqual(metrics["foot_slide_mean"], 0.2 / _HEIGHT, places=5)
        self.assertAlmostEqual(metrics["foot_slide_max_in_contact"], 0.2 / _HEIGHT, places=5)

    def test_a_swapping_stance_foot_is_counted_so_a_gate_can_be_calibrated(self):
        # Lift the left foot above the right for the back half of the clip.
        lift = [(0.0, 0.0, 0.0)] * 5 + [(0.0, 0.0, 0.2)] * 5
        frames, heads, tails = _clip(
            [index * 5.0 for index in range(10)],
            head_offsets={"LeftFoot": lift},
        )

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["stance_switch_count"], 1)
        self.assertEqual(
            [episode["foot"] for episode in metrics["stance_episodes"]],
            ["LeftFoot", "RightFoot"],
        )
        # One foot is on the ground throughout, so every frame is still a
        # contact frame -- the switch count is what tells a future gate this
        # clip is stepping rather than planted.
        self.assertEqual(metrics["stance_frames_fraction"], 1.0)

    def test_an_airborne_stretch_drops_out_of_the_contact_band(self):
        lift = [(0.0, 0.0, 0.0)] * 5 + [(0.0, 0.0, 0.5)] * 5
        frames, heads, tails = _clip(
            [index * 5.0 for index in range(10)],
            head_offsets={"LeftFoot": lift, "RightFoot": lift},
        )

        metrics = compute_motion_metrics(frames, heads, tails)

        # Both feet leave the ground for the back half, so only half the clip
        # can carry a foot-contact measurement at all.
        self.assertEqual(metrics["stance_frames_fraction"], 0.5)

    def test_a_footless_rig_reports_zero_slide_rather_than_failing(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])
        for name in ("LeftFoot", "RightFoot"):
            heads.pop(name)
            tails.pop(name)

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["foot_bones"], [])
        self.assertEqual(metrics["foot_slide_max"], 0.0)
        self.assertEqual(metrics["stance_frames_fraction"], 0.0)


class RootTravelTests(unittest.TestCase):
    def test_a_there_and_back_root_has_path_length_but_no_net_displacement(self):
        travel = [(0.9 * min(index, 9 - index) / 4.5, 0.0, 0.0) for index in range(10)]
        frames, heads, tails = _clip([0.0] * 10, head_offsets={"Hips": travel})

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["root_bone"], "Hips")
        self.assertAlmostEqual(metrics["root_travel_net"], 0.0, places=6)
        self.assertAlmostEqual(metrics["root_travel_path"], 1.6 / _HEIGHT, places=5)

    def test_a_travelling_root_reports_net_displacement(self):
        travel = [(0.9 * index / 9.0, 0.0, 0.0) for index in range(10)]
        frames, heads, tails = _clip([0.0] * 10, head_offsets={"Hips": travel})

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertAlmostEqual(metrics["root_travel_net"], 0.9 / _HEIGHT, places=5)
        self.assertAlmostEqual(metrics["root_travel_path"], 0.9 / _HEIGHT, places=5)


class UnmeasurableClipTests(unittest.TestCase):
    def test_a_single_frame_clip_is_reported_unmeasurable_not_zero(self):
        frames, heads, tails = _clip([0.0])

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertFalse(metrics["measurable"])
        self.assertEqual(metrics["sampled_frames"], 1)
        self.assertNotIn("joint_motion_coverage", metrics)

    def test_a_boneless_armature_is_reported_unmeasurable(self):
        metrics = compute_motion_metrics([0, 1], {}, {})

        self.assertFalse(metrics["measurable"])
        self.assertEqual(metrics["reason"], "no bones sampled")

    def test_the_synthetic_flag_and_source_action_are_carried_through(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])

        metrics = compute_motion_metrics(
            frames, heads, tails, synthetic=True, source_action="UniRigRestSway",
        )

        self.assertTrue(metrics["synthetic"])
        self.assertEqual(metrics["source_action"], "UniRigRestSway")


class SamplingIndependenceTests(unittest.TestCase):
    def test_angular_velocity_is_normalized_per_single_frame(self):
        # The same motion sampled every 3rd frame must not read as 3x faster
        # (which would hide a static tail behind a coarse sample step).
        angles = [0.0, 1.2, 2.4, 3.6] + [3.6] * 6
        dense_frames = list(range(10))
        _, heads, tails = _clip(angles)
        sparse = compute_motion_metrics(
            [frame * 3 for frame in dense_frames], heads, tails,
        )
        dense = compute_motion_metrics(dense_frames, heads, tails)

        # Sampled every 3rd frame the 1.2 deg step is 0.4 deg/frame -- below
        # the stillness threshold -- so the whole clip reads static.
        self.assertEqual(sparse["static_tail_fraction"], 1.0)
        # Sampled every frame it is 1.2 deg/frame, so only the real tail does.
        self.assertAlmostEqual(dense["static_tail_fraction"], 6 / 9, places=6)


if __name__ == "__main__":
    unittest.main()


class ContactEpisodeCountTests(unittest.TestCase):
    """A contact-filtered slide of 0.0 means two different things; the
    episode count is what separates them, and `pipeline_policy`'s S5 adapter
    refuses to derive `foot_contact_error` without it."""

    def test_a_planted_foot_reports_a_measured_contact_episode(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])

        metrics = compute_motion_metrics(frames, heads, tails)

        # Zero slide AND a qualifying episode: the zero is a measurement.
        self.assertEqual(metrics["foot_slide_max_in_contact"], 0.0)
        self.assertGreaterEqual(metrics["contact_stance_episodes"], 1)

    def test_a_fully_airborne_clip_reports_no_contact_episode_at_all(self):
        # Both feet leave the contact band after the first frame, so no run
        # of two consecutive in-contact frames on one foot ever forms.
        lift = [(0.0, 0.0, 0.0)] + [(0.0, 0.0, 0.5 + 0.05 * index) for index in range(9)]
        frames, heads, tails = _clip(
            [index * 5.0 for index in range(10)],
            head_offsets={"LeftFoot": lift, "RightFoot": lift},
        )

        metrics = compute_motion_metrics(frames, heads, tails)

        # Same 0.0 as the planted case above -- and here it is "no evidence".
        self.assertEqual(metrics["foot_slide_max_in_contact"], 0.0)
        self.assertEqual(metrics["contact_stance_episodes"], 0)

    def test_a_footless_rig_reports_no_contact_episodes(self):
        frames, heads, tails = _clip([index * 5.0 for index in range(10)])
        for name in ("LeftFoot", "RightFoot"):
            heads.pop(name)
            tails.pop(name)

        metrics = compute_motion_metrics(frames, heads, tails)

        self.assertEqual(metrics["contact_stance_episodes"], 0)
