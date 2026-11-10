"""Tests for facing-aware handedness in backend.app.skeleton_naming.

Synthetic 20-joint humanoids (pelvis/spine/neck/head, two arm chains, two leg
chains each with a toe) exercise the bug this module fixes: the "left" and
"right" name suffixes must follow the character's actual facing direction,
not a fixed +X/-X convention.
"""
import itertools
import unittest
from collections.abc import Callable

import numpy as np

from backend.app.skeleton_naming import (
    ChainCandidateScore,
    VERTICAL_AXIS,
    build_template_conformant_skeleton,
    distal_refinement_json,
    limb_radius_profile,
    measure_distal_span,
    refine_distal_anchor,
    refine_distal_anchors,
    optional_chain_layout,
    optional_chain_root,
    optional_chain_scale,
    template_anchor_bone,
    chain_candidate_extent_ratio,
    densify_chain,
    detect_forward_from_feet,
    detect_profile_from_names,
    detect_rig_profile,
    left_from_forward,
    medial_axis_chain,
    mesh_is_elongated,
    nearest_bone_blend_weights,
    remove_joints_reparent,
    resolve_position_overrides,
    score_chain_candidate,
    select_chain_continuation_child,
    semantic_joint_names,
    stitch_coincident_vertex_weights,
    template_side_conformance_findings,
)
from backend.app.skeleton_template import conformance_report, load_template


# Joint layout (indices), shared by every test in this file. Side A sits at
# +X, side B sits at -X; which one becomes "l" vs "r" depends on facing.
#   0 pelvis        6 upper_arm_A   12 thigh_A   16 thigh_B
#   1 spine_1       7 lower_arm_A   13 calf_A    17 calf_B
#   2 spine_2       8 hand_A        14 foot_A    18 foot_B
#   3 spine_3       9 upper_arm_B   15 toe_A     19 toe_B
#   4 neck         10 lower_arm_B
#   5 head         11 hand_B
_PARENTS_WITH_TOES = [
    -1, 0, 1, 2, 3, 4,
    3, 6, 7, 3, 9, 10,
    0, 12, 13, 14, 0, 16, 17, 18,
]
# Same skeleton with both toe joints (15, 19 above) dropped: foot_B's chain
# is renumbered down by one slot (16->15, 17->16, 18->17).
_PARENTS_NO_TOES = [
    -1, 0, 1, 2, 3, 4,
    3, 6, 7, 3, 9, 10,
    0, 12, 13, 0, 15, 16,
]


def _joints_with_toes(foot_forward_sign: int) -> np.ndarray:
    """foot_forward_sign > 0 => character faces +Y; < 0 => faces -Y."""
    y = foot_forward_sign
    return np.asarray([
        [0.0, 0.0, 1.00],   # 0 pelvis
        [0.0, 0.0, 1.15],   # 1 spine_1
        [0.0, 0.0, 1.30],   # 2 spine_2
        [0.0, 0.0, 1.45],   # 3 spine_3
        [0.0, 0.0, 1.55],   # 4 neck
        [0.0, 0.0, 1.65],   # 5 head
        [0.5, 0.0, 1.45],   # 6 upper_arm_A (+X)
        [0.9, 0.0, 1.45],   # 7 lower_arm_A
        [1.3, 0.0, 1.45],   # 8 hand_A
        [-0.5, 0.0, 1.45],  # 9 upper_arm_B (-X)
        [-0.9, 0.0, 1.45],  # 10 lower_arm_B
        [-1.3, 0.0, 1.45],  # 11 hand_B
        [0.15, 0.0, 0.90],       # 12 thigh_A (+X)
        [0.15, 0.0, 0.45],       # 13 calf_A
        [0.15, 0.10 * y, 0.05],  # 14 foot_A
        [0.15, 0.25 * y, 0.02],  # 15 toe_A
        [-0.15, 0.0, 0.90],       # 16 thigh_B (-X)
        [-0.15, 0.0, 0.45],       # 17 calf_B
        [-0.15, 0.10 * y, 0.05],  # 18 foot_B
        [-0.15, 0.25 * y, 0.02],  # 19 toe_B
    ], dtype=np.float64)


def _joints_no_toes(foot_forward_sign: int) -> np.ndarray:
    y = foot_forward_sign
    return np.asarray([
        [0.0, 0.0, 1.00],   # 0 pelvis
        [0.0, 0.0, 1.15],   # 1 spine_1
        [0.0, 0.0, 1.30],   # 2 spine_2
        [0.0, 0.0, 1.45],   # 3 spine_3
        [0.0, 0.0, 1.55],   # 4 neck
        [0.0, 0.0, 1.65],   # 5 head
        [0.5, 0.0, 1.45],   # 6 upper_arm_A (+X)
        [0.9, 0.0, 1.45],   # 7 lower_arm_A
        [1.3, 0.0, 1.45],   # 8 hand_A
        [-0.5, 0.0, 1.45],  # 9 upper_arm_B (-X)
        [-0.9, 0.0, 1.45],  # 10 lower_arm_B
        [-1.3, 0.0, 1.45],  # 11 hand_B
        [0.15, 0.0, 0.90],       # 12 thigh_A (+X)
        [0.15, 0.0, 0.45],       # 13 calf_A
        [0.15, 0.10 * y, 0.05],  # 14 foot_A (no toe child)
        [-0.15, 0.0, 0.90],       # 15 thigh_B (-X)
        [-0.15, 0.0, 0.45],       # 16 calf_B
        [-0.15, 0.10 * y, 0.05],  # 17 foot_B (no toe child)
    ], dtype=np.float64)


def _below_ankle_vertex_cloud(foot_forward_sign: int) -> np.ndarray:
    """A small vertex cloud standing in for a mesh, biased toward the
    character's forward direction, all below the feet (z < 0.05)."""
    y = foot_forward_sign
    return np.asarray([
        [0.05, 0.50 * y, 0.00],
        [-0.05, 0.50 * y, 0.00],
        [0.03, 0.60 * y, 0.01],
        [-0.03, 0.60 * y, 0.01],
    ], dtype=np.float64)


class SemanticJointNamesFacingTests(unittest.TestCase):
    def _assert_sides_correct(self, names: list[str], forward_sign: int, thigh_b_index: int) -> None:
        # forward_sign < 0 (facing -Y) => left is +X => side A (+X) is "l".
        # forward_sign > 0 (facing +Y) => left is -X => side B (-X) is "l".
        if forward_sign < 0:
            left_upper_arm, right_upper_arm = 6, 9
            left_thigh, right_thigh = 12, thigh_b_index
        else:
            left_upper_arm, right_upper_arm = 9, 6
            left_thigh, right_thigh = thigh_b_index, 12
        self.assertEqual(names[left_upper_arm], "upper_arm_l")
        self.assertEqual(names[right_upper_arm], "upper_arm_r")
        self.assertEqual(names[left_thigh], "thigh_l")
        self.assertEqual(names[right_thigh], "thigh_r")

    def test_toe_direction_facing_negative_y(self):
        joints = _joints_with_toes(-1)
        result = semantic_joint_names(joints, _PARENTS_WITH_TOES)
        self.assertEqual(result.facing_source, "foot-toe-direction")
        self.assertIsNone(result.warning)
        np.testing.assert_allclose(result.forward, [0.0, -1.0, 0.0], atol=1e-6)
        self._assert_sides_correct(result.names, forward_sign=-1, thigh_b_index=16)

    def test_toe_direction_facing_positive_y(self):
        joints = _joints_with_toes(1)
        result = semantic_joint_names(joints, _PARENTS_WITH_TOES)
        self.assertEqual(result.facing_source, "foot-toe-direction")
        np.testing.assert_allclose(result.forward, [0.0, 1.0, 0.0], atol=1e-6)
        self._assert_sides_correct(result.names, forward_sign=1, thigh_b_index=16)

    def test_vertex_fallback_facing_negative_y(self):
        joints = _joints_no_toes(-1)
        mesh_vertices = _below_ankle_vertex_cloud(-1)
        result = semantic_joint_names(joints, _PARENTS_NO_TOES, mesh_vertices)
        self.assertEqual(result.facing_source, "foot-region-vertices")
        self.assertIsNone(result.warning)
        forward = result.forward / np.linalg.norm(result.forward)
        self.assertLess(forward[1], -0.9)  # dominantly -Y
        self._assert_sides_correct(result.names, forward_sign=-1, thigh_b_index=15)

    def test_vertex_fallback_facing_positive_y(self):
        joints = _joints_no_toes(1)
        mesh_vertices = _below_ankle_vertex_cloud(1)
        result = semantic_joint_names(joints, _PARENTS_NO_TOES, mesh_vertices)
        self.assertEqual(result.facing_source, "foot-region-vertices")
        forward = result.forward / np.linalg.norm(result.forward)
        self.assertGreater(forward[1], 0.9)  # dominantly +Y
        self._assert_sides_correct(result.names, forward_sign=1, thigh_b_index=15)

    def test_assumes_negative_y_and_warns_when_undetectable(self):
        joints = _joints_no_toes(-1)
        result = semantic_joint_names(joints, _PARENTS_NO_TOES, mesh_vertices=None)
        self.assertEqual(result.facing_source, "assumed")
        self.assertIsNotNone(result.warning)
        np.testing.assert_allclose(result.forward, [0.0, -1.0, 0.0])
        # The assumed default happens to match this fixture's real facing,
        # so sides still land correctly.
        self._assert_sides_correct(result.names, forward_sign=-1, thigh_b_index=15)

    def test_arm_and_leg_chain_roles_are_fully_named(self):
        joints = _joints_with_toes(-1)
        result = semantic_joint_names(joints, _PARENTS_WITH_TOES)
        names = result.names
        self.assertEqual(names[0], "pelvis")
        self.assertEqual(names[1], "spine_1")
        self.assertEqual(names[2], "spine_2")
        self.assertEqual(names[3], "spine_3")
        self.assertEqual(names[4], "neck")
        self.assertEqual(names[5], "head")
        for base in ("upper_arm", "lower_arm", "hand", "thigh", "calf", "foot"):
            sides = {n[len(base) + 1:] for n in names if n in (f"{base}_l", f"{base}_r")}
            self.assertEqual(sides, {"l", "r"}, f"missing a side for {base}")
        # the toe (foot's child) gets an extra-role name, not "foot_l"/"foot_r"
        self.assertTrue(any(n.startswith("foot_extra_") for n in names))


class DetectForwardFromFeetConfidenceTests(unittest.TestCase):
    """Regression coverage for the near-cancellation bug: two feet whose
    toe directions disagree average to a near-zero vector that naive
    normalization turns into a confident, essentially arbitrary direction."""

    # A generic torso/leg skeleton (root pelvis, two thighs descending to
    # feet) shared by these tests -- only the toe positions vary.
    _JOINTS_BASE = np.asarray([
        [0.0, 0.0, 1.00],   # 0 pelvis
        [0.15, 0.0, 0.90],  # 1 thigh_A
        [0.15, 0.0, 0.45],  # 2 calf_A
        [0.15, 0.0, 0.05],  # 3 foot_A
        [-0.15, 0.0, 0.90],  # 4 thigh_B
        [-0.15, 0.0, 0.45],  # 5 calf_B
        [-0.15, 0.0, 0.05],  # 6 foot_B
    ], dtype=np.float64)

    def test_agreeing_feet_are_confident(self):
        joints = np.vstack([self._JOINTS_BASE, [
            [0.15, -0.15, 0.02],   # 7 toe_A: -Y, agrees with toe_B
            [-0.15, -0.14, 0.02],  # 8 toe_B: -Y
        ]])
        result = detect_forward_from_feet(joints, [3, 6], [[7], [8]])
        self.assertEqual(result.source, "foot-toe-direction")
        self.assertIsNone(result.warning)

    def test_nearly_opposed_feet_are_not_confident(self):
        # ~170 degrees apart: nowhere near cancelling to exactly zero, but
        # too disagreeing/small to trust as a confident measurement.
        joints = np.vstack([self._JOINTS_BASE, [
            [0.15, -0.15, 0.02],  # 7 toe_A: -Y
            [-0.15, 0.14, 0.04],  # 8 toe_B: nearly +Y
        ]])
        result = detect_forward_from_feet(joints, [3, 6], [[7], [8]])
        self.assertEqual(result.source, "foot-toe-direction-low-confidence")
        self.assertIsNotNone(result.warning)

    def test_exactly_opposed_equal_feet_fall_back_to_assumed(self):
        # Perfectly cancelling: the averaged vector is exactly zero and
        # cannot be normalized into any direction at all.
        joints = np.vstack([self._JOINTS_BASE, [
            [0.15, -0.15, 0.02],  # 7 toe_A: -Y
            [-0.15, 0.15, 0.02],  # 8 toe_B: exactly +Y, same magnitude
        ]])
        result = detect_forward_from_feet(joints, [3, 6], [[7], [8]])
        self.assertEqual(result.source, "assumed")
        self.assertIsNotNone(result.warning)

    def test_one_foot_only_is_still_confident(self):
        # A single real toe measurement has nothing to disagree with.
        joints = np.vstack([self._JOINTS_BASE, [
            [0.15, -0.15, 0.02],  # 7 toe_A: -Y
        ]])
        result = detect_forward_from_feet(joints, [3, 6], [[7], []])
        self.assertEqual(result.source, "foot-toe-direction")
        np.testing.assert_allclose(
            result.forward / np.linalg.norm(result.forward), [0.0, -1.0, 0.0], atol=1e-6,
        )

    def test_weak_toe_signal_beats_the_hardcoded_default_when_kept(self):
        # No mesh_vertices fallback available, so a "kept" weak result (if
        # any) must be the low-confidence toe reading, never silently
        # promoted to "foot-toe-direction".
        joints = np.vstack([self._JOINTS_BASE, [
            [0.15, -0.15, 0.02],
            [-0.15, 0.14, 0.04],
        ]])
        result = detect_forward_from_feet(joints, [3, 6], [[7], [8]], mesh_vertices=None)
        self.assertEqual(result.source, "foot-toe-direction-low-confidence")
        self.assertIsNotNone(result.warning)


class LeftFromForwardDegenerateTests(unittest.TestCase):
    def test_vertical_forward_does_not_raise_and_returns_unit_vector(self):
        forward = np.array([0.0, 0.0, 5.0])  # parallel to up (vertical_axis=2)
        left = left_from_forward(forward, vertical_axis=2)
        np.testing.assert_allclose(left, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(left)), 1.0)

    def test_negative_vertical_forward_does_not_raise(self):
        forward = np.array([0.0, 0.0, -3.0])
        left = left_from_forward(forward, vertical_axis=2)
        np.testing.assert_allclose(left, [1.0, 0.0, 0.0])

    def test_vertical_axis_is_x_falls_back_to_plus_y(self):
        forward = np.array([3.0, 0.0, 0.0])  # parallel to up when vertical_axis=0
        left = left_from_forward(forward, vertical_axis=0)
        np.testing.assert_allclose(left, [0.0, 1.0, 0.0])


class ResolvePositionOverridesTests(unittest.TestCase):
    def test_uses_stored_map_when_naming_changed_since_the_override_was_authored(self):
        # Before the facing fix, joint 7 was displayed (and saved) as
        # "upper_arm_l"; after the fix, the same joint is correctly renamed
        # "upper_arm_r" and joint 9 is now called "upper_arm_l" instead. The
        # override must still land on joint 7.
        fresh_names = [f"bone_{i}" for i in range(10)]
        fresh_names[7] = "upper_arm_r"
        fresh_names[9] = "upper_arm_l"
        stored_names = {"upper_arm_l": 7, "upper_arm_r": 9}
        overrides = {"upper_arm_l": [1.0, 2.0, 3.0]}

        result = resolve_position_overrides(overrides, fresh_names, stored_names)

        self.assertFalse(result.used_fallback)
        self.assertEqual(set(result.resolved), {7})
        np.testing.assert_allclose(result.resolved[7], [1.0, 2.0, 3.0])
        self.assertEqual(result.unresolved, [])

    def test_falls_back_to_fresh_names_without_a_stored_map(self):
        fresh_names = ["pelvis", "upper_arm_l", "upper_arm_r"]
        overrides = {"upper_arm_l": [4.0, 5.0, 6.0]}

        result = resolve_position_overrides(overrides, fresh_names, None)

        self.assertTrue(result.used_fallback)
        self.assertEqual(set(result.resolved), {1})
        np.testing.assert_allclose(result.resolved[1], [4.0, 5.0, 6.0])

    def test_reports_unresolved_override_names(self):
        fresh_names = ["pelvis", "upper_arm_l"]
        stored_names = {"pelvis": 0}
        overrides = {"pelvis": [0.0, 0.0, 0.0], "made_up_bone": [1.0, 1.0, 1.0]}

        result = resolve_position_overrides(overrides, fresh_names, stored_names)

        self.assertEqual(result.unresolved, ["made_up_bone"])
        self.assertEqual(set(result.resolved), {0})

    def test_ignores_stored_index_outside_current_joint_count(self):
        # A stored map can reference an index that no longer exists if the
        # skeleton itself changed shape between binds; must not crash or
        # silently write out of range.
        fresh_names = ["pelvis", "upper_arm_l"]
        stored_names = {"upper_arm_l": 99}
        overrides = {"upper_arm_l": [1.0, 1.0, 1.0]}

        result = resolve_position_overrides(overrides, fresh_names, stored_names)

        self.assertEqual(result.resolved, {})
        self.assertEqual(result.unresolved, ["upper_arm_l"])


class SelectChainContinuationChildTests(unittest.TestCase):
    def test_pelvis_prefers_named_spine_child_over_thighs(self):
        head = np.array([0.0, 0.0, 1.0])
        child_names = ["thigh_l", "thigh_r", "spine_1"]
        child_heads = [
            np.array([0.15, 0.0, 0.9]),
            np.array([-0.15, 0.0, 0.9]),
            np.array([0.0, 0.0, 1.15]),
        ]
        index = select_chain_continuation_child("pelvis", head, None, child_names, child_heads)
        self.assertEqual(child_names[index], "spine_1")

    def test_last_spine_bone_prefers_neck_over_upper_arms(self):
        head = np.array([0.0, 0.0, 1.45])
        parent_head = np.array([0.0, 0.0, 1.30])
        child_names = ["upper_arm_l", "upper_arm_r", "neck"]
        child_heads = [
            np.array([0.5, 0.0, 1.45]),
            np.array([-0.5, 0.0, 1.45]),
            np.array([0.0, 0.0, 1.55]),
        ]
        index = select_chain_continuation_child("spine_3", head, parent_head, child_names, child_heads)
        self.assertEqual(child_names[index], "neck")

    def test_falls_back_to_direction_closest_to_parent_bone_direction(self):
        head = np.array([0.0, 0.0, 1.0])
        parent_head = np.array([0.0, 0.0, 0.8])  # climbing +Z
        child_names = ["bone_a", "bone_b"]
        child_heads = [
            np.array([0.5, 0.0, 1.0]),  # sideways, does not continue
            np.array([0.0, 0.0, 1.2]),  # continues +Z
        ]
        index = select_chain_continuation_child("bone_root", head, parent_head, child_names, child_heads)
        self.assertEqual(child_names[index], "bone_b")

    def test_root_without_parent_or_name_match_prefers_upward_child(self):
        head = np.array([0.0, 0.0, 1.0])
        child_names = ["bone_a", "bone_b"]
        child_heads = [
            np.array([0.15, 0.0, 0.9]),  # descending
            np.array([0.0, 0.0, 1.2]),   # climbing
        ]
        index = select_chain_continuation_child("unnamed_root", head, None, child_names, child_heads)
        self.assertEqual(child_names[index], "bone_b")

    def test_single_child_is_always_chosen_without_evaluating_scores(self):
        head = np.array([0.0, 0.0, 1.0])
        child_names = ["only_child"]
        child_heads = [np.array([0.0, 0.0, 1.2])]
        index = select_chain_continuation_child("anything", head, None, child_names, child_heads)
        self.assertEqual(index, 0)


def _linear_chain(count: int, spacing: float = 1.0, root_at_start: bool = True):
    """`count` joints on a straight line along +X. `root_at_start` controls
    which end is the armature root (parent=None) -- the chain naming's head
    heuristic falls back to whichever end is nearest the root."""
    joints = np.stack(
        [np.arange(count, dtype=np.float64) * spacing, np.zeros(count), np.zeros(count)],
        axis=1,
    )
    if root_at_start:
        parents: list[int | None] = [None, *range(count - 1)]
    else:
        parents = [*range(1, count), None]
    return joints, parents


def _curved_chain(count: int, spacing: float = 1.0, curvature: float = 0.03, root_at_start: bool = True):
    """`count` joints on a gentle parabolic arc in the X-Z plane -- curved,
    but still elongated overall (principal-axis ratio comfortably above the
    chain threshold for these parameters)."""
    x = np.arange(count, dtype=np.float64) * spacing
    z = curvature * (x ** 2)
    joints = np.stack([x, np.zeros(count), z], axis=1)
    if root_at_start:
        parents: list[int | None] = [None, *range(count - 1)]
    else:
        parents = [*range(1, count), None]
    return joints, parents


def _tapered_vertex_cloud(
    thick_end: np.ndarray,
    thin_end: np.ndarray,
    thick_radius: float = 0.3,
    thin_radius: float = 0.02,
) -> np.ndarray:
    """A small vertex cloud straddling two ends of a chain, one end
    (`thick_end`) surrounded by a wider ring than the other -- standing in
    for a mesh where one tip (the head) is visibly thicker than the other
    (the tail)."""
    offsets = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    thick_cluster = thick_end + offsets * thick_radius
    thin_cluster = thin_end + offsets * thin_radius
    return np.vstack([thick_cluster, thin_cluster])


class ChainProfileTests(unittest.TestCase):
    """Coverage for the non-humanoid "chain" body plan (serpents, tails,
    tentacles -- anything with no limb branches): profile detection in
    `detect_rig_profile`/`detect_profile_from_names`, and chain naming via
    `semantic_joint_names`."""

    def test_humanoid_fixture_still_classifies_as_humanoid(self):
        # Regression guard: adding chain detection must not disturb the
        # existing humanoid path for an ordinary humanoid skeleton.
        joints = _joints_with_toes(-1)
        result = semantic_joint_names(joints, _PARENTS_WITH_TOES)
        self.assertEqual(result.profile, "humanoid")
        self.assertEqual(result.names[0], "pelvis")

    def test_straight_twelve_joint_chain_defaults_head_to_root_end(self):
        joints, parents = _linear_chain(12, root_at_start=True)
        self.assertEqual(detect_rig_profile(joints, parents), "chain")

        result = semantic_joint_names(joints, parents)
        self.assertEqual(result.profile, "chain")
        # No mesh evidence: head falls back to the end nearest UniRig's own
        # root, i.e. the root end itself (index 0).
        self.assertEqual(result.names[0], "head")
        self.assertEqual(result.names[11], "spine_00")
        self.assertEqual(result.names[1], "spine_10")
        for index in range(1, 11):
            self.assertRegex(result.names[index], r"^spine_\d{2}$")

    def test_straight_chain_head_flips_to_far_end_with_tapered_mesh(self):
        joints, parents = _linear_chain(12, root_at_start=True)
        # Root sits at index 0, but the mesh is visibly thicker at index 11
        # -- the head heuristic should override the root-proximity fallback.
        mesh_vertices = _tapered_vertex_cloud(thick_end=joints[11], thin_end=joints[0])

        result = semantic_joint_names(joints, parents, mesh_vertices)

        self.assertEqual(result.profile, "chain")
        self.assertEqual(result.names[11], "head")
        self.assertEqual(result.names[0], "spine_00")
        self.assertEqual(result.names[10], "spine_10")

    def test_curved_chain_root_at_start_defaults_head_to_root_end(self):
        joints, parents = _curved_chain(12, root_at_start=True)
        self.assertEqual(detect_rig_profile(joints, parents), "chain")

        result = semantic_joint_names(joints, parents)

        self.assertEqual(result.profile, "chain")
        self.assertEqual(result.names[0], "head")
        self.assertEqual(result.names[11], "spine_00")

    def test_curved_chain_root_at_far_end_defaults_head_to_that_end(self):
        # Same curved geometry, root relabeled to the opposite physical end
        # -- the "both orientations" case: head must follow the root, not a
        # fixed index.
        joints, parents = _curved_chain(12, root_at_start=False)
        self.assertEqual(detect_rig_profile(joints, parents), "chain")

        result = semantic_joint_names(joints, parents)

        self.assertEqual(result.profile, "chain")
        self.assertEqual(result.names[11], "head")
        self.assertEqual(result.names[0], "spine_00")

    def test_curved_chain_head_flips_via_tapered_mesh(self):
        joints, parents = _curved_chain(12, root_at_start=True)
        mesh_vertices = _tapered_vertex_cloud(thick_end=joints[11], thin_end=joints[0])

        result = semantic_joint_names(joints, parents, mesh_vertices)

        self.assertEqual(result.profile, "chain")
        self.assertEqual(result.names[11], "head")
        self.assertEqual(result.names[0], "spine_00")

    def test_chain_with_two_joint_spur_is_pruned_and_named_as_spur(self):
        joints, parents = _linear_chain(12, root_at_start=True)
        # A 2-joint spur (a fin/whisker) branching off joint 5.
        joints = np.vstack([joints, [
            [5.0, 0.3, 0.0],
            [5.0, 0.6, 0.0],
        ]])
        parents = [*parents, 5, 12]

        self.assertEqual(detect_rig_profile(joints, parents), "chain")
        result = semantic_joint_names(joints, parents)

        self.assertEqual(result.profile, "chain")
        # The main 12-joint path is unaffected by the spur.
        self.assertEqual(result.names[0], "head")
        self.assertEqual(result.names[11], "spine_00")
        parent_name = result.names[5]
        self.assertTrue(result.names[12].startswith(f"spur_{parent_name}_"))
        self.assertTrue(result.names[13].startswith(f"spur_{parent_name}_"))
        self.assertNotEqual(result.names[12], result.names[13])

    def test_root_hanging_off_a_path_elbow_via_parent_is_named_as_a_spur(self):
        # Regression for a real bug found on a live 26-joint UniRig
        # prediction: the skeleton's own root can sit off the diameter path
        # entirely, attached to a main-path "elbow" node (one with two
        # on-path children, both directions of the chain) only via that
        # node's PARENT link, not a rooted child. The old spur detector
        # only checked a path node's rooted children for stray branches, so
        # this one-joint stub went unclassified forever, exported as a
        # leftover un-renamed "bone_0" -- which then tripped the
        # certificate's chain.branching finding on an otherwise-clean chain.
        joints = np.asarray([
            [0.0, 1.0, 0.0],   # 0 root -- hangs off joint 1 via the parent edge
            [0.0, 0.0, 0.0],   # 1 elbow: two children form the main path
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [-4.0, 0.0, 0.0],
        ], dtype=np.float64)
        parents = [None, 0, 1, 2, 3, 4, 1, 6, 7, 8]

        self.assertEqual(detect_rig_profile(joints, parents), "chain")
        result = semantic_joint_names(joints, parents)

        self.assertEqual(result.profile, "chain")
        self.assertTrue(all(not name.startswith("bone_") for name in result.names))
        elbow_name = result.names[1]
        self.assertTrue(result.names[0].startswith(f"spur_{elbow_name}_"))

    def test_branch_at_or_above_spur_threshold_is_not_a_chain(self):
        # A genuine Y-shaped (really a 3-way) skeleton: a root with *three*
        # branches of 3+ joints each. Any two of them concatenate into the
        # tree's longest path, but the third is real limb-like branching --
        # too long to prune as a spur -- so this is not a chain. (A root
        # with exactly *two* long branches is not a counterexample: that's
        # just a chain that happens to bend at its root, same as the real
        # UniRig serpent this module was built against.)
        joints = np.asarray([
            [0.0, 0.0, 0.0],    # 0 root
            [1.0, 0.3, 0.0],    # 1 branch A
            [2.0, 0.6, 0.0],    # 2
            [3.0, 0.9, 0.0],    # 3
            [1.0, -0.3, 0.0],   # 4 branch B
            [2.0, -0.6, 0.0],   # 5
            [3.0, -0.9, 0.0],   # 6
            [1.0, 0.0, 0.3],    # 7 branch C
            [2.0, 0.0, 0.6],    # 8
            [3.0, 0.0, 0.9],    # 9
        ], dtype=np.float64)
        parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8]

        self.assertEqual(detect_rig_profile(joints, parents), "humanoid")

    def test_five_joint_chain_still_classifies_as_topological_chain(self):
        # Below the certificate's >= 8 spine-bone READY threshold (that's a
        # quality_policy concern -- see test_quality_policy.py), but still a
        # valid chain topology at the naming layer.
        joints, parents = _linear_chain(5, root_at_start=True)
        self.assertEqual(detect_rig_profile(joints, parents), "chain")


class DetectProfileFromNamesTests(unittest.TestCase):
    def test_chain_names_are_detected(self):
        names = ["spine_00", "spine_01", "spine_02", "head", "spur_spine_01_0"]
        self.assertEqual(detect_profile_from_names(names), "chain")

    def test_humanoid_spine_naming_is_not_mistaken_for_a_chain(self):
        # Humanoid spine bones are spine_1/spine_2/spine_3 -- never
        # zero-padded -- so this must not collide with the chain pattern.
        names = ["pelvis", "spine_1", "spine_2", "spine_3", "head",
                 "upper_arm_l", "lower_arm_l", "hand_l", "thigh_r", "calf_r", "foot_r"]
        self.assertEqual(detect_profile_from_names(names), "humanoid")

    def test_limb_role_names_disqualify_chain_even_with_chain_style_spine(self):
        names = ["spine_00", "spine_01", "head", "upper_arm_l"]
        self.assertEqual(detect_profile_from_names(names), "humanoid")

    def test_missing_head_disqualifies_chain(self):
        names = ["spine_00", "spine_01", "spine_02"]
        self.assertEqual(detect_profile_from_names(names), "humanoid")


class ScoreChainCandidateTests(unittest.TestCase):
    def test_clean_straight_chain_has_no_spurs_and_full_length(self):
        joints, parents = _linear_chain(10, root_at_start=True)
        score = score_chain_candidate(joints, parents)
        self.assertEqual(score, ChainCandidateScore(main_chain_length=10, spur_count=0, disqualified=False))

    def test_two_joint_spur_is_counted_but_not_disqualifying(self):
        joints, parents = _linear_chain(10, root_at_start=True)
        joints = np.vstack([joints, [[5.0, 0.3, 0.0], [5.0, 0.6, 0.0]]])
        parents = [*parents, 5, 10]
        score = score_chain_candidate(joints, parents)
        self.assertFalse(score.disqualified)
        self.assertEqual(score.main_chain_length, 10)
        self.assertEqual(score.spur_count, 1)

    def test_real_branch_disqualifies_the_candidate(self):
        # Same 3-branch structure as the topology test above.
        joints = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.6, 0.0], [3.0, 0.9, 0.0],
            [1.0, -0.3, 0.0], [2.0, -0.6, 0.0], [3.0, -0.9, 0.0],
            [1.0, 0.0, 0.3], [2.0, 0.0, 0.6], [3.0, 0.0, 0.9],
        ], dtype=np.float64)
        parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8]
        score = score_chain_candidate(joints, parents)
        self.assertTrue(score.disqualified)

    def test_scoring_prefers_longer_unbranched_chain_over_more_spurs(self):
        # A realistic seed-comparison: candidate A is longer with one spur,
        # candidate B is shorter with none -- A should sort first under
        # (not disqualified, main_chain_length, -spur_count).
        joints_a, parents_a = _linear_chain(10, root_at_start=True)
        joints_a = np.vstack([joints_a, [[5.0, 0.3, 0.0], [5.0, 0.6, 0.0]]])
        parents_a = [*parents_a, 5, 10]
        score_a = score_chain_candidate(joints_a, parents_a)

        joints_b, parents_b = _linear_chain(7, root_at_start=True)
        score_b = score_chain_candidate(joints_b, parents_b)

        candidates = sorted(
            [("A", score_a), ("B", score_b)],
            key=lambda item: (not item[1].disqualified, item[1].main_chain_length, -item[1].spur_count),
            reverse=True,
        )
        self.assertEqual(candidates[0][0], "A")


def _tube_mesh_and_centerline(radius_at: "Callable[[float], float]", sample_count: int = 80):
    """A tube mesh (a ring of vertices at each of `sample_count` samples)
    around the gentle curve used by `_curved_chain`-style tests, plus its
    true centerline, for exercising densify_chain's medial-axis snap."""
    def curve(t: float) -> np.ndarray:
        x = 10.0 * t
        z = 2.0 * t * t
        return np.array([x, 0.0, z])

    ts = np.linspace(0.0, 1.0, sample_count)
    centerline = np.asarray([curve(t) for t in ts])
    vertices = []
    up = np.array([0.0, 1.0, 0.0])
    for index, t in enumerate(ts):
        eps = 1e-4
        tangent = curve(min(t + eps, 1.0)) - curve(max(t - eps, 0.0))
        tangent = tangent / np.linalg.norm(tangent)
        side = np.cross(tangent, up)
        side = side / np.linalg.norm(side)
        perp = np.cross(tangent, side)
        ring_radius = radius_at(t)
        for angle in np.linspace(0.0, 2 * np.pi, 6, endpoint=False):
            offset = ring_radius * (np.cos(angle) * side + np.sin(angle) * perp)
            vertices.append(centerline[index] + offset)
    return np.asarray(vertices), centerline, curve


class DensifyChainTests(unittest.TestCase):
    def test_five_predicted_joints_densify_to_twelve_within_the_mesh(self):
        mesh_vertices, centerline, curve = _tube_mesh_and_centerline(
            radius_at=lambda t: 0.03 + 0.06 * t,  # thin tail, thick head
        )
        ts_sparse = [0.0, 0.25, 0.5, 0.75, 1.0]
        joints = np.asarray([curve(t) for t in ts_sparse])
        # Inject lateral noise at the middle predicted joint, simulating a
        # sparse/noisy UniRig prediction that isn't exactly on the medial
        # axis -- the slab snap should pull it back.
        joints[2] = joints[2] + np.array([0.0, 0.4, 0.0])
        parents = [None, 0, 1, 2, 3]

        result = densify_chain(joints, parents, target_joint_count=12, mesh_vertices=mesh_vertices)

        self.assertEqual(result.joints.shape, (12, 3))
        self.assertEqual(result.names, [f"spine_{i:02d}" for i in range(11)] + ["head"])
        self.assertEqual(result.parents, [None, *range(11)])

        # Head lands at the thick end (t=1), not the root end (t=0) --
        # the thickness heuristic overriding the root-proximity fallback.
        np.testing.assert_allclose(result.joints[-1], curve(1.0), atol=0.25)
        np.testing.assert_allclose(result.joints[0], curve(0.0), atol=0.25)

        # Monotonic arc length: each joint is strictly further along the
        # tail->head direction than the last.
        overall = result.joints[-1] - result.joints[0]
        overall_unit = overall / np.linalg.norm(overall)
        projections = (result.joints - result.joints[0]) @ overall_unit
        self.assertTrue(np.all(np.diff(projections) > 0))

        # Every resampled joint is within (a small margin of) the mesh.
        margin = 0.05
        mesh_min = mesh_vertices.min(axis=0) - margin
        mesh_max = mesh_vertices.max(axis=0) + margin
        self.assertTrue(np.all(result.joints >= mesh_min))
        self.assertTrue(np.all(result.joints <= mesh_max))

        # The snap recovered the true medial line even at the deliberately
        # noisy middle joint -- nowhere close to the injected 0.4 offset.
        distances_to_centerline = [
            float(np.min(np.linalg.norm(centerline - point, axis=1)))
            for point in result.joints
        ]
        self.assertLess(max(distances_to_centerline), 0.15)

    def test_spurs_are_dropped_from_the_densified_chain(self):
        mesh_vertices, _, curve = _tube_mesh_and_centerline(radius_at=lambda t: 0.05)
        joints = np.asarray([curve(t) for t in (0.0, 0.5, 1.0)])
        joints = np.vstack([joints, [[5.0, 0.3, 1.0]]])  # a 1-joint spur off the middle
        parents = [None, 0, 1, 2]

        result = densify_chain(joints, parents, target_joint_count=8, mesh_vertices=mesh_vertices)

        self.assertEqual(len(result.joints), 8)
        self.assertEqual(result.names[-1], "head")

    def test_rejects_a_disqualifying_branch(self):
        joints = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.6, 0.0], [3.0, 0.9, 0.0],
            [1.0, -0.3, 0.0], [2.0, -0.6, 0.0], [3.0, -0.9, 0.0],
            [1.0, 0.0, 0.3], [2.0, 0.0, 0.6], [3.0, 0.0, 0.9],
        ], dtype=np.float64)
        parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8]

        with self.assertRaises(ValueError):
            densify_chain(joints, parents, target_joint_count=12)

    def test_works_without_mesh_evidence(self):
        joints, parents = _linear_chain(6, root_at_start=True)
        result = densify_chain(joints, parents, target_joint_count=10)
        self.assertEqual(result.joints.shape, (10, 3))
        self.assertEqual(result.names[-1], "head")


def _s_curve_tube(sample_count: int = 150):
    """An S-shaped tube: gently sinusoidal in Y as it sweeps along X, thin
    at the tail (x=0) and thick at the head (x=12)."""
    def curve(t: float) -> np.ndarray:
        x = 12.0 * t
        y = 2.0 * np.sin(t * np.pi * 1.5)
        return np.array([x, y, 0.0])

    ts = np.linspace(0.0, 1.0, sample_count)
    centerline = np.asarray([curve(t) for t in ts])
    up = np.array([0.0, 0.0, 1.0])
    vertices = []
    for index, t in enumerate(ts):
        eps = 1e-4
        tangent = curve(min(t + eps, 1.0)) - curve(max(t - eps, 0.0))
        tangent = tangent / np.linalg.norm(tangent)
        side = np.cross(tangent, up)
        side = side / np.linalg.norm(side)
        perp = np.cross(tangent, side)
        radius = 0.2 + 0.5 * t  # thin tail, thick head
        for angle in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
            offset = radius * (np.cos(angle) * side + np.sin(angle) * perp)
            vertices.append(centerline[index] + offset)
    return np.asarray(vertices), centerline, curve


class MedialAxisChainTests(unittest.TestCase):
    def test_s_curve_tube_produces_twelve_joints_inside_the_mesh(self):
        vertices, centerline, curve = _s_curve_tube()

        result = medial_axis_chain(vertices, target_joint_count=12)

        self.assertEqual(result.joints.shape, (12, 3))
        self.assertEqual(result.names, [f"spine_{i:02d}" for i in range(11)] + ["head"])
        self.assertEqual(result.parents, [None, *range(11)])

        # Monotonic arc length: every consecutive segment has positive
        # length (the walk never doubles back on itself).
        segment_lengths = np.linalg.norm(np.diff(result.joints, axis=0), axis=1)
        self.assertTrue(np.all(segment_lengths > 0))

        # Every joint lands close to the mesh (within a small margin).
        margin = 0.5
        mesh_min = vertices.min(axis=0) - margin
        mesh_max = vertices.max(axis=0) + margin
        self.assertTrue(np.all(result.joints >= mesh_min))
        self.assertTrue(np.all(result.joints <= mesh_max))

        # Head lands at the thick end (x near 12), tail at the thin end
        # (x near 0) -- the thickness comparison, not just march order.
        self.assertGreater(result.joints[-1][0], 10.0)
        self.assertLess(result.joints[0][0], 2.0)

        # The march tracked the true curve, not a straight PCA chord --
        # every joint should sit close to the actual centerline.
        distances_to_centerline = [
            float(np.min(np.linalg.norm(centerline - point, axis=1)))
            for point in result.joints
        ]
        self.assertLess(max(distances_to_centerline), 0.3)

    def test_requires_at_least_two_target_joints(self):
        vertices, _, _ = _s_curve_tube()
        with self.assertRaises(ValueError):
            medial_axis_chain(vertices, target_joint_count=1)

    def test_requires_a_non_trivial_vertex_cloud(self):
        with self.assertRaises(ValueError):
            medial_axis_chain(np.zeros((2, 3)), target_joint_count=8)

    def test_accepts_a_lazy_vertex_callable(self):
        vertices, _, _ = _s_curve_tube()
        calls = []

        def lazy():
            calls.append(1)
            return vertices

        result = medial_axis_chain(lazy, target_joint_count=8)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.joints.shape, (8, 3))


class ChainCandidateExtentRatioTests(unittest.TestCase):
    def test_full_span_candidate_has_ratio_near_one(self):
        vertices, _, curve = _s_curve_tube()
        joints = np.asarray([curve(t) for t in np.linspace(0, 1, 10)])
        parents = [None, *range(9)]

        ratio = chain_candidate_extent_ratio(joints, parents, vertices)

        self.assertGreater(ratio, 0.8)

    def test_short_midbody_cluster_has_a_small_ratio(self):
        # Mirrors the real "5-bone star at mid-body" failure mode: a short
        # cluster of joints spanning a fraction of a long mesh.
        vertices, _, curve = _s_curve_tube()
        joints = np.asarray([curve(t) for t in (0.45, 0.48, 0.5, 0.52, 0.55)])
        parents = [None, 0, 1, 2, 3]

        ratio = chain_candidate_extent_ratio(joints, parents, vertices)

        self.assertLess(ratio, 0.2)

    def test_mesh_is_elongated_matches_the_internal_check(self):
        vertices, _, _ = _s_curve_tube()
        self.assertTrue(mesh_is_elongated(vertices))
        cube = np.random.default_rng(0).uniform(-1, 1, size=(200, 3))
        self.assertFalse(mesh_is_elongated(cube))


# Real UniRig-predicted skeleton for "Mary", a T-posed humanoid (arm span
# routinely exceeds height in a T-pose -- normal geometry, not evidence of
# a chain body plan). Regression fixture for a real incident: an auto-rig
# job on this exact skeleton produced profile "chain" via a mesh-elongation
# fallback that fired despite the skeleton already being correctly
# classified "humanoid", discarding a good rig
# (studio-data/rig-inputs/job-fad6c2f6ed104cd48148f5b759815ecc/predict_skeleton.npz).
_MARY_PARENTS = [
    None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 9, 10, 11, 9, 13, 14, 9, 16, 17, 9, 19, 20, 9, 22, 23, 3, 25, 26, 27, 28, 29,
    30, 28, 32, 33, 28, 35, 36, 28, 38, 39, 28, 41, 42, 0, 44, 45, 46, 0, 48, 49, 50,
]
_MARY_JOINTS = [
    [0.00390625, 0.01171875, 0.16796875], [0.00390625, 0.02734375, 0.26953125],
    [0.00390625, 0.03515625, 0.38671875], [0.00390625, 0.05078125, 0.51953125],
    [0.00390625, 0.07421875, 0.66796875], [0.00390625, 0.06640625, 0.74609375],
    [0.07421875, 0.07421875, 0.64453125], [0.21484375, 0.07421875, 0.60546875],
    [0.48046875, 0.07421875, 0.57421875], [0.75390625, 0.05859375, 0.57421875],
    [0.78515625, 0.01953125, 0.55859375], [0.81640625, -0.00390625, 0.55078125],
    [0.84765625, -0.03515625, 0.53515625], [0.87109375, 0.02734375, 0.56640625],
    [0.91796875, 0.01953125, 0.56640625], [0.95703125, 0.01953125, 0.56640625],
    [0.87890625, 0.05078125, 0.56640625], [0.92578125, 0.05078125, 0.56640625],
    [0.96484375, 0.05078125, 0.56640625], [0.87890625, 0.08203125, 0.55859375],
    [0.91796875, 0.08203125, 0.55859375], [0.94921875, 0.08984375, 0.55859375],
    [0.86328125, 0.10546875, 0.55078125], [0.89453125, 0.11328125, 0.55078125],
    [0.92578125, 0.12109375, 0.54296875], [-0.06640625, 0.07421875, 0.64453125],
    [-0.20703125, 0.07421875, 0.60546875], [-0.47265625, 0.07421875, 0.57421875],
    [-0.74609375, 0.05859375, 0.57421875], [-0.77734375, 0.01953125, 0.55859375],
    [-0.80859375, -0.00390625, 0.55078125], [-0.83984375, -0.03515625, 0.53515625],
    [-0.86328125, 0.02734375, 0.56640625], [-0.91015625, 0.01953125, 0.56640625],
    [-0.94921875, 0.01953125, 0.56640625], [-0.87109375, 0.05078125, 0.56640625],
    [-0.91796875, 0.05078125, 0.56640625], [-0.95703125, 0.05078125, 0.56640625],
    [-0.87109375, 0.08203125, 0.55859375], [-0.91015625, 0.08203125, 0.55859375],
    [-0.94140625, 0.08984375, 0.55859375], [-0.85546875, 0.10546875, 0.55078125],
    [-0.88671875, 0.11328125, 0.55078125], [-0.91796875, 0.12109375, 0.54296875],
    [0.09765625, 0.01171875, 0.11328125], [0.12109375, 0.01953125, -0.39453125],
    [0.14453125, 0.04296875, -0.85546875], [0.15234375, -0.12890625, -0.98046875],
    [-0.08984375, 0.01171875, 0.11328125], [-0.11328125, 0.01953125, -0.39453125],
    [-0.13671875, 0.03515625, -0.85546875], [-0.14453125, -0.12890625, -0.98046875],
]


class MaryHumanoidRegressionTests(unittest.TestCase):
    """Regression coverage for the real incident above: a T-posed humanoid
    must classify "humanoid" (never "chain") regardless of mesh/joint
    elongation, both with automatic detection and with an explicit hint."""

    def test_mary_classifies_humanoid_without_a_hint(self):
        joints = np.asarray(_MARY_JOINTS, dtype=np.float64)
        # Arm span (X: -0.95..0.96) is comparable to or exceeds height
        # (Z: -0.98..0.75) -- exactly the T-pose elongation that triggered
        # the incident -- yet the topology (real, multi-joint arm/leg
        # branches) must still win.
        self.assertGreater(float(np.ptp(joints[:, 0])), float(np.ptp(joints[:, 2])) * 0.9)
        self.assertEqual(detect_rig_profile(joints, _MARY_PARENTS), "humanoid")

        result = semantic_joint_names(joints, _MARY_PARENTS)

        self.assertEqual(result.profile, "humanoid")
        self.assertEqual(result.profile_source, "detected")
        self.assertEqual(result.names[0], "pelvis")
        for role in ("upper_arm_l", "upper_arm_r", "thigh_l", "thigh_r", "head"):
            self.assertIn(role, result.names)

    def test_mary_honors_an_explicit_humanoid_hint(self):
        joints = np.asarray(_MARY_JOINTS, dtype=np.float64)
        result = semantic_joint_names(joints, _MARY_PARENTS, profile_hint="humanoid")
        self.assertEqual(result.profile, "humanoid")
        self.assertEqual(result.profile_source, "hint")
        self.assertEqual(result.names[0], "pelvis")

    def test_mary_rejects_a_mismatched_chain_hint(self):
        # A "chain" hint on a real limbed skeleton must not force a chain
        # naming onto it -- the topology check overrules the hint and
        # falls back to (correct) automatic humanoid detection.
        joints = np.asarray(_MARY_JOINTS, dtype=np.float64)
        result = semantic_joint_names(joints, _MARY_PARENTS, profile_hint="chain")
        self.assertEqual(result.profile, "humanoid")
        self.assertEqual(result.profile_source, "detected")


class RemoveJointsReparentTests(unittest.TestCase):
    """A 6-joint linear chain (`spine_00`..`spine_04`, `head`), matching the
    shape of an exported chain rig's bone list."""

    def _chain(self):
        joints = np.asarray(
            [[0.0, 0.0, float(i)] for i in range(6)], dtype=np.float64
        )
        parents = [None, 0, 1, 2, 3, 4]
        names = ["spine_00", "spine_01", "spine_02", "spine_03", "spine_04", "head"]
        return joints, parents, names

    def test_removing_a_middle_joint_splices_its_neighbours_together(self):
        joints, parents, names = self._chain()

        result = remove_joints_reparent(joints, parents, names, None, ["spine_02"])

        self.assertEqual(result.removed, ["spine_02"])
        self.assertEqual(result.refused, [])
        self.assertEqual(result.names, ["spine_00", "spine_01", "spine_03", "spine_04", "head"])
        # spine_03 (now index 2) used to be parented to spine_02; it must
        # now point directly at spine_01 (index 1), the removed joint's
        # own parent.
        self.assertEqual(result.parents, [None, 0, 1, 2, 3])
        np.testing.assert_allclose(
            result.joints,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 5.0]],
        )

    def test_removing_a_joint_with_two_children_reparents_both(self):
        # A branch point: joint 1 has two children (2 and 3).
        joints = np.asarray([[0.0, 0.0, float(i)] for i in range(4)], dtype=np.float64)
        parents = [None, 0, 1, 1]
        names = ["root_bone", "hub", "left_tip", "right_tip"]

        result = remove_joints_reparent(joints, parents, names, None, ["hub"])

        self.assertEqual(result.removed, ["hub"])
        self.assertEqual(result.names, ["root_bone", "left_tip", "right_tip"])
        # Both former children of "hub" (now indices 1 and 2) reparent onto
        # "root_bone" (index 0), hub's own parent.
        self.assertEqual(result.parents, [None, 0, 0])

    def test_refuses_to_remove_the_root(self):
        joints, parents, names = self._chain()

        result = remove_joints_reparent(joints, parents, names, None, ["spine_00"])

        self.assertEqual(result.removed, [])
        self.assertEqual(result.refused, ["spine_00"])
        self.assertEqual(result.names, names)
        self.assertEqual(result.parents, parents)

    def test_refuses_an_unknown_joint_name(self):
        joints, parents, names = self._chain()

        result = remove_joints_reparent(joints, parents, names, None, ["not_a_real_bone"])

        self.assertEqual(result.removed, [])
        self.assertEqual(result.refused, ["not_a_real_bone"])
        self.assertEqual(result.names, names)

    def test_refuses_protected_structural_names_case_insensitively(self):
        joints = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64)
        parents = [None, 0, 1]
        names = ["Pelvis", "spine", "head"]

        result = remove_joints_reparent(joints, parents, names, None, ["Pelvis", "spine"])

        # "Pelvis" is also the root here, so it is refused on both counts;
        # "spine" (exact, case-insensitive) is refused as a protected role
        # even though it is not the root.
        self.assertEqual(result.removed, [])
        self.assertEqual(sorted(result.refused), ["Pelvis", "spine"])
        self.assertEqual(result.names, names)

    def test_a_chain_name_that_merely_contains_spine_is_still_removable(self):
        # "spine_02" != "spine" -- only an exact (case-insensitive) match
        # to the protected name is refused, matching a chain rig's own
        # spine_NN naming convention.
        joints, parents, names = self._chain()

        result = remove_joints_reparent(joints, parents, names, None, ["spine_02"])

        self.assertEqual(result.removed, ["spine_02"])
        self.assertEqual(result.refused, [])

    def test_multiple_removals_keep_indices_and_parents_consistent(self):
        joints, parents, names = self._chain()

        result = remove_joints_reparent(joints, parents, names, None, ["spine_01", "spine_03"])

        self.assertEqual(sorted(result.removed), ["spine_01", "spine_03"])
        self.assertEqual(len(result.joints), len(result.names))
        self.assertEqual(len(result.parents), len(result.names))
        for index, parent_index in enumerate(result.parents):
            self.assertTrue(parent_index is None or 0 <= parent_index < len(result.names))
            self.assertTrue(parent_index is None or parent_index < index)

    def test_removal_carries_tails_along(self):
        joints, parents, names = self._chain()
        tails = joints + np.array([0.0, 0.0, 0.5])

        result = remove_joints_reparent(joints, parents, names, tails, ["spine_02"])

        self.assertEqual(len(result.tails), len(result.names))
        np.testing.assert_allclose(result.tails[2], tails[3])  # spine_03's tail, unshifted


class NearestBoneBlendWeightsTests(unittest.TestCase):
    """`nearest_bone_blend_weights` replaces Blender's ARMATURE_ENVELOPE
    fallback: it must never let a bone's influence reach past its own
    segment's endpoints the way an envelope-distance capsule radius can.
    """

    def _three_collinear_bones(self):
        # bone0 [0,0,0]->[1,0,0], bone1 [1,0,0]->[2,0,0], bone2 [2,0,0]->[3,0,0]
        heads = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        tails = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        return heads, tails

    def test_vertex_beside_middle_of_a_bone_gets_that_bone_at_full_weight(self):
        heads, tails = self._three_collinear_bones()
        vertices = np.array([[0.5, 0.3, 0.0]])  # beside bone0's midpoint

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        self.assertEqual(bone_indices[0, 0], 0)
        self.assertEqual(bone_indices[0, 1], -1)
        self.assertAlmostEqual(weights[0, 0], 1.0)
        self.assertAlmostEqual(weights[0, 1], 0.0)

    def test_vertex_at_a_shared_joint_blends_roughly_evenly(self):
        heads, tails = self._three_collinear_bones()
        vertices = np.array([[1.0, 0.0, 0.0]])  # bone0's tail == bone1's head

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        # Default max_influences is now 4: bone2 (the third, distal bone)
        # is far outside blend_ratio and stays an unused -1 slot.
        assigned = sorted(i for i in bone_indices[0].tolist() if i != -1)
        self.assertEqual(assigned, [0, 1])
        self.assertAlmostEqual(weights[0, 0], 0.5, places=3)
        self.assertAlmostEqual(weights[0, 1], 0.5, places=3)

    def test_vertex_near_a_distal_bone_far_outside_blend_ratio_gets_only_that_bone(self):
        heads, tails = self._three_collinear_bones()
        vertices = np.array([[2.5, 0.01, 0.0]])  # tight against bone2's midpoint

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        self.assertEqual(bone_indices[0, 0], 2)
        self.assertEqual(bone_indices[0, 1], -1)
        self.assertAlmostEqual(weights[0, 0], 1.0)

    def test_thigh_mid_segment_vertex_is_not_pulled_in_by_the_calf(self):
        # Real-rig proportions from the Mary humanoid retarget regression:
        # thigh 0.72 long, calf 0.66 long, sharing the knee joint at the
        # origin. A vertex 0.05 off-axis beside the thigh's *midpoint*
        # (z=0.36) is the exact region an envelope-distance capsule on the
        # calf (envelope_distance ~= 0.66*0.4 = 0.264, reaching ~0.37 above
        # the knee per the diagnosed bug) would incorrectly claim.
        thigh_head = np.array([0.0, 0.0, 0.72])
        thigh_tail = np.array([0.0, 0.0, 0.0])
        calf_head = np.array([0.0, 0.0, 0.0])
        calf_tail = np.array([0.0, 0.0, -0.66])
        heads = np.array([thigh_head, calf_head])
        tails = np.array([thigh_tail, calf_tail])
        vertices = np.array([[0.05, 0.0, 0.36]])

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        self.assertEqual(bone_indices[0, 0], 0)  # thigh
        self.assertEqual(bone_indices[0, 1], -1)  # calf excluded
        self.assertAlmostEqual(weights[0, 0], 1.0)

    def test_weight_rows_always_sum_to_one(self):
        heads, tails = self._three_collinear_bones()
        rng = np.random.default_rng(7)
        vertices = rng.uniform(-1.0, 4.0, size=(50, 3))

        _, weights = nearest_bone_blend_weights(vertices, heads, tails)

        np.testing.assert_allclose(weights.sum(axis=1), np.ones(50), atol=1e-9)

    def test_degenerate_bone_is_treated_as_a_point_at_its_head(self):
        heads = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tails = np.array([[1.0, 0.0, 0.0], [1e-12, 0.0, 0.0]])  # bone1 is degenerate
        vertices = np.array([[0.0, 0.5, 0.0]])  # closest to the shared head

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        # Both bones' nearest point collapses to the origin here, so this
        # should blend like the shared-joint case rather than error out.
        # Only 2 bones exist, so the other two (of the default 4) slots
        # stay unused -1.
        assigned = sorted(i for i in bone_indices[0].tolist() if i != -1)
        self.assertEqual(assigned, [0, 1])
        self.assertAlmostEqual(weights[0].sum(), 1.0)

    def test_three_way_junction_blends_all_three_bones_near_the_joint(self):
        # A Y-junction like the neck (head / neck / chest meet) or a
        # shoulder (chest / upper-arm meet a third short bone): three
        # bones sharing one endpoint. A vertex sitting right at the
        # junction must pick up nonzero weight from all three, not just
        # the nearest two -- a strict nearest-two blend leaves a hard
        # boundary one bone short of the joint, which is exactly the
        # clipping/stretching symptom this widening fixes.
        heads = np.array([
            [0.0, 0.0, 0.0],   # bone0: junction -> +x
            [0.0, 0.0, 0.0],   # bone1: junction -> +y
            [0.0, 0.0, 0.0],   # bone2: junction -> +z
        ])
        tails = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        vertices = np.array([[0.0, 0.0, 0.0]])  # exactly at the shared joint

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        assigned = {
            int(index): float(weight)
            for index, weight in zip(bone_indices[0], weights[0])
            if index != -1
        }
        self.assertEqual(set(assigned), {0, 1, 2})
        for weight in assigned.values():
            self.assertGreater(weight, 0.0)
        self.assertAlmostEqual(sum(assigned.values()), 1.0)

    def test_influences_are_capped_even_when_more_bones_are_within_blend_ratio(self):
        # Five bones all meeting at one point, all equidistant from a
        # vertex sitting on the joint -- more candidates than the default
        # max_influences=4, so exactly 4 (not 5) must receive weight.
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        heads = np.zeros((5, 3))
        tails = directions
        vertices = np.array([[0.0, 0.0, 0.0]])

        bone_indices, weights = nearest_bone_blend_weights(vertices, heads, tails)

        self.assertEqual(bone_indices.shape[1], 4)
        assigned = [i for i in bone_indices[0].tolist() if i != -1]
        self.assertEqual(len(assigned), 4)
        self.assertAlmostEqual(weights[0].sum(), 1.0)

    def test_max_influences_can_still_be_overridden(self):
        heads = np.zeros((3, 3))
        tails = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        vertices = np.array([[0.0, 0.0, 0.0]])

        bone_indices, weights = nearest_bone_blend_weights(
            vertices, heads, tails, max_influences=2,
        )

        self.assertEqual(bone_indices.shape, (1, 2))
        assigned = [i for i in bone_indices[0].tolist() if i != -1]
        self.assertEqual(len(assigned), 2)


class StitchCoincidentVertexWeightsTests(unittest.TestCase):
    """`stitch_coincident_vertex_weights` is the appearance-preserving
    seam-tear fix: vertices sitting at the same 3D position (duplicated
    for UV/normal seams) must end up with identical bone weights so a pose
    cannot pull them apart.
    """

    def _sparse(self, rows: list[list[tuple[int, float]]], max_influences: int) -> tuple[np.ndarray, np.ndarray]:
        bone_indices = np.full((len(rows), max_influences), -1, dtype=np.int64)
        weights = np.zeros((len(rows), max_influences), dtype=np.float64)
        for row_index, row in enumerate(rows):
            for slot, (bone, weight) in enumerate(row):
                bone_indices[row_index, slot] = bone
                weights[row_index, slot] = weight
        return bone_indices, weights

    def test_two_coincident_vertices_with_different_single_bones_end_fifty_fifty(self):
        vertices = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        bone_indices, weights = self._sparse([[(0, 1.0)], [(1, 1.0)]], max_influences=2)

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        for row in range(2):
            assigned = {
                int(b): float(w) for b, w in zip(out_bones[row], out_weights[row]) if b != -1
            }
            self.assertEqual(set(assigned), {0, 1})
            self.assertAlmostEqual(assigned[0], 0.5)
            self.assertAlmostEqual(assigned[1], 0.5)

    def test_three_vertex_stack_averages(self):
        vertices = np.tile(np.array([5.0, -1.0, 0.25]), (3, 1))
        bone_indices, weights = self._sparse(
            [[(0, 1.0)], [(0, 0.6), (1, 0.4)], [(1, 1.0)]], max_influences=2
        )

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        expected_bone0 = (1.0 + 0.6 + 0.0) / 3.0
        expected_bone1 = (0.0 + 0.4 + 1.0) / 3.0
        for row in range(3):
            assigned = {
                int(b): float(w) for b, w in zip(out_bones[row], out_weights[row]) if b != -1
            }
            self.assertAlmostEqual(assigned[0], expected_bone0, places=6)
            self.assertAlmostEqual(assigned[1], expected_bone1, places=6)

    def test_non_coincident_vertices_are_untouched(self):
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        bone_indices, weights = self._sparse(
            [[(0, 1.0)], [(1, 0.7), (2, 0.3)], [(2, 1.0)]], max_influences=2
        )

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        np.testing.assert_array_equal(out_bones, bone_indices)
        np.testing.assert_allclose(out_weights, weights)

    def test_caps_influences_at_the_input_arrays_width(self):
        # Three coincident vertices whose union of bones is 3, but the
        # array only has room for max_influences=2 -- the weakest bone
        # must be dropped and the remaining two renormalized to sum to 1.
        vertices = np.zeros((3, 3))
        bone_indices, weights = self._sparse(
            [[(0, 1.0), (-1, 0.0)], [(1, 0.9), (2, 0.1)], [(2, 0.05), (1, 0.95)]],
            max_influences=2,
        )

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        self.assertEqual(out_bones.shape, (3, 2))
        for row in range(3):
            assigned = {
                int(b): float(w) for b, w in zip(out_bones[row], out_weights[row]) if b != -1
            }
            self.assertLessEqual(len(assigned), 2)
            self.assertAlmostEqual(sum(assigned.values()), 1.0, places=6)
            # bone2's total contribution (0 + 0.1 + 0.05)/3 = 0.05 is the
            # weakest of the three and must be the one dropped.
            self.assertNotIn(2, assigned)

    def test_stitched_rows_sum_to_one(self):
        rng = np.random.default_rng(3)
        num_groups = 20
        positions = rng.uniform(-5.0, 5.0, size=(num_groups, 3))
        vertices = []
        rows = []
        for group_index in range(num_groups):
            member_count = rng.integers(1, 4)
            for _ in range(member_count):
                vertices.append(positions[group_index])
                num_bones = int(rng.integers(1, 4))
                bone_choices = rng.choice(6, size=num_bones, replace=False)
                raw = rng.uniform(0.1, 1.0, size=num_bones)
                raw = raw / raw.sum()
                rows.append(list(zip(bone_choices.tolist(), raw.tolist())))
        vertices = np.array(vertices)
        bone_indices, weights = self._sparse(rows, max_influences=4)

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        row_sums = out_weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(vertices)), atol=1e-9)

    def test_group_members_with_zero_weight_at_all_are_left_untouched(self):
        # A coincident group where every member has no valid bone weight
        # at all should not error, and should be left as-is (nothing to
        # average).
        vertices = np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        bone_indices, weights = self._sparse([[], []], max_influences=2)

        out_bones, out_weights = stitch_coincident_vertex_weights(vertices, bone_indices, weights)

        np.testing.assert_array_equal(out_bones, bone_indices)
        np.testing.assert_allclose(out_weights, weights)


# --- N3: template-conformant humanoid skeleton building ---
#
# Fixture: a plausible (not anatomically precise -- direction/roll are the
# template's job, not the builder's) humanoid, laid out in the family house
# frame (Z-up, left = +X), with only the 18 bones UniRig's own semantic
# namer ever actually emits (never "root" or "clavicle_l/r" -- see
# `semantic_joint_names`'s docstring). `skeleton_height` is 1.5 (head Z 1.55
# minus foot Z 0.05).
_BUILD_FIXTURE_NAMES = [
    "pelvis", "spine_1", "spine_2", "spine_3", "neck", "head",
    "upper_arm_l", "lower_arm_l", "hand_l",
    "upper_arm_r", "lower_arm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l",
    "thigh_r", "calf_r", "foot_r",
]
_BUILD_FIXTURE_JOINTS = np.array([
    [0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [0.0, 0.0, 1.2], [0.0, 0.05, 1.35],
    [0.0, 0.0, 1.45], [0.0, 0.0, 1.55],
    [0.2, 0.0, 1.35], [0.45, 0.0, 1.35], [0.65, 0.0, 1.35],
    [-0.2, 0.0, 1.35], [-0.45, 0.0, 1.35], [-0.65, 0.0, 1.35],
    [0.1, 0.0, 1.0], [0.1, 0.0, 0.5], [0.1, 0.0, 0.05],
    [-0.1, 0.0, 1.0], [-0.1, 0.0, 0.5], [-0.1, 0.0, 0.05],
])


class BuildTemplateConformantSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("biped")

    def test_full_prediction_synthesizes_root_and_clavicles_with_no_missing_bones(self):
        # skip_optional_bones pins this to the required tier: growing the
        # optional tier is the default and has its own tests below.
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template,
            skip_optional_bones=True,
        )

        self.assertEqual(build.missing_required, [])
        self.assertEqual(set(build.names), {bone.name for bone in self.template.required_bones})
        self.assertEqual(build.synthesized, ["root", "clavicle_l", "clavicle_r"])
        self.assertAlmostEqual(build.skeleton_height, 1.5)

    def test_synthesized_root_is_anchored_at_pelvis_head(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        root_head = build.heads[build.names.index("root")]
        pelvis_head = build.heads[build.names.index("pelvis")]
        np.testing.assert_allclose(root_head, pelvis_head)
        np.testing.assert_allclose(root_head, [0.0, 0.0, 1.0])

    def test_synthesized_clavicle_head_is_midpoint_of_spine_3_and_upper_arm(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        spine_3 = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("spine_3")]
        upper_arm_l = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("upper_arm_l")]
        upper_arm_r = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("upper_arm_r")]

        clavicle_l_head = build.heads[build.names.index("clavicle_l")]
        clavicle_r_head = build.heads[build.names.index("clavicle_r")]
        np.testing.assert_allclose(clavicle_l_head, (spine_3 + upper_arm_l) / 2.0)
        np.testing.assert_allclose(clavicle_r_head, (spine_3 + upper_arm_r) / 2.0)

    def test_tails_follow_template_rest_direction_scaled_to_this_skeleton_height(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        for name in ("pelvis", "foot_l", "hand_r", "root"):
            bone = self.template.bone(name)
            index = build.names.index(name)
            expected_tail = build.heads[index] + np.asarray(bone.rest_direction) * (
                bone.relative_length * build.skeleton_height
            )
            np.testing.assert_allclose(build.tails[index], expected_tail)

    def test_rolls_are_copied_from_the_template_unchanged(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        for name in ("clavicle_l", "upper_arm_r", "foot_l"):
            index = build.names.index(name)
            self.assertEqual(build.rolls[index], self.template.bone(name).roll)

    def test_parents_are_template_parent_names(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        parent_by_name = dict(zip(build.names, build.parents))
        self.assertIsNone(parent_by_name["root"])
        self.assertEqual(parent_by_name["pelvis"], "root")
        self.assertEqual(parent_by_name["clavicle_l"], "spine_3")
        self.assertEqual(parent_by_name["upper_arm_l"], "clavicle_l")

    def test_missing_upper_arm_cascades_to_its_children_and_blocks_clavicle_synthesis(self):
        keep = [i for i, name in enumerate(_BUILD_FIXTURE_NAMES) if name != "upper_arm_l"]
        names = [_BUILD_FIXTURE_NAMES[i] for i in keep]
        joints = _BUILD_FIXTURE_JOINTS[keep]

        build = build_template_conformant_skeleton(names, joints, self.template)

        self.assertEqual(
            set(build.missing_required),
            {"upper_arm_l", "lower_arm_l", "hand_l", "clavicle_l"},
        )
        # Unaffected bones (including the right arm and both legs) are
        # still placed -- this is a cascading failure of one limb, not a
        # global one.
        self.assertIn("upper_arm_r", build.names)
        self.assertIn("thigh_l", build.names)

    def test_optional_bone_is_emitted_only_when_predicted_under_the_skip_gate(self):
        without_tip = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template,
            skip_optional_bones=True,
        )
        self.assertNotIn("head_tip", without_tip.names)
        # A missing optional bone is not a fitting failure.
        self.assertEqual(without_tip.missing_required, [])

        names = _BUILD_FIXTURE_NAMES + ["head_tip"]
        joints = np.concatenate([_BUILD_FIXTURE_JOINTS, [[0.0, 0.0, 1.6]]], axis=0)
        with_tip = build_template_conformant_skeleton(
            names, joints, self.template, skip_optional_bones=True,
        )

        self.assertIn("head_tip", with_tip.names)
        np.testing.assert_allclose(
            with_tip.heads[with_tip.names.index("head_tip")], [0.0, 0.0, 1.6]
        )

    def test_unmatched_names_do_not_leak_into_the_output(self):
        names = _BUILD_FIXTURE_NAMES + ["bone_99", "hand_extra_l_23"]
        joints = np.concatenate([_BUILD_FIXTURE_JOINTS, [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]], axis=0)

        build = build_template_conformant_skeleton(names, joints, self.template)

        self.assertNotIn("bone_99", build.names)
        self.assertNotIn("hand_extra_l_23", build.names)


class TemplateSideConformanceFindingsTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("biped")
        self.left = np.array([1.0, 0.0, 0.0])
        self.reference_point = np.array([0.0, 0.0, 1.0])

    def test_correctly_sided_rig_has_no_findings(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        findings = template_side_conformance_findings(
            build.names, build.heads, self.template, self.left, self.reference_point
        )

        self.assertEqual(findings, [])

    def test_leg_only_mirror_is_caught_by_position_even_though_direction_is_identical(self):
        # thigh_l/thigh_r's template rest_direction is literally identical
        # (both point straight down) -- this is exactly the N2 gap this
        # helper exists to close.
        names = ["thigh_l", "thigh_r"]
        heads = np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])  # physically swapped

        findings = template_side_conformance_findings(
            names, heads, self.template, self.left, self.reference_point
        )

        codes_by_bone = {f["message"] for f in findings}
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(f["code"] == "mirrored_side_position" for f in findings))
        self.assertTrue(any("thigh_l" in message for message in codes_by_bone))
        self.assertTrue(any("thigh_r" in message for message in codes_by_bone))

    def test_full_body_mirror_is_also_caught(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )
        mirrored_heads = build.heads.copy()
        mirrored_heads[:, 0] *= -1.0  # flip every bone's X (side) coordinate

        findings = template_side_conformance_findings(
            build.names, mirrored_heads, self.template, self.left, self.reference_point
        )

        flagged = {f["message"].split("'")[1] for f in findings}
        expected_lr_bones = {name for name in build.names if name.endswith(("_l", "_r"))}
        self.assertEqual(flagged, expected_lr_bones)

    def test_degenerate_left_vector_returns_no_findings(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )

        findings = template_side_conformance_findings(
            build.names, build.heads, self.template, np.zeros(3), self.reference_point
        )

        self.assertEqual(findings, [])


# --- mrmmn: growing the template's optional tier (fingers, toes, head_tip) ---
#
# UniRig predicts none of these 45 bones, so they are synthesized from the
# hands/feet/head the fitter did place. Every assertion below is pure math on
# the fixture humanoid above -- no bpy, no mesh file.

_LONG_MITT_REACH = 0.20  # ~13% of the fixture's 1.5 height: a big hand, not an absurd one
_LEFT_FINGER_ROOTS = ("index_1_l", "middle_1_l", "ring_1_l", "pinky_1_l", "thumb_1_l")
_LEFT_FINGER_TIPS = ("index_tip_l", "middle_tip_l", "ring_tip_l", "pinky_tip_l", "thumb_tip_l")


def _mitt_cloud(anchor_head, reach, *, count=2000, half_width=0.04, seed=0):
    """A crude hand blob: `count` points filling a box that starts at
    `anchor_head` and extends `reach` along +X (the biped template's own
    left-hand axis), which is the geometry `measure_distal_span` reads."""
    rng = np.random.default_rng(seed)
    return np.asarray(anchor_head) + np.column_stack([
        rng.uniform(0.0, reach, count),
        rng.uniform(-half_width, half_width, count),
        rng.uniform(-half_width, half_width, count),
    ])


class OptionalChainHelperTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("biped")

    def test_anchor_of_every_optional_bone_is_its_nearest_required_ancestor(self):
        self.assertEqual(template_anchor_bone(self.template, "index_1_l"), "hand_l")
        self.assertEqual(template_anchor_bone(self.template, "thumb_tip_l"), "hand_l")
        self.assertEqual(template_anchor_bone(self.template, "pinky_3_r"), "hand_r")
        self.assertEqual(template_anchor_bone(self.template, "toes_tip_r"), "foot_r")
        self.assertEqual(template_anchor_bone(self.template, "head_tip"), "head")
        # A required bone anchors itself; an unknown name anchors nothing.
        self.assertEqual(template_anchor_bone(self.template, "hand_l"), "hand_l")
        self.assertIsNone(template_anchor_bone(self.template, "bone_99"))

    def test_chain_root_is_the_first_optional_bone_in_the_chain(self):
        self.assertEqual(optional_chain_root(self.template, "index_tip_l"), "index_1_l")
        self.assertEqual(optional_chain_root(self.template, "index_1_l"), "index_1_l")
        self.assertEqual(optional_chain_root(self.template, "thumb_3_r"), "thumb_1_r")
        self.assertEqual(optional_chain_root(self.template, "toes_tip_l"), "toes_l")
        self.assertEqual(optional_chain_root(self.template, "head_tip"), "head_tip")
        # Required bones and unknown names root no chain.
        self.assertIsNone(optional_chain_root(self.template, "hand_l"))
        self.assertIsNone(optional_chain_root(self.template, "bone_99"))
        # Every optional bone belongs to exactly one of the 13 chains.
        roots = {
            optional_chain_root(self.template, bone.name)
            for bone in self.template.optional_bones
        }
        self.assertEqual(len(roots), 13)  # 5 fingers x 2 hands + toes x 2 + head_tip

    def test_chain_layout_covers_exactly_that_chain(self):
        index = optional_chain_layout(self.template, "index_1_l")
        self.assertEqual(set(index), {"index_1_l", "index_2_l", "index_3_l", "index_tip_l"})
        self.assertEqual(set(optional_chain_layout(self.template, "toes_r")), {"toes_r", "toes_tip_r"})
        self.assertEqual(set(optional_chain_layout(self.template, "head_tip")), {"head_tip"})
        # A required bone roots no chain at all.
        self.assertEqual(optional_chain_layout(self.template, "hand_l"), {})
        # The 13 chains partition the template's 45 optional bones.
        covered = []
        for bone in self.template.optional_bones:
            root = optional_chain_root(self.template, bone.name)
            covered.extend(optional_chain_layout(self.template, root))
        self.assertEqual(set(covered), {bone.name for bone in self.template.optional_bones})

    def test_chain_layout_starts_at_the_anchor_s_tail_and_chains_from_there(self):
        layout = optional_chain_layout(self.template, "index_1_l")
        hand = self.template.bone("hand_l")
        index_1 = self.template.bone("index_1_l")
        np.testing.assert_allclose(
            layout["index_1_l"], np.asarray(hand.rest_direction) * hand.relative_length
        )
        np.testing.assert_allclose(
            layout["index_2_l"],
            layout["index_1_l"] + np.asarray(index_1.rest_direction) * index_1.relative_length,
        )

    def test_distal_span_is_none_without_enough_region_vertices(self):
        anchor = np.zeros(3)
        axis = np.array([1.0, 0.0, 0.0])
        self.assertIsNone(measure_distal_span(anchor, axis, np.zeros((0, 3)), 1.0, 1.0))
        # Vertices behind the anchor along the axis are not the hand.
        behind = np.column_stack([np.full(50, -0.1), np.zeros(50), np.zeros(50)])
        self.assertIsNone(measure_distal_span(anchor, axis, behind, 1.0, 1.0))
        # Vertices further along the axis than the limb could be are not
        # the hand either.
        far = np.column_stack([np.full(50, 5.0), np.zeros(50), np.zeros(50)])
        self.assertIsNone(measure_distal_span(anchor, axis, far, 1.0, 1.0))
        # Nor are vertices off to the side of the chain's own axis (the
        # torso, when the axis runs down an arm).
        beside = np.column_stack([np.full(50, 0.1), np.full(50, 3.0), np.zeros(50)])
        self.assertIsNone(measure_distal_span(anchor, axis, beside, 1.0, 1.0))

    def test_distal_span_measures_reach_along_the_axis(self):
        span, saturated = measure_distal_span(
            np.zeros(3), np.array([1.0, 0.0, 0.0]), _mitt_cloud(np.zeros(3), 0.25),
            lateral_limit=1.0, along_limit=1.0,
        )
        self.assertAlmostEqual(span, 0.25, delta=0.01)
        self.assertFalse(saturated)

    def test_distal_span_reports_a_limb_that_runs_past_the_region(self):
        # A measurement that hits its own bound is a floor, not an answer.
        span, saturated = measure_distal_span(
            np.zeros(3), np.array([1.0, 0.0, 0.0]), _mitt_cloud(np.zeros(3), 0.50),
            lateral_limit=1.0, along_limit=0.20,
        )
        self.assertAlmostEqual(span, 0.20, delta=0.005)
        self.assertTrue(saturated)

    def test_chain_scale_falls_back_to_template_proportion_without_a_mesh(self):
        scale, source = optional_chain_scale(self.template, "index_1_l", np.zeros(3), 1.5, None)
        self.assertEqual(scale, 1.0)
        self.assertEqual(source, "template-proportion-no-mesh")

    def test_chain_scale_is_safe_on_a_degenerate_skeleton_height(self):
        scale, source = optional_chain_scale(
            self.template, "index_1_l", np.zeros(3), 0.0, _mitt_cloud(np.zeros(3), 0.25)
        )
        self.assertEqual(scale, 1.0)
        self.assertEqual(source, "template-proportion-degenerate-height")

    def test_chain_scale_clamps_a_hand_region_that_measures_absurdly_short(self):
        # A wrist the mesh barely extends past cannot shrink the chain to
        # nothing -- it clamps, and says so.
        scale, source = optional_chain_scale(
            self.template, "index_1_l", np.zeros(3), 1.5, _mitt_cloud(np.zeros(3), 0.005)
        )
        self.assertEqual(source, "measured-mesh-clamped")
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(scale, 0.35)


class GrowOptionalTierTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("biped")
        self.build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template
        )
        self.index = {name: i for i, name in enumerate(self.build.names)}

    def test_all_66_template_bones_are_emitted_by_default(self):
        self.assertEqual(set(self.build.names), self.template.bone_names())
        self.assertEqual(len(self.build.names), 66)
        self.assertEqual(len(self.build.synthesized_optional), 45)
        self.assertEqual(
            set(self.build.synthesized_optional),
            {bone.name for bone in self.template.optional_bones},
        )
        self.assertEqual(self.build.missing_required, [])

    def test_skip_optional_bones_restores_the_21_bone_rig(self):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template,
            skip_optional_bones=True,
        )

        self.assertEqual(len(build.names), 21)
        self.assertEqual(build.synthesized_optional, ())
        self.assertEqual(build.optional_chain_scales, ())

    def test_grown_rig_is_still_fully_template_conformant(self):
        findings = conformance_report(
            self.build.names, self.build.parents, self.build.heads, self.build.tails,
            self.build.rolls, template=self.template,
        )

        self.assertEqual(findings, [])

    def test_every_optional_bone_s_head_sits_exactly_on_its_parent_s_tail(self):
        # No gaps anywhere in a grown chain -- including at the wrist and
        # ankle, where the chain hangs off a required bone whose own length
        # this build never rescales.
        for bone in self.template.optional_bones:
            child = self.index[bone.name]
            parent = self.index[bone.parent]
            np.testing.assert_allclose(
                self.build.heads[child], self.build.tails[parent], atol=1e-12,
                err_msg=f"{bone.name} head is not on {bone.parent}'s tail",
            )

    def test_head_tip_is_exactly_the_head_bone_s_tail(self):
        head = self.template.bone("head")
        expected = self.build.heads[self.index["head"]] + np.asarray(
            head.rest_direction
        ) * (head.relative_length * self.build.skeleton_height)

        np.testing.assert_allclose(self.build.heads[self.index["head_tip"]], expected)

    def test_each_finger_follows_its_own_template_rest_direction(self):
        # The fan is the template's per-finger directions and nothing else,
        # so each grown segment must point exactly where its own spec says.
        for name in ("index_1_l", "middle_2_l", "ring_3_r", "pinky_1_r", "thumb_2_l", "toes_l"):
            i = self.index[name]
            direction = self.build.tails[i] - self.build.heads[i]
            direction = direction / np.linalg.norm(direction)
            # 1e-5, not exact: the template's stored directions are the
            # donor's own, rounded to six decimals, so they are not quite
            # unit vectors and re-normalizing moves the last digit.
            np.testing.assert_allclose(
                direction, self.template.bone(name).rest_direction, atol=1e-5
            )

    def test_the_five_finger_chains_fan_apart(self):
        tips = {name: self.build.heads[self.index[name]] for name in _LEFT_FINGER_TIPS}
        # All five start at the same point (the template records no knuckle
        # offsets), so every bit of separation comes from the directions.
        for a, b in itertools.combinations(tips, 2):
            self.assertGreater(
                float(np.linalg.norm(tips[a] - tips[b])), 1e-4, f"{a} and {b} coincide"
            )
        # Index tilts one way in Y, pinky the other; the thumb leaves the
        # palm plane entirely (a Z offset no finger has).
        self.assertLess(tips["index_tip_l"][1], tips["pinky_tip_l"][1])
        palm_z = self.build.heads[self.index["hand_l"]][2]
        for name in ("index_tip_l", "middle_tip_l", "ring_tip_l", "pinky_tip_l"):
            self.assertAlmostEqual(tips[name][2], palm_z, places=9)
        self.assertLess(tips["thumb_tip_l"][2], palm_z - 0.01)

    def test_left_and_right_chains_are_mirror_images(self):
        for left_name in ("index_tip_l", "thumb_2_l", "toes_tip_l"):
            right_name = self.template.mirror_of(left_name)
            left = self.build.heads[self.index[left_name]]
            right = self.build.heads[self.index[right_name]]
            # Same six-decimal rounding caveat as above: the donor's own
            # left and right directions differ in the last stored digit.
            np.testing.assert_allclose(left[0], -right[0], atol=1e-5)
            np.testing.assert_allclose(left[1:], right[1:], atol=1e-5)

    def test_chains_are_not_grown_when_the_skeleton_height_is_degenerate(self):
        flat = np.tile(_BUILD_FIXTURE_JOINTS[0], (len(_BUILD_FIXTURE_NAMES), 1))

        build = build_template_conformant_skeleton(_BUILD_FIXTURE_NAMES, flat, self.template)

        self.assertEqual(build.skeleton_height, 0.0)
        self.assertEqual(build.synthesized_optional, ())


class OptionalChainMeshScalingTests(unittest.TestCase):
    """The whole point of anchoring locally: an oversized mitt gets an
    oversized finger chain, the other hand is unaffected, and each finger is
    measured along its own axis rather than a hand-wide average."""

    def setUp(self):
        self.template = load_template("biped")
        self.hand_l = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("hand_l")]

    def _build(self, vertices):
        build = build_template_conformant_skeleton(
            _BUILD_FIXTURE_NAMES, _BUILD_FIXTURE_JOINTS, self.template,
            mesh_vertices=vertices,
        )
        return build, dict(zip(build.names, range(len(build.names))))

    def _scales(self, build):
        return {root: (scale, source) for root, scale, source in build.optional_chain_scales}

    def test_a_long_mitt_stretches_only_that_hand_s_chains(self):
        build, _ = self._build(_mitt_cloud(self.hand_l, _LONG_MITT_REACH))
        scales = self._scales(build)

        for root in _LEFT_FINGER_ROOTS:
            self.assertEqual(scales[root][1], "measured-mesh", root)
            self.assertGreater(scales[root][0], 1.2, root)
        # No vertices near the right hand or either foot: those chains keep
        # exact template proportion rather than borrowing the left hand's.
        for root in ("index_1_r", "thumb_1_r", "toes_l", "toes_r"):
            self.assertEqual(scales[root], (1.0, "template-proportion-no-region-vertices"), root)

    def test_grown_fingers_stay_inside_the_measured_mitt(self):
        reach = _LONG_MITT_REACH
        build, index = self._build(_mitt_cloud(self.hand_l, reach))

        for name in _LEFT_FINGER_TIPS:
            along_hand = float(build.heads[index[name]][0] - self.hand_l[0])
            self.assertGreater(along_hand, 0.0, name)
            self.assertLess(along_hand, reach, f"{name} pokes out past the mesh hand")
        # ...and they use most of it, rather than leaving the mitt's far
        # half unarticulated (that is the mitt symptom this bead fixes).
        middle = float(build.heads[index["middle_tip_l"]][0] - self.hand_l[0])
        self.assertGreater(middle, 0.6 * reach)

    def test_the_thumb_is_measured_along_its_own_axis_not_the_fingers(self):
        # The thumb leaves the palm plane, so on a flat slab of a mitt it
        # must not be stretched as far as the straight fingers -- that is
        # what pushed thumb joints out through the surface before chains
        # were scaled individually.
        build, _ = self._build(_mitt_cloud(self.hand_l, _LONG_MITT_REACH, half_width=0.02))
        scales = self._scales(build)

        self.assertLess(scales["thumb_1_l"][0], scales["middle_1_l"][0])

    def test_a_short_mitt_shrinks_the_chain_relative_to_template_proportion(self):
        long_build, long_index = self._build(_mitt_cloud(self.hand_l, _LONG_MITT_REACH))
        short_build, short_index = self._build(_mitt_cloud(self.hand_l, 0.10))

        self.assertLess(
            self._scales(short_build)["middle_1_l"][0],
            self._scales(long_build)["middle_1_l"][0],
        )
        self.assertLess(
            float(short_build.heads[short_index["middle_tip_l"]][0]),
            float(long_build.heads[long_index["middle_tip_l"]][0]),
        )

    def test_a_scaled_chain_keeps_its_segments_joined_and_stays_conformant(self):
        build, index = self._build(_mitt_cloud(self.hand_l, _LONG_MITT_REACH))

        for bone in self.template.optional_bones:
            np.testing.assert_allclose(
                build.heads[index[bone.name]], build.tails[index[bone.parent]], atol=1e-12,
                err_msg=f"{bone.name} detached from {bone.parent} after scaling",
            )
        self.assertEqual(
            conformance_report(
                build.names, build.parents, build.heads, build.tails, build.rolls,
                template=self.template,
            ),
            [],
        )

    def test_toes_are_scaled_by_the_foot_region_not_the_hand_region(self):
        foot_l = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("foot_l")]
        rng = np.random.default_rng(7)
        # A long foot: the toe chain runs forward (-Y) from the ankle.
        toe_cloud = foot_l + np.column_stack([
            rng.uniform(-0.04, 0.04, 2000),
            rng.uniform(-0.30, 0.0, 2000),
            rng.uniform(-0.04, 0.04, 2000),
        ])
        build, index = self._build(np.concatenate([toe_cloud, _mitt_cloud(self.hand_l, _LONG_MITT_REACH)]))
        scales = self._scales(build)

        self.assertEqual(scales["toes_l"][1], "measured-mesh")
        self.assertGreater(scales["toes_l"][0], 1.0)
        # The hand chains are unaffected by a foot measurement -- toes_r is
        # deliberately not asserted here: this fixture's ankles are 0.2
        # apart, inside the foot region radius, so one foot's vertices do
        # reach the other. That leak is real (a narrow stance measures both
        # feet as one) and is why the region radius is bounded at all.
        self.assertEqual(
            scales["index_1_r"], (1.0, "template-proportion-no-region-vertices")
        )
        # The left toe reaches forward into the measured foot, not sideways.
        self.assertLess(float(build.heads[index["toes_tip_l"]][1]), float(foot_l[1]))


# --- distal anchor refinement ------------------------------------------------
#
# A synthetic left arm with the real anatomy UniRig misreads: a forearm that
# narrows to a wrist, a palm bulge, then fingers. The fixture's elbow is the
# build fixture's own `lower_arm_l`; `hand_l` is deliberately fitted mid-
# forearm, which is what UniRig actually does (measured on zix: the fitted
# wrist sits 60% of the way up the forearm from the elbow).

_REFINE_ELBOW_L = _BUILD_FIXTURE_JOINTS[_BUILD_FIXTURE_NAMES.index("lower_arm_l")]
_REFINE_FOREARM_SPAN = 0.28
_REFINE_PALM_SPAN = 0.07
_REFINE_FINGER_SPAN = 0.05
_REFINE_LIMB_EXTENT = _REFINE_FOREARM_SPAN + _REFINE_PALM_SPAN + _REFINE_FINGER_SPAN
_REFINE_FOREARM_RADIUS = 0.030
_REFINE_PALM_RADIUS = 0.055
_REFINE_FINGER_RADIUS = 0.035
# Where the fitter put the wrist: mid-forearm, 0.14 out from the elbow.
_REFINE_FITTED_OFFSET = 0.14


def _tube_cloud(origin, axis, segments, *, rings_per_unit=1400, per_ring=16):
    """A hollow tube of stacked rings along `axis`, one `(length, radius)`
    segment after another. Points sit ON the surface, so the profile's p90
    lateral distance in any slab is that slab's radius exactly -- the shape
    is the fixture, with no sampling noise to argue about."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(axis, helper)
    side = side / np.linalg.norm(side)
    up = np.cross(axis, side)
    angles = np.linspace(0.0, 2.0 * np.pi, per_ring, endpoint=False)
    ring = np.cos(angles)[:, None] * side + np.sin(angles)[:, None] * up
    origin = np.asarray(origin, dtype=np.float64)
    points = []
    start = 0.0
    for length, radius in segments:
        count = max(2, int(round(length * rings_per_unit)))
        for offset in np.linspace(start, start + length, count, endpoint=False):
            points.append(origin + axis * offset + ring * radius)
        start += length
    return np.concatenate(points)


def _arm_cloud(*, palm_radius=_REFINE_PALM_RADIUS, origin=None, axis=(1.0, 0.0, 0.0),
               medial_shift=(0.0, 0.0, 0.0)):
    return _tube_cloud(
        (_REFINE_ELBOW_L if origin is None else origin) + np.asarray(medial_shift, dtype=np.float64),
        axis,
        [
            (_REFINE_FOREARM_SPAN, _REFINE_FOREARM_RADIUS),
            (_REFINE_PALM_SPAN, palm_radius),
            (_REFINE_FINGER_SPAN, _REFINE_FINGER_RADIUS),
        ],
    )


def _refine_fixture(hand_offset=_REFINE_FITTED_OFFSET):
    """The build fixture with `hand_l` moved to `hand_offset` out from the
    elbow along +X (the arm's own axis)."""
    names = list(_BUILD_FIXTURE_NAMES)
    joints = _BUILD_FIXTURE_JOINTS.copy()
    joints[names.index("hand_l")] = _REFINE_ELBOW_L + np.array([hand_offset, 0.0, 0.0])
    return names, joints


class LimbRadiusProfileTests(unittest.TestCase):
    def test_profile_reads_back_each_segment_s_own_radius(self):
        profile = limb_radius_profile(
            _REFINE_ELBOW_L, np.array([1.0, 0.0, 0.0]), _arm_cloud(),
            lateral_limit=0.225, along_limit=0.9,
        )

        self.assertIsNotNone(profile)
        self.assertFalse(profile.saturated)
        self.assertAlmostEqual(profile.extent, _REFINE_LIMB_EXTENT, delta=2 * profile.width)
        centres = (np.arange(profile.usable) + 0.5) * profile.width
        radii = np.asarray(profile.radii[:profile.usable])
        forearm = radii[centres < _REFINE_FOREARM_SPAN - profile.width]
        palm = radii[
            (centres > _REFINE_FOREARM_SPAN + profile.width)
            & (centres < _REFINE_FOREARM_SPAN + _REFINE_PALM_SPAN - profile.width)
        ]
        np.testing.assert_allclose(forearm, _REFINE_FOREARM_RADIUS, atol=1e-9)
        np.testing.assert_allclose(palm, _REFINE_PALM_RADIUS, atol=1e-9)

    def test_a_limb_running_past_the_region_bound_reports_saturated(self):
        profile = limb_radius_profile(
            _REFINE_ELBOW_L, np.array([1.0, 0.0, 0.0]), _arm_cloud(),
            lateral_limit=0.225, along_limit=0.2,
        )

        self.assertTrue(profile.saturated)

    def test_an_unbounded_lateral_limit_lets_the_opposite_limb_into_the_profile(self):
        # Why the region is capped at a fraction of the way to the mirror
        # anchor: a vertical knee->ankle axis with only a height-derived
        # bound swallows the other leg, and its silhouette invents an
        # "ankle" that is not on this limb at all.
        neighbour = _arm_cloud(origin=_REFINE_ELBOW_L + np.array([0.0, 0.2, 0.0]))
        cloud = np.concatenate([_arm_cloud(), neighbour])

        wide = limb_radius_profile(
            _REFINE_ELBOW_L, np.array([1.0, 0.0, 0.0]), cloud,
            lateral_limit=0.30, along_limit=0.9,
        )
        narrow = limb_radius_profile(
            _REFINE_ELBOW_L, np.array([1.0, 0.0, 0.0]), cloud,
            lateral_limit=0.09, along_limit=0.9,
        )

        self.assertGreater(wide.radii[4], 4 * _REFINE_FOREARM_RADIUS)
        self.assertAlmostEqual(narrow.radii[4], _REFINE_FOREARM_RADIUS, places=9)

    def test_an_empty_or_degenerate_region_is_no_profile_at_all(self):
        self.assertIsNone(limb_radius_profile(
            _REFINE_ELBOW_L, np.array([1.0, 0.0, 0.0]), np.zeros((0, 3)),
            lateral_limit=0.225, along_limit=0.9,
        ))
        self.assertIsNone(limb_radius_profile(
            _REFINE_ELBOW_L, np.zeros(3), _arm_cloud(),
            lateral_limit=0.225, along_limit=0.9,
        ))
        # Everything is behind the proximal joint: nothing to bin.
        self.assertIsNone(limb_radius_profile(
            _REFINE_ELBOW_L, np.array([-1.0, 0.0, 0.0]), _arm_cloud(),
            lateral_limit=0.225, along_limit=0.9,
        ))


class RefineDistalAnchorTests(unittest.TestCase):
    def setUp(self):
        self.names, self.joints = _refine_fixture()

    def _refine(self, cloud, **kwargs):
        return refine_distal_anchor(
            self.joints, self.names, cloud, ("lower_arm_l", "hand_l"),
            mirror_anchor="hand_r", **kwargs,
        )

    def test_a_clear_bulge_snaps_the_wrist_to_the_pre_bulge_narrow(self):
        report = self._refine(_arm_cloud())

        self.assertTrue(report.applied, report.reason)
        self.assertEqual(report.reason, "refined")
        along = float(np.asarray(report.wrist_to)[0] - _REFINE_ELBOW_L[0])
        # The anatomical wrist is where the forearm ends; allow the bin the
        # detector had to choose between, no more.
        self.assertAlmostEqual(along, _REFINE_FOREARM_SPAN, delta=0.02)
        self.assertGreater(report.bulge_ratio, 1.5)
        self.assertGreater(report.confidence, 0.9)

    def test_the_refined_anchor_leaves_the_whole_hand_for_the_finger_chains(self):
        report = self._refine(_arm_cloud())

        self.assertAlmostEqual(report.limb_extent, _REFINE_LIMB_EXTENT, delta=0.03)
        self.assertAlmostEqual(
            report.hand_extent, _REFINE_PALM_SPAN + _REFINE_FINGER_SPAN, delta=0.03,
        )
        # ...and never past the end of the mesh, which is the guard rail
        # that stops a chain from being anchored in mid-air.
        along = float(np.asarray(report.wrist_to)[0] - _REFINE_ELBOW_L[0])
        self.assertLess(along, report.limb_extent)

    def test_a_centred_limb_needs_no_sideways_correction(self):
        report = self._refine(_arm_cloud())

        self.assertLess(report.medial_offset, 0.002)
        np.testing.assert_allclose(
            np.asarray(report.wrist_to)[1:], _REFINE_ELBOW_L[1:], atol=0.002,
        )

    def test_the_anchor_is_pulled_onto_the_limb_s_own_medial_point(self):
        # The proximal->anchor line runs through the joint being corrected,
        # so it is only as centred as that joint was. Measured on zix and
        # Mary the line misses the hand's own centre by 0.025-0.031, which
        # is most of a finger's thickness.
        shift = np.array([0.0, 0.02, 0.0])

        report = self._refine(_arm_cloud(medial_shift=shift))

        self.assertTrue(report.applied, report.reason)
        self.assertGreater(report.medial_offset, 0.015)
        np.testing.assert_allclose(
            np.asarray(report.wrist_to)[1:], (_REFINE_ELBOW_L + shift)[1:], atol=0.004,
        )

    def test_a_mitt_with_no_bulge_is_skipped(self):
        flat = _tube_cloud(
            _REFINE_ELBOW_L, (1.0, 0.0, 0.0),
            [(_REFINE_LIMB_EXTENT, _REFINE_FOREARM_RADIUS)],
        )

        report = self._refine(flat)

        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "refine_skipped:no-bulge")
        self.assertIsNone(report.wrist_to)

    def test_a_bulge_too_shallow_to_be_a_hand_is_skipped(self):
        report = self._refine(_arm_cloud(palm_radius=_REFINE_FOREARM_RADIUS * 1.15))

        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "refine_skipped:no-bulge")
        self.assertLess(report.bulge_ratio, 1.25)

    def test_a_missing_joint_or_a_zero_length_segment_is_skipped(self):
        names, joints = _refine_fixture()
        del names[names.index("lower_arm_l")]
        joints = np.delete(joints, _BUILD_FIXTURE_NAMES.index("lower_arm_l"), axis=0)
        self.assertEqual(
            refine_distal_anchor(joints, names, _arm_cloud(), ("lower_arm_l", "hand_l")).reason,
            "refine_skipped:missing-joint",
        )

        names, joints = _refine_fixture(hand_offset=0.0)
        self.assertEqual(
            refine_distal_anchor(joints, names, _arm_cloud(), ("lower_arm_l", "hand_l")).reason,
            "refine_skipped:degenerate-axis",
        )

    def test_no_mesh_and_no_nearby_geometry_are_both_skipped(self):
        self.assertEqual(self._refine(None).reason, "refine_skipped:no-mesh")
        self.assertEqual(self._refine(np.zeros((0, 3))).reason, "refine_skipped:no-mesh")
        far = _arm_cloud(origin=_REFINE_ELBOW_L + np.array([0.0, 5.0, 0.0]))
        self.assertEqual(self._refine(far).reason, "refine_skipped:no-region-vertices")

    def test_a_flat_skeleton_cannot_size_its_own_region(self):
        names, joints = _refine_fixture()
        joints[:, VERTICAL_AXIS] = 1.35

        report = refine_distal_anchor(joints, names, _arm_cloud(), ("lower_arm_l", "hand_l"))

        self.assertEqual(report.reason, "refine_skipped:degenerate-height")

    def test_an_anchor_already_at_the_wrist_is_reported_not_nudged(self):
        names, joints = _refine_fixture(hand_offset=_REFINE_FOREARM_SPAN)

        report = refine_distal_anchor(
            joints, names, _arm_cloud(), ("lower_arm_l", "hand_l"), mirror_anchor="hand_r",
        )

        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "refine_skipped:already-placed")
        # It is still a measurement, not a non-event: the bulge was found.
        self.assertGreater(report.bulge_ratio, 1.5)

    def test_a_move_past_the_guard_rail_is_refused_rather_than_clamped(self):
        # Clamping would land the anchor somewhere neither the fitter nor
        # the detector believes in.
        names, joints = _refine_fixture(hand_offset=0.02)

        report = refine_distal_anchor(
            joints, names, _arm_cloud(), ("lower_arm_l", "hand_l"),
            mirror_anchor="hand_r", max_move_fraction=0.4,
        )

        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "refine_skipped:move-exceeds-guard")
        self.assertGreater(abs(report.moved), 0.4 * report.limb_extent)

    def test_a_confidence_floor_above_the_measurement_refuses_it(self):
        report = self._refine(_arm_cloud(), min_confidence=1.01)

        self.assertFalse(report.applied)
        self.assertEqual(report.reason, "refine_skipped:low-confidence")

    def test_a_limb_that_never_ends_inside_the_region_is_refused(self):
        long_arm = _tube_cloud(
            _REFINE_ELBOW_L, (1.0, 0.0, 0.0),
            [(_REFINE_FOREARM_SPAN, _REFINE_FOREARM_RADIUS),
             (_REFINE_PALM_SPAN, _REFINE_PALM_RADIUS),
             (3.0, _REFINE_FINGER_RADIUS)],
        )

        report = self._refine(long_arm)

        self.assertEqual(report.reason, "refine_skipped:region-saturated")


class RefineDistalAnchorsTests(unittest.TestCase):
    def setUp(self):
        self.names, self.joints = _refine_fixture()
        self.cloud = _arm_cloud()

    def test_only_the_measured_limb_moves_and_the_input_is_not_mutated(self):
        before = self.joints.copy()

        refined, reports = refine_distal_anchors(self.joints, self.names, self.cloud)

        np.testing.assert_array_equal(self.joints, before)
        moved = [self.names[i] for i in np.flatnonzero(
            np.linalg.norm(refined - self.joints, axis=1) > 1e-12
        )]
        self.assertEqual(moved, ["hand_l"])
        by_anchor = {report.anchor: report for report in reports}
        self.assertEqual(sorted(by_anchor), ["foot_l", "foot_r", "hand_l", "hand_r"])
        self.assertTrue(by_anchor["hand_l"].applied)
        for anchor in ("hand_r", "foot_l", "foot_r"):
            self.assertFalse(by_anchor[anchor].applied, anchor)
            self.assertEqual(
                by_anchor[anchor].reason, "refine_skipped:no-region-vertices", anchor,
            )

    def test_a_run_with_nothing_to_refine_leaves_the_rig_byte_for_byte(self):
        refined, reports = refine_distal_anchors(self.joints, self.names, None)

        np.testing.assert_array_equal(refined, self.joints)
        self.assertTrue(all(report.reason == "refine_skipped:no-mesh" for report in reports))

    def test_refinement_is_idempotent(self):
        once, _ = refine_distal_anchors(self.joints, self.names, self.cloud)
        twice, reports = refine_distal_anchors(once, self.names, self.cloud)

        np.testing.assert_allclose(twice, once, atol=1e-12)
        self.assertEqual(
            {report.anchor for report in reports if report.applied}, set(),
        )

    def test_a_hand_placed_joint_is_never_argued_with(self):
        refined, reports = refine_distal_anchors(
            self.joints, self.names, self.cloud, protected={"hand_l"},
        )

        np.testing.assert_array_equal(refined, self.joints)
        by_anchor = {report.anchor: report for report in reports}
        self.assertEqual(by_anchor["hand_l"].reason, "refine_skipped:manual-override")

    def test_the_function_is_pure_across_repeated_calls(self):
        first, first_reports = refine_distal_anchors(self.joints, self.names, self.cloud)
        second, second_reports = refine_distal_anchors(self.joints, self.names, self.cloud)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            [distal_refinement_json(r) for r in first_reports],
            [distal_refinement_json(r) for r in second_reports],
        )

    def test_the_json_report_names_every_limb_and_its_disposition(self):
        _, reports = refine_distal_anchors(self.joints, self.names, self.cloud)
        payload = [distal_refinement_json(report) for report in reports]

        self.assertEqual([entry["anchor"] for entry in payload],
                         ["hand_l", "hand_r", "foot_l", "foot_r"])
        hand = payload[0]
        self.assertEqual(hand["status"], "refined")
        self.assertIn("medial_offset", hand)
        self.assertEqual(len(hand["wrist_from"]), 3)
        self.assertEqual(len(hand["wrist_to"]), 3)
        self.assertGreater(hand["moved"], 0.0)
        self.assertEqual(payload[3]["wrist_to"], None)


class RefinedAnchorGrowsFingersIntoTheHandTests(unittest.TestCase):
    """The point of the whole exercise: the finger chains that
    `build_template_conformant_skeleton` grows must land in the mesh's own
    hand, not in the distal forearm."""

    def setUp(self):
        self.template = load_template("biped")
        self.names, self.joints = _refine_fixture()
        self.cloud = _arm_cloud()
        self.palm_start = float(_REFINE_ELBOW_L[0]) + _REFINE_FOREARM_SPAN
        self.mesh_tip = float(_REFINE_ELBOW_L[0]) + _REFINE_LIMB_EXTENT

    def _build(self, joints):
        build = build_template_conformant_skeleton(
            self.names, joints, self.template, mesh_vertices=self.cloud,
        )
        return build, dict(zip(build.names, range(len(build.names))))

    def test_without_refinement_the_knuckles_articulate_the_forearm(self):
        # The bead's actual symptom: a chain rooted at a wrist fitted
        # mid-forearm puts every knuckle in the forearm, whatever the tips
        # are then stretched to reach.
        build, index = self._build(self.joints)

        for root in _LEFT_FINGER_ROOTS:
            self.assertLess(float(build.heads[index[root]][0]), self.palm_start, root)

    def test_with_refinement_the_knuckles_and_tips_are_both_in_the_mesh_hand(self):
        refined, _ = refine_distal_anchors(self.joints, self.names, self.cloud)
        build, index = self._build(refined)

        for root in _LEFT_FINGER_ROOTS:
            self.assertGreater(float(build.heads[index[root]][0]), self.palm_start, root)
        for name in _LEFT_FINGER_TIPS:
            tip = float(build.heads[index[name]][0])
            self.assertGreater(tip, self.palm_start, name)
            self.assertLess(tip, self.mesh_tip, name)

    def test_the_refined_chain_measures_the_hand_instead_of_saturating(self):
        refined, _ = refine_distal_anchors(self.joints, self.names, self.cloud)
        build, _ = self._build(refined)
        sources = {root: source for root, _, source in build.optional_chain_scales}

        for root in ("index_1_l", "middle_1_l", "ring_1_l", "pinky_1_l"):
            self.assertEqual(sources[root], "measured-mesh", root)

    def test_the_refined_rig_is_still_template_conformant(self):
        refined, _ = refine_distal_anchors(self.joints, self.names, self.cloud)
        build, _ = self._build(refined)

        findings = conformance_report(
            build.names, build.parents, build.heads, build.tails, build.rolls,
            family="biped", template=self.template,
        )

        self.assertEqual([finding["code"] for finding in findings], [])


if __name__ == "__main__":
    unittest.main()
