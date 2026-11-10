import math
import unittest

import numpy as np

from backend.app.retarget_math import (
    basis_from_world,
    bvh_export_frame_range,
    desired_target_world_pose,
    propagate_unmapped,
    quat_align_frame,
    quat_between_vectors,
    quat_clamp_angle,
    quat_from_axis_angle,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    quat_rotation_angle,
    vector_angle_deg,
)

# Reference "along the bone" axis in every bone's own local rest frame,
# matching the convention used by the diagnostics and blender_worker (a
# bone's rest direction is whatever its local +Y axis points to once its
# rest quaternion is applied).
BONE_AXIS = (0.0, 1.0, 0.0)


def assert_quat_close(case, actual, expected, atol=1e-6):
    """Unit quaternions q and -q represent the same rotation."""
    a, e = np.asarray(actual), np.asarray(expected)
    case.assertTrue(
        np.allclose(a, e, atol=atol) or np.allclose(a, -e, atol=atol),
        f"{actual} != {expected} (up to sign)",
    )


def assert_vec_close(case, actual, expected, atol=1e-6):
    case.assertTrue(
        np.allclose(np.asarray(actual), np.asarray(expected), atol=atol),
        f"{actual} != {expected}",
    )


class QuaternionPrimitiveTests(unittest.TestCase):
    def test_multiply_identity_is_noop(self):
        q = quat_normalize((0.7, 0.1, 0.2, 0.3))
        assert_quat_close(self, quat_multiply(q, quat_identity()), q)
        assert_quat_close(self, quat_multiply(quat_identity(), q), q)

    def test_inverse_undoes_rotation(self):
        q = quat_from_axis_angle((0.0, 0.0, 1.0), 1.234)
        assert_quat_close(self, quat_multiply(q, quat_inverse(q)), quat_identity())

    def test_rotate_vector_matches_axis_angle(self):
        q = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
        rotated = quat_rotate_vector(q, (1.0, 0.0, 0.0))
        assert_vec_close(self, rotated, (0.0, 1.0, 0.0), atol=1e-6)

    def test_quat_between_vectors_maps_source_onto_target(self):
        q = quat_between_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert_vec_close(self, quat_rotate_vector(q, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    def test_quat_between_vectors_handles_opposite_vectors(self):
        q = quat_between_vectors((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
        assert_vec_close(self, quat_rotate_vector(q, (1.0, 0.0, 0.0)), (-1.0, 0.0, 0.0))

    def test_quat_between_vectors_is_stable_near_180_degrees(self):
        # A near-but-not-exact 180 degree case is the ill-conditioned branch
        # (dot ~ -1, cross product ~ zero-length) -- must still return a
        # valid, finite, unit rotation that closely maps v_from onto v_to,
        # not NaNs or a wildly wrong axis.
        almost_opposite = (-1.0, 0.02, 0.0)
        q = quat_between_vectors((1.0, 0.0, 0.0), almost_opposite)
        self.assertTrue(np.all(np.isfinite(np.asarray(q))))
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(q))), 1.0, places=6)
        rotated = quat_rotate_vector(q, (1.0, 0.0, 0.0))
        target_dir = np.asarray(almost_opposite) / np.linalg.norm(almost_opposite)
        self.assertLess(vector_angle_deg(rotated, target_dir), 1e-3)

    def test_vector_angle_deg(self):
        self.assertAlmostEqual(vector_angle_deg((1, 0, 0), (0, 1, 0)), 90.0, places=6)
        self.assertAlmostEqual(vector_angle_deg((1, 0, 0), (1, 0, 0)), 0.0, places=6)


class QuatClampAngleTests(unittest.TestCase):
    def test_clamp_leaves_small_angle_unchanged(self):
        q = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(10.0))
        assert_quat_close(self, quat_clamp_angle(q, math.radians(30.0)), q)

    def test_clamp_reduces_angle_keeping_axis(self):
        axis = (0.0, 1.0, 0.0)
        q = quat_from_axis_angle(axis, math.radians(50.0))
        clamped = quat_clamp_angle(q, math.radians(30.0))
        self.assertAlmostEqual(math.degrees(quat_rotation_angle(clamped)), 30.0, places=5)
        # Same rotation axis: rotating the axis vector itself is a no-op for
        # both the original and the clamped quaternion.
        assert_vec_close(self, quat_rotate_vector(clamped, axis), axis, atol=1e-6)

    def test_clamp_to_zero_gives_identity(self):
        q = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(45.0))
        assert_quat_close(self, quat_clamp_angle(q, 0.0), quat_identity())

    def test_clamp_identity_stays_identity(self):
        assert_quat_close(self, quat_clamp_angle(quat_identity(), math.radians(30.0)), quat_identity())


class IdenticalSkeletonTests(unittest.TestCase):
    """A2 test 1: identical source/target rest -> basis equals the source
    local delta (the world-delta transfer is a no-op relative to Blender's
    own rest-relative pose composition when both skeletons agree)."""

    def test_identical_rest_gives_source_local_delta_as_basis(self):
        rest = quat_from_axis_angle((0.0, 0.0, 1.0), 0.4)
        local_delta = quat_from_axis_angle((1.0, 0.0, 0.0), 0.3)
        source_rest_w = rest
        target_rest_w = rest  # identical skeletons
        source_pose_w = quat_multiply(source_rest_w, local_delta)

        source_dir = quat_rotate_vector(source_rest_w, BONE_AXIS)
        target_dir = quat_rotate_vector(target_rest_w, BONE_AXIS)
        align = quat_between_vectors(target_dir, source_dir)
        assert_quat_close(self, align, quat_identity())

        target_pose_w = desired_target_world_pose(
            source_pose_w, source_rest_w, align, target_rest_w,
        )
        # Root bone: virtual parent rest/pose is the (identity) armature object.
        basis = basis_from_world(
            target_rest_w, quat_identity(), quat_identity(), target_pose_w,
        )
        assert_quat_close(self, basis, local_delta)


class RollInvarianceTests(unittest.TestCase):
    """A2 test 2: rolling the target bone about its own rest axis must not
    change the transferred world direction."""

    def test_target_roll_does_not_perturb_transferred_direction(self):
        axis_dir = np.array([1.0, 1.0, 0.0])
        axis_dir /= np.linalg.norm(axis_dir)
        base = quat_between_vectors(BONE_AXIS, axis_dir)
        source_rest_w = base
        swing = quat_from_axis_angle((0.0, 0.0, 1.0), 0.9)
        source_pose_w = quat_multiply(swing, source_rest_w)

        directions = []
        for roll_degrees in (0.0, 90.0, 180.0):
            roll = quat_from_axis_angle(BONE_AXIS, math.radians(roll_degrees))
            target_rest_w = quat_multiply(base, roll)
            # Rest directions still agree (rolling about the bone's own axis
            # cannot change where that axis points), so alignment is a no-op.
            source_dir = quat_rotate_vector(source_rest_w, BONE_AXIS)
            target_dir = quat_rotate_vector(target_rest_w, BONE_AXIS)
            assert_vec_close(self, target_dir, source_dir, atol=1e-6)
            align = quat_between_vectors(target_dir, source_dir)

            target_pose_w = desired_target_world_pose(
                source_pose_w, source_rest_w, align, target_rest_w,
            )
            directions.append(quat_rotate_vector(target_pose_w, BONE_AXIS))

        for direction in directions[1:]:
            assert_vec_close(self, direction, directions[0], atol=1e-6)


class RestAlignmentTests(unittest.TestCase):
    """A2 test 3: an A-posed target against a T-posed source -- at the
    source's own rest frame the target must point along the SOURCE's rest
    direction, not its own (drooped) rest direction."""

    def test_target_points_along_source_rest_direction_at_source_rest(self):
        source_dir = np.array([1.0, 0.0, 0.0])  # T-pose: arm straight out
        droop = math.radians(15.0)
        target_dir = np.array([math.cos(droop), 0.0, -math.sin(droop)])  # A-pose droop

        source_rest_w = quat_between_vectors(BONE_AXIS, source_dir)
        target_rest_w = quat_between_vectors(BONE_AXIS, target_dir)
        align = quat_between_vectors(target_dir, source_dir)

        # Source at its own rest: S_pose_w == S_rest_w.
        target_pose_w = desired_target_world_pose(
            source_rest_w, source_rest_w, align, target_rest_w,
        )
        resulting_dir = quat_rotate_vector(target_pose_w, BONE_AXIS)
        assert_vec_close(self, resulting_dir, source_dir, atol=1e-6)
        # Sanity: this is NOT the target's own native (A-pose) rest direction.
        self.assertGreater(vector_angle_deg(resulting_dir, target_dir), 5.0)


class HierarchyChainTests(unittest.TestCase):
    """A2 test 4: a two-level chain where the parent is driven by a mapped
    source (rotated 30 degrees) and the child is unmapped -- the child must
    rigidly follow the parent (identity local basis)."""

    def test_unmapped_child_follows_rotated_parent(self):
        parent_rest_w = quat_identity()
        child_rest_w = quat_identity()  # straight extension of the parent at rest
        parent_pose_w = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(30.0))

        child_pose_w = propagate_unmapped(parent_pose_w, parent_rest_w, child_rest_w)
        expected_dir = quat_rotate_vector(parent_pose_w, BONE_AXIS)
        assert_vec_close(self, quat_rotate_vector(child_pose_w, BONE_AXIS), expected_dir)

        # The child's local basis must round-trip to identity: it contributes
        # no rotation of its own, it only inherits the parent's.
        basis = basis_from_world(child_rest_w, parent_rest_w, parent_pose_w, child_pose_w)
        assert_quat_close(self, basis, quat_identity())


class TorsoAlignmentClampRegressionTests(unittest.TestCase):
    """Regression coverage for the Mary/Sword_Attack torso-hunch fix
    (backend/workers/blender_worker.py::retarget_bvh's spine/head alignment
    clamp, tightened from 30 to 5 degrees).

    The rest-direction mismatches below are the ACTUAL values measured
    between Mary's rig (studio-data/artifacts/8e416b46...) and the SOMA
    donor rig for the Sword_Attack clip (see /tmp/k6/rest_quats.json and
    validate_align.py on the box; the investigation is summarized in the
    blender_worker.py comment above the alignment loop). All of them sit
    UNDER the previous 30-degree clamp -- meaning that clamp was a no-op
    for this rig pair and the full (uncapped) rest-direction alignment was
    baked into every frame, copying the donor's own absolute torso lean
    onto Mary's spine/neck/head instead of a bounded correction."""

    MEASURED_MISMATCH_DEG = {
        "Chest->spine_3": 23.63,
        "Neck1->neck": 20.51,
        "Spine1->spine_1": 10.85,
        "Spine2->spine_2": 4.88,
        "Head->head": 3.85,
    }

    def test_previous_30_degree_clamp_was_a_no_op_for_every_measured_bone(self):
        for label, mismatch_deg in self.MEASURED_MISMATCH_DEG.items():
            full_align = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(mismatch_deg))
            clamped_30 = quat_clamp_angle(full_align, math.radians(30.0))
            assert_quat_close(self, clamped_30, full_align, atol=1e-6)

    def test_reduced_5_degree_clamp_bounds_the_large_mismatches(self):
        for label, mismatch_deg in self.MEASURED_MISMATCH_DEG.items():
            full_align = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(mismatch_deg))
            clamped_5 = quat_clamp_angle(full_align, math.radians(5.0))
            resulting_deg = math.degrees(quat_rotation_angle(clamped_5))
            if mismatch_deg > 5.0:
                self.assertAlmostEqual(resulting_deg, 5.0, places=5, msg=label)
            else:
                # Already under the new bound -- unaffected (Spine2, Head).
                self.assertAlmostEqual(resulting_deg, mismatch_deg, places=5, msg=label)

    def test_5_degree_clamp_still_transfers_a_small_genuine_bend(self):
        # A real (small) spine-curve difference between the two rigs should
        # still pass through mostly intact under the tighter bound -- the
        # clamp is there to cap the donor's absolute-lean convention, not
        # to zero out every correction.
        small_bend = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(2.0))
        clamped = quat_clamp_angle(small_bend, math.radians(5.0))
        assert_quat_close(self, clamped, small_bend, atol=1e-6)


class QuatAlignFrameTests(unittest.TestCase):
    """Two-vector ("frame") limb rest alignment (BOU elbow-twist fix): the
    swing-only alignment `quat_between_vectors` leaves the rotation ABOUT
    the aligned bone axis arbitrary; `quat_align_frame` pins that residual
    roll using a secondary hint direction (a mapped child bone's own rest
    direction)."""

    def test_identity_when_frames_match(self):
        # Same bone direction AND same hint direction on both rigs -> no
        # correction needed at all.
        t_dir = (1.0, 0.2, 0.0)
        t_hint = (0.9, 0.1, 0.3)
        align = quat_align_frame(t_dir, t_hint, t_dir, t_hint)
        assert_quat_close(self, align, quat_identity())

    def test_pure_roll_when_directions_match_but_hints_differ_by_90_degrees(self):
        axis = np.array([1.0, 0.0, 0.0])
        # A hint perpendicular to the shared bone axis, then rotated 90
        # degrees about that axis -- the bone direction itself is
        # unchanged, only the hint's azimuth differs.
        s_hint = (0.0, 1.0, 0.0)
        rotate_90_about_axis = quat_from_axis_angle(axis, math.radians(90.0))
        t_hint = tuple(quat_rotate_vector(rotate_90_about_axis, np.array(s_hint)).tolist())

        align = quat_align_frame(tuple(axis), t_hint, tuple(axis), s_hint)
        self.assertAlmostEqual(math.degrees(quat_rotation_angle(align)), 90.0, places=3)
        # The rotation axis is the shared bone direction itself (a no-op
        # when applied to that axis).
        assert_vec_close(self, quat_rotate_vector(align, axis), axis, atol=1e-6)
        # And it actually carries the target hint onto the source hint.
        assert_vec_close(self, quat_rotate_vector(align, np.array(t_hint)), np.array(s_hint), atol=1e-6)

    def test_degenerate_hint_parallel_to_bone_falls_back_to_swing_only(self):
        # A straight-limbed T-pose rest: the "hint" (e.g. a forearm) points
        # along the SAME direction as its own bone (e.g. an upper arm) on
        # both rigs, so it carries no roll information.
        t_dir = (1.0, 0.3, 0.0)
        s_dir = (1.0, 0.0, 0.2)
        t_hint = t_dir  # exactly parallel -- the clearest degenerate case
        s_hint = s_dir
        align = quat_align_frame(t_dir, t_hint, s_dir, s_hint)
        expected = quat_between_vectors(t_dir, s_dir)
        assert_quat_close(self, align, expected)

    def test_near_degenerate_hint_below_threshold_falls_back_to_swing_only(self):
        # A hint just a couple of degrees off its own bone's direction --
        # below the default 5-degree trust threshold -- must still fall
        # back rather than trust a near-zero-length perpendicular component.
        t_dir = (1.0, 0.0, 0.0)
        s_dir = (1.0, 0.0, 0.0)
        tiny_deg = 2.0
        t_hint = (math.cos(math.radians(tiny_deg)), math.sin(math.radians(tiny_deg)), 0.0)
        s_hint = (math.cos(math.radians(tiny_deg)), 0.0, math.sin(math.radians(tiny_deg)))
        align = quat_align_frame(t_dir, t_hint, s_dir, s_hint, degenerate_deg=5.0)
        expected = quat_between_vectors(t_dir, s_dir)
        assert_quat_close(self, align, expected)

    def test_bone_direction_always_maps_onto_source_direction(self):
        # Regardless of the hint, the primary constraint (swing) must
        # always hold: the target bone direction ends up exactly on the
        # source bone direction.
        t_dir = np.array([0.3, 0.7, 0.1])
        s_dir = np.array([-0.2, 0.5, 0.6])
        t_hint = np.array([0.1, -0.2, 0.9])
        s_hint = np.array([0.4, 0.3, 0.7])
        align = quat_align_frame(tuple(t_dir), tuple(t_hint), tuple(s_dir), tuple(s_hint))
        resulting = quat_rotate_vector(align, t_dir / np.linalg.norm(t_dir))
        expected = s_dir / np.linalg.norm(s_dir)
        assert_vec_close(self, resulting, expected, atol=1e-6)


class BvhExportFrameRangeTests(unittest.TestCase):
    """K17: normalize_bvh_axes's bpy.ops.export_anim.bvh call exports the
    scene's current frame range, not the imported action's -- a shorter
    Kimodo clip used to get padded to Blender's untouched default scene
    frame_end (250) with its last real frame held static the rest of the
    way. bvh_export_frame_range is the frame-range math that call now
    seeds the scene with first, extracted here so it's testable without a
    real Blender/bpy install.
    """

    def test_exact_integer_range_is_unchanged(self):
        self.assertEqual(bvh_export_frame_range((1.0, 60.0)), (1, 60))

    def test_matches_the_reported_kimodo_case(self):
        # 2s request @30fps honored as ~60 real frames (1-indexed BVH import
        # frame_start=1), not the 250-frame padded clip that used to export.
        self.assertEqual(bvh_export_frame_range((1.0, 61.0)), (1, 61))

    def test_fractional_range_floors_start_and_ceils_end(self):
        # A bake that starts or ends mid-frame must never be truncated.
        self.assertEqual(bvh_export_frame_range((1.4, 60.1)), (1, 61))
        self.assertEqual(bvh_export_frame_range((0.9, 59.9)), (0, 60))

    def test_single_frame_action(self):
        self.assertEqual(bvh_export_frame_range((1.0, 1.0)), (1, 1))

    def test_negative_start_is_preserved_not_clamped(self):
        # This function only computes the range; clamping to a sane minimum,
        # if ever needed, is the caller's call -- not assumed here.
        self.assertEqual(bvh_export_frame_range((-2.0, 10.0)), (-2, 10))


if __name__ == "__main__":
    unittest.main()
