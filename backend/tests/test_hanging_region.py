"""Tests for the hanging-region weighting post-pass.

The failure being fixed: long geometry that hangs below the hips instead of
sitting on a leg (a robe/skirt shell, a coat tail, the fabric web bridging
the two legs at the crotch) has no skeletal representation, so bone-heat --
a nearest-wins scheme -- partitions it into rigid left/right halves with a
knife-edge boundary down the midline. Any scissoring pose then shears the
two halves in opposite directions: the shell collapses into a flat vertical
plank below the hem.

Synthetic fixtures, both built around the same leg skeleton:

* a **skirt**: one conical shell from the hips to the floor, enclosing both
  legs, with no leg geometry modelled at all (the real case measured on a
  floor-length robe).
* **trousers**: two separate leg cylinders, the fitted case that must come
  out a near no-op.
"""
import unittest

import numpy as np

from backend.app.skeleton_naming import (
    chain_segment_distance,
    detect_hanging_region,
    hanging_region_blend_weights,
    midline_lateral_gap,
    point_segment_distance,
)

UP = 2
LATERAL = 0
HIP_HEIGHT = 1.0
HIP_HALF_SPAN = 0.1
KNEE_HEIGHT = 0.5
ANKLE_HEIGHT = 0.08
FLOOR = 0.0

LEFT_CHAIN = [
    (np.array([0.1, 0.0, HIP_HEIGHT]), np.array([0.1, 0.0, KNEE_HEIGHT])),
    (np.array([0.1, 0.0, KNEE_HEIGHT]), np.array([0.1, 0.0, ANKLE_HEIGHT])),
    (np.array([0.1, 0.0, ANKLE_HEIGHT]), np.array([0.1, 0.06, FLOOR])),
]
RIGHT_CHAIN = [
    (np.array([-0.1, 0.0, HIP_HEIGHT]), np.array([-0.1, 0.0, KNEE_HEIGHT])),
    (np.array([-0.1, 0.0, KNEE_HEIGHT]), np.array([-0.1, 0.0, ANKLE_HEIGHT])),
    (np.array([-0.1, 0.0, ANKLE_HEIGHT]), np.array([-0.1, 0.06, FLOOR])),
]

PELVIS, THIGH_L, THIGH_R, CALF_L, CALF_R, FOOT_L, FOOT_R = 0, 1, 2, 3, 4, 5, 6
BONE_COUNT = 7


def cylinder(centre_x, radius, *, top, bottom, rings=40, segments=64):
    """A closed-ish tube of vertices around a vertical axis."""
    points = []
    for ring in range(rings):
        z = bottom + (top - bottom) * ring / (rings - 1)
        for segment in range(segments):
            angle = 2 * np.pi * segment / segments
            points.append([centre_x + radius * np.cos(angle), radius * np.sin(angle), z])
    return np.asarray(points, dtype=np.float64)


def skirt_shell(*, top=HIP_HEIGHT, bottom=FLOOR, top_radius=0.16, bottom_radius=0.34):
    """A cone widening towards the floor, centred between the legs: it
    encloses both leg axes at every height and models no leg surface."""
    # Mesh density, not a wireframe: `midline_lateral_gap` bins each height
    # band's lateral coordinate, so a sparse ring would read as a corridor
    # everywhere. Real canonicalized meshes carry thousands of vertices per
    # band; these fixtures carry hundreds.
    points = []
    rings, segments = 48, 96
    for ring in range(rings):
        t = ring / (rings - 1)
        z = top + (bottom - top) * t
        radius = top_radius + (bottom_radius - top_radius) * t
        for segment in range(segments):
            angle = 2 * np.pi * segment / segments
            points.append([radius * np.cos(angle), radius * np.sin(angle), z])
    return np.asarray(points, dtype=np.float64)


def trousers():
    """Two separate leg tubes, each snug around its own bone axis."""
    return np.concatenate([
        cylinder(0.1, 0.07, top=HIP_HEIGHT, bottom=FLOOR),
        cylinder(-0.1, 0.07, top=HIP_HEIGHT, bottom=FLOOR),
    ], axis=0)


def detect(vertices, **overrides):
    kwargs = dict(
        up_axis=UP,
        up_sign=1.0,
        lateral_axis=LATERAL,
        hip_height=HIP_HEIGHT,
        hip_half_span=HIP_HALF_SPAN,
        hip_midline=0.0,
        left_segments=LEFT_CHAIN,
        right_segments=RIGHT_CHAIN,
    )
    kwargs.update(overrides)
    return detect_hanging_region(vertices, **kwargs)


def nearest_wins_weights(vertices):
    """The binding this pass exists to replace: every vertex fully owned by
    whichever of the two leg chains is closer, thigh above the knee and calf
    below it. One influence, hard midline boundary."""
    left = chain_segment_distance(vertices, LEFT_CHAIN)
    right = chain_segment_distance(vertices, RIGHT_CHAIN)
    is_left = left <= right
    is_calf = vertices[:, UP] < KNEE_HEIGHT
    bone = np.where(
        is_calf,
        np.where(is_left, CALF_L, CALF_R),
        np.where(is_left, THIGH_L, THIGH_R),
    )
    bone_indices = np.full((vertices.shape[0], 4), -1, dtype=np.int64)
    weights = np.zeros((vertices.shape[0], 4), dtype=np.float64)
    bone_indices[:, 0] = bone
    weights[:, 0] = 1.0
    return bone_indices, weights


def blend(vertices, region, bone_indices, weights):
    return hanging_region_blend_weights(
        vertices,
        region,
        bone_indices,
        weights,
        up_axis=UP,
        up_sign=1.0,
        lateral_axis=LATERAL,
        hip_height=HIP_HEIGHT,
        hip_half_span=HIP_HALF_SPAN,
        hip_midline=0.0,
        lateral_left=0.1,
        lateral_right=-0.1,
        knee_height=KNEE_HEIGHT,
        ankle_height=ANKLE_HEIGHT,
        pelvis_bone=PELVIS,
        thigh_left_bone=THIGH_L,
        thigh_right_bone=THIGH_R,
        calf_left_bone=CALF_L,
        calf_right_bone=CALF_R,
        foot_left_bone=FOOT_L,
        foot_right_bone=FOOT_R,
    )


def dense(bone_indices, weights, bone_count=BONE_COUNT):
    out = np.zeros((bone_indices.shape[0], bone_count))
    for slot in range(bone_indices.shape[1]):
        column = bone_indices[:, slot]
        valid = column >= 0
        np.add.at(out, (np.nonzero(valid)[0], column[valid]), weights[valid, slot])
    return out


class PointSegmentDistanceTests(unittest.TestCase):
    def test_a_point_beside_the_segment_measures_its_perpendicular_offset(self):
        d = point_segment_distance(np.array([[0.3, 0.0, 0.5]]), *LEFT_CHAIN[1])
        self.assertAlmostEqual(float(d[0]), 0.2)

    def test_a_point_past_the_segment_end_measures_to_that_end_not_the_line(self):
        # LEFT_CHAIN[1] is knee -> ankle; a point below the ankle measures to
        # the ankle end, not to the infinite line through the shin.
        d = point_segment_distance(np.array([[0.1, 0.0, -0.4]]), *LEFT_CHAIN[1])
        self.assertAlmostEqual(float(d[0]), ANKLE_HEIGHT + 0.4)

    def test_a_degenerate_segment_is_treated_as_its_head_point(self):
        head = np.array([0.0, 0.0, 0.0])
        d = point_segment_distance(np.array([[0.0, 0.0, 3.0]]), head, head)
        self.assertAlmostEqual(float(d[0]), 3.0)

    def test_a_chain_reports_the_nearest_of_its_segments(self):
        d = chain_segment_distance(np.array([[0.1, 0.0, 0.25]]), LEFT_CHAIN)
        self.assertAlmostEqual(float(d[0]), 0.0)

    def test_an_empty_chain_is_infinitely_far_rather_than_an_error(self):
        d = chain_segment_distance(np.array([[0.0, 0.0, 0.0]]), [])
        self.assertTrue(np.isinf(d[0]))


class MidlineLateralGapTests(unittest.TestCase):
    def test_two_separated_limbs_leave_a_corridor_across_the_midline(self):
        lateral = np.concatenate([
            np.linspace(-0.17, -0.03, 200), np.linspace(0.03, 0.17, 200)
        ])
        self.assertGreater(midline_lateral_gap(lateral, HIP_HALF_SPAN), 0.4)

    def test_an_enclosing_shell_leaves_no_corridor(self):
        angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        self.assertEqual(midline_lateral_gap(0.3 * np.cos(angles), HIP_HALF_SPAN), 0.0)

    def test_a_gap_far_off_the_midline_is_not_counted(self):
        lateral = np.concatenate([np.linspace(-0.2, 0.2, 400), np.linspace(0.9, 1.0, 100)])
        self.assertEqual(midline_lateral_gap(lateral, HIP_HALF_SPAN), 0.0)

    def test_a_handful_of_stray_interior_points_cannot_fill_the_corridor(self):
        lateral = np.concatenate([
            np.linspace(-0.17, -0.03, 2000),
            np.linspace(0.03, 0.17, 2000),
            np.array([0.0, 0.001, -0.001]),
        ])
        self.assertGreater(midline_lateral_gap(lateral, HIP_HALF_SPAN), 0.4)

    def test_too_few_points_to_judge_reports_no_corridor(self):
        self.assertEqual(midline_lateral_gap(np.array([0.0, 1.0]), HIP_HALF_SPAN), 0.0)


class DetectHangingRegionTests(unittest.TestCase):
    def test_a_skirt_is_detected_as_one_enclosing_shell(self):
        region = detect(skirt_shell())
        self.assertEqual(region.mode, "shell")
        self.assertGreater(region.enclosure_depth, 0.9)
        # No band leaves a corridor across the midline: the shell's front
        # and back walls both cross it at every height.
        self.assertTrue(all(gap < 0.35 for gap in region.band_gaps), region.band_gaps)

    def test_a_skirt_claims_essentially_all_of_its_own_geometry(self):
        vertices = skirt_shell()
        region = detect(vertices)
        self.assertGreater(int(region.mask.sum()), 0.85 * vertices.shape[0])

    def test_trousers_are_a_near_no_op(self):
        vertices = trousers()
        region = detect(vertices)
        # Separated legs leave a corridor in every band, so the gapless run
        # down from the hips is empty and the pass has nothing to do at all.
        self.assertIn(region.mode, ("none", "fitted"))
        self.assertLess(region.enclosure_depth, 0.5)
        self.assertLess(int(region.mask.sum()), 0.02 * vertices.shape[0])
        self.assertTrue(all(gap > 0.35 for gap in region.band_gaps) or not region.band_gaps)

    def test_the_detector_never_claims_an_ordinary_inner_thigh(self):
        # The inner face of the left trouser leg: as close to its own bone
        # as any other leg vertex, and the classic false positive for a
        # "roughly between the legs" test.
        vertices = trousers()
        inner = (
            (np.abs(vertices[:, 1]) < 0.02)
            & (vertices[:, 0] > 0.0)
            & (vertices[:, 0] < 0.1)
            & (vertices[:, 2] > 0.1)
            & (vertices[:, 2] < 0.9)
        )
        self.assertGreater(int(inner.sum()), 10)
        region = detect(vertices)
        self.assertEqual(int(np.count_nonzero(region.mask & inner)), 0)

    def test_a_knee_length_skirt_stops_at_its_hem(self):
        # Bare separated legs below the hem end the gapless run there, so
        # only the skirt itself is claimed.
        vertices = np.concatenate([
            skirt_shell(top=HIP_HEIGHT, bottom=0.55, top_radius=0.16, bottom_radius=0.28),
            trousers(),
        ], axis=0)
        region = detect(vertices)
        claimed = vertices[region.mask]
        self.assertGreater(int(region.mask.sum()), 100)
        self.assertGreater(float(claimed[:, UP].min()), 0.4)

    def test_strength_fades_out_towards_the_hips_rather_than_stepping(self):
        vertices = skirt_shell()
        region = detect(vertices)
        near_hip = region.strength[vertices[:, UP] > 0.98]
        mid = region.strength[np.abs(vertices[:, UP] - 0.5) < 0.03]
        self.assertLess(float(near_hip.max()), 0.35)
        self.assertGreater(float(mid.min()), 0.95)

    def test_strength_never_leaves_the_unit_interval(self):
        region = detect(skirt_shell())
        self.assertGreaterEqual(float(region.strength.min()), 0.0)
        self.assertLessEqual(float(region.strength.max()), 1.0)
        self.assertTrue(np.array_equal(region.mask, region.strength > 1e-6))

    def test_geometry_above_the_hips_is_never_claimed(self):
        vertices = np.concatenate([skirt_shell(), cylinder(0.0, 0.2, top=1.8, bottom=1.02)], axis=0)
        region = detect(vertices)
        self.assertEqual(int(np.count_nonzero(region.mask & (vertices[:, UP] >= HIP_HEIGHT))), 0)

    def test_an_empty_mesh_is_handled_without_claiming_anything(self):
        region = detect(np.zeros((0, 3)))
        self.assertEqual(region.mode, "none")
        self.assertEqual(int(region.mask.sum()), 0)

    def test_a_missing_leg_chain_disables_the_pass(self):
        region = detect(skirt_shell(), left_segments=[])
        self.assertEqual(region.mode, "none")

    def test_a_degenerate_hip_span_disables_the_pass(self):
        region = detect(skirt_shell(), hip_half_span=0.0)
        self.assertEqual(region.mode, "none")

    def test_detection_does_not_mutate_its_input(self):
        vertices = skirt_shell()
        before = vertices.copy()
        detect(vertices)
        self.assertTrue(np.array_equal(vertices, before))


class HangingRegionBlendTests(unittest.TestCase):
    def setUp(self):
        self.vertices = skirt_shell()
        self.region = detect(self.vertices)
        self.before_bones, self.before_weights = nearest_wins_weights(self.vertices)
        self.after_bones, self.after_weights = blend(
            self.vertices, self.region, self.before_bones, self.before_weights
        )
        self.after = dense(self.after_bones, self.after_weights)

    def test_every_blended_row_still_sums_to_one(self):
        totals = self.after_weights.sum(axis=1)
        self.assertTrue(np.allclose(totals, 1.0, atol=1e-9))

    def test_no_row_exceeds_four_influences(self):
        counts = np.count_nonzero(self.after_weights > 1e-9, axis=1)
        self.assertLessEqual(int(counts.max()), 4)

    def test_the_midline_of_the_hem_is_driven_equally_by_both_legs(self):
        # This is the whole fix: nearest-wins gives a midline hem vertex
        # 100% of one calf, so the two halves of the shell shear apart.
        hem = (self.vertices[:, UP] < 0.06) & (np.abs(self.vertices[:, LATERAL]) < 0.02)
        self.assertGreater(int(hem.sum()), 0)
        left = self.after[hem][:, [THIGH_L, CALF_L]].sum(axis=1)
        right = self.after[hem][:, [THIGH_R, CALF_R]].sum(axis=1)
        self.assertTrue(np.allclose(left, right, atol=0.05))
        self.assertGreater(float(left.min()), 0.3)

    def test_nearest_wins_gave_that_same_midline_a_single_leg(self):
        hem = (self.vertices[:, UP] < 0.06) & (np.abs(self.vertices[:, LATERAL]) < 0.02)
        before = dense(self.before_bones, self.before_weights)
        left = before[hem][:, [THIGH_L, CALF_L]].sum(axis=1)
        right = before[hem][:, [THIGH_R, CALF_R]].sum(axis=1)
        self.assertGreater(float(np.abs(left - right).min()), 0.99)

    def test_the_outer_edge_still_follows_the_leg_on_its_own_side(self):
        outer = (self.vertices[:, UP] < 0.06) & (self.vertices[:, LATERAL] > 0.3)
        self.assertGreater(int(outer.sum()), 0)
        left = self.after[outer][:, [THIGH_L, CALF_L]].sum(axis=1)
        right = self.after[outer][:, [THIGH_R, CALF_R]].sum(axis=1)
        self.assertTrue(np.all(left > right))

    def test_the_lateral_handover_is_monotone_and_gradual(self):
        # One ring of the hem: strength varies with height, so mixing rings
        # would compare vertices at different blend strengths.
        band = self.vertices[:, UP] < 0.005
        lateral = self.vertices[band][:, LATERAL]
        left = self.after[band][:, [THIGH_L, CALF_L]].sum(axis=1)
        order = np.argsort(lateral)
        self.assertTrue(np.all(np.diff(left[order]) > -1e-9))
        # Gradual, not a step: the handover spans a real width of the hem
        # rather than flipping at one vertex, which is what nearest-wins did.
        handover = lateral[(left > 0.3) & (left < 0.7)]
        self.assertGreater(int(handover.size), 0)
        self.assertGreater(float(np.ptp(handover)), 0.08)

    def test_the_upper_skirt_is_pelvis_dominant_and_the_hem_is_not(self):
        # The skirt rides the pelvis where it is worn and is handed to the
        # legs towards the hem. (The last hand's-breadth right under the
        # hip line is deliberately left alone -- that band is legitimately
        # gapless on every character, so the strength ramp fades out there.)
        upper = (self.vertices[:, UP] > 0.6) & (self.vertices[:, UP] < 0.8)
        hem = self.vertices[:, UP] < 0.06
        self.assertGreater(float(self.after[upper][:, PELVIS].mean()), 0.5)
        self.assertLess(float(self.after[hem][:, PELVIS].max()), 0.15)

    def test_the_hem_is_driven_by_the_calves_and_the_hip_band_by_the_thighs(self):
        hem = self.vertices[:, UP] < 0.06
        upper = (self.vertices[:, UP] > 0.7) & (self.vertices[:, UP] < 0.85)
        self.assertGreater(float(self.after[hem][:, [CALF_L, CALF_R]].sum(axis=1).mean()), 0.55)
        self.assertGreater(float(self.after[upper][:, [THIGH_L, THIGH_R]].sum(axis=1).mean()),
                           float(self.after[upper][:, [CALF_L, CALF_R]].sum(axis=1).mean()))

    def test_the_feet_keep_a_minority_share_of_the_lowest_hem(self):
        # A required bone left with no vertices at all fails the rig
        # certificate (`rig.required_joint_unused`), which is exactly what a
        # floor-length hem did to `foot_l`/`foot_r` before the foot level
        # existed -- but the feet swing hardest of any bone, so their share
        # stays a minority of the calves'.
        hem = self.vertices[:, UP] < 0.03
        foot = self.after[hem][:, [FOOT_L, FOOT_R]].sum(axis=1)
        calf = self.after[hem][:, [CALF_L, CALF_R]].sum(axis=1)
        self.assertGreater(float(foot.mean()), 0.05)
        self.assertLess(float(foot.mean()), float(calf.mean()))

    def test_a_rig_without_feet_still_produces_normalized_weights(self):
        vertices = skirt_shell()
        region = detect(vertices)
        bones, weights = nearest_wins_weights(vertices)
        out_bones, out_weights = hanging_region_blend_weights(
            vertices, region, bones, weights,
            up_axis=UP, up_sign=1.0, lateral_axis=LATERAL,
            hip_height=HIP_HEIGHT, hip_half_span=HIP_HALF_SPAN, hip_midline=0.0,
            lateral_left=0.1, lateral_right=-0.1,
            knee_height=KNEE_HEIGHT, ankle_height=ANKLE_HEIGHT,
            pelvis_bone=PELVIS, thigh_left_bone=THIGH_L, thigh_right_bone=THIGH_R,
            calf_left_bone=CALF_L, calf_right_bone=CALF_R,
        )
        self.assertTrue(np.allclose(out_weights.sum(axis=1), 1.0, atol=1e-9))
        self.assertEqual(int(np.count_nonzero(out_bones == FOOT_L)), 0)

    def test_vertices_outside_the_region_are_returned_untouched(self):
        vertices = np.concatenate([skirt_shell(), cylinder(0.0, 0.2, top=1.8, bottom=1.02)], axis=0)
        region = detect(vertices)
        bones, weights = nearest_wins_weights(vertices)
        out_bones, out_weights = blend(vertices, region, bones, weights)
        outside = ~region.mask
        self.assertTrue(np.array_equal(out_bones[outside], bones[outside]))
        self.assertTrue(np.array_equal(out_weights[outside], weights[outside]))

    def test_trousers_come_out_a_near_no_op(self):
        vertices = trousers()
        region = detect(vertices)
        bones, weights = nearest_wins_weights(vertices)
        out_bones, out_weights = blend(vertices, region, bones, weights)
        moved = np.abs(dense(out_bones, out_weights) - dense(bones, weights)).sum(axis=1)
        self.assertLess(int(np.count_nonzero(moved > 0.05)), 0.02 * vertices.shape[0])

    def test_an_unresolved_target_bone_hands_its_share_to_the_others(self):
        vertices = skirt_shell()
        region = detect(vertices)
        bones, weights = nearest_wins_weights(vertices)
        out_bones, out_weights = hanging_region_blend_weights(
            vertices, region, bones, weights,
            up_axis=UP, up_sign=1.0, lateral_axis=LATERAL,
            hip_height=HIP_HEIGHT, hip_half_span=HIP_HALF_SPAN, hip_midline=0.0,
            lateral_left=0.1, lateral_right=-0.1,
            knee_height=KNEE_HEIGHT, ankle_height=ANKLE_HEIGHT,
            pelvis_bone=-1, thigh_left_bone=THIGH_L, thigh_right_bone=THIGH_R,
            calf_left_bone=CALF_L, calf_right_bone=CALF_R,
            foot_left_bone=FOOT_L, foot_right_bone=FOOT_R,
        )
        self.assertTrue(np.allclose(out_weights.sum(axis=1), 1.0, atol=1e-9))
        self.assertEqual(int(np.count_nonzero(out_bones == PELVIS)), 0)

    def test_an_empty_region_returns_the_weights_unchanged(self):
        vertices = trousers()
        region = detect(vertices, hip_half_span=0.0)
        bones, weights = nearest_wins_weights(vertices)
        out_bones, out_weights = blend(vertices, region, bones, weights)
        self.assertTrue(np.array_equal(out_bones, bones))
        self.assertTrue(np.array_equal(out_weights, weights))

    def test_the_blend_does_not_mutate_its_inputs(self):
        bones, weights = nearest_wins_weights(self.vertices)
        bones_before, weights_before = bones.copy(), weights.copy()
        blend(self.vertices, self.region, bones, weights)
        self.assertTrue(np.array_equal(bones, bones_before))
        self.assertTrue(np.array_equal(weights, weights_before))

    def test_a_second_application_barely_moves_anything(self):
        # Not idempotent by construction -- the pass mixes the target into
        # whatever weights it is given, so re-running converges towards the
        # target rather than reproducing the previous result exactly. What
        # matters is that it converges rather than drifting.
        before = dense(self.before_bones, self.before_weights)
        again_bones, again_weights = blend(
            self.vertices, self.region, self.after_bones, self.after_weights
        )
        again = dense(again_bones, again_weights)
        first_move = float(np.abs(self.after - before).sum())
        second_move = float(np.abs(again - self.after).sum())
        self.assertLess(second_move, 0.1 * first_move)


class ScissorPoseTests(unittest.TestCase):
    """The artifact itself: rotate the two legs in opposite directions about
    the lateral axis and measure how far the shell's own edges are stretched.
    Nearest-wins shears the shell into two rigid halves; the blend does not."""

    def _deform(self, vertices, bone_indices, weights, degrees=30.0):
        angle = np.radians(degrees)

        def rotation(sign, pivot):
            c, s = np.cos(sign * angle), np.sin(sign * angle)
            matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
            return matrix, pivot

        transforms = {
            PELVIS: (np.eye(3), np.array([0.0, 0.0, HIP_HEIGHT])),
            THIGH_L: rotation(1.0, np.array([0.1, 0.0, HIP_HEIGHT])),
            THIGH_R: rotation(-1.0, np.array([-0.1, 0.0, HIP_HEIGHT])),
            CALF_L: rotation(1.0, np.array([0.1, 0.0, HIP_HEIGHT])),
            CALF_R: rotation(-1.0, np.array([-0.1, 0.0, HIP_HEIGHT])),
            FOOT_L: rotation(1.0, np.array([0.1, 0.0, HIP_HEIGHT])),
            FOOT_R: rotation(-1.0, np.array([-0.1, 0.0, HIP_HEIGHT])),
        }
        out = np.zeros_like(vertices)
        for slot in range(bone_indices.shape[1]):
            for bone, (matrix, pivot) in transforms.items():
                rows = bone_indices[:, slot] == bone
                if not np.any(rows):
                    continue
                moved = (vertices[rows] - pivot) @ matrix.T + pivot
                out[rows] += moved * weights[rows, slot][:, None]
        return out

    def _worst_stretch(self, vertices, bone_indices, weights):
        posed = self._deform(vertices, bone_indices, weights)
        # ring-neighbour edges of the synthetic cone
        segments = 32
        count = vertices.shape[0]
        a = np.arange(count)
        b = (a // segments) * segments + (a + 1) % segments
        rest = np.linalg.norm(vertices[a] - vertices[b], axis=1)
        keep = rest > 1e-9
        posed_length = np.linalg.norm(posed[a] - posed[b], axis=1)
        return float(np.percentile(posed_length[keep] / rest[keep], 99))

    def test_the_blend_cuts_the_shear_a_scissor_pose_puts_on_a_skirt(self):
        vertices = skirt_shell()
        region = detect(vertices)
        before_bones, before_weights = nearest_wins_weights(vertices)
        after_bones, after_weights = blend(vertices, region, before_bones, before_weights)
        before = self._worst_stretch(vertices, before_bones, before_weights)
        after = self._worst_stretch(vertices, after_bones, after_weights)
        self.assertGreater(before, 1.3)
        self.assertLess(after, before * 0.7)


if __name__ == "__main__":
    unittest.main()
