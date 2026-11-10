"""Tests for the torso-shell and inter-leg-bridge weighting post-passes.

Two failures `detect_hanging_region` cannot reach:

* A loose garment ABOVE the hips -- an open vest or jacket standing off the
  ribcage. Nearest-wins hands its flaps to whichever bone is closest, which
  near the shoulders is an ARM bone, so half a vest ends up glued to a
  swinging arm and tears off the trunk. The structural signal is the
  corridor idea one dimension over: a fitted torso's cross-section is a
  single closed ring, crossed ONCE along any radial direction from the
  spine; a shell offset from the body is crossed TWICE, with air between.
* Material bridging BETWEEN the legs (the crotch web), which needs the two
  legs to share the stride tension symmetrically with the pelvis holding
  the middle -- and, above all, needs the weight GRADIENT across it
  flattened, since that is what a sheet actually tears along.

Fixtures: a torso cylinder about a vertical spine, optionally wearing an
open vest shell and/or holding its arms out sideways; and a pair of
trouser legs, optionally webbed at the crotch.
"""
import itertools
import unittest

import numpy as np

from backend.app.skeleton_naming import (
    BridgeRegion,
    TorsoShellRegion,
    bridge_blend_weights,
    detect_bridge_region,
    detect_torso_shell,
    mix_smoothed_weights,
    radial_shell_split,
    torso_shell_blend_weights,
)

UP = 2
LATERAL = 0
DEPTH = 1

HIP_HEIGHT = 0.90
SHOULDER_HEIGHT = 1.45
BODY_RADIUS = 0.15
VEST_RADIUS = 0.26

SPINE_NAMES = ("pelvis", "spine_1", "spine_2", "spine_3", "neck")
SPINE_HEIGHTS = (0.90, 1.05, 1.20, 1.35, 1.50)
SPINE_BONE_HEIGHTS = SPINE_HEIGHTS[:4]
CLAVICLE_HEIGHT = 1.42
CLAVICLE_SPAN = 0.30

PELVIS, SPINE_1, SPINE_2, SPINE_3 = 0, 1, 2, 3
CLAVICLE_L, CLAVICLE_R = 4, 5
UPPER_ARM_L, UPPER_ARM_R = 6, 7
THIGH_L, THIGH_R = 8, 9
BONE_COUNT = 10
SPINE_BONES = (PELVIS, SPINE_1, SPINE_2, SPINE_3)

SPINE_SEGMENTS = [
    (np.array([0.0, 0.0, low]), np.array([0.0, 0.0, high]))
    for low, high in itertools.pairwise(SPINE_HEIGHTS)
]
ARM_SEGMENTS = [
    (np.array([0.16, 0.0, 1.45]), np.array([0.60, 0.0, 1.45])),
    (np.array([-0.16, 0.0, 1.45]), np.array([-0.60, 0.0, 1.45])),
]


def tube(radius, *, bottom, top, rings=60, segments=64, angle_from=-np.pi, angle_to=np.pi):
    """Vertices on a vertical tube (or an arc of one) about the origin."""
    points = []
    for ring in range(rings):
        z = bottom + (top - bottom) * ring / max(rings - 1, 1)
        for index in range(segments):
            angle = angle_from + (angle_to - angle_from) * index / max(segments - 1, 1)
            points.append([radius * np.cos(angle), radius * np.sin(angle), z])
    return np.asarray(points, dtype=np.float64)


def arm_tube(sign, *, radius=0.05, rings=30, segments=16):
    """A horizontal arm sticking out sideways from the shoulder."""
    points = []
    for ring in range(rings):
        x = sign * (0.16 + (0.60 - 0.16) * ring / max(rings - 1, 1))
        for index in range(segments):
            angle = 2 * np.pi * index / segments
            points.append([x, radius * np.cos(angle), 1.45 + radius * np.sin(angle)])
    return np.asarray(points, dtype=np.float64)


def torso_only():
    return tube(BODY_RADIUS, bottom=0.88, top=SHOULDER_HEIGHT)


def torso_with_vest():
    """A body cylinder wearing an open vest: a shell at a larger radius,
    open at the front (a 90-degree gap centred on +depth)."""
    return np.concatenate([
        torso_only(),
        tube(VEST_RADIUS, bottom=0.95, top=1.38, rings=44, segments=56,
             angle_from=np.pi / 4, angle_to=2 * np.pi - np.pi / 4),
    ])


def detect(vertices, *, arm_segments=()):
    return detect_torso_shell(
        vertices,
        up_axis=UP,
        up_sign=1.0,
        lateral_axis=LATERAL,
        depth_axis=DEPTH,
        hip_height=HIP_HEIGHT,
        shoulder_height=SHOULDER_HEIGHT,
        spine_heights=SPINE_HEIGHTS,
        spine_lateral=[0.0] * len(SPINE_HEIGHTS),
        spine_depth=[0.0] * len(SPINE_HEIGHTS),
        spine_segments=SPINE_SEGMENTS,
        arm_segments=list(arm_segments),
    )


def single_bone_weights(count, bone):
    bone_indices = np.full((count, 4), -1, dtype=np.int64)
    weights = np.zeros((count, 4), dtype=np.float64)
    bone_indices[:, 0] = bone
    weights[:, 0] = 1.0
    return bone_indices, weights


def blend(vertices, region, bone_indices, weights, **overrides):
    kwargs = {
        "up_axis": UP,
        "up_sign": 1.0,
        "lateral_axis": LATERAL,
        "spine_bones": SPINE_BONES,
        "spine_bone_heights": SPINE_BONE_HEIGHTS,
        "clavicle_left_bone": CLAVICLE_L,
        "clavicle_right_bone": CLAVICLE_R,
        "clavicle_height": CLAVICLE_HEIGHT,
        "clavicle_span": CLAVICLE_SPAN,
        "lateral_midline": 0.0,
    }
    kwargs.update(overrides)
    return torso_shell_blend_weights(vertices, region, bone_indices, weights, **kwargs)


def dominant(bone_indices, weights, rows):
    return bone_indices[rows, np.argmax(weights[rows], axis=1)]


class RadialShellSplitTests(unittest.TestCase):
    def test_one_cluster_has_no_meaningful_gap(self):
        radii = 0.15 + np.linspace(-0.004, 0.004, 40)
        gap, _ = radial_shell_split(radii, 0.15)
        self.assertLess(gap, 0.1)

    def test_two_clusters_report_the_gap_and_the_inner_edge(self):
        inner = 0.15 + np.linspace(-0.004, 0.004, 20)
        outer = 0.26 + np.linspace(-0.004, 0.004, 20)
        gap, split = radial_shell_split(np.concatenate([inner, outer]), 0.15)
        self.assertGreater(gap, 0.6)
        self.assertAlmostEqual(split, float(inner.max()), places=6)

    def test_a_single_stray_vertex_is_not_a_shell(self):
        radii = np.concatenate([0.15 + np.linspace(-0.004, 0.004, 30), [0.40]])
        gap, _ = radial_shell_split(radii, 0.15, cluster_minimum=3)
        self.assertLess(gap, 0.15)

    def test_too_few_samples_says_nothing(self):
        gap, split = radial_shell_split(np.array([0.1, 0.2, 0.3]), 0.15)
        self.assertEqual((gap, split), (0.0, 0.0))


class DetectTorsoShellTests(unittest.TestCase):
    def test_fitted_torso_is_a_no_op(self):
        region = detect(torso_only())
        self.assertEqual(int(region.mask.sum()), 0)
        self.assertFalse(np.any(region.strength))

    def test_open_vest_is_detected(self):
        vertices = torso_with_vest()
        region = detect(vertices)
        self.assertGreater(int(region.mask.sum()), 200)
        radius = np.hypot(vertices[:, LATERAL], vertices[:, DEPTH])
        # everything selected is outer material, not the body cylinder
        self.assertGreater(float(radius[region.mask].min()), BODY_RADIUS * 1.2)
        self.assertLessEqual(float(region.strength.max()), 0.7 + 1e-9)

    def test_detection_reports_the_gap_it_measured(self):
        region = detect(torso_with_vest())
        self.assertGreater(region.widest_gap, 0.15)
        self.assertGreater(region.core_radius, 0.0)
        self.assertGreaterEqual(region.region_bottom, HIP_HEIGHT)
        self.assertLessEqual(region.region_top, SHOULDER_HEIGHT)

    def test_outstretched_arms_are_not_a_shell(self):
        """A T-posed character's arms are a huge outer radial cluster in the
        lateral sectors. Without the arm exclusion the detector fires on
        every character alike, which is the bug this test pins."""
        vertices = np.concatenate([torso_only(), arm_tube(1.0), arm_tube(-1.0)])
        region = detect(vertices, arm_segments=ARM_SEGMENTS)
        self.assertEqual(int(region.mask.sum()), 0)

    def test_vest_still_found_with_arms_present(self):
        vertices = np.concatenate([torso_with_vest(), arm_tube(1.0), arm_tube(-1.0)])
        region = detect(vertices, arm_segments=ARM_SEGMENTS)
        self.assertGreater(int(region.mask.sum()), 100)
        lateral = np.abs(vertices[region.mask, LATERAL])
        self.assertLess(float(lateral.max()), 0.30)

    def test_strength_fades_in_above_the_hip_line(self):
        """The pass degrades continuously into `detect_hanging_region`'s
        territory below the hips rather than stepping at the hip line."""
        vertices = torso_with_vest()
        short_ramp = detect_torso_shell(
            vertices, up_axis=UP, up_sign=1.0, lateral_axis=LATERAL, depth_axis=DEPTH,
            hip_height=HIP_HEIGHT, shoulder_height=SHOULDER_HEIGHT,
            spine_heights=SPINE_HEIGHTS, spine_lateral=[0.0] * 5, spine_depth=[0.0] * 5,
            spine_segments=SPINE_SEGMENTS, arm_segments=[], hip_margin=0.001,
        )
        long_ramp = detect_torso_shell(
            vertices, up_axis=UP, up_sign=1.0, lateral_axis=LATERAL, depth_axis=DEPTH,
            hip_height=HIP_HEIGHT, shoulder_height=SHOULDER_HEIGHT,
            spine_heights=SPINE_HEIGHTS, spine_lateral=[0.0] * 5, spine_depth=[0.0] * 5,
            spine_segments=SPINE_SEGMENTS, arm_segments=[], hip_margin=0.6,
        )
        self.assertLess(float(long_ramp.strength.sum()), float(short_ramp.strength.sum()))
        heights = vertices[:, UP]
        low = heights < 1.05
        self.assertLess(float(long_ramp.strength[low].sum()), float(short_ramp.strength[low].sum()))

    def test_smoothing_strength_is_the_uncapped_shape(self):
        region = detect(torso_with_vest())
        rows = region.mask
        self.assertTrue(np.all(region.smoothing_strength[rows] >= region.strength[rows] - 1e-12))
        self.assertLessEqual(float(region.smoothing_strength.max()), 1.0 + 1e-9)


class TorsoShellBlendTests(unittest.TestCase):
    def setUp(self):
        self.vertices = torso_with_vest()
        self.region = detect(self.vertices)
        self.rows = np.nonzero(self.region.mask)[0]
        self.assertGreater(self.rows.size, 0)

    def full_strength_region(self):
        return TorsoShellRegion(
            mask=self.region.mask,
            strength=np.where(self.region.mask, 1.0, 0.0),
            smoothing_strength=self.region.smoothing_strength,
            core_radius=self.region.core_radius,
            region_top=self.region.region_top,
            region_bottom=self.region.region_bottom,
            band_shell_counts=self.region.band_shell_counts,
            widest_gap=self.region.widest_gap,
        )

    def test_vest_glued_to_an_arm_is_re_anchored_to_the_spine(self):
        """The punk's failure: nearest-wins gave the vest to an arm bone and
        the swing tore it off the trunk. After the blend the trunk owns it."""
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(
            self.vertices, self.full_strength_region(), bone_indices, weights
        )
        winners = dominant(out_bones, out_weights, self.rows)
        self.assertTrue(np.all(np.isin(winners, SPINE_BONES)))
        arm_share = np.sum(np.where(out_bones[self.rows] == UPPER_ARM_L, out_weights[self.rows], 0.0))
        self.assertEqual(arm_share, 0.0)

    def test_vertices_outside_the_region_are_untouched(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(self.vertices, self.region, bone_indices, weights)
        outside = np.nonzero(~self.region.mask)[0]
        np.testing.assert_array_equal(out_bones[outside], bone_indices[outside])
        np.testing.assert_array_equal(out_weights[outside], weights[outside])

    def test_weights_stay_normalized_and_within_the_influence_cap(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(
            self.vertices, self.full_strength_region(), bone_indices, weights
        )
        totals = out_weights[self.rows].sum(axis=1)
        np.testing.assert_allclose(totals, 1.0, atol=1e-9)
        self.assertLessEqual(int(np.max((out_bones[self.rows] >= 0).sum(axis=1))), 4)

    def test_low_material_is_driven_by_the_lower_spine(self):
        region = self.full_strength_region()
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(self.vertices, region, bone_indices, weights)
        heights = self.vertices[self.rows, UP]
        low = self.rows[heights < 1.05]
        if low.size:
            winners = dominant(out_bones, out_weights, low)
            self.assertTrue(np.all(np.isin(winners, (PELVIS, SPINE_1))))

    def test_a_front_flap_gets_no_shoulder_weight(self):
        """Lateral spread towards the clavicles is what lets a shoulder panel
        follow the shoulder. Material on the midline must not get it, at any
        height, or the fix recreates the bug in a milder form."""
        region = self.full_strength_region()
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(self.vertices, region, bone_indices, weights)
        midline = self.rows[np.abs(self.vertices[self.rows, LATERAL]) < 0.01]
        self.assertGreater(midline.size, 0)
        clavicle = np.isin(out_bones[midline], (CLAVICLE_L, CLAVICLE_R))
        share = np.sum(np.where(clavicle, out_weights[midline], 0.0), axis=1)
        self.assertLess(float(share.max()), 0.01)

    def test_shoulder_share_grows_with_lateral_offset(self):
        region = self.full_strength_region()
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(self.vertices, region, bone_indices, weights)
        clavicle = np.isin(out_bones, (CLAVICLE_L, CLAVICLE_R))
        share = np.sum(np.where(clavicle, out_weights, 0.0), axis=1)
        heights = self.vertices[self.rows, UP]
        high = self.rows[heights > 1.30]
        lateral = np.abs(self.vertices[high, LATERAL])
        inner = share[high[lateral < 0.08]]
        outer = share[high[lateral > 0.20]]
        if inner.size and outer.size:
            self.assertLess(float(inner.mean()), float(outer.mean()))

    def test_shoulder_material_may_take_the_clavicle_on_its_own_side(self):
        region = self.full_strength_region()
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(self.vertices, region, bone_indices, weights)
        heights = self.vertices[self.rows, UP]
        lateral = self.vertices[self.rows, LATERAL]
        left_high = self.rows[(heights > 1.30) & (lateral > 0.20)]
        if left_high.size:
            left_share = np.sum(np.where(out_bones[left_high] == CLAVICLE_L, out_weights[left_high], 0.0))
            right_share = np.sum(np.where(out_bones[left_high] == CLAVICLE_R, out_weights[left_high], 0.0))
            self.assertGreater(left_share, right_share)

    def test_unresolved_clavicles_degrade_to_spine_only(self):
        region = self.full_strength_region()
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        out_bones, out_weights = blend(
            self.vertices, region, bone_indices, weights,
            clavicle_left_bone=-1, clavicle_right_bone=-1,
        )
        winners = dominant(out_bones, out_weights, self.rows)
        self.assertTrue(np.all(np.isin(winners, SPINE_BONES)))
        np.testing.assert_allclose(out_weights[self.rows].sum(axis=1), 1.0, atol=1e-9)


# --- inter-leg bridge ------------------------------------------------------

BRIDGE_HIP_HEIGHT = 1.0
BRIDGE_HALF_SPAN = 0.10
LEFT_LEG_SEGMENTS = [
    (np.array([0.10, 0.0, 1.0]), np.array([0.10, 0.0, 0.5])),
    (np.array([0.10, 0.0, 0.5]), np.array([0.10, 0.0, 0.05])),
]
RIGHT_LEG_SEGMENTS = [
    (np.array([-0.10, 0.0, 1.0]), np.array([-0.10, 0.0, 0.5])),
    (np.array([-0.10, 0.0, 0.5]), np.array([-0.10, 0.0, 0.05])),
]


def leg_tube(centre_x, radius=0.06, *, rings=40, segments=32, bottom=0.05, top=1.0):
    points = []
    for ring in range(rings):
        z = bottom + (top - bottom) * ring / max(rings - 1, 1)
        for index in range(segments):
            angle = 2 * np.pi * index / segments
            points.append([centre_x + radius * np.cos(angle), radius * np.sin(angle), z])
    return np.asarray(points, dtype=np.float64)


def trousers():
    return np.concatenate([leg_tube(0.10), leg_tube(-0.10)])


def trousers_with_web(rows=24, columns=15):
    """A sheet of fabric spanning the crotch, clear of both legs."""
    web = []
    for row in range(rows):
        z = 0.74 + (0.99 - 0.74) * row / max(rows - 1, 1)
        for column in range(columns):
            x = -0.055 + 0.110 * column / max(columns - 1, 1)
            web.append([x, 0.0, z])
            web.append([x, 0.02, z])
    return np.concatenate([trousers(), np.asarray(web, dtype=np.float64)])


def detect_bridge(vertices, **overrides):
    kwargs = {
        "up_axis": UP,
        "up_sign": 1.0,
        "lateral_axis": LATERAL,
        "hip_height": BRIDGE_HIP_HEIGHT,
        "hip_half_span": BRIDGE_HALF_SPAN,
        "hip_midline": 0.0,
        "left_segments": LEFT_LEG_SEGMENTS,
        "right_segments": RIGHT_LEG_SEGMENTS,
    }
    kwargs.update(overrides)
    return detect_bridge_region(vertices, **kwargs)


def blend_bridge(vertices, region, bone_indices, weights, **overrides):
    kwargs = {
        "up_axis": UP,
        "up_sign": 1.0,
        "lateral_axis": LATERAL,
        "hip_height": BRIDGE_HIP_HEIGHT,
        "hip_half_span": BRIDGE_HALF_SPAN,
        "hip_midline": 0.0,
        "pelvis_bone": PELVIS,
        "thigh_left_bone": THIGH_L,
        "thigh_right_bone": THIGH_R,
    }
    kwargs.update(overrides)
    return bridge_blend_weights(vertices, region, bone_indices, weights, **kwargs)


class DetectBridgeRegionTests(unittest.TestCase):
    def test_bare_legs_have_no_bridge(self):
        region = detect_bridge(trousers())
        self.assertEqual(int(region.mask.sum()), 0)

    def test_crotch_web_is_detected(self):
        vertices = trousers_with_web()
        region = detect_bridge(vertices)
        self.assertGreater(int(region.mask.sum()), 100)
        # nothing outside the corridor is selected
        self.assertLessEqual(
            float(np.abs(vertices[region.mask, LATERAL]).max()), 0.65 * BRIDGE_HALF_SPAN + 1e-9
        )
        self.assertTrue(np.all(vertices[region.mask, UP] < BRIDGE_HIP_HEIGHT))

    def test_the_corridor_is_what_bounds_the_selection(self):
        """Distances alone fire on inner thighs; being BETWEEN the legs is
        what makes the rule bridging-specific, so the corridor is a hard
        lateral bound on everything selected."""
        vertices = trousers_with_web()
        wide = detect_bridge(vertices, corridor=0.25)
        narrow = detect_bridge(vertices, corridor=0.10)
        self.assertGreater(int(wide.mask.sum()), int(narrow.mask.sum()))
        for region, corridor in ((wide, 0.25), (narrow, 0.10)):
            selected = np.abs(vertices[region.mask, LATERAL]) / BRIDGE_HALF_SPAN
            self.assertLessEqual(float(selected.max()), corridor + 1e-9)

    def test_material_sitting_on_a_leg_is_not_bridging(self):
        """A sheet inside the corridor but at a leg's own radius is skin,
        and `clearance` is what keeps it out."""
        sheet = []
        for row in range(24):
            z = 0.74 + (0.99 - 0.74) * row / 23.0
            for depth in (0.0, 0.02):
                sheet.append([0.045, depth, z])
        vertices = np.concatenate([trousers(), np.asarray(sheet, dtype=np.float64)])
        region = detect_bridge(vertices)
        self.assertEqual(int(region.mask.sum()), 0)

    def test_smoothing_strength_is_the_uncapped_shape(self):
        region = detect_bridge(trousers_with_web())
        rows = region.mask
        self.assertTrue(np.all(region.smoothing_strength[rows] >= region.strength[rows] - 1e-12))
        self.assertLessEqual(float(region.strength.max()), 0.3 + 1e-9)


class BridgeBlendTests(unittest.TestCase):
    def setUp(self):
        self.vertices = trousers_with_web()
        detected = detect_bridge(self.vertices)
        self.region = BridgeRegion(
            mask=detected.mask,
            strength=np.where(detected.mask, 1.0, 0.0),
            smoothing_strength=detected.smoothing_strength,
            region_top=detected.region_top,
            region_bottom=detected.region_bottom,
        )
        self.rows = np.nonzero(detected.mask)[0]
        self.assertGreater(self.rows.size, 0)

    def test_midline_material_is_driven_equally_by_both_thighs(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(self.vertices, self.region, bone_indices, weights)
        midline = self.rows[np.abs(self.vertices[self.rows, LATERAL]) < 1e-9]
        self.assertGreater(midline.size, 0)
        left = np.sum(np.where(out_bones[midline] == THIGH_L, out_weights[midline], 0.0), axis=1)
        right = np.sum(np.where(out_bones[midline] == THIGH_R, out_weights[midline], 0.0), axis=1)
        np.testing.assert_allclose(left, right, atol=1e-9)

    def test_the_rule_is_symmetric_across_the_midline(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(self.vertices, self.region, bone_indices, weights)

        def share(row, bone):
            return float(np.sum(np.where(out_bones[row] == bone, out_weights[row], 0.0)))

        lateral = self.vertices[:, LATERAL]
        height = self.vertices[:, UP]
        for row in self.rows[:200]:
            mirrored = np.nonzero(
                self.region.mask
                & (np.abs(lateral + lateral[row]) < 1e-9)
                & (np.abs(height - height[row]) < 1e-9)
                & (np.abs(self.vertices[:, DEPTH] - self.vertices[row, DEPTH]) < 1e-9)
            )[0]
            if mirrored.size != 1:
                continue
            other = int(mirrored[0])
            self.assertAlmostEqual(share(row, THIGH_L), share(other, THIGH_R), places=9)
            self.assertAlmostEqual(share(row, PELVIS), share(other, PELVIS), places=9)

    def test_pelvis_never_drops_below_the_stabilizer_floor(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(self.vertices, self.region, bone_indices, weights)
        pelvis = np.sum(
            np.where(out_bones[self.rows] == PELVIS, out_weights[self.rows], 0.0), axis=1
        )
        self.assertGreaterEqual(float(pelvis.min()), 0.3 - 1e-9)

    def test_the_floor_is_configurable_and_respected(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(
            self.vertices, self.region, bone_indices, weights, pelvis_floor=0.6
        )
        pelvis = np.sum(
            np.where(out_bones[self.rows] == PELVIS, out_weights[self.rows], 0.0), axis=1
        )
        self.assertGreaterEqual(float(pelvis.min()), 0.6 - 1e-9)

    def test_vertices_outside_the_region_are_untouched(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(self.vertices, self.region, bone_indices, weights)
        outside = np.nonzero(~self.region.mask)[0]
        np.testing.assert_array_equal(out_bones[outside], bone_indices[outside])
        np.testing.assert_array_equal(out_weights[outside], weights[outside])

    def test_an_unresolved_pelvis_still_leaves_a_normalized_weighting(self):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        out_bones, out_weights = blend_bridge(
            self.vertices, self.region, bone_indices, weights, pelvis_bone=-1
        )
        np.testing.assert_allclose(out_weights[self.rows].sum(axis=1), 1.0, atol=1e-9)
        # the pelvis share is handed to the thigh pair, not left dangling
        self.assertTrue(np.all(np.isin(dominant(out_bones, out_weights, self.rows),
                                       (THIGH_L, THIGH_R))))


class MixSmoothedWeightsTests(unittest.TestCase):
    def setUp(self):
        self.bone_indices = np.array([[0, 1, -1, -1], [2, -1, -1, -1]], dtype=np.int64)
        self.weights = np.array([[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        self.smoothed_bones = np.array([[3, -1, -1, -1], [4, 5, -1, -1]], dtype=np.int64)
        self.smoothed_weights = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])

    def test_zero_strength_is_a_no_op(self):
        out_bones, out_weights = mix_smoothed_weights(
            self.bone_indices, self.weights, self.smoothed_bones, self.smoothed_weights,
            np.array([0, 1]), np.array([0.0, 0.0]),
        )
        np.testing.assert_array_equal(out_bones, self.bone_indices)
        np.testing.assert_array_equal(out_weights, self.weights)

    def test_full_strength_takes_the_smoothed_weighting(self):
        out_bones, out_weights = mix_smoothed_weights(
            self.bone_indices, self.weights, self.smoothed_bones, self.smoothed_weights,
            np.array([0]), np.array([1.0]),
        )
        self.assertEqual(int(out_bones[0, 0]), 3)
        self.assertAlmostEqual(float(out_weights[0, 0]), 1.0)

    def test_half_strength_averages_both(self):
        out_bones, out_weights = mix_smoothed_weights(
            self.bone_indices, self.weights, self.smoothed_bones, self.smoothed_weights,
            np.array([0]), np.array([0.5]),
        )
        row = {int(b): float(w) for b, w in zip(out_bones[0], out_weights[0]) if b >= 0}
        self.assertAlmostEqual(row[0], 0.3)
        self.assertAlmostEqual(row[1], 0.2)
        self.assertAlmostEqual(row[3], 0.5)

    def test_untouched_rows_are_byte_identical(self):
        out_bones, out_weights = mix_smoothed_weights(
            self.bone_indices, self.weights, self.smoothed_bones, self.smoothed_weights,
            np.array([0]), np.array([1.0]),
        )
        np.testing.assert_array_equal(out_bones[1], self.bone_indices[1])
        np.testing.assert_array_equal(out_weights[1], self.weights[1])

    def test_result_is_normalized_and_capped(self):
        bone_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)
        weights = np.array([[0.25, 0.25, 0.25, 0.25]])
        smoothed_bones = np.array([[4, 5, 6, 7]], dtype=np.int64)
        smoothed_weights = np.array([[0.25, 0.25, 0.25, 0.25]])
        out_bones, out_weights = mix_smoothed_weights(
            bone_indices, weights, smoothed_bones, smoothed_weights,
            np.array([0]), np.array([0.5]), max_influences=4,
        )
        self.assertEqual(int(np.count_nonzero(out_bones[0] >= 0)), 4)
        self.assertAlmostEqual(float(out_weights[0].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
