"""The left/right split of the torso-shell and bridge blends follows the
rig's own bones, not a fixed lateral sign.

`skeleton_naming` assigns the l/r suffixes from the character's FACING, so
on a +Y-facing character `thigh_l` / `clavicle_l` sit at the LOWER lateral
coordinate. `hanging_region_blend_weights` already ramps across the signed
`lateral_left - lateral_right` span; these tests pin the two sibling blends
to the same convention, on fixtures mirrored from the ones in
`test_torso_shell_region` (so `lateral_left < lateral_right`), and check
that left-side material lands on the `*_l` bone.
"""
import unittest

import numpy as np

from backend.app.skeleton_naming import BridgeRegion
from backend.tests.test_torso_shell_region import (
    BRIDGE_HALF_SPAN,
    CLAVICLE_L,
    CLAVICLE_R,
    CLAVICLE_SPAN,
    LATERAL,
    LEFT_LEG_SEGMENTS,
    RIGHT_LEG_SEGMENTS,
    THIGH_L,
    THIGH_R,
    UP,
    UPPER_ARM_L,
    blend,
    blend_bridge,
    detect,
    detect_bridge,
    single_bone_weights,
    torso_with_vest,
    trousers_with_web,
)


def mirrored(vertices):
    out = np.asarray(vertices, dtype=np.float64).copy()
    out[:, LATERAL] *= -1.0
    return out


def mirrored_segments(segments):
    return [(mirrored(head[None])[0], mirrored(tail[None])[0]) for head, tail in segments]


def share(bone_indices, weights, rows, bone):
    return np.sum(np.where(bone_indices[rows] == bone, weights[rows], 0.0), axis=1)


class MirroredBridgeBlendTests(unittest.TestCase):
    """thigh_l at -0.10, thigh_r at +0.10: the web on the -lateral side is
    the LEFT leg's and must be driven by THIGH_L."""

    def setUp(self):
        self.vertices = mirrored(trousers_with_web())
        detected = detect_bridge(
            self.vertices,
            left_segments=mirrored_segments(LEFT_LEG_SEGMENTS),
            right_segments=mirrored_segments(RIGHT_LEG_SEGMENTS),
        )
        self.region = BridgeRegion(
            mask=detected.mask,
            strength=np.where(detected.mask, 1.0, 0.0),
            smoothing_strength=detected.smoothing_strength,
            region_top=detected.region_top,
            region_bottom=detected.region_bottom,
        )
        self.rows = np.nonzero(detected.mask)[0]
        self.assertGreater(self.rows.size, 0)
        self.left_side = self.rows[self.vertices[self.rows, LATERAL] < -0.02]
        self.right_side = self.rows[self.vertices[self.rows, LATERAL] > 0.02]
        self.assertGreater(self.left_side.size, 0)
        self.assertGreater(self.right_side.size, 0)

    def blend_mirrored(self, **overrides):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], THIGH_L)
        kwargs = {"lateral_left": -BRIDGE_HALF_SPAN, "lateral_right": BRIDGE_HALF_SPAN}
        kwargs.update(overrides)
        return blend_bridge(self.vertices, self.region, bone_indices, weights, **kwargs)

    def test_left_side_web_is_driven_by_the_left_thigh(self):
        out_bones, out_weights = self.blend_mirrored()
        left_l = share(out_bones, out_weights, self.left_side, THIGH_L)
        left_r = share(out_bones, out_weights, self.left_side, THIGH_R)
        self.assertTrue(np.all(left_l > left_r))
        right_l = share(out_bones, out_weights, self.right_side, THIGH_L)
        right_r = share(out_bones, out_weights, self.right_side, THIGH_R)
        self.assertTrue(np.all(right_r > right_l))

    def test_mirrored_rig_reproduces_the_unmirrored_weighting(self):
        """Reflecting the whole scene (mesh AND bones) must reflect the
        weights with it: thigh_l's share at +x on the original equals
        thigh_l's share at -x on the mirror."""
        original = trousers_with_web()
        detected = detect_bridge(original)
        region = BridgeRegion(
            mask=detected.mask,
            strength=np.where(detected.mask, 1.0, 0.0),
            smoothing_strength=detected.smoothing_strength,
            region_top=detected.region_top,
            region_bottom=detected.region_bottom,
        )
        np.testing.assert_array_equal(region.mask, self.region.mask)
        bone_indices, weights = single_bone_weights(original.shape[0], THIGH_L)
        base_bones, base_weights = blend_bridge(
            original, region, bone_indices, weights,
            lateral_left=BRIDGE_HALF_SPAN, lateral_right=-BRIDGE_HALF_SPAN,
        )
        out_bones, out_weights = self.blend_mirrored()
        for bone in (THIGH_L, THIGH_R):
            np.testing.assert_allclose(
                share(out_bones, out_weights, self.rows, bone),
                share(base_bones, base_weights, self.rows, bone),
                atol=1e-12,
            )

    def test_midline_stays_an_even_split(self):
        out_bones, out_weights = self.blend_mirrored()
        midline = self.rows[np.abs(self.vertices[self.rows, LATERAL]) < 1e-9]
        self.assertGreater(midline.size, 0)
        np.testing.assert_allclose(
            share(out_bones, out_weights, midline, THIGH_L),
            share(out_bones, out_weights, midline, THIGH_R),
            atol=1e-9,
        )

    def test_without_bone_coordinates_the_split_defaults_to_plus_lateral(self):
        """The fallback keeps the old +lateral = left behaviour for callers
        that pass no thigh coordinates -- which is exactly what a mirrored
        rig must NOT rely on, so the worker passes them."""
        out_bones, out_weights = self.blend_mirrored(lateral_left=None, lateral_right=None)
        left_l = share(out_bones, out_weights, self.left_side, THIGH_L)
        left_r = share(out_bones, out_weights, self.left_side, THIGH_R)
        self.assertTrue(np.all(left_r > left_l))


class MirroredTorsoShellBlendTests(unittest.TestCase):
    """clavicle_l at -0.15, clavicle_r at +0.15 (signed span -0.30): the
    shoulder panel on the -lateral side is the LEFT shoulder's.

    The mesh is the unmirrored vest (its opening is centred on +lateral,
    so it has shoulder material at x < -0.20 but only up to x ~ 0.18); the
    only thing mirrored is where the rig's clavicles are."""

    def setUp(self):
        self.vertices = torso_with_vest()
        detected = detect(self.vertices)
        self.region = detected._replace(strength=np.where(detected.mask, 1.0, 0.0))
        self.rows = np.nonzero(detected.mask)[0]
        self.assertGreater(self.rows.size, 0)
        heights = self.vertices[self.rows, UP]
        lateral = self.vertices[self.rows, LATERAL]
        self.left_high = self.rows[(heights > 1.30) & (lateral < -0.20)]
        self.right_high = self.rows[(heights > 1.30) & (lateral > 0.12)]
        self.assertGreater(self.left_high.size, 0)
        self.assertGreater(self.right_high.size, 0)

    def blend_mirrored(self, **overrides):
        bone_indices, weights = single_bone_weights(self.vertices.shape[0], UPPER_ARM_L)
        kwargs = {"clavicle_span": -CLAVICLE_SPAN}
        kwargs.update(overrides)
        return blend(self.vertices, self.region, bone_indices, weights, **kwargs)

    def test_left_shoulder_panel_takes_the_left_clavicle(self):
        out_bones, out_weights = self.blend_mirrored()
        left_l = float(share(out_bones, out_weights, self.left_high, CLAVICLE_L).sum())
        left_r = float(share(out_bones, out_weights, self.left_high, CLAVICLE_R).sum())
        self.assertGreater(left_l, 0.0)
        self.assertGreater(left_l, left_r)
        right_l = float(share(out_bones, out_weights, self.right_high, CLAVICLE_L).sum())
        right_r = float(share(out_bones, out_weights, self.right_high, CLAVICLE_R).sum())
        self.assertGreater(right_r, right_l)

    def test_sign_only_swaps_the_clavicles(self):
        """The span's magnitude drives reach and softness; flipping its sign
        must move the SAME shoulder share to the other clavicle and leave
        the spine share untouched."""
        neg_bones, neg_weights = self.blend_mirrored()
        pos_bones, pos_weights = self.blend_mirrored(clavicle_span=CLAVICLE_SPAN)
        np.testing.assert_allclose(
            share(neg_bones, neg_weights, self.rows, CLAVICLE_L),
            share(pos_bones, pos_weights, self.rows, CLAVICLE_R),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            share(neg_bones, neg_weights, self.rows, CLAVICLE_R),
            share(pos_bones, pos_weights, self.rows, CLAVICLE_L),
            atol=1e-12,
        )
        np.testing.assert_allclose(neg_weights[self.rows].sum(axis=1), 1.0, atol=1e-9)

    def test_midline_material_still_gets_no_shoulder_weight(self):
        out_bones, out_weights = self.blend_mirrored()
        midline = self.rows[np.abs(self.vertices[self.rows, LATERAL]) < 0.02]
        self.assertGreater(midline.size, 0)
        clavicle = np.isin(out_bones[midline], (CLAVICLE_L, CLAVICLE_R))
        self.assertLess(float(np.sum(np.where(clavicle, out_weights[midline], 0.0), axis=1).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
