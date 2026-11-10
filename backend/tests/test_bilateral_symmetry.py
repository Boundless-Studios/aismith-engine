import math
import unittest

import numpy as np

from backend.app.bilateral_symmetry import (
    SYMMETRY_MIRROR_BAND,
    measure_bilateral_symmetry,
)
from backend.app.skeleton_template import load_template


TEMPLATE = load_template("biped")
_BONES_BY_NAME = {bone.name: bone for bone in TEMPLATE.bones}
# The template's house frame: Z-up, -Y facing, left = +X.
LEFT = np.array([1.0, 0.0, 0.0])


def _template_heads() -> dict[str, np.ndarray]:
    """Rest-pose head of every template bone, walked down the hierarchy.

    The template stores per-bone `rest_direction` * `relative_length` in the
    family's house frame, so a bone's head is its parent's head plus the
    parent's own rest offset -- the "ideal" placement a perfect fitter would
    reproduce.
    """
    heads: dict[str, np.ndarray] = {}

    def head_of(name: str) -> np.ndarray:
        if name in heads:
            return heads[name]
        bone = _BONES_BY_NAME[name]
        if bone.parent is None:
            heads[name] = np.zeros(3, dtype=np.float64)
        else:
            parent = _BONES_BY_NAME[bone.parent]
            heads[name] = head_of(bone.parent) + (
                np.asarray(parent.rest_direction, dtype=np.float64) * parent.relative_length
            )
        return heads[name]

    for bone in TEMPLATE.bones:
        head_of(bone.name)
    return heads


def _template_directions() -> dict[str, np.ndarray]:
    return {
        bone.name: np.asarray(bone.rest_direction, dtype=np.float64) * bone.relative_length
        for bone in TEMPLATE.bones
    }


TEMPLATE_PAIRS = [
    (bone.name, TEMPLATE.mirror_of(bone.name))
    for bone in TEMPLATE.bones
    if bone.name.endswith("_l") and TEMPLATE.mirror_of(bone.name)
]
MODEL_HEIGHT = 1.0


def _measure(positions, directions, pairs=None, **overrides):
    kwargs = {
        "joint_positions": positions,
        "joint_directions": directions,
        "pairs": pairs if pairs is not None else TEMPLATE_PAIRS,
        "left": LEFT,
        "plane_point": positions.get("pelvis", np.zeros(3)),
        "model_height": MODEL_HEIGHT,
    }
    kwargs.update(overrides)
    return measure_bilateral_symmetry(**kwargs)


class PerfectMirrorSkeletonTests(unittest.TestCase):
    """The shipped biped template placed exactly as authored is symmetric."""

    def setUp(self):
        self.positions = _template_heads()
        self.directions = _template_directions()

    def test_a_template_perfect_rig_measures_zero_error(self):
        report = _measure(self.positions, self.directions)

        self.assertTrue(report["measurable"])
        self.assertEqual(report["pairs_measured"], len(TEMPLATE_PAIRS))
        self.assertEqual(report["pairs_skipped"], [])
        self.assertEqual(report["pairs_outside_band_count"], 0)
        # The donor's own authored rolls carry sub-micron mirror jitter; the
        # measurement is zero to any resolution the band cares about.
        self.assertLess(report["mirror_error_p95"], 1e-4)
        self.assertLess(report["mirror_error_max"], 1e-4)
        self.assertEqual(report["band"], SYMMETRY_MIRROR_BAND)

    def test_every_required_pair_is_measured(self):
        report = _measure(self.positions, self.directions)

        measured = {(item["left"], item["right"]) for item in report["pairs"]}
        required = {
            (bone.name, TEMPLATE.mirror_of(bone.name))
            for bone in TEMPLATE.required_bones
            if bone.name.endswith("_l") and TEMPLATE.mirror_of(bone.name)
        }
        self.assertTrue(required.issubset(measured))

    def test_a_mirrored_pair_has_a_matching_direction_and_length(self):
        report = _measure(self.positions, self.directions)
        upper_arm = next(item for item in report["pairs"] if item["left"] == "upper_arm_l")

        self.assertAlmostEqual(upper_arm["direction_angle_degrees"], 0.0, places=2)
        self.assertAlmostEqual(upper_arm["length_ratio"], 1.0, places=4)

    def test_a_displaced_joint_is_the_worst_offender(self):
        self.positions["upper_arm_r"] = self.positions["upper_arm_r"] + np.array(
            [0.0, 0.0, 0.05 * MODEL_HEIGHT]
        )

        report = _measure(self.positions, self.directions)

        self.assertEqual(report["worst_offenders"][0]["left"], "upper_arm_l")
        self.assertAlmostEqual(
            report["worst_offenders"][0]["mirror_error_normalized"], 0.05, places=4
        )
        self.assertEqual(report["pairs_outside_band_count"], 1)
        self.assertGreater(report["mirror_error_p95"], 0.0)

    def test_a_rotated_right_bone_shows_up_as_a_direction_angle(self):
        # Rotate the right upper arm 90deg about the vertical axis: its head
        # is untouched, so only the direction number may move.
        direction = self.directions["upper_arm_r"]
        self.directions["upper_arm_r"] = np.array(
            [-direction[1], direction[0], direction[2]]
        )

        report = _measure(self.positions, self.directions)
        upper_arm = next(item for item in report["pairs"] if item["left"] == "upper_arm_l")

        self.assertAlmostEqual(upper_arm["direction_angle_degrees"], 90.0, places=2)
        self.assertAlmostEqual(upper_arm["mirror_error_normalized"], 0.0, places=4)
        self.assertAlmostEqual(upper_arm["length_ratio"], 1.0, places=4)


class SkippedAndUnmeasurablePairsTests(unittest.TestCase):
    def setUp(self):
        self.positions = _template_heads()
        self.directions = _template_directions()

    def test_a_pair_missing_its_right_bone_is_skipped_and_listed(self):
        del self.positions["hand_r"]

        report = _measure(self.positions, self.directions)

        self.assertTrue(report["measurable"])
        self.assertIn("hand_l/hand_r", report["pairs_skipped"])
        self.assertNotIn("hand_l", [item["left"] for item in report["pairs"]])
        self.assertEqual(report["pairs_measured"], len(TEMPLATE_PAIRS) - 1)
        self.assertTrue(any("skipped" in caveat for caveat in report["caveats"]))

    def test_a_pair_without_directions_still_measures_its_placement(self):
        self.directions["hand_l"] = None
        self.directions["hand_r"] = None

        report = _measure(self.positions, self.directions)
        hand = next(item for item in report["pairs"] if item["left"] == "hand_l")

        self.assertIsNone(hand["length_ratio"])
        self.assertIsNone(hand["direction_angle_degrees"])
        self.assertIsNotNone(hand["mirror_error_normalized"])

    def test_a_degenerate_left_vector_is_not_measurable(self):
        report = _measure(self.positions, self.directions, left=np.zeros(3))

        self.assertFalse(report["measurable"])
        self.assertIsNone(report["mirror_error_p95"])
        self.assertIsNone(report["pairs_outside_band_count"])
        self.assertTrue(any("degenerate" in caveat for caveat in report["caveats"]))

    def test_a_non_positive_model_height_is_not_measurable(self):
        report = _measure(self.positions, self.directions, model_height=0.0)

        self.assertFalse(report["measurable"])
        self.assertIsNone(report["mirror_error_max"])
        self.assertTrue(any("height" in caveat for caveat in report["caveats"]))

    def test_no_measurable_pair_is_not_measurable(self):
        report = _measure({}, {})

        self.assertFalse(report["measurable"])
        self.assertEqual(report["pairs_measured"], 0)
        self.assertEqual(len(report["pairs_skipped"]), len(TEMPLATE_PAIRS))
        self.assertTrue(any("nothing to compare" in caveat for caveat in report["caveats"]))


class MirrorGeometryTests(unittest.TestCase):
    """The plane is the RIG's midline, not the world origin."""

    def test_an_offset_pair_is_symmetric_about_its_own_plane_point(self):
        offset = np.array([3.0, 0.0, 0.0])
        positions = {
            "pelvis": offset,
            "hand_l": offset + np.array([0.4, 0.1, 1.0]),
            "hand_r": offset + np.array([-0.4, 0.1, 1.0]),
        }

        report = _measure(positions, {}, pairs=[("hand_l", "hand_r")])

        self.assertTrue(report["measurable"])
        self.assertAlmostEqual(report["mirror_error_max"], 0.0, places=9)

    def test_the_same_pair_is_asymmetric_about_the_world_origin(self):
        offset = np.array([3.0, 0.0, 0.0])
        positions = {
            "hand_l": offset + np.array([0.4, 0.1, 1.0]),
            "hand_r": offset + np.array([-0.4, 0.1, 1.0]),
        }

        report = _measure(
            positions, {}, pairs=[("hand_l", "hand_r")], plane_point=np.zeros(3)
        )

        self.assertAlmostEqual(report["mirror_error_max"], 6.0, places=6)

    def test_a_longer_right_limb_reports_a_length_ratio_below_one(self):
        positions = {"pelvis": np.zeros(3), "calf_l": np.array([0.2, 0.0, 0.5])}
        positions["calf_r"] = np.array([-0.2, 0.0, 0.5])
        directions = {
            "calf_l": np.array([0.0, 0.0, -0.4]),
            "calf_r": np.array([0.0, 0.0, -0.5]),
        }

        report = _measure(positions, directions, pairs=[("calf_l", "calf_r")])
        calf = report["pairs"][0]

        self.assertAlmostEqual(calf["length_ratio"], 0.8, places=6)
        self.assertAlmostEqual(calf["direction_angle_degrees"], 0.0, places=6)

    def test_the_angle_is_measured_against_the_mirrored_left_direction(self):
        # An unmirrored comparison would call these two 180deg apart; the
        # mirror makes an outward-pointing pair agree.
        positions = {"pelvis": np.zeros(3)}
        positions["hand_l"] = np.array([0.5, 0.0, 1.0])
        positions["hand_r"] = np.array([-0.5, 0.0, 1.0])
        directions = {
            "hand_l": np.array([1.0, 0.0, 0.0]),
            "hand_r": np.array([-1.0, 0.0, 0.0]),
        }

        report = _measure(positions, directions, pairs=[("hand_l", "hand_r")])

        self.assertAlmostEqual(report["pairs"][0]["direction_angle_degrees"], 0.0, places=6)

    def test_the_band_is_configurable(self):
        positions = {
            "pelvis": np.zeros(3),
            "hand_l": np.array([0.5, 0.0, 1.0]),
            "hand_r": np.array([-0.5, 0.0, 1.02]),
        }

        loose = _measure(positions, {}, pairs=[("hand_l", "hand_r")], band=0.5)
        tight = _measure(positions, {}, pairs=[("hand_l", "hand_r")], band=0.001)

        self.assertEqual(loose["pairs_outside_band_count"], 0)
        self.assertEqual(tight["pairs_outside_band_count"], 1)
        self.assertEqual(tight["band"], 0.001)


class ReportShapeTests(unittest.TestCase):
    def test_worst_offenders_are_capped_and_ordered(self):
        positions = {"pelvis": np.zeros(3)}
        directions: dict[str, np.ndarray | None] = {}
        pairs = []
        for index in range(20):
            left_name, right_name = f"bone_{index}_l", f"bone_{index}_r"
            positions[left_name] = np.array([0.5, 0.0, float(index)])
            positions[right_name] = np.array([-0.5, float(index) * 0.01, float(index)])
            pairs.append((left_name, right_name))

        report = _measure(positions, directions, pairs=pairs)
        errors = [item["mirror_error_normalized"] for item in report["worst_offenders"]]

        self.assertEqual(len(report["worst_offenders"]), 12)
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertEqual(report["mirror_error_max"], max(errors))
        self.assertEqual(report["method"], "sagittal-mirror-bind-pose")
        self.assertEqual(report["schema_version"], 1)

    def test_the_p95_is_a_percentile_of_the_measured_pairs(self):
        positions = {"pelvis": np.zeros(3)}
        pairs = []
        for index in range(10):
            left_name, right_name = f"bone_{index}_l", f"bone_{index}_r"
            positions[left_name] = np.array([0.5, 0.0, 0.0])
            positions[right_name] = np.array([-0.5, float(index) * 0.1, 0.0])
            pairs.append((left_name, right_name))

        report = _measure(positions, {}, pairs=pairs)
        expected = float(
            np.percentile([abs(index) * 0.1 for index in range(10)], 95)
        )

        self.assertAlmostEqual(report["mirror_error_p95"], expected, places=6)
        self.assertAlmostEqual(report["mirror_error_max"], 0.9, places=6)
        self.assertTrue(math.isfinite(report["mirror_error_p95"]))


if __name__ == "__main__":
    unittest.main()
