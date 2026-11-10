import unittest

import numpy as np

from backend.app.joint_containment import (
    enclosed_volume,
    measure_joint_containment,
    points_inside_mesh,
)


def _box_triangles(low=(-1.0, -1.0, -1.0), high=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Closed axis-aligned box as a 12-triangle soup."""
    x0, y0, z0 = low
    x1, y1, z1 = high
    corners = np.asarray([
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ], dtype=np.float64)
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 5, 1), (0, 4, 5),  # -y
        (3, 2, 6), (3, 6, 7),  # +y
        (0, 3, 7), (0, 7, 4),  # -x
        (1, 5, 6), (1, 6, 2),  # +x
    ]
    return corners[np.asarray(faces)]


def _hollow_box(wall=0.05) -> np.ndarray:
    """A 2x2x2 box with an inverted inner box `wall` inside it -- a shell whose
    cavity is empty, the shape a "point in solid material" test gets wrong."""
    inner = _box_triangles(
        low=(-1.0 + wall,) * 3, high=(1.0 - wall,) * 3,
    )[:, ::-1]  # reversed winding: the cavity's normals face inward
    return np.concatenate([_box_triangles(), inner], axis=0)


class PointsInsideMeshTests(unittest.TestCase):
    def test_a_point_at_the_centre_of_a_closed_box_is_inside(self):
        inside = points_inside_mesh(np.asarray([[0.0, 0.0, 0.0]]), _box_triangles())

        self.assertTrue(bool(inside[0]))

    def test_a_point_beyond_every_face_is_outside(self):
        inside = points_inside_mesh(np.asarray([[5.0, 0.0, 0.0]]), _box_triangles())

        self.assertFalse(bool(inside[0]))

    def test_points_are_classified_independently(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [0.9, 0.9, 0.9],
            [1.5, 0.0, 0.0],
            [0.0, -3.0, 0.0],
        ])

        inside = points_inside_mesh(points, _box_triangles())

        self.assertEqual(inside.tolist(), [True, True, False, False])

    def test_a_point_inside_a_translated_box_is_still_inside(self):
        box = _box_triangles(low=(9.0, 9.0, 9.0), high=(11.0, 11.0, 11.0))

        inside = points_inside_mesh(np.asarray([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]]), box)

        self.assertEqual(inside.tolist(), [True, False])

    def test_an_empty_triangle_soup_classifies_everything_as_outside(self):
        inside = points_inside_mesh(np.asarray([[0.0, 0.0, 0.0]]), np.empty((0, 3, 3)))

        self.assertEqual(inside.tolist(), [False])

    def test_no_points_returns_an_empty_result(self):
        self.assertEqual(len(points_inside_mesh(np.empty((0, 3)), _box_triangles())), 0)

    def test_a_point_in_a_hollow_shells_cavity_is_inside(self):
        # The live shape this method exists for: Mary's rigged mesh is
        # watertight but encloses ~2% of its bounding box, so every joint
        # sits in the cavity rather than in solid material. A parity test
        # calls all of them outside; the S3 contract means "inside the body".
        hollow = _hollow_box()

        inside = points_inside_mesh(np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), hollow)

        self.assertEqual(inside.tolist(), [True, False])


class EnclosedVolumeTests(unittest.TestCase):
    def test_a_solid_box_encloses_its_own_volume(self):
        self.assertAlmostEqual(enclosed_volume(_box_triangles()), 8.0, places=6)

    def test_a_hollow_shell_encloses_almost_nothing(self):
        hollow = _hollow_box()

        # 2^3 - 1.9^3 = 1.141, against a bounding-box volume of 8: the shell
        # signature that rules out a solid-material containment test.
        self.assertAlmostEqual(enclosed_volume(hollow), 8.0 - 1.9 ** 3, places=6)

    def test_an_empty_soup_encloses_nothing(self):
        self.assertEqual(enclosed_volume(np.empty((0, 3, 3))), 0.0)


class MeasureJointContainmentTests(unittest.TestCase):
    def test_a_well_fitted_rig_reports_every_joint_inside(self):
        names = ["pelvis", "spine", "head"]
        positions = np.asarray([[0.0, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

        report = measure_joint_containment(names, positions, _box_triangles(), model_height=2.0)

        self.assertTrue(report["measurable"])
        self.assertEqual(report["joints_inside_mesh_fraction"], 1.0)
        self.assertEqual(report["joints_outside_mesh_count"], 0)
        self.assertEqual(report["inside_joints"], 3)
        self.assertEqual(report["worst_offenders"], [])

    def test_escaped_joints_are_counted_and_named_worst_first(self):
        names = ["pelvis", "hand_l", "hand_r"]
        positions = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

        report = measure_joint_containment(names, positions, _box_triangles(), model_height=2.0)

        self.assertEqual(report["joints_outside_mesh_count"], 2)
        self.assertAlmostEqual(report["joints_inside_mesh_fraction"], 1 / 3, places=6)
        self.assertEqual(
            [item["joint"] for item in report["worst_offenders"]],
            ["hand_l", "hand_r"],
        )
        self.assertGreater(
            report["worst_offenders"][0]["distance_to_surface"],
            report["worst_offenders"][1]["distance_to_surface"],
        )

    def test_the_offender_distance_is_normalized_by_model_height(self):
        report = measure_joint_containment(
            ["hand_l"], np.asarray([[4.0, 0.0, 0.0]]), _box_triangles(), model_height=2.0,
        )

        offender = report["worst_offenders"][0]
        self.assertAlmostEqual(
            offender["normalized_distance"], offender["distance_to_surface"] / 2.0, places=5,
        )

    def test_a_rig_without_skinned_triangles_is_unmeasurable_not_zero(self):
        report = measure_joint_containment(
            ["pelvis"], np.asarray([[0.0, 0.0, 0.0]]), np.empty((0, 3, 3)),
        )

        self.assertFalse(report["measurable"])
        self.assertEqual(report["reason"], "no skinned triangles")
        self.assertNotIn("joints_inside_mesh_fraction", report)

    def test_a_mesh_without_joints_is_unmeasurable(self):
        report = measure_joint_containment([], np.empty((0, 3)), _box_triangles())

        self.assertFalse(report["measurable"])
        self.assertEqual(report["reason"], "no joints")

    def test_mismatched_names_and_positions_are_rejected(self):
        with self.assertRaises(ValueError):
            measure_joint_containment(
                ["pelvis", "spine"], np.asarray([[0.0, 0.0, 0.0]]), _box_triangles(),
            )


if __name__ == "__main__":
    unittest.main()
