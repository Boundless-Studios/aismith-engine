import unittest

import numpy as np

from backend.app.geometry_contract import (
    extent_profile,
    scale_uniformity,
    uv_unwrap_stats,
)


class UvUnwrapStatsTests(unittest.TestCase):
    def test_a_real_unwrap_is_recorded_as_unwrapped_with_its_area(self):
        triangles = np.asarray([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ])

        stats = uv_unwrap_stats(triangles)

        self.assertTrue(stats["uv_unwrapped"])
        self.assertEqual(stats["uv_faces"], 2)
        self.assertAlmostEqual(stats["uv_total_area"], 1.0, places=6)
        self.assertEqual(stats["uv_non_degenerate_face_fraction"], 1.0)
        self.assertEqual(stats["uv_degenerate_faces"], 0)

    def test_a_mesh_with_no_uv_layer_is_not_unwrapped(self):
        stats = uv_unwrap_stats(np.empty((0, 3, 2)))

        self.assertFalse(stats["uv_unwrapped"])
        self.assertEqual(stats["uv_faces"], 0)
        self.assertEqual(stats["uv_total_area"], 0.0)

    def test_a_uv_layer_collapsed_onto_one_point_is_not_an_unwrap(self):
        # This is the failure a bare "does a UV layer exist?" check misses:
        # the layer is present but every texel maps to the same spot.
        triangles = np.zeros((32, 3, 2))

        stats = uv_unwrap_stats(triangles)

        self.assertFalse(stats["uv_unwrapped"])
        self.assertEqual(stats["uv_degenerate_faces"], 32)
        self.assertEqual(stats["uv_non_degenerate_face_fraction"], 0.0)

    def test_a_handful_of_degenerate_faces_still_counts_as_unwrapped(self):
        good = np.tile(
            np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), (400, 1, 1),
        )
        triangles = np.concatenate([good, np.zeros((1, 3, 2))], axis=0)

        stats = uv_unwrap_stats(triangles)

        self.assertTrue(stats["uv_unwrapped"])
        self.assertEqual(stats["uv_degenerate_faces"], 1)

    def test_winding_direction_does_not_flip_the_measured_area_negative(self):
        clockwise = np.asarray([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])

        self.assertAlmostEqual(uv_unwrap_stats(clockwise)["uv_total_area"], 0.5, places=6)


class ScaleUniformityTests(unittest.TestCase):
    def test_identity_transforms_are_uniform_and_identity(self):
        stats = scale_uniformity([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])

        self.assertTrue(stats["uniform_scale_ok"])
        self.assertTrue(stats["identity_scale"])
        self.assertEqual(stats["objects"], 2)
        self.assertEqual(stats["max_scale_anisotropy"], 0.0)

    def test_a_uniformly_scaled_object_is_uniform_but_not_identity(self):
        stats = scale_uniformity([(0.01, 0.01, 0.01)])

        self.assertTrue(stats["uniform_scale_ok"])
        self.assertFalse(stats["identity_scale"])

    def test_a_squashed_object_is_not_uniform(self):
        stats = scale_uniformity([(1.0, 1.0, 1.0), (1.0, 0.5, 1.0)])

        self.assertFalse(stats["uniform_scale_ok"])
        self.assertAlmostEqual(stats["max_scale_anisotropy"], 0.5, places=6)

    def test_a_mirrored_object_is_judged_on_magnitude_not_sign(self):
        stats = scale_uniformity([(1.0, -1.0, 1.0)])

        self.assertTrue(stats["uniform_scale_ok"])
        self.assertTrue(stats["identity_scale"])

    def test_no_objects_is_not_silently_reported_as_uniform(self):
        stats = scale_uniformity([])

        self.assertFalse(stats["uniform_scale_ok"])
        self.assertEqual(stats["objects"], 0)


class ExtentProfileTests(unittest.TestCase):
    def test_the_profile_is_scale_free(self):
        self.assertEqual(extent_profile((1.0, 2.0, 4.0)), extent_profile((10.0, 20.0, 40.0)))

    def test_a_degenerate_extent_does_not_divide_by_zero(self):
        self.assertEqual(extent_profile((0.0, 0.0, 0.0)), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
