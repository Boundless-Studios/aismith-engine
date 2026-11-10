import unittest

from backend.app.skin_weights import (
    blend_projected_weights,
    combine_weight_maps,
    smooth_weight_maps,
)


class SkinWeightProjectionTests(unittest.TestCase):
    def test_exact_donor_vertex_preserves_its_anatomical_weights(self):
        result = blend_projected_weights([
            (0.0, {"LeftUpLeg": 0.8, "Hips": 0.2}),
            (0.5, {"LeftLeg": 1.0}),
        ])

        self.assertAlmostEqual(result["LeftUpLeg"], 0.8)
        self.assertAlmostEqual(result["Hips"], 0.2)
        self.assertLess(result.get("LeftLeg", 0.0), 1e-9)

    def test_projection_is_normalized_and_limited_to_four_influences(self):
        result = blend_projected_weights([
            (0.25, {f"bone-{index}": 1.0})
            for index in range(6)
        ])

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_unweighted_donor_neighbors_return_no_assignment(self):
        self.assertEqual(blend_projected_weights([(0.1, {})]), {})

    def test_barycentric_maps_stay_on_one_donor_surface(self):
        result = combine_weight_maps([
            (0.6, {"LeftUpLeg": 1.0}),
            (0.3, {"LeftUpLeg": 0.5, "LeftLeg": 0.5}),
            (0.1, {"LeftLeg": 1.0}),
        ])

        self.assertAlmostEqual(result["LeftUpLeg"], 0.75)
        self.assertAlmostEqual(result["LeftLeg"], 0.25)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_topology_smoothing_softens_an_abrupt_joint_boundary(self):
        result = smooth_weight_maps(
            [{"upperarm": 1.0}, {"upperarm": 1.0}, {"forearm": 1.0}],
            [(0, 1), (1, 2)],
            factor=0.5,
            iterations=1,
        )

        self.assertEqual(result[0], {"upperarm": 1.0})
        self.assertGreater(result[1]["forearm"], 0.0)
        self.assertGreater(result[2]["upperarm"], 0.0)
        self.assertAlmostEqual(sum(result[1].values()), 1.0)

    def test_topology_smoothing_never_bleeds_across_disconnected_layers(self):
        result = smooth_weight_maps(
            [{"spine": 1.0}, {"spine": 1.0}, {"hand": 1.0}],
            [(0, 1)],
            factor=1.0,
            iterations=3,
        )

        self.assertEqual(result[2], {"hand": 1.0})

    def test_topology_smoothing_truncates_only_after_blending_own_weights(self):
        result = smooth_weight_maps(
            [
                {"shared": 1.0},
                {"a": 1.0},
                {"b": 1.0},
                {"c": 1.0},
                {"d": 1.0},
                {"shared": 1.0},
            ],
            [(0, neighbor) for neighbor in range(1, 6)],
            factor=0.5,
            iterations=1,
        )

        self.assertEqual(len(result[0]), 4)
        self.assertAlmostEqual(result[0]["shared"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
