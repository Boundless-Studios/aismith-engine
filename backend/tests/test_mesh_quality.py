import unittest

import numpy as np
import trimesh

from backend.app.mesh_quality import analyze_mesh


class MeshQualityTests(unittest.TestCase):
    def test_closed_box_reports_clean_objective_metrics(self):
        report = analyze_mesh(trimesh.creation.box(extents=(1.0, 2.0, 0.5)))

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["components"], 1)
        self.assertEqual(report["boundary_edges"], 0)
        self.assertEqual(report["non_manifold_edges"], 0)
        self.assertTrue(report["watertight"])
        self.assertTrue(report["finite"])
        self.assertEqual(report["bounds"]["extents"], [1.0, 2.0, 0.5])

    def test_disconnected_geometry_reports_component_ratio(self):
        large = trimesh.creation.box()
        small = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        small.apply_translation((3.0, 0.0, 0.0))

        report = analyze_mesh(trimesh.util.concatenate((large, small)))

        self.assertEqual(report["components"], 2)
        self.assertEqual(report["largest_component_face_ratio"], 0.5)

    def test_uv_style_duplicate_vertices_do_not_create_false_components(self):
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )

        report = analyze_mesh(mesh)

        self.assertEqual(report["components"], 1)
        self.assertEqual(report["boundary_edges"], 4)
        self.assertEqual(report["vertices"], 6)


if __name__ == "__main__":
    unittest.main()
