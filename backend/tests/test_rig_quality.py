import struct
import unittest

import numpy as np

from backend.app.rig_quality import _accessor_reader, _point_set_diameter, analyze_rig_document


class RigQualityTests(unittest.TestCase):
    def test_point_set_diameter_ignores_duplicate_vertex_density(self):
        sparse = np.asarray([[0, 0, 0], [0, 0, 2]], dtype=float)
        dense = np.asarray([[0, 0, 0]] * 100 + [[0, 0, 2]], dtype=float)

        self.assertEqual(_point_set_diameter(sparse), _point_set_diameter(dense))

    def test_point_set_diameter_is_independent_of_vertex_order(self):
        points = np.asarray([
            [-5, 4, 9], [-8, -7, 9], [6, 8, 2], [-5, -6, -2],
        ], dtype=float)

        self.assertEqual(
            _point_set_diameter(points),
            _point_set_diameter(points[[2, 0, 3, 1]]),
        )

    def test_landmarks_ignore_unreferenced_attribute_vertices(self):
        document = {
            "nodes": [{"name": "hand_r"}, {"mesh": 0, "skin": 0}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "indices": 3,
            }]}],
            "skins": [{"joints": [0]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 100]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]] * 3, dtype=int),
            2: np.asarray([[1, 0, 0, 0]] * 3, dtype=float),
            3: np.asarray([0, 1], dtype=int),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 1.0)
        self.assertEqual(report["skinned_vertices"], 2)

    def test_landmarks_ignore_nodes_outside_the_default_scene(self):
        document = {
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}, {"nodes": [0, 2]}],
            "nodes": [
                {"name": "hand_r"},
                {"mesh": 0, "skin": 0},
                {"name": "AISmith Sword Blade", "mesh": 1, "skin": 0},
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}}]},
                {"primitives": [{"attributes": {"POSITION": 1, "JOINTS_0": 2, "WEIGHTS_0": 3}}]},
            ],
            "skins": [{"joints": [0]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1]], dtype=float),
            1: np.asarray([[0, 0, 0], [0, 0, 100]], dtype=float),
            2: np.asarray([[0, 0, 0, 0]] * 2, dtype=int),
            3: np.asarray([[1, 0, 0, 0]] * 2, dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 1.0)
        self.assertEqual(report["skinned_primitives"], 1)

    def test_anatomical_extent_excludes_long_skinned_attachments(self):
        document = {
            "nodes": [
                {"name": "hand_r"},
                {"name": "sword_attachment"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 100], [0, 0, 200]], dtype=float),
            1: np.asarray([
                [0, 0, 0, 0], [0, 0, 0, 0],
                [1, 0, 0, 0], [1, 0, 0, 0],
            ], dtype=int),
            2: np.asarray([[1, 0, 0, 0]] * 4, dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 1.0)

    def test_anatomical_extent_excludes_hand_named_weapon_socket(self):
        document = {
            "nodes": [
                {"name": "hand_r"},
                {"name": "weapon_hand_socket"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 100]], dtype=float),
            1: np.asarray([
                [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
            ], dtype=int),
            2: np.asarray([
                [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
            ], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 1.0)

    def test_landmark_centroid_excludes_secondary_attachment_weights(self):
        document = {
            "nodes": [
                {"name": "head"},
                {"name": "hair_helper"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 100]], dtype=float),
            1: np.asarray([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0], [1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)
        head = next(item for item in report["joint_weight_coverage"] if item["joint"] == "head")

        self.assertEqual(head["weight_centroid"], [0.0, 0.0, 0.5])

    def test_bind_pose_errors_preserve_duplicate_joint_node_identity(self):
        second_inverse_bind = np.eye(4)
        second_inverse_bind[0, 3] = -2.0
        document = {
            "nodes": [
                {"name": "helper"},
                {"name": "helper"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0,
                "JOINTS_0": 1,
                "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
        }
        arrays = {
            0: np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=float),
            3: np.asarray([
                np.eye(4).reshape(16, order="F"),
                second_inverse_bind.reshape(16, order="F"),
            ]),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(
            [item["joint_position"] for item in report["bind_pose_landmark_errors"]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )

    def test_secondary_anatomical_joint_retains_a_landmark(self):
        document = {
            "nodes": [{"name": "head"}, {"name": "neck"}, {"mesh": 0, "skin": 0}],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 1]], dtype=float),
            1: np.asarray([[1, 0, 0, 0]], dtype=int),
            2: np.asarray([[0.51, 0.49, 0, 0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertIn("head", [item["joint"] for item in report["bind_pose_landmark_errors"]])

    def test_dominant_joint_ties_prefer_anatomical_over_helper_slots(self):
        document = {
            "nodes": [
                {"name": "hair_helper"},
                {"name": "head"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 1]], dtype=float),
            1: np.asarray([[0, 1, 0, 0]], dtype=int),
            2: np.asarray([[0.5, 0.5, 0, 0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertIn("head", [item["joint"] for item in report["bind_pose_landmark_errors"]])

        arrays[1] = np.asarray([[0, 1, 1, 0]], dtype=int)
        arrays[2] = np.asarray([[0.4, 0.3, 0.3, 0.0]], dtype=float)
        repeated = analyze_rig_document(document, arrays.__getitem__)
        self.assertIn(
            "head",
            [item["joint"] for item in repeated["bind_pose_landmark_errors"]],
        )

    def test_inverse_bind_origin_is_transformed_from_mesh_to_world_space(self):
        inverse_bind = np.eye(4)
        document = {
            "nodes": [{"name": "hand_r"}, {"mesh": 0, "skin": 0, "translation": [10, 0, 0]}],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0], "inverseBindMatrices": 3}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0]], dtype=float),
            3: np.asarray([inverse_bind.reshape(16, order="F")]),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["bind_pose_landmark_errors"][0]["joint_position"], [10.0, 0.0, 0.0])

    def test_shared_joint_bind_positions_are_compared_per_skin_instance(self):
        translated_inverse_bind = np.eye(4)
        translated_inverse_bind[0, 3] = 10.0
        document = {
            "nodes": [
                {"name": "hand_r"},
                {"mesh": 0, "skin": 0},
                {"mesh": 1, "skin": 1, "translation": [10, 0, 0]},
            ],
            "meshes": [
                {"primitives": [{"attributes": {
                    "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
                }}]},
                {"primitives": [{"attributes": {
                    "POSITION": 5, "JOINTS_0": 1, "WEIGHTS_0": 2,
                }}]},
            ],
            "skins": [
                {"joints": [0], "inverseBindMatrices": 3},
                {"joints": [0], "inverseBindMatrices": 4},
            ],
        }
        arrays = {
            0: np.asarray([[0, 0, 0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0]], dtype=float),
            3: np.asarray([np.eye(4).reshape(16, order="F")]),
            4: np.asarray([translated_inverse_bind.reshape(16, order="F")]),
            5: np.asarray([[-10, 0, 0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(
            [item["distance"] for item in report["bind_pose_landmark_errors"]],
            [0.0, 0.0],
        )

    def test_model_scale_is_invariant_to_scene_rotation(self):
        base_nodes = [{"name": "head"}, {"mesh": 0, "skin": 0}]
        rotated_nodes = [
            {"name": "head"},
            {"mesh": 0, "skin": 0, "rotation": [0.0, 0.0, 0.3826834, 0.9238795]},
        ]
        mesh = [{"primitives": [{"attributes": {
            "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
        }}]}]
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 2]], dtype=float),
            1: np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
        }

        base = analyze_rig_document({"nodes": base_nodes, "meshes": mesh, "skins": [{"joints": [0]}]}, arrays.__getitem__)
        rotated = analyze_rig_document({"nodes": rotated_nodes, "meshes": mesh, "skins": [{"joints": [0]}]}, arrays.__getitem__)

        self.assertEqual(base["model_height"], rotated["model_height"])

    def test_mismatched_bind_pose_cannot_inflate_its_normalization_scale(self):
        head_inverse_bind = np.eye(4)
        head_inverse_bind[2, 3] = -1.25
        document = {
            "nodes": [{"name": "pelvis"}, {"name": "head"}, {"mesh": 0, "skin": 0}],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 0, 1]], dtype=float),
            1: np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
            3: np.asarray([
                np.eye(4).reshape(16, order="F"),
                head_inverse_bind.reshape(16, order="F"),
            ]),
        }

        report = analyze_rig_document(document, arrays.__getitem__)
        head = next(
            item for item in report["bind_pose_landmark_errors"]
            if item["joint"] == "head"
        )

        self.assertEqual(report["model_height"], 1.0)
        self.assertEqual(head["normalized_error"], 0.25)

    def test_rigid_skinned_prop_is_not_an_anatomical_landmark_or_scale_source(self):
        document = {
            "nodes": [
                {"name": "pelvis"},
                {"name": "hand_r"},
                {"mesh": 0, "skin": 0},
                {"name": "AISmith Sword Blade", "mesh": 1, "skin": 0},
            ],
            "meshes": [
                {"primitives": [{"attributes": {
                    "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
                }}]},
                {"primitives": [{"attributes": {
                    "POSITION": 4, "JOINTS_0": 5, "WEIGHTS_0": 6,
                }}]},
            ],
            "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
        }
        arrays = {
            0: np.asarray([
                [0, 0, 2 * index / 29]
                for index in range(30)
            ], dtype=float),
            1: np.asarray([
                [0 if index < 15 else 1, 0, 0, 0]
                for index in range(30)
            ], dtype=int),
            2: np.asarray([[1, 0, 0, 0]] * 30, dtype=float),
            3: np.asarray([
                np.eye(4).reshape(16, order="F"),
                np.eye(4).reshape(16, order="F"),
            ]),
            4: np.asarray([[100, 0, 0], [101, 0, 0]], dtype=float),
            5: np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=int),
            6: np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 2.0)
        self.assertEqual(
            [item["joint"] for item in report["bind_pose_landmark_errors"]],
            ["pelvis", "hand_r"],
        )

    def test_reports_normalized_bind_pose_landmark_error(self):
        document = {
            "nodes": [
                {"name": "hand_r", "translation": [0.0, 0.0, 1.0]},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0,
                "JOINTS_0": 1,
                "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0], "inverseBindMatrices": 3}],
        }
        arrays = {
            0: np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=float),
            3: np.asarray([
                np.asarray([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]).reshape(16, order="F"),
            ]),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["model_height"], 2.0)
        self.assertEqual(report["bind_pose_landmark_errors"], [{
            "joint_node": 0,
            "joint": "hand_r",
            "distance": 0.0,
            "normalized_error": 0.0,
            "joint_position": [0.0, 0.0, 1.0],
            "weight_centroid": [0.0, 0.0, 1.0],
        }])

    def test_omitted_inverse_bind_matrices_use_identity_in_mesh_space(self):
        document = {
            "nodes": [
                {"name": "hand_r", "translation": [0.0, 0.0, 1.0]},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0]}],
        }
        arrays = {
            0: np.asarray([[0.0, 0.0, 1.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(
            report["bind_pose_landmark_errors"][0]["joint_position"],
            [0.0, 0.0, 0.0],
        )

    def test_default_morph_weights_are_applied_to_landmark_geometry(self):
        document = {
            "nodes": [{"name": "hand_r"}, {"mesh": 0, "skin": 0, "weights": [1.0]}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "targets": [{"POSITION": 3}],
            }]}],
            "skins": [{"joints": [0]}],
        }
        arrays = {
            0: np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            3: np.asarray([[0.0, 0.0, 2.0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(
            report["bind_pose_landmark_errors"][0]["weight_centroid"],
            [0.0, 0.0, 2.0],
        )

    def test_reports_skin_weights_and_animation_coverage(self):
        document = {
            "nodes": [
                {"name": "mesh", "mesh": 0, "skin": 0},
                {"name": "root", "children": [2]},
                {"name": "pelvis", "children": [3]},
                {"name": "head"},
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "JOINTS_0": 1,
                                "WEIGHTS_0": 2,
                            }
                        }
                    ]
                }
            ],
            "skins": [{"joints": [1, 2, 3], "skeleton": 1}],
            "animations": [
                {
                    "name": "Sword_Attack",
                    "samplers": [{"input": 3, "output": 4}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 1, "path": "rotation"}}
                    ],
                }
            ],
        }
        accessors = {
            0: np.zeros((3, 3), dtype=np.float32),
            1: np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
            2: np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
            3: np.array([0.0, 1.5], dtype=np.float32),
            4: np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.1, 0.0, 0.995]], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["skins"], 1)
        self.assertEqual(report["joints"], 3)
        self.assertEqual(report["skinned_vertices"], 3)
        self.assertEqual(report["unweighted_vertices"], 1)
        self.assertEqual(report["non_normalized_vertices"], 1)
        self.assertEqual(report["maximum_influences"], 2)
        self.assertEqual(report["unused_joints"], 1)
        self.assertEqual(report["animations"][0]["name"], "Sword_Attack")
        self.assertEqual(report["animations"][0]["duration_seconds"], 1.5)
        self.assertEqual(report["animations"][0]["animated_joint_coverage"], 1 / 3)
        self.assertEqual(report["animations"][0]["animated_joint_names"], ["root"])
        pelvis = next(
            item for item in report["joint_weight_coverage"]
            if item["joint"] == "pelvis"
        )
        self.assertEqual(pelvis["weight_centroid"], [0.0, 0.0, 0.0])

    def test_constant_export_sampling_does_not_count_as_animated_joint_coverage(self):
        document = {
            "nodes": [{"name": "pelvis"}, {"name": "head"}],
            "skins": [{"joints": [0, 1]}],
            "animations": [{
                "samplers": [{"input": 0, "output": 1}],
                "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["animations"][0]["animated_joints"], 0)
        self.assertEqual(report["animations"][0]["animated_joint_coverage"], 0.0)

    def test_antipodal_quaternion_samples_are_the_same_orientation(self):
        document = {
            "nodes": [{"name": "pelvis"}],
            "skins": [{"joints": [0]}],
            "animations": [{
                "samplers": [{"input": 0, "output": 1}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["animations"][0]["animated_joints"], 0)

    def test_cubic_tangents_count_as_motion_when_key_values_match(self):
        document = {
            "nodes": [{"name": "pelvis"}],
            "skins": [{"joints": [0]}],
            "animations": [{
                "samplers": [{"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            ], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["animations"][0]["animated_joints"], 1)

    def test_cubic_quaternion_tangents_parallel_to_keys_do_not_count_as_rotation(self):
        document = {
            "nodes": [{"name": "pelvis"}],
            "skins": [{"joints": [0]}],
            "animations": [{
                "samplers": [{"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([
                [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 5.0],
            ], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["animations"][0]["animated_joints"], 0)

    def test_unused_cubic_endpoint_tangents_do_not_count_as_motion(self):
        document = {
            "nodes": [{"name": "pelvis"}],
            "skins": [{"joints": [0]}],
            "animations": [{
                "samplers": [{"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            ], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["animations"][0]["animated_joints"], 0)

    def test_invalid_animation_sampler_counts_as_invalid_target(self):
        document = {
            "nodes": [{"name": "pelvis"}],
            "skins": [{"joints": [0]}],
            "animations": [{
                "samplers": [],
                "channels": [{"sampler": 3, "target": {"node": 0, "path": "rotation"}}],
            }],
        }

        report = analyze_rig_document(document, lambda _index: np.zeros((2, 4), dtype=np.float32))

        self.assertEqual(report["animations"][0]["invalid_targets"], 1)

    def test_animation_coverage_uses_certified_character_skin_in_multi_skin_asset(self):
        document = {
            "nodes": [
                {"name": "pelvis"},
                {"name": "head"},
                {"name": "sword_root"},
                {"name": "sword_tip"},
                {"name": "finger_helper"},
                {"name": "twist_helper"},
            ],
            "skins": [{"joints": [0, 1, 4, 5]}, {"joints": [2, 3]}],
            "animations": [{
                "samplers": [
                    {"input": 0, "output": 1},
                    {"input": 0, "output": 2},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                ],
            }],
        }
        accessors = {
            0: np.array([0.0, 1.0], dtype=np.float32),
            1: np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.1, 0.0, 0.995]], dtype=np.float32),
            2: np.array([[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.995]], dtype=np.float32),
        }

        report = analyze_rig_document(
            document,
            lambda index: accessors[index],
            preferred_joint_names={"pelvis", "head"},
        )

        animation = report["animations"][0]
        self.assertEqual(animation["animation_joint_scope"], 2)
        self.assertEqual(animation["animated_joints"], 2)
        self.assertEqual(animation["animated_joint_coverage"], 1.0)

    def test_unrigged_mesh_is_reported_without_crashing(self):
        document = {
            "nodes": [{"name": "mesh", "mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        }

        report = analyze_rig_document(
            document,
            lambda _index: np.zeros((8, 3), dtype=np.float32),
        )

        self.assertEqual(report["skins"], 0)
        self.assertEqual(report["joints"], 0)
        self.assertEqual(report["unskinned_primitives"], 1)

    def test_rigid_joint_attachment_is_not_unskinned_character_geometry(self):
        document = {
            "nodes": [
                {"name": "hand", "children": [1]},
                {"name": "sword", "mesh": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "skins": [{"joints": [0]}],
        }

        report = analyze_rig_document(
            document,
            lambda _index: np.zeros((120, 3), dtype=np.float32),
        )

        self.assertEqual(report["unskinned_vertices"], 0)
        self.assertEqual(report["rigid_attachment_vertices"], 120)
        self.assertEqual(report["rigid_attachment_primitives"], 1)
        self.assertEqual(report["animations"], [])

    def test_aggregates_secondary_skin_weight_attribute_sets(self):
        document = {
            "nodes": [
                {"name": "mesh", "mesh": 0, "skin": 0},
                {"name": "pelvis"},
                {"name": "head"},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0,
                "JOINTS_0": 1,
                "WEIGHTS_0": 2,
                "JOINTS_1": 3,
                "WEIGHTS_1": 4,
            }}]}],
            "skins": [{"joints": [1, 2]}],
        }
        accessors = {
            0: np.zeros((1, 3), dtype=np.float32),
            1: np.array([[0, 0, 0, 0]], dtype=np.int32),
            2: np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32),
            3: np.array([[1, 0, 0, 0]], dtype=np.int32),
            4: np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["non_normalized_vertices"], 0)
        self.assertEqual(report["maximum_influences"], 2)
        self.assertEqual(report["unused_joints"], 0)

    def test_materializes_sparse_accessor_over_zero_initialized_base(self):
        binary = struct.pack("<H", 1) + b"\x00\x00" + struct.pack("<fff", 0.25, 0.5, 0.75)
        document = {
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 2},
                {"buffer": 0, "byteOffset": 4, "byteLength": 12},
            ],
            "accessors": [{
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5123},
                    "values": {"bufferView": 1},
                },
            }],
        }

        values = _accessor_reader(document, binary)(0)

        np.testing.assert_array_equal(values[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(values[1], [0.25, 0.5, 0.75])
        np.testing.assert_array_equal(values[2], [0.0, 0.0, 0.0])

    def test_reports_vertices_from_unskinned_character_geometry(self):
        document = {
            "nodes": [
                {"name": "body", "mesh": 0},
                {"name": "shirt", "mesh": 1, "skin": 0},
                {"name": "pelvis"},
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0}}]},
                {"primitives": [{"attributes": {"POSITION": 1, "JOINTS_0": 2, "WEIGHTS_0": 3}}]},
            ],
            "skins": [{"joints": [2]}],
        }
        accessors = {
            0: np.zeros((100, 3), dtype=np.float32),
            1: np.zeros((5, 3), dtype=np.float32),
            2: np.zeros((5, 4), dtype=np.int32),
            3: np.array([[1.0, 0.0, 0.0, 0.0]] * 5, dtype=np.float32),
        }

        report = analyze_rig_document(document, lambda index: accessors[index])

        self.assertEqual(report["unskinned_vertices"], 100)

    def _humanoid_handedness_document(self, mirrored: bool) -> dict:
        # Exported GLBs from this pipeline are Y-up (vertical=1); the toe
        # (foot's child) points -Z here, so the true facing is -Z and
        # left = up x forward = (-1, 0, 0) -- left is -X.
        left_x, right_x = (-0.5, 0.5) if not mirrored else (0.5, -0.5)
        left_thigh_x, right_thigh_x = (-0.15, 0.15) if not mirrored else (0.15, -0.15)
        return {
            "nodes": [
                {"name": "pelvis", "translation": [0.0, 1.0, 0.0], "children": [9]},
                {"name": "upper_arm_l", "translation": [left_x, 1.5, 0.0]},
                {"name": "upper_arm_r", "translation": [right_x, 1.5, 0.0]},
                {"name": "thigh_l", "translation": [left_thigh_x, 0.9, 0.0]},
                {"name": "thigh_r", "translation": [right_thigh_x, 0.9, 0.0]},
                {"name": "foot_l", "translation": [left_thigh_x, 0.05, -0.1], "children": [7]},
                {"name": "foot_r", "translation": [right_thigh_x, 0.05, -0.1], "children": [8]},
                {"name": "toe_l", "translation": [left_thigh_x, 0.02, -0.25]},
                {"name": "toe_r", "translation": [right_thigh_x, 0.02, -0.25]},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
            }]}],
            "skins": [{"joints": [0, 1, 2, 3, 4, 5, 6, 7, 8]}],
        }

    def _humanoid_handedness_accessors(self) -> dict:
        return {
            0: np.asarray([[0.0, 1.0, 0.0], [0.0, 1.1, 0.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]] * 2, dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=float),
        }

    def test_handedness_flags_mirrored_left_right_bones(self):
        document = self._humanoid_handedness_document(mirrored=True)
        arrays = self._humanoid_handedness_accessors()

        report = analyze_rig_document(document, arrays.__getitem__)

        handedness = report["handedness"]
        self.assertIsNotNone(handedness)
        self.assertEqual(handedness["facing_source"], "foot-toe-direction")
        mismatched_roles = {item["role"] for item in handedness["mismatches"]}
        self.assertEqual(
            mismatched_roles,
            {"upper_arm_l", "upper_arm_r", "thigh_l", "thigh_r"},
        )

    def test_handedness_clean_when_sides_match_facing(self):
        document = self._humanoid_handedness_document(mirrored=False)
        arrays = self._humanoid_handedness_accessors()

        report = analyze_rig_document(document, arrays.__getitem__)

        handedness = report["handedness"]
        self.assertIsNotNone(handedness)
        self.assertEqual(handedness["facing_source"], "foot-toe-direction")
        self.assertEqual(handedness["mismatches"], [])

    def test_handedness_is_none_without_resolvable_feet(self):
        document = {
            "nodes": [
                {"name": "bone_0", "translation": [0.0, 1.0, 0.0]},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
            }]}],
            "skins": [{"joints": [0]}],
        }
        arrays = {
            0: np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]], dtype=int),
            2: np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertIsNone(report["handedness"])

    def test_handedness_role_aliases_agree_with_quality_policy_aliases(self):
        # rig_quality._HANDEDNESS_ROLE_ALIASES intentionally duplicates a
        # subset of quality_policy._REQUIRED_HUMANOID_ROLES (kept local to
        # avoid rig_quality depending on the policy layer) -- guard against
        # the two silently drifting apart.
        from backend.app.rig_quality import _HANDEDNESS_ROLE_ALIASES
        from backend.app.quality_policy import _REQUIRED_HUMANOID_ROLES

        key_map = {
            "pelvis": "pelvis",
            "upper_arm_l": "left_upper_arm",
            "upper_arm_r": "right_upper_arm",
            "thigh_l": "left_upper_leg",
            "thigh_r": "right_upper_leg",
            "foot_l": "left_foot",
            "foot_r": "right_foot",
        }
        self.assertEqual(set(_HANDEDNESS_ROLE_ALIASES), set(key_map))
        for rig_quality_role, policy_role in key_map.items():
            self.assertEqual(
                _HANDEDNESS_ROLE_ALIASES[rig_quality_role],
                _REQUIRED_HUMANOID_ROLES[policy_role],
                f"alias mismatch: rig_quality[{rig_quality_role!r}] vs "
                f"quality_policy[{policy_role!r}]",
            )


_RING_OFFSETS = np.asarray([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def _linear_chain_document(joint_names: list[str], world_positions: np.ndarray, mesh_ring_centers: np.ndarray, ring_radius: float = 0.05):
    """A straight-line chain rig: `joint_names[i]` is a node at
    `world_positions[i]`, linearly parented (node i's parent is node i-1).
    Each joint gets 4 rigidly-weighted mesh vertices in a small ring around
    the corresponding `mesh_ring_centers[i]` -- usually the same as the
    joint's own position, except where a test deliberately separates them
    to simulate a bone pulled off the mesh's medial axis.
    """
    count = len(joint_names)
    nodes = []
    for i, name in enumerate(joint_names):
        local = world_positions[i] - (world_positions[i - 1] if i > 0 else np.zeros(3))
        children = [i + 1] if i + 1 < count else []
        nodes.append({"name": name, "translation": local.tolist(), "children": children})
    mesh_node_index = count
    nodes.append({"mesh": 0, "skin": 0})

    positions = np.concatenate([mesh_ring_centers[i] + _RING_OFFSETS * ring_radius for i in range(count)], axis=0)
    joints = np.concatenate([
        np.tile([i, 0, 0, 0], (4, 1)) for i in range(count)
    ], axis=0)
    weights = np.tile([1.0, 0.0, 0.0, 0.0], (count * 4, 1))

    document = {
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
        "skins": [{"joints": list(range(count))}],
    }
    arrays = {0: positions, 1: joints, 2: weights}
    return document, arrays, mesh_node_index


class ChainRigAnalysisTests(unittest.TestCase):
    """Coverage for the chain-profile additions to analyze_rig_document:
    profile detection from bone names, per-bone medial-axis deviation, and
    branch detection from the exported node hierarchy."""

    def _straight_chain(self, spine_count: int = 10):
        joint_names = [f"spine_{i:02d}" for i in range(spine_count)] + ["head"]
        positions = np.asarray([[float(i), 0.0, 0.0] for i in range(spine_count + 1)])
        return joint_names, positions

    def test_profile_is_detected_and_recorded_for_a_chain_rig(self):
        joint_names, positions = self._straight_chain()
        document, arrays, _ = _linear_chain_document(joint_names, positions, positions)

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "chain")
        self.assertEqual(report["chain_topology"], {"is_single_chain": True, "branch_joints": []})

    def test_profile_is_humanoid_for_an_ordinary_humanoid_rig_name_set(self):
        joint_names = ["pelvis", "spine", "head", "upperarm_l", "lowerarm_l", "hand_l"]
        positions = np.asarray([[0.0, float(i), 0.0] for i in range(len(joint_names))])
        document, arrays, _ = _linear_chain_document(joint_names, positions, positions)

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "humanoid")
        self.assertIsNone(report["chain_topology"])
        self.assertEqual(report["chain_axis_errors"], [])

    def test_chain_axis_error_flags_a_bone_pulled_off_the_medial_axis(self):
        joint_names, positions = self._straight_chain()
        # The mesh stays on the straight x-axis tube everywhere, but the
        # bone at "spine_05" (index 5) is itself pulled 2 units sideways --
        # its own vertex ring, however, is left on-axis at its correct body
        # location, exactly as if the bone's *position* were wrong while
        # the *surface* it deforms is not.
        mesh_ring_centers = positions.copy()
        off_axis_positions = positions.copy()
        off_axis_positions[5] = [5.0, 2.0, 0.0]

        document, arrays, _ = _linear_chain_document(joint_names, off_axis_positions, mesh_ring_centers)
        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "chain")
        errors_by_joint = {item["joint"]: item for item in report["chain_axis_errors"]}
        self.assertGreater(errors_by_joint["spine_05"]["axis_deviation_ratio"], 0.9)
        for name in ("spine_02", "spine_08", "head"):
            self.assertLess(errors_by_joint[name]["axis_deviation_ratio"], 0.01)

    def test_chain_axis_errors_are_unaffected_by_collapsed_skinning_weights(self):
        # Regression for a real false-positive: the off-axis measurement
        # used to be built from each joint's *dominantly-weighted* skin
        # vertices, not a genuine spatial slab. A bone-heat/envelope solver
        # collapsing weight onto one end bone (the real "slither test rig"
        # failure mode this module's weighting fallback exists for) then
        # starved every other joint of any measurement at all, and handed
        # the collapsed end bone a vertex cloud spanning the whole chain --
        # wildly inflating its deviation despite the mesh geometry itself
        # (and every bone's true position) being perfectly on-axis. The
        # fixed measurement is purely spatial (a slab of mesh vertices near
        # each joint's own position), so it must be unaffected by this
        # collapse: every joint should still measure a small, correctly
        # local deviation.
        joint_names, positions = self._straight_chain()
        document, arrays, _ = _linear_chain_document(joint_names, positions, positions)
        collapsed_joints = np.zeros_like(arrays[1])  # every vertex "weighted" to joint 0
        arrays = {0: arrays[0], 1: collapsed_joints, 2: arrays[2]}

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "chain")
        errors_by_joint = {item["joint"]: item for item in report["chain_axis_errors"]}
        for name in ("spine_00", "spine_05", "head"):
            self.assertIn(name, errors_by_joint)
            self.assertLess(errors_by_joint[name]["axis_deviation_ratio"], 0.01)

    def test_chain_axis_errors_ignore_a_neighbouring_coil_of_the_same_body(self):
        # Regression for the live serpent false positive: a coiled body
        # brings another stretch of itself within a joint's slab. The slab
        # at each joint of the outgoing run cut straight through the
        # returning run 0.3 units away, so "local radius" and the centroid
        # offset were both dominated by the other coil and every mid-body
        # joint reported ratio ~0.95 while sitting exactly on its own
        # medial line. Vertices must be attributed to their nearest chain
        # segment before a joint measures its slab.
        joint_names = [f"spine_{i:02d}" for i in range(10)] + ["head"]
        outgoing = [[float(i), 0.0, 0.0] for i in range(5)]           # spine_00..spine_04
        turn = [[4.5, 0.0, 0.15]]                                       # spine_05
        returning = [[float(4 - i), 0.0, 0.3] for i in range(5)]       # spine_06..spine_09, head
        positions = np.asarray(outgoing + turn + returning)
        document, arrays, _ = _linear_chain_document(joint_names, positions, positions)

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "chain")
        errors_by_joint = {item["joint"]: item for item in report["chain_axis_errors"]}
        for name in ("spine_01", "spine_02", "spine_03", "spine_07", "spine_08"):
            self.assertLess(errors_by_joint[name]["axis_deviation_ratio"], 0.05, name)
            self.assertLess(errors_by_joint[name]["local_radius"], 0.1, name)

    def test_chain_topology_detects_a_branch(self):
        joint_names, positions = self._straight_chain()
        document, arrays, _ = _linear_chain_document(joint_names, positions, positions)
        # Graft a second chain-vocabulary branch onto spine_05 (index 5),
        # alongside its real continuation (spine_06, index 6) -- a genuine
        # fork the exported hierarchy shows even though every bone name
        # individually looks like valid chain vocabulary. Appended after
        # the mesh node, so no existing node index shifts.
        extra_index = len(document["nodes"])
        document["nodes"][5]["children"].append(extra_index)
        document["nodes"].append({"name": "spine_99", "translation": [0.0, 1.0, 0.0], "children": []})
        document["skins"][0]["joints"].append(extra_index)

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["profile"], "chain")
        self.assertFalse(report["chain_topology"]["is_single_chain"])
        self.assertIn("spine_05", report["chain_topology"]["branch_joints"])

    def test_chain_topology_root_as_elbow_of_a_single_path_is_not_a_branch(self):
        # Mirrors the real UniRig serpent this module was built against: the
        # predicted root sits mid-body, with the skeleton's true diameter
        # path passing through *two* of the root's branches (one climbing
        # to the head, one descending to the tail). A root with exactly two
        # chain-vocabulary children is the elbow of one valid path, not a
        # fork -- regression coverage for a false positive the undirected
        # degree check below replaced a children-count-only check to fix.
        from backend.app.rig_quality import _chain_topology
        joint_names_by_index = {
            0: "spine_02", 1: "spine_01", 2: "spine_00",
            3: "spine_03", 4: "spine_04", 5: "head",
        }
        children_by_node = {0: [1, 3], 1: [2], 2: [], 3: [4], 4: [5], 5: []}
        parent_by_node = {1: 0, 2: 1, 3: 0, 4: 3, 5: 4}
        chain_indices = set(joint_names_by_index)

        result = _chain_topology(chain_indices, children_by_node, parent_by_node, joint_names_by_index)

        self.assertEqual(result, {"is_single_chain": True, "branch_joints": []})

    def test_chain_topology_root_with_three_children_is_a_branch(self):
        # A root has no parent, so all its neighbors are children -- three
        # chain-vocabulary children (undirected degree 3) is a genuine fork
        # even though the root itself has no "incoming" edge to count.
        from backend.app.rig_quality import _chain_topology
        joint_names_by_index = {0: "spine_00", 1: "spine_01", 2: "spine_02", 3: "head"}
        children_by_node = {0: [1, 2, 3], 1: [], 2: [], 3: []}
        parent_by_node = {1: 0, 2: 0, 3: 0}
        chain_indices = set(joint_names_by_index)

        result = _chain_topology(chain_indices, children_by_node, parent_by_node, joint_names_by_index)

        self.assertFalse(result["is_single_chain"])
        self.assertIn("spine_00", result["branch_joints"])


class TemplateConformanceExtrasTests(unittest.TestCase):
    """N3: `unirig_bind_worker` computes biped-template conformance findings
    at bind time (with exact bpy edit-bone data) and stashes them as a
    custom property on the armature object, which `export_extras=True`
    carries through to the exported GLB as that node's `extras`. These
    tests exercise `analyze_rig_document` reading that back -- no bpy or
    real GLB export involved, just the same raw-document-dict fixtures the
    rest of this file uses."""

    def _document(self, armature_extras):
        return {
            "nodes": [
                {"name": "UniRigArmature", "extras": armature_extras},
                {"name": "pelvis"},
                {"name": "head"},
                {"mesh": 0, "skin": 0},
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "indices": 3,
            }]}],
            "skins": [{"joints": [1, 2]}],
        }

    def _arrays(self):
        return {
            0: np.asarray([[0, 0, 0], [0, 0, 1]], dtype=float),
            1: np.asarray([[0, 0, 0, 0]] * 2, dtype=int),
            2: np.asarray([[1, 0, 0, 0]] * 2, dtype=float),
            3: np.asarray([0, 1], dtype=int),
        }

    def test_conformant_rig_carries_empty_findings_list_not_none(self):
        document = self._document({"template_conformance": []})

        report = analyze_rig_document(document, self._arrays().__getitem__)

        self.assertEqual(report["template_conformance"], [])

    def test_non_conformant_findings_round_trip(self):
        finding = {
            "code": "missing_required_bone",
            "severity": "error",
            "message": "required bone 'hand_l' is missing from the candidate rig",
            "metric": "required_bone_present",
            "actual": False,
            "expected": "'hand_l' present",
        }
        document = self._document({"template_conformance": [finding]})

        report = analyze_rig_document(document, self._arrays().__getitem__)

        self.assertEqual(report["template_conformance"], [finding])

    def test_rig_with_no_conformance_extras_reports_none(self):
        document = self._document({})

        report = analyze_rig_document(document, self._arrays().__getitem__)

        self.assertIsNone(report["template_conformance"])

    def test_rig_with_no_extras_at_all_reports_none(self):
        document = {
            "nodes": [{"name": "UniRigArmature"}, {"name": "pelvis"}, {"name": "head"},
                      {"mesh": 0, "skin": 0}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "indices": 3,
            }]}],
            "skins": [{"joints": [1, 2]}],
        }

        report = analyze_rig_document(document, self._arrays().__getitem__)

        self.assertIsNone(report["template_conformance"])


# A minimal template-named biped in the EXPORTED convention (Y-up), facing
# +Z, so `left = up x forward` is +X. Positions are world-space; the builder
# below turns them into local translations plus a matching set of inverse
# bind matrices.
_PAIRED_SKELETON = [
    ("root", None, (0.0, 0.0, 0.0)),
    ("pelvis", "root", (0.0, 1.00, 0.0)),
    ("spine_1", "pelvis", (0.0, 1.15, 0.0)),
    ("spine_2", "spine_1", (0.0, 1.30, 0.0)),
    ("spine_3", "spine_2", (0.0, 1.45, 0.0)),
    ("neck", "spine_3", (0.0, 1.60, 0.0)),
    ("head", "neck", (0.0, 1.70, 0.0)),
    ("clavicle_l", "spine_3", (0.05, 1.55, 0.0)),
    ("upper_arm_l", "clavicle_l", (0.20, 1.50, 0.0)),
    ("lower_arm_l", "upper_arm_l", (0.45, 1.50, 0.0)),
    ("hand_l", "lower_arm_l", (0.70, 1.50, 0.0)),
    ("clavicle_r", "spine_3", (-0.05, 1.55, 0.0)),
    ("upper_arm_r", "clavicle_r", (-0.20, 1.50, 0.0)),
    ("lower_arm_r", "upper_arm_r", (-0.45, 1.50, 0.0)),
    ("hand_r", "lower_arm_r", (-0.70, 1.50, 0.0)),
    ("thigh_l", "pelvis", (0.10, 0.95, 0.0)),
    ("calf_l", "thigh_l", (0.10, 0.55, 0.0)),
    ("foot_l", "calf_l", (0.10, 0.10, 0.0)),
    ("toes_l", "foot_l", (0.10, 0.05, 0.15)),
    ("thigh_r", "pelvis", (-0.10, 0.95, 0.0)),
    ("calf_r", "thigh_r", (-0.10, 0.55, 0.0)),
    ("foot_r", "calf_r", (-0.10, 0.10, 0.0)),
    ("toes_r", "foot_r", (-0.10, 0.05, 0.15)),
]
# Symmetric about every axis, so a joint's vertex ring never shifts the
# facing measurement or the model's own centroid.
_SYMMETRIC_RING = np.asarray([
    [0.03, 0.0, 0.0], [-0.03, 0.0, 0.0], [0.0, 0.0, 0.03], [0.0, 0.0, -0.03],
])


def _paired_skeleton_document(displacements=None, drop_toes=False):
    """A template-named humanoid glTF document with real inverse bind
    matrices, so joint heads read back at the positions declared above
    rather than collapsing onto the mesh node's origin.

    `displacements` moves named joints AFTER the mirror-perfect layout is
    built (the failure the symmetry measure exists to catch); `drop_toes`
    removes the toe joints, which is what turns the facing detection from
    "foot-toe-direction" into an assumed guess.
    """
    entries = [
        entry for entry in _PAIRED_SKELETON
        if not (drop_toes and entry[0].startswith("toes_"))
    ]
    world = {name: np.asarray(position, dtype=float) for name, _, position in entries}
    for name, offset in (displacements or {}).items():
        world[name] = world[name] + np.asarray(offset, dtype=float)

    index_by_name = {name: index for index, (name, _, _) in enumerate(entries)}
    nodes = []
    for name, parent, _ in entries:
        local = world[name] - (world[parent] if parent else np.zeros(3))
        children = [
            index_by_name[child]
            for child, child_parent, _ in entries
            if child_parent == name
        ]
        nodes.append({"name": name, "translation": local.tolist(), "children": children})
    nodes.append({"mesh": 0, "skin": 0})

    joint_indices = list(range(len(entries)))
    positions = np.concatenate(
        [world[name] + _SYMMETRIC_RING for name, _, _ in entries], axis=0
    )
    joints = np.concatenate(
        [np.tile([index, 0, 0, 0], (len(_SYMMETRIC_RING), 1)) for index in joint_indices],
        axis=0,
    )
    weights = np.tile([1.0, 0.0, 0.0, 0.0], (len(positions), 1))
    inverse_binds = []
    for name, _, _ in entries:
        matrix = np.eye(4)
        matrix[:3, 3] = -world[name]
        inverse_binds.append(matrix.flatten(order="F"))

    document = {
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
        }}]}],
        "skins": [{"joints": joint_indices, "inverseBindMatrices": 3}],
    }
    arrays = {
        0: positions,
        1: joints,
        2: weights,
        3: np.asarray(inverse_binds, dtype=float),
    }
    return document, arrays


class BilateralSymmetryReportTests(unittest.TestCase):
    """`analyze_rig_document` measures left/right placement and mirrors the
    scalars `pipeline_policy`'s S3 check reads at the report's top level."""

    def test_a_mirror_perfect_rig_is_measurable_and_clean(self):
        document, arrays = _paired_skeleton_document()

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertEqual(report["handedness"]["facing_source"], "foot-toe-direction")
        symmetry = report["bilateral_symmetry"]
        self.assertTrue(symmetry["measurable"])
        self.assertEqual(symmetry["method"], "sagittal-mirror-bind-pose")
        # clavicle, upper_arm, lower_arm, hand, thigh, calf, foot, toes.
        self.assertEqual(symmetry["pairs_measured"], 8)
        self.assertAlmostEqual(symmetry["mirror_error_max"], 0.0, places=6)
        self.assertEqual(report["symmetry_mirror_error_p95"], symmetry["mirror_error_p95"])
        self.assertEqual(report["symmetry_pairs_outside_band_count"], 0)

    def test_the_finger_pairs_the_rig_lacks_are_reported_as_skipped(self):
        document, arrays = _paired_skeleton_document()

        symmetry = analyze_rig_document(document, arrays.__getitem__)["bilateral_symmetry"]

        self.assertIn("index_1_l/index_1_r", symmetry["pairs_skipped"])
        self.assertTrue(any("skipped" in caveat for caveat in symmetry["caveats"]))

    def test_a_displaced_right_hand_is_a_real_asymmetry(self):
        document, arrays = _paired_skeleton_document(
            displacements={"hand_r": (0.0, 0.2, 0.0)}
        )

        report = analyze_rig_document(document, arrays.__getitem__)
        symmetry = report["bilateral_symmetry"]

        self.assertEqual(symmetry["worst_offenders"][0]["left"], "hand_l")
        self.assertEqual(report["symmetry_pairs_outside_band_count"], 1)
        self.assertGreater(report["symmetry_mirror_error_p95"], 0.0)

    def test_a_rig_without_confident_facing_is_not_measurable(self):
        document, arrays = _paired_skeleton_document(drop_toes=True)

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertNotIn(
            report["handedness"]["facing_source"],
            ("foot-toe-direction", "foot-region-vertices"),
        )
        self.assertFalse(report["bilateral_symmetry"]["measurable"])
        self.assertIsNone(report["symmetry_mirror_error_p95"])
        self.assertIsNone(report["symmetry_pairs_outside_band_count"])
        self.assertTrue(
            any("facing" in caveat for caveat in report["bilateral_symmetry"]["caveats"])
        )

    def test_a_rig_with_no_handedness_measurement_is_not_measurable(self):
        # No feet, so `_detect_rig_handedness` declines to measure at all.
        document = {
            "nodes": [{"name": "pelvis"}, {"name": "head"}, {"mesh": 0, "skin": 0}],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2,
            }}]}],
            "skins": [{"joints": [0, 1]}],
        }
        arrays = {
            0: np.asarray([[0, 0, 0], [0, 1, 0]], dtype=float),
            1: np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=int),
            2: np.asarray([[1, 0, 0, 0]] * 2, dtype=float),
        }

        report = analyze_rig_document(document, arrays.__getitem__)

        self.assertIsNone(report["handedness"])
        self.assertFalse(report["bilateral_symmetry"]["measurable"])
        self.assertIsNone(report["symmetry_mirror_error_p95"])

    def test_child_joints_give_the_measured_bones_a_direction(self):
        document, arrays = _paired_skeleton_document()

        symmetry = analyze_rig_document(document, arrays.__getitem__)["bilateral_symmetry"]
        pairs = {item["left"]: item for item in symmetry["pairs"]}

        self.assertAlmostEqual(pairs["upper_arm_l"]["direction_angle_degrees"], 0.0, places=6)
        self.assertAlmostEqual(pairs["upper_arm_l"]["length_ratio"], 1.0, places=6)
        # A leaf bone has no child to infer a tail from.
        self.assertIsNone(pairs["hand_l"]["direction_angle_degrees"])


if __name__ == "__main__":
    unittest.main()
