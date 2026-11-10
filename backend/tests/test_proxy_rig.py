import unittest

from backend.app import proxy_rig
from backend.app.proxy_rig import (
    DEFORMATION_PROXY_BAKE_RAY_MULTIPLIER,
    add_local_finger_coverage,
    chain_endpoints_from_centroids,
    deformation_proxy_decimation_ratio,
    deformation_surface_can_promote,
    descendant_names,
    face_render_signature,
    is_proxy_deform_bone,
    largest_face_component,
    resolve_limb_chains,
    resolve_right_hand_bone,
    select_deformation_surface,
    select_action_name,
    sword_component_offsets,
    unique_face_indices,
    unique_position_face_indices,
    weighted_centroid,
)


class ProxyRigTests(unittest.TestCase):
    def test_fitted_hand_moves_every_descendant_finger_bone(self):
        parents = {
            "hand_l": "lowerarm_l",
            "index_01_l": "hand_l",
            "index_02_l": "index_01_l",
            "index_leaf_l": "index_02_l",
            "unrelated": "spine",
        }

        self.assertEqual(
            descendant_names(parents, "hand_l"),
            ["index_01_l", "index_02_l", "index_leaf_l"],
        )

    def test_finger_weight_policy_recognizes_deforming_segments_only(self):
        selector = getattr(proxy_rig, "is_finger_deform_bone", None)

        self.assertIsNotNone(selector)
        self.assertTrue(selector("index_01_l"))
        self.assertTrue(selector("thumb_03_r"))
        self.assertFalse(selector("hand_l"))
        self.assertFalse(selector("middle_04_leaf_l"))

    def test_finger_weights_take_only_local_hand_influence(self):
        blend = getattr(proxy_rig, "blend_local_finger_weights", None)

        self.assertIsNotNone(blend)
        self.assertEqual(
            blend(
                {"hand_l": 0.8, "lowerarm_l": 0.2},
                {"index_01_l": 0.0, "middle_01_l": 0.01},
                hand_bone="hand_l",
                maximum_distance=0.02,
            ),
            {"hand_l": 0.08, "lowerarm_l": 0.2, "index_01_l": 0.72},
        )

    def test_finger_weights_ignore_vertices_outside_the_hand_envelope(self):
        blend = getattr(proxy_rig, "blend_local_finger_weights", None)
        current = {"hand_l": 0.6, "lowerarm_l": 0.4}

        self.assertIsNotNone(blend)
        self.assertEqual(
            blend(
                current,
                {"index_01_l": 0.03},
                hand_bone="hand_l",
                maximum_distance=0.02,
            ),
            current,
        )

    def test_unused_terminal_finger_takes_only_bounded_same_side_influence(self):
        self.assertEqual(
            add_local_finger_coverage(
                {"hand_l": 0.2, "thumb_02_l": 0.7, "lowerarm_l": 0.1},
                target_bone="thumb_03_l",
                side="l",
            ),
            {"hand_l": 0.2, "thumb_02_l": 0.595, "lowerarm_l": 0.1, "thumb_03_l": 0.105},
        )

    def test_only_original_visible_surface_can_auto_promote(self):
        self.assertTrue(deformation_surface_can_promote("visible-original"))
        self.assertFalse(deformation_surface_can_promote("voxel-proxy"))

    def test_sword_grip_is_centered_on_the_hand_anchor(self):
        self.assertEqual(
            sword_component_offsets(blade_length=4.0, grip_length=1.0),
            {
                "blade": 2.5,
                "guard": 0.5,
                "grip": 0.0,
            },
        )

    def test_voxel_proxy_uses_local_texture_projection_rays(self):
        self.assertEqual(DEFORMATION_PROXY_BAKE_RAY_MULTIPLIER, 1.0)

    def test_deformation_proxy_decimation_targets_a_bounded_surface(self):
        self.assertEqual(deformation_proxy_decimation_ratio(20_000), 1.0)
        self.assertAlmostEqual(deformation_proxy_decimation_ratio(100_000), 0.25)
        with self.assertRaisesRegex(ValueError, "positive face count"):
            deformation_proxy_decimation_ratio(0)

    def test_largest_face_component_discards_disconnected_voxel_crumbs(self):
        neighbors = [
            {1},
            {0, 2},
            {1},
            {4},
            {3},
            set(),
        ]

        self.assertEqual(largest_face_component(neighbors), {0, 1, 2})

    def test_deformation_surface_selects_visible_control_or_watertight_proxy(self):
        self.assertEqual(select_deformation_surface("visible-original"), "visible-original")
        self.assertEqual(select_deformation_surface("voxel-proxy"), "voxel-proxy")
        with self.assertRaisesRegex(ValueError, "Unsupported deformation surface"):
            select_deformation_surface("magic")

    def test_limb_chain_resolver_supports_mixamo_and_rigify_names(self):
        chains = resolve_limb_chains([
            "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
            "DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R",
            "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
            "DEF-thigh.R", "DEF-shin.R", "DEF-foot.R",
        ])

        self.assertEqual(chains, [
            ("mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand"),
            ("DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R"),
            ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot"),
            ("DEF-thigh.R", "DEF-shin.R", "DEF-foot.R"),
        ])

    def test_limb_chain_resolver_allows_an_unsupported_donor(self):
        self.assertEqual(resolve_limb_chains(["Bone", "Bone.001"]), [])

    def test_chain_endpoints_put_joints_between_weighted_regions(self):
        endpoints = chain_endpoints_from_centroids([
            (0.0, 0.0, 0.0),
            (2.0, -2.0, 0.0),
            (3.0, -4.0, 0.0),
        ])

        self.assertEqual(endpoints, [
            (-1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (2.5, -3.0, 0.0),
            (3.5, -5.0, 0.0),
        ])

    def test_render_signature_distinguishes_uv_layouts(self):
        positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]

        first = face_render_signature(positions, [(0, 0), (1, 0), (0, 1)])
        second = face_render_signature(positions, [(1, 0), (0, 0), (0, 1)])

        self.assertNotEqual(first, second)

    def test_duplicate_face_cleanup_keeps_one_winding_independent_triangle(self):
        kept = unique_face_indices([
            (0, 1, 2),
            (2, 1, 0),
            (1, 2, 0),
            (0, 2, 3),
        ])

        self.assertEqual(kept, [0, 3])

    def test_position_duplicate_cleanup_handles_split_glb_vertices(self):
        kept = unique_position_face_indices(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            [(0, 1, 2), (3, 4, 5), (0, 1, 6)],
        )

        self.assertEqual(kept, [0, 2])

    def test_position_cleanup_preserves_opposite_winding_for_culled_cards(self):
        kept = unique_position_face_indices(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 1, 2), (2, 1, 0)],
            face_groups=[4, 4],
            preserve_opposite_winding_groups={4},
        )

        self.assertEqual(kept, [0, 1])

    def test_position_cleanup_keeps_coincident_faces_with_distinct_materials(self):
        kept = unique_position_face_indices(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 1, 2), (2, 1, 0)],
            face_groups=[1, 2],
        )

        self.assertEqual(kept, [0, 1])

    def test_position_cleanup_preserves_distinct_concave_ngon_ordering(self):
        kept = unique_position_face_indices(
            [(0, 0, 0), (2, 0, 0), (1, 1, 0), (2, 2, 0), (0, 2, 0)],
            [(0, 1, 2, 3, 4), (0, 1, 3, 2, 4)],
        )

        self.assertEqual(kept, [0, 1])

    def test_position_cleanup_compares_transformed_world_coordinates(self):
        kept = unique_position_face_indices(
            [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.004), (1.0, 0.0, 0.004), (0.0, 1.0, 0.004),
            ],
            [(0, 1, 2), (3, 4, 5)],
        )

        self.assertEqual(kept, [0, 1])

    def test_position_cleanup_deduplicates_across_mesh_objects(self):
        seen = set()
        first = unique_position_face_indices(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(0, 1, 2)],
            face_groups=["shared-material"],
            seen_keys=seen,
        )
        second = unique_position_face_indices(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(2, 1, 0)],
            face_groups=["shared-material"],
            seen_keys=seen,
        )

        self.assertEqual(first, [0])
        self.assertEqual(second, [])

    def test_exact_action_wins_over_related_prefixed_actions(self):
        selected = select_action_name(
            ["Sword_Attack", "Sword_Attack_Air_Vertical"],
            "Sword_Attack",
        )

        self.assertEqual(selected, "Sword_Attack")

    def test_unique_importer_suffixed_action_is_accepted(self):
        selected = select_action_name(["Sword_Attack_Armature"], "Sword_Attack")

        self.assertEqual(selected, "Sword_Attack_Armature")

    def test_importer_suffix_wins_over_related_root_motion_variant(self):
        selected = select_action_name(
            ["Sword_Attack_Armature", "Sword_Attack_RM_Armature"],
            "Sword_Attack",
        )

        self.assertEqual(selected, "Sword_Attack_Armature")

    def test_unique_blender_collision_suffix_is_accepted(self):
        selected = select_action_name(
            ["Sword_Attack_Armature.001", "Sword_Attack_RM_Armature.001"],
            "Sword_Attack",
        )

        self.assertEqual(selected, "Sword_Attack_Armature.001")

    def test_right_hand_resolver_supports_common_humanoid_names(self):
        self.assertEqual(resolve_right_hand_bone(["hand_r"]), "hand_r")
        self.assertEqual(resolve_right_hand_bone(["RightHand"]), "RightHand")
        self.assertEqual(
            resolve_right_hand_bone(["mixamorig:RightHand"]),
            "mixamorig:RightHand",
        )

    def test_weighted_centroid_follows_the_exported_hand_weights(self):
        centroid = weighted_centroid([
            ((0.0, 0.0, 0.0), 1.0),
            ((4.0, 2.0, 0.0), 3.0),
        ])

        self.assertEqual(centroid, (3.0, 1.5, 0.0))

    def test_low_detail_proxy_uses_hands_but_not_finger_bones(self):
        self.assertTrue(is_proxy_deform_bone("hand_l"))
        self.assertTrue(is_proxy_deform_bone("hand_r"))
        self.assertFalse(is_proxy_deform_bone("thumb_03_l"))
        self.assertFalse(is_proxy_deform_bone("index_01_r"))
        self.assertFalse(is_proxy_deform_bone("middle_04_leaf_l"))

    def test_low_detail_proxy_excludes_non_deforming_root_and_leaf_bones(self):
        self.assertFalse(is_proxy_deform_bone("root"))
        self.assertFalse(is_proxy_deform_bone("head_leaf"))
        self.assertTrue(is_proxy_deform_bone("head"))
        self.assertTrue(is_proxy_deform_bone("lowerarm_l"))


if __name__ == "__main__":
    unittest.main()
