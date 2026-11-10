import unittest

from backend.app.retarget import (
    resolve_motion_mapping,
    select_character_armature,
    select_deepest_source_per_target,
    select_motion_translation_bones,
)


class RetargetMappingTests(unittest.TestCase):
    def test_uses_bvh_root_translation_when_root_is_mapped(self):
        self.assertEqual(
            select_motion_translation_bones(
                {"Root", "Hips"},
                {"Root": "root", "Hips": "pelvis"},
            ),
            ("Root", "root"),
        )

    def test_falls_back_to_hips_translation_without_a_root_channel(self):
        self.assertEqual(
            select_motion_translation_bones(
                {"Hips"},
                {"Hips": "pelvis"},
            ),
            ("Hips", "pelvis"),
        )

    def test_prefers_certified_leg_roles_over_ambiguous_exact_names(self):
        mapping = resolve_motion_mapping(
            {"LeftLeg", "LeftShin", "RightLeg", "RightShin"},
            {"LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg"},
            {
                "left_upper_leg": "LeftUpLeg",
                "left_lower_leg": "LeftLeg",
                "right_upper_leg": "RightUpLeg",
                "right_lower_leg": "RightLeg",
            },
        )

        self.assertEqual(mapping["LeftLeg"], "LeftUpLeg")
        self.assertEqual(mapping["LeftShin"], "LeftLeg")
        self.assertEqual(mapping["RightLeg"], "RightUpLeg")
        self.assertEqual(mapping["RightShin"], "RightLeg")

    def test_prefers_unique_exact_spine_chain_before_generic_certificate_role(self):
        mapping = resolve_motion_mapping(
            {"Hips", "Spine1", "Spine2", "Chest", "Head"},
            {"pelvis", "spine_01", "spine_02", "spine_03", "head"},
            {"pelvis": "pelvis", "spine": "spine_01", "head": "head"},
        )

        self.assertEqual(mapping["Spine1"], "spine_01")
        self.assertEqual(mapping["Spine2"], "spine_02")
        self.assertEqual(mapping["Chest"], "spine_03")
        self.assertEqual(len(set(mapping.values())), len(mapping))

    def test_combines_extra_source_spine_segments_for_single_spine_rig(self):
        mapping = resolve_motion_mapping(
            {"Hips", "Spine1", "Spine2", "Chest", "Head"},
            {"Hips", "Spine", "Head"},
            {"pelvis": "Hips", "spine": "Spine", "head": "Head"},
        )

        self.assertEqual(mapping["Spine1"], "Spine")
        self.assertEqual(mapping["Spine2"], "Spine")
        self.assertEqual(mapping["Chest"], "Spine")

    def test_collapses_missing_neck_and_shoulders_into_mapped_descendants(self):
        mapping = resolve_motion_mapping(
            {"Neck1", "Head", "LeftShoulder", "LeftArm", "RightShoulder", "RightArm"},
            {"Head", "LeftArm", "RightArm"},
            {"head": "Head", "left_upper_arm": "LeftArm", "right_upper_arm": "RightArm"},
        )

        self.assertEqual(mapping["Neck1"], "Head")
        self.assertEqual(mapping["Head"], "Head")
        self.assertEqual(mapping["LeftShoulder"], "LeftArm")
        self.assertEqual(mapping["LeftArm"], "LeftArm")

    def test_unirig_spine_and_neck_names_resolve_to_distinct_bones(self):
        # Real UniRig output (studio-data/artifacts/57d3bd80...glb): spine_1,
        # spine_2, spine_3 and neck are all separate bones -- unlike the
        # mesh2motion-style spine_01/02/03 + neck_01 names, so each SOMA
        # segment should land on its own target instead of collapsing onto
        # the single certified "spine" role.
        mapping = resolve_motion_mapping(
            {"Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head"},
            {"pelvis", "spine_1", "spine_2", "spine_3", "neck", "head"},
            {"pelvis": "pelvis", "spine": "spine_3", "head": "head"},
        )

        self.assertEqual(mapping["Spine1"], "spine_1")
        self.assertEqual(mapping["Spine2"], "spine_2")
        self.assertEqual(mapping["Chest"], "spine_3")
        self.assertEqual(mapping["Neck1"], "neck")
        self.assertEqual(mapping["Head"], "head")
        self.assertEqual(len(set(mapping.values())), len(mapping))

    def test_old_mesh2motion_spine_and_neck_names_still_resolve(self):
        mapping = resolve_motion_mapping(
            {"Spine1", "Spine2", "Chest", "Neck1"},
            {"spine_01", "spine_02", "spine_03", "neck_01"},
            {},
        )

        self.assertEqual(mapping["Spine1"], "spine_01")
        self.assertEqual(mapping["Spine2"], "spine_02")
        self.assertEqual(mapping["Chest"], "spine_03")
        self.assertEqual(mapping["Neck1"], "neck_01")

    def test_deepest_source_wins_when_spine_segments_collapse_onto_one_bone(self):
        mapping = {"Spine1": "spine_3", "Spine2": "spine_3", "Chest": "spine_3"}

        self.assertEqual(
            select_deepest_source_per_target(mapping),
            {"spine_3": "Chest"},
        )

    def test_deepest_source_wins_for_head_and_shoulder_collapses(self):
        mapping = {
            "Neck1": "head", "Head": "head",
            "LeftShoulder": "upper_arm_l", "LeftArm": "upper_arm_l",
        }

        chosen = select_deepest_source_per_target(mapping)

        self.assertEqual(chosen["head"], "Head")
        self.assertEqual(chosen["upper_arm_l"], "LeftArm")

    def test_deepest_source_keeps_non_colliding_targets_unchanged(self):
        mapping = {"Hips": "pelvis", "LeftHand": "hand_l", "RightHand": "hand_r"}

        self.assertEqual(
            select_deepest_source_per_target(mapping),
            {"pelvis": "Hips", "hand_l": "LeftHand", "hand_r": "RightHand"},
        )

    def test_selects_armature_containing_the_certified_character_bones(self):
        selected = select_character_armature(
            {
                "Character": {"Hips", "Spine", "Head", "LeftArm"},
                "Sword": {"weapon_root", "blade"},
            },
            {"pelvis": "Hips", "spine": "Spine", "head": "Head"},
        )

        self.assertEqual(selected, "Character")


if __name__ == "__main__":
    unittest.main()
