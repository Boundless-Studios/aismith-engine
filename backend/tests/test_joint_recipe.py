import unittest

from pydantic import ValidationError

from backend.app.joint_recipe import JointRecipe


class JointRecipeTests(unittest.TestCase):
    def payload(self):
        return {
            "schema_version": 1,
            "clean_mesh_artifact_id": "mesh-1",
            "skeleton_template": "Human",
            "bone_naming": "preserve",
            "mirror_enabled": True,
            "transforms": [{
                "name": "pelvis",
                "position": [0, 1, 0],
                "quaternion": [0, 0, 0, 1],
            }],
            "editor_version": "mesh2motion-v1",
        }

    def test_round_trips_a_finite_recipe(self):
        recipe = JointRecipe.model_validate(self.payload())
        self.assertEqual(JointRecipe.model_validate_json(recipe.model_dump_json()), recipe)

    def test_rejects_duplicate_bones(self):
        payload = self.payload()
        payload["transforms"] *= 2
        with self.assertRaisesRegex(ValidationError, "unique"):
            JointRecipe.model_validate(payload)

    def test_rejects_non_finite_transforms(self):
        payload = self.payload()
        payload["transforms"][0]["position"][0] = float("nan")
        with self.assertRaisesRegex(ValidationError, "finite"):
            JointRecipe.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
