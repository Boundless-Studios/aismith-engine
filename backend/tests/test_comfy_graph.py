from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.app.comfy_graph import find_nodes, ui_workflow_to_api_prompt

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "comfy"


class HandBuiltWorkflowTests(unittest.TestCase):
    """LoadImage -> Reroute -> KSampler, a Note (skipped), and a converted
    widget-to-input, converted against a small hand-built object_info."""

    def setUp(self):
        self.object_info = {
            "LoadImage": {
                "input": {
                    "required": {"image": [["input.png", "other.png"], {}]},
                    "optional": {},
                }
            },
            "KSampler": {
                "input": {
                    "required": {
                        "image": ["IMAGE", {}],
                        "seed": ["INT", {"control_after_generate": True}],
                        "steps": ["INT", {}],
                    },
                    "optional": {},
                }
            },
            "ConvertedWidgetNode": {
                "input": {"required": {"value": ["FLOAT", {}]}, "optional": {}}
            },
            "FloatSource": {"input": {"required": {}, "optional": {}}},
        }

        self.workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "LoadImage",
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [100]},
                        {"name": "MASK", "type": "MASK", "links": []},
                    ],
                    "widgets_values": ["input.png", "image"],
                },
                {
                    "id": 2,
                    "type": "Note",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": ["a note, not a real node"],
                },
                {
                    "id": 3,
                    "type": "Reroute",
                    "inputs": [{"name": "", "type": "*", "link": 100}],
                    "outputs": [{"name": "", "type": "*", "links": [101]}],
                },
                {
                    "id": 4,
                    "type": "KSampler",
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 101}],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                    "widgets_values": [123, "randomize", 20],
                },
                {
                    "id": 5,
                    "type": "ConvertedWidgetNode",
                    "inputs": [
                        {
                            "name": "value",
                            "type": "FLOAT",
                            "widget": {"name": "value"},
                            "link": 102,
                        }
                    ],
                    "outputs": [],
                    "widgets_values": [9.99],
                },
                {
                    "id": 6,
                    "type": "FloatSource",
                    "inputs": [],
                    "outputs": [{"name": "FLOAT", "type": "FLOAT", "links": [102]}],
                },
            ],
            "links": [
                [100, 1, 0, 3, 0, "IMAGE"],
                [101, 3, 0, 4, 0, "IMAGE"],
                [102, 6, 0, 5, 0, "FLOAT"],
            ],
        }

    def test_converts_to_exact_api_prompt(self):
        prompt = ui_workflow_to_api_prompt(self.workflow, self.object_info)
        self.assertEqual(
            prompt,
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
                "4": {
                    "class_type": "KSampler",
                    "inputs": {"image": ["1", 0], "seed": 123, "steps": 20},
                },
                "5": {"class_type": "ConvertedWidgetNode", "inputs": {"value": ["6", 0]}},
                "6": {"class_type": "FloatSource", "inputs": {}},
            },
        )

    def test_skips_ui_only_nodes(self):
        prompt = ui_workflow_to_api_prompt(self.workflow, self.object_info)
        self.assertNotIn("2", prompt)
        self.assertNotIn("3", prompt)

    def test_reroute_resolves_to_original_source(self):
        prompt = ui_workflow_to_api_prompt(self.workflow, self.object_info)
        self.assertEqual(prompt["4"]["inputs"]["image"], ["1", 0])

    def test_find_nodes(self):
        prompt = ui_workflow_to_api_prompt(self.workflow, self.object_info)
        self.assertEqual(find_nodes(prompt, "KSampler"), ["4"])
        self.assertEqual(find_nodes(prompt, "Note"), [])

    def test_dict_form_widgets_values(self):
        workflow = json.loads(json.dumps(self.workflow))
        for node in workflow["nodes"]:
            if node["id"] == 4:
                node["widgets_values"] = {"seed": 123, "steps": 20, "control_after_generate": "randomize"}
        prompt = ui_workflow_to_api_prompt(workflow, self.object_info)
        self.assertEqual(prompt["4"]["inputs"], {"image": ["1", 0], "seed": 123, "steps": 20})


class DynamicComboDefaultFallbackTests(unittest.TestCase):
    """COMFY_DYNAMICCOMBO_V3's selected option's sub-fields are required by
    ComfyUI's server-side validator unconditionally once that option is
    selected -- unlike an ordinary top-level widget it may NOT be omitted
    when widgets_values runs out short; it must fall back to its own
    object_info "default" instead, or the live server rejects the node with
    "Required input is missing" (BOU comfy-native-3d-trial live-run finding).
    They also land under dotted keys ("<combo_name>.<sub_name>"), matching a
    live 0.34.4 node_errors payload's extra_info.input_name."""

    def setUp(self):
        self.object_info = {
            "DynComboNode": {
                "input": {
                    "required": {
                        "mode": [
                            "COMFY_DYNAMICCOMBO_V3",
                            {
                                "options": [
                                    {
                                        "key": "a",
                                        "inputs": {
                                            "required": {
                                                "x": ["BOOLEAN", {"default": True}],
                                                "y": ["INT", {"default": 5}],
                                            }
                                        },
                                    },
                                    {
                                        "key": "b",
                                        "inputs": {"required": {"z": ["FLOAT", {"default": 1.5}]}},
                                    },
                                ]
                            },
                        ],
                        "tail": ["INT", {}],
                    },
                    "optional": {},
                }
            }
        }

    def _node(self, widgets_values):
        return {
            "nodes": [
                {"id": 1, "type": "DynComboNode", "inputs": [], "outputs": [], "widgets_values": widgets_values}
            ],
            "links": [],
        }

    def test_sub_inputs_fall_back_to_default_when_widgets_values_runs_out(self):
        prompt = ui_workflow_to_api_prompt(self._node(["a"]), self.object_info)
        self.assertEqual(prompt["1"]["inputs"], {"mode": "a", "mode.x": True, "mode.y": 5})

    def test_sub_inputs_use_provided_widgets_values_when_present(self):
        prompt = ui_workflow_to_api_prompt(self._node(["a", False, 9, 3]), self.object_info)
        self.assertEqual(prompt["1"]["inputs"], {"mode": "a", "mode.x": False, "mode.y": 9, "tail": 3})

    def test_other_option_gets_its_own_defaults(self):
        prompt = ui_workflow_to_api_prompt(self._node(["b"]), self.object_info)
        self.assertEqual(prompt["1"]["inputs"], {"mode": "b", "mode.z": 1.5})


class RealTemplateFixtureTests(unittest.TestCase):
    """End-to-end conversion of the real ComfyUI TRELLIS.2/Pixal3D template
    against a subset of the real server's live /object_info response."""

    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads((FIXTURES_DIR / "template_3d.json").read_text())
        cls.object_info = json.loads((FIXTURES_DIR / "object_info_3d_subset.json").read_text())
        cls.prompt = ui_workflow_to_api_prompt(cls.workflow, cls.object_info)

    def test_every_non_ui_node_converted(self):
        from backend.app.comfy_graph import SKIP_NODE_TYPES

        expected_ids = {
            str(n["id"]) for n in self.workflow["nodes"] if n["type"] not in SKIP_NODE_TYPES
        }
        self.assertEqual(set(self.prompt.keys()), expected_ids)

    def test_no_node_has_an_unresolved_link(self):
        nodes_by_id = {n["id"]: n for n in self.workflow["nodes"]}
        for node in self.workflow["nodes"]:
            node_id_str = str(node["id"])
            if node_id_str not in self.prompt:
                continue
            api_inputs = self.prompt[node_id_str]["inputs"]
            for inp in node.get("inputs", []):
                if inp.get("link") is None:
                    continue
                self.assertIn(
                    inp["name"],
                    api_inputs,
                    f"node {node_id_str} ({node['type']}) missing linked input {inp['name']!r}",
                )
                value = api_inputs[inp["name"]]
                self.assertIsInstance(value, list)
                self.assertEqual(len(value), 2)
                source_id = value[0]
                self.assertIn(int(source_id), nodes_by_id)

    def test_expected_node_classes_present(self):
        for class_type in (
            "Trellis2Conditioning",
            "Pixal3DConditioning",
            "LoadImage",
            "Save3DAdvanced",
            "MeshToFile3D",
            "Trellis2UpsampleStage",
        ):
            self.assertTrue(
                find_nodes(self.prompt, class_type),
                f"expected at least one {class_type} node in the converted prompt",
            )

    def test_upsample_target_resolution_widget_lands_correctly(self):
        upsample_ids = find_nodes(self.prompt, "Trellis2UpsampleStage")
        self.assertEqual(len(upsample_ids), 1)
        inputs = self.prompt[upsample_ids[0]]["inputs"]
        self.assertIn("target_resolution", inputs)
        self.assertEqual(inputs["target_resolution"], "1536")

    def _inputs_of(self, class_type: str) -> dict:
        ids = find_nodes(self.prompt, class_type)
        self.assertEqual(len(ids), 1, f"expected exactly one {class_type} node")
        return self.prompt[ids[0]]["inputs"]

    def test_voxel_to_mesh_combo_widget(self):
        # V3-schema COMBO: type is the string "COMBO" with choices in
        # options["options"], not a bare list -- must still be classified as
        # a widget, or `algorithm` is skipped and `threshold` inherits its
        # value ("basic" landing on the FLOAT slot).
        self.assertEqual(
            self._inputs_of("VoxelToMesh"),
            {"voxel": ["119", 0], "algorithm": "basic", "threshold": 0.6},
        )

    def test_remesh_mesh_dynamic_combo_expands_inline(self):
        # COMFY_DYNAMICCOMBO_V3: sign_mode picks "udf", which then owns
        # sub-widgets (qef, drop_inverted_components, drop_enclosed_components)
        # consumed inline, landing as DOTTED keys ("sign_mode.qef", not
        # "qef") -- confirmed against a live ComfyUI 0.34.4 /prompt
        # node_errors payload, whose extra_info.input_name for a missing
        # sub-field was literally "sign_mode.drop_inverted_components".
        self.assertEqual(
            self._inputs_of("RemeshMesh"),
            {
                "mesh": ["202", 0],
                "resolution": 768,
                "sign_mode": "udf",
                "sign_mode.qef": False,
                "sign_mode.drop_inverted_components": False,
                "sign_mode.drop_enclosed_components": False,
                "band": 1,
                "project_back": 0,
                "fix_poles": False,
                "smooth_iters": 20,
                "drop_small_components": 0.01,
                "precluster_max_verts": 20000000,
            },
        )

    def test_unwrap_mesh_combo_widget_and_converted_input(self):
        self.assertEqual(
            self._inputs_of("UnwrapMesh"),
            {
                "mesh": ["238", 0],
                "segmenter": "pec",
                "resolution": ["288", 0],
                "padding": 1,
                "weld_distance": 0.0002,
            },
        )

    def test_vae_decode_structure_trellis2_combo_widget(self):
        self.assertEqual(
            self._inputs_of("VaeDecodeStructureTrellis2"),
            {"samples": ["3", 0], "vae": ["117", 0], "resolution": "32"},
        )

    def test_moge_geometry_to_fov_combo_widgets(self):
        self.assertEqual(
            self._inputs_of("MoGeGeometryToFOV"),
            {"moge_geometry": ["56", 0], "axis": "horizontal", "unit": "degrees"},
        )

    def test_load_moge_model_combo_widget(self):
        self.assertEqual(
            self._inputs_of("LoadMoGeModel"),
            {"model_name": "moge_2_vitl_normal_fp16.safetensors"},
        )

    def test_load_background_removal_model_combo_widget(self):
        self.assertEqual(
            self._inputs_of("LoadBackgroundRemovalModel"),
            {"bg_removal_name": "birefnet.safetensors"},
        )

    def test_save_3d_advanced_viewport_state_is_a_widget_not_a_socket(self):
        # viewport_state (LOAD_3D) is frontend-managed and never appears in
        # node["inputs"] -- it must still consume its widgets_values slot, or
        # width/height shift left (width landing on "").
        ids = find_nodes(self.prompt, "Save3DAdvanced")
        self.assertEqual(len(ids), 1)
        self.assertEqual(
            self.prompt[ids[0]]["inputs"],
            {
                "model_3d": ["285", 0],
                "filename_prefix": "3d/ComfyUI",
                "viewport_state": "",
                "width": 1024,
                "height": 1024,
            },
        )

    def test_preview_3d_advanced_viewport_state_is_a_widget_not_a_socket(self):
        for node_id, source in (("246", ["247", 0]), ("323", ["282", 0])):
            self.assertEqual(
                self.prompt[node_id]["inputs"],
                {"model_3d": source, "viewport_state": "", "width": 1024, "height": 1024},
            )


if __name__ == "__main__":
    unittest.main()
