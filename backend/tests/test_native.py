import importlib.util
import json
import os
import struct
import sys
import tomllib
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import trimesh

from backend.app import native

# pymeshfix is the optional `meshfix` extra (GPL-3.0); CI installs it.
requires_pymeshfix = unittest.skipUnless(
    importlib.util.find_spec("pymeshfix") is not None,
    "pymeshfix is not installed (pip install -e '.[meshfix]')",
)


class NativeAdapterTests(unittest.TestCase):
    @staticmethod
    def _write_glb(path: Path, document: dict) -> None:
        json_chunk = json.dumps(document).encode("utf-8")
        json_chunk += b" " * (-len(json_chunk) % 4)
        total_length = 12 + 8 + len(json_chunk)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, total_length)
            + struct.pack("<I4s", len(json_chunk), b"JSON")
            + json_chunk
        )

    def test_glb_material_detection_follows_primitive_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            unmaterialed = root / "unmaterialed.glb"
            materialed = root / "materialed.glb"
            self._write_glb(unmaterialed, {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}]})
            self._write_glb(
                materialed,
                {
                    "asset": {"version": "2.0"},
                    "materials": [{}],
                    "meshes": [{"primitives": [{"material": 0}]}],
                },
            )

            self.assertFalse(native._glb_has_material(unmaterialed))
            self.assertTrue(native._glb_has_material(materialed))

    def test_glb_texture_detection_rejects_clay_materials(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            clay = root / "clay.glb"
            textured = root / "textured.glb"
            self._write_glb(
                clay,
                {
                    "asset": {"version": "2.0"},
                    "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
                    "meshes": [{"primitives": [{"material": 0}]}],
                },
            )
            self._write_glb(
                textured,
                {
                    "asset": {"version": "2.0"},
                    "images": [{"bufferView": 0, "mimeType": "image/png"}],
                    "textures": [{"source": 0}],
                    "materials": [
                        {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                    ],
                    "meshes": [{"primitives": [{"material": 0}]}],
                },
            )

            self.assertFalse(native._glb_has_texture(clay))
            self.assertTrue(native._glb_has_texture(textured))

    def test_glb_rigging_contract_rejects_mesh_node_transforms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "identity.glb"
            rotated = root / "rotated.glb"
            self._write_glb(
                identity,
                {"asset": {"version": "2.0"}, "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{}]}]},
            )
            self._write_glb(
                rotated,
                {
                    "asset": {"version": "2.0"},
                    "nodes": [{"mesh": 0, "rotation": [0.7071068, 0, 0, 0.7071068]}],
                    "meshes": [{"primitives": [{}]}],
                },
            )

            self.assertTrue(native._glb_mesh_nodes_have_identity_transforms(identity))
            self.assertFalse(native._glb_mesh_nodes_have_identity_transforms(rotated))

    def test_glb_rigging_contract_requires_y_up_vertex_coordinates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            y_up = root / "y-up.glb"
            z_up = root / "z-up.glb"
            common = {
                "asset": {"version": "2.0"},
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            }
            self._write_glb(y_up, {**common, "accessors": [{"min": [-1, -2, -0.5], "max": [1, 2, 0.5]}]})
            self._write_glb(z_up, {**common, "accessors": [{"min": [-1, -0.5, -2], "max": [1, 0.5, 2]}]})

            self.assertTrue(native._glb_mesh_vertices_are_y_up(y_up))
            self.assertFalse(native._glb_mesh_vertices_are_y_up(z_up))

    def test_retopology_bake_skips_appearance_bake_without_source_material(self):
        command = []

        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.glb"
            output = Path(directory) / "output.glb"

            def capture(args, **_kwargs):
                command.extend(args)
                self._write_glb(
                    output,
                    {
                        "asset": {"version": "2.0"},
                        "nodes": [{"mesh": 0}],
                        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                        "accessors": [{"min": [-1, -2, -0.5], "max": [1, 2, 0.5]}],
                    },
                )

            self._write_glb(source, {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}]})
            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
            ):
                native.blender_bake_retopology(source, Path("remeshed.obj"), output)

        spec = json.loads(command[-1])
        self.assertFalse(spec["bake_appearance"])

    def test_linux_instant_meshes_requires_native_executable(self):
        with TemporaryDirectory() as directory:
            install = Path(directory)
            linux_binary = install / "Instant Meshes"
            windows_binary = install / "Instant Meshes.exe"
            linux_binary.write_bytes(b"linux")
            windows_binary.write_bytes(b"windows")
            linux_binary.chmod(0o644)
            with (
                patch.object(native, "instant_meshes_dir", return_value=install),
                patch.object(native.sys, "platform", "linux"),
            ):
                self.assertIsNone(native.instant_meshes_exe())
                linux_binary.chmod(0o755)
                self.assertEqual(native.instant_meshes_exe(), linux_binary.resolve())

    def test_curated_sword_donor_resolves_path_hash_and_is_immutable(self):
        donor = native.curated_sword_donor()
        donor_path = native.ROOT / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb"
        self.assertEqual(donor.path, donor_path)
        self.assertEqual(
            donor.sha256,
            "406eb0a8dc4ab366e623b79b6e3005a4951392e1bda78ae39c1099d31147733c",
        )
        self.assertEqual(donor.action_name, "Sword_Regular_A")
        with self.assertRaises(FrozenInstanceError):
            donor.path = donor_path.parent / "other.glb"

    def test_curated_sword_donor_reports_missing_bundled_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(native, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "The bundled grounded Sword Attack library is missing",
                ) as raised:
                    native.curated_sword_donor()

            self.assertIn("human-base-animations.glb", str(raised.exception))

    def test_curated_sword_donor_rejects_modified_bundled_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            donor_path = root / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb"
            donor_path.parent.mkdir(parents=True)
            donor_path.write_bytes(b"modified grounded sword donor")

            with patch.object(native, "ROOT", root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SHA-256.*does not match",
                ):
                    native.curated_sword_donor()

    def test_curated_sword_donor_rejects_symlinked_bundled_library(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            donor_path = root / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb"
            donor_path.parent.mkdir(parents=True)
            target = root / "donor-target.glb"
            target.write_bytes(b"grounded sword donor")
            try:
                donor_path.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")

            with patch.object(native, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                    native.curated_sword_donor()

    def test_target_proxy_rig_invokes_isolated_blender_operation(self):
        command = []

        def capture(args, **_kwargs):
            command.extend(args)

        with (
            patch.object(native, "find_blender", return_value=Path("/tools/blender")),
            patch.object(native, "run_process", side_effect=capture),
        ):
            result = native.run_target_proxy_rig(
                Path("mary.glb"),
                Path("donor.glb"),
                Path("candidate.glb"),
                action_name="Sword_Attack_Air_Vertical",
                attach_sword=True,
                deformation_surface="voxel-proxy",
            )

        spec = json.loads(command[-1])
        self.assertEqual(spec["op"], "target-proxy-rig")
        self.assertEqual(spec["input"], "mary.glb")
        self.assertEqual(spec["donor"], "donor.glb")
        self.assertEqual(spec["output"], "candidate.glb")
        self.assertEqual(spec["action_name"], "Sword_Attack_Air_Vertical")
        self.assertTrue(spec["attach_sword"])
        self.assertEqual(spec["deformation_surface"], "voxel-proxy")
        self.assertEqual(result["weighting"], "target-voxel-proxy")
        self.assertEqual(result["deformation_surface"], "voxel-proxy")
        self.assertEqual(result["attached_prop"], "right-hand-sword")

    @staticmethod
    def _diagonal(mesh: trimesh.Trimesh) -> float:
        bounds = mesh.bounds
        return float(np.linalg.norm(bounds[1] - bounds[0]))

    def test_repair_meshfix_names_the_optional_extra_when_pymeshfix_is_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.dict(sys.modules, {"pymeshfix": None}),
                self.assertRaisesRegex(RuntimeError, r"\.\[meshfix\]"),
            ):
                native.repair_meshfix(root / "in.obj", root / "out.obj")

    def test_pymeshfix_is_an_optional_extra_not_a_base_dependency(self):
        project = tomllib.loads((native.ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertFalse([dep for dep in project["dependencies"] if dep.lower().startswith("pymeshfix")])
        self.assertTrue([dep for dep in project["optional-dependencies"]["meshfix"] if dep.startswith("pymeshfix")])

    @requires_pymeshfix
    def test_repair_meshfix_fills_hole_and_preserves_bounds(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        holed_faces = np.delete(sphere.faces, 0, axis=0)
        holed = trimesh.Trimesh(vertices=sphere.vertices, faces=holed_faces, process=False)
        self.assertFalse(holed.is_watertight)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            holed.export(input_obj, file_type="obj")

            result = native.repair_meshfix(input_obj, output_obj)

            self.assertTrue(output_obj.exists())
            self.assertNotIn("warning", result)
            self.assertEqual(result["components"], 1)
            self.assertEqual(result["repaired_components"], 1)
            self.assertEqual(result["kept_components"], 0)

            repaired = trimesh.load(output_obj, force="mesh", process=False)
            self.assertTrue(repaired.is_watertight)
            source_diagonal = self._diagonal(holed)
            repaired_diagonal = self._diagonal(repaired)
            self.assertGreater(repaired_diagonal, source_diagonal * 0.75)
            self.assertLess(repaired_diagonal, source_diagonal * 1.25)

    @requires_pymeshfix
    def test_repair_meshfix_keeps_good_component_and_passes_through_scrap(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        # A degenerate, disconnected scrap far from the sphere: too small a
        # fraction of the total faces to be worth running MeshFix on.
        scrap_vertices = np.array(
            [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0001], [100.0001, 100.0, 100.0]]
        )
        scrap_faces = np.array([[0, 1, 2]])
        scrap = trimesh.Trimesh(vertices=scrap_vertices, faces=scrap_faces, process=False)
        combined = trimesh.util.concatenate([sphere, scrap])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            combined.export(input_obj, file_type="obj")

            result = native.repair_meshfix(input_obj, output_obj)

            self.assertTrue(output_obj.exists())
            self.assertEqual(result["components"], 2)
            self.assertEqual(result["repaired_components"], 1)
            self.assertEqual(result["kept_components"], 1)

            # The good sphere is repaired without ballooning in face count;
            # the scrap is passed through untouched rather than dropped.
            self.assertGreaterEqual(result["output_faces"], len(sphere.faces))
            self.assertLessEqual(result["output_faces"], len(sphere.faces) * 2.5 + len(scrap.faces))

            repaired = trimesh.load(output_obj, force="mesh", process=False)
            self.assertTrue(
                np.any(np.all(np.isclose(repaired.vertices, scrap_vertices[0], atol=1e-3), axis=1))
            )

    @requires_pymeshfix
    def test_repair_meshfix_repairs_two_non_tiny_holed_components(self):
        sphere_a = trimesh.creation.icosphere(subdivisions=2)
        holed_a = trimesh.Trimesh(vertices=sphere_a.vertices, faces=np.delete(sphere_a.faces, 0, axis=0), process=False)
        sphere_b = trimesh.creation.icosphere(subdivisions=2)
        sphere_b.apply_translation([5.0, 0.0, 0.0])
        holed_b = trimesh.Trimesh(vertices=sphere_b.vertices, faces=np.delete(sphere_b.faces, 0, axis=0), process=False)
        combined = trimesh.util.concatenate([holed_a, holed_b])
        self.assertFalse(combined.is_watertight)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            combined.export(input_obj, file_type="obj")

            result = native.repair_meshfix(input_obj, output_obj)

            self.assertTrue(output_obj.exists())
            self.assertNotIn("warning", result)
            self.assertEqual(result["components"], 2)
            self.assertEqual(result["repaired_components"], 2)
            self.assertEqual(result["kept_components"], 0)

            repaired = trimesh.load(output_obj, force="mesh", process=False)
            self.assertTrue(repaired.is_watertight)

    @requires_pymeshfix
    def test_repair_meshfix_raises_when_every_component_fails_and_holes_remain(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        holed = trimesh.Trimesh(vertices=sphere.vertices, faces=np.delete(sphere.faces, 0, axis=0), process=False)
        self.assertFalse(holed.is_watertight)

        class _FakeMeshFix:
            def __init__(self, vertices, faces, verbose=False):
                pass

            def repair(self, joincomp=False, remove_smallest_components=False):
                # Simulate pymeshfix discarding almost everything, as seen on
                # the dense, heavily non-manifold Instant Meshes output.
                self.points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]])
                self.faces = np.array([[0, 1, 2]])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            holed.export(input_obj, file_type="obj")

            with patch("pymeshfix.MeshFix", _FakeMeshFix):
                with self.assertRaisesRegex(RuntimeError, "unfilled holes"):
                    native.repair_meshfix(input_obj, output_obj)

    @requires_pymeshfix
    def test_repair_meshfix_raises_when_whole_mesh_bounds_guard_trips(self):
        sphere_a = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
        sphere_b = trimesh.creation.icosphere(subdivisions=1)  # 80 faces
        sphere_b.apply_translation([3.0, 0.0, 0.0])
        combined = trimesh.util.concatenate([sphere_a, sphere_b])

        class _FakeMeshFix:
            def __init__(self, vertices, faces, verbose=False):
                self._vertices = np.asarray(vertices)
                self._faces = np.asarray(faces)

            def repair(self, joincomp=False, remove_smallest_components=False):
                if len(self._faces) > 200:
                    # Same size and face count as the input component (so it
                    # passes the per-component envelope), but drifted far
                    # from its original position - catastrophic once
                    # reassembled with the other, untouched component.
                    self.points = self._vertices + np.array([1000.0, 0.0, 0.0])
                else:
                    self.points = self._vertices
                self.faces = self._faces

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            combined.export(input_obj, file_type="obj")

            with patch("pymeshfix.MeshFix", _FakeMeshFix):
                with self.assertRaisesRegex(RuntimeError, "safe limit"):
                    native.repair_meshfix(input_obj, output_obj)

    @requires_pymeshfix
    def test_repair_meshfix_keeps_original_and_warns_when_every_component_fails(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        self.assertTrue(sphere.is_watertight)

        class _FakeMeshFix:
            def __init__(self, vertices, faces, verbose=False):
                self._vertices = vertices
                self._faces = faces

            def repair(self, joincomp=False, remove_smallest_components=False):
                # Simulate pymeshfix discarding almost everything, as seen on
                # the dense, heavily non-manifold Instant Meshes output.
                self.points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]])
                self.faces = np.array([[0, 1, 2]])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            sphere.export(input_obj, file_type="obj")

            with patch("pymeshfix.MeshFix", _FakeMeshFix):
                result = native.repair_meshfix(input_obj, output_obj)

            self.assertTrue(output_obj.exists())
            self.assertEqual(result["components"], 1)
            self.assertEqual(result["repaired_components"], 0)
            self.assertEqual(result["kept_components"], 1)
            self.assertIn("warning", result)

            repaired = trimesh.load(output_obj, force="mesh", process=False)
            self.assertEqual(len(repaired.faces), len(sphere.faces))
            source_diagonal = self._diagonal(sphere)
            repaired_diagonal = self._diagonal(repaired)
            self.assertAlmostEqual(repaired_diagonal, source_diagonal, delta=1e-4)

    @requires_pymeshfix
    def test_repair_meshfix_skips_repair_when_mesh_is_fragmented_beyond_repair(self):
        # A quad/triangle soup: thousands of mutually-disconnected 1-face
        # components, none sharing a vertex with any other (as observed on
        # real jobs where welding still left ~179k components). This should
        # be recognized as unrepairable up front rather than run through the
        # per-component MeshFix loop (where every component is "tiny" and
        # none gets repaired) and then raise.
        component_count = 2500
        triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vertices = np.concatenate(
            [triangle + np.array([index * 10.0, 0.0, 0.0]) for index in range(component_count)]
        )
        faces = np.array([[3 * index, 3 * index + 1, 3 * index + 2] for index in range(component_count)])
        soup = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_obj = root / "remeshed.obj"
            output_obj = root / "remeshed-repaired.obj"
            soup.export(input_obj, file_type="obj")

            result = native.repair_meshfix(input_obj, output_obj)

            self.assertTrue(output_obj.exists())
            self.assertEqual(result["components"], component_count)
            self.assertEqual(result["repaired_components"], 0)
            self.assertEqual(result["kept_components"], component_count)
            self.assertEqual(result["skipped_reason"], "fragmented")
            self.assertIn("warning", result)
            self.assertIn(str(component_count), result["warning"])

            repaired = trimesh.load(output_obj, force="mesh", process=False)
            self.assertEqual(len(repaired.faces), component_count)

    def _run_instant_meshes_with_source_metadata(self, target_vertices, source_metadata):
        def fake_run_process(args, **_kwargs):
            Path(args[2]).write_bytes(b"0" * 128)
            return ""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_glb = root / "source.glb"
            output_glb = root / "output.glb"
            with (
                patch.object(native, "instant_meshes_exe", return_value=Path("/fake/instant-meshes")),
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "blender_export_obj", return_value=None),
                patch.object(native, "run_process", side_effect=fake_run_process),
                patch.object(native, "blender_bake_retopology", return_value=None),
            ):
                return native.run_instant_meshes(
                    input_glb,
                    output_glb,
                    target_vertices,
                    repair_holes=False,
                    source_metadata=source_metadata,
                )

    def test_run_instant_meshes_warns_when_upscaling_source_by_more_than_1_5x(self):
        result = self._run_instant_meshes_with_source_metadata(
            200000, {"mesh_quality": {"vertices": 100000}}
        )

        self.assertIn("warnings", result)
        self.assertTrue(any("1.5x" in warning for warning in result["warnings"]))
        self.assertTrue(any("fragments it" in warning for warning in result["warnings"]))

    def test_run_instant_meshes_does_not_warn_at_1x_source_vertices(self):
        result = self._run_instant_meshes_with_source_metadata(
            100000, {"mesh_quality": {"vertices": 100000}}
        )

        self.assertNotIn("warnings", result)

    def test_run_instant_meshes_warns_when_source_is_already_a_retopo_output(self):
        result = self._run_instant_meshes_with_source_metadata(
            50000, {"stage": "retopo", "mesh_quality": {"vertices": 100000}}
        )

        self.assertIn("warnings", result)
        self.assertTrue(any("already a retopo output" in warning for warning in result["warnings"]))

    def test_run_trellis_fp8_spec_carries_view_images_only_for_multiview(self):
        command = []

        def capture(args, **_kwargs):
            command.extend(args)

        with (
            patch.object(native, "trellis_refiner_ready", return_value=True),
            patch.object(native, "trellis_refiner_python", return_value=Path("/tools/python")),
            patch.object(native, "trellis_refiner_dir", return_value=Path("/tools/runtime")),
            patch.object(native, "run_process", side_effect=capture),
            patch.object(native, "_glb_has_texture", return_value=True),
        ):
            result = native.run_trellis_fp8(
                Path("front.png"),
                Path("output.glb"),
                1024,
                view_images={
                    "front": Path("front.png"),
                    "back": Path("back.png"),
                },
            )

        spec = json.loads(command[-1])
        self.assertIn("view_images", spec)
        self.assertEqual(spec["view_images"], {"front": "front.png", "back": "back.png"})
        self.assertEqual(spec["source_image"], "front.png")
        self.assertEqual(result["views"], ["front", "back"])

    def test_run_trellis_fp8_spec_omits_view_images_for_single_view(self):
        command = []

        def capture(args, **_kwargs):
            command.extend(args)

        with (
            patch.object(native, "trellis_refiner_ready", return_value=True),
            patch.object(native, "trellis_refiner_python", return_value=Path("/tools/python")),
            patch.object(native, "trellis_refiner_dir", return_value=Path("/tools/runtime")),
            patch.object(native, "run_process", side_effect=capture),
            patch.object(native, "_glb_has_texture", return_value=True),
        ):
            result = native.run_trellis_fp8(
                Path("front.png"),
                Path("output.glb"),
                1024,
            )
            result_with_single_view_dict = native.run_trellis_fp8(
                Path("front.png"),
                Path("output.glb"),
                1024,
                view_images={"front": Path("front.png")},
            )

        for entry in (result, result_with_single_view_dict):
            self.assertNotIn("views", entry)
        spec = json.loads(command[-1])
        self.assertNotIn("view_images", spec)
        self.assertEqual(spec["source_image"], "front.png")


class RenderGuideViewsTests(unittest.TestCase):
    """task9: the render-guide-views launcher spec, mocked (no Blender)."""

    def test_spec_carries_roles_extra_roles_and_size(self):
        command = []

        def capture(args, **_kwargs):
            command.extend(args)

        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "guides"
            guides = {"image_size": 768, "roles": {}, "bounds_low": [0, 0, 0], "bounds_high": [1, 1, 1], "diagonal": 1.0}
            def capture_and_write(args, **_kwargs):
                command.extend(args)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "guides.json").write_text(json.dumps(guides), encoding="utf-8")

            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture_and_write),
            ):
                result = native.render_guide_views(
                    Path("mesh.glb"),
                    output_dir,
                    roles=("front", "back", "left", "right"),
                    size=768,
                    extra_roles=("front-left",),
                )

        spec = json.loads(command[-1])
        self.assertEqual(spec["op"], "render-guide-views")
        self.assertEqual(spec["input"], "mesh.glb")
        self.assertEqual(spec["output_dir"], str(output_dir))
        self.assertEqual(spec["roles"], ["front", "back", "left", "right"])
        self.assertEqual(spec["extra_roles"], ["front-left"])
        self.assertEqual(spec["size"], 768)
        self.assertEqual(result, guides)

    def test_missing_blender_raises(self):
        with patch.object(native, "find_blender", return_value=None):
            with self.assertRaises(RuntimeError):
                native.render_guide_views(Path("mesh.glb"), Path("out"))

    def test_missing_guides_json_raises_a_clear_error(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(native, "find_blender", return_value=Path("/tools/blender")),
            patch.object(native, "run_process", side_effect=lambda *_a, **_k: None),
        ):
            with self.assertRaisesRegex(RuntimeError, "guides.json"):
                native.render_guide_views(Path("mesh.glb"), Path(directory) / "guides")


class RunProjectViewsCamerasTests(unittest.TestCase):
    """task9: settings["cameras"] passes through run_project_views unchanged."""

    def test_settings_cameras_pass_through_unchanged(self):
        command = []
        cameras = {
            "front-left": {
                "azimuth_degrees": 45.0,
                "center": [0.0, 0.0, 1.0],
                "ortho_scale": 2.5,
                "camera_distance": 4.0,
            },
        }

        with TemporaryDirectory() as directory:
            output_glb = Path(directory) / "out.glb"
            report_path = Path(str(output_glb) + ".projection.json")

            def capture(args, **_kwargs):
                command.extend(args)
                output_glb.write_bytes(b"glb")
                report_path.write_text(json.dumps({"views_used": []}), encoding="utf-8")

            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
            ):
                native.run_project_views(
                    Path(directory) / "in.glb",
                    output_glb,
                    {"front-left": Path(directory) / "front-left_clay.png"},
                    {"cameras": cameras, "cycles_samples": 8},
                )

        spec = json.loads(command[-1])
        self.assertEqual(spec["settings"]["cameras"], cameras)


class RunRebuildViewsTests(unittest.TestCase):
    """`overlaid.glb` is the deliverable, `final.glb` the inspectable rebuild;
    approach9.sh exits 0 even when a step fails, so the files are the proof.

    The Pixal3D scripts are optional (the public repository leaves them out),
    so each test stands in its own script path."""

    def setUp(self):
        scripts = TemporaryDirectory()
        self.addCleanup(scripts.cleanup)
        script = Path(scripts.name) / "scripts" / "pixal_mv" / "approach9.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        patcher = patch.object(native, "REBUILD_VIEWS_SCRIPT", script)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _write_workdir_outputs(workdir: Path, *, overlaid: bool = True) -> None:
        (workdir / "overlay" / "gate").mkdir(parents=True, exist_ok=True)
        if overlaid:
            (workdir / "overlaid.glb").write_bytes(b"overlaid-glb")
        (workdir / "final.glb").write_bytes(b"final-glb")
        (workdir / "report.json").write_text(
            json.dumps({"mean_iou": 0.94, "front": {"iou": 0.95}}), encoding="utf-8"
        )
        (workdir / "overlay" / "gate" / "multiview.json").write_text(
            json.dumps({"front": {"masked_ssim": 0.544, "face_ssim": 0.32}}), encoding="utf-8"
        )

    def test_runs_approach9_and_returns_the_iou_report_and_gate(self):
        command: list[str] = []

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_glb = root / "out.glb"
            workdir = root / "out-rebuild"

            def capture(args, **_kwargs):
                command.extend(args)
                self._write_workdir_outputs(workdir)
                return "approach9 log"

            with patch.object(native, "run_process", side_effect=capture):
                report = native.run_rebuild_views(
                    root / "in.glb",
                    output_glb,
                    {"front": root / "front.png", "back": root / "back.png"},
                    {},
                )

            self.assertEqual(output_glb.read_bytes(), b"overlaid-glb")

        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("scripts/pixal_mv/approach9.sh"))
        self.assertEqual(command[2:4], ["--mesh", str(root / "in.glb")])
        self.assertEqual(command[4:6], ["--front", str(root / "front.png")])
        self.assertEqual(command[6:8], ["--out", str(workdir)])
        self.assertEqual(command[8:], ["--back", str(root / "back.png")])
        self.assertEqual(report["backend"], "pixal3d-mv")
        self.assertEqual(report["mv_iou"]["mean_iou"], 0.94)
        self.assertEqual(report["gate"]["front"]["face_ssim"], 0.32)
        self.assertEqual(report["pixal_mv_glb"], str(workdir / "final.glb"))
        self.assertEqual(report["workdir"], str(workdir))

    def test_missing_overlaid_glb_raises_with_the_script_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workdir = root / "out-rebuild"

            def capture(_args, **_kwargs):
                self._write_workdir_outputs(workdir, overlaid=False)
                return "inference_mv failed (see log)"

            with (
                patch.object(native, "run_process", side_effect=capture),
                self.assertRaisesRegex(RuntimeError, "inference_mv failed"),
            ):
                native.run_rebuild_views(
                    root / "in.glb", root / "out.glb", {"front": root / "front.png"}, {},
                )

    def test_a_checkout_without_the_scripts_refuses_before_launching(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(native, "REBUILD_VIEWS_SCRIPT", root / "absent" / "approach9.sh"),
                patch.object(native, "run_process") as run_process,
                self.assertRaisesRegex(RuntimeError, "optional Pixal3D multi-view scripts"),
            ):
                native.run_rebuild_views(
                    root / "in.glb", root / "out.glb", {"front": root / "front.png"}, {},
                )
            run_process.assert_not_called()
            self.assertFalse((root / "out-rebuild").exists())

    def test_a_missing_front_reference_is_refused_before_launching(self):
        with (
            patch.object(native, "run_process") as run_process,
            self.assertRaisesRegex(ValueError, "front reference view"),
        ):
            native.run_rebuild_views(
                Path("in.glb"), Path("out.glb"), {"back": Path("back.png")}, {},
            )
        run_process.assert_not_called()


class GpuDeviceTests(unittest.TestCase):
    @staticmethod
    def _fake_nvidia_smi(used_mb: int, total_mb: int):
        return type("FakeResult", (), {"returncode": 0, "stdout": f"{used_mb}, {total_mb}\n"})()

    def test_gpu_memory_snapshot_reports_free_and_busy_external(self):
        with patch.object(native.subprocess, "run", return_value=self._fake_nvidia_smi(20000, 24564)):
            snapshot = native.gpu_memory_snapshot()
        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.used_mb, 20000)
        self.assertEqual(snapshot.total_mb, 24564)
        self.assertEqual(snapshot.free_mb, 4564)
        self.assertTrue(snapshot.busy_external)

    def test_gpu_memory_snapshot_not_busy_below_threshold(self):
        with patch.object(native.subprocess, "run", return_value=self._fake_nvidia_smi(2000, 24564)):
            snapshot = native.gpu_memory_snapshot()
        self.assertFalse(snapshot.busy_external)

    def test_gpu_memory_snapshot_unavailable_when_nvidia_smi_missing(self):
        with patch.object(native.subprocess, "run", side_effect=FileNotFoundError):
            snapshot = native.gpu_memory_snapshot()
        self.assertFalse(snapshot.available)
        self.assertFalse(snapshot.busy_external)

    def test_resolve_trellis_low_vram_auto_off_when_vram_plentiful(self):
        snapshot = native.GpuMemorySnapshot(available=True, used_mb=1000, total_mb=24564, free_mb=23564, busy_external=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AISMITH_TRELLIS_LOW_VRAM", None)
            self.assertFalse(native.resolve_trellis_low_vram(snapshot))

    def test_resolve_trellis_low_vram_auto_on_when_vram_scarce(self):
        snapshot = native.GpuMemorySnapshot(available=True, used_mb=20000, total_mb=24564, free_mb=4564, busy_external=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AISMITH_TRELLIS_LOW_VRAM", None)
            self.assertTrue(native.resolve_trellis_low_vram(snapshot))

    def test_resolve_trellis_low_vram_auto_on_when_gpu_unknown(self):
        snapshot = native.GpuMemorySnapshot(available=False, used_mb=0, total_mb=0, free_mb=0, busy_external=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AISMITH_TRELLIS_LOW_VRAM", None)
            self.assertTrue(native.resolve_trellis_low_vram(snapshot))

    def test_resolve_trellis_low_vram_env_override_forces_on(self):
        snapshot = native.GpuMemorySnapshot(available=True, used_mb=1000, total_mb=24564, free_mb=23564, busy_external=False)
        with patch.dict(os.environ, {"AISMITH_TRELLIS_LOW_VRAM": "1"}):
            self.assertTrue(native.resolve_trellis_low_vram(snapshot))

    def test_resolve_trellis_low_vram_env_override_forces_off(self):
        snapshot = native.GpuMemorySnapshot(available=True, used_mb=20000, total_mb=24564, free_mb=4564, busy_external=True)
        with patch.dict(os.environ, {"AISMITH_TRELLIS_LOW_VRAM": "0"}):
            self.assertFalse(native.resolve_trellis_low_vram(snapshot))

    def test_wait_for_gpu_preflight_returns_immediately_when_free(self):
        free_snapshot = native.GpuMemorySnapshot(True, 1000, 24564, 23564, False)
        with (
            patch.object(native, "gpu_memory_snapshot", return_value=free_snapshot),
            patch.object(native, "gpu_holders", return_value=[]),
        ):
            snapshot = native.wait_for_gpu_preflight(max_wait_seconds=5)
        self.assertFalse(snapshot.busy_external)

    def test_wait_for_gpu_preflight_polls_until_free(self):
        snapshots = [
            native.GpuMemorySnapshot(True, 20000, 24564, 4564, True),
            native.GpuMemorySnapshot(True, 20000, 24564, 4564, True),
            native.GpuMemorySnapshot(True, 1000, 24564, 23564, False),
        ]
        with (
            patch.object(native, "gpu_memory_snapshot", side_effect=snapshots),
            patch.object(native, "gpu_holders", return_value=[]),
            patch.object(native.time, "sleep") as sleep_mock,
        ):
            snapshot = native.wait_for_gpu_preflight(poll_seconds=1, max_wait_seconds=30)
        self.assertFalse(snapshot.busy_external)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_wait_for_gpu_preflight_raises_naming_holders_after_max_wait(self):
        """Was 'gives up ... proceeds anyway' -- the 2026-08-31 incident (bead
        xckqr) showed a preflight that never fails just hides a genuinely
        stuck GPU. It must now fail loudly, naming the holder(s), instead of
        silently degrading a job onto a GPU that's still busy."""
        busy = native.GpuMemorySnapshot(True, 20000, 24564, 4564, True)
        holder = native.GpuHolder(pid=4242, used_mb=20000, cmdline="some-external-renderer --port 9999", label="external")
        with (
            patch.object(native, "gpu_memory_snapshot", return_value=busy),
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=None),
            patch.object(native, "_active_process_pid", return_value=None),
            patch.object(native.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                native.wait_for_gpu_preflight(poll_seconds=10, max_wait_seconds=15)
        self.assertIn("GPU busy timeout after 15s", str(ctx.exception))
        self.assertIn("4242", str(ctx.exception))
        self.assertIn("external", str(ctx.exception))

    def test_wait_for_gpu_preflight_logs_countdown_while_waiting(self):
        snapshots = [
            native.GpuMemorySnapshot(True, 20000, 24564, 4564, True),
            native.GpuMemorySnapshot(True, 1000, 24564, 23564, False),
        ]
        messages: list[str] = []
        with (
            patch.object(native, "gpu_memory_snapshot", side_effect=snapshots),
            patch.object(native, "gpu_holders", return_value=[]),
            patch.object(native.time, "sleep"),
        ):
            native.wait_for_gpu_preflight(progress=messages.append, poll_seconds=15, max_wait_seconds=600)
        countdown_messages = [message for message in messages if message.startswith("GPU busy:")]
        self.assertEqual(len(countdown_messages), 1)
        self.assertIn("waiting 0s of 600s", countdown_messages[0])

    def test_wait_for_gpu_preflight_stops_when_job_is_cancelled_mid_wait(self):
        """A job cancelled while it's still stuck waiting for the GPU -- before
        any worker subprocess exists for cancel_active_process() to kill --
        must not be left to spawn a brand new GPU worker once the GPU frees
        up later. Reproduces the exact gap hit live during bead xckqr
        verification: job 94be8a73 was cancelled mid-preflight-wait and, pre-
        fix, would have gone on to spawn its worker regardless."""
        busy = native.GpuMemorySnapshot(True, 20000, 24564, 4564, True)
        with (
            patch.object(native, "gpu_memory_snapshot", return_value=busy),
            patch.object(native, "gpu_holders", return_value=[]),
            patch.object(native.time, "sleep"),
        ):
            with self.assertRaises(InterruptedError):
                native.wait_for_gpu_preflight(
                    poll_seconds=1, max_wait_seconds=60, cancelled=lambda: True
                )

    def test_wait_for_gpu_preflight_not_cancelled_proceeds_normally(self):
        free_snapshot = native.GpuMemorySnapshot(True, 1000, 24564, 23564, False)
        with (
            patch.object(native, "gpu_memory_snapshot", return_value=free_snapshot),
            patch.object(native, "gpu_holders", return_value=[]),
        ):
            snapshot = native.wait_for_gpu_preflight(max_wait_seconds=5, cancelled=lambda: False)
        self.assertFalse(snapshot.busy_external)

    def _run_trellis_fp8_with_snapshot(self, snapshot):
        captured_env = {}
        captured_spec = {}

        def capture(args, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            captured_spec.update(json.loads(args[-1]))
            return "DEVICE_REPORT_JSON:" + json.dumps({"device": "cuda", "free_gb": 4.5, "total_gb": 24.0})

        with (
            patch.object(native, "trellis_refiner_ready", return_value=True),
            patch.object(native, "trellis_refiner_python", return_value=Path("/tools/python")),
            patch.object(native, "trellis_refiner_dir", return_value=Path("/tools/runtime")),
            patch.object(native, "run_process", side_effect=capture),
            patch.object(native, "_glb_has_texture", return_value=True),
            patch.object(native, "wait_for_gpu_preflight", return_value=snapshot),
            patch.object(native, "trellis2_worker_env", return_value={"LD_LIBRARY_PATH": "/fake"}),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AISMITH_TRELLIS_LOW_VRAM", None)
            result = native.run_trellis_fp8(Path("front.png"), Path("output.glb"), 1024)
        return result, captured_spec, captured_env

    def test_run_trellis_fp8_auto_low_vram_when_vram_scarce(self):
        scarce = native.GpuMemorySnapshot(True, 20000, 24564, 4564, False)
        result, captured_spec, captured_env = self._run_trellis_fp8_with_snapshot(scarce)

        self.assertFalse(captured_spec["gpu_first"])
        self.assertTrue(result["low_vram"])
        self.assertEqual(captured_env, {"LD_LIBRARY_PATH": "/fake"})
        self.assertEqual(result["device"], "cuda")

    def test_run_trellis_fp8_auto_full_gpu_when_vram_plentiful(self):
        plentiful = native.GpuMemorySnapshot(True, 1000, 24564, 23564, False)
        result, captured_spec, _ = self._run_trellis_fp8_with_snapshot(plentiful)

        self.assertTrue(captured_spec["gpu_first"])
        self.assertFalse(result["low_vram"])

    def test_blender_worker_env_prepends_wsl_lib_when_present(self):
        with TemporaryDirectory() as directory:
            wsl_lib = Path(directory) / "wsl-lib"
            wsl_lib.mkdir()
            with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing/path"}, clear=False):
                env = native.blender_worker_env(wsl_lib=wsl_lib)
            self.assertEqual(env["LD_LIBRARY_PATH"], f"{wsl_lib}{os.pathsep}/existing/path")

    def test_blender_worker_env_sets_ld_library_path_when_previously_unset(self):
        with TemporaryDirectory() as directory:
            wsl_lib = Path(directory) / "wsl-lib"
            wsl_lib.mkdir()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LD_LIBRARY_PATH", None)
                env = native.blender_worker_env(wsl_lib=wsl_lib)
            self.assertEqual(env["LD_LIBRARY_PATH"], str(wsl_lib))

    def test_blender_worker_env_unchanged_when_wsl_lib_missing(self):
        missing = Path("/does/not/exist/wsl-lib")
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing/path"}, clear=False):
            env = native.blender_worker_env(wsl_lib=missing)
        self.assertEqual(env.get("LD_LIBRARY_PATH"), "/existing/path")

    def test_blender_worker_env_passes_through_device_override(self):
        with patch.dict(os.environ, {"AISMITH_BLENDER_DEVICE": "cuda"}, clear=False):
            env = native.blender_worker_env(wsl_lib=Path("/does/not/exist"))
        self.assertEqual(env["AISMITH_BLENDER_DEVICE"], "cuda")

    @staticmethod
    def _write_glb(path: Path, document: dict) -> None:
        json_chunk = json.dumps(document).encode("utf-8")
        json_chunk += b" " * (-len(json_chunk) % 4)
        total_length = 12 + 8 + len(json_chunk)
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, total_length)
            + struct.pack("<I4s", len(json_chunk), b"JSON")
            + json_chunk
        )

    def test_blender_bake_retopology_launches_with_blender_worker_env(self):
        captured_env = {}

        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.glb"
            output = Path(directory) / "output.glb"

            def capture(args, **kwargs):
                captured_env.update(kwargs.get("env") or {})
                self._write_glb(
                    output,
                    {
                        "asset": {"version": "2.0"},
                        "nodes": [{"mesh": 0}],
                        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                        "accessors": [{"min": [-1, -2, -0.5], "max": [1, 2, 0.5]}],
                    },
                )

            self._write_glb(source, {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}]})
            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
                patch.object(native, "blender_worker_env", return_value={"LD_LIBRARY_PATH": "/usr/lib/wsl/lib"}),
            ):
                native.blender_bake_retopology(source, Path("remeshed.obj"), output)

        self.assertEqual(captured_env, {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib"})

    def test_run_deformation_diagnostics_render_review_launches_with_blender_worker_env(self):
        captured_envs = []

        def capture(args, **kwargs):
            captured_envs.append(kwargs.get("env"))
            spec = json.loads(args[-1])
            if spec["op"] == "deformation-diagnostics":
                Path(spec["output"]).write_text(json.dumps({}), encoding="utf-8")
            else:
                Path(spec["output"]).write_bytes(b"fake-png")
            return ""

        with TemporaryDirectory() as directory:
            input_glb = Path(directory) / "input.glb"
            input_glb.write_bytes(b"glb")
            output_json = Path(directory) / "report.json"
            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
                patch.object(
                    native,
                    "deformation_review_frame_specs",
                    return_value=[("frontish", 0.35, False, None)],
                ),
                patch.object(native, "blender_worker_env", return_value={"LD_LIBRARY_PATH": "/usr/lib/wsl/lib"}),
            ):
                report = native.run_deformation_diagnostics(input_glb, output_json)

        self.assertIsNone(captured_envs[0])
        self.assertEqual(captured_envs[1], {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib"})
        self.assertEqual(len(report["_review_files"]), 1)

    def test_run_skeleton_sheet_removes_its_working_directory_but_keeps_review_files(self):
        from PIL import Image

        render_dirs = []

        def render_views(args, **_kwargs):
            spec = json.loads(args[-1])
            self.assertEqual(spec["op"], "render-skeleton-sheet")
            render_dirs.append(Path(spec["output_dir"]))
            for role in ("front", "right", "left", "back"):
                Image.new("RGB", (8, 8), (40, 40, 40)).save(Path(spec["output_dir"]) / f"{role}.png")
            return ""

        with TemporaryDirectory() as directory:
            input_glb = Path(directory) / "input.glb"
            input_glb.write_bytes(b"glb")
            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=render_views),
                patch.object(native, "blender_worker_env", return_value={}),
            ):
                report = native.run_skeleton_sheet(input_glb)

        review_files = report["_review_files"]
        try:
            self.assertEqual(
                [item["label"] for item in review_files],
                ["skeleton-sheet", "front", "right", "left", "back"],
            )
            self.assertEqual(len(render_dirs), 1)
            self.assertFalse(render_dirs[0].exists(), "mkdtemp working directory must be removed")
            for item in review_files:
                path = Path(item["path"])
                self.assertTrue(path.is_file(), f"{item['label']} must survive for the caller to persist")
                self.assertFalse(path.is_relative_to(render_dirs[0]))
            with Image.open(review_files[0]["path"]) as sheet:
                self.assertEqual(sheet.size, (16, 16))
        finally:
            for item in review_files:
                Path(item["path"]).unlink(missing_ok=True)

    def test_run_skeleton_sheet_removes_its_working_directory_on_failure(self):
        def render_nothing(args, **_kwargs):
            self.render_dir = Path(json.loads(args[-1])["output_dir"])
            return ""

        with TemporaryDirectory() as directory:
            input_glb = Path(directory) / "input.glb"
            input_glb.write_bytes(b"glb")
            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=render_nothing),
                patch.object(native, "blender_worker_env", return_value={}),
            ):
                with self.assertRaises(RuntimeError):
                    native.run_skeleton_sheet(input_glb)
        self.assertFalse(self.render_dir.exists())


class GpuHolderClassificationTests(unittest.TestCase):
    """Pure classification logic for bead xckqr's GPU-busy transparency work:
    who is holding VRAM, and what (if anything) is safe to do about it."""

    def _holder(self, pid: int, cmdline: str = "", used_mb: int = 1000) -> native.GpuHolder:
        return native.GpuHolder(pid=pid, used_mb=used_mb, cmdline=cmdline, label="unclassified")

    def test_classifies_aismith_own_comfyui_by_pid(self):
        holder = self._holder(3261482, cmdline="venv/bin/python main.py --listen 0.0.0.0 --port 8288")
        label = native.classify_gpu_holder(holder, comfyui_pid=3261482, active_pid=None)
        self.assertEqual(label, "aismith-comfyui")

    def test_classifies_currently_active_worker(self):
        holder = self._holder(555, cmdline="python backend/workers/trellis2_fp8_worker.py {}")
        label = native.classify_gpu_holder(holder, comfyui_pid=None, active_pid=555)
        self.assertEqual(label, "aismith-worker-active")

    def test_classifies_orphaned_worker_by_script_marker(self):
        holder = self._holder(999, cmdline="python backend/workers/trellis2_fp8_worker.py {}")
        label = native.classify_gpu_holder(holder, comfyui_pid=None, active_pid=None)
        self.assertEqual(label, "aismith-worker-orphaned")

    def test_active_worker_takes_priority_over_orphan_marker(self):
        # Same script marker, but it IS the pid we're currently tracking as
        # active -- must never be classified (and later reclaimed) as an orphan.
        holder = self._holder(777, cmdline="python backend/workers/trellis2_fp8_worker.py {}")
        label = native.classify_gpu_holder(holder, comfyui_pid=None, active_pid=777)
        self.assertEqual(label, "aismith-worker-active")

    def test_classifies_unrecognized_process_as_external(self):
        holder = self._holder(111, cmdline="/opt/comfyui/venv/bin/python main.py --port 8189")
        label = native.classify_gpu_holder(holder, comfyui_pid=None, active_pid=None)
        self.assertEqual(label, "external")

    def test_describe_gpu_holders_names_pid_label_and_cmdline(self):
        holders = [native.GpuHolder(pid=42, used_mb=12345, cmdline="main.py --port 8288", label="aismith-comfyui")]
        description = native.describe_gpu_holders(holders)
        self.assertIn("pid 42", description)
        self.assertIn("aismith-comfyui", description)
        self.assertIn("12345 MiB", description)
        self.assertIn("main.py --port 8288", description)

    def test_describe_gpu_holders_empty_is_not_blank(self):
        self.assertTrue(native.describe_gpu_holders([]))


class ReclaimGpuHoldersTests(unittest.TestCase):
    """Auto-reclaim decisions: free AISmith's own idle ComfyUI, kill orphaned
    AISmith workers, and never touch anything external (bead xckqr, part 2)."""

    def test_frees_own_comfyui_when_queue_is_empty(self):
        holder = native.GpuHolder(pid=42, used_mb=12900, cmdline="main.py --port 8288", label="unclassified")
        with (
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=42),
            patch.object(native, "_active_process_pid", return_value=None),
            patch.object(native, "_comfyui_base_url", return_value="http://127.0.0.1:8288"),
            patch.object(native, "_comfyui_queue_empty", return_value=True),
            patch.object(native, "_comfyui_free", return_value=True) as free_mock,
        ):
            observed = native.reclaim_gpu_holders()
        free_mock.assert_called_once_with("http://127.0.0.1:8288")
        self.assertEqual(observed[0].label, "aismith-comfyui")

    def test_leaves_own_comfyui_alone_when_queue_is_not_empty(self):
        holder = native.GpuHolder(pid=42, used_mb=12900, cmdline="main.py --port 8288", label="unclassified")
        with (
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=42),
            patch.object(native, "_active_process_pid", return_value=None),
            patch.object(native, "_comfyui_queue_empty", return_value=False),
            patch.object(native, "_comfyui_free") as free_mock,
        ):
            native.reclaim_gpu_holders()
        free_mock.assert_not_called()

    def test_kills_orphaned_worker(self):
        holder = native.GpuHolder(pid=999, used_mb=8000, cmdline="python backend/workers/trellis2_fp8_worker.py {}", label="unclassified")
        with (
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=None),
            patch.object(native, "_active_process_pid", return_value=None),
            patch.object(native, "terminate_process_group", return_value=True) as kill_mock,
        ):
            native.reclaim_gpu_holders()
        kill_mock.assert_called_once()
        self.assertEqual(kill_mock.call_args.args[0], 999)

    def test_never_kills_the_currently_active_worker(self):
        holder = native.GpuHolder(pid=555, used_mb=8000, cmdline="python backend/workers/trellis2_fp8_worker.py {}", label="unclassified")
        with (
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=None),
            patch.object(native, "_active_process_pid", return_value=555),
            patch.object(native, "terminate_process_group") as kill_mock,
            patch.object(native, "_comfyui_free") as free_mock,
        ):
            native.reclaim_gpu_holders()
        kill_mock.assert_not_called()
        free_mock.assert_not_called()

    def test_never_touches_external_holder(self):
        holder = native.GpuHolder(pid=321, used_mb=12000, cmdline="other-comfyui-renderer", label="unclassified")
        with (
            patch.object(native, "gpu_holders", return_value=[holder]),
            patch.object(native, "_pid_listening_on_port", return_value=None),
            patch.object(native, "_active_process_pid", return_value=None),
            patch.object(native, "terminate_process_group") as kill_mock,
            patch.object(native, "_comfyui_free") as free_mock,
        ):
            native.reclaim_gpu_holders()
        kill_mock.assert_not_called()
        free_mock.assert_not_called()


class TerminateProcessGroupTests(unittest.TestCase):
    """cancel_active_process()/terminate_process_group(): SIGTERM, escalate
    to SIGKILL after a grace period, confirm exit (bead xckqr, part 1)."""

    def test_terminate_process_group_returns_true_when_already_dead(self):
        self.assertTrue(native.terminate_process_group(999999, is_alive=lambda: False))

    def test_escalates_to_sigkill_when_sigterm_ignored(self):
        # Simulate a worker that ignores SIGTERM (still "alive" through the
        # whole term grace window) but dies once SIGKILLed.
        alive_after_term = {"value": True}

        def is_alive():
            return alive_after_term["value"]

        killpg_calls: list[tuple[int, int]] = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))
            if sig == native.signal.SIGKILL:
                alive_after_term["value"] = False

        with (
            patch.object(native.os, "getpgid", return_value=4242),
            patch.object(native.os, "killpg", side_effect=fake_killpg),
            patch.object(native.time, "sleep"),
        ):
            result = native.terminate_process_group(
                4242, is_alive=is_alive, term_grace_seconds=0.4, kill_confirm_seconds=0.4
            )
        self.assertTrue(result)
        self.assertEqual([sig for _pgid, sig in killpg_calls], [native.signal.SIGTERM, native.signal.SIGKILL])

    def test_reports_failure_when_sigkill_does_not_reclaim_it(self):
        with (
            patch.object(native.os, "getpgid", return_value=4242),
            patch.object(native.os, "killpg"),
            patch.object(native.time, "sleep"),
        ):
            result = native.terminate_process_group(4242, is_alive=lambda: True, term_grace_seconds=0.2, kill_confirm_seconds=0.2)
        self.assertFalse(result)

    def test_cancel_active_process_false_when_nothing_active(self):
        native._set_active_process(None)
        self.assertFalse(native.cancel_active_process())

    def test_cancel_active_process_kills_a_real_process_group(self):
        """Integration-style: spawn a real subprocess that itself forks a
        grandchild (mirroring a worker whose GPU library spawns helper
        processes), track it as the active worker, cancel it, and confirm
        via os.kill(pid, 0) that BOTH the parent and the grandchild are gone
        -- proving the process-GROUP kill, not just a single pid.terminate()."""
        import subprocess as subprocess_module
        import time as time_module

        parent = subprocess_module.Popen(
            ["bash", "-c", "sleep 60 & child=$!; echo $child; wait $child"],
            stdout=subprocess_module.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            grandchild_line = parent.stdout.readline().strip()
            self.assertTrue(grandchild_line.isdigit(), f"expected a pid, got {grandchild_line!r}")
            grandchild_pid = int(grandchild_line)
            self.assertTrue(native._pid_alive(grandchild_pid))

            native._set_active_process(parent)
            killed = native.cancel_active_process()
            self.assertTrue(killed)

            self.assertIsNotNone(parent.poll())
            deadline = time_module.monotonic() + 3
            while native._pid_alive(grandchild_pid) and time_module.monotonic() < deadline:
                time_module.sleep(0.1)
            self.assertFalse(
                native._pid_alive(grandchild_pid),
                "grandchild survived cancel_active_process(); process-group kill did not reach it",
            )
        finally:
            native._set_active_process(None)
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


class RunProcessProcessGroupTests(unittest.TestCase):
    def test_run_process_spawns_in_its_own_session(self):
        captured_kwargs: dict = {}

        class FakeProcess:
            pid = 4321
            stdout = None

            def wait(self, timeout=None):
                return 0

        def fake_popen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeProcess()

        environment = {k: v for k, v in os.environ.items() if k != native.JOB_PROCESS_GROUP_ENV}
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(native.subprocess, "Popen", side_effect=fake_popen):
            native.run_process(["/bin/true"])

        self.assertTrue(captured_kwargs.get("start_new_session"))

    def test_run_process_stays_in_the_job_group_under_the_engine_runner(self):
        captured_kwargs: dict = {}

        class FakeProcess:
            pid = 4321
            stdout = None

            def wait(self, timeout=None):
                return 0

        def fake_popen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeProcess()

        with patch.dict(os.environ, {native.JOB_PROCESS_GROUP_ENV: "1"}), \
             patch.object(native.subprocess, "Popen", side_effect=fake_popen):
            native.run_process(["/bin/true"])

        self.assertNotIn("start_new_session", captured_kwargs)

    def test_a_worker_in_a_shared_group_is_killed_without_a_group_kill(self):
        """Under AISMITH_JOB_PROCESS_GROUP=1 a worker shares the stage's group;
        terminate_process_group must kill it and its children but never
        killpg that shared group, which would kill the stage itself."""
        import subprocess as subprocess_module
        import time as time_module

        parent = subprocess_module.Popen(
            ["bash", "-c", "sleep 60 & child=$!; echo $child; wait $child"],
            stdout=subprocess_module.PIPE,
            text=True,
        )
        real_killpg = os.killpg
        own_group = os.getpgrp()

        def guarded_killpg(pgid, sig):
            if pgid == own_group:
                raise AssertionError("killpg on the caller's own process group")
            return real_killpg(pgid, sig)

        try:
            grandchild_pid = int(parent.stdout.readline().strip())
            with patch.object(native.os, "killpg", side_effect=guarded_killpg):
                killed = native.terminate_process_group(
                    parent.pid, is_alive=lambda: parent.poll() is None, term_grace_seconds=2, kill_confirm_seconds=3
                )
            self.assertTrue(killed)
            deadline = time_module.monotonic() + 10
            while native._pid_alive(grandchild_pid) and time_module.monotonic() < deadline:
                time_module.sleep(0.1)
            self.assertFalse(native._pid_alive(grandchild_pid))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


class CanonicalizeSimilarityTests(unittest.TestCase):
    """S2 appearance-preservation gate: pure numpy scoring on synthetic
    arrays, independent of Blender rendering."""

    def test_identical_images_score_near_perfect_similarity(self):
        image = np.random.default_rng(0).random((64, 64))

        result = native.compute_view_similarity(image, image.copy())

        self.assertAlmostEqual(result["mad"], 0.0)
        self.assertAlmostEqual(result["histogram_distance"], 0.0)
        self.assertAlmostEqual(result["similarity"], 1.0)

    def test_opposite_images_score_low_similarity(self):
        black = np.zeros((32, 32))
        white = np.ones((32, 32))

        result = native.compute_view_similarity(black, white)

        self.assertAlmostEqual(result["mad"], 1.0)
        self.assertLess(result["similarity"], 0.1)

    def test_similarity_is_clamped_into_zero_one(self):
        black = np.zeros((8, 8))
        white = np.ones((8, 8))

        result = native.compute_view_similarity(black, white)

        self.assertGreaterEqual(result["similarity"], 0.0)
        self.assertLessEqual(result["similarity"], 1.0)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            native.compute_view_similarity(np.zeros((8, 8)), np.zeros((4, 4)))

    def test_partial_difference_lands_between_identical_and_opposite(self):
        base = np.full((16, 16), 0.5)
        slightly_off = base.copy()
        slightly_off[:4, :4] = 0.6

        identical = native.compute_view_similarity(base, base.copy())
        partial = native.compute_view_similarity(base, slightly_off)

        self.assertLess(partial["similarity"], identical["similarity"])
        self.assertGreater(partial["similarity"], 0.5)

    def test_evaluate_canonicalize_similarity_aggregates_and_emits_required_key(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"views": {}}
            for view, tint in (("front", 40), ("back", 200)):
                source_path = root / f"{view}_source.png"
                canonical_path = root / f"{view}_canonical.png"
                Image.new("RGB", (16, 16), (tint, tint, tint)).save(source_path)
                Image.new("RGB", (16, 16), (tint, tint, tint)).save(canonical_path)
                manifest["views"][view] = {"source": str(source_path), "canonical": str(canonical_path)}

            report = native.evaluate_canonicalize_similarity(manifest)

        # Identical renders per view -> perfect aggregate similarity.
        self.assertIn("s1_s2_image_similarity_score", report)
        self.assertAlmostEqual(report["s1_s2_image_similarity_score"], 1.0, places=3)
        self.assertEqual(set(report["views"]), {"front", "back"})
        for view_report in report["views"].values():
            self.assertIn("similarity", view_report)


class CanonicalizeDetailPreservationTests(unittest.TestCase):
    """The S1/S2 detail-preservation measurement (2026-09-01 P1).

    The MAD + histogram similarity score above reads 64x64 grayscale
    thumbnails, so it cannot see a texture-resolution loss at all -- a
    visibly muddy canonical face scored 0.997. These tests pin the
    high-frequency measurement that can, on synthetic renders shaped like the
    real ones: a bright textured subject on a near-black ground.
    """

    @staticmethod
    def _render(size=192, seed=0, background=0.05, subject_level=0.55, texture=0.25):
        """A bright, finely textured square subject on a dark ground."""
        rng = np.random.default_rng(seed)
        image = np.full((size, size), background)
        margin = size // 4
        patch = rng.random((size - 2 * margin, size - 2 * margin))
        image[margin:-margin, margin:-margin] = subject_level + texture * (patch - 0.5)
        return image

    @staticmethod
    def _blur(image, passes=1):
        """`passes` rounds of a 3x3 neighbour average -- no scipy needed."""
        blurred = np.asarray(image, dtype=np.float64)
        for _ in range(passes):
            blurred = sum(
                np.roll(blurred, (row, column), axis=(0, 1))
                for row in (-1, 0, 1)
                for column in (-1, 0, 1)
            ) / 9.0
        return blurred

    def test_laplacian_variance_is_zero_on_a_flat_image(self):
        self.assertAlmostEqual(native.laplacian_variance(np.full((32, 32), 0.4)), 0.0)

    def test_laplacian_variance_drops_when_an_image_is_blurred(self):
        sharp = self._render()

        self.assertLess(
            native.laplacian_variance(self._blur(sharp)), native.laplacian_variance(sharp)
        )

    def test_laplacian_variance_is_zero_for_a_degenerate_crop(self):
        self.assertEqual(native.laplacian_variance(np.zeros((2, 40))), 0.0)

    def test_laplacian_variance_rejects_a_non_2d_array(self):
        with self.assertRaises(ValueError):
            native.laplacian_variance(np.zeros((8, 8, 3)))

    def test_laplacian_variance_rejects_a_mismatched_mask(self):
        with self.assertRaises(ValueError):
            native.laplacian_variance(np.zeros((8, 8)), np.ones((4, 4), dtype=bool))

    def test_a_mask_confines_the_measurement_to_the_selected_pixels(self):
        image = np.full((32, 32), 0.4)
        image[:16] += np.random.default_rng(1).random((16, 32)) * 0.3
        quiet_half = np.zeros((32, 32), dtype=bool)
        quiet_half[20:30, 2:30] = True

        self.assertAlmostEqual(native.laplacian_variance(image, quiet_half), 0.0)
        self.assertGreater(native.laplacian_variance(image), 0.0)

    def test_lit_surface_mask_selects_the_subject_and_drops_the_background(self):
        image = self._render()

        mask = native.lit_surface_mask(image, image.copy())

        self.assertGreater(mask.sum(), 0)
        # Background is never selected, and the silhouette edge is eroded away.
        self.assertFalse(mask[:4, :4].any())
        self.assertTrue(mask[96, 96])

    def test_lit_surface_mask_is_the_intersection_of_both_renders(self):
        image = self._render()
        half_dark = image.copy()
        half_dark[:, 96:] = 0.05

        mask = native.lit_surface_mask(image, half_dark)

        self.assertFalse(mask[:, 100:].any())
        self.assertTrue(mask[:, :90].any())

    def test_lit_surface_mask_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            native.lit_surface_mask(np.zeros((8, 8)), np.zeros((4, 4)))

    def test_identical_renders_preserve_all_detail(self):
        sharp = self._render()

        result = native.compute_view_detail(sharp, sharp.copy())

        self.assertAlmostEqual(result["detail_ratio"], 1.0, places=6)
        self.assertGreater(result["detail_pixels"], 0)

    def test_a_blurred_canonical_render_scores_well_below_one(self):
        sharp = self._render()

        result = native.compute_view_detail(sharp, self._blur(sharp))

        self.assertLess(result["detail_ratio"], 0.5)
        self.assertGreater(result["detail_ratio"], 0.0)
        self.assertLess(result["canonical_detail"], result["source_detail"])

    def test_more_blur_scores_lower_than_less_blur(self):
        sharp = self._render()

        gentle = native.compute_view_detail(sharp, self._blur(sharp, passes=1))
        heavy = native.compute_view_detail(sharp, self._blur(sharp, passes=4))

        self.assertLess(heavy["detail_ratio"], gentle["detail_ratio"])

    def test_the_silhouette_alone_cannot_hide_a_blurred_texture(self):
        # The whole-frame measurement this replaced scored 0.96 on a visibly
        # muddy bake because the subject's outline -- identical geometry in
        # both renders -- dominated the high-frequency energy.
        sharp = self._render()
        blurred_texture = sharp.copy()
        blurred_texture[48:-48, 48:-48] = self._blur(sharp, passes=3)[48:-48, 48:-48]

        masked = native.compute_view_detail(sharp, blurred_texture)["detail_ratio"]
        whole_frame = (
            native.laplacian_variance(blurred_texture) / native.laplacian_variance(sharp)
        )

        self.assertLess(masked, whole_frame)
        self.assertLess(masked, 0.6)

    def test_ratio_is_capped_so_a_noisy_canonical_cannot_dominate_the_aggregate(self):
        flat_subject = self._render(texture=0.0)
        noisy = self._render(seed=3, texture=0.25)

        result = native.compute_view_detail(flat_subject, noisy)

        self.assertEqual(result["detail_ratio"], native.MAX_DETAIL_RATIO)

    def test_two_featureless_renders_count_as_fully_preserved(self):
        flat = self._render(texture=0.0)

        result = native.compute_view_detail(flat, flat.copy())

        self.assertEqual(result["detail_ratio"], 1.0)

    def _write_manifest(self, root, views):
        from PIL import Image

        manifest = {"views": {}, "face_fraction": 0.3, "size": 192}
        for view, (source, canonical, face_source, face_canonical) in views.items():
            paths = {}
            for name, array in (
                ("source", source), ("canonical", canonical),
                ("source_face", face_source), ("canonical_face", face_canonical),
            ):
                path = root / f"{view}_{name}.png"
                Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").save(path)
                paths[name] = str(path)
            manifest["views"][view] = paths
        return manifest

    def test_evaluate_canonicalize_similarity_records_the_detail_ratio(self):
        sharp = self._render()
        blurred = self._blur(sharp, passes=3)

        with TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), {
                "front": (sharp, sharp, sharp, blurred),
                "back": (sharp, sharp, sharp, blurred),
            })

            report = native.evaluate_canonicalize_similarity(manifest)

        self.assertIn("s1_s2_detail_ratio", report)
        self.assertLess(report["s1_s2_detail_ratio"], 0.5)
        self.assertEqual(report["detail_views_measured"], 2)
        self.assertEqual(report["face_fraction"], 0.3)
        for view_report in report["views"].values():
            self.assertIn("detail_ratio", view_report)
            self.assertIn("similarity", view_report)

    def test_a_view_with_no_lit_surface_is_left_out_of_the_aggregate(self):
        sharp = self._render()
        blurred = self._blur(sharp, passes=3)
        dark = np.full((192, 192), 0.05)

        with TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), {
                "front": (sharp, sharp, sharp, blurred),
                "left": (sharp, sharp, dark, dark),
            })

            report = native.evaluate_canonicalize_similarity(manifest)

        self.assertEqual(report["detail_views_measured"], 1)
        self.assertEqual(
            report["views"]["left"]["detail_excluded"], "insufficient lit surface"
        )

    def test_a_manifest_without_face_renders_omits_the_detail_ratio(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "front_source.png"
            canonical_path = root / "front_canonical.png"
            Image.new("RGB", (16, 16), (90, 90, 90)).save(source_path)
            Image.new("RGB", (16, 16), (90, 90, 90)).save(canonical_path)
            manifest = {"views": {"front": {
                "source": str(source_path), "canonical": str(canonical_path),
            }}}

            report = native.evaluate_canonicalize_similarity(manifest)

        self.assertNotIn("s1_s2_detail_ratio", report)
        self.assertIn("s1_s2_image_similarity_score", report)

    def test_full_resolution_load_does_not_thumbnail_the_render(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            path = Path(directory) / "render.png"
            Image.new("L", (128, 128), 120).save(path)

            self.assertEqual(native._load_grayscale_array(path, None).shape, (128, 128))
            self.assertEqual(native._load_grayscale_array(path, 64).shape, (64, 64))


class RunCanonicalizeInstantMeshesTests(unittest.TestCase):
    """Worker-level orchestration of the `method="instant-meshes"` remesh
    path -- kept reachable for comparison against the default voxel path
    below (2026-08-30: Instant Meshes shattered one character's S1 mesh
    into 800+ shards at every vertex target). Mocks every
    Blender-boundary collaborator but exercises the real `repair_meshfix`
    output through the real `mesh_quality.analyze_mesh` so the retry-knob
    loop is tested against genuine measurements."""

    def _run(self, component_sequence, target_vertices=20_000):
        attempt_index = {"value": 0}

        def fake_run_process(args, **_kwargs):
            output_index = args.index("--output") + 1
            Path(args[output_index]).write_bytes(b"0" * 128)
            return ""

        def fake_repair_meshfix(input_obj, output_obj, progress=None):
            trimesh.creation.box().export(output_obj, file_type="obj")
            return {"repaired_components": 1, "kept_components": 0}

        def fake_analyze_mesh(mesh):
            index = min(attempt_index["value"], len(component_sequence) - 1)
            attempt_index["value"] += 1
            components = component_sequence[index]
            return {
                "components": components,
                "watertight": components == 1,
                "non_manifold_edges": 0,
                "faces": 30_000,
            }

        def fake_normalize_floor(input_glb, output_glb, progress=None):
            Path(output_glb).write_bytes(b"floor-normalized")
            return {"floor_offset_z": 0.0}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_glb = root / "source.glb"
            input_glb.write_bytes(b"glb")
            output_glb = root / "output.glb"
            with (
                patch.object(native, "instant_meshes_exe", return_value=Path("/fake/instant-meshes")),
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "blender_export_obj", return_value=None),
                patch.object(native, "run_process", side_effect=fake_run_process),
                patch.object(native, "repair_meshfix", side_effect=fake_repair_meshfix),
                patch("backend.app.mesh_quality.analyze_mesh", side_effect=fake_analyze_mesh),
                patch.object(native, "blender_bake_retopology", return_value=None),
                patch.object(native, "_blender_normalize_floor", side_effect=fake_normalize_floor),
                patch.object(native, "_glb_mesh_nodes_have_identity_transforms", return_value=True),
                patch.object(native, "_glb_axis_profile_matches", return_value=True),
                patch.object(
                    native, "render_canonicalize_similarity_views",
                    return_value={"views": {"front": {"source": "s.png", "canonical": "c.png"}}},
                ),
                patch.object(
                    native, "evaluate_canonicalize_similarity",
                    return_value={
                        "schema_version": 1, "views": {},
                        "s1_s2_image_similarity_score": 0.95, "aggregate_distance": 0.05,
                    },
                ),
            ):
                return native.run_canonicalize(
                    input_glb, output_glb, target_vertices, method="instant-meshes",
                )

    def test_reaches_contract_on_the_first_attempt(self):
        result = self._run(component_sequence=[1])

        self.assertTrue(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertNotIn("warnings", result)
        self.assertEqual(result["s1_s2_image_similarity_score"], 0.95)
        self.assertEqual(result["similarity"]["s1_s2_image_similarity_score"], 0.95)
        self.assertEqual(result["remesh_method"], "instant-meshes")
        self.assertEqual(result["backend"], "instant-meshes+meshfix+blender-canonicalize-bake")

    def test_iterates_and_lowers_the_vertex_target_until_the_contract_is_met(self):
        result = self._run(component_sequence=[4, 2, 1], target_vertices=20_000)

        self.assertTrue(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 3)
        vertex_targets = [attempt["target_vertices"] for attempt in result["attempts"]]
        self.assertEqual(vertex_targets, sorted(vertex_targets, reverse=True))
        self.assertGreater(vertex_targets[0], vertex_targets[-1])
        self.assertEqual(vertex_targets[0], 20_000)

    def test_gives_up_after_at_most_two_retries_with_a_clear_finding(self):
        result = self._run(component_sequence=[5, 5, 5])

        self.assertFalse(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("warnings", result)
        self.assertIn("1 component", result["warnings"][0])
        # every attempt's measurement is preserved for the operator, not
        # just the last one.
        self.assertEqual([attempt["components"] for attempt in result["attempts"]], [5, 5, 5])

    def test_missing_instant_meshes_raises_only_for_this_method(self):
        with (
            patch.object(native, "instant_meshes_exe", return_value=None),
            patch.object(native, "find_blender", return_value=Path("/fake/blender")),
        ):
            with self.assertRaises(RuntimeError):
                native.run_canonicalize(
                    Path("/tmp/source.glb"), Path("/tmp/output.glb"), 20_000,
                    method="instant-meshes",
                )


class RunCanonicalizeVoxelTests(unittest.TestCase):
    """Worker-level orchestration of the default `method="voxel"` remesh
    path (2026-08-30: replaces Instant Meshes as the primary S2 backend --
    see docs/PIPELINE_POLICY.md). Mocks the Blender-
    boundary voxel remesh call itself (its bisection/decimation arithmetic
    is exercised directly by `VoxelRemeshPolicyTests` against the real
    `backend.app.voxel_remesh` helpers) but runs the real
    `mesh_quality.analyze_mesh` over the produced geometry, exactly like the
    Instant Meshes variant above."""

    def _run(self, component_sequence, target_vertices=20_000, method=None, settings=None):
        attempt_index = {"value": 0}
        voxel_calls: list[dict] = []

        def fake_voxel_remesh(input_glb, output_obj, target_faces, raw_face_min, raw_face_max, progress=None):
            voxel_calls.append({
                "target_faces": target_faces,
                "raw_face_min": raw_face_min,
                "raw_face_max": raw_face_max,
            })
            trimesh.creation.box().export(output_obj, file_type="obj")
            return {
                "method": "voxel",
                "target_faces": target_faces,
                "raw_face_band": [raw_face_min, raw_face_max],
                "voxel_size": 0.02,
                "raw_faces": 200_000,
                "decimated_faces": target_faces,
                "decimate_ratio": target_faces / 200_000,
            }

        def fake_analyze_mesh(mesh):
            index = min(attempt_index["value"], len(component_sequence) - 1)
            attempt_index["value"] += 1
            components = component_sequence[index]
            return {
                "components": components,
                "watertight": components == 1,
                "non_manifold_edges": 0,
                "faces": 30_000,
            }

        def fake_normalize_floor(input_glb, output_glb, progress=None):
            Path(output_glb).write_bytes(b"floor-normalized")
            return {"floor_offset_z": 0.0}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_glb = root / "source.glb"
            input_glb.write_bytes(b"glb")
            output_glb = root / "output.glb"
            with (
                patch.object(native, "find_blender", return_value=Path("/fake/blender")),
                patch.object(native, "blender_voxel_remesh_canonicalize", side_effect=fake_voxel_remesh),
                patch("backend.app.mesh_quality.analyze_mesh", side_effect=fake_analyze_mesh),
                patch.object(native, "blender_bake_retopology", return_value=None),
                patch.object(native, "_blender_normalize_floor", side_effect=fake_normalize_floor),
                patch.object(native, "_glb_mesh_nodes_have_identity_transforms", return_value=True),
                patch.object(native, "_glb_axis_profile_matches", return_value=True),
                patch.object(
                    native, "render_canonicalize_similarity_views",
                    return_value={"views": {"front": {"source": "s.png", "canonical": "c.png"}}},
                ),
                patch.object(
                    native, "evaluate_canonicalize_similarity",
                    return_value={
                        "schema_version": 1, "views": {},
                        "s1_s2_image_similarity_score": 0.95, "aggregate_distance": 0.05,
                    },
                ),
            ):
                kwargs = {} if method is None else {"method": method}
                if settings is not None:
                    kwargs["settings"] = settings
                result = native.run_canonicalize(input_glb, output_glb, target_vertices, **kwargs)
            return result, voxel_calls

    def test_voxel_is_the_default_method(self):
        result, _calls = self._run(component_sequence=[1])

        self.assertEqual(result["remesh_method"], "voxel")
        self.assertEqual(result["backend"], "blender-voxel-remesh+blender-canonicalize-bake")
        self.assertIsNone(result["final_target_vertices"])
        self.assertEqual(result["final_target_faces"], native.DEFAULT_CANONICALIZE_TARGET_FACES)

    def test_reaches_contract_on_the_first_attempt(self):
        result, calls = self._run(component_sequence=[1])

        self.assertTrue(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertNotIn("warnings", result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["s1_s2_image_similarity_score"], 0.95)

    def test_iterates_and_grows_the_raw_band_until_the_contract_is_met(self):
        result, calls = self._run(component_sequence=[4, 2, 1])

        self.assertTrue(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 3)
        raw_mins = [call["raw_face_min"] for call in calls]
        self.assertEqual(raw_mins, sorted(raw_mins))
        self.assertGreater(raw_mins[-1], raw_mins[0])
        self.assertEqual(raw_mins[0], native.DEFAULT_VOXEL_RAW_FACE_MIN)

    def test_gives_up_after_at_most_two_retries_with_a_clear_finding(self):
        result, _calls = self._run(component_sequence=[5, 5, 5])

        self.assertFalse(result["contract_met"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("warnings", result)
        self.assertIn("1 component", result["warnings"][0])
        self.assertEqual([attempt["components"] for attempt in result["attempts"]], [5, 5, 5])

    def test_provenance_records_method_and_voxel_report(self):
        result, _calls = self._run(component_sequence=[1])

        attempt = result["attempts"][0]
        self.assertEqual(attempt["method"], "voxel")
        self.assertIn("voxel_remesh", attempt)
        self.assertEqual(attempt["voxel_remesh"]["raw_faces"], 200_000)
        self.assertEqual(attempt["target_faces"], native.DEFAULT_CANONICALIZE_TARGET_FACES)

    def test_a_settings_free_run_is_labelled_standard_and_unchanged(self):
        result, calls = self._run(component_sequence=[1])

        self.assertEqual(result["quality_tier"], "standard")
        self.assertIsNone(result["quality_tier_requested"])
        self.assertEqual(result["target_triangles"], 30_000)
        self.assertEqual(result["triangle_budget_band"], [15_000, 45_000])
        self.assertEqual(calls[0]["target_faces"], 30_000)
        self.assertEqual((calls[0]["raw_face_min"], calls[0]["raw_face_max"]), (150_000, 350_000))

    def test_the_ultra_tier_scales_the_decimation_target_and_the_raw_search(self):
        result, calls = self._run(component_sequence=[1], settings={"quality_tier": "ultra"})

        self.assertEqual(result["quality_tier"], "ultra")
        self.assertEqual(result["target_triangles"], 150_000)
        # Ultra's raw search is 5x its own target, exactly the oversampling
        # Standard has always had -- not the 30k-tuned 150k-350k band, which
        # would have made the decimation a no-op.
        self.assertEqual((calls[0]["raw_face_min"], calls[0]["raw_face_max"]), (750_000, 1_750_000))
        self.assertEqual(calls[0]["target_faces"], 150_000)

    def test_the_gate_band_recorded_is_the_selected_tier_s_band(self):
        result, _calls = self._run(component_sequence=[1], settings={"quality_tier": "ultra"})

        self.assertEqual(result["triangle_budget_band"], [75_000, 225_000])
        self.assertEqual(result["bake_atlas_minimum"], 4096)

    def test_a_custom_triangle_target_overrides_the_tier_end_to_end(self):
        result, calls = self._run(
            component_sequence=[1], settings={"quality_tier": "draft", "target_triangles": 96_000},
        )

        self.assertEqual(result["quality_tier"], "custom")
        self.assertEqual(result["quality_tier_requested"], "draft")
        self.assertEqual(result["target_triangles"], 96_000)
        self.assertEqual(result["triangle_budget_band"], [48_000, 144_000])
        self.assertEqual(calls[0]["target_faces"], 96_000)

    def test_the_retry_escalation_scales_from_the_selected_target(self):
        _result, calls = self._run(
            component_sequence=[4, 2, 1], settings={"quality_tier": "high"},
        )

        raw_mins = [call["raw_face_min"] for call in calls]
        self.assertEqual(raw_mins[0], 400_000)  # 5x High's 80k target
        self.assertEqual(raw_mins, sorted(raw_mins))
        self.assertGreater(raw_mins[-1], raw_mins[0])
        # Every retry re-derives from the target, so the absolute raw ceiling
        # is applied on the escalated band too, not just the first one.
        from backend.app.voxel_remesh import VOXEL_RAW_FACE_CEILING
        self.assertLessEqual(max(call["raw_face_max"] for call in calls), VOXEL_RAW_FACE_CEILING)

    def test_the_achieved_triangle_count_is_reported_next_to_the_target(self):
        result, _calls = self._run(component_sequence=[1], settings={"quality_tier": "ultra"})

        # `analyze_mesh` is faked at 30k here; the point is that what the mesh
        # CAME OUT at is recorded next to what was asked for, so the verdict
        # card can show both.
        self.assertEqual(result["achieved_triangles"], 30_000)
        self.assertEqual(result["achieved_triangles"], result["attempts"][-1]["faces"])

    def test_an_unknown_tier_is_refused_before_any_blender_work(self):
        with self.assertRaises(ValueError):
            self._run(component_sequence=[1], settings={"quality_tier": "cinematic"})


class VoxelRemeshPolicyTests(unittest.TestCase):
    """Pure arithmetic backing the voxel remesh's search strategy
    (`backend.app.voxel_remesh`), unit tested directly without any Blender
    boundary -- the same split `proxy_rig.deformation_proxy_decimation_ratio`
    uses. `blender_worker.voxel_remesh_canonicalize` is a thin bpy-calling
    wrapper around exactly these functions."""

    def test_initial_voxel_size_targets_120_voxels_across_the_longest_axis(self):
        from backend.app.voxel_remesh import initial_voxel_size

        self.assertAlmostEqual(initial_voxel_size(1.2), 0.01)
        self.assertAlmostEqual(initial_voxel_size(2.4, voxels_across_longest_axis=240.0), 0.01)

    def test_initial_voxel_size_rejects_non_positive_dimension(self):
        from backend.app.voxel_remesh import initial_voxel_size

        with self.assertRaises(ValueError):
            initial_voxel_size(0.0)
        with self.assertRaises(ValueError):
            initial_voxel_size(-1.0)

    def test_next_voxel_size_shrinks_when_under_the_floor(self):
        from backend.app.voxel_remesh import next_voxel_size

        size, low, high = next_voxel_size(0.02, raw_faces=1_000, raw_face_min=150_000, raw_face_max=350_000, low_voxel=None, high_voxel=None)

        self.assertLess(size, 0.02)
        self.assertIsNone(low)
        self.assertEqual(high, 0.02)

    def test_next_voxel_size_grows_when_over_the_ceiling(self):
        from backend.app.voxel_remesh import next_voxel_size

        size, low, high = next_voxel_size(0.02, raw_faces=900_000, raw_face_min=150_000, raw_face_max=350_000, low_voxel=None, high_voxel=None)

        self.assertGreater(size, 0.02)
        self.assertEqual(low, 0.02)
        self.assertIsNone(high)

    def test_next_voxel_size_converges_via_bisection_once_bounds_are_set(self):
        from backend.app.voxel_remesh import next_voxel_size

        # Both bounds set: the next guess must land strictly between them.
        size, low, high = next_voxel_size(0.03, raw_faces=900_000, raw_face_min=150_000, raw_face_max=350_000, low_voxel=0.01, high_voxel=0.05)

        self.assertEqual((low, high), (0.03, 0.05))
        self.assertAlmostEqual(size, 0.04)

    def test_voxel_candidate_preferred_over_the_floor_beats_under_the_floor(self):
        from backend.app.voxel_remesh import voxel_candidate_preferred

        self.assertTrue(voxel_candidate_preferred(160_000, 90_000, raw_face_min=150_000))
        self.assertFalse(voxel_candidate_preferred(90_000, 160_000, raw_face_min=150_000))

    def test_voxel_candidate_preferred_prefers_leaner_when_both_over(self):
        from backend.app.voxel_remesh import voxel_candidate_preferred

        self.assertTrue(voxel_candidate_preferred(160_000, 300_000, raw_face_min=150_000))
        self.assertFalse(voxel_candidate_preferred(300_000, 160_000, raw_face_min=150_000))

    def test_voxel_candidate_preferred_prefers_denser_when_both_under(self):
        from backend.app.voxel_remesh import voxel_candidate_preferred

        self.assertTrue(voxel_candidate_preferred(120_000, 50_000, raw_face_min=150_000))
        self.assertFalse(voxel_candidate_preferred(50_000, 120_000, raw_face_min=150_000))

    def test_canonicalize_decimate_ratio_targets_the_family_band_midpoint(self):
        from backend.app.voxel_remesh import canonicalize_decimate_ratio

        self.assertAlmostEqual(canonicalize_decimate_ratio(200_000, 30_000), 0.15)

    def test_canonicalize_decimate_ratio_never_exceeds_one(self):
        from backend.app.voxel_remesh import canonicalize_decimate_ratio

        self.assertEqual(canonicalize_decimate_ratio(10_000, 30_000), 1.0)

    def test_canonicalize_decimate_ratio_rejects_non_positive_inputs(self):
        from backend.app.voxel_remesh import canonicalize_decimate_ratio

        with self.assertRaises(ValueError):
            canonicalize_decimate_ratio(0, 30_000)
        with self.assertRaises(ValueError):
            canonicalize_decimate_ratio(30_000, 0)


class VoxelRawFaceBandScalingTests(unittest.TestCase):
    """`voxel_remesh.voxel_raw_face_band` (bead i63ky) -- the raw
    over-tessellation band the voxel size is bisected toward, which has to
    follow the operator's triangle target instead of staying pinned to the
    30k-tuned 150k-350k constants."""

    def test_the_standard_target_reproduces_the_hand_tuned_constants(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        self.assertEqual(voxel_raw_face_band(30_000), (150_000, 350_000))

    def test_native_default_constants_are_derived_from_that_band(self):
        # The constants are now derived, not written down twice; if the
        # multipliers ever drift, this is the test that says so.
        self.assertEqual(
            (native.DEFAULT_VOXEL_RAW_FACE_MIN, native.DEFAULT_VOXEL_RAW_FACE_MAX),
            (150_000, 350_000),
        )

    def test_the_band_scales_proportionally_with_the_target(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        self.assertEqual(voxel_raw_face_band(10_000), (50_000, 116_667))
        self.assertEqual(voxel_raw_face_band(80_000), (400_000, 933_333))

    def test_ultra_keeps_the_same_oversampling_standard_has(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        low, high = voxel_raw_face_band(150_000)

        # The bug this exists to prevent: an Ultra 150k target decimated down
        # from a 150k-350k raw mesh is barely a decimation at all.
        self.assertEqual((low, high), (750_000, 1_750_000))
        self.assertAlmostEqual(low / 150_000, 5.0)

    def test_the_floor_never_reaches_the_target_itself(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        low, _high = voxel_raw_face_band(400_000)

        # Ceiling-clamped, but still strictly denser than the budget -- a raw
        # mesh at or below the target makes the decimation step a no-op.
        self.assertGreater(low, 400_000)

    def test_the_absolute_ceiling_bounds_a_pathological_custom_target(self):
        from backend.app.voxel_remesh import VOXEL_RAW_FACE_CEILING, voxel_raw_face_band

        _low, high = voxel_raw_face_band(500_000)

        self.assertEqual(high, VOXEL_RAW_FACE_CEILING)

    def test_growth_escalates_the_band_on_a_retry(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        first = voxel_raw_face_band(30_000)
        second = voxel_raw_face_band(30_000, growth=1.5)

        self.assertEqual(second, (225_000, 525_000))
        self.assertGreater(second[0], first[0])

    def test_growth_still_honours_the_ceiling(self):
        from backend.app.voxel_remesh import VOXEL_RAW_FACE_CEILING, voxel_raw_face_band

        _low, high = voxel_raw_face_band(150_000, growth=1.5 * 1.5)

        self.assertEqual(high, VOXEL_RAW_FACE_CEILING)

    def test_non_positive_inputs_are_refused(self):
        from backend.app.voxel_remesh import voxel_raw_face_band

        with self.assertRaises(ValueError):
            voxel_raw_face_band(0)
        with self.assertRaises(ValueError):
            voxel_raw_face_band(30_000, growth=0)


if __name__ == "__main__":
    unittest.main()
