import contextlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from backend.workers import trellis2_fp8_worker as worker


class SparseConv3d:
    pass


class Linear:
    pass


class Trellis2Fp8WorkerTests(unittest.TestCase):
    def test_foreground_preparation_removes_opaque_background(self):
        from PIL import Image

        calls = []

        def remove_background(image):
            calls.append(image.mode)
            return Image.new("RGBA", image.size, (10, 20, 30, 0))

        prepared = worker._prepare_foreground(
            Image.new("RGB", (2, 2), "white"),
            remove_background,
        )

        self.assertEqual(calls, ["RGB"])
        self.assertEqual(prepared.getchannel("A").getextrema(), (0, 0))

    def test_foreground_preparation_preserves_existing_alpha(self):
        from PIL import Image

        image = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
        prepared = worker._prepare_foreground(
            image,
            lambda _image: self.fail("background removal was repeated"),
        )

        self.assertEqual(prepared.getchannel("A").getextrema(), (0, 0))

    def _fake_rembg(self, new_session):
        module = types.ModuleType("rembg")
        module.new_session = new_session
        module.remove = lambda image, session=None: image
        return module

    def test_matting_session_uses_configured_default_model(self):
        calls = []

        def new_session(name, **kwargs):
            calls.append((name, kwargs.get("providers")))
            return f"session:{name}"

        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISMITH_MATTING_MODEL", None)
                session = worker._create_matting_session()

        self.assertEqual([name for name, _ in calls], ["birefnet-general"])
        self.assertEqual(session, "session:birefnet-general")

    def test_matting_session_requests_cuda_before_cpu_provider(self):
        calls = []

        def new_session(name, **kwargs):
            calls.append((name, kwargs.get("providers")))
            return f"session:{name}"

        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISMITH_MATTING_MODEL", None)
                worker._create_matting_session()

        self.assertEqual(calls[0][1], ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_matting_session_logs_actual_device_from_inner_session(self):
        class FakeInnerSession:
            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        class FakeSession:
            def __init__(self, name):
                self.name = name
                self.inner_session = FakeInnerSession()

        def new_session(name, **kwargs):
            return FakeSession(name)

        buffer = io.StringIO()
        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISMITH_MATTING_MODEL", None)
                with contextlib.redirect_stdout(buffer):
                    worker._create_matting_session()

        log_output = buffer.getvalue()
        self.assertIn("device: cuda", log_output)

    def test_matting_session_logs_cpu_when_cuda_provider_unavailable(self):
        class FakeInnerSession:
            def get_providers(self):
                return ["CPUExecutionProvider"]

        class FakeSession:
            def __init__(self, name):
                self.name = name
                self.inner_session = FakeInnerSession()

        def new_session(name, **kwargs):
            return FakeSession(name)

        buffer = io.StringIO()
        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISMITH_MATTING_MODEL", None)
                with contextlib.redirect_stdout(buffer):
                    worker._create_matting_session()

        log_output = buffer.getvalue()
        self.assertIn("device: cpu", log_output)

    def test_matting_session_respects_env_override(self):
        calls = []

        def new_session(name, **kwargs):
            calls.append(name)
            return f"session:{name}"

        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {"AISMITH_MATTING_MODEL": "custom-model"}):
                session = worker._create_matting_session()

        self.assertEqual(calls, ["custom-model"])
        self.assertEqual(session, "session:custom-model")

    def test_matting_session_falls_back_to_lite_on_failure(self):
        calls = []

        def new_session(name, **kwargs):
            calls.append(name)
            if name == "birefnet-general":
                raise RuntimeError("model not downloaded")
            return f"session:{name}"

        buffer = io.StringIO()
        with patch.dict(sys.modules, {"rembg": self._fake_rembg(new_session)}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AISMITH_MATTING_MODEL", None)
                with contextlib.redirect_stdout(buffer):
                    session = worker._create_matting_session()

        self.assertEqual(calls, ["birefnet-general", "birefnet-general-lite"])
        self.assertEqual(session, "session:birefnet-general-lite")
        log_output = buffer.getvalue()
        self.assertIn("birefnet-general", log_output)
        self.assertIn("birefnet-general-lite", log_output)

    def test_harden_foreground_binarizes_and_frames_subject(self):
        from PIL import Image

        size = 40
        alpha = np.zeros((size, size), dtype=np.uint8)
        # Soft-alpha blob well above the hardening threshold.
        alpha[10:30, 10:30] = 200
        # Faint noise below the threshold that should be dropped entirely.
        alpha[2:5, 2:5] = 10

        image = Image.new("RGBA", (size, size), (255, 0, 0, 255))
        image.putalpha(Image.fromarray(alpha, mode="L"))

        result = worker.harden_foreground(image)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.width, result.height)

        result_alpha = np.asarray(result.getchannel("A"))
        center = result.height // 2, result.width // 2
        self.assertEqual(result_alpha[center], 255)
        # Corners are in the padding margin and must stay transparent.
        self.assertEqual(result_alpha[0, 0], 0)
        self.assertEqual(result_alpha[-1, -1], 0)

        opaque_rows = np.where(result_alpha.sum(axis=1) > 0)[0]
        opaque_cols = np.where(result_alpha.sum(axis=0) > 0)[0]
        top, bottom = int(opaque_rows.min()), int(opaque_rows.max())
        left, right = int(opaque_cols.min()), int(opaque_cols.max())
        subject_side = max(bottom - top + 1, right - left + 1)
        canvas_side = result.width
        pad_top = top
        pad_left = left
        # ~8% padding on each side of the (possibly blur-dilated) subject box.
        self.assertAlmostEqual(pad_top / canvas_side, 0.08, delta=0.04)
        self.assertAlmostEqual(pad_left / canvas_side, 0.08, delta=0.04)
        self.assertLess(subject_side, canvas_side)

    def test_prepare_foreground_frames_existing_alpha_without_matting(self):
        from PIL import Image

        size = 30
        alpha = np.zeros((size, size), dtype=np.uint8)
        alpha[5:25, 5:25] = 200
        image = Image.new("RGBA", (size, size), (0, 200, 0, 255))
        image.putalpha(Image.fromarray(alpha, mode="L"))

        prepared = worker._prepare_foreground(
            image,
            lambda _image: self.fail("background removal should be skipped"),
        )

        self.assertEqual(prepared.mode, "RGBA")
        self.assertEqual(prepared.width, prepared.height)
        prepared_alpha = np.asarray(prepared.getchannel("A"))
        self.assertGreater(prepared_alpha.max(), 0)
        self.assertEqual(prepared_alpha[0, 0], 0)

    def test_spconv_sparse_modules_use_float16_compute(self):
        self.assertEqual(
            worker._fp8_compute_dtype_name(SparseConv3d(), "spconv"),
            "float16",
        )
        self.assertEqual(
            worker._fp8_compute_dtype_name(Linear(), "spconv"),
            "bfloat16",
        )

    def test_refine_uses_complete_reduced_decoder(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class LoadMesh:
            def load(self, _path):
                return ("mesh",)

        class MeshRefiner:
            def process(self, *args):
                captured["use_tiled_decoder"] = args[20]
                captured["downsampling"] = args[15]
                raise ProbeComplete

        class Nodes:
            Trellis2LoadMesh = LoadMesh
            Trellis2MeshRefiner = MeshRefiner

        with self.assertRaises(ProbeComplete):
            worker._refine(
                Nodes,
                object(),
                object(),
                "input.glb",
                "output.glb",
                512,
                60_000,
                False,
            )

        self.assertFalse(captured["use_tiled_decoder"])
        self.assertEqual(captured["downsampling"], 32)

    def test_generate_preserves_sparse_occupancy_before_reconstruction(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class Generator:
            def process(self, *args):
                captured["seed"] = args[2]
                captured["fill_holes"] = args[13]
                captured["hole_fill_algorithm"] = args[15]
                captured["keep_only_shell"] = args[16]
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelGenerator = Generator

        with self.assertRaises(ProbeComplete):
            worker._generate(
                Nodes,
                object(),
                object(),
                "output.glb",
                512,
                generate_texture=False,
            )

        self.assertFalse(captured["fill_holes"])
        self.assertEqual(captured["hole_fill_algorithm"], "flood_fill")
        self.assertFalse(captured["keep_only_shell"])
        self.assertEqual(captured["seed"], 1337)

    def test_generate_uses_high_quality_reconstruction_chain(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class Generator:
            def process(self, *_args):
                return ("decoded", None)

        class CuFill:
            def process(self, mesh, perimeter):
                captured["cu_fill"] = (mesh, perimeter)
                return ("cu-filled",)

        class Reconstruct:
            def process(self, mesh, band, resolution, floaters, inner):
                captured["reconstruct"] = (
                    mesh,
                    band,
                    resolution,
                    floaters,
                    inner,
                )
                return ("reconstructed",)

        class MeshlibFill:
            def process(self, mesh):
                captured["meshlib_input"] = mesh
                return ("meshlib-filled", 1)

        class Convert:
            def process(self, mesh, _orientation):
                captured["converted"] = mesh
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelGenerator = Generator
            Trellis2FillHolesWithCuMesh = CuFill
            Trellis2ReconstructMeshWithQuad = Reconstruct
            Trellis2FillHolesNicelyWithMeshlib = MeshlibFill
            Trellis2MeshWithVoxelToTrimesh = Convert

        with self.assertRaises(ProbeComplete):
            worker._generate(
                Nodes,
                object(),
                object(),
                "output.glb",
                512,
                generate_texture=False,
            )

        self.assertEqual(captured.get("cu_fill"), ("decoded", 1.0))
        self.assertEqual(
            captured["reconstruct"],
            ("cu-filled", 1.0, 512, True, True),
        )
        self.assertEqual(captured["meshlib_input"], "reconstructed")
        self.assertEqual(captured["converted"], "meshlib-filled")

    def test_textured_generation_removes_detached_components_before_export(self):
        captured = {}

        class Generator:
            def process(self, *_args):
                return ("decoded", None)

        class TexturedExport:
            def process(self, mesh, texture_size, atlas_size, face_budget):
                captured["texture_export"] = (mesh, texture_size, atlas_size, face_budget)
                return ("textured-with-crumbs",)

        class Nodes:
            Trellis2MeshWithVoxelGenerator = Generator
            Trellis2OvoxelExportToGLB = TexturedExport

        with (
            patch.object(worker, "_retain_dominant_component", return_value="clean-textured") as retain,
            patch.object(worker, "_validate_character_mesh") as validate,
            patch.object(worker, "_export") as export,
        ):
            worker._generate(
                Nodes,
                object(),
                object(),
                "output.glb",
                512,
                generate_texture=True,
            )

        self.assertEqual(captured["texture_export"], ("decoded", 512, 2048, 250000))
        retain.assert_called_once_with("textured-with-crumbs")
        validate.assert_called_once_with("clean-textured")
        export.assert_called_once_with("clean-textured", "output.glb")

    def test_character_cleanup_retains_only_valid_dominant_shell(self):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [3, 0, 0],
                    [4, 0, 0],
                    [3, 1, 0],
                ],
                dtype=float,
            ),
            faces=np.asarray(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [0, 3, 2],
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            process=False,
        )

        cleaned = worker._retain_dominant_component(mesh)

        self.assertEqual(len(cleaned.faces), 4)
        self.assertEqual(len(cleaned.vertices), 4)
        worker._validate_character_mesh(cleaned)

    def test_character_cleanup_preserves_textured_vertex_uvs(self):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [3, 0, 0], [4, 0, 0], [3, 1, 0],
            ], dtype=float),
            faces=np.asarray([
                [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6],
            ]),
            visual=trimesh.visual.TextureVisuals(
                uv=np.asarray([
                    [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1],
                ], dtype=float),
                material=trimesh.visual.material.SimpleMaterial(name="Mary texture"),
            ),
            process=False,
        )

        cleaned = worker._retain_dominant_component(mesh)

        self.assertEqual(len(cleaned.visual.uv), len(cleaned.vertices))
        self.assertEqual(cleaned.visual.material.name, "Mary texture")

    def test_character_cleanup_does_not_split_body_at_uv_seams(self):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray([
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [3, 0, 0], [4, 0, 0], [3, 1, 0],
            ], dtype=float),
            faces=np.asarray([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            visual=trimesh.visual.TextureVisuals(
                uv=np.asarray([
                    [0, 0], [0.5, 0], [0, 1],
                    [0.5, 0], [1, 1], [0, 1],
                    [0, 0], [1, 0], [0, 1],
                ], dtype=float),
            ),
            process=False,
        )

        cleaned = worker._retain_dominant_component(mesh)

        self.assertEqual(len(cleaned.faces), 2)
        self.assertEqual(len(cleaned.visual.uv), len(cleaned.vertices))

    def test_character_validation_allows_bounded_open_boundaries(self):
        import numpy as np
        import trimesh

        size = 10
        vertices = np.array(
            [
                (x, y, 0.01 * x * y)
                for y in range(size)
                for x in range(size)
            ],
            dtype=float,
        )
        faces = []
        for y in range(size - 1):
            for x in range(size - 1):
                lower_left = y * size + x
                faces.extend(
                    (
                        (lower_left, lower_left + 1, lower_left + size),
                        (lower_left + 1, lower_left + size + 1, lower_left + size),
                    )
                )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        worker._validate_character_mesh(mesh)

    def test_character_validation_treats_many_open_boundaries_as_diagnostic(self):
        import numpy as np
        import trimesh

        boundary_count = 536
        angles = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False)
        vertices = np.vstack((
            np.array([[0.0, 0.0, 0.1]]),
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros(boundary_count))),
        ))
        faces = np.asarray([
            (0, index + 1, ((index + 1) % boundary_count) + 1)
            for index in range(boundary_count)
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        worker._validate_character_mesh(mesh)

    def test_character_validation_treats_nonmanifold_edges_as_diagnostic(self):
        import trimesh

        vertices = np.asarray([
            [0, 0, 0], [1, 0, 0],
            *[[0.5, index + 1, 0.1 * index] for index in range(18)],
        ], dtype=float)
        faces = np.asarray([(0, 1, index + 2) for index in range(18)])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        worker._validate_character_mesh(mesh)

    def test_geometry_export_includes_smooth_vertex_normals(self):
        import json
        import struct

        import trimesh

        mesh = trimesh.creation.box()
        worker._ensure_smooth_vertex_normals(mesh)
        exported = mesh.export(file_type="glb")
        json_length, _ = struct.unpack("<II", exported[12:20])
        document = json.loads(exported[20:20 + json_length].decode("utf-8"))

        attributes = document["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("NORMAL", attributes)

    def test_simplification_requests_nonmanifold_repair_without_relaxing_gate(self):
        import trimesh

        calls = []

        class Mesh:
            def __init__(self, vertex_matrix, face_matrix):
                self._vertices = vertex_matrix
                self._faces = face_matrix

            def vertex_matrix(self):
                return self._vertices

            def face_matrix(self):
                return self._faces

            def face_number(self):
                return len(self._faces)

        class MeshSet:
            def add_mesh(self, mesh):
                self.mesh = mesh

            def current_mesh(self):
                return self.mesh

            def meshing_repair_non_manifold_edges(self, **kwargs):
                calls.append(("edges", kwargs))

            def meshing_repair_non_manifold_vertices(self, **kwargs):
                calls.append(("vertices", kwargs))

            def meshing_remove_unreferenced_vertices(self):
                calls.append(("unreferenced", {}))

            def meshing_close_holes(self, **kwargs):
                calls.append(("close_holes", kwargs))

        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [0, -1, 0], [0, 0, -1],
                ],
                dtype=float,
            ),
            faces=np.asarray(
                [
                    [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3],
                    [0, 1, 4], [0, 5, 1], [0, 4, 5], [1, 5, 4],
                ]
            ),
            process=False,
        )

        with patch.dict(
            sys.modules,
            {"pymeshlab": SimpleNamespace(Mesh=Mesh, MeshSet=MeshSet)},
        ):
            worker._simplify_mesh(mesh, target_faces=8)

        self.assertEqual(
            calls,
            [
                ("edges", {"method": "Remove Faces"}),
                ("vertices", {"vertdispratio": 0.0}),
                ("unreferenced", {}),
                (
                    "close_holes",
                    {
                        "maxholesize": 100,
                        "newfaceselected": False,
                        "selfintersection": False,
                    },
                ),
                ("edges", {"method": "Remove Faces"}),
                ("vertices", {"vertdispratio": 0.0}),
                ("unreferenced", {}),
                (
                    "close_holes",
                    {
                        "maxholesize": 100,
                        "newfaceselected": False,
                        "selfintersection": False,
                    },
                ),
            ],
        )

    def test_native_contract_accepts_the_current_gpu_architecture(self):
        with TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "native-contract.json").write_text(
                json.dumps({"cuda_architecture": "8.9"})
            )

            worker._validate_native_contract(runtime, (8, 9))

    def test_native_contract_rejects_extensions_for_another_gpu(self):
        with TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "native-contract.json").write_text(
                json.dumps({"cuda_architecture": "8.6"})
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "built for CUDA architecture 8.6.*current GPU requires 8.9",
            ):
                worker._validate_native_contract(runtime, (8, 9))

    def test_safe_reconstruction_preserves_decoded_mesh(self):
        captured = {}

        class LoadMesh:
            def load(self, _path):
                return ("source",)

        class MeshRefiner:
            def process(self, *args):
                decoded = SimpleNamespace(
                    vertices=np.eye(3),
                    faces=np.asarray([[0, 1, 2]]),
                )
                captured["decoded"] = decoded
                return (decoded, None)

        class Reconstruct:
            def process(self, *args):
                raise AssertionError("redundant reconstruction selected")

        class Simplify:
            def process(self, mesh, *_args):
                captured["simplified"] = mesh
                captured["simplifier"] = _args[-1]
                return ("simplified",)

        class Mesh:
            vertices = np.eye(3)
            faces = np.asarray([[0, 1, 2]])

        class Convert:
            def process(self, mesh, *_args):
                captured["converted"] = mesh
                return (Mesh(),)

        class Nodes:
            Trellis2LoadMesh = LoadMesh
            Trellis2MeshRefiner = MeshRefiner
            Trellis2ReconstructMesh = Reconstruct
            Trellis2ReconstructMeshWithQuad = Reconstruct
            Trellis2SimplifyMesh = Simplify
            Trellis2MeshWithVoxelToTrimesh = Convert

        with (
            patch.object(worker, "_texture", return_value="textured"),
            patch.object(worker, "_export"),
        ):
            worker._refine(
                Nodes,
                object(),
                object(),
                "input.glb",
                "output.glb",
                512,
                60_000,
                False,
                safe_reconstruction=True,
            )

        self.assertIs(captured["converted"], captured["decoded"])

    def test_generate_multiview_conditions_on_every_provided_view(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class MultiViewGenerator:
            def process(self, **kwargs):
                captured.update(kwargs)
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelMultiViewGenerator = MultiViewGenerator

        view_images = {"front": "front-tensor", "left": "left-tensor"}

        with self.assertRaises(ProbeComplete):
            worker._generate_multiview(
                Nodes,
                object(),
                view_images,
                "output.glb",
                1024,
                generate_texture=False,
            )

        self.assertEqual(captured["front_image"], "front-tensor")
        self.assertEqual(captured["left_image"], "left-tensor")
        self.assertIsNone(captured["back_image"])
        self.assertIsNone(captured["right_image"])
        self.assertEqual(captured["pipeline_type"], "1024_cascade")
        self.assertEqual(captured["front_axis"], "z")
        # Node widget default; see the comment in _generate_multiview.
        self.assertEqual(captured["blend_temperature"], 1.0)
        self.assertFalse(captured["generate_texture_slat"])

    @staticmethod
    def _solid_blob(width, height, canvas_size=100):
        from PIL import Image

        image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        ox, oy = (canvas_size - width) // 2, (canvas_size - height) // 2
        for x in range(width):
            for y in range(height):
                image.putpixel((ox + x, oy + y), (200, 50, 50, 255))
        return image

    def test_multiview_mattes_every_view_when_any_view_lacks_real_alpha(self):
        from PIL import Image

        matted = []

        def remove_background(image):
            matted.append(image.mode)
            return self._solid_blob(4, 4)

        # "front" is fully opaque (no real alpha, e.g. a raw upload); "back"
        # already carries real alpha (e.g. a prepared artifact). Because at
        # least one view lacks real alpha, BOTH must be matted identically --
        # a view is never allowed to skip matting just because it happens to
        # already have an alpha channel.
        opaque_front = Image.new("RGB", (20, 20), "white")
        transparent_back = self._solid_blob(6, 6)

        framed = worker._prepare_multiview_foregrounds(
            {"front": opaque_front, "back": transparent_back}, remove_background
        )

        self.assertEqual(len(matted), 2)
        self.assertEqual(set(framed.keys()), {"front", "back"})

    def test_multiview_skips_matting_when_every_view_already_has_alpha(self):
        def remove_background(_image):
            self.fail("matting must not run when every view already has real alpha")

        framed = worker._prepare_multiview_foregrounds(
            {"front": self._solid_blob(20, 20), "back": self._solid_blob(20, 20)},
            remove_background,
        )

        self.assertEqual(set(framed.keys()), {"front", "back"})

    def test_multiview_scales_every_view_to_one_shared_subject_size(self):
        import numpy as np

        def remove_background(_image):
            self.fail("both views already carry real alpha; matting must not run")

        large = self._solid_blob(40, 40)
        small = self._solid_blob(20, 20)

        framed = worker._prepare_multiview_foregrounds(
            {"front": large, "back": small}, remove_background
        )

        # Identical output pixel dimensions for every view.
        self.assertEqual(framed["front"].size, framed["back"].size)

        def subject_extent(image):
            alpha = np.asarray(image.getchannel("A"))
            rows = np.where(alpha.sum(axis=1) > 0)[0]
            cols = np.where(alpha.sum(axis=0) > 0)[0]
            return (int(cols.max() - cols.min()) + 1, int(rows.max() - rows.min()) + 1)

        front_extent = subject_extent(framed["front"])
        back_extent = subject_extent(framed["back"])

        # Equal output size and equal apparent subject scale, even though the
        # two source bboxes (after the same hardening pass) are not the same
        # size to begin with.
        self.assertAlmostEqual(front_extent[0], back_extent[0], delta=1)
        self.assertAlmostEqual(front_extent[1], back_extent[1], delta=1)

        def raw_hardened_side(image):
            mask = worker._harden_alpha(image.getchannel("A"))
            bbox = mask.getbbox()
            return max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        front_raw_side = raw_hardened_side(large)
        back_raw_side = raw_hardened_side(small)
        self.assertGreater(front_raw_side, back_raw_side)  # sanity: inputs really differ

        # front already sits at the shared side length, so it is ~unscaled;
        # back's bbox is smaller, so it must be scaled up noticeably to reach
        # the same final extent as front (resize/blur introduce a few percent
        # of measurement noise, so this checks direction and magnitude rather
        # than an exact ratio).
        self.assertAlmostEqual(front_extent[0] / front_raw_side, 1.0, delta=0.1)
        self.assertGreater(back_extent[0] / back_raw_side, 1.3)

    def test_multiview_warns_when_bbox_heights_diverge(self):
        buffer = io.StringIO()

        def remove_background(_image):
            self.fail("both views already carry real alpha; matting must not run")

        with contextlib.redirect_stdout(buffer):
            worker._prepare_multiview_foregrounds(
                {"front": self._solid_blob(20, 60), "back": self._solid_blob(20, 20)},
                remove_background,
            )

        log_output = buffer.getvalue()
        self.assertIn("Multi-view crops:", log_output)
        self.assertIn("WARNING", log_output)
        self.assertIn("bbox heights differ", log_output)

    def test_main_routes_multiview_spec_to_multiview_node(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class MultiViewGenerator:
            def process(self, **kwargs):
                captured.update(kwargs)
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelMultiViewGenerator = MultiViewGenerator

        spec = json.dumps({
            "operation": "generate",
            "runtime": "runtime-dir",
            "output_glb": "output.glb",
            "resolution": 1024,
            "view_images": {
                "front": "front.png",
                "left": "left.png",
                "right": "right.png",
            },
        })

        def fake_multiview_tensors(_nodes, view_paths):
            return {label: f"tensor:{path.name}" for label, path in view_paths.items()}

        with (
            patch.object(sys, "argv", ["prog", spec]),
            patch.object(worker, "_validate_native_contract"),
            patch.object(worker, "_load_nodes", return_value=Nodes),
            patch.object(worker, "_load_pipeline", return_value=object()),
            patch.object(worker, "_multiview_tensors", side_effect=fake_multiview_tensors),
        ):
            with self.assertRaises(ProbeComplete):
                worker.main()

        self.assertEqual(captured["front_image"], "tensor:front.png")
        self.assertEqual(captured["left_image"], "tensor:left.png")
        self.assertEqual(captured["right_image"], "tensor:right.png")
        self.assertIsNone(captured["back_image"])

    def test_main_raises_clear_error_when_wrapper_lacks_multiview_node(self):
        # Preflight check: a stale/older wrapper install without the
        # multi-view node must fail before any tensors are built, with a
        # message pointing at the fix (reinstall/upgrade the runtime).
        class Nodes:
            pass  # deliberately missing Trellis2MeshWithVoxelMultiViewGenerator

        spec = json.dumps({
            "operation": "generate",
            "runtime": "runtime-dir",
            "output_glb": "output.glb",
            "resolution": 1024,
            "view_images": {"front": "front.png", "back": "back.png"},
        })

        def fail_if_called(*_args, **_kwargs):
            self.fail("tensors must not be built before the multiview preflight check")

        with (
            patch.object(sys, "argv", ["prog", spec]),
            patch.object(worker, "_validate_native_contract"),
            patch.object(worker, "_load_nodes", return_value=Nodes),
            patch.object(worker, "_load_pipeline", return_value=object()),
            patch.object(worker, "_multiview_tensors", side_effect=fail_if_called),
        ):
            with self.assertRaisesRegex(RuntimeError, "multi-view conditioning requires"):
                worker.main()

    def test_main_routes_single_view_spec_to_legacy_node(self):
        captured = {}

        class ProbeComplete(Exception):
            pass

        class Generator:
            def process(self, *args):
                captured["seed"] = args[2]
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelGenerator = Generator

        spec = json.dumps({
            "operation": "generate",
            "runtime": "runtime-dir",
            "output_glb": "output.glb",
            "resolution": 1024,
            "source_image": "front.png",
            "seed": 4242,
        })

        def fake_image_tensor(_nodes, path):
            return f"tensor:{Path(path).name}"

        with (
            patch.object(sys, "argv", ["prog", spec]),
            patch.object(worker, "_validate_native_contract"),
            patch.object(worker, "_load_nodes", return_value=Nodes),
            patch.object(worker, "_load_pipeline", return_value=object()),
            patch.object(worker, "_image_tensor", side_effect=fake_image_tensor),
        ):
            with self.assertRaises(ProbeComplete):
                worker.main()

        self.assertEqual(captured["seed"], 4242)

    def test_main_routes_single_entry_view_images_to_legacy_node(self):
        # A view_images spec with only "front" must not be treated as
        # multi-view: it should still hit the byte-for-byte legacy path.
        captured = {}

        class ProbeComplete(Exception):
            pass

        class Generator:
            def process(self, *args):
                captured["called"] = True
                raise ProbeComplete

        class Nodes:
            Trellis2MeshWithVoxelGenerator = Generator

        spec = json.dumps({
            "operation": "generate",
            "runtime": "runtime-dir",
            "output_glb": "output.glb",
            "resolution": 1024,
            "source_image": "front.png",
            "view_images": {"front": "front.png"},
        })

        def fake_image_tensor(_nodes, path):
            return f"tensor:{Path(path).name}"

        with (
            patch.object(sys, "argv", ["prog", spec]),
            patch.object(worker, "_validate_native_contract"),
            patch.object(worker, "_load_nodes", return_value=Nodes),
            patch.object(worker, "_load_pipeline", return_value=object()),
            patch.object(worker, "_image_tensor", side_effect=fake_image_tensor),
        ):
            with self.assertRaises(ProbeComplete):
                worker.main()

        self.assertTrue(captured.get("called"))


if __name__ == "__main__":
    unittest.main()
