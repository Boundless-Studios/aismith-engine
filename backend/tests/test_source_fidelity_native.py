import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from backend.app import native
from backend.app.source_fidelity import subject_bbox, subject_mask_from_image

WIDTH, HEIGHT = 240, 360
HEAD_CENTER = (120, 55)
HEAD_RADIUS = 30
BODY_BOX = (90, 85, 150, 280)
ARM_BOX = (60, 110, 90, 190)
SKIN = (216, 178, 140, 255)
BLACK = (0, 0, 0, 255)

CAMERA_SPEC = {
    "kind": "perspective",
    "width": WIDTH,
    "height": HEIGHT,
    "center": [0.0, 0.0, 0.9],
    "distance": 2.5,
    "yaw_degrees": 0.0,
    "pitch_degrees": 0.0,
    "fov_x_degrees": 35.0,
    "ortho_scale": None,
}


def _character(body_color=BLACK, head_color=SKIN, arm_color=SKIN):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = HEAD_CENTER
    draw.ellipse([cx - HEAD_RADIUS, cy - HEAD_RADIUS, cx + HEAD_RADIUS, cy + HEAD_RADIUS], fill=head_color)
    draw.rectangle(list(BODY_BOX), fill=body_color)
    draw.rectangle(list(ARM_BOX), fill=arm_color)
    return img


def _fake_report(**overrides) -> dict:
    report = {
        "camera": CAMERA_SPEC,
        "mesh_bounds": {"low": [-0.3, -0.1, 0.0], "high": [0.3, 0.1, 1.8]},
        "mesh_count": 1,
        "device": "CPU",
        "samples": 32,
        "elapsed_seconds": 0.01,
    }
    report.update(overrides)
    return report


class MeasureSourceFidelityTests(unittest.TestCase):
    def test_measure_source_fidelity_perfect_render(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo_path = root / "photo.png"
            photo_img = _character()
            photo_img.save(photo_path)
            mesh_glb = root / "mesh.glb"
            mesh_glb.write_bytes(b"glTF")
            out_dir = root / "out"

            captured: list[dict] = []

            def capture(command, **_kwargs):
                spec = json.loads(command[-1])
                captured.append(spec)
                self.assertEqual(spec["op"], "render-source-view")
                photo_img.save(spec["output"])
                Path(spec["report"]).write_text(json.dumps(_fake_report()), encoding="utf-8")
                return "ok"

            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
            ):
                report = native.measure_source_fidelity(mesh_glb, photo_path, out_dir)

            self.assertEqual(report["silhouette_iou"], 1.0)
            self.assertGreaterEqual(report["masked_ssim"], 0.99)
            self.assertTrue((out_dir / "render.png").is_file())
            self.assertTrue((out_dir / "side_by_side.png").is_file())
            self.assertTrue((out_dir / "report.json").is_file())

            spec = captured[0]
            self.assertEqual(spec["framing"]["image_size"], [WIDTH, HEIGHT])
            mask, _ = subject_mask_from_image(np.array(photo_img))
            self.assertEqual(spec["framing"]["subject_bbox"], list(subject_bbox(mask)))

    def test_measure_source_fidelity_camera_override(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo_path = root / "photo.png"
            photo_img = _character()
            photo_img.save(photo_path)
            mesh_glb = root / "mesh.glb"
            mesh_glb.write_bytes(b"glTF")
            out_dir = root / "out"

            captured: list[dict] = []

            def capture(command, **_kwargs):
                spec = json.loads(command[-1])
                captured.append(spec)
                photo_img.save(spec["output"])
                Path(spec["report"]).write_text(json.dumps(_fake_report()), encoding="utf-8")
                return "ok"

            with (
                patch.object(native, "find_blender", return_value=Path("/tools/blender")),
                patch.object(native, "run_process", side_effect=capture),
            ):
                native.measure_source_fidelity(mesh_glb, photo_path, out_dir, camera=CAMERA_SPEC)

            self.assertEqual(captured[0]["framing"], {"camera": CAMERA_SPEC})

    def test_render_source_view_requires_blender(self):
        with patch.object(native, "find_blender", return_value=None):
            with self.assertRaises(RuntimeError):
                native.render_source_view(Path("mesh.glb"), Path("out.png"), {"camera": CAMERA_SPEC})


class ResolveSourcePhotoTests(unittest.TestCase):
    @staticmethod
    def _sidecar(root: Path, artifact_id: str, kind: str, *, source_id: str | None = None, file_name: str | None = None) -> None:
        payload: dict = {"id": artifact_id, "kind": kind, "metadata": {}}
        if file_name:
            image_path = root / file_name
            image_path.write_bytes(b"\x89PNG\r\n")
            payload["path"] = str(image_path)
        if source_id:
            payload["metadata"]["source_artifact_id"] = source_id
        (root / f"{artifact_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_resolve_source_photo_returns_root_upload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._sidecar(root, "upload", "image", file_name="upload.png")
            self._sidecar(root, "tpose", "image", source_id="upload", file_name="tpose.png")
            self._sidecar(root, "mesh", "glb", source_id="tpose")

            path, artifact_id, reason = native.resolve_source_photo("mesh", artifacts_dir=root)
            self.assertEqual(artifact_id, "upload")
            self.assertEqual(path, root / "upload.png")
            self.assertIn("upload", reason)

            palette_path, palette_id, _ = native.resolve_palette_reference("mesh", artifacts_dir=root)
            self.assertEqual(palette_id, "tpose")
            self.assertEqual(palette_path, root / "tpose.png")

    def test_resolve_source_photo_missing_and_cycle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifact_id, reason = native.resolve_source_photo("does-not-exist", artifacts_dir=root)
            self.assertIsNone(path)
            self.assertIsNone(artifact_id)
            self.assertTrue(reason)

            self._sidecar(root, "a", "glb", source_id="b")
            self._sidecar(root, "b", "glb", source_id="a")
            path, artifact_id, reason = native.resolve_source_photo("a", artifacts_dir=root)
            self.assertIsNone(path)
            self.assertIsNone(artifact_id)
            self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
