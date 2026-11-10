import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import trimesh
from PIL import Image

from backend.app import native

GREY = [160, 160, 160, 255]


def _build_fixture(directory: Path) -> Path:
    """Box (torso) + off-centre sphere (head), asymmetric on purpose so a
    mirrored render scores low. glTF is Y-up; Blender's importer converts to
    Z-up, so the box's +Y height (1.6) stands along Blender's +Z."""
    box = trimesh.creation.box(extents=(0.4, 1.6, 0.2))
    box.visual.vertex_colors = GREY
    sphere = trimesh.creation.uv_sphere(radius=0.25)
    sphere.apply_translation((0.3, 1.05, 0.0))
    sphere.visual.vertex_colors = GREY
    path = directory / "fixture.glb"
    trimesh.Scene([box, sphere]).export(path, file_type="glb")
    return path


class SourceFidelityBlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if native.find_blender() is None:
            raise unittest.SkipTest("Blender not found (BLENDER_EXE / PATH)")

    def _self_render_matches(self, fov_x_degrees):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            glb = _build_fixture(root)
            photo_png = root / "photo.png"
            framing = {
                "image_size": [300, 400],
                "subject_bbox": [60, 40, 240, 360],
                "fov_x_degrees": fov_x_degrees,
                "yaw_degrees": 20.0,
                "pitch_degrees": 10.0,
            }
            native.render_source_view(glb, photo_png, framing, samples=16)

            out_dir = root / "out"
            report = native.measure_source_fidelity(
                glb,
                photo_png,
                out_dir,
                fov_x_degrees=fov_x_degrees,
                yaw_degrees=20.0,
                pitch_degrees=10.0,
                samples=16,
            )

            self.assertGreaterEqual(report["silhouette_iou"], 0.97)
            self.assertGreaterEqual(report["masked_ssim"], 0.95)
            self.assertLessEqual(abs(report["alignment"]["scale"] - 1), 0.03)

            requested_height = framing["subject_bbox"][3] - framing["subject_bbox"][1]
            x0, y0, x1, y1 = report["subject_bbox"]
            self.assertLessEqual(abs((y1 - y0) - requested_height), 6)
            return report, photo_png

    def test_self_render_is_a_perfect_match_perspective(self):
        self._self_render_matches(35.0)

    def test_self_render_is_a_perfect_match_ortho(self):
        self._self_render_matches(None)

    def test_mirrored_photo_scores_low(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            glb = _build_fixture(root)
            photo_png = root / "photo.png"
            framing = {
                "image_size": [300, 400],
                "subject_bbox": [60, 40, 240, 360],
                "fov_x_degrees": 35.0,
                "yaw_degrees": 20.0,
                "pitch_degrees": 10.0,
            }
            native.render_source_view(glb, photo_png, framing, samples=16)

            mirrored_png = root / "photo_mirrored.png"
            mirrored = Image.open(photo_png).convert("RGBA")
            mirrored = mirrored.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored.save(mirrored_png)

            out_dir = root / "out"
            report = native.measure_source_fidelity(
                glb,
                mirrored_png,
                out_dir,
                fov_x_degrees=35.0,
                yaw_degrees=20.0,
                pitch_degrees=10.0,
                samples=16,
            )
            self.assertLess(report["silhouette_iou"], 0.9)


if __name__ == "__main__":
    unittest.main()
