import math
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from backend.app import source_fidelity as sf

# The geometry metrics under test here are independent of the learned
# identity backbones, so pin those off: otherwise every compute call in this
# file would pay a torch import, and the numbers would depend on which weights
# happen to be cached on the box. test_source_fidelity_identity.py owns them.
_REAL_IDENTITY_MODELS = None


def setUpModule():
    global _REAL_IDENTITY_MODELS
    _REAL_IDENTITY_MODELS = sf._IDENTITY_MODELS
    sf._IDENTITY_MODELS = sf.IdentityModels(warnings=("identity metrics disabled for this test module",))


def tearDownModule():
    sf._IDENTITY_MODELS = _REAL_IDENTITY_MODELS


WIDTH, HEIGHT = 240, 360
HEAD_CENTER = (120, 55)
HEAD_RADIUS = 30
BODY_BOX = (90, 85, 150, 280)
ARM_BOX = (60, 110, 90, 190)
SKIN = (216, 178, 140, 255)
BLACK = (0, 0, 0, 255)


def _character(body_color=BLACK, head_color=SKIN, arm_color=SKIN):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = HEAD_CENTER
    draw.ellipse([cx - HEAD_RADIUS, cy - HEAD_RADIUS, cx + HEAD_RADIUS, cy + HEAD_RADIUS], fill=head_color)
    draw.rectangle(list(BODY_BOX), fill=body_color)
    draw.rectangle(list(ARM_BOX), fill=arm_color)
    return np.array(img)


def _textured_character():
    """`_character()` is flat-filled -- zero high-frequency energy anywhere,
    which makes a detail RATIO undefined (0/0). Detail-ratio tests need a
    photo that actually carries texture, so this overlays fixed-seed per-
    pixel noise on the opaque pixels only."""
    photo = _character().astype(np.int16)
    rng = np.random.default_rng(1)
    noise = rng.integers(-40, 40, size=photo.shape[:2], endpoint=True)
    opaque = photo[..., 3] > 0
    for channel in range(3):
        photo[..., channel] = np.where(opaque, np.clip(photo[..., channel] + noise, 0, 255), photo[..., channel])
    return photo.astype(np.uint8)


def _composite_over_color(rgba: np.ndarray, color) -> np.ndarray:
    arr = rgba.astype(np.float64) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    color_arr = np.array(color, dtype=np.float64).reshape(1, 1, 3)
    return rgb * alpha + color_arr * (1 - alpha)


class ComputeSourceFidelityTests(unittest.TestCase):
    def test_identical_render_scores_perfect(self):
        photo = _character()
        report = sf.compute_source_fidelity(photo, photo)
        self.assertEqual(report["policy_version"], "source-fidelity-v1")
        self.assertEqual(report["mask_source"], "alpha")
        self.assertEqual(report["silhouette_iou"], 1.0)
        self.assertGreaterEqual(report["masked_ssim"], 0.99)
        self.assertGreaterEqual(report["face_ssim"], 0.99)
        self.assertLessEqual(report["perceptual_distance"], 0.01)
        self.assertLess(abs(report["masked_luma_delta"]), 1e-3)
        self.assertGreaterEqual(report["face_pixels"], sf.MIN_FACE_PIXELS)

    def test_mirrored_render_scores_low(self):
        photo = _character()
        identical = sf.compute_source_fidelity(photo, photo)
        mirrored = sf.compute_source_fidelity(photo, np.fliplr(photo))
        self.assertLess(mirrored["silhouette_iou"], 0.9)
        self.assertLessEqual(mirrored["masked_ssim"], identical["masked_ssim"] - 0.15)

    def test_alignment_removes_scale_and_offset(self):
        photo = _character()
        photo_img = Image.fromarray(photo, mode="RGBA")
        shrunk = photo_img.resize((round(WIDTH * 0.8), round(HEIGHT * 0.8)), Image.BILINEAR)
        canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        canvas.paste(shrunk, (15, 10), shrunk)
        render = np.array(canvas)

        report = sf.compute_source_fidelity(photo, render)
        self.assertGreaterEqual(report["silhouette_iou"], 0.95)
        self.assertGreaterEqual(report["masked_ssim"], 0.9)
        self.assertLess(abs(report["alignment"]["scale"] - 1.25), 0.05)
        self.assertLess(abs(report["alignment"]["bbox_width_ratio"] - 1.0), 0.05)

    def test_grey_body_is_a_luminance_error(self):
        photo = _character()
        render = _character(body_color=(128, 128, 128, 255))
        identical = sf.compute_source_fidelity(photo, photo)
        report = sf.compute_source_fidelity(photo, render)
        self.assertEqual(report["silhouette_iou"], 1.0)
        self.assertGreater(report["masked_luma_delta"], 0.15)
        self.assertGreater(report["masked_rgb_l1"], 0.15)
        self.assertLess(report["masked_ssim"], identical["masked_ssim"])

    def test_face_band_isolates_the_head(self):
        photo = _character()
        render = _character(head_color=(20, 20, 20, 255))
        report = sf.compute_source_fidelity(photo, render)
        self.assertLess(report["face_ssim"], 0.8)
        self.assertGreater(report["masked_ssim"], report["face_ssim"])

    def test_border_colour_mask_matches_alpha(self):
        photo = _character()
        alpha_mask, source = sf.subject_mask_from_image(photo)
        self.assertEqual(source, "alpha")

        flattened = _composite_over_color(photo, (0.0, 0.6, 0.0))
        border_mask, border_source = sf.subject_mask_from_image(flattened)
        self.assertEqual(border_source, "border-colour")

        iou = sf.silhouette_iou(alpha_mask, border_mask)
        self.assertGreaterEqual(iou, 0.97)


class DetailEnergyTests(unittest.TestCase):
    def test_sharp_beats_its_own_blurred_copy(self):
        size = 64
        checker = (np.indices((size, size)).sum(axis=0) % 2).astype(np.float64)
        mask = np.ones((size, size), dtype=bool)
        blurred = sf.gaussian_blur(checker, sigma=2.0)
        self.assertGreater(sf.detail_energy(checker, mask), sf.detail_energy(blurred, mask))

    def test_constant_image_has_zero_energy(self):
        flat = np.full((40, 40), 0.5)
        mask = np.ones((40, 40), dtype=bool)
        self.assertEqual(sf.detail_energy(flat, mask), 0.0)

    def test_empty_mask_is_none(self):
        flat = np.full((40, 40), 0.5)
        mask = np.zeros((40, 40), dtype=bool)
        self.assertIsNone(sf.detail_energy(flat, mask))


class HeadBboxTests(unittest.TestCase):
    def test_excludes_an_arm_bar_that_enters_the_band_from_below(self):
        height, width = 300, 200
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - 100) ** 2 + (yy - 40) ** 2 <= 25**2
        mask = mask.copy()
        mask[65:250, 80:120] = True  # body, below the band
        mask[45:55, 20:180] = True  # arm bar: inside the band, but below its top 40%
        band = sf.face_band_bbox(sf.subject_bbox(mask))
        self.assertEqual(band, (20, 15, 180, 62))

        box = sf.head_bbox(mask)
        self.assertGreater(box[0], band[0] + 20)
        self.assertLess(box[2], band[2] - 20)

    def test_matches_face_band_bbox_on_the_existing_character(self):
        # This fixture's arm sits entirely below the band (y=110, band ends
        # at y=76), so face_band_bbox's x0 is dead space inherited from the
        # arm's declared bbox, not real arm pixels -- head_bbox excludes it
        # and is a TIGHTER box, not merely a "few pixels" approximation of
        # face_band_bbox on this particular fixture.
        photo = _character()
        mask, _ = sf.subject_mask_from_image(photo)
        band = sf.face_band_bbox(sf.subject_bbox(mask))
        box = sf.head_bbox(mask)
        self.assertLessEqual(abs(box[1] - band[1]), 3)
        self.assertLessEqual(abs(box[2] - band[2]), 3)
        self.assertLessEqual(abs(box[3] - band[3]), 3)
        self.assertGreater(box[0], band[0] + 20)


class DetailRatioTests(unittest.TestCase):
    def test_identical_render_has_unit_detail_ratios(self):
        photo = _textured_character()
        report = sf.compute_source_fidelity(photo, photo)
        self.assertAlmostEqual(report["detail_ratio"], 1.0, places=6)
        self.assertAlmostEqual(report["face_detail_ratio"], 1.0, places=6)

    def test_blurred_render_has_a_low_detail_ratio_but_a_perfect_silhouette(self):
        photo = _textured_character()
        rgb_blurred = Image.fromarray(photo, mode="RGBA").convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
        render = np.dstack([np.array(rgb_blurred), photo[..., 3]])  # blur texture only, keep the silhouette exact

        report = sf.compute_source_fidelity(photo, render)
        self.assertEqual(report["silhouette_iou"], 1.0)
        self.assertLess(report["detail_ratio"], 0.7)


class FitSourceCameraTests(unittest.TestCase):
    def test_fit_source_camera_perspective_and_ortho(self):
        mesh_low = (-0.4, -0.2, 0.0)
        mesh_high = (0.4, 0.2, 2.0)
        image_size = (400, 800)
        subject_bbox = (100, 80, 300, 720)

        half_fov = math.radians(40.0) / 2.0
        expected_distance = max(
            2.0 / 0.8 / (2 * math.tan(half_fov) * 2.0),
            0.8 / 0.5 / (2 * math.tan(half_fov)),
        )
        perspective = sf.fit_source_camera(mesh_low, mesh_high, image_size, subject_bbox, fov_x_degrees=40.0)
        self.assertEqual(perspective.kind, "perspective")
        self.assertAlmostEqual(perspective.distance, expected_distance, places=6)
        self.assertEqual(perspective.center, (0.0, 0.0, 1.0))
        self.assertIsNone(perspective.ortho_scale)

        ortho = sf.fit_source_camera(mesh_low, mesh_high, image_size, subject_bbox)
        self.assertEqual(ortho.kind, "orthographic")
        expected_ortho_scale = max(2.0 / 0.8 * 0.5, 0.8 / 0.5)
        self.assertAlmostEqual(ortho.ortho_scale, expected_ortho_scale, places=6)
        self.assertIsNone(ortho.fov_x_degrees)

        spec = perspective.to_spec()
        self.assertEqual(sf.SourceCamera.from_spec(spec), perspective)

        yaw90 = sf.SourceCamera(
            kind="orthographic", width=10, height=10, center=(0, 0, 0), distance=1.0,
            yaw_degrees=90.0, pitch_degrees=0.0,
        )
        np.testing.assert_allclose(yaw90.direction(), (1.0, 0.0, 0.0), atol=1e-9)

        pitch30 = sf.SourceCamera(
            kind="orthographic", width=10, height=10, center=(0, 0, 0), distance=1.0,
            yaw_degrees=0.0, pitch_degrees=30.0,
        )
        self.assertAlmostEqual(pitch30.direction()[2], math.sin(math.radians(30.0)), places=9)


class SsimAndSilhouetteBasicsTests(unittest.TestCase):
    def test_ssim_map_basics(self):
        rng = np.random.default_rng(0)
        identical = rng.random((64, 64))
        ssim_identical = sf.ssim_map(identical, identical)
        np.testing.assert_allclose(ssim_identical, 1.0, atol=1e-6)

        flat = np.full((64, 64), 0.5)
        noise = rng.random((64, 64))
        ssim_noisy = sf.ssim_map(flat, noise)
        self.assertLess(float(np.mean(ssim_noisy)), 0.2)

        mask_a = np.zeros((20, 20), dtype=bool)
        mask_a[:10, :] = True
        mask_b = np.zeros((20, 20), dtype=bool)
        mask_b[10:, :] = True
        self.assertEqual(sf.silhouette_iou(mask_a, mask_b), 0.0)
        self.assertEqual(sf.silhouette_iou(np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)), 0.0)


class SideBySideTests(unittest.TestCase):
    def test_side_by_side_layout(self):
        photo = _character()
        mask, _ = sf.subject_mask_from_image(photo)
        face_bbox = sf.face_band_bbox(sf.subject_bbox(mask))
        composite = sf.side_by_side(photo, photo, face_bbox)
        self.assertEqual(composite.mode, "RGB")
        self.assertEqual(composite.size, (3 * WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main()
