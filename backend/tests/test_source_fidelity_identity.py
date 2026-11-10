"""Identity metrics (CLIP-I / DINOv2 cosine / LPIPS) on the source-fidelity
gate.

The real backbones are never loaded here: they are monkeypatched to
deterministic stand-ins (embedding = mean colour per channel, LPIPS = mean
absolute difference) so the suite asserts the plumbing -- which crops the
metrics see, which direction they move, what happens with no weights on the
box -- without depending on a download.
"""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from backend.app import source_fidelity as sf

WIDTH, HEIGHT = 240, 360
HEAD_CENTER = (120, 55)
HEAD_RADIUS = 30
BODY_BOX = (90, 85, 150, 280)
ARM_BOX = (60, 110, 90, 190)
SKIN = (216, 178, 140, 255)
BLACK = (0, 0, 0, 255)

# Every key compute_source_fidelity produced before the identity metrics
# existed; these must be untouched by their arrival.
GEOMETRY_KEYS = (
    "silhouette_iou",
    "masked_ssim",
    "face_ssim",
    "perceptual_distance",
    "masked_luma_delta",
    "masked_rgb_l1",
    "photo_detail",
    "render_detail",
    "detail_ratio",
    "face_detail_ratio",
    "subject_bbox",
    "face_bbox",
    "face_pixels",
    "region_pixels",
)


def _character(body_color=BLACK, head_color=SKIN, arm_color=SKIN):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = HEAD_CENTER
    draw.ellipse([cx - HEAD_RADIUS, cy - HEAD_RADIUS, cx + HEAD_RADIUS, cy + HEAD_RADIUS], fill=head_color)
    draw.rectangle(list(BODY_BOX), fill=body_color)
    draw.rectangle(list(ARM_BOX), fill=arm_color)
    return np.array(img)


def _unavailable_models(reason="torch not importable (test)"):
    return sf.IdentityModels(warnings=(f"clip_i/dino_cos/lpips unavailable: {reason}",))


def _fake_models():
    """A holder whose three slots are merely non-None -- the stand-in metric
    functions below never look at them."""
    return sf.IdentityModels(clip="clip", clip_processor="proc", dino="dino", dino_processor="proc", lpips="lpips")


def _mean_colour(models, image):
    return np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(axis=0)


def _mean_abs_difference(models, a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _with_fakes(*, models=None, record=None):
    """Patch the three backbone calls and the loader. `record`, when given, is
    appended to with the shape of every array a metric is handed."""
    holder = _fake_models() if models is None else models

    def clip_embedding(m, image):
        if record is not None:
            record.append(np.asarray(image).shape)
        return _mean_colour(m, image)

    return (
        patch.object(sf, "_clip_embedding", clip_embedding),
        patch.object(sf, "_dino_embedding", _mean_colour),
        patch.object(sf, "_lpips_forward", _mean_abs_difference),
        patch.object(sf, "load_identity_models", lambda **kwargs: holder),
    )


class _FakeModelTestCase(unittest.TestCase):
    def enter_fakes(self, *, models=None, record=None):
        for patcher in _with_fakes(models=models, record=record):
            patcher.start()
            self.addCleanup(patcher.stop)


class MaskedCropTests(unittest.TestCase):
    def test_crops_the_box_and_greys_everything_outside_the_mask(self):
        rgb = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 3:6] = True

        crop = sf.masked_crop(rgb, mask, (3, 2, 6, 5))

        self.assertEqual(crop.shape, (3, 3, 3))
        np.testing.assert_allclose(crop, rgb[2:5, 3:6])

        wider = sf.masked_crop(rgb, mask, (1, 1, 7, 7))
        self.assertEqual(wider.shape, (6, 6, 3))
        outside = ~mask[1:7, 1:7]
        np.testing.assert_allclose(wider[outside], sf.NEUTRAL_BACKGROUND)
        np.testing.assert_allclose(wider[~outside], rgb[1:7, 1:7][~outside])


class IdentityMetricDirectionTests(_FakeModelTestCase):
    """Directions, not magic numbers: the stand-in embeddings have no
    calibrated scale, so only the ORDERING of the three cases is meaningful."""

    def setUp(self):
        self.enter_fakes()
        rng = np.random.default_rng(7)
        self.image = rng.random((48, 48, 3)).astype(np.float32)
        self.brighter = np.clip(self.image + 0.10, 0.0, 1.0).astype(np.float32)
        # A different colour, not a different exposure: the red and blue
        # channels trade places, so the mean-colour direction really moves.
        self.recoloured = np.clip(self.image * np.array([1.6, 0.7, 0.4], dtype=np.float32), 0.0, 1.0).astype(np.float32)

    def test_identical_images_score_perfectly(self):
        self.assertAlmostEqual(sf.clip_similarity(self.image, self.image), 1.0, places=9)
        self.assertAlmostEqual(sf.dino_similarity(self.image, self.image), 1.0, places=9)
        self.assertAlmostEqual(sf.lpips_distance(self.image, self.image), 0.0, places=9)

    def test_a_brightness_shift_moves_less_than_a_colour_change(self):
        for similarity in (sf.clip_similarity, sf.dino_similarity):
            with self.subTest(metric=similarity.__name__):
                same = similarity(self.image, self.image)
                bright = similarity(self.image, self.brighter)
                colour = similarity(self.image, self.recoloured)
                self.assertLess(bright, same)
                self.assertLess(colour, bright)

        same = sf.lpips_distance(self.image, self.image)
        bright = sf.lpips_distance(self.image, self.brighter)
        colour = sf.lpips_distance(self.image, self.recoloured)
        self.assertGreater(bright, same)
        self.assertGreater(colour, bright)

    def test_cosine_of_a_zero_embedding_is_none_not_a_nan(self):
        black = np.zeros((16, 16, 3), dtype=np.float32)
        self.assertIsNone(sf.clip_similarity(black, self.image))


class IdentityMetricCropTests(_FakeModelTestCase):
    def test_metrics_see_exactly_the_crops_ssim_scores(self):
        photo = _character()
        record: list[tuple] = []
        self.enter_fakes(record=record)

        report = sf.compute_source_fidelity(photo, photo)

        mask, _ = sf.subject_mask_from_image(photo)
        x0, y0, x1, y1 = sf.subject_bbox(mask)
        fx0, fy0, fx1, fy1 = sf.head_bbox(mask)
        subject_shape = (y1 - y0, x1 - x0, 3)
        face_shape = (fy1 - fy0, fx1 - fx0, 3)

        self.assertEqual(report["subject_bbox"], [x0, y0, x1, y1])
        self.assertEqual(report["face_bbox"], [fx0, fy0, fx1, fy1])
        # Photo and render crop for the subject, then the same pair for the face.
        self.assertEqual(record, [subject_shape, subject_shape, face_shape, face_shape])

    def test_a_face_crop_below_the_ssim_floor_is_skipped(self):
        photo = _character()
        record: list[tuple] = []
        self.enter_fakes(record=record)

        with patch.object(sf, "MIN_FACE_PIXELS", 10**9):
            report = sf.compute_source_fidelity(photo, photo)

        self.assertIsNone(report["face_ssim"])
        self.assertIsNone(report["clip_i_face"])
        self.assertIsNone(report["dino_cos_face"])
        self.assertIsNone(report["lpips_face"])
        self.assertIsNotNone(report["clip_i"])
        self.assertEqual(len(record), 2)  # subject only


class ReportShapeTests(_FakeModelTestCase):
    def test_report_carries_the_six_identity_keys(self):
        self.enter_fakes()
        photo = _character()

        report = sf.compute_source_fidelity(photo, photo)

        self.assertAlmostEqual(report["clip_i"], 1.0, places=4)
        self.assertAlmostEqual(report["clip_i_face"], 1.0, places=4)
        self.assertAlmostEqual(report["dino_cos"], 1.0, places=4)
        self.assertAlmostEqual(report["dino_cos_face"], 1.0, places=4)
        self.assertEqual(report["lpips"], 0.0)
        self.assertEqual(report["lpips_face"], 0.0)
        self.assertEqual(report["metric_warnings"], [])
        self.assertEqual(report["identity_models"]["clip"], sf.CLIP_MODEL_ID)

    def test_a_recoloured_body_moves_the_identity_metrics(self):
        self.enter_fakes()
        photo = _character()
        render = _character(body_color=(220, 40, 40, 255))

        same = sf.compute_source_fidelity(photo, photo)
        drifted = sf.compute_source_fidelity(photo, render)

        self.assertLess(drifted["clip_i"], same["clip_i"])
        self.assertGreater(drifted["lpips"], same["lpips"])


class UnavailableModelTests(_FakeModelTestCase):
    def test_missing_models_report_none_and_a_warning(self):
        photo = _character()
        with patch.object(sf, "load_identity_models", lambda **kwargs: _unavailable_models()):
            report = sf.compute_source_fidelity(photo, photo)

        for key in sf.IDENTITY_METRIC_KEYS:
            self.assertIsNone(report[key], key)
        self.assertEqual(len(report["metric_warnings"]), 1)
        self.assertIn("unavailable", report["metric_warnings"][0])

    def test_the_existing_metrics_are_byte_identical_with_and_without_models(self):
        photo = _character()
        render = _character(body_color=(128, 128, 128, 255), head_color=(20, 20, 20, 255))

        with patch.object(sf, "load_identity_models", lambda **kwargs: _unavailable_models()):
            without = sf.compute_source_fidelity(photo, render)
        self.enter_fakes()
        with_models = sf.compute_source_fidelity(photo, render)

        self.assertEqual(
            {key: without[key] for key in GEOMETRY_KEYS},
            {key: with_models[key] for key in GEOMETRY_KEYS},
        )
        # Pinned so a future change to the shared crops has to admit it here.
        self.assertEqual(without["silhouette_iou"], 1.0)
        self.assertEqual(without["masked_ssim"], 0.1577)
        self.assertEqual(without["face_ssim"], 0.2138)
        self.assertEqual(without["masked_luma_delta"], 0.2364)
        self.assertEqual(without["masked_rgb_l1"], 0.4488)
        self.assertEqual(without["detail_ratio"], 0.3392)

    def test_one_missing_backbone_does_not_take_the_others_down(self):
        photo = _character()
        models = _fake_models()
        models.clip = None
        models.warnings = ("clip_i unavailable: weights not cached",)
        self.enter_fakes(models=models)

        report = sf.compute_source_fidelity(photo, photo)

        self.assertIsNone(report["clip_i"])
        self.assertIsNone(report["clip_i_face"])
        self.assertAlmostEqual(report["dino_cos"], 1.0, places=4)
        self.assertEqual(report["lpips"], 0.0)
        self.assertEqual(report["metric_warnings"], ["clip_i unavailable: weights not cached"])


class LoaderTests(unittest.TestCase):
    def test_the_loader_is_a_cached_singleton(self):
        sentinel = _unavailable_models()
        with patch.object(sf, "_IDENTITY_MODELS", None), patch.object(
            sf, "_load_identity_models", side_effect=[sentinel, _unavailable_models("second")]
        ) as load:
            first = sf.load_identity_models()
            second = sf.load_identity_models()
            third = sf.load_identity_models(reload=True)

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertIsNot(third, sentinel)
        self.assertEqual(load.call_count, 2)

    def test_a_torch_import_failure_is_a_warning_not_an_exception(self):
        with patch.dict("sys.modules", {"torch": None}):
            models = sf._load_identity_models()

        self.assertIsNone(models.clip)
        self.assertIsNone(models.dino)
        self.assertIsNone(models.lpips)
        self.assertEqual(len(models.warnings), 1)
        self.assertIn("torch not importable", models.warnings[0])

    def test_downloads_are_opt_in(self):
        with patch.dict("os.environ", {sf.IDENTITY_DOWNLOAD_ENV: ""}, clear=False):
            self.assertFalse(sf._download_allowed())
        with patch.dict("os.environ", {sf.IDENTITY_DOWNLOAD_ENV: "0"}, clear=False):
            self.assertFalse(sf._download_allowed())
        with patch.dict("os.environ", {sf.IDENTITY_DOWNLOAD_ENV: "1"}, clear=False):
            self.assertTrue(sf._download_allowed())


if __name__ == "__main__":
    unittest.main()
