import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.score_views_2d as score_views_2d  # noqa: E402


def _solid_png(path: Path, *, size=(64, 64), background=(200, 200, 200), subject_box=(16, 8, 48, 56), subject=(20, 20, 20)) -> None:
    image = Image.new("RGB", size, background)
    for y in range(subject_box[1], subject_box[3]):
        for x in range(subject_box[0], subject_box[2]):
            image.putpixel((x, y), subject)
    image.save(path, format="PNG")


import unittest  # noqa: E402


class ParsePairsTests(unittest.TestCase):
    def test_parse_pairs_splits_role_painted_and_reference(self):
        pairs = score_views_2d.parse_pairs(
            "front=out/front.png:tpose/front.png,back=out/back.png:tpose/back.png"
        )
        self.assertEqual(
            pairs,
            {
                "front": (Path("out/front.png"), Path("tpose/front.png")),
                "back": (Path("out/back.png"), Path("tpose/back.png")),
            },
        )


class ComputeSourceFidelityIdenticalImageTests(unittest.TestCase):
    def test_identical_images_score_iou_1_and_high_ssim(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "same.png"
            _solid_png(path)
            rgba = score_views_2d.load_rgba(path)
            mask, _ = score_views_2d.subject_mask_from_image(rgba)

            metrics = score_views_2d.compute_source_fidelity(
                rgba, rgba, photo_mask=mask, render_mask=mask,
            )

        self.assertEqual(metrics["silhouette_iou"], 1.0)
        self.assertGreaterEqual(metrics["masked_ssim"], 0.99)


class ScoreViews2DDriverTests(unittest.TestCase):
    def test_scorer_writes_table_and_scores_json_for_identical_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            front_a = root / "front-a.png"
            front_b = root / "front-b.png"
            _solid_png(front_a)
            _solid_png(front_b)
            out_dir = root / "out"

            argv = [
                "score_views_2d.py",
                "--pairs", f"front={front_a}:{front_b}",
                "--out", str(out_dir),
            ]
            with patch.object(sys, "argv", argv):
                score_views_2d.main()

            scores_path = out_dir / "scores.json"
            self.assertTrue(scores_path.is_file())
            payload = json.loads(scores_path.read_text())

            self.assertIn("front", payload["scores"])
            self.assertEqual(payload["scores"]["front"]["silhouette_iou"], 1.0)
            self.assertGreaterEqual(payload["scores"]["front"]["masked_ssim"], 0.99)
            self.assertIn("mean", payload)
            self.assertEqual(payload["mean"]["silhouette_iou"], 1.0)

    def test_markdown_table_includes_the_documented_columns_and_mean_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            painted = root / "painted.png"
            reference = root / "reference.png"
            _solid_png(painted)
            _solid_png(reference)

            painted_rgba = score_views_2d.load_rgba(painted)
            reference_rgba = score_views_2d.load_rgba(reference)
            painted_mask, _ = score_views_2d.subject_mask_from_image(painted_rgba)
            reference_mask, _ = score_views_2d.subject_mask_from_image(reference_rgba)
            metrics = score_views_2d.compute_source_fidelity(
                reference_rgba, painted_rgba,
                photo_mask=reference_mask, render_mask=painted_mask,
            )
            table, means = score_views_2d.render_markdown_table({"front": metrics})

        for header, _ in score_views_2d.TABLE_COLUMNS:
            self.assertIn(header, table)
        self.assertIn("| role |", table)
        self.assertIn("mean", table)
        self.assertEqual(means["silhouette_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
