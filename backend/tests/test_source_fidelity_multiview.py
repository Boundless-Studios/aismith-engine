"""`scripts/source_fidelity_multiview.py`: scores a mesh against several
source photos at once, one per view role, so a T-pose asset's drift can't
hide behind a single flattering front-view number."""
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "source_fidelity_multiview.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("source_fidelity_multiview", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_report(role: str, yaw_degrees: float) -> dict:
    return {
        "silhouette_iou": 0.9,
        "masked_ssim": 0.8,
        "face_ssim": 0.7,
        "clip_i": 0.92,
        "clip_i_face": 0.89,
        "dino_cos": 0.87,
        "dino_cos_face": 0.84,
        "lpips": 0.12,
        "lpips_face": 0.13,
        "metric_warnings": [],
        "detail_ratio": 0.6,
        "face_detail_ratio": 0.5,
        "masked_luma_delta": 0.01,
        "role": role,
        "yaw_degrees": yaw_degrees,
    }


class SourceFidelityMultiviewTests(unittest.TestCase):
    def test_scores_each_given_view_with_the_right_yaw_and_writes_outputs(self):
        module = _load_module()

        def fake_measure(mesh, photo, out_dir, *, yaw_degrees, fov_x_degrees=None, samples=24):
            role = Path(out_dir).name
            return _fake_report(role, yaw_degrees)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "out"
            front = root / "front.png"
            back = root / "back.png"
            mesh = root / "character.glb"

            argv = [
                "source_fidelity_multiview.py",
                "--mesh", str(mesh),
                "--out", str(out_dir),
                "--front", str(front),
                "--back", str(back),
            ]
            stdout = io.StringIO()
            with (
                patch.object(module.native, "measure_source_fidelity", side_effect=fake_measure) as measure,
                patch.object(sys, "argv", argv),
                redirect_stdout(stdout),
            ):
                module.main()

            self.assertEqual(measure.call_count, 2)
            yaw_by_role = {}
            for call in measure.call_args_list:
                role = Path(call.args[2]).name
                yaw_by_role[role] = call.kwargs["yaw_degrees"]
            self.assertEqual(yaw_by_role, {"front": 0.0, "back": 180.0})

            written = json.loads((out_dir / "multiview.json").read_text(encoding="utf-8"))
            self.assertEqual(set(written), {"front", "back"})
            self.assertEqual(written["front"]["yaw_degrees"], 0.0)

            self.assertEqual(written["front"]["clip_i_face"], 0.89)  # face variants: JSON only

            table = stdout.getvalue()
            self.assertIn("face detail", table)
            for column in ("CLIP-I", "DINO", "LPIPS"):
                self.assertIn(column, table)
            self.assertNotIn("clip_i_face", table)
            self.assertIn("0.9200", table)
            self.assertIn("| front |", table)
            self.assertIn("| back |", table)
            self.assertIn("| mean |", table)

    def test_a_missing_backbone_is_explained_under_the_table(self):
        module = _load_module()

        def fake_measure(mesh, photo, out_dir, *, yaw_degrees, fov_x_degrees=None, samples=24):
            report = _fake_report(Path(out_dir).name, yaw_degrees)
            report.update({"clip_i": None, "metric_warnings": ["clip_i unavailable: weights not cached"]})
            return report

        with TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "source_fidelity_multiview.py",
                "--mesh", str(root / "character.glb"),
                "--out", str(root / "out"),
                "--front", str(root / "front.png"),
                "--back", str(root / "back.png"),
            ]
            stdout = io.StringIO()
            with (
                patch.object(module.native, "measure_source_fidelity", side_effect=fake_measure),
                patch.object(sys, "argv", argv),
                redirect_stdout(stdout),
            ):
                module.main()

            output = stdout.getvalue()
            # Deduplicated: both views carry the same warning, it prints once.
            self.assertEqual(output.count("clip_i unavailable: weights not cached"), 1)

    def test_requires_at_least_one_view(self):
        module = _load_module()
        with TemporaryDirectory() as directory:
            argv = [
                "source_fidelity_multiview.py",
                "--mesh", str(Path(directory) / "character.glb"),
                "--out", str(Path(directory) / "out"),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    module.main()


if __name__ == "__main__":
    unittest.main()
