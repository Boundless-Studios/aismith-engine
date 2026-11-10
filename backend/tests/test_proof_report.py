import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.proof.build_report import build_report


class ProofReportTests(unittest.TestCase):
    def test_report_contains_verdicts_settings_renders_and_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "front.png").write_bytes(b"png")
            (root / "mesh.glb").write_bytes(b"glb")
            manifest = root / "run.json"
            manifest.write_text(json.dumps({
                "title": "Mary A-pose baseline",
                "source": "front.png",
                "stages": [{
                    "name": "Generate",
                    "verdict": "accepted",
                    "settings": {"seed": 0, "resolution": 512},
                    "renders": ["front.png"],
                    "artifacts": ["mesh.glb"],
                    "animated_glb": "mesh.glb",
                }],
            }), encoding="utf-8")

            output = build_report(manifest, root / "report")
            html = output.read_text(encoding="utf-8")

            self.assertIn("Mary A-pose baseline", html)
            self.assertIn("accepted", html)
            self.assertIn("resolution", html)
            self.assertIn("front.png", html)
            self.assertIn("mesh.glb", html)
            self.assertIn("animation-name=", html)


if __name__ == "__main__":
    unittest.main()
