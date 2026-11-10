"""Tests for the workstream N4 biped clip store
(assets/skeletons/biped/clips/): the one-time offline port of the 87
curated donor animation clips onto the canonical biped template skeleton.

Pure-python / no bpy -- only checks the manifest and GLB produced by
scripts/port_curated_clips_to_biped.py. Bone-set/hierarchy conformance and
motion-survives-the-round-trip checks require Blender and live in
scripts/verify_biped_clip_store.py instead (that script is the source of
truth for those; this file covers what a plain pytest run can check).
"""
import json
import unittest
from pathlib import Path

from backend.app.native import CURATED_SWORD_DONOR_SHA256
from backend.app.skeleton_template import load_template

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "assets" / "skeletons" / "biped" / "clips"
MANIFEST_PATH = CLIPS_DIR / "clips-manifest.json"
STORE_GLB_PATH = CLIPS_DIR / "biped-clips-v1.glb"
EXPECTED_CLIP_COUNT = 87
ACTION_NAME_SUFFIX = "_Armature"


@unittest.skipUnless(MANIFEST_PATH.is_file(), f"clip store not yet built: {MANIFEST_PATH}")
class BipedClipManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_store_glb_exists_and_is_reasonably_sized(self):
        self.assertTrue(STORE_GLB_PATH.is_file(), f"missing {STORE_GLB_PATH}")
        size = STORE_GLB_PATH.stat().st_size
        # Armature + animations only (meshes stripped): expect the same
        # order of magnitude as the donor GLB (a few MB), not a bare-bones
        # empty file and not a mesh-bloated one.
        self.assertGreater(size, 200_000, f"store GLB suspiciously small: {size} bytes")
        self.assertLess(size, 20_000_000, f"store GLB suspiciously large: {size} bytes")

    def test_family_and_provenance(self):
        self.assertEqual(self.manifest["family"], "biped")
        provenance = self.manifest["provenance"]
        self.assertEqual(provenance["donor_sha256"], CURATED_SWORD_DONOR_SHA256)
        template = load_template("biped")
        self.assertEqual(provenance["template_version"], template.version)

    def test_exactly_87_clips_no_duplicates(self):
        clips = self.manifest["clips"]
        self.assertEqual(len(clips), EXPECTED_CLIP_COUNT)
        self.assertEqual(self.manifest["clip_count"], EXPECTED_CLIP_COUNT)
        names = [c["clip_name"] for c in clips]
        self.assertEqual(len(names), len(set(names)), "duplicate clip_name entries")

    def test_clip_name_convention_and_display_name_contract(self):
        for clip in self.manifest["clips"]:
            clip_name = clip["clip_name"]
            with self.subTest(clip_name=clip_name):
                self.assertTrue(
                    clip_name.endswith(ACTION_NAME_SUFFIX),
                    f"{clip_name!r} does not carry the importer-stable _Armature suffix",
                )
                self.assertEqual(
                    clip["display_name"],
                    clip_name[: -len(ACTION_NAME_SUFFIX)],
                    "display_name must be clip_name with only the trailing "
                    "_Armature suffix stripped -- no other normalization",
                )
                self.assertGreater(clip["frame_count"], 0)
                self.assertGreaterEqual(clip["duration_seconds"], 0.0)

    def test_known_sample_clips_present(self):
        names = {c["clip_name"] for c in self.manifest["clips"]}
        for expected in ("Roll_Armature", "Sword_Attack_Armature", "Walk_Armature"):
            self.assertIn(expected, names)

    def test_lookup_contract_documented(self):
        self.assertIn("lookup_contract", self.manifest)
        self.assertIn("clip_name", self.manifest["lookup_contract"])


if __name__ == "__main__":
    unittest.main()
