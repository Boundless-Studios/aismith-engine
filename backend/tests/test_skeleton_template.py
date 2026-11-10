"""Tests for backend.app.skeleton_template: the skeleton family registry
loader and the S3-conformance checker (PIPELINE_POLICY.md).

Candidate rigs are synthesized directly from the shipped biped template
(`assets/skeletons/biped/template.json`) rather than hand-authored numbers,
so these tests track the real asset. Each synthetic rig uses head=(0,0,0)
for every bone and tail=<some direction vector>; only bone *direction*
(unit head->tail) is checked by `conformance_report`, never absolute
position, so a shared head origin is a faithful, minimal fixture.
"""
import copy
import math
import unittest

import numpy as np

from backend.app.skeleton_template import (
    DEFAULT_DIRECTION_TOLERANCE_DEGREES,
    Template,
    conformance_report,
    load_template,
    template_from_dict,
)


def _candidate_from_bones(bones):
    """Build (names, parents, heads, tails, rolls) from a sequence of
    `BoneSpec`-like objects (or the shipped template's own `.bones`),
    reproducing each bone's template direction exactly (head at origin,
    tail = unit rest_direction)."""
    names = [b.name for b in bones]
    parents = [b.parent for b in bones]
    heads = [[0.0, 0.0, 0.0] for _ in bones]
    tails = [list(b.rest_direction) for b in bones]
    rolls = [b.roll for b in bones]
    return names, parents, heads, tails, rolls


def _rotate(vector, axis, degrees):
    """Rodrigues' rotation formula -- rotate `vector` by `degrees` around
    the unit `axis`. Local helper so the test doesn't need scipy (the
    shipped module can't depend on it either -- see its module docstring)."""
    v = np.asarray(vector, dtype=np.float64)
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    theta = math.radians(degrees)
    return (
        v * math.cos(theta)
        + np.cross(k, v) * math.sin(theta)
        + k * np.dot(k, v) * (1 - math.cos(theta))
    )


class LoadTemplateTests(unittest.TestCase):
    def test_biped_template_loads_and_validates(self):
        template = load_template("biped")
        self.assertEqual(template.family, "biped")
        self.assertEqual(template.version, 1)
        self.assertEqual(len(template.required_bones), 21)
        self.assertEqual(len(template.optional_bones), 45)
        self.assertEqual(len(template.bones), 66)
        # exactly one root
        roots = [b for b in template.bones if b.parent is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].name, "root")
        # every parent reference resolves to a bone in the template
        names = template.bone_names()
        for bone in template.bones:
            if bone.parent is not None:
                self.assertIn(bone.parent, names)
        # core biped roles are present and required
        for expected in ("pelvis", "spine_1", "spine_2", "spine_3", "neck", "head",
                          "clavicle_l", "clavicle_r", "upper_arm_l", "upper_arm_r",
                          "lower_arm_l", "lower_arm_r", "hand_l", "hand_r",
                          "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r"):
            self.assertEqual(template.bone(expected).tier, "required", expected)
        # fingers and toes are optional even though the donor authors them
        for expected in ("thumb_1_l", "index_2_r", "toes_l", "toes_r"):
            self.assertEqual(template.bone(expected).tier, "optional", expected)
        # ROM clip metadata is populated and plausible
        self.assertTrue(template.rom_clip.name)
        self.assertGreater(template.rom_clip.frame_end, template.rom_clip.frame_start)
        self.assertGreaterEqual(template.rom_clip.joint_motion_coverage_pct, 90.0)
        # provenance points at the pinned curated donor
        self.assertEqual(
            template.provenance.donor_sha256,
            "406eb0a8dc4ab366e623b79b6e3005a4951392e1bda78ae39c1099d31147733c",
        )

    def test_duplicate_bone_names_rejected(self):
        template = load_template("biped")
        raw = _template_to_dict(template)
        raw["bones"].append(dict(raw["bones"][0]))  # duplicate the first bone's entry
        with self.assertRaises(ValueError):
            template_from_dict(raw)

    def test_dangling_parent_reference_rejected(self):
        template = load_template("biped")
        raw = _template_to_dict(template)
        raw["bones"][-1]["parent"] = "no_such_bone"
        with self.assertRaises(ValueError):
            template_from_dict(raw)

    def test_unknown_family_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_template("not-a-real-family")


def _template_to_dict(template: Template) -> dict:
    """Round-trip a loaded Template back to the raw-dict shape
    `template_from_dict` expects, for mutate-and-reload validation tests."""
    return {
        "family": template.family,
        "version": template.version,
        "provenance": vars(template.provenance),
        "coordinate_frame": vars(template.coordinate_frame),
        "skeleton_height_donor_units": template.skeleton_height_donor_units,
        "rom_clip": vars(template.rom_clip),
        "bones": [
            {
                "name": b.name,
                "parent": b.parent,
                "tier": b.tier,
                "donor_name": b.donor_name,
                "rest_direction": list(b.rest_direction),
                "relative_length": b.relative_length,
                "roll": b.roll,
            }
            for b in template.bones
        ],
    }


class ConformanceReportTests(unittest.TestCase):
    def setUp(self):
        self.template = load_template("biped")

    def _codes(self, findings):
        return [f["code"] for f in findings]

    def test_templates_own_bones_are_fully_conformant(self):
        names, parents, heads, tails, rolls = _candidate_from_bones(self.template.bones)
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        self.assertEqual(findings, [], findings)

    def test_optional_bones_absent_still_passes(self):
        required_only = self.template.required_bones
        names, parents, heads, tails, rolls = _candidate_from_bones(required_only)
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        self.assertEqual(findings, [], findings)

    def test_missing_required_bone_is_flagged(self):
        bones = [b for b in self.template.bones if b.name != "thigh_l"]
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        missing = [f for f in findings if f["code"] == "missing_required_bone"]
        self.assertEqual(len(missing), 1, findings)
        self.assertEqual(missing[0]["expected"], "'thigh_l' present")
        # removing a bone from the candidate list must not spuriously flag
        # its former children as reparented -- their parent name is
        # untouched, the parent bone is just absent (a separate finding).
        self.assertNotIn("reparented_bone", self._codes(findings))

    def test_reparented_bone_is_flagged(self):
        bones = list(self.template.bones)
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        calf_l_index = names.index("calf_l")
        self.assertEqual(parents[calf_l_index], "thigh_l")
        parents[calf_l_index] = "hand_l"  # wrong parent: calf reattached under the hand
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        reparented = [f for f in findings if f["code"] == "reparented_bone"]
        self.assertEqual(len(reparented), 1, findings)
        self.assertEqual(reparented[0]["actual"], "hand_l")
        self.assertEqual(reparented[0]["expected"], "thigh_l")

    def test_mirrored_left_right_is_flagged(self):
        # upper_arm_l/r have a strong lateral (x-sign) component in rest
        # pose -- unlike the legs, which point straight down and are
        # direction-identical between sides (see the biped README's
        # "surprises" section) -- so swapping their directions is a clean,
        # unambiguous mirror-detection fixture.
        bones = list(self.template.bones)
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        left_index = names.index("upper_arm_l")
        right_index = names.index("upper_arm_r")
        tails[left_index], tails[right_index] = tails[right_index], tails[left_index]
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        mirrored = [f for f in findings if f["code"] == "mirrored_left_right"]
        self.assertGreaterEqual(len(mirrored), 1, findings)
        flagged_bones = {n for n in ("upper_arm_l", "upper_arm_r")
                          if any(n in f["message"] for f in mirrored)}
        self.assertTrue(flagged_bones, findings)

    def test_off_direction_bone_is_flagged(self):
        # thigh_l/thigh_r are direction-identical in rest pose (both point
        # straight down), so perturbing thigh_l can never accidentally
        # score closer to the mirrored side -- this isolates the plain
        # direction-tolerance check from the mirror check.
        bones = list(self.template.bones)
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        index = names.index("thigh_l")
        original = np.array(tails[index])
        rotated = _rotate(original, axis=[0.0, 1.0, 0.0], degrees=30.0)
        tails[index] = rotated.tolist()
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        deviated = [f for f in findings if f["code"] == "rest_direction_deviation"]
        self.assertEqual(len(deviated), 1, findings)
        self.assertIn("thigh_l", deviated[0]["message"])
        self.assertGreater(deviated[0]["actual"], DEFAULT_DIRECTION_TOLERANCE_DEGREES)
        self.assertNotIn("mirrored_left_right", self._codes(findings))

    def test_unknown_bone_name_is_flagged(self):
        bones = list(self.template.bones)
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        names.append("bone_wearing_a_hat")
        parents.append("pelvis")
        heads.append([0.0, 0.0, 0.0])
        tails.append([0.0, 0.0, 1.0])
        rolls.append(0.0)
        findings = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        unknown = [f for f in findings if f["code"] == "unknown_bone_name"]
        self.assertEqual(len(unknown), 1, findings)
        self.assertEqual(unknown[0]["actual"], "bone_wearing_a_hat")

    def test_custom_direction_tolerance_is_honored(self):
        bones = list(self.template.bones)
        names, parents, heads, tails, rolls = _candidate_from_bones(bones)
        index = names.index("thigh_l")
        rotated = _rotate(np.array(tails[index]), axis=[0.0, 1.0, 0.0], degrees=10.0)
        tails[index] = rotated.tolist()
        # 10deg exceeds a tight 5deg tolerance...
        tight = conformance_report(
            names, parents, heads, tails, rolls, template=self.template, direction_tolerance_degrees=5.0,
        )
        self.assertIn("rest_direction_deviation", [f["code"] for f in tight])
        # ...but is within the default 15deg tolerance.
        loose = conformance_report(names, parents, heads, tails, rolls, template=self.template)
        self.assertNotIn("rest_direction_deviation", [f["code"] for f in loose])


if __name__ == "__main__":
    unittest.main()
