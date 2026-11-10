"""Skeleton family registry: typed access to `assets/skeletons/<family>/template.json`
and a conformance checker for a candidate rig against a family's template.

Part of workstream N2 (PIPELINE_POLICY.md's S3 contract: "The skeleton IS the
family template: exact bone set, names, hierarchy, orientations/rolls.
Fitters ... only PLACE the template's joints; they never emit bespoke
topology, extra bones, or renamed bones."). This module owns the template
data model and the conformance check; wiring it into the bind worker /
quality_policy gate is a later bead.

Pure python/numpy -- no bpy. `skeleton_naming.py` carries the same
constraint (it is imported inside Blender's bundled python, which has numpy
but not scipy), so this module follows the same discipline: no scipy, no
bpy, importable both on the host and inside Blender.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKELETONS_DIR = ROOT / "assets" / "skeletons"

Severity = Literal["warning", "error"]
Tier = Literal["required", "optional"]

# S3's blame-table entry for limb roll/twist is unconditional ("template
# rolls are law"), but donor-authored rolls still carry harmless per-bone
# jitter (see clavicle_l/r: 0.0429 rad ~= 2.5deg) that a geometrically
# faithful fit will not reproduce bit-for-bit. 15deg is generous enough to
# absorb that jitter while still catching an actually wrong orientation
# (a swapped axis or a mirrored limb is >> 15deg off).
DEFAULT_DIRECTION_TOLERANCE_DEGREES = 15.0
DEFAULT_ROLL_TOLERANCE_DEGREES = 20.0

_EPS = 1e-9


@dataclass(frozen=True)
class BoneSpec:
    """One bone's entry in a skeleton family template.

    `rest_direction` is the unit head->tail vector in the family's house
    frame (Z-up, -Y-facing, left = +X -- see `CoordinateFrame`).
    `relative_length` is the bone's rest length as a fraction of the
    donor skeleton's total height, so the template is scale-independent.
    `roll` is the Blender edit-bone roll angle, in radians, that defines
    the bone's local twist/up-axis convention.
    """

    name: str
    parent: str | None
    tier: Tier
    donor_name: str
    rest_direction: tuple[float, float, float]
    relative_length: float
    roll: float


@dataclass(frozen=True)
class Provenance:
    donor_path: str
    donor_sha256: str
    donor_armature_name: str
    donor_bone_count: int


@dataclass(frozen=True)
class CoordinateFrame:
    up_axis: str
    forward_axis: str
    left_axis: str
    note: str


@dataclass(frozen=True)
class RomClip:
    """The family's range-of-motion calibration clip (S4 gate input)."""

    name: str
    frame_start: int
    frame_end: int
    joint_motion_coverage_pct: float
    coverage_metric: str
    selection_note: str


@dataclass(frozen=True)
class Template:
    family: str
    version: int
    provenance: Provenance
    coordinate_frame: CoordinateFrame
    skeleton_height_donor_units: float
    rom_clip: RomClip
    bones: tuple[BoneSpec, ...]

    def __post_init__(self) -> None:
        names = [b.name for b in self.bones]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"template {self.family!r} has duplicate bone names: {duplicates}")
        known = set(names)
        for bone in self.bones:
            if bone.parent is not None and bone.parent not in known:
                raise ValueError(
                    f"template {self.family!r} bone {bone.name!r} has unknown parent {bone.parent!r}"
                )
            if bone.tier not in ("required", "optional"):
                raise ValueError(f"template {self.family!r} bone {bone.name!r} has invalid tier {bone.tier!r}")
        roots = [b for b in self.bones if b.parent is None]
        if len(roots) != 1:
            raise ValueError(
                f"template {self.family!r} must have exactly one root bone (parent=None); found {len(roots)}"
            )

    @property
    def required_bones(self) -> tuple[BoneSpec, ...]:
        return tuple(b for b in self.bones if b.tier == "required")

    @property
    def optional_bones(self) -> tuple[BoneSpec, ...]:
        return tuple(b for b in self.bones if b.tier == "optional")

    def bone(self, name: str) -> BoneSpec:
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(f"template {self.family!r} has no bone named {name!r}")

    def bone_names(self) -> frozenset[str]:
        return frozenset(b.name for b in self.bones)

    def mirror_of(self, name: str) -> str | None:
        """The opposite-side canonical name for a `_l`/`_r` bone, or None."""
        if name.endswith("_l"):
            return name[: -len("_l")] + "_r"
        if name.endswith("_r"):
            return name[: -len("_r")] + "_l"
        return None


def load_template(family: str, *, skeletons_dir: Path | None = None) -> Template:
    """Load and validate `assets/skeletons/<family>/template.json`."""
    base = skeletons_dir if skeletons_dir is not None else SKELETONS_DIR
    path = base / family / "template.json"
    if not path.is_file():
        raise FileNotFoundError(f"no template for skeleton family {family!r}: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return template_from_dict(raw)


def template_from_dict(raw: dict[str, Any]) -> Template:
    bones = tuple(
        BoneSpec(
            name=b["name"],
            parent=b["parent"],
            tier=b["tier"],
            donor_name=b["donor_name"],
            rest_direction=(
                float(b["rest_direction"][0]),
                float(b["rest_direction"][1]),
                float(b["rest_direction"][2]),
            ),
            relative_length=float(b["relative_length"]),
            roll=float(b["roll"]),
        )
        for b in raw["bones"]
    )
    return Template(
        family=raw["family"],
        version=int(raw["version"]),
        provenance=Provenance(**raw["provenance"]),
        coordinate_frame=CoordinateFrame(**raw["coordinate_frame"]),
        skeleton_height_donor_units=float(raw["skeleton_height_donor_units"]),
        rom_clip=RomClip(**raw["rom_clip"]),
        bones=bones,
    )


def _finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    metric: str,
    actual: Any,
    expected: str,
) -> dict[str, Any]:
    """Same shape as `quality_policy._finding` (code/severity/message/metric/
    actual/expected) so wiring this into that gate later is a straight
    concatenation of finding lists, not a reshape."""
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metric": metric,
        "actual": actual,
        "expected": expected,
    }


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return None
    return vector / norm


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _roll_deviation_degrees(actual_roll: float, expected_roll: float) -> float:
    """Smallest angular difference between two roll angles (radians in,
    degrees out), correctly wrapped across the +-pi seam."""
    delta = (actual_roll - expected_roll + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(delta))


def conformance_report(
    names: Sequence[str],
    parents: Sequence[str | None],
    heads: Sequence[Sequence[float]] | np.ndarray,
    tails: Sequence[Sequence[float]] | np.ndarray,
    rolls: Sequence[float] | None = None,
    *,
    family: str = "biped",
    template: Template | None = None,
    direction_tolerance_degrees: float = DEFAULT_DIRECTION_TOLERANCE_DEGREES,
    roll_tolerance_degrees: float = DEFAULT_ROLL_TOLERANCE_DEGREES,
) -> list[dict[str, Any]]:
    """Check a candidate rig's bone set against a skeleton family template.

    `names`, `parents`, `heads`, `tails` (and optional `rolls`) are aligned
    by index and describe the candidate rig's rest pose: `parents[i]` is the
    *name* of bone i's parent (or None for a root), `heads`/`tails` are
    world-space rest positions in the family's house frame (see
    `Template.coordinate_frame`) -- the same frame the fitter is contracted
    to place joints in, so no additional transform is applied here.

    Checks (S3 contract, PIPELINE_POLICY.md):
      - every `template.required_bones` name is present (`missing_required_bone`)
      - every candidate name not known to the template is flagged
        (`unknown_bone_name`) -- known *optional* bones being absent is not
        an error, but a present bone with an unrecognized name is
      - for every candidate bone matched to a template bone, its parent
        name matches the template's (`reparented_bone`)
      - for every candidate bone matched to a template bone, its rest
        direction (unit head->tail) is within `direction_tolerance_degrees`
        of the template's (`rest_direction_deviation`)
      - for `_l`/`_r` bones, the candidate direction is closer to its own
        side's template direction than to the mirrored side's
        (`mirrored_left_right`) -- catches a rig whose left/right bones
        were swapped even when each individually still resembles *a*
        plausible arm/leg direction
      - if `rolls` is given, each matched bone's roll is within
        `roll_tolerance_degrees` of the template's (`roll_deviation`)

    Returns a list of finding dicts shaped like `quality_policy`'s
    (code/severity/message/metric/actual/expected); empty means fully
    conformant.
    """
    tmpl = template if template is not None else load_template(family)
    heads = np.asarray(heads, dtype=np.float64)
    tails = np.asarray(tails, dtype=np.float64)
    if len(names) != len(parents) or len(names) != len(heads) or len(names) != len(tails):
        raise ValueError("names, parents, heads, tails must all have the same length")
    if rolls is not None and len(rolls) != len(names):
        raise ValueError("rolls must have the same length as names")

    findings: list[dict[str, Any]] = []

    by_name: dict[str, int] = {}
    for i, name in enumerate(names):
        by_name[name] = i  # last one wins; duplicate names are not this check's concern

    known_names = tmpl.bone_names()

    for bone in tmpl.required_bones:
        if bone.name not in by_name:
            findings.append(_finding(
                "missing_required_bone",
                "error",
                f"required bone {bone.name!r} is missing from the candidate rig",
                metric="required_bone_present",
                actual=False,
                expected=f"{bone.name!r} present",
            ))

    for name in names:
        if name not in known_names:
            findings.append(_finding(
                "unknown_bone_name",
                "error",
                f"bone name {name!r} is not part of the {tmpl.family!r} template "
                "(fitters must not emit renamed or bespoke-topology bones)",
                metric="bone_name_known",
                actual=name,
                expected=f"one of {sorted(known_names)}",
            ))

    for name, index in by_name.items():
        if name not in known_names:
            continue
        bone = tmpl.bone(name)
        actual_parent = parents[index]
        if actual_parent != bone.parent:
            findings.append(_finding(
                "reparented_bone",
                "error",
                f"bone {name!r} has parent {actual_parent!r}, template expects {bone.parent!r}",
                metric="parent_name",
                actual=actual_parent,
                expected=bone.parent,
            ))

        direction = _unit(tails[index] - heads[index])
        if direction is None:
            findings.append(_finding(
                "degenerate_bone_length",
                "error",
                f"bone {name!r} has zero-length rest pose (head == tail); cannot check direction",
                metric="rest_length",
                actual=0.0,
                expected="> 0",
            ))
        else:
            template_direction = np.array(bone.rest_direction, dtype=np.float64)
            angle = _angle_degrees(direction, template_direction)

            mirror_name = tmpl.mirror_of(name)
            flagged_as_mirrored = False
            if mirror_name is not None and mirror_name in known_names:
                mirror_direction = np.array(tmpl.bone(mirror_name).rest_direction, dtype=np.float64)
                mirror_angle = _angle_degrees(direction, mirror_direction)
                if mirror_angle < angle:
                    findings.append(_finding(
                        "mirrored_left_right",
                        "error",
                        f"bone {name!r} points closer to {mirror_name!r}'s template direction "
                        f"than to its own; left/right sides look swapped",
                        metric="angle_to_own_side_vs_mirrored_side_deg",
                        actual=[round(angle, 3), round(mirror_angle, 3)],
                        expected=f"angle_to_own ({round(angle, 3)}) <= angle_to_mirrored ({round(mirror_angle, 3)})",
                    ))
                    flagged_as_mirrored = True

            if not flagged_as_mirrored and angle > direction_tolerance_degrees:
                findings.append(_finding(
                    "rest_direction_deviation",
                    "error",
                    f"bone {name!r} rest direction is {round(angle, 2)}deg from the template "
                    f"(tolerance {direction_tolerance_degrees}deg)",
                    metric="direction_angle_deg",
                    actual=round(angle, 3),
                    expected=f"<= {direction_tolerance_degrees}",
                ))

        if rolls is not None:
            roll_delta = _roll_deviation_degrees(float(rolls[index]), bone.roll)
            if roll_delta > roll_tolerance_degrees:
                findings.append(_finding(
                    "roll_deviation",
                    "error",
                    f"bone {name!r} roll is {round(roll_delta, 2)}deg from the template "
                    f"(tolerance {roll_tolerance_degrees}deg)",
                    metric="roll_angle_deg",
                    actual=round(roll_delta, 3),
                    expected=f"<= {roll_tolerance_degrees}",
                ))

    return findings
