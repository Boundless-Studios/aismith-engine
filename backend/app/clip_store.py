"""Family clip store lookup (workstream N5's read-only consumer of N4's
`assets/skeletons/<family>/clips/` store).

Pure python -- no bpy, no scipy. Importable both on the host (main.py's
apply-clip job validation) and inside Blender's bundled interpreter
(backend/workers/blender_worker.py's apply_clip op), same discipline
skeleton_template.py and skeleton_naming.py already follow.

Lookup contract (assets/skeletons/<family>/clips/clips-manifest.json's own
"lookup_contract" field is authoritative -- this module is a thin, typed
wrapper around exactly that): a consumer selects a clip by exact match on
"clip_name" (e.g. "Sword_Attack_Armature") against `bpy.data.actions` after
importing the family's store GLB, and assigns it to the armature's
`animation_data.action`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .skeleton_template import load_template

ROOT = Path(__file__).resolve().parents[2]
SKELETONS_DIR = ROOT / "assets" / "skeletons"


@dataclass(frozen=True)
class ClipManifestEntry:
    clip_name: str
    display_name: str
    frame_start: int
    frame_end: int
    frame_count: int
    duration_seconds: float


def clips_dir(family: str) -> Path:
    return SKELETONS_DIR / family / "clips"


def manifest_path(family: str) -> Path:
    return clips_dir(family) / "clips-manifest.json"


def load_manifest(family: str) -> dict[str, Any]:
    """Load `assets/skeletons/<family>/clips/clips-manifest.json`.

    Raises FileNotFoundError for a family with no clip store -- the caller
    (main.py's apply-clip job validation) turns that into a clear 400/409,
    never a 500.
    """
    path = manifest_path(family)
    if not path.is_file():
        raise FileNotFoundError(f"no clip store for family {family!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def clip_store_path(family: str) -> Path:
    """Path to the family's clip store GLB (import this into Blender and
    pull an action by exact clip_name -- see the manifest's own
    lookup_contract)."""
    manifest = load_manifest(family)
    return clips_dir(family) / str(manifest["store_file"])


def find_clip(family: str, clip_name: str) -> ClipManifestEntry | None:
    """Exact clip_name lookup against the family manifest (the manifest's own
    lookup contract: "a consumer selects a clip by exact match on
    'clip_name'"). Returns None if the family has no clip by that name -- a
    family with no clip store at all still raises FileNotFoundError, from
    load_manifest above."""
    manifest = load_manifest(family)
    for clip in manifest.get("clips", []):
        if clip.get("clip_name") == clip_name:
            return ClipManifestEntry(
                clip_name=clip["clip_name"],
                display_name=clip["display_name"],
                frame_start=int(clip["frame_start"]),
                frame_end=int(clip["frame_end"]),
                frame_count=int(clip["frame_count"]),
                duration_seconds=float(clip["duration_seconds"]),
            )
    return None


def missing_animated_bones(
    animated_bones: "list[str] | set[str] | frozenset[str]",
    rig_bone_names: "list[str] | set[str] | frozenset[str]",
) -> list[str]:
    """S5's `s5.exact_bone_name_match` (docs/PIPELINE_POLICY.md,
    pipeline_policy.py's S5_CHECKS): bones an action's fcurves target that
    the target rig does not carry, by exact name.

    Direct-attach (this workstream's whole point) has no retarget math to
    paper over a mismatch -- see `classify_missing_bones` below for how a
    non-empty result here is actually dispositioned (blender_worker.py's
    apply_clip op calls both in sequence). Pure and sorted so it is
    independently unit-testable without bpy or a real rig file.
    """
    return sorted(set(animated_bones) - set(rig_bone_names))


def classify_missing_bones(missing_bones: "list[str]", family: str) -> "tuple[list[str], list[str]]":
    """Partition `missing_animated_bones`'s output into (hard_fail, skip)
    using the family template's own required/optional tiers
    (skeleton_template.Template, S3's contract).

    A clip's motion touching an OPTIONAL template bone the rig doesn't
    carry (a finger segment, a toe, head_tip -- exactly the store's own 66
    template bones minus the ~21 required ones) is a valid, expected gap:
    S3 already treats an optional bone's absence as conformant ("missing
    OPTIONAL bones being absent is not an error"), and every clip in the
    store was baked on the FULL 66-bone canonical template, so it carries a
    channel for every one of those bones whether or not a given rig
    instance actually has them. Direct-attach must not hard-fail on that --
    the orphaned channels for a bone the rig never had are simply inert
    once the action is assigned (Blender does not error on an fcurve whose
    pose-bone target does not exist; it just never drives anything).

    A clip touching a REQUIRED template bone the rig doesn't carry would
    mean the rig was never actually template-conformant in the first place
    (certification already requires every required bone) -- that stays a
    hard fail. Anything not even a template bone at all is also a hard
    fail (this rig, or this clip, is not what it claims to be).
    """
    template = load_template(family)
    hard_fail: list[str] = []
    skip: list[str] = []
    for name in missing_bones:
        try:
            bone = template.bone(name)
        except KeyError:
            hard_fail.append(name)
            continue
        (hard_fail if bone.tier == "required" else skip).append(name)
    return sorted(hard_fail), sorted(skip)
