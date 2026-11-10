from __future__ import annotations

from typing import Any

from . import native
from .rig_quality import _accessor_reader, _read_glb

# Blender's glTF importer (io_scene_gltf2, exercised by
# backend/workers/unirig_bind_worker.py:export_soma_bvh when it imports this
# donor) recreates every armature-owned action with the source armature
# object's name appended -- this donor's only armature object is named
# "Armature", so every action lands as "<raw-glTF-animation-name>_Armature".
# Confirmed empirically: calling POST /api/animations/export-curated-bvh with
# an unknown `animation` makes export_soma_bvh raise "donor library has no
# action named ...; available: [...]", and every one of the 87 names in that
# list carries this exact suffix. The raw glTF JSON `animations[].name`
# values read directly out of the GLB (below) do NOT carry it, so it must be
# appended here to produce the exact name callers have to send back to that
# endpoint.
_ACTION_NAME_SUFFIX = "_Armature"

_cache: dict[str, Any] | None = None


def curated_action_name(raw_animation_name: str) -> str:
    """Blender action name for a raw glTF `animations[].name` from the donor."""
    return f"{raw_animation_name}{_ACTION_NAME_SUFFIX}"


def curated_clip_label(exact_action_name: str) -> str:
    """Prettify an exact donor action name for display.

    Strip a trailing "_Armature", replace underscores with spaces, and
    title-case the remaining words. `exact_action_name` is passed through
    unchanged everywhere else -- this is presentation only.
    """
    stem = exact_action_name
    if stem.endswith(_ACTION_NAME_SUFFIX):
        stem = stem[: -len(_ACTION_NAME_SUFFIX)]
    words = [word for word in stem.replace("_", " ").split(" ") if word]
    return " ".join(word.capitalize() for word in words)


def _clip_stats(document: dict[str, Any], binary: bytes, animation: dict[str, Any]) -> tuple[int, float]:
    """(frame_count, duration_seconds) for one glTF `animations[]` entry.

    frame_count is the largest keyframe count among the animation's
    samplers; duration_seconds is the largest sampler input accessor max
    (glTF accessors declare min/max on their own, so this donor never needs
    the binary chunk decoded -- `_accessor_reader` is only a fallback for an
    accessor that omits the declared max).
    """
    accessors = document.get("accessors", [])
    reader = None
    frame_count = 0
    duration_seconds = 0.0
    for sampler in animation.get("samplers", []):
        input_index = sampler.get("input")
        if input_index is None or input_index >= len(accessors):
            continue
        accessor = accessors[input_index]
        frame_count = max(frame_count, int(accessor.get("count", 0)))
        declared_max = accessor.get("max")
        if declared_max:
            sampler_duration = float(declared_max[0])
        else:
            if reader is None:
                reader = _accessor_reader(document, binary)
            values = reader(input_index)
            sampler_duration = float(values.max()) if len(values) else 0.0
        duration_seconds = max(duration_seconds, sampler_duration)
    return frame_count, duration_seconds


def curated_clips() -> dict[str, Any]:
    """List every playable clip in the curated donor library.

    Reads the donor GLB's glTF JSON `animations[]` directly (no Blender
    needed) and cached in-process -- the donor file is pinned by sha256 in
    `native.curated_sword_donor` and never changes at runtime.
    """
    global _cache
    if _cache is not None:
        return _cache
    donor = native.curated_sword_donor()
    document, binary = _read_glb(donor.path)
    clips: list[dict[str, Any]] = []
    for animation in document.get("animations", []):
        raw_name = animation.get("name")
        if not raw_name:
            continue
        exact_name = curated_action_name(raw_name)
        frame_count, duration_seconds = _clip_stats(document, binary, animation)
        clips.append({
            "animation": exact_name,
            "label": curated_clip_label(exact_name),
            "frames": frame_count,
            "duration_seconds": duration_seconds,
        })
    clips.sort(key=lambda clip: clip["label"])
    _cache = {
        "donor": str(donor.path.relative_to(native.ROOT)),
        "clips": clips,
    }
    return _cache
