"""Motion-quality metrics for a posed animation clip (pure numpy, no bpy).

Ported from the 2026-08-31 blame-split diagnostic that identified "the baked
clip goes dead after frame ~25 of 100" as an S5 animation-bake defect rather
than an S4 skinning one: the donor BVH and the curated source both measured a
`static_tail_fraction` of ~0.05, while the baked result measured 0.76. Edge
stretch -- the only thing `deformation_review` measured before this module --
was identical across all four, so the existing gate could not see the defect
at all.

The Blender-side sampling (posing frames, reading world-space bone head/tail
positions) lives in `backend/workers/blender_worker.deformation_diagnostics`;
everything here is numpy so every number is unit-testable without Blender.

Metrics
-------
``joint_motion_coverage``
    Fraction of length-bearing bones whose world-space direction (head->tail)
    departs by more than `JOINT_MOTION_THRESHOLD_DEGREES` from its first-frame
    value at some point in the clip. A stick-figure / dead-rig clip measures
    near zero; every healthy clip in the 08-31 corpus measured >= 0.98.

``foot_slide_max`` / ``foot_slide_mean``
    Horizontal drift of the stance foot, as a fraction of skeleton height (so
    BVH units and glTF metres cancel). "Stance foot" is the lowest foot bone
    on each frame. KNOWN CAVEAT: on a genuinely stepping clip the lowest-foot
    heuristic charges real locomotion to slide, so these numbers OVERCOUNT
    there and must not be gated on blind -- `stance_frames_fraction` and
    `stance_switch_count` are recorded alongside precisely so a future gate
    can be calibrated per motion family. `foot_slide_*_in_contact` is the
    contact-filtered pair; read it with `contact_stance_episodes`, which is 0
    when no stance episode qualified as ground contact and the 0.0 slide
    therefore means "no evidence", not "planted".

``static_tail_fraction``
    Longest suffix of the clip (by sampled-frame transitions) during which no
    bone turns faster than `STATIC_FRAME_THRESHOLD_DEGREES_PER_FRAME`. This is
    the number that caught the 08-31 defect.

``root_travel_path`` / ``root_travel_net``
    Total horizontal path length and net start->end displacement of the root
    bone, as fractions of skeleton height. Feeds a future root-motion-policy
    check; recorded without a threshold here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1

# A bone counts as "moved" once its world direction departs this far from its
# first-frame direction. 5 degrees is the reference diagnostic's threshold:
# large enough to ignore float noise and rounding in a baked clip, small
# enough that a bone carrying any real animation clears it.
JOINT_MOTION_THRESHOLD_DEGREES = 5.0

# Per-single-frame angular velocity below which a bone is considered still.
STATIC_FRAME_THRESHOLD_DEGREES_PER_FRAME = 0.5

# A foot is "in contact" when it sits within this fraction of skeleton height
# of the clip's own lowest foot position. Used only for the contact-filtered
# slide numbers and `stance_frames_fraction`; the unfiltered numbers keep the
# reference diagnostic's semantics so the 08-31 corpus stays comparable.
STANCE_CONTACT_BAND_FRACTION = 0.05

# Bones with a mean length under this (in scene units, relative to skeleton
# height) carry no meaningful direction -- leaf/zero-length helper bones.
_MIN_BONE_LENGTH = 1e-6

_FOOT_BONE_PATTERN = re.compile(r"foot|ball")
# Checked in priority order: a specific hips/pelvis bone is preferred over a
# generic "root" wrapper joint that some rigs keep static as a retarget anchor.
_ROOT_BONE_PATTERNS = ("hips", "pelvis", "root")
_HEAD_BONE_PATTERNS = ("head",)


def find_bone(names: Sequence[str], patterns: Sequence[str]) -> str | None:
    """First bone whose lowercased name contains one of `patterns`.

    Patterns are honored in order, so an earlier pattern always beats a later
    one regardless of the bone ordering.
    """
    for pattern in patterns:
        for name in names:
            if pattern in name.lower():
                return name
    return None


def foot_bone_names(names: Sequence[str]) -> list[str]:
    """Bones that look like feet (foot/ball, including toe and leaf joints)."""
    return [name for name in names if _FOOT_BONE_PATTERN.search(name.lower())]


def root_bone_name(names: Sequence[str]) -> str | None:
    return find_bone(names, _ROOT_BONE_PATTERNS)


def head_bone_name(names: Sequence[str]) -> str | None:
    return find_bone(names, _HEAD_BONE_PATTERNS)


def resolve_vertical_axis(
    head_position: np.ndarray | None,
    foot_positions: Sequence[np.ndarray],
) -> int:
    """Axis index along which the head sits furthest from the feet.

    Determined empirically rather than assumed, because a BVH source, a
    Blender scene and a glTF export do not agree on which axis is up. Falls
    back to Blender's native Z when the rig has no head or no feet.
    """
    if head_position is None or not len(foot_positions):
        return 2
    foot_mean = np.mean(np.asarray(foot_positions, dtype=np.float64), axis=0)
    return int(np.argmax(np.abs(np.asarray(head_position, dtype=np.float64) - foot_mean)))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def angle_between_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(np.dot(_unit(first), _unit(second)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _direction_angles_from_first(directions: np.ndarray) -> np.ndarray:
    """Angle (degrees) of every row of `directions` against row 0."""
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    units = directions / np.maximum(lengths, 1e-12)
    dots = np.clip(units @ units[0], -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _consecutive_angles(directions: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each consecutive pair of rows."""
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    units = directions / np.maximum(lengths, 1e-12)
    dots = np.clip(np.einsum("ij,ij->i", units[:-1], units[1:]), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def compute_motion_metrics(
    frames: Sequence[int],
    heads: Mapping[str, np.ndarray],
    tails: Mapping[str, np.ndarray],
    *,
    synthetic: bool = False,
    source_action: str | None = None,
) -> dict:
    """Measure motion quality from sampled world-space bone poses.

    `heads` / `tails` map bone name -> an (n_frames, 3) array of world
    positions, in the same order as `frames`.
    """
    frame_list = [int(frame) for frame in frames]
    bone_names = list(heads)
    frame_count = len(frame_list)
    if frame_count < 2 or not bone_names:
        return {
            "schema_version": SCHEMA_VERSION,
            "measurable": False,
            "reason": (
                "fewer than two sampled frames" if frame_count < 2 else "no bones sampled"
            ),
            "sampled_frames": frame_count,
            "synthetic": bool(synthetic),
            "source_action": source_action,
        }

    head_array = {name: np.asarray(heads[name], dtype=np.float64) for name in bone_names}
    tail_array = {name: np.asarray(tails[name], dtype=np.float64) for name in bone_names}

    feet = foot_bone_names(bone_names)
    root = root_bone_name(bone_names) or bone_names[0]
    head_bone = head_bone_name(bone_names)
    vertical_axis = resolve_vertical_axis(
        head_array[head_bone][0] if head_bone else None,
        [head_array[name][0] for name in feet],
    )
    horizontal_axes = [axis for axis in (0, 1, 2) if axis != vertical_axis]

    first_frame_points = np.concatenate(
        [np.stack([head_array[name][0] for name in bone_names]),
         np.stack([tail_array[name][0] for name in bone_names])],
        axis=0,
    )
    character_height = max(float(np.ptp(first_frame_points[:, vertical_axis])), 1e-9)

    # --- joint motion coverage -------------------------------------------
    moved_bones: list[str] = []
    measured_bones = 0
    per_bone_max_delta: dict[str, float] = {}
    for name in bone_names:
        directions = tail_array[name] - head_array[name]
        if float(np.mean(np.linalg.norm(directions, axis=1))) < _MIN_BONE_LENGTH:
            continue  # zero-length / leaf bone: no meaningful direction
        measured_bones += 1
        max_delta = float(np.max(_direction_angles_from_first(directions)))
        per_bone_max_delta[name] = max_delta
        if max_delta > JOINT_MOTION_THRESHOLD_DEGREES:
            moved_bones.append(name)
    coverage = len(moved_bones) / measured_bones if measured_bones else 0.0

    # --- static tail ------------------------------------------------------
    frame_gaps = np.maximum(np.diff(np.asarray(frame_list, dtype=np.float64)), 1.0)
    per_transition_max = np.zeros(frame_count - 1, dtype=np.float64)
    for name in bone_names:
        directions = tail_array[name] - head_array[name]
        lengths = np.linalg.norm(directions, axis=1)
        if float(np.mean(lengths)) < _MIN_BONE_LENGTH:
            continue
        velocities = _consecutive_angles(directions) / frame_gaps
        # A bone that collapses to zero length mid-clip yields a meaningless
        # angle; treat those transitions as still rather than as motion.
        degenerate = (lengths[:-1] < _MIN_BONE_LENGTH) | (lengths[1:] < _MIN_BONE_LENGTH)
        velocities = np.where(degenerate, 0.0, velocities)
        per_transition_max = np.maximum(per_transition_max, velocities)

    still = per_transition_max < STATIC_FRAME_THRESHOLD_DEGREES_PER_FRAME
    static_tail_frames = 0
    for value in reversed(still.tolist()):
        if not value:
            break
        static_tail_frames += 1
    static_tail_fraction = static_tail_frames / len(still) if len(still) else 0.0

    # --- root travel ------------------------------------------------------
    root_horizontal = head_array[root][:, horizontal_axes]
    path_length = float(np.sum(np.linalg.norm(np.diff(root_horizontal, axis=0), axis=1)))
    net_displacement = float(np.linalg.norm(root_horizontal[-1] - root_horizontal[0]))

    # --- foot slide -------------------------------------------------------
    episodes: list[dict] = []
    contact_episodes: list[dict] = []
    stance_switches = 0
    contact_frames = 0
    if feet:
        foot_heights = np.stack([head_array[name][:, vertical_axis] for name in feet], axis=1)
        stance_index = np.argmin(foot_heights, axis=1)
        stance_height = foot_heights[np.arange(frame_count), stance_index]
        contact_band = float(np.min(stance_height)) + STANCE_CONTACT_BAND_FRACTION * character_height
        in_contact = stance_height <= contact_band
        contact_frames = int(np.count_nonzero(in_contact))
        stance_switches = int(np.count_nonzero(np.diff(stance_index) != 0))

        def _episodes(mask: np.ndarray | None) -> list[dict]:
            found: list[dict] = []
            start = 0
            for index in range(1, frame_count + 1):
                boundary = (
                    index == frame_count
                    or stance_index[index] != stance_index[start]
                    or (mask is not None and mask[index] != mask[start])
                )
                if not boundary:
                    continue
                if index - start >= 2 and (mask is None or bool(mask[start])):
                    name = feet[int(stance_index[start])]
                    points = head_array[name][start:index][:, horizontal_axes]
                    drift = float(np.max(np.linalg.norm(points - points[0], axis=1)))
                    found.append({
                        "foot": name,
                        "frames": [frame_list[start], frame_list[index - 1]],
                        "drift": round(drift / character_height, 6),
                    })
                start = index
            return found

        episodes = _episodes(None)
        contact_episodes = _episodes(in_contact)

    def _slide(items: list[dict]) -> tuple[float, float]:
        drifts = [item["drift"] for item in items]
        if not drifts:
            return 0.0, 0.0
        return float(max(drifts)), float(np.mean(drifts))

    slide_max, slide_mean = _slide(episodes)
    contact_slide_max, contact_slide_mean = _slide(contact_episodes)

    worst = sorted(per_bone_max_delta.items(), key=lambda item: item[1])[:8]
    return {
        "schema_version": SCHEMA_VERSION,
        "measurable": True,
        "synthetic": bool(synthetic),
        "source_action": source_action,
        "sampled_frames": frame_count,
        "frame_range": [frame_list[0], frame_list[-1]],
        "vertical_axis": vertical_axis,
        "character_height": round(character_height, 6),
        "root_bone": root,
        "head_bone": head_bone,
        "foot_bones": feet,
        "joint_motion_coverage": round(coverage, 6),
        "moved_bones": len(moved_bones),
        "measured_bones": measured_bones,
        "least_moved_bones": [
            {"bone": name, "max_direction_change_degrees": round(value, 4)}
            for name, value in worst
        ],
        "foot_slide_max": round(slide_max, 6),
        "foot_slide_mean": round(slide_mean, 6),
        "foot_slide_max_in_contact": round(contact_slide_max, 6),
        "foot_slide_mean_in_contact": round(contact_slide_mean, 6),
        "stance_frames_fraction": round(contact_frames / frame_count, 6),
        "stance_switch_count": stance_switches,
        "stance_episodes": episodes,
        # How many stance episodes actually qualified as ground contact. The
        # contact-filtered slide figures are 0.0 both for a perfectly planted
        # foot and for a clip where no episode qualified (an airborne roll);
        # without this count a consumer cannot tell "measured zero" from "no
        # evidence", and would read the second as a pass.
        "contact_stance_episodes": len(contact_episodes),
        "static_tail_fraction": round(static_tail_fraction, 6),
        "static_tail_frames": static_tail_frames,
        "measured_transitions": int(len(still)),
        "root_travel_path": round(path_length / character_height, 6),
        "root_travel_net": round(net_displacement / character_height, 6),
        "foot_slide_caveat": (
            "stance foot is the lowest foot bone per frame; on a genuinely "
            "stepping clip this charges real locomotion to slide. Calibrate a "
            "gate against stance_frames_fraction / stance_switch_count."
        ),
    }
