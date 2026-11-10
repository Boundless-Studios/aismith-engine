"""Is the fitted skeleton actually left/right symmetric? (pure numpy, no bpy)

`docs/PIPELINE_POLICY.md`'s S3 contract requires left/right joint placement
to sit "within the family's symmetry bands". The only thing measuring
anything called symmetry today is `skeleton_template.conformance_report`'s
`mirrored_left_right` finding, and that is a NAMING check: it asks whether
`upper_arm_l` points more like the template's left arm than like its right
arm. A rig can pass it with both arms named correctly and still have the
right hand 8cm further out than the left -- which is the failure operators
actually see in the viewport, and the reason S3's symmetry check has never
had a geometric number behind it.

Method: mirror every left joint's bind-pose head across the rig's own
sagittal plane and measure how far it lands from its right partner, in
model heights. The plane is defined by the measured left vector (from the
rig's detected facing, not from an assumed +X) through the pelvis, so a
character modelled off-origin or facing an unexpected axis is measured
against its own midline rather than against the world's.

Why bind-pose heads rather than a mesh-vertex comparison: the S3 question
is about where the FITTER put the joints. A mesh that is itself
asymmetric (a character with one pauldron, a tail flicked to one side)
would drag a vertex-based measure red for something S1/S2 owns, and the
blame table's "metric red but the rig is fine -> blame the METRIC" row is
exactly what that produces. Comparing joints to joints keeps the
measurement inside the stage that can fix it.

Three numbers per pair, because they fail independently:
  - `mirror_error_normalized` -- placement. The headline.
  - `length_ratio` -- one limb longer than the other, even when both heads
    are mirrored correctly (the head is shared with the parent, so a
    stretched forearm shows up here first).
  - `direction_angle_degrees` -- orientation. A pair can be placed and
    scaled symmetrically and still have one bone rolled or aimed
    differently, which is what breaks a mirrored animation.

Known limitation: this measures the rig against ITS OWN midline, so a rig
that is uniformly skewed -- every joint shifted a centimetre to the left --
reads as perfectly symmetric. That is deliberate. Absolute midline
placement is `s3.medial_line_distance`'s question, not this one.
"""

from __future__ import annotations

import math

import numpy as np


SCHEMA_VERSION = 1
METHOD = "sagittal-mirror-bind-pose"

# A pair is "outside the band" when its mirrored head lands more than 3% of
# the model's height from its partner. On a 1.8m character that is ~5cm --
# visible in the viewport, well above the fitter's own placement jitter, and
# below the ~8cm errors seen on the mirrored-rig failures this was built for.
SYMMETRY_MIRROR_BAND = 0.03
# Worst offenders reported by name.
MAX_REPORTED_OFFENDERS = 12

_EPS = 1e-12


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def mirror_point(point: np.ndarray, left: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    """Reflect `point` across the sagittal plane (normal `left`, through
    `plane_point`)."""
    offset = np.asarray(point, dtype=np.float64) - np.asarray(plane_point, dtype=np.float64)
    return np.asarray(point, dtype=np.float64) - 2.0 * float(np.dot(offset, left)) * left


def mirror_vector(vector: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Reflect a free vector across the sagittal plane (no plane point: a
    direction has no position to offset)."""
    vector = np.asarray(vector, dtype=np.float64)
    return vector - 2.0 * float(np.dot(vector, left)) * left


def _unmeasurable(caveats: list[str], pairs_skipped: list[str], band: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "measurable": False,
        "method": METHOD,
        "pairs_measured": 0,
        "pairs_skipped": pairs_skipped,
        "pairs": [],
        "worst_offenders": [],
        "mirror_error_p95": None,
        "mirror_error_max": None,
        "pairs_outside_band_count": None,
        "band": band,
        "caveats": caveats,
    }


def unmeasurable_report(caveat: str, *, band: float = SYMMETRY_MIRROR_BAND) -> dict:
    """The `measurable: False` report shape, for callers that cannot even
    assemble the inputs -- no family template, no confident facing, no
    midline joint. Keeps "we could not measure this" one shape everywhere
    rather than two."""
    return _unmeasurable([caveat], [], band)


def measure_bilateral_symmetry(
    *,
    joint_positions: dict[str, np.ndarray],
    joint_directions: dict[str, np.ndarray | None],
    pairs: list[tuple[str, str]],
    left: np.ndarray,
    plane_point: np.ndarray,
    model_height: float,
    band: float = SYMMETRY_MIRROR_BAND,
) -> dict:
    """Per-pair sagittal mirror error, plus aggregates and worst offenders.

    `left` is the rig's measured left direction (the sagittal plane's
    normal); `plane_point` a point on the midline (the pelvis head).
    Distances are reported as fractions of `model_height` so the numbers are
    comparable across characters.
    """
    caveats: list[str] = []
    left_unit = _unit(np.asarray(left, dtype=np.float64))
    if left_unit is None:
        caveats.append("left/right axis is degenerate; no sagittal plane to mirror across")
    if not model_height > 0.0:
        caveats.append("model height is not positive; mirror error cannot be normalized")
    if left_unit is None or not model_height > 0.0:
        return _unmeasurable(caveats, [f"{l}/{r}" for l, r in pairs], band)

    plane_origin = np.asarray(plane_point, dtype=np.float64)
    measured: list[dict] = []
    skipped: list[str] = []
    for left_name, right_name in pairs:
        left_position = joint_positions.get(left_name)
        right_position = joint_positions.get(right_name)
        if left_position is None or right_position is None:
            skipped.append(f"{left_name}/{right_name}")
            continue
        mirrored = mirror_point(
            np.asarray(left_position, dtype=np.float64), left_unit, plane_origin
        )
        error = float(np.linalg.norm(mirrored - np.asarray(right_position, dtype=np.float64)))

        left_direction = joint_directions.get(left_name)
        right_direction = joint_directions.get(right_name)
        left_unit_direction = (
            _unit(np.asarray(left_direction, dtype=np.float64))
            if left_direction is not None
            else None
        )
        right_unit_direction = (
            _unit(np.asarray(right_direction, dtype=np.float64))
            if right_direction is not None
            else None
        )
        if left_unit_direction is not None and right_unit_direction is not None:
            length_ratio = float(
                np.linalg.norm(left_direction) / np.linalg.norm(right_direction)
            )
            dot = float(
                np.clip(
                    np.dot(mirror_vector(left_unit_direction, left_unit), right_unit_direction),
                    -1.0,
                    1.0,
                )
            )
            direction_angle = math.degrees(math.acos(dot))
        else:
            length_ratio = None
            direction_angle = None

        measured.append({
            "left": left_name,
            "right": right_name,
            "mirror_error_normalized": round(error / model_height, 6),
            "length_ratio": round(length_ratio, 6) if length_ratio is not None else None,
            "direction_angle_degrees": (
                round(direction_angle, 4) if direction_angle is not None else None
            ),
        })

    if not measured:
        caveats.append("no left/right pair had both bones present; nothing to compare")
        return _unmeasurable(caveats, skipped, band)

    if skipped:
        caveats.append(
            f"{len(skipped)} of {len(pairs)} template pairs were skipped "
            "because one side is missing from the rig"
        )
    errors = np.asarray([item["mirror_error_normalized"] for item in measured], dtype=np.float64)
    offenders = sorted(
        measured, key=lambda item: item["mirror_error_normalized"], reverse=True
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "measurable": True,
        "method": METHOD,
        "pairs_measured": len(measured),
        "pairs_skipped": skipped,
        "pairs": measured,
        "worst_offenders": offenders[:MAX_REPORTED_OFFENDERS],
        "mirror_error_p95": round(float(np.percentile(errors, 95)), 6),
        "mirror_error_max": round(float(np.max(errors)), 6),
        "pairs_outside_band_count": int(np.count_nonzero(errors > band)),
        "band": band,
        "caveats": caveats,
    }
