"""Cheap S2 geometry-contract measurements (pure numpy, no bpy).

`docs/PIPELINE_POLICY.md`'s S2 contract asks three questions the pipeline
already half-answers as assertions ("raise if wrong") but never records as a
number: is the mesh UV-unwrapped, are its object transforms identity/uniform,
and did canonicalization preserve the source's axis profile. `pipeline_policy`
reports all three as "not-evaluable" purely because nothing writes them down.

The Blender-side collection lives in
`backend/workers/blender_worker.normalize_floor` (the last worker step that
touches the final canonical GLB, so it measures exactly the shipped artifact);
the math here is numpy so it is unit-testable without Blender.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SCHEMA_VERSION = 1

# Total UV area below which a "UV layer" is a placeholder rather than a real
# unwrap (every loop collapsed onto one point).
MIN_UV_TOTAL_AREA = 1e-9
# A face is degenerate in UV space below this area; Blender's smart-project
# leaves a handful on near-zero-area triangles, so a real unwrap is judged by
# the fraction of faces that carry area, not by "zero degenerate faces".
MIN_UV_FACE_AREA = 1e-12
# At least this fraction of faces must carry UV area for the mesh to count as
# unwrapped. Smart-project on a clean voxel-remeshed mesh lands at ~1.0.
MIN_UV_NON_DEGENERATE_FACE_FRACTION = 0.99

# Object scale is "uniform" when its three components agree to this relative
# tolerance, and "identity" when they are additionally all 1 to this tolerance.
SCALE_TOLERANCE = 1e-4


def uv_unwrap_stats(uv_triangles: np.ndarray) -> dict:
    """Measure a UV unwrap from per-triangle UV coordinates.

    `uv_triangles` is an (n_faces, 3, 2) array of UV coordinates. An empty
    array means the mesh carries no UV layer at all.
    """
    triangles = np.asarray(uv_triangles, dtype=np.float64).reshape(-1, 3, 2)
    face_count = len(triangles)
    if face_count == 0:
        return {
            "uv_unwrapped": False,
            "uv_faces": 0,
            "uv_total_area": 0.0,
            "uv_non_degenerate_face_fraction": 0.0,
            "uv_degenerate_faces": 0,
        }
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    areas = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) * 0.5
    total_area = float(np.sum(areas))
    non_degenerate = int(np.count_nonzero(areas > MIN_UV_FACE_AREA))
    fraction = non_degenerate / face_count
    return {
        "uv_unwrapped": bool(
            total_area > MIN_UV_TOTAL_AREA
            and fraction >= MIN_UV_NON_DEGENERATE_FACE_FRACTION
        ),
        "uv_faces": face_count,
        "uv_total_area": round(total_area, 8),
        "uv_non_degenerate_face_fraction": round(fraction, 6),
        "uv_degenerate_faces": face_count - non_degenerate,
    }


def scale_uniformity(scales: Sequence[Sequence[float]]) -> dict:
    """Measure per-object scale uniformity from a list of (x, y, z) scales.

    "Uniform" means every object scales its three axes equally; "identity"
    additionally requires that factor to be 1, which is what a rigger-ready
    canonical mesh must have (the axis conversion baked into vertices, not
    left in a node transform).
    """
    matrix = np.asarray(scales, dtype=np.float64).reshape(-1, 3)
    if len(matrix) == 0:
        return {
            "uniform_scale_ok": False,
            "identity_scale": False,
            "objects": 0,
            "max_scale_anisotropy": 0.0,
            "max_scale_deviation": 0.0,
        }
    magnitudes = np.abs(matrix)
    anisotropy = float(np.max(np.ptp(magnitudes, axis=1) / np.maximum(np.max(magnitudes, axis=1), 1e-12)))
    deviation = float(np.max(np.abs(magnitudes - 1.0)))
    return {
        "uniform_scale_ok": bool(anisotropy <= SCALE_TOLERANCE),
        "identity_scale": bool(deviation <= SCALE_TOLERANCE),
        "objects": int(len(matrix)),
        "max_scale_anisotropy": round(anisotropy, 8),
        "max_scale_deviation": round(deviation, 8),
    }


def extent_profile(extents: Sequence[float]) -> list[float]:
    """Per-axis extents normalized by the largest axis (scale-free shape)."""
    values = [float(value) for value in extents]
    largest = max(values) or 1.0
    return [round(value / largest, 6) for value in values]
