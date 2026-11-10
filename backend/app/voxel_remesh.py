"""Pure policy helpers for S2 canonicalization's voxel-remesh backend.

Blender's voxel remesh (`blender_worker.voxel_remesh_canonicalize`) rebuilds
a mesh's surface from a signed-distance field, which makes it watertight and
a single connected component *by construction* -- unlike Instant Meshes'
quad-field solver, which shatters badly-conditioned input (one character's
S1 mesh: 800+ shards at every vertex target tried, 2026-08-30; see
docs/PIPELINE_POLICY.md S2). This module holds the arithmetic that
strategy depends on -- picking a starting voxel size from the source's
bounds, bisecting toward a "comfortably over-tessellated" raw face-count
band, choosing between two out-of-band candidates, and computing the
decimation ratio down to the family's triangle budget -- kept separate from
`blender_worker.py`'s `bpy` calls so it can be unit tested without a running
Blender process (the same split `proxy_rig.py` uses for
`deformation_proxy_decimation_ratio`).
"""
from __future__ import annotations

# Voxels spanning the source mesh's longest bounding-box dimension, for the
# first voxel-size guess.
DEFAULT_VOXELS_ACROSS_LONGEST_AXIS = 120.0

# The raw voxel-remeshed mesh (before decimation) has to land comfortably ABOVE
# the triangle target: decimating down from an over-tessellated, already-
# manifold mesh is safe, and there is no way to "undecimate" an undershoot.
#
# These two multipliers ARE the band that used to be written as the constants
# 150_000-350_000 in `native.py`, divided by the 30k target they were tuned
# against. Expressing them as ratios is what makes the search scale with an
# operator's tier (bead i63ky): before this, an Ultra 150k target would have
# been decimated down from a 150k-350k raw mesh -- a decimate ratio at or above
# 1.0, i.e. no oversampling at all, and a candidate the tier's own band would
# then reject. Ultra now searches for a ~750k-1.75M raw mesh, exactly the same
# 5x-11.7x oversampling Standard has always had.
VOXEL_RAW_FACE_MIN_MULTIPLIER = 5.0  # 150_000 / 30_000
VOXEL_RAW_FACE_MAX_MULTIPLIER = 35.0 / 3.0  # 350_000 / 30_000
# Absolute ceiling on the raw search, independent of the target. A voxel remesh
# is an SDF rebuild plus (worst case) eight bisection iterations, so the raw
# face count is the single biggest driver of canonicalize wall time and Blender
# memory. At the Ultra tier the proportional band (1.75M) sits under this, so
# the ceiling only engages for custom targets past ~171k -- where it still
# leaves the decimator a >10x oversampled source.
VOXEL_RAW_FACE_CEILING = 2_000_000


def voxel_raw_face_band(
    target_faces: int,
    *,
    growth: float = 1.0,
    ceiling: int = VOXEL_RAW_FACE_CEILING,
) -> tuple[int, int]:
    """Raw (pre-decimation) face band to search for, given a triangle target.

    `growth` is the per-retry escalation factor `native.run_canonicalize`
    applies when a decimated candidate still fails the mesh contract: search
    for an even-more-over-tessellated raw mesh rather than loosening the gate.
    Applying it here rather than by multiplying the previous band keeps the
    ceiling honoured on every retry.
    """
    if target_faces <= 0:
        raise ValueError(
            f"voxel_raw_face_band needs a positive triangle target, got {target_faces}"
        )
    if growth <= 0:
        raise ValueError(f"voxel_raw_face_band needs a positive growth factor, got {growth}")
    if ceiling <= 0:
        raise ValueError(f"voxel_raw_face_band needs a positive ceiling, got {ceiling}")
    low = int(round(target_faces * VOXEL_RAW_FACE_MIN_MULTIPLIER * growth))
    high = int(round(target_faces * VOXEL_RAW_FACE_MAX_MULTIPLIER * growth))
    high = min(high, ceiling)
    # The floor never rises above the ceiling-clamped roof, and never drops to
    # or below the target itself -- searching for a raw mesh no denser than the
    # budget would make the decimation step a no-op.
    low = max(target_faces + 1, min(low, high))
    return low, max(low, high)


def initial_voxel_size(
    max_dimension: float,
    voxels_across_longest_axis: float = DEFAULT_VOXELS_ACROSS_LONGEST_AXIS,
) -> float:
    """Starting voxel size: ~120 voxels across the source's longest axis."""
    if max_dimension <= 0:
        raise ValueError("Voxel remesh requires a positive bounding-box dimension")
    if voxels_across_longest_axis <= 0:
        raise ValueError("voxels_across_longest_axis must be positive")
    return max_dimension / voxels_across_longest_axis


def next_voxel_size(
    voxel_size: float,
    raw_faces: int,
    raw_face_min: int,
    raw_face_max: int,
    low_voxel: float | None,
    high_voxel: float | None,
) -> tuple[float, float | None, float | None]:
    """One bisection step toward landing `raw_faces` inside
    `[raw_face_min, raw_face_max]`.

    Returns `(next_voxel_size, next_low_voxel, next_high_voxel)`. Smaller
    voxels produce more faces (finer detail); larger voxels produce fewer.
    The caller stops once `raw_face_min <= raw_faces <= raw_face_max`; this
    function assumes it is being asked precisely because that was not yet
    the case.
    """
    if raw_faces < raw_face_min:
        # Too coarse: shrink the voxel for more detail (more faces).
        next_high = voxel_size if high_voxel is None else min(high_voxel, voxel_size)
        next_size = voxel_size / 2.0 if low_voxel is None else (voxel_size + low_voxel) / 2.0
        return next_size, low_voxel, next_high
    # Too fine: grow the voxel for less detail (fewer faces).
    next_low = voxel_size if low_voxel is None else max(low_voxel, voxel_size)
    next_size = voxel_size * 2.0 if high_voxel is None else (voxel_size + high_voxel) / 2.0
    return next_size, next_low, high_voxel


def voxel_candidate_preferred(candidate_faces: int, incumbent_faces: int, raw_face_min: int) -> bool:
    """Tie-break between two out-of-band voxel remesh attempts.

    Over-the-floor (`raw_faces >= raw_face_min`) always beats under-the-floor,
    since decimation can only remove faces, never invent missing geometry.
    Between two over-the-floor candidates, prefer the leaner one (closer to
    the floor from above -- less for the decimator to remove); between two
    under-the-floor candidates, prefer the denser one (closer to the floor
    from below).
    """
    candidate_over = candidate_faces >= raw_face_min
    incumbent_over = incumbent_faces >= raw_face_min
    if candidate_over != incumbent_over:
        return candidate_over
    if candidate_over:
        return candidate_faces < incumbent_faces
    return candidate_faces > incumbent_faces


def canonicalize_decimate_ratio(raw_faces: int, target_faces: int) -> float:
    """Blender decimation ratio bringing a voxel-remeshed mesh down to budget."""
    if raw_faces <= 0 or target_faces <= 0:
        raise ValueError("Voxel canonicalize decimation requires positive face counts")
    return min(1.0, max(1e-6, target_faces / raw_faces))
