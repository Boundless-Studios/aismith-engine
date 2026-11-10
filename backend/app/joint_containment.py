"""Are the fitted joints actually inside the mesh? (pure numpy, no bpy)

`docs/PIPELINE_POLICY.md`'s S3 contract requires every placed joint to sit
inside the mesh volume, and `pipeline_policy`'s `s3.joints_inside_mesh` check
reports "not-evaluable" because nothing measures it. `rig_quality` already
walks the GLB's skinned primitives in world space, so the triangle soup and
the bind-pose joint positions are both in hand; this module turns them into a
containment fraction plus named worst offenders.

Method: a joint is "inside" when the mesh blocks it along BOTH halves of
every probe axis -- five fixed generic directions, ten half-rays, each
resolved with a vectorized Moller-Trumbore intersector.

The obvious alternative, ray-parity ("odd crossings = inside"), was tried
first and is wrong for this pipeline. Parity answers "is this point in solid
material", and the meshes coming out of S1/S2 are frequently hollow shells:
measured live on 2026-09-01, Mary's rigged mesh
(`8c3a3ba7b2114085bf423e4545b0f4a6`) is watertight, winding-consistent and
genus 1, but encloses a volume of 0.004 against a 0.224 bounding box -- a
~0.005-thick wall. Every joint sits in the cavity, so parity called all 21 of
them "outside" while the rig is in fact fitted correctly. The S3 contract
means "the joint is inside the body", not "the joint is embedded in the
surface's wall thickness", and the blocking test answers that question for a
solid and a shell alike.

Known limitation of the blocking test: it is not exact for concavities. A
joint parked in a deep pocket -- between an arm and the torso, say -- can be
blocked along all ten half-rays and read as inside. It is deliberately
tuned to catch the failure S3 cares about (a joint escaping the silhouette
entirely, which leaves at least one half-ray unblocked) rather than to be a
general point-in-polyhedron oracle.

The reported distance for an outside joint is the distance to the nearest
mesh VERTEX, not to the nearest surface point: an over-estimate on a coarse
mesh, but enough to rank offenders, and it avoids building a BVH.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SCHEMA_VERSION = 1

# Fixed, deterministic probe axes, deliberately generic: no axis alignment, no
# equal components, no simple ratios between components. That matters more
# than it sounds -- a "nice" direction like (1,1,1)/sqrt(3) leaves an
# axis-aligned box exactly through a corner and is counted by all six
# triangles meeting there, and (1,-3,1) leaves exactly along a quad's
# diagonal and is counted twice. Only the sign-free notion of "is this axis
# blocked" is used, so those degeneracies inflate a count that is then only
# compared against zero.
_PROBE_DIRECTIONS = np.asarray([
    (0.3137851, 0.5126347, 0.7990261),
    (-0.6219733, 0.2833107, 0.7302941),
    (0.4581337, -0.7162719, 0.5261103),
    (0.8143279, 0.4409631, -0.3771482),
    (-0.2694513, -0.5837211, -0.7659374),
], dtype=np.float64)
_PROBE_DIRECTIONS /= np.linalg.norm(_PROBE_DIRECTIONS, axis=1, keepdims=True)

_EPS = 1e-12
# Above this triangle count the soup is strided down before the parity test,
# bounding the cost of the O(joints x triangles) sweep. A 60-joint rig against
# 120k triangles is ~7M ray-triangle tests per direction, which is still well
# under a second in numpy.
MAX_PROBE_TRIANGLES = 120_000
# Worst offenders reported by name.
MAX_REPORTED_OFFENDERS = 12


def _axis_blocked(origin: np.ndarray, direction: np.ndarray, vertex0, edge1, edge2) -> bool:
    """Does the mesh lie on BOTH sides of `origin` along `direction`?

    Vectorized Moller-Trumbore over the whole triangle soup at once. `t` is
    signed, so one pass answers for both half-rays.
    """
    pvec = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(determinant) < _EPS
    safe = np.where(parallel, 1.0, determinant)
    inverse = 1.0 / safe
    tvec = origin - vertex0
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inverse
    t = np.einsum("ij,ij->i", edge2, qvec) * inverse
    hit = (
        ~parallel
        & (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
    )
    return bool(np.any(hit & (t > _EPS)) and np.any(hit & (t < -_EPS)))


def points_inside_mesh(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Boolean "is inside" per point: every probe axis blocked both ways."""
    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangle_array = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(point_array) == 0:
        return np.zeros(0, dtype=bool)
    if len(triangle_array) == 0:
        return np.zeros(len(point_array), dtype=bool)
    if len(triangle_array) > MAX_PROBE_TRIANGLES:
        stride = int(np.ceil(len(triangle_array) / MAX_PROBE_TRIANGLES))
        triangle_array = triangle_array[::stride]
    vertex0 = triangle_array[:, 0]
    edge1 = triangle_array[:, 1] - vertex0
    edge2 = triangle_array[:, 2] - vertex0
    inside = np.zeros(len(point_array), dtype=bool)
    for index, origin in enumerate(point_array):
        inside[index] = all(
            _axis_blocked(origin, direction, vertex0, edge1, edge2)
            for direction in _PROBE_DIRECTIONS
        )
    return inside


def enclosed_volume(triangles: np.ndarray) -> float:
    """Signed volume enclosed by the triangle soup (divergence theorem).

    Recorded alongside the containment fraction because it is the fact that
    explains the method: a value near zero relative to the bounding box means
    the mesh is a hollow shell, and any "is this point in solid material"
    test would then call a correctly fitted rig's joints outside.
    """
    triangle_array = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(triangle_array) == 0:
        return 0.0
    first, second, third = triangle_array[:, 0], triangle_array[:, 1], triangle_array[:, 2]
    return float(abs(np.sum(np.einsum("ij,ij->i", first, np.cross(second, third))) / 6.0))


def _nearest_vertex_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    if len(vertices) == 0:
        return np.full(len(points), np.inf)
    return np.asarray([
        float(np.min(np.linalg.norm(vertices - point, axis=1)))
        for point in points
    ])


def measure_joint_containment(
    joint_names: Sequence[str],
    joint_positions: np.ndarray,
    triangles: np.ndarray,
    *,
    model_height: float = 0.0,
) -> dict:
    """Fraction of joint heads inside the mesh volume, plus worst offenders."""
    names = list(joint_names)
    positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1, 3)
    triangle_array = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if len(names) != len(positions):
        raise ValueError("joint_names and joint_positions must have the same length")
    if not names or len(triangle_array) == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "measurable": False,
            "reason": "no joints" if not names else "no skinned triangles",
            "measured_joints": len(names),
            "triangles": int(len(triangle_array)),
        }

    inside = points_inside_mesh(positions, triangle_array)
    outside_indices = np.flatnonzero(~inside)
    vertices = np.unique(triangle_array.reshape(-1, 3), axis=0)
    extents = np.ptp(vertices, axis=0)
    volume = enclosed_volume(triangle_array)
    box_volume = float(np.prod(extents))
    distances = (
        _nearest_vertex_distance(positions[outside_indices], vertices)
        if len(outside_indices)
        else np.zeros(0)
    )
    scale = max(float(model_height), 1e-12)
    offenders = sorted(
        (
            {
                "joint": names[int(index)],
                "distance_to_surface": round(float(distance), 6),
                "normalized_distance": round(float(distance) / scale, 6),
            }
            for index, distance in zip(outside_indices, distances)
        ),
        key=lambda item: item["distance_to_surface"],
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "measurable": True,
        "method": f"ray-blocking-{len(_PROBE_DIRECTIONS)}-axes",
        "measured_joints": len(names),
        "inside_joints": int(np.count_nonzero(inside)),
        "joints_outside_mesh_count": int(len(outside_indices)),
        "joints_inside_mesh_fraction": round(float(np.mean(inside)), 6),
        "triangles": int(len(triangle_array)),
        "enclosed_volume": round(volume, 8),
        "bounding_box_volume": round(box_volume, 8),
        # Near zero means the mesh is a hollow shell rather than a solid --
        # the fact that rules out a "point in solid material" test here.
        "enclosed_volume_ratio": round(volume / box_volume, 6) if box_volume > 0 else 0.0,
        "worst_offenders": offenders[:MAX_REPORTED_OFFENDERS],
        "distance_caveat": (
            "distance is to the nearest mesh vertex, not the nearest surface "
            "point -- an over-estimate on a coarse mesh, used only for ranking"
        ),
        "method_caveat": (
            "a joint is inside when the mesh blocks both halves of all five "
            "probe axes; a joint parked in a deep concavity can read as inside"
        ),
    }
