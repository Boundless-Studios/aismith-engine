"""Pure rotation math for BVH-to-rig pose retargeting (no bpy/mathutils).

`mathutils` is only importable inside Blender's own Python, so the bake-time
transfer logic in ``backend/workers/blender_worker.py`` factors its
quaternion algebra out to this module, which is unit-testable with a plain
interpreter. Every rotation is a unit quaternion in Blender's own
``(w, x, y, z)`` convention and Hamilton-product order (``q1 @ q2`` in
mathutils is ``quat_multiply(q1, q2)`` here), so callers can round-trip
through ``mathutils.Quaternion(quat_multiply(...))`` with no axis remapping.

World-delta retarget formula::

    T_pose_w(b, f) = S_pose_w(s, f) . S_rest_w(s)^-1 . A_b . T_rest_w(b)

``S_rest_w(s)``   source bone rest orientation, world space
``S_pose_w(s, f)``source bone orientation at frame f, world space
``T_rest_w(b)``   target bone rest orientation, world space
``A_b``           minimal rotation taking ``T_rest_w(b)``'s direction onto
                  ``S_rest_w(s)``'s direction (`quat_between_vectors`)

Blender keyframes local *basis* quaternions, not world orientations. The
object's ``matrix_world`` is constant, so it cancels out of both sides of
Blender's armature-space composition rule, leaving the same rule one level
up in true world space::

    pose_w(b) = pose_w(parent) . rest_w(parent)^-1 . rest_w(b) . basis(b)

`basis_from_world` solves that for ``basis(b)`` given a desired world pose;
`propagate_unmapped` gives the world pose of a bone that keeps an identity
local basis (no mapped source drives it directly, e.g. an intermediate
bone). Walking the whole target hierarchy in depth order with these two
halves is what `backend/workers/blender_worker.py::retarget_bvh` does per
frame.
"""
from __future__ import annotations

import math

import numpy as np

Quat = tuple[float, float, float, float]

_EPS = 1e-9


def _as_array(q) -> np.ndarray:
    array = np.asarray(q, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"expected a length-4 (w, x, y, z) quaternion, got shape {array.shape}")
    return array


def quat_identity() -> Quat:
    return (1.0, 0.0, 0.0, 0.0)


def quat_normalize(q) -> Quat:
    array = _as_array(q)
    norm = float(np.linalg.norm(array))
    if norm < _EPS:
        raise ValueError("cannot normalize a zero-length quaternion")
    return tuple((array / norm).tolist())


def quat_multiply(*quats) -> Quat:
    """Hamilton product; ``quat_multiply(a, b)`` applies ``b`` first, then ``a``
    (equivalent to mathutils' ``a @ b``). Any number of quaternions may be
    chained left to right: ``quat_multiply(a, b, c) == a * b * c``."""
    if not quats:
        return quat_identity()
    result = _as_array(quats[0])
    for q in quats[1:]:
        w1, x1, y1, z1 = result
        w2, x2, y2, z2 = _as_array(q)
        result = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    return tuple(result.tolist())


def quat_conjugate(q) -> Quat:
    w, x, y, z = _as_array(q)
    return (w, -x, -y, -z)


def quat_inverse(q) -> Quat:
    array = _as_array(q)
    norm_sq = float(np.dot(array, array))
    if norm_sq < _EPS:
        raise ValueError("cannot invert a zero-length quaternion")
    w, x, y, z = array
    return (w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)


def quat_rotate_vector(q, v) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion: ``v' = q v q*``."""
    qv: Quat = (0.0, *tuple(np.asarray(v, dtype=np.float64).tolist()))
    rotated = quat_multiply(q, qv, quat_conjugate(q))
    return np.array(rotated[1:])


def quat_from_axis_angle(axis, angle_rad: float) -> Quat:
    array = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm < _EPS:
        raise ValueError("rotation axis must be non-zero")
    array = array / norm
    half = angle_rad / 2.0
    return (math.cos(half), *(array * math.sin(half)).tolist())


def quat_between_vectors(v_from, v_to) -> Quat:
    """Minimal-arc rotation that takes ``v_from`` onto ``v_to`` (need not be
    unit length). Returns identity when either vector is zero-length or the
    vectors already point the same way."""
    a = np.asarray(v_from, dtype=np.float64)
    b = np.asarray(v_to, dtype=np.float64)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a < _EPS or norm_b < _EPS:
        return quat_identity()
    a, b = a / norm_a, b / norm_b
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return quat_identity()
    if dot < -1.0 + 1e-9:
        # 180 degrees apart: any axis perpendicular to `a` works.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, fallback)
        axis = axis / np.linalg.norm(axis)
        return (0.0, *axis.tolist())
    axis = np.cross(a, b)
    return quat_normalize(np.array([1.0 + dot, *axis.tolist()]))


def quat_rotation_angle(q) -> float:
    """Total rotation angle of a unit quaternion, in radians, in [0, pi].

    q and -q represent the same rotation, so the canonical (w >= 0) form is
    used before taking acos(w).
    """
    w = quat_normalize(q)[0]
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(w))))


def quat_clamp_angle(q, max_angle_rad: float) -> Quat:
    """Clamp a unit quaternion's rotation angle to at most `max_angle_rad`,
    keeping its rotation axis unchanged. Used to limit how much a rest-
    direction alignment (`quat_between_vectors`) is allowed to rotate a
    torso bone away from its own rest orientation -- SOMA and a target
    rig's bone-drawing conventions can disagree by tens of degrees at the
    spine/neck/head even when the underlying rig is not defective, and a
    full alignment there would visibly pitch the torso every frame."""
    if max_angle_rad < 0:
        raise ValueError("max_angle_rad must be non-negative")
    w, x, y, z = quat_normalize(q)
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    if angle <= max_angle_rad:
        return (w, x, y, z)
    axis = np.array([x, y, z])
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < _EPS:
        return quat_identity()
    axis = axis / axis_norm
    half = max_angle_rad / 2.0
    return (math.cos(half), *(axis * math.sin(half)).tolist())


def vector_angle_deg(a, b) -> float:
    """Angle between two vectors in degrees; 180.0 if either is zero-length."""
    array_a = np.asarray(a, dtype=np.float64)
    array_b = np.asarray(b, dtype=np.float64)
    norm_a, norm_b = float(np.linalg.norm(array_a)), float(np.linalg.norm(array_b))
    if norm_a < _EPS or norm_b < _EPS:
        return 180.0
    cos_angle = float(np.clip(np.dot(array_a, array_b) / (norm_a * norm_b), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def quat_align_frame(t_dir, t_hint, s_dir, s_hint, degenerate_deg: float = 5.0) -> Quat:
    """Two-vector ("frame") rest alignment: the minimal rotation that takes
    the target bone's rest DIRECTION onto the source bone's rest direction
    (the same swing `quat_between_vectors` computes), PLUS a roll about that
    now-shared axis so a secondary "hint" direction lines up too.

    `quat_between_vectors` alone only pins the bone's primary axis; the
    rotation ABOUT that axis is whatever the minimal-arc construction
    happens to produce (see its cross-product implementation) -- physically
    arbitrary, not derived from either rig's own geometry. For a bone whose
    two rigs disagree about that residual roll at rest (e.g. no separate
    clavicle bone, so an auto-rigged upper arm inherits a rest roll that
    doesn't match the donor mocap skeleton's convention), that arbitrary
    roll rides through the whole limb every frame as a CONSTANT twist,
    because the world-delta transfer's per-frame swing correction can only
    ever undo the FRAME's own swing, never a baked-in rest-roll offset.

    The hint (typically a mapped child bone's own rest direction -- a
    forearm's hint for an upper arm, a hand's hint for a forearm) resolves
    that ambiguity: after `t_hint` is carried through the same swing that
    aligns `t_dir` onto `s_dir`, its component perpendicular to the shared
    axis is rotated (about that axis only) onto `s_hint`'s own perpendicular
    component. Both hints are projected relative to the SAME shared axis
    (`s_dir`) so the comparison is apples-to-apples once the swing has
    already brought the two bone directions into agreement.

    Falls back to swing-only (`quat_between_vectors(t_dir, s_dir)`) when
    either hint's perpendicular component is too small relative to its own
    length -- i.e. the hint is (nearly) parallel to its own bone's rest
    direction, e.g. a straight-limbed T-pose rest where a forearm gives no
    roll information about its upper arm. `degenerate_deg` is the minimum
    angle (from the bone's own direction) a hint must clear to be trusted;
    below it, the projected component is dominated by whatever fraction of
    a degree of incidental/authoring noise happens to sit in each rig's
    rest pose, not a meaningful anatomical signal, and callers should
    prefer a hint further down the chain (e.g. a hand instead of a
    forearm) before falling all the way back to swing-only.
    """
    swing = quat_between_vectors(t_dir, s_dir)
    s_axis = np.asarray(s_dir, dtype=np.float64)
    s_axis_norm = float(np.linalg.norm(s_axis))
    if s_axis_norm < _EPS:
        return swing
    s_axis = s_axis / s_axis_norm

    t_hint_arr = np.asarray(t_hint, dtype=np.float64)
    s_hint_arr = np.asarray(s_hint, dtype=np.float64)
    t_hint_norm = float(np.linalg.norm(t_hint_arr))
    s_hint_norm = float(np.linalg.norm(s_hint_arr))
    if t_hint_norm < _EPS or s_hint_norm < _EPS:
        return swing

    t_hint_swung = quat_rotate_vector(swing, t_hint_arr)
    t_hint_perp = t_hint_swung - np.dot(t_hint_swung, s_axis) * s_axis
    s_hint_perp = s_hint_arr - np.dot(s_hint_arr, s_axis) * s_axis

    min_component = math.sin(math.radians(degenerate_deg))
    if (float(np.linalg.norm(t_hint_perp)) < min_component * t_hint_norm
            or float(np.linalg.norm(s_hint_perp)) < min_component * s_hint_norm):
        return swing

    roll = quat_between_vectors(t_hint_perp, s_hint_perp)
    return quat_multiply(roll, swing)


def desired_target_world_pose(source_pose_w, source_rest_w, align, target_rest_w) -> Quat:
    """``T_pose_w = S_pose_w . S_rest_w^-1 . A_b . T_rest_w`` (task A2)."""
    return quat_multiply(source_pose_w, quat_inverse(source_rest_w), align, target_rest_w)


def propagate_unmapped(parent_pose_w, parent_rest_w, bone_rest_w) -> Quat:
    """World pose of a bone with an identity local basis, i.e. no mapped
    source drives it: ``pose_w(b) = pose_w(parent) . rest_w(parent)^-1 . rest_w(b)``."""
    return quat_multiply(parent_pose_w, quat_inverse(parent_rest_w), bone_rest_w)


def basis_from_world(bone_rest_w, parent_rest_w, parent_pose_w, bone_pose_w) -> Quat:
    """Solve Blender's composition rule
    ``pose(b) = pose(parent) . rest(parent)^-1 . rest(b) . basis(b)``
    for ``basis(b)`` given the desired world pose orientation of the bone."""
    return quat_multiply(
        quat_inverse(bone_rest_w), parent_rest_w, quat_inverse(parent_pose_w), bone_pose_w,
    )


# --- Frame range (K17) ---------------------------------------------------
#
# Not rotation math, but kept in this module for the same reason as
# everything else here: it backs a bpy call in blender_worker.py
# (normalize_bvh_axes's bpy.ops.export_anim.bvh) and needs to be
# unit-testable with a plain interpreter, which a module that `import bpy`s
# at module scope cannot be.


def bvh_export_frame_range(action_frame_range: tuple[float, float]) -> tuple[int, int]:
    """The integer (frame_start, frame_end) an export should use for an
    action spanning `action_frame_range`, inclusive on both ends.

    bpy.ops.export_anim.bvh exports whatever bpy.context.scene.frame_start/
    frame_end currently are, not the action's own range -- normalize_bvh_axes
    used to leave the scene's frame_end at Blender's untouched default (250)
    after importing a Kimodo BVH, so a real, shorter action (Kimodo actually
    honoring a 2s / ~60-frame request) got exported padded out to 250 frames,
    with the action's last real frame held constant the rest of the way
    (confirmed on a real clip: rows 60-250 were byte-identical). Floors the
    start and ceils the end so a fractional action range (e.g. a bake that
    starts mid-frame) is never truncated -- same rounding
    _bake_motion_to_fitted_armature already uses for the identical purpose.
    """
    start, end = action_frame_range
    return int(math.floor(start)), int(math.ceil(end))
