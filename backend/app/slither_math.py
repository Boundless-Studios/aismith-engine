"""Pure math for the `slither-bake` Blender op (no bpy import): a travelling
lateral bend animating a chain-rigged creature (task F3 of the project's
clone-progress plan).

The Blender op (`backend/workers/blender_worker.py::slither_bake`) imports a
chain-rigged GLB -- bones `spine_00..spine_NN` tail to head plus a terminal
`head` bone, this repo's chain-naming convention (see
`backend/app/skeleton_naming.py`) -- and keyframes each bone's LOCAL
rotation about world-up (expressed in that bone's own rest frame; see
`slither_bake`'s `_up_axis_in_bone_local_space`) so the whole body weaves
like a travelling sine wave along its own length. The wave travels head to
tail -- real serpentine locomotion pushes the muscular wave backward along
the body relative to the direction of travel -- and the chain's root bone
(the tail end, `spine_00`, `s == 0`) is anchored: it never absolutely yaws,
it only ever inherits the relative bend of its neighbor (see
`chain_rotations`).

Every bone i sits at a normalized arc position `s_i` along the chain (0 =
tail, 1 = head; see `arc_positions_from_lengths`). The underlying continuous
curve's tangent angle at arc position `s` and normalized clip time `t` (in
`[0, 1)`, one `speed`-scaled wave per unit time) is::

    theta(s, t) = A * sin(2*pi * (s / wavelength + speed * t))

The `+` (rather than `-`) on `speed * t` is what makes a fixed phase value
move toward *decreasing* `s` as `t` increases -- i.e. from head toward tail.
`wavelength` and `speed` are used directly in the caller's units (fraction
of body length, waves per clip); `speed` must be an integer number of waves
per clip or the clip could not loop seamlessly (frame 0 and the wrapped
"frame N" would disagree). `A` is the underlying curve's angular amplitude
in radians; the public `amplitude` setting (a fraction-of-body-length sway
control) is mapped onto it with a fixed `amplitude * pi` scale. That
constant is arbitrary -- the only thing any test here depends on is that
the mapping is *linear* in `amplitude`.

A bone's own LOCAL rotation is the *difference* between the curve's tangent
angle at its own arc position and at its tail-ward neighbor's, so the chain
actually bends relative to its parent bone (the pose looks like a snake),
rather than each bone independently swinging in world space. The root bone
(`s == 0`) is the one exception: its "tail-ward neighbor" is itself, so its
own local rotation is always exactly zero and it stays anchored in the
armature's rest orientation (see `chain_rotations`). `head_stabilize`
dampens the difference multiplicatively per bone (`taper`, linear in `s`)
rather than damping the shared curve itself: at `head_stabilize == 1.0`,
`taper(1.0) == 0.0` and the head bone's own local rotation is therefore
*exactly* zero regardless of how the rest of the body is moving -- which is
what "stabilize the head" should mean operationally. (Damping the curve
itself before differencing would only pull the head bone's rotation toward
its neighbor's value, never to a clean zero.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_EPS = 1e-9


@dataclass(frozen=True)
class SlitherParams:
    """Math-relevant `slither-bake` settings. `frames`, `fps`, and `advance`
    are handled by the Blender op itself and do not affect this module's
    per-bone rotation math."""

    wavelength: float = 1.0
    amplitude: float = 0.12
    speed: float = 1.0
    head_stabilize: float = 0.5

    def __post_init__(self) -> None:
        if self.wavelength <= _EPS:
            raise ValueError("wavelength must be positive")
        if not 0.0 <= self.head_stabilize <= 1.0:
            raise ValueError("head_stabilize must be in [0, 1]")
        if abs(self.speed - round(self.speed)) > 1e-9:
            raise ValueError(
                "speed must be an integer number of waves per clip -- a "
                "fractional speed leaves the curve's phase at t=1 disagreeing "
                "with t=0, so the clip could never loop seamlessly"
            )


def from_settings(settings: dict) -> SlitherParams:
    """Build `SlitherParams` from a `slither-bake` job's raw `settings`
    dict, applying the documented defaults for any missing key."""
    defaults = SlitherParams()
    return SlitherParams(
        wavelength=float(settings.get("wavelength", defaults.wavelength)),
        amplitude=float(settings.get("amplitude", defaults.amplitude)),
        speed=float(settings.get("speed", defaults.speed)),
        head_stabilize=float(settings.get("head_stabilize", defaults.head_stabilize)),
    )


def curve_tangent_angle(s: float, t: float, params: SlitherParams) -> float:
    """The travelling curve's own tangent angle (radians) at arc position
    `s` and clip time `t`, before any per-bone taper is applied. The `+`
    sign on `speed * t` makes the wave travel head (`s=1`) to tail (`s=0`)
    as `t` increases -- see the module docstring."""
    angular_amplitude = params.amplitude * math.pi
    phase = 2.0 * math.pi * (s / params.wavelength + params.speed * t)
    return angular_amplitude * math.sin(phase)


def taper(s: float, head_stabilize: float) -> float:
    """Linear damping of a bone's own local rotation toward the head end.
    `taper(0, x) == 1` (the tail keeps full motion); `taper(1, 1) == 0` (a
    fully stabilized head bone gets exactly zero local rotation)."""
    return 1.0 - head_stabilize * max(0.0, min(1.0, s))


def chain_rotations(s: Sequence[float], t: float, params: SlitherParams) -> list[float]:
    """Per-bone LOCAL rotation angle (radians) about world-up, tail to
    head, for chain bones sitting at arc positions `s` (see
    `arc_positions_from_lengths`) at normalized clip time `t`.

    Bone 0 (the chain root, `s == 0`) is anchored: its "tail-ward neighbor"
    is itself, so its own local rotation is always exactly zero regardless
    of the wave -- an un-anchored root would absolutely yaw the whole
    armature's rest orientation by the curve's full angle every frame,
    which is not a bend, it's a spin. Every other bone i bends relative to
    bone i-1's own (untapered) curve angle -- see the module docstring for
    why taper is applied to the difference, not to the shared curve.
    """
    if not s:
        return []
    angles = [curve_tangent_angle(si, t, params) for si in s]
    rotations: list[float] = []
    previous_angle = angles[0]
    for si, angle in zip(s, angles):
        rotations.append(taper(si, params.head_stabilize) * (angle - previous_angle))
        previous_angle = angle
    return rotations


def arc_positions_from_lengths(lengths: Sequence[float]) -> list[float]:
    """Normalize per-bone lengths (tail to head order) to arc positions
    `s_i` in `[0, 1]`: `s_0 == 0.0` (tail), `s_{-1} == 1.0` (head),
    monotonically increasing in between at each bone's own midpoint."""
    n = len(lengths)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    cumulative_before = 0.0
    midpoints = []
    for length in lengths:
        length = max(0.0, float(length))
        midpoints.append(cumulative_before + length / 2.0)
        cumulative_before += length
    lo, hi = midpoints[0], midpoints[-1]
    span = hi - lo
    if span <= _EPS:
        # Degenerate chain (all-zero or near-equal lengths): fall back to an
        # even spread so the rest of the pipeline still gets a valid,
        # monotonic tail-to-head ordering.
        return [i / (n - 1) for i in range(n)]
    return [(m - lo) / span for m in midpoints]


def bone_rotation_quaternion(
    angle_rad: float, axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> tuple[float, float, float, float]:
    """Unit quaternion `(w, x, y, z)` for a rotation of `angle_rad` about
    `axis`. `axis` defaults to world-up (Z) for tests and simple callers,
    but `slither_bake` always passes world-up *re-expressed in the bone's
    own local rest frame* (see `_up_axis_in_bone_local_space`): a chain
    bone's local Z is not world-up in general (Blender bone roll varies
    rig to rig), so rotating about a hardcoded local axis would corkscrew
    the body wherever roll is non-zero. This is the only place bone
    orientation is touched: a unit quaternion has no scale component, so
    composing per-bone local rotations from this helper alone can never
    change any bone's length -- the chain's total arc length is invariant
    by construction (pure rotation, no scale)."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < _EPS:
        raise ValueError("rotation axis must be non-zero")
    ax, ay, az = ax / norm, ay / norm, az / norm
    half = angle_rad / 2.0
    sin_half = math.sin(half)
    return (math.cos(half), ax * sin_half, ay * sin_half, az * sin_half)
