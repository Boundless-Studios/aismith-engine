import math
import unittest

from backend.app.slither_math import (
    SlitherParams,
    arc_positions_from_lengths,
    bone_rotation_quaternion,
    chain_rotations,
    curve_tangent_angle,
    from_settings,
    taper,
)


class SlitherParamsTests(unittest.TestCase):
    def test_defaults_match_the_documented_settings(self):
        params = from_settings({})
        self.assertEqual(params.wavelength, 1.0)
        self.assertEqual(params.amplitude, 0.12)
        self.assertEqual(params.speed, 1.0)
        self.assertEqual(params.head_stabilize, 0.5)

    def test_from_settings_overrides_only_supplied_keys(self):
        params = from_settings({"amplitude": 0.3, "speed": 2.0})
        self.assertEqual(params.amplitude, 0.3)
        self.assertEqual(params.speed, 2.0)
        self.assertEqual(params.wavelength, 1.0)
        self.assertEqual(params.head_stabilize, 0.5)

    def test_rejects_non_positive_wavelength(self):
        with self.assertRaises(ValueError):
            SlitherParams(wavelength=0.0)

    def test_rejects_head_stabilize_out_of_range(self):
        with self.assertRaises(ValueError):
            SlitherParams(head_stabilize=1.5)
        with self.assertRaises(ValueError):
            SlitherParams(head_stabilize=-0.1)

    def test_rejects_non_integer_speed(self):
        # A fractional speed leaves the curve's phase at t=1 disagreeing
        # with t=0, so the clip could never loop seamlessly (review F3-3).
        with self.assertRaises(ValueError):
            SlitherParams(speed=1.5)
        with self.assertRaises(ValueError):
            SlitherParams(speed=0.1)

    def test_accepts_integer_valued_speed_including_zero_and_negative(self):
        for speed in (0.0, 1.0, -1.0, 3.0, -2.0):
            SlitherParams(speed=speed)  # must not raise


class ChainRotationsTests(unittest.TestCase):
    """The five behaviors the slither-bake op requires (task F3 of the
    project's clone-progress plan)."""

    def setUp(self):
        # 9 spine bones + 1 head bone, evenly spread tail (0.0) to head (1.0).
        self.s = [i / 9.0 for i in range(10)]

    def test_seamless_loop_frame_zero_equals_frame_n(self):
        params = SlitherParams(wavelength=0.4, amplitude=0.25, speed=1.0, head_stabilize=0.3)
        start = chain_rotations(self.s, 0.0, params)
        end = chain_rotations(self.s, 1.0, params)
        for a, b in zip(start, end):
            self.assertAlmostEqual(a, b, places=9)

    def test_seamless_loop_holds_for_multi_cycle_speed(self):
        # speed is "waves per clip" -- any integer speed must still loop.
        params = SlitherParams(wavelength=0.3, amplitude=0.2, speed=3.0, head_stabilize=0.0)
        start = chain_rotations(self.s, 0.0, params)
        end = chain_rotations(self.s, 1.0, params)
        for a, b in zip(start, end):
            self.assertAlmostEqual(a, b, places=9)

    def test_zero_amplitude_gives_all_zeros(self):
        params = SlitherParams(amplitude=0.0)
        for t in (0.0, 0.2, 0.5, 0.9):
            rotations = chain_rotations(self.s, t, params)
            self.assertTrue(all(r == 0.0 for r in rotations))

    def test_amplitude_scales_linearly(self):
        base = SlitherParams(amplitude=0.1, wavelength=0.6, speed=1.0, head_stabilize=0.4)
        doubled = SlitherParams(amplitude=0.2, wavelength=0.6, speed=1.0, head_stabilize=0.4)
        r1 = chain_rotations(self.s, 0.37, base)
        r2 = chain_rotations(self.s, 0.37, doubled)
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(b, 2.0 * a, places=9)

    def test_head_stabilize_one_zeroes_the_head_bone_rotation(self):
        params = SlitherParams(amplitude=0.3, wavelength=0.5, speed=1.0, head_stabilize=1.0)
        for t in (0.0, 0.13, 0.5, 0.81):
            rotations = chain_rotations(self.s, t, params)
            self.assertAlmostEqual(rotations[-1], 0.0, places=9)

    def test_head_stabilize_zero_does_not_special_case_the_head(self):
        params_stable = SlitherParams(amplitude=0.3, wavelength=0.5, speed=1.0, head_stabilize=1.0)
        params_unstable = SlitherParams(amplitude=0.3, wavelength=0.5, speed=1.0, head_stabilize=0.0)
        stable = chain_rotations(self.s, 0.2, params_stable)
        unstable = chain_rotations(self.s, 0.2, params_unstable)
        self.assertNotAlmostEqual(unstable[-1], 0.0, places=6)
        self.assertAlmostEqual(stable[-1], 0.0, places=9)

    def test_phase_increases_with_s_at_fixed_time_zero(self):
        # At t=0, theta(s, 0) = A*sin(2*pi*s/wavelength) has a phase
        # strictly increasing in s (positive coefficient on s) -- a
        # quarter-wavelength window from the tail is a clean monotonic ramp.
        # (At t=0 the +speed*t term vanishes, so this holds regardless of
        # which direction the wave travels over time -- see
        # test_wave_travels_head_to_tail_over_time for that.)
        params = SlitherParams(wavelength=1.0, amplitude=0.5, speed=1.0, head_stabilize=0.0)
        quarter = params.wavelength / 4.0
        samples = [i * quarter / 10.0 for i in range(11)]
        angles = [curve_tangent_angle(s, 0.0, params) for s in samples]
        for earlier, later in zip(angles, angles[1:]):
            self.assertLess(earlier, later)

    def test_wave_travels_head_to_tail_over_time(self):
        # theta(s, t) = A*sin(2*pi*(s/wavelength + speed*t)): the phase
        # value present at arc position `s` at t=0 reappears, unchanged, at
        # position `s - speed*wavelength*dt` at time `dt` -- i.e. the wave's
        # pattern shifts toward DECREASING s (head -> tail) as t increases.
        # This is exact algebra, not a numerical approximation:
        #   phase(s - shift, dt) == phase(s, 0)  when shift = speed*wavelength*dt
        params = SlitherParams(wavelength=0.4, amplitude=0.3, speed=1.0, head_stabilize=0.0)
        dt = 0.01
        shift = params.speed * params.wavelength * dt
        for s in (0.6, 0.35, 0.9):
            self.assertAlmostEqual(
                curve_tangent_angle(s, 0.0, params),
                curve_tangent_angle(s - shift, dt, params),
                places=9,
            )

    def test_root_bone_is_anchored_regardless_of_wave(self):
        # The chain root (s=0, the tail bone that hangs directly off the
        # armature) must never absolutely yaw -- only bones 1..N bend
        # relative to their tail-ward neighbor (review F3-4).
        params = SlitherParams(amplitude=0.4, wavelength=0.6, speed=2.0, head_stabilize=0.2)
        for t in (0.0, 0.1, 0.37, 0.9):
            rotations = chain_rotations(self.s, t, params)
            self.assertEqual(rotations[0], 0.0)

    def test_seamless_loop_on_the_ops_actual_frame_grid(self):
        # The Blender op samples t = frame / frames for frame in
        # range(frames); "frame == frames" (one past the last real frame,
        # i.e. where the next loop iteration's frame 0 would land) must
        # reproduce frame 0 exactly for the bake to loop without a pop.
        frames = 48
        params = SlitherParams(wavelength=0.4, amplitude=0.25, speed=2.0, head_stabilize=0.3)
        first = chain_rotations(self.s, 0 / frames, params)
        wrapped = chain_rotations(self.s, frames / frames, params)
        for a, b in zip(first, wrapped):
            self.assertAlmostEqual(a, b, places=9)

    def test_head_peak_deviation_degrees_two_bone_chain(self):
        # Analytically tractable case: a 2-bone chain (root + head) with a
        # wavelength spanning exactly half the chain puts the two bones in
        # perfect antiphase, so the head's LOCAL rotation (relative to the
        # anchored root) has a clean closed-form peak of 2*A radians, where
        # A = amplitude * pi. amplitude=0.2 -> peak == exactly 72 degrees.
        params = SlitherParams(wavelength=2.0, amplitude=0.2, speed=1.0, head_stabilize=0.0)
        s = [0.0, 1.0]
        samples = 2000
        peak = max(abs(chain_rotations(s, i / samples, params)[-1]) for i in range(samples))
        self.assertAlmostEqual(math.degrees(peak), 72.0, places=1)

    def test_empty_chain_returns_empty(self):
        params = SlitherParams()
        self.assertEqual(chain_rotations([], 0.0, params), [])

    def test_default_params_produce_a_genuine_s_curve_not_a_rigid_tilt(self):
        # Regression for a rendered defect: a synthetic 10-bone chain baked
        # with the documented defaults (wavelength=1.0, amplitude=0.12,
        # speed=1, head_stabilize=0.5) rendered as a rigidly-tilted straight
        # rod instead of an S -- root-quaternion-identity and
        # head-tip-X-oscillates checks alone don't catch this, because a
        # rigid tilt about the anchored root satisfies both. The actual
        # defect turned out to be the test rig's mesh/weights, not this
        # math, but the math itself must be provably incapable of
        # producing only a rigid tilt: a single overall bend has per-bone
        # LOCAL rotations that never change sign, and a monotonic
        # cumulative (world) angle along the chain. A real S needs the
        # chain to curve one way over part of its length and the opposite
        # way over the rest.
        params = SlitherParams()  # documented defaults
        rotations = chain_rotations(self.s, 0.25, params)

        signs = [1 if r > 1e-9 else (-1 if r < -1e-9 else 0) for r in rotations[1:]]
        self.assertIn(1, signs, "expected at least one bone bending one way")
        self.assertIn(-1, signs, "expected at least one bone bending the other way")

        cumulative = []
        total = 0.0
        for rotation in rotations:
            total += rotation
            cumulative.append(total)
        increasing = all(a < b for a, b in zip(cumulative, cumulative[1:]))
        decreasing = all(a > b for a, b in zip(cumulative, cumulative[1:]))
        self.assertFalse(
            increasing or decreasing,
            "cumulative world angle must not be monotonic along the chain -- "
            "a monotonic profile is a rigid tilt/C-bend, not an S",
        )


class TaperTests(unittest.TestCase):
    def test_taper_at_tail_is_always_one(self):
        for head_stabilize in (0.0, 0.3, 1.0):
            self.assertEqual(taper(0.0, head_stabilize), 1.0)

    def test_taper_at_head_matches_one_minus_head_stabilize(self):
        for head_stabilize in (0.0, 0.3, 1.0):
            self.assertAlmostEqual(taper(1.0, head_stabilize), 1.0 - head_stabilize)

    def test_taper_clamps_out_of_range_s(self):
        self.assertEqual(taper(-0.5, 0.5), 1.0)
        self.assertEqual(taper(1.5, 0.5), 0.5)


class ArcPositionsFromLengthsTests(unittest.TestCase):
    def test_endpoints_are_zero_and_one(self):
        positions = arc_positions_from_lengths([1.0, 2.0, 1.0, 3.0])
        self.assertAlmostEqual(positions[0], 0.0)
        self.assertAlmostEqual(positions[-1], 1.0)

    def test_positions_are_monotonically_increasing(self):
        positions = arc_positions_from_lengths([0.5, 1.5, 0.7, 2.0, 0.3])
        for earlier, later in zip(positions, positions[1:]):
            self.assertLess(earlier, later)

    def test_uniform_lengths_are_evenly_spread(self):
        positions = arc_positions_from_lengths([1.0] * 5)
        expected = [i / 4.0 for i in range(5)]
        for actual, expect in zip(positions, expected):
            self.assertAlmostEqual(actual, expect)

    def test_single_bone_returns_zero(self):
        self.assertEqual(arc_positions_from_lengths([1.0]), [0.0])

    def test_empty_returns_empty(self):
        self.assertEqual(arc_positions_from_lengths([]), [])

    def test_degenerate_zero_lengths_fall_back_to_even_spread(self):
        positions = arc_positions_from_lengths([0.0, 0.0, 0.0, 0.0])
        expected = [i / 3.0 for i in range(4)]
        for actual, expect in zip(positions, expected):
            self.assertAlmostEqual(actual, expect)


class BoneRotationQuaternionTests(unittest.TestCase):
    def test_zero_angle_is_identity(self):
        self.assertEqual(bone_rotation_quaternion(0.0), (1.0, 0.0, 0.0, 0.0))

    def test_result_is_always_unit_norm(self):
        for angle in (0.1, 1.0, -2.3, math.pi):
            w, x, y, z = bone_rotation_quaternion(angle)
            norm = math.sqrt(w * w + x * x + y * y + z * z)
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_matches_axis_angle_half_angle_formula_about_z(self):
        angle = math.radians(60.0)
        w, x, y, z = bone_rotation_quaternion(angle, axis=(0.0, 0.0, 1.0))
        self.assertAlmostEqual(w, math.cos(angle / 2.0))
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, math.sin(angle / 2.0))

    def test_normalizes_a_non_unit_axis(self):
        angle = math.radians(45.0)
        direct = bone_rotation_quaternion(angle, axis=(0.0, 0.0, 2.0))
        normalized = bone_rotation_quaternion(angle, axis=(0.0, 0.0, 1.0))
        for a, b in zip(direct, normalized):
            self.assertAlmostEqual(a, b, places=9)

    def test_rejects_zero_axis(self):
        with self.assertRaises(ValueError):
            bone_rotation_quaternion(1.0, axis=(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
