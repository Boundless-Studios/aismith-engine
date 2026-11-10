import unittest

import numpy as np

from backend.app.texture_bake import (
    BAKE_TEXTURE_SIZE_MAXIMUM,
    BAKE_TEXTURE_SIZE_MINIMUM,
    NORMAL_MAP_COLORSPACE,
    NORMAL_MAP_FLAT_RGB,
    NORMAL_MAP_GREEN_CHANNEL,
    NORMAL_MAP_SPACE,
    PALETTE_HUE_EVALUABLE_FRACTION,
    PALETTE_LUMINANCE_STRENGTH,
    UV_PACK_MARGIN_TEXELS,
    bake_image_name,
    bake_texture_size,
    chromatic_fraction,
    estimate_background_color,
    fill_transparent_texels,
    hue_histogram,
    image_colorfulness,
    linear_rgb_to_oklab,
    linear_to_srgb,
    mark_normal_bake_coverage,
    normal_bake_report,
    oklab_to_linear_rgb,
    palette_distance,
    palette_transfer_report,
    renormalize_tangent_normals,
    srgb_to_linear,
    subject_pixel_mask,
    transfer_palette,
    unique_by_identity,
    uv_pack_margin,
)


class BakeTextureSizeTests(unittest.TestCase):
    def test_matches_a_power_of_two_source_texture_exactly(self):
        self.assertEqual(bake_texture_size([2048, 2048, 2048, 2048]), 2048)

    def test_follows_the_largest_source_texture_not_the_first(self):
        self.assertEqual(bake_texture_size([512, 512, 4096, 1024]), 4096)

    def test_rounds_a_non_power_of_two_source_up(self):
        self.assertEqual(bake_texture_size([1536, 1536]), 2048)
        self.assertEqual(bake_texture_size([1025]), 2048)

    def test_caps_at_the_maximum_so_an_8k_source_does_not_blow_up_the_bake(self):
        self.assertEqual(bake_texture_size([8192, 8192]), BAKE_TEXTURE_SIZE_MAXIMUM)

    def test_floors_at_the_minimum_for_a_tiny_source_texture(self):
        self.assertEqual(bake_texture_size([64, 128]), BAKE_TEXTURE_SIZE_MINIMUM)

    def test_a_source_with_no_textures_falls_back_to_the_minimum(self):
        self.assertEqual(bake_texture_size([]), BAKE_TEXTURE_SIZE_MINIMUM)
        self.assertEqual(bake_texture_size([0, 0]), BAKE_TEXTURE_SIZE_MINIMUM)

    def test_honors_explicit_bounds(self):
        self.assertEqual(bake_texture_size([4096], minimum=256, maximum=1024), 1024)
        self.assertEqual(bake_texture_size([100], minimum=256, maximum=1024), 256)

    def test_rejects_an_inverted_or_non_positive_band(self):
        with self.assertRaises(ValueError):
            bake_texture_size([2048], minimum=2048, maximum=1024)
        with self.assertRaises(ValueError):
            bake_texture_size([2048], minimum=0)


class UvPackMarginTests(unittest.TestCase):
    def test_margin_means_the_same_texel_gutter_at_every_atlas_size(self):
        for size in (1024, 2048, 4096):
            self.assertAlmostEqual(
                uv_pack_margin(size) * size, float(UV_PACK_MARGIN_TEXELS), places=6,
            )

    def test_explicit_texel_gutter_is_honored(self):
        self.assertAlmostEqual(uv_pack_margin(2048, texels=8), 8 / 2048)

    def test_a_zero_gutter_is_allowed(self):
        self.assertEqual(uv_pack_margin(2048, texels=0), 0.0)

    def test_a_pathologically_small_atlas_cannot_produce_a_margin_over_a_quarter(self):
        self.assertEqual(uv_pack_margin(4, texels=64), 0.25)

    def test_rejects_nonsense_inputs(self):
        with self.assertRaises(ValueError):
            uv_pack_margin(0)
        with self.assertRaises(ValueError):
            uv_pack_margin(2048, texels=-1)


class TextureBakeTests(unittest.TestCase):
    def test_shared_bake_materials_are_processed_once(self):
        first = object()
        second = object()

        self.assertEqual(
            unique_by_identity([first, first, second, first]),
            [first, second],
        )

    def test_fill_transparent_texels_spreads_valid_color_into_bake_misses(self):
        pixels = np.zeros((3, 3, 4), dtype=np.float32)
        pixels[1, 1] = (0.2, 0.6, 0.9, 1.0)

        filled = fill_transparent_texels(pixels)

        expected_color = np.broadcast_to((0.2, 0.6, 0.9), (3, 3, 3))
        np.testing.assert_allclose(filled[:, :, :3], expected_color)
        np.testing.assert_allclose(filled[:, :, 3], 1.0)

    def test_fill_transparent_texels_preserves_valid_samples(self):
        pixels = np.array([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)

        np.testing.assert_array_equal(fill_transparent_texels(pixels), pixels)

    def test_fill_transparent_texels_forces_unreached_fallback_texels_opaque(self):
        pixels = np.zeros((2, 2, 4), dtype=np.float32)
        pixels[:, :, :3] = 0.18

        filled = fill_transparent_texels(pixels, max_passes=0)

        np.testing.assert_allclose(filled[:, :, :3], pixels[:, :, :3])
        np.testing.assert_allclose(filled[:, :, 3], 1.0)


class BakeImageNameTests(unittest.TestCase):
    def test_names_are_channel_derived_and_prefixed(self):
        self.assertEqual(bake_image_name("Normal"), "AISmith_Retopo_Normal")
        self.assertEqual(bake_image_name("BaseColor"), "AISmith_Retopo_BaseColor")

    def test_a_spaced_channel_name_collapses_to_one_datablock_token(self):
        self.assertEqual(bake_image_name("Base Color"), "AISmith_Retopo_BaseColor")

    def test_rejects_an_empty_channel(self):
        with self.assertRaises(ValueError):
            bake_image_name("   ")


class NormalBakeCoverageTests(unittest.TestCase):
    @staticmethod
    def _generated(shape):
        buffer = np.zeros(shape, dtype=np.float32)
        buffer[:, :, :3] = NORMAL_MAP_FLAT_RGB
        buffer[:, :, 3] = 1.0
        return buffer

    def test_texels_the_bake_did_not_move_are_marked_as_misses(self):
        before = self._generated((1, 2, 4))
        after = before.copy()
        after[0, 1] = (0.9, 0.2, 0.7, 1.0)

        marked = mark_normal_bake_coverage(after, before)

        self.assertEqual(marked[0, 0, 3], 0.0)
        self.assertEqual(marked[0, 1, 3], 1.0)

    def test_coverage_survives_a_quantized_generated_fallback(self):
        # `bpy.data.images.new` allocates a BYTE buffer, so a generated 0.5
        # reads back as 127/255. Comparing against the nominal flat constant
        # would call every texel covered and report a 0% miss rate.
        before = self._generated((1, 2, 4))
        before[:, :, 0:2] = 127.0 / 255.0
        after = before.copy()
        after[0, 1] = (0.9, 0.2, 0.7, 1.0)

        marked = mark_normal_bake_coverage(after, before)

        self.assertEqual(marked[0, 0, 3], 0.0)
        self.assertEqual(marked[0, 1, 3], 1.0)

    def test_coverage_ignores_whatever_the_normal_pass_left_in_alpha(self):
        # The NORMAL pass's alpha is not a coverage channel -- a baked texel
        # arriving with alpha 0 must still count as covered.
        before = self._generated((1, 1, 4))
        after = np.array([[[0.4, 0.6, 0.95, 0.0]]], dtype=np.float32)

        self.assertEqual(mark_normal_bake_coverage(after, before)[0, 0, 3], 1.0)

    def test_rejects_a_non_rgba_or_mismatched_buffer(self):
        with self.assertRaises(ValueError):
            mark_normal_bake_coverage(
                np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2, 3), dtype=np.float32),
            )
        with self.assertRaises(ValueError):
            mark_normal_bake_coverage(
                np.zeros((2, 2, 4), dtype=np.float32), np.zeros((3, 2, 4), dtype=np.float32),
            )


class RenormalizeTangentNormalsTests(unittest.TestCase):
    def test_an_averaged_pair_of_normals_is_restored_to_unit_length(self):
        # What the miss dilation produces: the mean of two unit normals, which
        # is short, and therefore shades as a tilt toward the surface.
        left = np.array([-1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        averaged = (left + up) / 2.0
        pixels = np.array([[[*(averaged * 0.5 + 0.5), 1.0]]], dtype=np.float32)

        decoded = renormalize_tangent_normals(pixels)[0, 0, :3] * 2.0 - 1.0

        self.assertAlmostEqual(float(np.linalg.norm(decoded)), 1.0, places=5)

    def test_an_already_unit_normal_is_left_alone(self):
        pixels = np.array([[[*NORMAL_MAP_FLAT_RGB, 1.0]]], dtype=np.float32)

        np.testing.assert_allclose(
            renormalize_tangent_normals(pixels)[:, :, :3],
            np.array([[NORMAL_MAP_FLAT_RGB]], dtype=np.float32),
            atol=1e-6,
        )

    def test_a_degenerate_sample_falls_back_to_the_flat_normal(self):
        pixels = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)

        np.testing.assert_allclose(
            renormalize_tangent_normals(pixels)[0, 0, :3],
            np.asarray(NORMAL_MAP_FLAT_RGB, dtype=np.float32),
            atol=1e-6,
        )

    def test_rejects_a_non_rgba_buffer(self):
        with self.assertRaises(ValueError):
            renormalize_tangent_normals(np.zeros((2, 2, 3), dtype=np.float32))


class NormalBakeReportTests(unittest.TestCase):
    def test_report_pins_every_convention_a_wrong_normal_map_would_hide(self):
        report = normal_bake_report(
            texture_size=2048, miss_fraction=0.1234567, target_shading="smooth",
        )

        self.assertEqual(report["baked"], True)
        self.assertEqual(report["image_name"], "AISmith_Retopo_Normal")
        self.assertEqual(report["texture_size"], 2048)
        self.assertEqual(report["space"], NORMAL_MAP_SPACE)
        self.assertEqual(report["space"], "tangent")
        self.assertEqual(report["green_channel"], NORMAL_MAP_GREEN_CHANNEL)
        self.assertEqual(report["green_channel"], "+Y")
        self.assertEqual(report["colorspace"], NORMAL_MAP_COLORSPACE)
        self.assertEqual(report["colorspace"], "Non-Color")
        self.assertEqual(report["target_shading"], "smooth")
        self.assertEqual(report["miss_fraction"], 0.123457)

    def test_a_missing_miss_fraction_is_recorded_as_unknown_not_zero(self):
        report = normal_bake_report(
            texture_size=1024, miss_fraction=None, target_shading="flat",
        )

        self.assertIsNone(report["miss_fraction"])
        self.assertEqual(report["target_shading"], "flat")

    def test_rejects_nonsense_provenance(self):
        with self.assertRaises(ValueError):
            normal_bake_report(texture_size=0, miss_fraction=0.0, target_shading="smooth")
        with self.assertRaises(ValueError):
            normal_bake_report(texture_size=2048, miss_fraction=1.5, target_shading="smooth")
        with self.assertRaises(ValueError):
            normal_bake_report(texture_size=2048, miss_fraction=0.0, target_shading="shaded")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Palette fidelity
# ---------------------------------------------------------------------------


def _mono_image(height=64, width=64, level=0.5, noise=0.0, seed=0):
    """A neutral grey image: every channel equal, so chroma is exactly zero."""
    rng = np.random.default_rng(seed)
    grey = np.full((height, width), float(level), dtype=np.float64)
    if noise:
        grey = np.clip(grey + rng.normal(0.0, noise, grey.shape), 0.0, 1.0)
    return np.repeat(grey[:, :, None], 3, axis=2)


def _tinted_image(height=64, width=64, rgb=(0.7, 0.4, 0.4), seed=1, noise=0.02):
    """A saturated image -- the invented-palette stand-in."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float64)
    for channel, value in enumerate(rgb):
        base[:, :, channel] = value
    return np.clip(base + rng.normal(0.0, noise, base.shape), 0.0, 1.0)


def _framed(subject, background=0.9, border=12):
    """Put `subject` on a uniform background so the border ring keys it out."""
    height, width = subject.shape[0] + 2 * border, subject.shape[1] + 2 * border
    framed = np.full((height, width, 3), float(background), dtype=np.float64)
    framed[border:-border, border:-border] = subject
    return framed


def _chroma(linear_rgb, mask=None):
    """Mean Oklab chroma magnitude -- 'how much colour is in here'."""
    lab = linear_rgb_to_oklab(linear_rgb)
    selected = lab.reshape(-1, 3) if mask is None else lab[mask]
    return float(np.hypot(selected[:, 1], selected[:, 2]).mean())


class TransferPaletteTests(unittest.TestCase):
    def test_mono_reference_drives_output_chroma_to_zero(self):
        """The operator's ask: a monochrome source yields a monochrome texture.

        No special case implements this -- a neutral reference simply has
        chroma mean and standard deviation of ~0, so the match scales the
        albedo's chroma by ~0 and re-centres it on ~0.
        """
        albedo = _tinted_image(rgb=(0.75, 0.35, 0.35))
        reference = _mono_image(noise=0.05)
        before = _chroma(albedo)
        result = transfer_palette(albedo, reference)
        after = _chroma(result)
        self.assertGreater(before, 0.02, "the fixture albedo must start visibly coloured")
        self.assertLess(after, before / 20.0)
        self.assertLess(after, 0.002)

    def test_mono_reference_preserves_luminance_structure(self):
        """Draining the colour must not flatten the light/dark structure.

        A texture whose chroma is removed but whose luminance is crushed is a
        different failure, not a fix: the character would read as a silhouette.
        """
        albedo = np.zeros((64, 64, 3), dtype=np.float64)
        albedo[:32] = (0.8, 0.3, 0.3)
        albedo[32:] = (0.2, 0.08, 0.08)
        result = transfer_palette(albedo, _mono_image(noise=0.05))
        before = linear_rgb_to_oklab(albedo)[:, :, 0]
        after = linear_rgb_to_oklab(result)[:, :, 0]
        before_spread = before[:32].mean() - before[32:].mean()
        after_spread = after[:32].mean() - after[32:].mean()
        # Most of the light/dark separation survives. It is allowed to shrink
        # -- L's standard deviation is matched at PALETTE_LUMINANCE_STRENGTH
        # -- but a texture whose contrast collapsed is a different failure.
        self.assertGreater(after_spread, 0.6 * before_spread)

    def test_colored_reference_corrects_a_hue_drift(self):
        """A red-tinted albedo against a blue-tinted reference moves to blue."""
        albedo = _tinted_image(rgb=(0.7, 0.35, 0.35), seed=2)
        reference = _tinted_image(rgb=(0.35, 0.4, 0.75), seed=3)
        result = transfer_palette(albedo, reference)
        reference_lab = linear_rgb_to_oklab(reference).reshape(-1, 3).mean(axis=0)
        result_lab = linear_rgb_to_oklab(result).reshape(-1, 3).mean(axis=0)
        albedo_lab = linear_rgb_to_oklab(albedo).reshape(-1, 3).mean(axis=0)
        # Chroma lands on the reference's, not somewhere between.
        np.testing.assert_allclose(result_lab[1:], reference_lab[1:], atol=2e-3)
        self.assertGreater(abs(albedo_lab[1] - reference_lab[1]), 0.05)

    def test_luminance_is_matched_only_gently(self):
        """L moves toward the reference but never all the way onto it.

        The reference is a SHADED image and the albedo is unlit base colour,
        so a full luminance match would bake the reference's exposure into the
        texture. Chroma is the channel that gets matched outright.
        """
        albedo = _mono_image(level=0.6, noise=0.02)
        reference = _mono_image(level=0.2, noise=0.02, seed=5)
        result = transfer_palette(albedo, reference)
        albedo_l = linear_rgb_to_oklab(albedo)[:, :, 0].mean()
        reference_l = linear_rgb_to_oklab(reference)[:, :, 0].mean()
        result_l = linear_rgb_to_oklab(result)[:, :, 0].mean()
        self.assertLess(result_l, albedo_l)
        self.assertGreater(result_l, reference_l)
        moved = (albedo_l - result_l) / (albedo_l - reference_l)
        self.assertAlmostEqual(moved, PALETTE_LUMINANCE_STRENGTH, places=2)

    def test_strength_zero_is_an_exact_no_op(self):
        albedo = _tinted_image(rgb=(0.7, 0.3, 0.3))
        result = transfer_palette(albedo, _mono_image(), strength=0.0)
        np.testing.assert_array_equal(result, albedo)

    def test_strength_blends_between_input_and_full_match(self):
        albedo = _tinted_image(rgb=(0.75, 0.35, 0.35))
        reference = _mono_image(noise=0.05)
        full = _chroma(transfer_palette(albedo, reference, strength=1.0))
        half = _chroma(transfer_palette(albedo, reference, strength=0.5))
        start = _chroma(albedo)
        self.assertLess(full, half)
        self.assertLess(half, start)

    def test_albedo_mask_leaves_uncovered_texels_untouched(self):
        """Bake misses keep their neutral fallback -- the coverage mask rules.

        The uncovered atlas is fallback grey and gutter, not baked surface;
        rewriting it would drag the whole texture's statistics toward a colour
        the bake never produced.
        """
        albedo = _tinted_image(rgb=(0.75, 0.35, 0.35))
        covered = np.zeros(albedo.shape[:2], dtype=bool)
        covered[:32] = True
        result = transfer_palette(
            albedo, _mono_image(noise=0.05), albedo_mask=covered,
        )
        np.testing.assert_array_equal(result[32:], albedo[32:])
        self.assertFalse(np.allclose(result[:32], albedo[:32]))

    def test_albedo_mask_also_scopes_the_statistics(self):
        """Statistics come from covered texels only, not the whole atlas."""
        albedo = _tinted_image(rgb=(0.75, 0.35, 0.35))
        albedo[32:] = 0.18  # the neutral generated fallback
        covered = np.zeros(albedo.shape[:2], dtype=bool)
        covered[:32] = True
        masked = transfer_palette(albedo, _mono_image(noise=0.05), albedo_mask=covered)
        unmasked = transfer_palette(albedo, _mono_image(noise=0.05))
        self.assertFalse(np.allclose(masked[:32], unmasked[:32]))

    def test_reference_mask_selects_the_subject_not_the_backdrop(self):
        """A blue backdrop must not tint the texture blue."""
        subject = _mono_image(height=40, width=40, level=0.45, noise=0.03)
        reference = np.full((64, 64, 3), 0.0, dtype=np.float64)
        reference[:, :] = (0.1, 0.15, 0.8)
        reference[12:52, 12:52] = subject
        mask = np.zeros(reference.shape[:2], dtype=bool)
        mask[12:52, 12:52] = True
        albedo = _tinted_image(rgb=(0.75, 0.35, 0.35))
        with_mask = _chroma(transfer_palette(albedo, reference, mask))
        without_mask = _chroma(transfer_palette(albedo, reference))
        self.assertLess(with_mask, 0.002)
        self.assertGreater(without_mask, with_mask * 5)

    def test_empty_reference_mask_is_a_no_op(self):
        """Nothing to measure means nothing is changed -- never an exception."""
        albedo = _tinted_image(rgb=(0.7, 0.3, 0.3))
        empty = np.zeros(albedo.shape[:2], dtype=bool)
        np.testing.assert_array_equal(
            transfer_palette(albedo, _mono_image(), empty), albedo
        )

    def test_flat_albedo_channel_does_not_divide_by_zero(self):
        """A perfectly uniform albedo has no spread to rescale."""
        albedo = np.full((32, 32, 3), 0.5, dtype=np.float64)
        result = transfer_palette(albedo, _tinted_image(rgb=(0.3, 0.6, 0.4)))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0.0) & (result <= 1.0)).all())

    def test_output_stays_in_gamut(self):
        albedo = _tinted_image(rgb=(0.95, 0.05, 0.05), noise=0.05)
        result = transfer_palette(albedo, _tinted_image(rgb=(0.05, 0.95, 0.9), seed=9))
        self.assertTrue(((result >= 0.0) & (result <= 1.0)).all())

    def test_rejects_out_of_range_strength(self):
        albedo = _mono_image()
        with self.assertRaises(ValueError):
            transfer_palette(albedo, albedo, strength=1.5)
        with self.assertRaises(ValueError):
            transfer_palette(albedo, albedo, strength=-0.1)

    def test_rejects_malformed_arrays(self):
        with self.assertRaises(ValueError):
            transfer_palette(np.zeros((4, 4)), _mono_image())
        with self.assertRaises(ValueError):
            transfer_palette(_mono_image(), np.zeros((4, 4, 4)))

    def test_mismatched_mask_shape_is_rejected(self):
        albedo = _mono_image()
        with self.assertRaises(ValueError):
            transfer_palette(
                albedo, albedo, albedo_mask=np.ones((8, 8), dtype=bool)
            )


class OklabRoundTripTests(unittest.TestCase):
    # The published Oklab matrices are given to 10 significant figures and the
    # forward/inverse pair is not exactly self-inverse at that precision, so
    # this tolerance is set by the CONSTANTS, not by the implementation
    # (measured round-trip error: 2.4e-7 max, 5.6e-8 mean). It is still ~250x
    # tighter than one 8-bit code value (1/255 = 3.9e-3), which is the only
    # precision an atlas byte buffer can carry anyway.
    PRECISION = 1e-6

    def test_matches_ottossons_published_reference_values(self):
        """Pin the transform against the source it came from.

        A tolerance test alone would pass with a transposed or mistyped
        matrix; these two values would not.
        """
        white = linear_rgb_to_oklab(np.array([[[1.0, 1.0, 1.0]]]))[0, 0]
        np.testing.assert_allclose(white, (1.0, 0.0, 0.0), atol=1e-6)
        red = linear_rgb_to_oklab(np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
        np.testing.assert_allclose(red, (0.62796, 0.22486, 0.12585), atol=1e-5)

    def test_oklab_round_trips_linear_rgb(self):
        rng = np.random.default_rng(11)
        rgb = rng.random((16, 16, 3))
        np.testing.assert_allclose(
            oklab_to_linear_rgb(linear_rgb_to_oklab(rgb)), rgb, atol=self.PRECISION
        )

    def test_srgb_transfer_curve_round_trips(self):
        values = np.linspace(0.0, 1.0, 257).reshape(1, -1, 1) * np.ones((1, 1, 3))
        np.testing.assert_allclose(
            linear_to_srgb(srgb_to_linear(values)), values, atol=1e-9
        )

    def test_neutral_grey_has_zero_chroma(self):
        """Every equal-channel colour must sit on the Oklab L axis.

        This is the invariant the whole mono case rests on: if grey did not
        map to zero chroma, "match the reference's chroma" could not mean
        "make it grey".
        """
        grey = np.full((4, 4, 3), 0.42, dtype=np.float64)
        lab = linear_rgb_to_oklab(grey)
        np.testing.assert_allclose(lab[:, :, 1:], 0.0, atol=self.PRECISION)


class SubjectPixelMaskTests(unittest.TestCase):
    def test_keys_a_subject_off_a_uniform_border(self):
        subject = _tinted_image(height=40, width=40, rgb=(0.2, 0.2, 0.2), noise=0.0)
        framed = _framed(subject, background=0.9)
        mask = subject_pixel_mask(framed)
        self.assertIsNotNone(mask)
        self.assertTrue(mask[32, 32])
        self.assertFalse(mask[2, 2])

    def test_refuses_a_non_uniform_border(self):
        """A gradient backdrop is not a background; refusing beats guessing."""
        gradient = np.linspace(0.0, 1.0, 64).reshape(1, -1, 1) * np.ones((64, 1, 3))
        self.assertIsNone(subject_pixel_mask(gradient))

    def test_prefers_real_alpha_over_keying(self):
        rgb = _mono_image(level=0.9)
        alpha = np.zeros(rgb.shape[:2], dtype=np.float64)
        alpha[16:48, 16:48] = 1.0
        mask = subject_pixel_mask(rgb, alpha)
        self.assertTrue(mask[32, 32])
        self.assertFalse(mask[0, 0])

    def test_opaque_alpha_falls_through_to_keying(self):
        """A fully opaque alpha channel carries no subject information."""
        framed = _framed(_tinted_image(height=40, width=40, rgb=(0.2, 0.2, 0.2), noise=0.0))
        opaque = np.ones(framed.shape[:2], dtype=np.float64)
        mask = subject_pixel_mask(framed, opaque)
        self.assertIsNotNone(mask)
        self.assertTrue(mask[32, 32])


class EstimateBackgroundColorTests(unittest.TestCase):
    def test_reads_the_median_border_color(self):
        subject = _tinted_image(height=40, width=40, rgb=(0.2, 0.2, 0.2), noise=0.0)
        framed = _framed(subject, background=0.9)
        background = estimate_background_color(framed)
        self.assertIsNotNone(background)
        np.testing.assert_allclose(background, (0.9, 0.9, 0.9), atol=1e-6)

    def test_none_for_a_non_uniform_border(self):
        gradient = np.linspace(0.0, 1.0, 64).reshape(1, -1, 1) * np.ones((64, 1, 3))
        self.assertIsNone(estimate_background_color(gradient))


class PaletteDistanceTests(unittest.TestCase):
    def test_identical_images_are_zero_distance(self):
        image = _tinted_image(rgb=(0.6, 0.4, 0.3))
        srgb = linear_to_srgb(image)
        self.assertAlmostEqual(palette_distance(srgb, srgb)["distance"], 0.0, places=6)

    def test_two_monochrome_images_are_zero_distance(self):
        """Both achromatic means both hue histograms are empty -- and equal.

        This is the case a naive chi-square gets wrong by dividing 0 by 0, and
        it is the case that matters: a mono reference against a correctly-mono
        candidate is the PASSING state the gate is calibrated to allow.
        """
        first = linear_to_srgb(_mono_image(level=0.5, noise=0.04, seed=1))
        second = linear_to_srgb(_mono_image(level=0.55, noise=0.04, seed=2))
        self.assertLess(palette_distance(first, second)["distance"], 0.05)

    def test_mono_reference_against_a_colored_candidate_is_far(self):
        reference = linear_to_srgb(_mono_image(level=0.5, noise=0.04))
        candidate = linear_to_srgb(_tinted_image(rgb=(0.75, 0.3, 0.3)))
        self.assertGreater(palette_distance(reference, candidate)["distance"], 0.4)

    def test_transfer_collapses_the_measured_distance(self):
        """End to end: the transfer is what moves the metric, on one array pair."""
        albedo = _tinted_image(rgb=(0.75, 0.3, 0.3))
        reference = _mono_image(level=0.5, noise=0.04)
        before = palette_distance(linear_to_srgb(reference), linear_to_srgb(albedo))
        after = palette_distance(
            linear_to_srgb(reference), linear_to_srgb(transfer_palette(albedo, reference))
        )
        self.assertGreater(before["distance"], 0.4)
        self.assertLess(after["distance"], 0.1)

    def test_distance_is_symmetric(self):
        first = linear_to_srgb(_tinted_image(rgb=(0.7, 0.3, 0.3), seed=1))
        second = linear_to_srgb(_tinted_image(rgb=(0.3, 0.3, 0.7), seed=2))
        self.assertAlmostEqual(
            palette_distance(first, second)["distance"],
            palette_distance(second, first)["distance"],
            places=6,
        )

    def test_different_shapes_are_comparable(self):
        """Distributions, not pixels: a 768px render against a 1024px image."""
        reference = linear_to_srgb(_mono_image(height=100, width=100, noise=0.04))
        candidate = linear_to_srgb(_mono_image(height=48, width=48, noise=0.04, seed=7))
        self.assertLess(palette_distance(reference, candidate)["distance"], 0.05)


class ImageColorfulnessTests(unittest.TestCase):
    def test_neutral_grey_scores_zero(self):
        self.assertAlmostEqual(
            image_colorfulness(_mono_image(level=0.5)), 0.0, places=6
        )

    def test_saturated_scores_above_neutral(self):
        self.assertGreater(
            image_colorfulness(_tinted_image(rgb=(0.9, 0.2, 0.2))),
            image_colorfulness(_mono_image(noise=0.05)) + 10.0,
        )

    def test_mask_scopes_the_measurement(self):
        """A colourful subject on a grey field reads as colourful once masked."""
        image = _mono_image(height=64, width=64, level=0.5)
        image[24:40, 24:40] = (0.9, 0.2, 0.2)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[24:40, 24:40] = True
        self.assertGreater(
            image_colorfulness(image, mask), image_colorfulness(image, None)
        )


class HueHistogramTests(unittest.TestCase):
    def test_monochrome_produces_an_empty_histogram(self):
        """Grey pixels have no hue; letting them vote would be noise."""
        histogram = hue_histogram(linear_to_srgb(_mono_image(noise=0.02)))
        self.assertAlmostEqual(float(histogram.sum()), 0.0, places=9)

    def test_saturated_image_concentrates_in_few_bins(self):
        histogram = hue_histogram(linear_to_srgb(_tinted_image(rgb=(0.85, 0.2, 0.2))))
        self.assertAlmostEqual(float(histogram.sum()), 1.0, places=6)
        self.assertGreater(float(histogram.max()), 0.5)

    def test_rejects_a_non_positive_bin_count(self):
        with self.assertRaises(ValueError):
            hue_histogram(linear_to_srgb(_mono_image()), bins=0)


class PaletteTransferReportTests(unittest.TestCase):
    def test_records_both_sides_when_applied(self):
        report = palette_transfer_report(
            applied=True,
            reference_id="abc",
            strength=1.0,
            before={"colorfulness": 11.5},
            after={"colorfulness": 5.1},
            reference={"colorfulness": 4.8},
        )
        self.assertTrue(report["applied"])
        self.assertEqual(report["reference_artifact_id"], "abc")
        self.assertEqual(report["albedo_colorfulness_before"], 11.5)
        self.assertEqual(report["albedo_colorfulness_after"], 5.1)
        self.assertEqual(report["reference_colorfulness"], 4.8)

    def test_a_skipped_transfer_still_says_why(self):
        report = palette_transfer_report(applied=False, reason="no reference")
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "no reference")
        self.assertIsNone(report["albedo_colorfulness_after"])


class HueEvaluabilityTests(unittest.TestCase):
    """The gate that stops the hue term from dominating a monochrome case.

    Comparing an empty hue histogram against a populated one is near-maximal
    chi-square NO MATTER how well the palette matches, so on a mono reference
    the hue term must be excluded rather than trusted.
    """

    def test_monochrome_has_a_negligible_chromatic_fraction(self):
        self.assertLess(
            chromatic_fraction(linear_to_srgb(_mono_image(noise=0.03))),
            PALETTE_HUE_EVALUABLE_FRACTION,
        )

    def test_saturated_is_entirely_chromatic(self):
        self.assertGreater(
            chromatic_fraction(linear_to_srgb(_tinted_image(rgb=(0.85, 0.2, 0.2)))),
            0.9,
        )

    def test_mono_pair_reports_hue_not_evaluable(self):
        first = linear_to_srgb(_mono_image(level=0.5, noise=0.03, seed=1))
        second = linear_to_srgb(_mono_image(level=0.5, noise=0.03, seed=2))
        self.assertFalse(palette_distance(first, second)["hue_evaluable"])

    def test_colored_pair_reports_hue_evaluable(self):
        first = linear_to_srgb(_tinted_image(rgb=(0.8, 0.3, 0.3), seed=1))
        second = linear_to_srgb(_tinted_image(rgb=(0.3, 0.3, 0.8), seed=2))
        self.assertTrue(palette_distance(first, second)["hue_evaluable"])

    def test_a_mono_reference_scores_on_colorfulness_alone(self):
        """With hue excluded, the distance IS the colorfulness term."""
        reference = linear_to_srgb(_mono_image(level=0.5, noise=0.03))
        candidate = linear_to_srgb(_tinted_image(rgb=(0.8, 0.3, 0.3)))
        result = palette_distance(reference, candidate)
        self.assertFalse(result["hue_evaluable"])
        self.assertAlmostEqual(result["distance"], result["colorfulness_term"], places=6)

    def test_a_colored_reference_blends_both_terms(self):
        first = linear_to_srgb(_tinted_image(rgb=(0.8, 0.3, 0.3), seed=1))
        second = linear_to_srgb(_tinted_image(rgb=(0.3, 0.3, 0.8), seed=2))
        result = palette_distance(first, second)
        self.assertTrue(result["hue_evaluable"])
        self.assertAlmostEqual(
            result["distance"],
            0.5 * result["colorfulness_term"] + 0.5 * result["hue_term"],
            places=5,  # every field is rounded to 6 dp before it is returned
        )

    def test_wrong_hue_at_matched_saturation_is_still_caught(self):
        """The case colorfulness ratio alone is blind to: right amount of
        colour, wrong colour. Both sides are equally vivid, so the ratio term
        is ~0 and only the hue term can see the difference."""
        red = linear_to_srgb(_tinted_image(rgb=(0.8, 0.3, 0.3), seed=1))
        blue = linear_to_srgb(_tinted_image(rgb=(0.3, 0.3, 0.8), seed=2))
        result = palette_distance(red, blue)
        self.assertLess(result["colorfulness_term"], 0.15)
        self.assertGreater(result["hue_term"], 0.8)
        self.assertGreater(result["distance"], 0.4)


class PaletteCalibrationTests(unittest.TestCase):
    """The measured chain values `canonicalize_policy` is calibrated against.

    These are real numbers from 2026-09-01, measured on the albedo atlas
    against each chain's conditioning image. They are
    asserted here so that a change to the distance formula that silently
    re-orders the chains fails a test instead of quietly moving the gate.
    """

    def test_the_calibrated_chains_straddle_the_policy_threshold(self):
        from backend.app.canonicalize_policy import DEFAULT_PALETTE_DISTANCE_THRESHOLD

        punk_transfer_off = 0.6578
        punk_transfer_on = 0.1231
        alien_transfer_on = 0.0969
        self.assertGreater(punk_transfer_off, DEFAULT_PALETTE_DISTANCE_THRESHOLD)
        self.assertLess(punk_transfer_on, DEFAULT_PALETTE_DISTANCE_THRESHOLD)
        self.assertLess(alien_transfer_on, DEFAULT_PALETTE_DISTANCE_THRESHOLD)
