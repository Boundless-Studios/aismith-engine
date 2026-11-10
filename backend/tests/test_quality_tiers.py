"""Tests for `backend/app/quality_tiers.py` (bead i63ky).

The operator-facing promise is "pick a density and the whole pipeline agrees
with you", so what is tested here is the RESOLUTION -- precedence between a
tier and a custom number, the band every consumer derives from it, and the
refusals that must happen before a job starts rather than twenty minutes into
one.
"""
from __future__ import annotations

import unittest

from backend.app.quality_tiers import (
    ATLAS_MINIMUM_BASE,
    ATLAS_MINIMUM_LADDER,
    CUSTOM_TARGET_TRIANGLES_MAXIMUM,
    CUSTOM_TARGET_TRIANGLES_MINIMUM,
    DEFAULT_QUALITY_TIER,
    QUALITY_TIER_TARGETS,
    atlas_minimum_for_target,
    resolve_quality_target,
    tier_triangle_target,
    triangle_band,
)


class TierTargetTests(unittest.TestCase):
    def test_the_four_tiers_have_the_advertised_triangle_targets(self):
        self.assertEqual(
            QUALITY_TIER_TARGETS,
            {"draft": 10_000, "standard": 30_000, "high": 80_000, "ultra": 150_000},
        )

    def test_standard_is_the_default_and_reproduces_the_pre_tier_behaviour(self):
        # The whole compatibility argument: an operator who never opens the
        # selector gets the 30k mesh the pipeline hardcoded before i63ky.
        self.assertEqual(DEFAULT_QUALITY_TIER, "standard")
        self.assertEqual(resolve_quality_target({}).target_triangles, 30_000)

    def test_tier_names_are_case_and_whitespace_insensitive(self):
        self.assertEqual(tier_triangle_target(" Ultra "), 150_000)

    def test_an_unknown_tier_is_refused_rather_than_defaulted(self):
        with self.assertRaises(ValueError):
            tier_triangle_target("cinematic")
        with self.assertRaises(ValueError):
            resolve_quality_target({"quality_tier": "cinematic"})


class TriangleBandTests(unittest.TestCase):
    def test_the_band_is_half_to_one_and_a_half_times_the_target(self):
        self.assertEqual(triangle_band(30_000), (15_000, 45_000))
        self.assertEqual(triangle_band(150_000), (75_000, 225_000))

    def test_every_tier_target_sits_inside_its_own_band(self):
        for tier, target in QUALITY_TIER_TARGETS.items():
            low, high = triangle_band(target)
            with self.subTest(tier=tier):
                self.assertLess(low, target)
                self.assertGreater(high, target)

    def test_a_custom_target_uses_the_same_ratio_as_a_tier(self):
        self.assertEqual(triangle_band(120_000), (60_000, 180_000))

    def test_a_non_positive_target_is_refused(self):
        with self.assertRaises(ValueError):
            triangle_band(0)


class AtlasFloorTests(unittest.TestCase):
    def test_the_floor_climbs_with_density(self):
        # Measured, not guessed: a 150k mesh smart-projects into ~5x the UV
        # islands a 30k one does, and on a shared 2048 atlas that collapsed
        # utilisation from 41.8% to 9.4% and pushed the detail ratio below
        # its threshold (alien chain, 2026-09-01).
        self.assertEqual(atlas_minimum_for_target(150_000), 4096)
        self.assertEqual(atlas_minimum_for_target(80_000), 2048)

    def test_light_targets_keep_the_base_floor(self):
        self.assertEqual(atlas_minimum_for_target(10_000), ATLAS_MINIMUM_BASE)
        self.assertEqual(atlas_minimum_for_target(30_000), ATLAS_MINIMUM_BASE)

    def test_the_floor_never_exceeds_what_the_bake_can_allocate(self):
        from backend.app.texture_bake import BAKE_TEXTURE_SIZE_MAXIMUM

        for _threshold, floor in ATLAS_MINIMUM_LADDER:
            with self.subTest(floor=floor):
                self.assertLessEqual(floor, BAKE_TEXTURE_SIZE_MAXIMUM)

    def test_a_custom_target_gets_the_floor_its_density_earns(self):
        # Keyed off the resolved triangle count, not the tier NAME, so a
        # custom 120k is treated like the tier nearest it, not like "no tier".
        self.assertEqual(resolve_quality_target({"target_triangles": 120_000}).atlas_minimum, 2048)
        self.assertEqual(resolve_quality_target({"target_triangles": 200_000}).atlas_minimum, 4096)
        self.assertEqual(resolve_quality_target({"target_triangles": 12_000}).atlas_minimum, ATLAS_MINIMUM_BASE)


class ResolutionPrecedenceTests(unittest.TestCase):
    def test_a_selected_tier_drives_the_target_and_the_band(self):
        resolved = resolve_quality_target({"quality_tier": "ultra"})

        self.assertEqual(resolved.tier, "ultra")
        self.assertEqual(resolved.target_triangles, 150_000)
        self.assertEqual(resolved.band, (75_000, 225_000))
        self.assertEqual(resolved.source, "tier")

    def test_a_custom_triangle_target_overrides_a_selected_tier(self):
        resolved = resolve_quality_target({"quality_tier": "draft", "target_triangles": 96_000})

        self.assertEqual(resolved.tier, "custom")
        self.assertEqual(resolved.target_triangles, 96_000)
        self.assertEqual(resolved.band, (48_000, 144_000))
        self.assertEqual(resolved.source, "custom-triangles")
        # The overridden tier is still recorded, so provenance can say what
        # the operator had selected when they typed a number over it.
        self.assertEqual(resolved.requested_tier, "draft")

    def test_a_custom_vertex_target_overrides_a_selected_tier_too(self):
        resolved = resolve_quality_target({"quality_tier": "high", "target_vertices": 40_000})

        self.assertEqual(resolved.tier, "custom")
        self.assertEqual(resolved.target_triangles, 80_000)
        self.assertEqual(resolved.target_vertices, 40_000)
        self.assertEqual(resolved.source, "custom-vertices")

    def test_a_triangle_target_wins_over_a_vertex_target(self):
        resolved = resolve_quality_target({"target_triangles": 50_000, "target_vertices": 40_000})

        self.assertEqual(resolved.target_triangles, 50_000)
        self.assertEqual(resolved.source, "custom-triangles")

    def test_a_request_level_vertex_target_is_honoured_when_settings_omit_it(self):
        resolved = resolve_quality_target({}, request_target_vertices=12_345)

        self.assertEqual(resolved.tier, "custom")
        self.assertEqual(resolved.target_triangles, 24_690)
        self.assertEqual(resolved.target_vertices, 12_345)

    def test_settings_beat_the_request_level_vertex_target(self):
        resolved = resolve_quality_target({"target_vertices": 6_000}, request_target_vertices=12_345)

        self.assertEqual(resolved.target_vertices, 6_000)

    def test_vertex_targets_round_trip_so_the_legacy_knob_is_unchanged(self):
        # `main._parse_canonicalize_settings` hands the resolved vertex count
        # to the Instant Meshes path; an operator's number must come back out
        # of the triangle round trip exactly as they typed it.
        for vertices in (500, 6_000, 8_000, 12_345, 25_000):
            with self.subTest(vertices=vertices):
                self.assertEqual(
                    resolve_quality_target({"target_vertices": vertices}).target_vertices, vertices
                )

    def test_an_invalid_tier_is_refused_even_when_a_custom_number_overrides_it(self):
        # Eager validation: a typo'd tier is an operator mistake worth
        # surfacing, not something to silently discard because the custom
        # number would have won anyway.
        with self.assertRaises(ValueError):
            resolve_quality_target({"quality_tier": "ultraa", "target_triangles": 50_000})


class CustomTargetValidationTests(unittest.TestCase):
    def test_a_non_integer_custom_target_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_quality_target({"target_triangles": "lots"})

    def test_a_zero_or_negative_custom_target_is_refused(self):
        for value in (0, -5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_quality_target({"target_triangles": value})

    def test_targets_outside_the_supported_range_are_refused(self):
        with self.assertRaises(ValueError):
            resolve_quality_target({"target_triangles": CUSTOM_TARGET_TRIANGLES_MINIMUM - 1})
        with self.assertRaises(ValueError):
            resolve_quality_target({"target_triangles": CUSTOM_TARGET_TRIANGLES_MAXIMUM + 1})

    def test_the_range_boundaries_themselves_are_accepted(self):
        for value in (CUSTOM_TARGET_TRIANGLES_MINIMUM, CUSTOM_TARGET_TRIANGLES_MAXIMUM):
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_quality_target({"target_triangles": value}).target_triangles, value
                )


class ProvenanceTests(unittest.TestCase):
    def test_provenance_carries_every_field_the_gate_and_the_ui_read(self):
        provenance = resolve_quality_target({"quality_tier": "high"}).as_provenance()

        self.assertEqual(provenance, {
            "quality_tier": "high",
            "quality_tier_requested": "high",
            "quality_target_source": "tier",
            "target_triangles": 80_000,
            "triangle_budget_band": [40_000, 120_000],
            "bake_atlas_minimum": 2048,
        })

    def test_a_default_run_is_labelled_as_defaulted_not_as_a_selection(self):
        provenance = resolve_quality_target({}).as_provenance()

        self.assertEqual(provenance["quality_tier"], "standard")
        self.assertIsNone(provenance["quality_tier_requested"])
        self.assertEqual(provenance["quality_target_source"], "default-tier")


if __name__ == "__main__":
    unittest.main()
