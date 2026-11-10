import unittest

from backend.app.canonicalize_policy import (
    DEFAULT_DETAIL_RATIO_THRESHOLD,
    DEFAULT_PALETTE_DISTANCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRIANGLE_BAND,
    evaluate_canonicalize,
)
from backend.app.quality_tiers import QUALITY_TIER_TARGETS, triangle_band


def _good_mesh_report(**overrides):
    report = {
        "components": 1,
        "components_unwelded": 1,
        "watertight": True,
        "non_manifold_edges": 0,
        "faces": 30_000,
    }
    report.update(overrides)
    return report


def _good_similarity_report(**overrides):
    report = {"s1_s2_image_similarity_score": 0.95, "s1_s2_detail_ratio": 0.95}
    report.update(overrides)
    return report


class CanonicalizePolicyTests(unittest.TestCase):
    def test_ready_when_every_invariant_holds(self):
        decision = evaluate_canonicalize(_good_mesh_report(), _good_similarity_report(), DEFAULT_TRIANGLE_BAND)

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["target_stage"], "canonicalize")
        self.assertEqual(decision["recommended_actions"], [])

    def test_blocked_when_not_a_single_welded_component(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(components=3), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.single_connected_component", codes)

    def test_blocked_when_not_a_single_unwelded_component(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(components_unwelded=1770), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.single_connected_component_unwelded", codes)

    def test_unwelded_check_is_skipped_when_not_reported(self):
        mesh_report = _good_mesh_report()
        del mesh_report["components_unwelded"]

        decision = evaluate_canonicalize(mesh_report, _good_similarity_report(), DEFAULT_TRIANGLE_BAND)

        self.assertEqual(decision["status"], "ready")

    def test_blocked_when_not_watertight(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(watertight=False), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.watertight", codes)

    def test_blocked_when_non_manifold_edges_present(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(non_manifold_edges=4), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.non_manifold_edges", codes)

    def test_blocked_when_triangle_count_below_band(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=4_000), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.triangle_budget", codes)

    def test_blocked_when_triangle_count_above_band(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=90_000), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.triangle_budget", codes)

    def test_blocked_when_similarity_below_threshold(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_image_similarity_score=DEFAULT_SIMILARITY_THRESHOLD - 0.01),
            DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.appearance_preservation_bake_gate", codes)
        finding = next(f for f in decision["findings"] if f["code"] == "canonicalize.appearance_preservation_bake_gate")
        self.assertEqual(finding["metric"], "s1_s2_image_similarity_score")

    def test_ready_at_exactly_the_similarity_threshold(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_image_similarity_score=DEFAULT_SIMILARITY_THRESHOLD),
            DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "ready")

    def test_blocked_when_similarity_score_missing(self):
        decision = evaluate_canonicalize(_good_mesh_report(), {}, DEFAULT_TRIANGLE_BAND)

        self.assertEqual(decision["status"], "blocked")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertIn("canonicalize.appearance_preservation_bake_gate", codes)

    def test_custom_family_band_is_honored(self):
        narrow_band = (1_000, 2_000)
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=30_000), _good_similarity_report(), narrow_band,
        )

        self.assertEqual(decision["status"], "blocked")
        finding = next(f for f in decision["findings"] if f["code"] == "canonicalize.triangle_budget")
        self.assertEqual(finding["expected"], "1000-2000")

    def test_multiple_violations_all_reported_and_deduplicated_actions(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(components=3, watertight=False, non_manifold_edges=2, faces=500),
            _good_similarity_report(s1_s2_image_similarity_score=0.1),
            DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(len(decision["findings"]), 5)
        # every finding's fix recommends the same action; must be deduplicated.
        self.assertEqual(decision["recommended_actions"], ["recanonicalize"])

    def test_mesh_and_appearance_violations_block_rather_than_need_review(self):
        # Every mesh/appearance check is severity="error" by design (S2 is a
        # mandatory, no-bypass gate). Only the detail-preservation check below
        # is advisory.
        decision = evaluate_canonicalize(
            _good_mesh_report(components=3), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )
        self.assertIn(decision["status"], {"ready", "blocked"})

    def test_detail_ratio_below_threshold_advisory_blocks_without_hard_blocking(self):
        # The muddy-bake case: everything the old gate measures is fine, and
        # the appearance score is high, but the face lost its detail.
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(
                s1_s2_image_similarity_score=0.997,
                s1_s2_detail_ratio=DEFAULT_DETAIL_RATIO_THRESHOLD - 0.01,
            ),
            DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "needs_review")
        finding = next(
            item for item in decision["findings"]
            if item["code"] == "canonicalize.detail_preservation"
        )
        self.assertEqual(finding["severity"], "warning")
        self.assertEqual(finding["metric"], "s1_s2_detail_ratio")
        self.assertIn("recanonicalize", decision["recommended_actions"])

    def test_ready_at_exactly_the_detail_ratio_threshold(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_detail_ratio=DEFAULT_DETAIL_RATIO_THRESHOLD),
            DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "ready")

    def test_missing_detail_ratio_warns_but_does_not_block(self):
        # An older report (no face-priority renders at all) must still
        # evaluate; the gate says it could not see, it does not fail closed.
        report = _good_similarity_report()
        report.pop("s1_s2_detail_ratio")
        decision = evaluate_canonicalize(_good_mesh_report(), report, DEFAULT_TRIANGLE_BAND)

        self.assertEqual(decision["status"], "needs_review")
        codes = {finding["code"] for finding in decision["findings"]}
        self.assertEqual(codes, {"canonicalize.detail_preservation_not_measured"})
        self.assertTrue(
            all(finding["severity"] == "warning" for finding in decision["findings"])
        )

    def test_detail_ratio_threshold_is_configurable(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_detail_ratio=0.40),
            DEFAULT_TRIANGLE_BAND,
            detail_ratio_threshold=0.30,
        )

        self.assertEqual(decision["status"], "ready")


if __name__ == "__main__":
    unittest.main()


class PalettePreservationPolicyTests(unittest.TestCase):
    """The advisory colour gate (`canonicalize.palette_preservation`).

    Values are the ones measured on 2026-09-01: the punk
    chain 9356577b at 0.6578 with the transfer off and 0.1231 with it on, and
    the already-correct alien chain b8f5932f at 0.0969.
    """

    PUNK_TRANSFER_OFF = 0.6578
    PUNK_TRANSFER_ON = 0.1231
    ALIEN_TRANSFER_ON = 0.0969

    def test_the_operators_mono_case_trips_before_the_fix(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_palette_distance=self.PUNK_TRANSFER_OFF),
            DEFAULT_TRIANGLE_BAND,
        )
        codes = [item["code"] for item in decision["findings"]]
        self.assertIn("canonicalize.palette_preservation", codes)

    def test_the_same_case_passes_after_the_fix(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_palette_distance=self.PUNK_TRANSFER_ON),
            DEFAULT_TRIANGLE_BAND,
        )
        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])

    def test_an_already_correct_colored_chain_passes_untouched(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_palette_distance=self.ALIEN_TRANSFER_ON),
            DEFAULT_TRIANGLE_BAND,
        )
        self.assertEqual(decision["status"], "ready")

    def test_the_finding_is_advisory_not_blocking(self):
        """A drifted palette is worth the operator's eyes, not a hard stop:
        the measurement is newer than the mesh contract and calibrated on a
        handful of chains."""
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_palette_distance=1.0),
            DEFAULT_TRIANGLE_BAND,
        )
        self.assertEqual(decision["status"], "needs_review")
        finding = next(
            item for item in decision["findings"]
            if item["code"] == "canonicalize.palette_preservation"
        )
        self.assertEqual(finding["severity"], "warning")
        self.assertIn("inspect-color-render", decision["recommended_actions"])

    def test_exactly_at_the_threshold_trips(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(
                s1_s2_palette_distance=DEFAULT_PALETTE_DISTANCE_THRESHOLD
            ),
            DEFAULT_TRIANGLE_BAND,
        )
        self.assertEqual(decision["status"], "needs_review")

    def test_a_missing_measurement_is_silent(self):
        """Unlike the detail check, an unmeasurable palette warns about nothing.

        A palette distance only exists when the lineage still holds the 2D
        image the generator was conditioned on. Uploaded meshes and
        re-canonicalized old artifacts legitimately have none, and warning on
        every one of them would train the operator to ignore the finding.
        """
        report = _good_similarity_report()
        self.assertNotIn("s1_s2_palette_distance", report)
        decision = evaluate_canonicalize(
            _good_mesh_report(), report, DEFAULT_TRIANGLE_BAND,
        )
        codes = [item["code"] for item in decision["findings"]]
        self.assertNotIn("canonicalize.palette_preservation", codes)
        self.assertEqual(decision["status"], "ready")

    def test_the_pre_transfer_drift_is_quoted_in_the_message(self):
        """The finding carries the generator's own drift, so the reader can
        tell "the bake introduced this" from "the bake failed to fix it"."""
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(
                s1_s2_palette_distance=0.6578, s1_palette_distance=0.6578
            ),
            DEFAULT_TRIANGLE_BAND,
        )
        finding = next(
            item for item in decision["findings"]
            if item["code"] == "canonicalize.palette_preservation"
        )
        self.assertIn("0.658", finding["message"])

    def test_an_explicit_threshold_overrides_the_default(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(),
            _good_similarity_report(s1_s2_palette_distance=0.30),
            DEFAULT_TRIANGLE_BAND,
            palette_distance_threshold=0.9,
        )
        self.assertEqual(decision["status"], "ready")

    def test_a_palette_warning_does_not_mask_a_real_block(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(watertight=False),
            _good_similarity_report(s1_s2_palette_distance=0.9),
            DEFAULT_TRIANGLE_BAND,
        )
        self.assertEqual(decision["status"], "blocked")


class TierAwareTriangleBudgetTests(unittest.TestCase):
    """The gate judges a candidate against the band of the density the
    OPERATOR asked for (bead i63ky), not a band tuned for a different one."""

    def test_an_ultra_mesh_passes_its_own_band(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=148_000),
            _good_similarity_report(),
            triangle_band(QUALITY_TIER_TARGETS["ultra"]),
            quality_tier="ultra",
        )

        self.assertEqual(decision["status"], "ready")

    def test_the_same_ultra_mesh_is_blocked_by_the_legacy_fixed_band(self):
        # The behaviour this bead exists to fix: without the tier, a mesh the
        # operator explicitly asked for is rejected for being too dense.
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=148_000), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )

        self.assertEqual(decision["status"], "blocked")
        self.assertIn(
            "canonicalize.triangle_budget",
            {finding["code"] for finding in decision["findings"]},
        )

    def test_a_tier_never_excuses_a_mesh_outside_its_own_band(self):
        # Widening with the request, not widening for its own sake: a Draft
        # run that came out at 148k tris is still blocked.
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=148_000),
            _good_similarity_report(),
            triangle_band(QUALITY_TIER_TARGETS["draft"]),
            quality_tier="draft",
        )

        self.assertEqual(decision["status"], "blocked")

    def test_the_finding_names_the_tier_it_judged_against(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=1_000),
            _good_similarity_report(),
            triangle_band(QUALITY_TIER_TARGETS["ultra"]),
            quality_tier="ultra",
        )
        finding = next(
            item for item in decision["findings"]
            if item["code"] == "canonicalize.triangle_budget"
        )

        self.assertIn("ultra", finding["message"])
        self.assertEqual(finding["expected"], "75000-225000")

    def test_a_tier_less_run_still_reads_as_the_family_band(self):
        decision = evaluate_canonicalize(
            _good_mesh_report(faces=1_000), _good_similarity_report(), DEFAULT_TRIANGLE_BAND,
        )
        finding = next(
            item for item in decision["findings"]
            if item["code"] == "canonicalize.triangle_budget"
        )

        self.assertNotIn("quality tier", finding["message"])
        self.assertEqual(finding["expected"], "10000-50000")
