import unittest

from backend.app.projection_policy import evaluate_projection


def _good_report(**overrides):
    report = {
        "schema_version": 1,
        "views_used": ["back", "front", "left", "right"],
        "per_role_coverage": {"front": 0.4, "back": 0.3, "left": 0.2, "right": 0.25},
        "projected_texel_fraction": 0.72,
        "fallback_texel_fraction": 0.28,
        "occlusion_rejected_fraction": {"front": 0.05, "back": 0.1, "left": 0.02, "right": 0.03},
        "mean_color_delta_vs_bake": 0.04,
        "atlas_size": 2048,
        "frame_alignment": {
            role: {"scale": 1.0, "dx": 0.0, "dy": 0.0} for role in ("front", "back", "left", "right")
        },
        "caveats": [],
    }
    report.update(overrides)
    return report


class EvaluateProjectionTests(unittest.TestCase):
    def test_green_when_coverage_is_healthy(self):
        decision = evaluate_projection(_good_report())
        self.assertEqual(decision["status"], "green")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["policy_version"], "projection-v1")

    def test_needs_review_below_projected_texel_fraction_threshold(self):
        decision = evaluate_projection(_good_report(projected_texel_fraction=0.3))
        self.assertEqual(decision["status"], "needs_review")
        codes = [finding["code"] for finding in decision["findings"]]
        self.assertIn("project_views.low_projected_coverage", codes)

    def test_needs_review_missing_projected_texel_fraction(self):
        report = _good_report()
        del report["projected_texel_fraction"]
        decision = evaluate_projection(report)
        self.assertEqual(decision["status"], "needs_review")

    def test_warning_when_a_role_contributed_little(self):
        decision = evaluate_projection(
            _good_report(per_role_coverage={"front": 0.4, "back": 0.3, "left": 0.05, "right": 0.25})
        )
        self.assertEqual(decision["status"], "needs_review")
        finding = next(
            item for item in decision["findings"] if item["code"] == "project_views.low_role_coverage"
        )
        self.assertIn("left", finding["message"])
        self.assertEqual(finding["metric"], "per_role_coverage.left")

    def test_blocked_when_fewer_than_two_views_used(self):
        decision = evaluate_projection(_good_report(views_used=["front"]))
        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(len(decision["findings"]), 1)
        self.assertEqual(decision["findings"][0]["severity"], "error")
        self.assertEqual(decision["findings"][0]["code"], "project_views.insufficient_views")

    def test_blocked_when_no_views_used(self):
        decision = evaluate_projection(_good_report(views_used=[]))
        self.assertEqual(decision["status"], "blocked")

    def test_textured_texel_fraction_preferred_when_present(self):
        # projected_texel_fraction alone would trip the low-coverage finding,
        # but textured_texel_fraction (post-dilation, projected UNION filled)
        # clears the threshold and must be what the finding reads instead.
        decision = evaluate_projection(
            _good_report(projected_texel_fraction=0.3, textured_texel_fraction=0.9)
        )
        self.assertEqual(decision["status"], "green")
        self.assertEqual(decision["findings"], [])

    def test_textured_texel_fraction_is_the_reported_metric_when_it_trips(self):
        decision = evaluate_projection(
            _good_report(projected_texel_fraction=0.9, textured_texel_fraction=0.1)
        )
        finding = next(
            item for item in decision["findings"] if item["code"] == "project_views.low_projected_coverage"
        )
        self.assertEqual(finding["metric"], "textured_texel_fraction")
        self.assertEqual(finding["actual"], 0.1)

    def test_recommended_actions_are_deduplicated(self):
        decision = evaluate_projection(
            _good_report(
                projected_texel_fraction=0.1,
                per_role_coverage={"front": 0.4, "back": 0.01, "left": 0.02, "right": 0.25},
            )
        )
        self.assertEqual(
            decision["recommended_actions"].count("inspect-projection-coverage"), 1
        )


if __name__ == "__main__":
    unittest.main()
