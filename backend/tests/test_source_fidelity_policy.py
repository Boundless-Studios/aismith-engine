import unittest

from backend.app.source_fidelity_policy import (
    ADVISORY_METRICS,
    evaluate_source_fidelity,
)


def _good_report(**overrides):
    report = {
        "silhouette_iou": 0.95,
        "masked_ssim": 0.8,
        "face_ssim": 0.75,
        "perceptual_distance": 0.05,
        "masked_luma_delta": 0.01,
        "masked_rgb_l1": 0.02,
        "detail_ratio": 0.95,
        "face_detail_ratio": 0.9,
        "clip_i": 0.93,
        "clip_i_face": 0.91,
        "dino_cos": 0.88,
        "dino_cos_face": 0.85,
        "lpips": 0.11,
        "lpips_face": 0.12,
        "metric_warnings": [],
        "face_pixels": 4096,
        "alignment": {"bbox_width_ratio": 1.02},
        "camera": {"fov_x_degrees": 40.0},
        "policy_version": "source-fidelity-v1",
    }
    report.update(overrides)
    return report


class EvaluateSourceFidelityTests(unittest.TestCase):
    def test_green_when_every_measurement_is_healthy(self):
        decision = evaluate_source_fidelity(_good_report())
        self.assertEqual(decision["status"], "green")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["policy_version"], "source-fidelity-policy-v1")
        self.assertEqual(decision["summary"], {
            "silhouette_iou": 0.95,
            "masked_ssim": 0.8,
            "face_ssim": 0.75,
            "perceptual_distance": 0.05,
            "masked_luma_delta": 0.01,
            "masked_rgb_l1": 0.02,
            "detail_ratio": 0.95,
            "face_detail_ratio": 0.9,
            "clip_i": 0.93,
            "clip_i_face": 0.91,
            "dino_cos": 0.88,
            "dino_cos_face": 0.85,
            "lpips": 0.11,
            "lpips_face": 0.12,
            "metric_warnings": [],
        })

    def test_low_silhouette_iou_needs_review(self):
        decision = evaluate_source_fidelity(_good_report(silhouette_iou=0.5))
        self.assertEqual(decision["status"], "needs_review")
        codes = [f["code"] for f in decision["findings"]]
        self.assertIn("source_fidelity.silhouette", codes)

    def test_low_face_ssim_needs_review(self):
        decision = evaluate_source_fidelity(_good_report(face_ssim=0.2))
        codes = [f["code"] for f in decision["findings"]]
        self.assertIn("source_fidelity.face", codes)

    def test_face_ssim_none_reports_face_pixels_as_the_metric(self):
        decision = evaluate_source_fidelity(_good_report(face_ssim=None, face_pixels=12))
        finding = next(f for f in decision["findings"] if f["code"] == "source_fidelity.face")
        self.assertEqual(finding["metric"], "face_pixels")
        self.assertEqual(finding["actual"], 12)
        self.assertIn("face crop too small to measure", finding["message"].lower())

    def test_low_face_detail_ratio_is_a_detail_finding(self):
        decision = evaluate_source_fidelity(_good_report(face_detail_ratio=0.3))
        finding = next(f for f in decision["findings"] if f["code"] == "source_fidelity.detail")
        self.assertEqual(finding["metric"], "face_detail_ratio")
        self.assertEqual(finding["actual"], 0.3)
        self.assertEqual(decision["summary"]["face_detail_ratio"], 0.3)
        self.assertEqual(decision["summary"]["detail_ratio"], 0.95)

    def test_low_masked_ssim_needs_review(self):
        decision = evaluate_source_fidelity(_good_report(masked_ssim=0.1))
        codes = [f["code"] for f in decision["findings"]]
        self.assertIn("source_fidelity.appearance", codes)

    def test_masked_ssim_none_needs_review(self):
        decision = evaluate_source_fidelity(_good_report(masked_ssim=None))
        codes = [f["code"] for f in decision["findings"]]
        self.assertIn("source_fidelity.appearance", codes)

    def test_high_masked_rgb_l1_is_a_colour_finding(self):
        decision = evaluate_source_fidelity(_good_report(masked_rgb_l1=0.3))
        finding = next(f for f in decision["findings"] if f["code"] == "source_fidelity.colour")
        self.assertEqual(finding["metric"], "masked_rgb_l1")
        self.assertEqual(finding["actual"], 0.3)

    def test_wide_bbox_width_ratio_is_a_pose_width_finding(self):
        decision = evaluate_source_fidelity(
            _good_report(alignment={"bbox_width_ratio": 1.6})
        )
        finding = next(f for f in decision["findings"] if f["code"] == "source_fidelity.pose_width")
        self.assertEqual(finding["metric"], "alignment.bbox_width_ratio")
        self.assertEqual(finding["actual"], 1.6)

    def test_skipped_report_is_not_evaluable(self):
        decision = evaluate_source_fidelity({"skipped": "no photo"})
        self.assertEqual(decision["status"], "not_evaluable")
        self.assertEqual(len(decision["findings"]), 1)
        finding = decision["findings"][0]
        self.assertEqual(finding["code"], "source_fidelity.not_measured")
        self.assertIn("no photo", finding["message"])
        self.assertEqual(decision["recommended_actions"], [])

    def test_identity_metrics_are_advisory(self):
        # The three identity metrics at their worst possible values, with
        # every gated metric healthy: reported in full, still green. They have
        # no calibration on our assets, so they must not be able to move a
        # verdict on their own.
        decision = evaluate_source_fidelity(_good_report(
            clip_i=-1.0, clip_i_face=-1.0, dino_cos=-1.0, dino_cos_face=-1.0, lpips=1.0, lpips_face=1.0,
        ))
        self.assertEqual(decision["status"], "green")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["summary"]["clip_i"], -1.0)
        self.assertEqual(decision["summary"]["lpips_face"], 1.0)

    def test_a_subject_detail_ratio_alone_is_advisory(self):
        # A canonical render can carry MORE high-frequency energy than its
        # photo, so only the face ratio gates; the subject one is reported.
        decision = evaluate_source_fidelity(_good_report(detail_ratio=0.1))
        self.assertEqual(decision["status"], "green")
        self.assertEqual(decision["summary"]["detail_ratio"], 0.1)

    def test_unavailable_identity_metrics_are_reported_not_gated(self):
        decision = evaluate_source_fidelity(_good_report(
            metric_warnings=["clip_i unavailable: weights not cached"],
            **dict.fromkeys(m for m in ADVISORY_METRICS if m != "detail_ratio"),
        ))
        self.assertEqual(decision["status"], "green")
        self.assertIsNone(decision["summary"]["clip_i"])
        self.assertEqual(decision["summary"]["metric_warnings"], ["clip_i unavailable: weights not cached"])

    def test_recommended_actions_are_deduplicated(self):
        decision = evaluate_source_fidelity(
            _good_report(silhouette_iou=0.1, face_ssim=0.1, masked_rgb_l1=0.9)
        )
        self.assertEqual(
            decision["recommended_actions"].count("compare-against-source-photo"), 1
        )


if __name__ == "__main__":
    unittest.main()
