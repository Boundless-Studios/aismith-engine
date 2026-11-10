"""S1/S2 source-fidelity verdict (docs/PIPELINE_POLICY.md).

`native.measure_source_fidelity` renders a mesh in its source photo's own
estimated framing and scores it against that photo; this module turns that
report into an operator-facing verdict. The mesh it measures is already a
complete, shippable asset -- a low score describes DRIFT from the source
photo, not a defect a downstream stage could reject, so every finding here
is a "warning": this gate never blocks.

This module intentionally duplicates `projection_policy`'s `_finding` shape
rather than importing it, for the same reason that module gives for
duplicating `canonicalize_policy`'s: this gate stays decoupled from S1/S2
verdicts owned by other work.
"""
from __future__ import annotations

from typing import Any, Literal

from .pipeline_policy import (
    SOURCE_FIDELITY_FACE_DETAIL_RATIO_THRESHOLD,
    SOURCE_FIDELITY_FACE_SSIM_THRESHOLD,
    SOURCE_FIDELITY_MASKED_SSIM_THRESHOLD,
    SOURCE_FIDELITY_RGB_L1_THRESHOLD,
    SOURCE_FIDELITY_SILHOUETTE_IOU_THRESHOLD,
    SOURCE_FIDELITY_WIDTH_RATIO_TOLERANCE,
)

Severity = Literal["warning"]

POLICY_VERSION = "source-fidelity-policy-v1"

# Reported, never gated on. Two separate reasons:
#
#   clip_i / dino_cos / lpips (and their _face variants) -- the learned
#   identity metrics. The literature's operating points are CLIP-I >= 0.90
#   (roughly where Hunyuan3D-Paint 2.1 sits) and LPIPS <= 0.13; DINOv2 cosine
#   has no published threshold anyone agrees on. None of the three has been
#   calibrated against OUR assets, so a threshold now would be a guess wearing
#   a number's clothes. They go in the summary so the calibration data
#   accumulates on real runs.
#
#   detail_ratio -- the SUBJECT-level sharpness ratio. A canonical render can
#   legitimately carry more high-frequency energy than its photo (remesh
#   faceting, UV seams), so only the FACE ratio, where that argument does not
#   apply, produces a finding.
ADVISORY_METRICS = (
    "detail_ratio",
    "clip_i",
    "clip_i_face",
    "dino_cos",
    "dino_cos_face",
    "lpips",
    "lpips_face",
)


def _finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    metric: str,
    actual: Any,
    expected: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metric": metric,
        "actual": actual,
        "expected": expected,
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "silhouette_iou": report.get("silhouette_iou"),
        "masked_ssim": report.get("masked_ssim"),
        "face_ssim": report.get("face_ssim"),
        "perceptual_distance": report.get("perceptual_distance"),
        "masked_luma_delta": report.get("masked_luma_delta"),
        "masked_rgb_l1": report.get("masked_rgb_l1"),
        "face_detail_ratio": report.get("face_detail_ratio"),
        **{metric: report.get(metric) for metric in ADVISORY_METRICS},
        # Why an identity metric above is None, when it is.
        "metric_warnings": list(report.get("metric_warnings") or []),
    }


def evaluate_source_fidelity(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a `native.measure_source_fidelity`-shaped report.

    Never blocks -- the worst status is "needs_review" for a measured
    report that drifted from its source photo, or "not_evaluable" for a
    report that could not be produced at all (no resolvable source photo).
    """
    skipped = report.get("skipped")
    if skipped is not None:
        return {
            "status": "not_evaluable",
            "findings": [
                _finding(
                    "source_fidelity.not_measured", "warning", str(skipped),
                    metric="skipped", actual=skipped, expected="a measured report",
                )
            ],
            "recommended_actions": [],
            "policy_version": POLICY_VERSION,
            "summary": _summary(report),
        }

    findings: list[dict[str, Any]] = []
    actions: list[str] = []

    silhouette_iou = report.get("silhouette_iou")
    if silhouette_iou is not None and silhouette_iou < SOURCE_FIDELITY_SILHOUETTE_IOU_THRESHOLD:
        findings.append(_finding(
            "source_fidelity.silhouette", "warning",
            "The mesh's silhouette overlaps the source photo's less than expected.",
            metric="silhouette_iou", actual=silhouette_iou,
            expected=f">= {SOURCE_FIDELITY_SILHOUETTE_IOU_THRESHOLD}",
        ))
        actions.append("compare-against-source-photo")

    face_ssim = report.get("face_ssim")
    if face_ssim is None:
        findings.append(_finding(
            "source_fidelity.face", "warning",
            "Face crop too small to measure.",
            metric="face_pixels", actual=report.get("face_pixels"),
            expected="a measurable face crop",
        ))
        actions.append("compare-against-source-photo")
    elif face_ssim < SOURCE_FIDELITY_FACE_SSIM_THRESHOLD:
        findings.append(_finding(
            "source_fidelity.face", "warning",
            "The mesh's face resembles the source photo's less than expected.",
            metric="face_ssim", actual=face_ssim,
            expected=f">= {SOURCE_FIDELITY_FACE_SSIM_THRESHOLD}",
        ))
        actions.append("compare-against-source-photo")

    face_detail_ratio = report.get("face_detail_ratio")
    if face_detail_ratio is not None and face_detail_ratio < SOURCE_FIDELITY_FACE_DETAIL_RATIO_THRESHOLD:
        findings.append(_finding(
            "source_fidelity.detail", "warning",
            "The render's face carries less than "
            f"{SOURCE_FIDELITY_FACE_DETAIL_RATIO_THRESHOLD:.0%} of the photo's high-frequency "
            "detail -- a blurred or washed texture.",
            metric="face_detail_ratio", actual=face_detail_ratio,
            expected=f">= {SOURCE_FIDELITY_FACE_DETAIL_RATIO_THRESHOLD}",
        ))
        actions.append("compare-against-source-photo")

    masked_ssim = report.get("masked_ssim")
    if masked_ssim is None or masked_ssim < SOURCE_FIDELITY_MASKED_SSIM_THRESHOLD:
        findings.append(_finding(
            "source_fidelity.appearance", "warning",
            "The mesh's overall appearance resembles the source photo's less than expected.",
            metric="masked_ssim", actual=masked_ssim,
            expected=f">= {SOURCE_FIDELITY_MASKED_SSIM_THRESHOLD}",
        ))

    masked_rgb_l1 = report.get("masked_rgb_l1")
    if masked_rgb_l1 is not None and masked_rgb_l1 > SOURCE_FIDELITY_RGB_L1_THRESHOLD:
        findings.append(_finding(
            "source_fidelity.colour", "warning",
            "The subject's colours differ from the photo's -- SSIM's luminance ratio is "
            "blind to lighter shifts such as skin vs mid-grey, so this is the check that "
            "catches a washed-out or re-tinted texture (masked_luma_delta="
            f"{report.get('masked_luma_delta')}).",
            metric="masked_rgb_l1", actual=masked_rgb_l1,
            expected=f"<= {SOURCE_FIDELITY_RGB_L1_THRESHOLD}",
        ))
        actions.append("compare-against-source-photo")

    alignment = report.get("alignment") or {}
    width_ratio = alignment.get("bbox_width_ratio")
    if width_ratio is not None and abs(width_ratio - 1) > SOURCE_FIDELITY_WIDTH_RATIO_TOLERANCE:
        findings.append(_finding(
            "source_fidelity.pose_width", "warning",
            "The subject's width after height-alignment differs from the photo's -- a pose "
            "mismatch such as a T-pose against arms-down.",
            metric="alignment.bbox_width_ratio", actual=width_ratio,
            expected=f"within {SOURCE_FIDELITY_WIDTH_RATIO_TOLERANCE} of 1.0",
        ))

    status = "needs_review" if findings else "green"
    return {
        "status": status,
        "findings": findings,
        "recommended_actions": list(dict.fromkeys(actions)),
        "policy_version": POLICY_VERSION,
        "summary": _summary(report),
    }


__all__ = ["ADVISORY_METRICS", "POLICY_VERSION", "evaluate_source_fidelity"]
