"""S2 view-projection verdict (docs/PIPELINE_POLICY.md).

`view_projection.py` blends the T-pose conditioning views onto the canonical
mesh's texture, falling back to the existing bake wherever no view reaches a
texel; this module turns that pass's own coverage report into an operator-
facing verdict. Unlike `canonicalize_policy`'s mandatory contract, a poor
projection never blocks the mesh -- the fallback bake is always a complete,
shippable texture -- so the worst outcome here is "needs review", reserving
"blocked" for the one case where the pass could not run meaningfully at all
(fewer than two usable conditioning views).

This module intentionally duplicates `canonicalize_policy`'s `_finding` dict
shape rather than importing it, for the same reason that module gives for
duplicating `quality_policy`'s: this gate should stay decoupled from
S2/S3 verdicts owned by other work.
"""
from __future__ import annotations

from typing import Any, Literal

Severity = Literal["warning", "error"]

POLICY_VERSION = "projection-v1"

# Below this projected fraction, most of the atlas is still the pre-existing
# bake rather than conditioning-accurate evidence -- worth a look, never a
# block, since the fallback bake underneath is still a complete texture.
MIN_PROJECTED_TEXEL_FRACTION = 0.5
# A view that reached less than this fraction of the covered atlas
# contributed close to nothing -- usually a bad frame fit or an occluded
# camera angle -- and is worth flagging by name.
MIN_ROLE_COVERAGE = 0.15
# Below two views there is no meaningful multi-view projection to speak of;
# the pass could not do its job at all.
MIN_VIEWS_USED = 2


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


def evaluate_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a `view_projection.projection_report`-shaped dict.

    Blocked only when fewer than `MIN_VIEWS_USED` conditioning views
    contributed anything at all -- everything else is advisory, since a poor
    projection still leaves the mesh with its prior (TRELLIS-baked) texture
    intact.
    """
    views_used = report.get("views_used") or []
    if len(views_used) < MIN_VIEWS_USED:
        finding = _finding(
            "project_views.insufficient_views",
            "error",
            f"Only {len(views_used)} conditioning view(s) contributed to the "
            f"projection; at least {MIN_VIEWS_USED} are required.",
            metric="views_used",
            actual=len(views_used),
            expected=f">= {MIN_VIEWS_USED}",
        )
        return {
            "status": "blocked",
            "findings": [finding],
            "recommended_actions": ["resolve-conditioning-views"],
            "policy_version": POLICY_VERSION,
        }

    findings: list[dict[str, Any]] = []
    actions: list[str] = []

    # `textured_texel_fraction` (projected UNION filled, from dilation) is the
    # true "not still the blurred fallback bake" measure when a report has
    # one; an older report -- or dilation disabled -- falls back to the
    # pre-dilation `projected_texel_fraction`.
    coverage_metric = "projected_texel_fraction"
    coverage_fraction = report.get("projected_texel_fraction")
    if report.get("textured_texel_fraction") is not None:
        coverage_metric = "textured_texel_fraction"
        coverage_fraction = report.get("textured_texel_fraction")
    if coverage_fraction is None or float(coverage_fraction) < MIN_PROJECTED_TEXEL_FRACTION:
        findings.append(
            _finding(
                "project_views.low_projected_coverage",
                "warning",
                "Less than half the atlas received a projected texel; most of "
                "the texture still comes from the existing bake.",
                metric=coverage_metric,
                actual=(
                    round(float(coverage_fraction), 4) if coverage_fraction is not None else None
                ),
                expected=f">= {MIN_PROJECTED_TEXEL_FRACTION}",
            )
        )
        actions.append("inspect-projection-coverage")

    for role, coverage in (report.get("per_role_coverage") or {}).items():
        if float(coverage) < MIN_ROLE_COVERAGE:
            findings.append(
                _finding(
                    "project_views.low_role_coverage",
                    "warning",
                    f"view {role} contributed little",
                    metric=f"per_role_coverage.{role}",
                    actual=round(float(coverage), 4),
                    expected=f">= {MIN_ROLE_COVERAGE}",
                )
            )
            actions.append("inspect-projection-coverage")

    status = "needs_review" if findings else "green"
    return {
        "status": status,
        "findings": findings,
        "recommended_actions": list(dict.fromkeys(actions)),
        "policy_version": POLICY_VERSION,
    }


__all__ = ["evaluate_projection", "POLICY_VERSION"]
