"""S2 Canonicalization gate (docs/PIPELINE_POLICY.md).

The canonicalize stage is the mandatory, no-bypass raw-mesh -> game-ready-mesh
contract: one connected component, watertight, zero non-manifold edges, a
triangle count inside the family's game-ready band, and appearance
preservation -- the baked S2 texture must still look like the S1 source from
N matched viewpoints. Any violation blocks promotion; this module never
loosens the contract to make a candidate pass.

This module intentionally duplicates `quality_policy.py`'s `_finding`/
`_decision` dict shapes rather than importing them, so the canonicalize gate
stays decoupled from the rig/animation/deformation promotion gates another
agent owns concurrently. Keep the shapes in sync by hand if either changes.
"""
from __future__ import annotations

from typing import Any, Literal

Severity = Literal["warning", "error"]

# S2's explicit default band (docs/PIPELINE_POLICY.md: "triangle budget within
# the family's band (default 10-50k)"). A caller with a real family registry
# entry should pass its own band; this is the fallback for families that
# don't have one yet.
DEFAULT_TRIANGLE_BAND: tuple[int, int] = (10_000, 50_000)

# Appearance-preservation threshold on the aggregate similarity score (1.0 ==
# identical, 0.0 == unrelated). Calibrated 2026-08-30 against one
# character's S1 mesh during N1 live verification. Equivalent to requiring
# the combined MAD + histogram "distance" metric
# (native.compute_view_similarity) to stay below ~0.12.
DEFAULT_SIMILARITY_THRESHOLD = 0.88

# Detail-preservation threshold on `s1_s2_detail_ratio` (the ratio of
# high-frequency energy between the S2 and S1 face-priority renders; see
# native.compute_view_detail). 1.0 means the canonical bake is exactly as
# sharp as the source.
#
# This exists because the similarity score above is structurally blind to
# resolution loss: it compares 64x64 grayscale thumbnails of a whole-body
# render, where a character's face is a few dozen pixels. On 2026-09-01 the
# alien's canonical mesh scored 0.997 on that metric with a visibly muddy
# face -- caught by eye, not by the gate.
#
# Calibrated 2026-09-01 on that exact chain (S1 2ce7ee60, four matched
# face-priority views at 768px):
#   muddy bake  (canonical b063338b, 6.8% UV atlas utilisation) -> 0.507
#   fixed bake  (canonical 5a739cf8, 41.8% utilisation)         -> 0.684
# 0.60 is the midpoint: the bake the operator rejected by eye fails, the
# fixed one passes.
DEFAULT_DETAIL_RATIO_THRESHOLD = 0.60


# Palette-fidelity threshold on `s1_s2_palette_distance` (native.py's
# `merge_palette_measurement`: how far the baked albedo's colour distribution
# sits from the conditioning image the generator was given, 0.0 == matched).
#
# This exists because BOTH metrics above are computed on grayscale images, so
# neither can see colour at all. On 2026-09-01 the operator's run 62107a15
# canonicalized a strictly black-and-white punk character into a model with
# pink skin and blue jeans -- TRELLIS.2's texture head falling back to its own
# coloured prior because the conditioning image carried no palette to copy --
# and it scored 0.996 on appearance similarity while doing it.
#
# Calibrated 2026-09-01 on three real chains, measured on the unlit albedo
# atlas against each chain's conditioning image (see PALETTE_CALIBRATION):
#   punk 9356577b, transfer OFF (the shipped defect)  -> TRIPS
#   punk 9356577b, transfer ON  (the fix)             -> passes
#   alien b8f5932f, already-correct colour, ON and OFF-> passes untouched
# The threshold is the midpoint between the highest passing chain and the
# failing one.
DEFAULT_PALETTE_DISTANCE_THRESHOLD = 0.25


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


def _decision(
    target_stage: str,
    findings: list[dict[str, Any]],
    recommended_actions: list[str],
) -> dict[str, Any]:
    if any(item["severity"] == "error" for item in findings):
        status = "blocked"
    elif findings:
        status = "needs_review"
    else:
        status = "ready"
    return {
        "policy_version": 1,
        "target_stage": target_stage,
        "status": status,
        "findings": findings,
        "recommended_actions": list(dict.fromkeys(recommended_actions)),
    }


def _detail_findings(
    similarity_report: dict[str, Any], threshold: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Advisory detail-preservation findings (warnings, never errors).

    Deliberately advisory: the measurement is newer than the mesh contract
    and its threshold is calibrated on one chain, so a low ratio moves the
    decision to `needs_review` (the operator looks at the face render) rather
    than hard-blocking a candidate the rest of the contract accepts. A
    missing measurement is also only a warning, so a report written by an
    older worker -- which rendered no face-priority views at all -- still
    evaluates instead of failing closed on every historical artifact.
    """
    detail_ratio = similarity_report.get("s1_s2_detail_ratio")
    if detail_ratio is None:
        return [
            _finding(
                "canonicalize.detail_preservation_not_measured",
                "warning",
                "No detail-preservation measurement was recorded; the S1/S2 "
                "appearance score alone cannot see a texture-resolution loss "
                "(it compares 64x64 grayscale thumbnails of a whole-body "
                "render).",
                metric="s1_s2_detail_ratio",
                actual=None,
                expected=f">= {threshold}",
            )
        ], ["rerender-similarity-views"]
    if float(detail_ratio) < threshold:
        return [
            _finding(
                "canonicalize.detail_preservation",
                "warning",
                "The baked S2 texture carries visibly less high-frequency "
                "detail than the S1 source on the face-priority crop "
                "(09-01: a 6.8%-utilised UV atlas made the canonical face "
                "muddy while the appearance score still read 0.997).",
                metric="s1_s2_detail_ratio",
                actual=round(float(detail_ratio), 4),
                expected=f">= {threshold}",
            )
        ], ["recanonicalize", "inspect-face-render"]
    return [], []


def _palette_findings(
    similarity_report: dict[str, Any], threshold: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Advisory palette-fidelity findings (warnings, never errors).

    Advisory for the same reason `_detail_findings` is: the measurement is
    newer than the mesh contract and its threshold is calibrated on a handful
    of chains, so a drifted palette moves the decision to `needs_review` (the
    operator looks at the render) rather than hard-blocking a candidate the
    rest of the contract accepts.

    A MISSING measurement is silent here -- no finding at all -- which is the
    one place this differs from the detail check. A palette distance can only
    be measured when the artifact's lineage still holds the 2D image the
    generator was conditioned on, and plenty of legitimate artifacts have no
    such lineage: an uploaded mesh, a re-canonicalized old artifact, anything
    whose conditioning image has been cleaned up. Warning on all of those
    would train the operator to ignore the finding that matters.
    """
    distance = similarity_report.get("s1_s2_palette_distance")
    if distance is None:
        return [], []
    if float(distance) < threshold:
        return [], []
    before = similarity_report.get("s1_palette_distance")
    drift = (
        f" The generator's own output measured {float(before):.3f} before the bake."
        if before is not None
        else ""
    )
    return [
        _finding(
            "canonicalize.palette_preservation",
            "warning",
            "The baked S2 texture's colour distribution does not match the "
            "image the generator was conditioned on (09-01: a black-and-white "
            "source produced pink skin and blue jeans while the appearance "
            "score still read 0.996 -- that score is computed on grayscale "
            "thumbnails and is structurally blind to colour)." + drift,
            metric="s1_s2_palette_distance",
            actual=round(float(distance), 4),
            expected=f"< {threshold}",
        )
    ], ["recanonicalize", "inspect-color-render"]


def evaluate_canonicalize(
    mesh_report: dict[str, Any],
    similarity_report: dict[str, Any],
    family_band: tuple[int, int] = DEFAULT_TRIANGLE_BAND,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    detail_ratio_threshold: float = DEFAULT_DETAIL_RATIO_THRESHOLD,
    palette_distance_threshold: float = DEFAULT_PALETTE_DISTANCE_THRESHOLD,
    quality_tier: str | None = None,
) -> dict[str, Any]:
    """Evaluate an S2 canonicalize candidate against the mandatory contract.

    `mesh_report` is a `mesh_quality.analyze_mesh`-shaped dict (the
    canonicalized output's own measurement -- welded-topology components,
    watertight, non_manifold_edges, faces). It may optionally carry
    `components_unwelded` (native.run_canonicalize adds this); when present
    it is checked too, since a welded measure alone can miss seam-splitting
    duplicate geometry (docs/PIPELINE_POLICY.md's S2 contract explicitly
    calls out both measures -- the "08-30 Mary hips" blame-table case).

    `similarity_report` is a `native.evaluate_canonicalize_similarity`-shaped
    dict; only `s1_s2_image_similarity_score` is consulted here, but the
    full per-view detail should still be persisted alongside this decision
    for evidence.

    `family_band` is the (min, max) triangle-count band the candidate is judged
    against. Since bead i63ky that is the band of the DENSITY THE OPERATOR
    ASKED FOR -- `quality_tiers.resolve_quality_target(...).band`, recorded on
    the artifact as `triangle_budget_band` -- so selecting Ultra moves the gate
    with the request instead of blocking a 150k mesh against a 10-50k band.
    Callers with neither a tier nor a per-family registry entry still pass
    `DEFAULT_TRIANGLE_BAND`; this function never widens a band on its own.
    `quality_tier` is provenance only: it names the selected tier in the
    triangle-budget finding so a blocked candidate says which density it was
    judged against.

    Blocked (status="blocked") unless every one of: 1 connected component
    (welded, and unwelded when reported), watertight, 0 non-manifold edges,
    triangle count within `family_band`, and similarity >= threshold.
    """
    findings: list[dict[str, Any]] = []
    actions: list[str] = []

    components = int(mesh_report.get("components", 0))
    if components != 1:
        findings.append(
            _finding(
                "canonicalize.single_connected_component",
                "error",
                "Canonicalized mesh must be a single connected component "
                "(welded measure).",
                metric="components",
                actual=components,
                expected="1",
            )
        )
        actions.append("recanonicalize")

    if "components_unwelded" in mesh_report:
        components_unwelded = int(mesh_report.get("components_unwelded", 0))
        if components_unwelded != 1:
            findings.append(
                _finding(
                    "canonicalize.single_connected_component_unwelded",
                    "error",
                    "Canonicalized mesh must be a single connected component "
                    "under the unwelded (duplicate-vertex) measure -- catches "
                    "seam-splitting duplicate geometry a welded measure alone "
                    "would miss (08-30 Mary hips).",
                    metric="components_unwelded",
                    actual=components_unwelded,
                    expected="1",
                )
            )
            actions.append("recanonicalize")

    watertight = bool(mesh_report.get("watertight", False))
    if not watertight:
        findings.append(
            _finding(
                "canonicalize.watertight",
                "error",
                "Canonicalized mesh must be watertight.",
                metric="watertight",
                actual=watertight,
                expected="true",
            )
        )
        actions.append("recanonicalize")

    non_manifold_edges = int(mesh_report.get("non_manifold_edges", 0))
    if non_manifold_edges != 0:
        findings.append(
            _finding(
                "canonicalize.non_manifold_edges",
                "error",
                "Canonicalized mesh must have zero non-manifold edges.",
                metric="non_manifold_edges",
                actual=non_manifold_edges,
                expected="0",
            )
        )
        actions.append("recanonicalize")

    faces = int(mesh_report.get("faces", 0))
    band_low, band_high = family_band
    if not (band_low <= faces <= band_high):
        tier_note = (
            f" (selected quality tier: {quality_tier})" if quality_tier else ""
        )
        findings.append(
            _finding(
                "canonicalize.triangle_budget",
                "error",
                "Triangle count must fall within the game-ready budget band "
                "for the selected mesh density." + tier_note,
                metric="faces",
                actual=faces,
                expected=f"{band_low}-{band_high}",
            )
        )
        actions.append("recanonicalize")

    similarity_score = similarity_report.get("s1_s2_image_similarity_score")
    if similarity_score is None:
        findings.append(
            _finding(
                "canonicalize.appearance_preservation_bake_gate",
                "error",
                "No appearance-similarity measurement was recorded; the S2 "
                "bake gate cannot be evaluated without it.",
                metric="s1_s2_image_similarity_score",
                actual=None,
                expected=f">= {similarity_threshold}",
            )
        )
        actions.append("recanonicalize")
    elif float(similarity_score) < similarity_threshold:
        findings.append(
            _finding(
                "canonicalize.appearance_preservation_bake_gate",
                "error",
                "Baked S2 texture must preserve S1's appearance above the "
                "similarity threshold (08-30: re-generating the surface "
                "instead of baking destroyed the character's face -- an S2 "
                "violation that must never ship).",
                metric="s1_s2_image_similarity_score",
                actual=round(float(similarity_score), 4),
                expected=f">= {similarity_threshold}",
            )
        )
        actions.append("recanonicalize")

    detail_findings, detail_actions = _detail_findings(
        similarity_report, detail_ratio_threshold
    )
    findings.extend(detail_findings)
    actions.extend(detail_actions)

    palette_findings, palette_actions = _palette_findings(
        similarity_report, palette_distance_threshold
    )
    findings.extend(palette_findings)
    actions.extend(palette_actions)

    return _decision("canonicalize", findings, actions)


def decide_canonicalize(
    metadata: dict[str, Any],
    family_band_override: Any = None,
) -> dict[str, Any]:
    """The S2 gate decision for a finished `native.run_canonicalize` run.

    Shared by the app's job runner and the engine's `clean` stage, so both
    judge a candidate the same way. Judges the mesh canonicalize actually
    iterated on (its last attempt), not a re-measurement of the exported GLB:
    export and floor normalisation can change the GLB's own component count.

    Band precedence: an explicit `family_band` override, then the band of the
    selected density (`triangle_budget_band`, recorded by run_canonicalize),
    then the fixed legacy band. Records the band actually used back into
    `metadata["triangle_budget_band"]`, which pipeline_policy reads.
    """
    attempts = metadata.get("attempts") or []
    mesh_report = dict(attempts[-1]) if attempts else dict(metadata["mesh_quality"])
    similarity_report = metadata.get("similarity") or {}
    band = tuple(family_band_override or metadata.get("triangle_budget_band") or DEFAULT_TRIANGLE_BAND)
    metadata["triangle_budget_band"] = [int(band[0]), int(band[1])]
    return evaluate_canonicalize(
        mesh_report,
        similarity_report,
        band,
        quality_tier=metadata.get("quality_tier"),
    )
