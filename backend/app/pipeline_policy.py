"""Policy-as-code for `docs/PIPELINE_POLICY.md` (workstream N6).

This module encodes the asset pipeline policy as data (stages, contracts,
the blame table) plus pure evaluation logic (contract checks, blame
resolution, the deterministic run/evaluate/stop-for-remediation loop). It
is deliberately standalone: it does not execute jobs, call Blender, touch
the filesystem, or import any in-flux module from this worktree's other
in-flight work (`skeleton_template.py`, the workers, `main.py`). Wiring
real measurements into `StageArtifact.metrics` and driving the loop from
job execution is later work -- see PIPELINE_POLICY.md's own note that this
module is "to be built -- workstream N" and the follow-on wiring bead.

Pure python + numpy only (no bpy, no scipy at module level) -- this
mirrors the same constraint `skeleton_naming.py` and `skeleton_template.py`
carry, since policy evaluation may eventually run inside Blender's bundled
interpreter as well as on the host.

Finding/decision dict shapes intentionally mirror `quality_policy.py`'s
`_finding`/`_decision` helpers (code/severity/message/metric/actual/expected
for findings; policy_version/status/findings for decisions) for consistency
across the codebase's policy modules, but this module does not import
`quality_policy` -- the shapes are duplicated on purpose to avoid coupling
two independently-evolving policy surfaces.

## How the pieces fit together

- `Stage` / `STAGE_ORDER`: the S0..S6 pipeline stages, in order.
- `StageArtifact`: one stage's recorded output -- the `metrics` a
  `ContractCheck` evaluates, plus `provenance` for the determinism
  principle.
- `ContractCheck` / `StageContract`: every invariant PIPELINE_POLICY.md
  states for a stage, as a declarative, independently testable unit. A
  check that needs a measurement nothing in the codebase computes yet
  (S2's bake image-similarity gate) declares `requires` naming that metric;
  the generic `ContractCheck.run` wrapper turns a missing/`None` required
  metric into a `"not-evaluable"` finding for EVERY check uniformly, so a
  contract can never silently "pass" for lack of data.
- `BLAME_TABLE` / `blame_stage` / `resolve_blame`: the doc's 7-row
  symptom -> owning-stage table as data, plus resolution (including the
  two rows whose owning stage is conditional on an earlier stage's
  contract having been green).
- `PipelineRun` / `next_action`: the pure run state machine implementing
  "evaluate stage by stage in order; on the first blocking violation,
  consult the blame table and stop; once every stage evaluates clean,
  done."
- `detect_stale_stages` / `StaleStage`: the other half of Principle 2's
  remediation loop -- "invalidate from the remediated stage, re-run
  downstream". A stage whose artifact does not descend from a remediated
  (or re-run) upstream stage's current artifact is stale, and
  `next_action` treats it as not-run so the loop re-enters it.
- `provenance_complete`: the standalone determinism-principle check,
  reused by S1's provenance contract check and available for any other
  stage that wants to assert it later.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Collection, Literal

Severity = Literal["warning", "error", "not-evaluable"]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class Stage(Enum):
    """PIPELINE_POLICY.md's S0..S6, in pipeline order."""

    S0_SOURCE = "S0_SOURCE"
    S1_GENERATION = "S1_GENERATION"
    S2_CANONICALIZATION = "S2_CANONICALIZATION"
    S3_SKELETON_FIT = "S3_SKELETON_FIT"
    S4_SKINNING = "S4_SKINNING"
    S5_ANIMATION = "S5_ANIMATION"
    S6_EXPORT = "S6_EXPORT"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.S0_SOURCE,
    Stage.S1_GENERATION,
    Stage.S2_CANONICALIZATION,
    Stage.S3_SKELETON_FIT,
    Stage.S4_SKINNING,
    Stage.S5_ANIMATION,
    Stage.S6_EXPORT,
)

# Finding/check ids follow the convention "s{n}.<slug>" (e.g. "s2.watertight")
# so a stage can always be recovered from a code alone -- this is the "safe
# default" the blame table falls back on. `provenance.*` codes (from
# `provenance_complete`) are the one deliberate exception; they are mapped
# to S1_GENERATION explicitly below since that is where the doc places the
# provenance requirement, even though the function itself is reusable.
_STAGE_CODE_PREFIX: dict[str, Stage] = {
    "s0": Stage.S0_SOURCE,
    "s1": Stage.S1_GENERATION,
    "s2": Stage.S2_CANONICALIZATION,
    "s3": Stage.S3_SKELETON_FIT,
    "s4": Stage.S4_SKINNING,
    "s5": Stage.S5_ANIMATION,
    "s6": Stage.S6_EXPORT,
    "provenance": Stage.S1_GENERATION,
}


def stage_index(stage: Stage) -> int:
    return STAGE_ORDER.index(stage)


def next_stage(stage: Stage) -> Stage | None:
    idx = stage_index(stage)
    if idx + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[idx + 1]


def _stage_from_code(finding_code: str) -> Stage:
    """The "safe default": blame the stage that emitted the finding, parsed
    from its id prefix. Falls back to the earliest stage (S0) for a code
    that matches no known prefix -- never raises, so blame resolution can
    never crash the loop."""
    prefix = finding_code.split(".", 1)[0].lower()
    return _STAGE_CODE_PREFIX.get(prefix, Stage.S0_SOURCE)


# ---------------------------------------------------------------------------
# Stage artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageArtifact:
    """One stage's recorded output.

    `metrics` carries whatever measurements that stage's contract checks
    evaluate (populated by wiring that is later work -- this module only
    consumes the dict). `provenance` carries the determinism-principle
    record (seed, tool/model versions, source artifact id, ...).
    `artifact_id` identifies this exact artifact for lineage/binding (S6's
    operator-accept gate binds to it).
    """

    stage: Stage
    artifact_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Findings / decisions
# ---------------------------------------------------------------------------


def finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    metric: str,
    actual: Any,
    expected: str,
) -> dict[str, Any]:
    """Build a finding dict. Shape mirrors `quality_policy._finding`."""
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metric": metric,
        "actual": actual,
        "expected": expected,
    }


def _not_evaluable_finding(check: "ContractCheck", missing: list[str]) -> dict[str, Any]:
    return finding(
        check.id,
        "not-evaluable",
        f"Cannot evaluate '{check.description}': required metric(s) not "
        f"recorded on this artifact yet: {', '.join(missing)}.",
        metric=missing[0] if len(missing) == 1 else ", ".join(missing),
        actual=None,
        expected=f"metric(s) present: {', '.join(missing)}",
    )


def _stage_decision(stage: Stage, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll a stage's findings up into a decision, matching
    `quality_policy._decision`'s status-precedence idea (error > everything
    else) but adding "not_evaluable" as its own status so a caller can tell
    "we checked and it's clean" apart from "we couldn't check"."""
    if any(f["severity"] == "error" for f in findings):
        status = "blocked"
    elif any(f["severity"] == "not-evaluable" for f in findings):
        status = "not_evaluable"
    elif findings:
        status = "needs_review"
    else:
        status = "green"
    return {
        "policy_version": 1,
        "stage": stage.value,
        "status": status,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------

CheckResult = dict[str, Any] | list[dict[str, Any]] | None


@dataclass(frozen=True)
class ContractCheck:
    """One declarative invariant from PIPELINE_POLICY.md.

    `evaluate` assumes every metric named in `requires` is present and
    non-None on the artifact -- `run()` is the entry point that enforces
    that precondition and turns a gap into a `"not-evaluable"` finding
    instead of calling `evaluate` (and instead of silently treating a
    missing metric as a pass).
    """

    id: str
    stage: Stage
    description: str
    severity: Severity
    requires: tuple[str, ...]
    evaluate: Callable[["StageArtifact"], CheckResult]

    def run(self, artifact: StageArtifact) -> list[dict[str, Any]]:
        missing = [
            key for key in self.requires
            if key not in artifact.metrics or artifact.metrics[key] is None
        ]
        if missing:
            return [_not_evaluable_finding(self, missing)]
        result = self.evaluate(artifact)
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]


@dataclass(frozen=True)
class StageContract:
    stage: Stage
    checks: tuple[ContractCheck, ...]

    def evaluate(self, artifact: StageArtifact) -> dict[str, Any]:
        if artifact.stage is not self.stage:
            raise ValueError(
                f"artifact is for {artifact.stage}, not {self.stage}"
            )
        findings: list[dict[str, Any]] = []
        for check in self.checks:
            findings.extend(check.run(artifact))
        return _stage_decision(self.stage, findings)


def evaluate_stage(stage: Stage, artifact: StageArtifact) -> dict[str, Any]:
    return CONTRACTS_BY_STAGE[stage].evaluate(artifact)


# ---------------------------------------------------------------------------
# Threshold constants
#
# Numbers the doc states explicitly are cited inline. Numbers the doc does
# NOT state a concrete value for (ROM/clip stretch thresholds, foot-contact
# band, symmetry/medial-line tolerances, proportion sanity, ambiguity
# cutoff) are placeholder defaults so every check is fully specified and
# testable now; they are named constants specifically so a later bead can
# retune them per-family without touching check logic. See the report's
# "ambiguities" section for the full list.
# ---------------------------------------------------------------------------

RESOLUTION_FLOOR_PX = 512  # S0: "resolution floor" -- no number given in doc
FAMILY_CONFIDENCE_AMBIGUOUS_THRESHOLD = 0.7  # S0: "ambiguity is an operator question"
DEFAULT_TRIANGLE_BUDGET: tuple[int, int] = (10_000, 50_000)  # S2: doc's explicit "default 10-50k"
BAKE_IMAGE_SIMILARITY_THRESHOLD = 0.9  # S2: bake gate -- now measured by source_fidelity.masked_ssim
SOURCE_FIDELITY_FACE_SSIM_THRESHOLD = 0.6  # S1/S2: face crop vs source photo
SOURCE_FIDELITY_SILHOUETTE_IOU_THRESHOLD = 0.85  # S1/S2: silhouette overlap vs source photo
SOURCE_FIDELITY_MASKED_SSIM_THRESHOLD = 0.5  # S1/S2: masked appearance vs source photo
SOURCE_FIDELITY_WIDTH_RATIO_TOLERANCE = 0.25  # S1/S2: post-align bbox width drift (pose mismatch)
SOURCE_FIDELITY_RGB_L1_THRESHOLD = 0.15  # S1/S2: masked colour error vs source photo
SOURCE_FIDELITY_FACE_DETAIL_RATIO_THRESHOLD = 0.6  # S1/S2: face high-frequency energy vs source photo
FLOOR_CONTACT_EPS = 1e-3  # S2: "floor-contact at z=0"
DIRECTION_TOLERANCE_DEGREES = 15.0  # S3: orientation -- mirrors skeleton_template.py's own default
ROLL_TOLERANCE_DEGREES = 20.0  # S3: roll -- mirrors skeleton_template.py's own default
PROPORTION_DEVIATION_THRESHOLD = 0.25  # S3: "proportion sanity vs the template"
MAX_INFLUENCES = 4  # S4: doc's explicit "Max 4 influences"
WEIGHT_SUM_EPS = 1e-3  # S4: "weights sum to 1"
ROM_EDGE_STRETCH_P99_THRESHOLD = 7.0  # S4: "p99 edge stretch ... under family thresholds"
# Retuned from the original 0.15 placeholder when the wiring bead (x8qhy)
# connected these checks to the codebase's ONLY calibrated stretch
# measurement, `deformation_review.maximum_p99_edge_stretch`. That number
# is a multiple of the rest edge length (1.0 == unchanged) and
# `quality_policy.evaluate_deformation_review` gates it at 8.0
# catastrophic / 6.0 needs-review. The adapter converts it to fractional
# stretch (measured - 1.0), so the equivalent blocking threshold is 7.0.
# Keeping 0.15 would have made every real clip fail a check that had
# never been calibrated against a real measurement -- precisely the
# blame table's "stretch metric red but visual clean -> blame the METRIC"
# row.
ROM_SELF_INTERSECTION_THRESHOLD = 0.0  # S4: "self-intersection under family thresholds"
# S5: PIPELINE_POLICY.md's "foot contact within band". Calibrated 2026-09-01
# (bead erbye) by `scripts/calibrate_foot_contact.py` over all 87 clips of
# `assets/skeletons/biped/clips/biped-clips-v1.glb`, sampled on the store's
# OWN template armature -- the ideal case, with no fitted rig, no skin and no
# retarget error available to blame. Contact-filtered stance-foot drift
# (`motion_review.foot_slide_max_in_contact`, in skeleton heights):
#
#   family       clips     min     p50     p75     p90     p95     max
#   locomotion       8  0.0285  0.3570  0.5573  0.5775  0.5880  0.5984
#   attack          28  0.0000  0.0016  0.1762  0.4465  0.7690  1.3293
#   idle            26  0.0000  0.0025  0.0100  0.0241  0.0639  0.1897
#   acrobatic       15  0.0000  0.0027  0.0850  0.6247  0.9854  1.1037
#   other           10  0.0000  0.0009  0.0016  0.0286  0.1480  0.2674
#   ALL             87  0.0000  0.0024  0.0850  0.3955  0.5895  1.3293
#
# There is no tight line, per family or overall, and the reason is that the
# measure charges an in-place locomotion cycle's STRIDE to slide.
# `Walk_Armature` has `root_travel_net` 0.0 -- the root never leaves the
# origin -- and still measures 0.3055, because the stance foot travels
# backwards under a stationary body, which is exactly what a correct in-place
# walk does. The old 0.02 placeholder failed 30 of the 87 reference clips;
# 0.2 fails 17; 0.6 still fails 4. Anything loose enough to pass the family's
# own authored clips is already blind to everything but gross breakage.
#
# So this is set as a GROSS-SLIP ADVISORY rather than a gate: 1.4 sits above
# the worst clip the curated store contains (`Sword_Regular_Combo_Armature`,
# 1.3293) with margin, and fails 0 of 87. An applied clip that exceeds it slid
# further, on this asset, than any authored clip does on the ideal skeleton --
# which can only be damage this asset's rig or skin introduced. The severity
# is "warning" for the same reason the number is loose: it cannot separate
# stride from slip, so it must never block a run on its own.
#
# The real gate wants either a per-clip reference table (this store's own
# per-clip measurement is the natural baseline: compare an applied clip
# against the same clip on the template) or a footlock-aware measure that
# knows the ground contract. Both are follow-ups, not a number.
FOOT_CONTACT_BAND_THRESHOLD = 1.4
# S3: PIPELINE_POLICY.md's "left/right joint placement within the family's
# symmetry bands", as the p95 over the template's mirrored pairs of
# `rig_quality.bilateral_symmetry`'s sagittal mirror error (in model
# heights).
#
# Calibrated 2026-09-04 by `scripts/calibrate_symmetry.py` over the GPU
# box's full artifact corpus (75 auto-rig/rig-adjust GLBs, 59 measurable --
# the other 16 lacked a confident facing detection and are reported
# unmeasurable, never as a false pass):
#
#   good rigs   0.0000-0.0117 (almost all), two outliers at 0.0233 / 0.0235
#               (artifacts 66e2e2fa, 8c3a3ba7)
#   known-bad   0.0505 (36bfef83, 2 pairs outside band)
#               0.0683 (7ccbf37f, 21), 0.0692 (5610c84a, 2)
#               0.0866 (c0297d5e, 23 -- the zix rig-adjust: a 2.49x
#                 length_ratio on the right finger chain, exactly the
#                 mirrored-rig defect this metric exists to catch)
#               0.0892 x3 (2f1bfcca, 72b01bc2, d4e9def0, 22 each)
#               0.1526 (def96dea, 22), 0.3321 (eb2b9225, 1)
#   accepted    edd0bb20 "punk v2" 0.0064, 6cdce431 "mary" 0.0043
#
# There is a clean gap between the highest good rig (0.0235) and the lowest
# known-bad one (0.0505): 0.03 sits in that gap. The severity stays
# "warning" -- a single calibration pass on today's corpus is not the same
# guarantee FOOT_CONTACT_BAND_THRESHOLD has, and the blame table's "metric
# red but the rig is fine -> blame the METRIC" row is exactly what an
# over-confident gate produces on a corpus this size.
#
# Rule: flag when mirror_error_p95 > 0.03.
SYMMETRY_MIRROR_ERROR_P95_THRESHOLD = 0.03
CLIP_STRETCH_P99_THRESHOLD = 7.0  # S5: "p99 stretch under threshold" (same calibration as ROM above)

# Registry families that exist today (docs/PIPELINE_POLICY.md's "Skeleton
# family registry"). Not enforced as a closed set by S0's checks -- the
# doc's own "process for adding a family" means the registry grows -- but
# useful as a reference / for tests.
KNOWN_FAMILIES: frozenset[str] = frozenset({
    "biped", "serpent", "quadruped-canine", "quadruped-ungulate", "bird",
})


# ---------------------------------------------------------------------------
# Small check-builder helpers (reduce boilerplate for the common shapes)
# ---------------------------------------------------------------------------


def _is_true_check(
    check_id: str,
    stage: Stage,
    description: str,
    metric_key: str,
    *,
    severity: Severity = "error",
    message: str | None = None,
    requires: tuple[str, ...] | None = None,
) -> ContractCheck:
    msg = message or f"{description}"

    def _evaluate(artifact: StageArtifact) -> dict[str, Any] | None:
        value = artifact.metrics[metric_key]
        if value is not True:
            return finding(check_id, severity, msg, metric=metric_key, actual=value, expected="true")
        return None

    return ContractCheck(
        id=check_id, stage=stage, description=description, severity=severity,
        requires=requires or (metric_key,), evaluate=_evaluate,
    )


def _is_false_check(
    check_id: str,
    stage: Stage,
    description: str,
    metric_key: str,
    *,
    severity: Severity = "error",
    message: str | None = None,
) -> ContractCheck:
    msg = message or f"{description}"

    def _evaluate(artifact: StageArtifact) -> dict[str, Any] | None:
        value = artifact.metrics[metric_key]
        if value is not False:
            return finding(check_id, severity, msg, metric=metric_key, actual=value, expected="false")
        return None

    return ContractCheck(
        id=check_id, stage=stage, description=description, severity=severity,
        requires=(metric_key,), evaluate=_evaluate,
    )


def _equals_check(
    check_id: str,
    stage: Stage,
    description: str,
    metric_key: str,
    expected_value: Any,
    *,
    severity: Severity = "error",
    message: str | None = None,
) -> ContractCheck:
    msg = message or f"{description}"

    def _evaluate(artifact: StageArtifact) -> dict[str, Any] | None:
        value = artifact.metrics[metric_key]
        if value != expected_value:
            return finding(
                check_id, severity, msg, metric=metric_key, actual=value,
                expected=repr(expected_value),
            )
        return None

    return ContractCheck(
        id=check_id, stage=stage, description=description, severity=severity,
        requires=(metric_key,), evaluate=_evaluate,
    )


def _max_threshold_check(
    check_id: str,
    stage: Stage,
    description: str,
    metric_key: str,
    threshold: float,
    *,
    severity: Severity = "error",
    message: str | None = None,
) -> ContractCheck:
    msg = message or f"{description}"

    def _evaluate(artifact: StageArtifact) -> dict[str, Any] | None:
        value = artifact.metrics[metric_key]
        if value > threshold:
            return finding(
                check_id, severity, msg, metric=metric_key, actual=value,
                expected=f"<= {threshold}",
            )
        return None

    return ContractCheck(
        id=check_id, stage=stage, description=description, severity=severity,
        requires=(metric_key,), evaluate=_evaluate,
    )


def _min_threshold_check(
    check_id: str,
    stage: Stage,
    description: str,
    metric_key: str,
    threshold: float,
    *,
    severity: Severity = "error",
    message: str | None = None,
) -> ContractCheck:
    msg = message or f"{description}"

    def _evaluate(artifact: StageArtifact) -> dict[str, Any] | None:
        value = artifact.metrics[metric_key]
        if value < threshold:
            return finding(
                check_id, severity, msg, metric=metric_key, actual=value,
                expected=f">= {threshold}",
            )
        return None

    return ContractCheck(
        id=check_id, stage=stage, description=description, severity=severity,
        requires=(metric_key,), evaluate=_evaluate,
    )


def _source_fidelity_silhouette_check(stage_prefix: str, stage: Stage) -> ContractCheck:
    return _min_threshold_check(
        f"{stage_prefix}.source_fidelity_silhouette", stage,
        "Silhouette IoU against the source photo rendered from its estimated camera.",
        "source_fidelity_silhouette_iou", SOURCE_FIDELITY_SILHOUETTE_IOU_THRESHOLD,
        severity="warning",
    )


def _source_fidelity_face_check(stage_prefix: str, stage: Stage) -> ContractCheck:
    return _min_threshold_check(
        f"{stage_prefix}.source_fidelity_face", stage,
        "Face-crop SSIM against the source photo.",
        "source_fidelity_face_ssim", SOURCE_FIDELITY_FACE_SSIM_THRESHOLD,
        severity="warning",
    )


# ---------------------------------------------------------------------------
# S0 Source & classification
# ---------------------------------------------------------------------------


def _s0_family_classified(artifact: StageArtifact) -> dict[str, Any] | None:
    family = artifact.metrics["subject_family"]
    if not isinstance(family, str) or not family.strip():
        return finding(
            "s0.family_classified", "error",
            "subject_family must be classified from the family registry.",
            metric="subject_family", actual=family, expected="non-empty family name",
        )
    return None


S0_CHECKS: tuple[ContractCheck, ...] = (
    _is_true_check(
        "s0.full_subject_visible", Stage.S0_SOURCE,
        "Full subject visible in the source image(s).", "full_subject_visible",
    ),
    _equals_check(
        "s0.single_subject", Stage.S0_SOURCE,
        "Exactly one subject present in the source image(s).", "subject_count", 1,
    ),
    _min_threshold_check(
        "s0.resolution_floor", Stage.S0_SOURCE,
        "Source image resolution meets the floor.", "min_resolution_px",
        RESOLUTION_FLOOR_PX,
    ),
    ContractCheck(
        id="s0.family_classified", stage=Stage.S0_SOURCE,
        description="subject_family classified from the registry, with confidence recorded.",
        severity="error", requires=("subject_family", "family_confidence"),
        evaluate=_s0_family_classified,
    ),
    _min_threshold_check(
        "s0.family_confidence_ambiguous", Stage.S0_SOURCE,
        "Family classification confidence is not ambiguous "
        "(low confidence is an operator question at S0, not a downstream surprise).",
        "family_confidence", FAMILY_CONFIDENCE_AMBIGUOUS_THRESHOLD, severity="warning",
    ),
)
S0_CONTRACT = StageContract(Stage.S0_SOURCE, S0_CHECKS)


# ---------------------------------------------------------------------------
# S1 Generation (image -> raw mesh)
# ---------------------------------------------------------------------------


def provenance_complete(artifact_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Principle 3 (Determinism): "Every artifact records provenance
    (inputs, settings, seed, stage impl version)." Requires, per the doc,
    a seed, model/tool version(s), and the source artifact id. Returns the
    (possibly empty) list of violations -- reusable standalone, and also
    what backs S1's `s1.provenance_recorded` contract check.
    """
    findings: list[dict[str, Any]] = []
    if artifact_meta.get("seed") is None:
        findings.append(finding(
            "provenance.missing_seed", "error",
            "Provenance is missing the generation seed required for a deterministic re-run.",
            metric="seed", actual=None, expected="present",
        ))
    if not artifact_meta.get("tool_versions"):
        findings.append(finding(
            "provenance.missing_tool_versions", "error",
            "Provenance is missing model/tool version identifiers.",
            metric="tool_versions", actual=artifact_meta.get("tool_versions"), expected="present",
        ))
    if not artifact_meta.get("source_artifact_id"):
        findings.append(finding(
            "provenance.missing_source_artifact_id", "error",
            "Provenance is missing the source artifact id this artifact was derived from.",
            metric="source_artifact_id", actual=artifact_meta.get("source_artifact_id"), expected="present",
        ))
    return findings


def _s1_provenance_recorded(artifact: StageArtifact) -> list[dict[str, Any]] | None:
    return provenance_complete(artifact.provenance) or None


S1_CHECKS: tuple[ContractCheck, ...] = (
    _is_true_check(
        "s1.recognizable_subject", Stage.S1_GENERATION,
        "Raw mesh depicts a recognizable single subject.", "recognizable_subject",
    ),
    _is_true_check(
        "s1.textures_present", Stage.S1_GENERATION,
        "Raw mesh has textures.", "textures_present",
    ),
    _is_true_check(
        "s1.source_views_recorded", Stage.S1_GENERATION,
        "Source view(s) used for generation are recorded in provenance.",
        "source_views_recorded",
    ),
    ContractCheck(
        id="s1.provenance_recorded", stage=Stage.S1_GENERATION,
        description="Provenance recorded (source views, seed, model). No topology "
                    "requirements here by design -- that is S2's job.",
        severity="error", requires=(), evaluate=_s1_provenance_recorded,
    ),
    _source_fidelity_silhouette_check("s1", Stage.S1_GENERATION),
    _source_fidelity_face_check("s1", Stage.S1_GENERATION),
)
S1_CONTRACT = StageContract(Stage.S1_GENERATION, S1_CHECKS)


# ---------------------------------------------------------------------------
# S2 Canonicalization (mandatory, no bypass)
# ---------------------------------------------------------------------------


def _s2_triangle_budget(artifact: StageArtifact) -> dict[str, Any] | None:
    count = artifact.metrics["triangle_count"]
    low, high = artifact.metrics.get("triangle_budget_band", DEFAULT_TRIANGLE_BUDGET)
    if not (low <= count <= high):
        tier = artifact.metrics.get("quality_tier")
        return finding(
            "s2.triangle_budget", "error",
            "Triangle count is outside the budget band for the selected mesh "
            "density." + (f" (quality tier: {tier})" if tier else ""),
            metric="triangle_count", actual=count, expected=f"[{low}, {high}]",
        )
    return None


def _s2_axis_convention(artifact: StageArtifact) -> dict[str, Any] | None:
    up_axis = artifact.metrics["up_axis"]
    facing_axis = artifact.metrics["facing_axis"]
    if up_axis != "Z" or facing_axis != "-Y":
        return finding(
            "s2.axis_convention", "error",
            "Mesh must use the uniform axis convention: Z-up, facing -Y.",
            metric="up_axis/facing_axis", actual=f"{up_axis}/{facing_axis}",
            expected="Z/-Y",
        )
    return None


S2_CHECKS: tuple[ContractCheck, ...] = (
    _equals_check(
        "s2.single_connected_component_welded", Stage.S2_CANONICALIZATION,
        "One connected component under the welded-vertex measure.",
        "connected_components_welded", 1,
    ),
    _equals_check(
        "s2.single_connected_component_unwelded", Stage.S2_CANONICALIZATION,
        "One connected component under the unwelded (duplicate-vertex) measure "
        "-- catches seam-splitting duplicate geometry a welded measure alone would miss.",
        "connected_components_unwelded", 1,
    ),
    _is_true_check(
        "s2.watertight", Stage.S2_CANONICALIZATION, "Mesh is watertight.", "watertight",
    ),
    _equals_check(
        "s2.non_manifold_edges", Stage.S2_CANONICALIZATION,
        "Zero non-manifold edges.", "non_manifold_edges", 0,
    ),
    ContractCheck(
        id="s2.triangle_budget", stage=Stage.S2_CANONICALIZATION,
        description=(
            "Triangle count within the budget band for the selected mesh density "
            "(the quality tier's band; 10-50k for a run that selected none)."
        ),
        severity="error", requires=("triangle_count",), evaluate=_s2_triangle_budget,
    ),
    _is_true_check(
        "s2.uv_unwrapped", Stage.S2_CANONICALIZATION, "Mesh is UV-unwrapped.", "uv_unwrapped",
    ),
    ContractCheck(
        id="s2.axis_convention", stage=Stage.S2_CANONICALIZATION,
        description=(
            "Uniform axis convention: Z-up, facing -Y. HALF-MEASURED: "
            "`up_axis` is recorded by `canonicalize.geometry_contract` "
            "(alongside `axis_convention_ok`, which asserts identity mesh-node "
            "transforms and no Y/Z swap against the source profile, and is the "
            "load-bearing axis evidence). `facing_axis` is NOT measured "
            "anywhere and is deliberately not guessed -- a subject-forward "
            "direction needs a rig (`skeleton_naming.detect_forward_from_feet` "
            "is the obvious source) and S2 runs before one exists. So this "
            "check stays not-evaluable, but now on `facing_axis` alone rather "
            "than on both halves, and it will never pass on half the evidence."
        ),
        severity="error", requires=("up_axis", "facing_axis"), evaluate=_s2_axis_convention,
    ),
    _is_true_check(
        "s2.uniform_scale", Stage.S2_CANONICALIZATION, "Mesh uses uniform scale.", "uniform_scale",
    ),
    _max_threshold_check(
        "s2.floor_contact", Stage.S2_CANONICALIZATION,
        "Mesh has floor-contact at z=0.", "floor_contact_z_abs", FLOOR_CONTACT_EPS,
    ),
    _min_threshold_check(
        "s2.appearance_preservation_bake_gate", Stage.S2_CANONICALIZATION,
        "Baked S2 texture preserves S1's appearance: image-similarity between S1 "
        "and S2 renders from N fixed viewpoints, above threshold. (08-30: "
        "re-generating the surface instead of baking destroyed the character's "
        "face -- an S2 violation that must never ship.) No image-similarity "
        "measurement exists in the codebase yet, so this check is structurally "
        "not-evaluable until that measurement is wired in.",
        "s1_s2_image_similarity_score", BAKE_IMAGE_SIMILARITY_THRESHOLD,
    ),
    _source_fidelity_silhouette_check("s2", Stage.S2_CANONICALIZATION),
    _source_fidelity_face_check("s2", Stage.S2_CANONICALIZATION),
)
S2_CONTRACT = StageContract(Stage.S2_CANONICALIZATION, S2_CHECKS)


# ---------------------------------------------------------------------------
# S3 Skeleton fit
# ---------------------------------------------------------------------------


def _s3_exact_bone_set(artifact: StageArtifact) -> dict[str, Any] | None:
    bone_names = set(artifact.metrics["bone_names"])
    template_names = set(artifact.metrics["template_bone_names"])
    if bone_names != template_names:
        return finding(
            "s3.exact_bone_set", "error",
            "Fitted rig's bone set must exactly match the family template's bone set "
            "(fitters only place joints; they never emit bespoke topology).",
            metric="bone_names", actual=sorted(bone_names), expected=sorted(template_names),
        )
    return None


def _s3_bone_hierarchy(artifact: StageArtifact) -> dict[str, Any] | None:
    parents = artifact.metrics["bone_parents"]
    template_parents = artifact.metrics["template_bone_parents"]
    if parents != template_parents:
        return finding(
            "s3.bone_hierarchy_matches_template", "error",
            "Fitted rig's bone hierarchy (and naming) must exactly match the family template.",
            metric="bone_parents", actual=parents, expected=template_parents,
        )
    return None


S3_CHECKS: tuple[ContractCheck, ...] = (
    ContractCheck(
        id="s3.exact_bone_set", stage=Stage.S3_SKELETON_FIT,
        description="Exact bone set matches the family template.",
        severity="error", requires=("bone_names", "template_bone_names"),
        evaluate=_s3_exact_bone_set,
    ),
    ContractCheck(
        id="s3.bone_hierarchy_matches_template", stage=Stage.S3_SKELETON_FIT,
        description="Exact bone hierarchy/names match the family template.",
        severity="error", requires=("bone_parents", "template_bone_parents"),
        evaluate=_s3_bone_hierarchy,
    ),
    _max_threshold_check(
        "s3.bone_rest_direction_matches_template", Stage.S3_SKELETON_FIT,
        "Bone rest directions (orientations) match the family template within tolerance "
        "(template orientations are law).",
        "max_rest_direction_deviation_degrees", DIRECTION_TOLERANCE_DEGREES,
    ),
    _max_threshold_check(
        "s3.bone_roll_matches_template", Stage.S3_SKELETON_FIT,
        "Bone rolls match the family template within tolerance (template rolls are law).",
        "max_roll_deviation_degrees", ROLL_TOLERANCE_DEGREES,
    ),
    _equals_check(
        "s3.joints_inside_mesh", Stage.S3_SKELETON_FIT,
        "All placed joints are inside the mesh.", "joints_outside_mesh_count", 0,
    ),
    _equals_check(
        "s3.left_right_symmetry", Stage.S3_SKELETON_FIT,
        "Left/right joint placement is within the family's symmetry bands.",
        "symmetry_band_violation_count", 0,
    ),
    # The geometric companion to `s3.left_right_symmetry` above: that one
    # counts NAMING mismatches from the conformance report, this one measures
    # how far the fitted left joints land from their right partners.
    _max_threshold_check(
        "s3.bilateral_symmetry_p95", Stage.S3_SKELETON_FIT,
        "Mirrored left/right joint placement is within the family's symmetry band "
        "(p95 of the sagittal mirror error, in model heights).",
        "symmetry_mirror_error_p95", SYMMETRY_MIRROR_ERROR_P95_THRESHOLD,
        severity="warning",
    ),
    _equals_check(
        "s3.medial_line_distance", Stage.S3_SKELETON_FIT,
        "Joints are within the family's medial-line distance bounds.",
        "medial_line_distance_violation_count", 0,
    ),
    _max_threshold_check(
        "s3.proportion_sanity", Stage.S3_SKELETON_FIT,
        "Fitted proportions are sane relative to the family template.",
        "proportion_deviation_ratio", PROPORTION_DEVIATION_THRESHOLD, severity="warning",
    ),
    _is_true_check(
        "s3.fit_converged", Stage.S3_SKELETON_FIT,
        "The fitter converged on a placement. A non-convergent fit is RCA'd to S2 "
        "(bad mesh) or S0 (misclassified family) per the doc -- it is never "
        "resolved by loosening the fit.",
        "fit_converged",
    ),
)
S3_CONTRACT = StageContract(Stage.S3_SKELETON_FIT, S3_CHECKS)

# The doc states a blame rule for "cannot fit" in prose, not in the 7-row
# blame table (which is reproduced verbatim in BLAME_TABLE below). Recorded
# separately so the 7-row table stays a faithful transcription of the doc's
# table while this rule is still honored by `resolve_blame`. The doc offers
# two candidates (S2 bad mesh, or S0 misclassified family) without deciding
# between them; S2 is picked as the default since it is listed first and is
# the more common cause in practice -- see the report's ambiguities section.
_FIT_NOT_CONVERGED_DEFAULT_OWNER = Stage.S2_CANONICALIZATION


# ---------------------------------------------------------------------------
# S4 Skinning
# ---------------------------------------------------------------------------


def _s4_solver_primary(artifact: StageArtifact) -> dict[str, Any] | None:
    solver = artifact.metrics["solver_used"]
    if solver != "primary":
        return finding(
            "s4.primary_solver_engaged", "error",
            "The primary solver (bone-heat / learned) must succeed; "
            "envelope-style fallbacks are guards only, and their engagement "
            "means the mesh made the primary solver fail (blame S2).",
            metric="solver_used", actual=solver, expected="primary",
        )
    return None


S4_CHECKS: tuple[ContractCheck, ...] = (
    ContractCheck(
        id="s4.primary_solver_engaged", stage=Stage.S4_SKINNING,
        description="Primary weight solver succeeded (no fallback engaged).",
        severity="error", requires=("solver_used",), evaluate=_s4_solver_primary,
    ),
    _max_threshold_check(
        "s4.max_influences", Stage.S4_SKINNING,
        "At most 4 bone influences per vertex.", "max_influences_per_vertex", MAX_INFLUENCES,
    ),
    _max_threshold_check(
        "s4.weights_sum_to_one", Stage.S4_SKINNING,
        "Skin weights sum to 1 per vertex.", "max_weight_sum_deviation", WEIGHT_SUM_EPS,
    ),
    _equals_check(
        "s4.coincident_vertices_share_weights", Stage.S4_SKINNING,
        "Coincident vertices share identical weights (assertion, not a fixup).",
        "coincident_weight_mismatch_count", 0,
    ),
    _equals_check(
        "s4.per_joint_coverage", Stage.S4_SKINNING,
        "Per-joint weight coverage is within the family's bands.",
        "joint_coverage_out_of_band_count", 0, severity="warning",
    ),
    _max_threshold_check(
        "s4.rom_edge_stretch_p99", Stage.S4_SKINNING,
        "ROM calibration clip: p99 edge stretch under the family threshold.",
        "rom_edge_stretch_p99", ROM_EDGE_STRETCH_P99_THRESHOLD,
    ),
    _max_threshold_check(
        "s4.rom_self_intersection", Stage.S4_SKINNING,
        "ROM calibration clip: self-intersection under the family threshold.",
        "rom_self_intersection_ratio", ROM_SELF_INTERSECTION_THRESHOLD,
    ),
    _is_false_check(
        "s4.rom_seam_tearing", Stage.S4_SKINNING,
        "No seam splitting/tearing observed while posing the ROM calibration clip.",
        "rom_seam_tearing_detected",
    ),
)
S4_CONTRACT = StageContract(Stage.S4_SKINNING, S4_CHECKS)


# ---------------------------------------------------------------------------
# S5 Animation
# ---------------------------------------------------------------------------


def _s5_direct_clip_attach(artifact: StageArtifact) -> dict[str, Any] | None:
    mechanism = artifact.metrics["mechanism"]
    if mechanism != "direct-attach":
        return finding(
            "s5.direct_clip_attach", "warning",
            "N5 contract: clips are pre-ported onto the family template and attach "
            "directly by exact bone name, with no per-asset retarget. A clip that "
            "arrived by retarget still plays, so this is advisory rather than "
            "blocking -- but it means this asset did not take the contract path.",
            metric="mechanism", actual=mechanism, expected="direct-attach",
        )
    return None


S5_CHECKS: tuple[ContractCheck, ...] = (
    _is_true_check(
        "s5.exact_bone_name_match", Stage.S5_ANIMATION,
        "Clip's bone names exactly match the family template (clips apply "
        "directly, no per-asset retarget).", "clip_bone_names_match_template",
    ),
    _is_true_check(
        "s5.root_motion_policy_honored", Stage.S5_ANIMATION,
        "Clip honors the family's root-motion policy.", "root_motion_policy_honored",
    ),
    _max_threshold_check(
        "s5.foot_contact_band", Stage.S5_ANIMATION,
        "Foot contact stays within the family's band. Calibrated over the "
        "whole biped clip store as a gross-slip advisory, not a gate -- see "
        "FOOT_CONTACT_BAND_THRESHOLD for the per-family distribution and for "
        "why a tight line would fail the family's own reference clips.",
        "foot_contact_error",
        FOOT_CONTACT_BAND_THRESHOLD, severity="warning",
    ),
    _max_threshold_check(
        "s5.stretch_p99_under_threshold", Stage.S5_ANIMATION,
        "p99 stretch on the actual clip is under threshold.", "clip_p99_stretch",
        CLIP_STRETCH_P99_THRESHOLD,
    ),
    _is_false_check(
        "s5.rest_pose_hunch_detected", Stage.S5_ANIMATION,
        "No hunch / inherited donor posture detected while playing the clip.",
        "hunch_detected",
    ),
    ContractCheck(
        id="s5.direct_clip_attach", stage=Stage.S5_ANIMATION,
        description="Clip arrived by direct attach onto the family template, not a per-asset retarget.",
        severity="warning", requires=("mechanism",), evaluate=_s5_direct_clip_attach,
    ),
)
S5_CONTRACT = StageContract(Stage.S5_ANIMATION, S5_CHECKS)


# ---------------------------------------------------------------------------
# S6 Export / accept
# ---------------------------------------------------------------------------


def _s6_operator_accept_recorded(artifact: StageArtifact) -> dict[str, Any] | None:
    decision = artifact.metrics["operator_accept_decision"]
    accepted_artifact_id = artifact.metrics["operator_accept_artifact_id"]
    if decision != "accept":
        return finding(
            "s6.operator_accept_recorded", "error",
            "Export requires a recorded operator accept decision.",
            metric="operator_accept_decision", actual=decision, expected="accept",
        )
    if accepted_artifact_id != artifact.artifact_id:
        return finding(
            "s6.operator_accept_recorded", "error",
            "Operator accept must bind to this exact artifact id.",
            metric="operator_accept_artifact_id", actual=accepted_artifact_id,
            expected=artifact.artifact_id,
        )
    return None


S6_CHECKS: tuple[ContractCheck, ...] = (
    ContractCheck(
        id="s6.operator_accept_recorded", stage=Stage.S6_EXPORT,
        description="Operator accept gate is recorded and binds to the exact artifact.",
        severity="error",
        requires=("operator_accept_decision", "operator_accept_artifact_id"),
        evaluate=_s6_operator_accept_recorded,
    ),
)
S6_CONTRACT = StageContract(Stage.S6_EXPORT, S6_CHECKS)


ALL_CONTRACTS: tuple[StageContract, ...] = (
    S0_CONTRACT, S1_CONTRACT, S2_CONTRACT, S3_CONTRACT, S4_CONTRACT, S5_CONTRACT, S6_CONTRACT,
)
CONTRACTS_BY_STAGE: dict[Stage, StageContract] = {c.stage: c for c in ALL_CONTRACTS}


# ---------------------------------------------------------------------------
# Blame table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlameRow:
    """One row of PIPELINE_POLICY.md's blame table.

    `owning_stage` is the default/primary owner; `None` is reserved for the
    one row whose owner is "the METRIC" itself, not a pipeline stage (fix
    the measurement, file it -- there is nothing to re-run).
    `conditional_stage`/`condition_check_stage` capture the two rows whose
    blame depends on run history ("S4 if S2 contract was green", "S5 only
    if S3 green; else S3") -- resolved by `resolve_blame`, which is the
    only place that consults a `PipelineRun`. `blame_stage` (no run
    context) always resolves to the row's default `owning_stage`.
    """

    symptom_code: str
    symptom: str
    finding_codes: tuple[str, ...]
    owning_stage: Stage | None
    rationale: str
    incident_ref: str
    conditional_stage: Stage | None = None
    condition_check_stage: Stage | None = None
    condition_description: str | None = None


BLAME_TABLE: tuple[BlameRow, ...] = (
    BlameRow(
        symptom_code="seams_split_tearing",
        symptom="Seams split / tearing in pose",
        finding_codes=("s4.rom_seam_tearing",),
        owning_stage=Stage.S2_CANONICALIZATION,
        conditional_stage=Stage.S4_SKINNING,
        condition_check_stage=Stage.S2_CANONICALIZATION,
        condition_description="S2 contract was green",
        rationale="unwelded dupes",
        incident_ref="08-30 Mary hips",
    ),
    BlameRow(
        symptom_code="wrong_mirrored_limb_naming",
        symptom="Wrong/mirrored limb naming",
        finding_codes=("s3.exact_bone_set", "s3.bone_hierarchy_matches_template"),
        owning_stage=Stage.S3_SKELETON_FIT,
        rationale="template conformance",
        incident_ref="08-29 mirrored rig",
    ),
    BlameRow(
        symptom_code="hunch_inherited_donor_posture",
        symptom="Hunch / inherited donor posture",
        finding_codes=("s5.rest_pose_hunch_detected",),
        owning_stage=Stage.S3_SKELETON_FIT,
        conditional_stage=Stage.S5_ANIMATION,
        condition_check_stage=Stage.S3_SKELETON_FIT,
        condition_description="S3 green",
        rationale="torso clamp; else S3 orientations",
        incident_ref="08-30 torso clamp",
    ),
    BlameRow(
        symptom_code="limb_roll_twist",
        symptom="Limb roll/twist",
        finding_codes=("s3.bone_roll_matches_template",),
        owning_stage=Stage.S3_SKELETON_FIT,
        rationale="template rolls are law",
        incident_ref="08-30 elbow",
    ),
    BlameRow(
        symptom_code="face_texture_degraded",
        symptom="Face/texture degraded",
        finding_codes=("s2.appearance_preservation_bake_gate",),
        owning_stage=Stage.S2_CANONICALIZATION,
        rationale="bake gate",
        incident_ref="08-30 AI-Refine regression",
    ),
    BlameRow(
        symptom_code="stretch_metric_red_visual_clean",
        symptom="Stretch metric red but visual clean",
        finding_codes=("s4.rom_edge_stretch_p99", "s5.stretch_p99_under_threshold"),
        owning_stage=None,
        rationale="the METRIC (fix measurement, file it)",
        incident_ref="seam-triangle pollution",
    ),
    BlameRow(
        symptom_code="solver_fallback_engaged",
        symptom="Solver fallback engaged",
        finding_codes=("s4.primary_solver_engaged",),
        owning_stage=Stage.S2_CANONICALIZATION,
        rationale="envelope weights saga",
        incident_ref="envelope weights saga",
    ),
)

_BLAME_ROW_BY_FINDING_CODE: dict[str, BlameRow] = {
    code: row for row in BLAME_TABLE for code in row.finding_codes
}


def blame_stage(finding_code: str) -> Stage:
    """Context-free blame resolution: the row's default owning stage, or
    the safe default (the stage that emitted the finding) for a code not
    in the table (or for the one row whose owner is "the METRIC", which
    is not itself a re-runnable stage)."""
    row = _BLAME_ROW_BY_FINDING_CODE.get(finding_code)
    if row is None:
        if finding_code == "s3.fit_converged":
            return _FIT_NOT_CONVERGED_DEFAULT_OWNER
        return _stage_from_code(finding_code)
    return row.owning_stage if row.owning_stage is not None else _stage_from_code(finding_code)


def _stage_is_strictly_green(run: "PipelineRun", stage: Stage) -> bool:
    """"Contract was green" is read strictly here: no findings at all (not
    even warnings) -- a `needs_review` (warnings-only) verdict is treated
    as not-yet-green for the purpose of the two conditional blame rows,
    since a warning still means the stage hasn't fully certified clean."""
    artifact = run.stage_artifacts.get(stage)
    if artifact is None:
        return False
    decision = evaluate_stage(stage, artifact)
    return decision["status"] == "green"


def resolve_blame(finding_code: str, run: "PipelineRun | None" = None) -> Stage:
    """Run-history-aware blame resolution. Without a run, behaves exactly
    like `blame_stage`. With a run, applies the two conditional rows
    (seams -> S4 if S2 was green; hunch -> S5 if S3 was green)."""
    row = _BLAME_ROW_BY_FINDING_CODE.get(finding_code)
    if row is None:
        return blame_stage(finding_code)
    if run is not None and row.conditional_stage is not None and row.condition_check_stage is not None:
        if _stage_is_strictly_green(run, row.condition_check_stage):
            return row.conditional_stage
    return row.owning_stage if row.owning_stage is not None else _stage_from_code(finding_code)


# ---------------------------------------------------------------------------
# The deterministic loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineRun:
    """Pure run state. Immutable -- `with_stage_artifact` returns a new
    run rather than mutating this one (Principle 4: artifacts, and by
    extension run state, are never edited in place).

    `seeds` and `findings_history` are the run's lineage/audit trail
    (Principle 3 determinism, Principle 4 immutability): `seeds` records
    every seed spent across the run (S1's generation seed and any
    downstream stochastic seeds), `findings_history` is the append-only
    log of every finding surfaced across every evaluation performed on
    this run so far.
    """

    source: str
    family: str
    seeds: tuple[int, ...] = ()
    stage_artifacts: dict[Stage, StageArtifact] = field(default_factory=dict)
    findings_history: tuple[dict[str, Any], ...] = ()

    def with_stage_artifact(self, artifact: StageArtifact) -> "PipelineRun":
        new_artifacts = dict(self.stage_artifacts)
        new_artifacts[artifact.stage] = artifact
        decision = evaluate_stage(artifact.stage, artifact)
        return dataclasses.replace(
            self,
            stage_artifacts=new_artifacts,
            findings_history=self.findings_history + tuple(decision["findings"]),
        )


@dataclass(frozen=True)
class Advance:
    """Run this stage next; every prior stage evaluates clean enough to proceed."""
    stage: Stage


@dataclass(frozen=True)
class StopForRemediation:
    """A stage's contract was violated. `triggering_stage` observed the
    violation; `stage` (per the blame table) is where the remediation
    lands and where the run resumes after it does -- "invalidate from the
    remediated stage, re-run downstream, re-evaluate contracts.\""""
    stage: Stage
    triggering_stage: Stage
    findings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Done:
    """S6 is green: an S6-green run, reproducible from its recorded provenance."""


PipelineAction = Advance | StopForRemediation | Done


def next_action(
    run: PipelineRun, stale: "Collection[Stage]" = ()
) -> PipelineAction:
    """Pure, deterministic: same `run.stage_artifacts` always yields the
    same action (no side effects, no job submission -- the orchestrator
    that submits stage jobs is later work).

    Walks S0..S6 in order. The first stage with no recorded artifact yet
    is next to run (`Advance`). The first stage whose recorded artifact's
    contract is `"blocked"` (has an error finding) stops the run: the
    owning stage is resolved via the blame table (earliest-stage-wins if
    multiple error findings implicate different stages, per Principle 2:
    "assigned to the earliest stage whose invariant would have prevented
    it"). `needs_review` (warnings only) and `not_evaluable` stages do not
    block -- there's nothing actionable to remediate from a missing
    measurement or an advisory warning -- so the walk continues past them.
    Once every stage from S0 through S6 has an artifact and none is
    blocked, the run is `Done`.

    `stale` names stages whose recorded artifact was built on a
    now-superseded upstream artifact (see `detect_stale_stages`). They are
    treated exactly as if they had never run: PIPELINE_POLICY.md's loop
    after a remediation is "invalidate from the remediated stage, re-run
    downstream", so the loop must advance INTO the stale stage rather than
    step over work that describes the pre-fix asset. Their contract
    verdicts are still reported -- `summarize_pipeline_policy` keeps them,
    flagged -- they just cannot satisfy a stage or stop the run.
    """
    stale_stages = frozenset(stale)
    for stage in STAGE_ORDER:
        artifact = run.stage_artifacts.get(stage)
        if artifact is None or stage in stale_stages:
            return Advance(stage=stage)
        decision = evaluate_stage(stage, artifact)
        if decision["status"] == "blocked":
            error_findings = tuple(f for f in decision["findings"] if f["severity"] == "error")
            owning_stages = [resolve_blame(f["code"], run) for f in error_findings]
            owning_stage = min(owning_stages, key=stage_index)
            return StopForRemediation(
                stage=owning_stage, triggering_stage=stage, findings=error_findings,
            )
    return Done()


# ===========================================================================
# Wiring: real job metadata -> StageArtifact metrics (bead x8qhy)
#
# Everything above this line is the pure policy: stages, contracts, the
# blame table, and the run loop. Everything below is the *adapter* layer
# that maps what this codebase actually measures onto those contracts, so
# real runs can be evaluated instead of hand-built fixtures.
#
# Two rules govern this layer, and they are the whole point of it:
#
#  1. **Never fabricate a value.** A contract check whose required metric
#     has no measurement anywhere in the pipeline must reach
#     `ContractCheck.run` with that metric absent, so it surfaces as the
#     check's own "not-evaluable" finding. There is no default, no
#     optimistic fill-in, and no "assume it passed because the job
#     completed".
#  2. **Say where every value came from.** Each populated metric is
#     recorded in `METRIC_SOURCES`, and `summarize_pipeline_policy`
#     emits the sources for the metrics it actually populated. A reader
#     of a policy block can always answer "where did this number come
#     from?" without reading this file.
#
# One convention needs stating explicitly, because it is the only place a
# number is *derived* rather than read: several recorded measurements are
# **violation lists**, not magnitudes -- `skeleton_template.conformance_report`
# records a `rest_direction_deviation` finding only for bones that exceed
# the tolerance, and records nothing for bones that pass. For those, the
# adapter reports the maximum deviation actually recorded, and `0.0` when
# the report ran and recorded none. That `0.0` means "the conformance
# report, run at exactly this check's tolerance, found no deviation worth
# recording" -- it is a floor, not a measured maximum. Metrics derived
# this way are labelled "(floor)" in `METRIC_SOURCES` so the derivation is
# never invisible. When the report was never produced at all
# (`template_conformance is None` -- a chain rig, or a rig predating the
# template contract) nothing is populated and every dependent check is
# honestly not-evaluable.
# ===========================================================================


# glTF/GLB is exported Y-up (`backend/workers/blender_worker.py`'s
# `export_yup`), while the family house frame is Z-up. `mesh_quality`
# measures bounds on the *exported* GLB, so the up axis in a recorded
# `bounds` triple is index 1.
GLB_UP_AXIS_INDEX = 1

# `weighting` values emitted by the rig workers. Only Blender's bone-heat
# solver is the doc's "primary solver"; the envelope and nearest-bone
# paths are exactly the "envelope-style fallbacks are guards only" the S4
# contract names, and the proxy/donor paths are neither.
PRIMARY_WEIGHT_SOLVER = "blender-bone-heat"

# Job kinds (`metadata["stage"]`) mapped onto the policy's stages. One
# artifact can carry two stages: a rig job places joints (S3) *and* solves
# weights (S4) in a single pass, so its metadata feeds both contracts.
#
# `retopologize`/`retopo` are deliberately absent: they are mesh cleanup
# passes that PIPELINE_POLICY.md's S0..S6 does not model, and mapping them
# onto S1 would let a retopo artifact (which records no generation seed or
# model) mask the real S1 generation artifact's provenance.
ARTIFACT_STAGES: dict[str, tuple[Stage, ...]] = {
    "prepare-image": (Stage.S0_SOURCE,),
    "synthesize-views": (Stage.S0_SOURCE,),
    # Bead 4vquq's composite: the same S0 reference work (T-pose front plus
    # the three views fanned off it) done as one job instead of two.
    "create-tpose-set": (Stage.S0_SOURCE,),
    "generate": (Stage.S1_GENERATION,),
    # `native.run_trellis_fp8`/`run_trellis_gguf` overwrite metadata["stage"]
    # with their own label, so the generate job's output is recorded as
    # "textured-geometry", not "generate".
    "textured-geometry": (Stage.S1_GENERATION,),
    "canonicalize": (Stage.S2_CANONICALIZATION,),
    # Projects the T-pose conditioning views onto the canonical texture; an
    # S2 refinement pass over an already-canonicalized mesh, not a stage of
    # its own.
    "project-views": (Stage.S2_CANONICALIZATION,),
    # Repaints the canonical texture from agent-picked reference views (the
    # approach9 agent-in-the-loop step); like project-views it refines the
    # canonicalized mesh's texture and is a Clean-stage output in the studio
    # (frontend journey/workspace CLEAN_OUTPUT_STAGES), so it speaks for S2.
    "rebuild-views": (Stage.S2_CANONICALIZATION,),
    "auto-rig": (Stage.S3_SKELETON_FIT, Stage.S4_SKINNING),
    "rig-adjust": (Stage.S3_SKELETON_FIT, Stage.S4_SKINNING),
    "rig-transfer": (Stage.S3_SKELETON_FIT, Stage.S4_SKINNING),
    "target-proxy-rig": (Stage.S3_SKELETON_FIT, Stage.S4_SKINNING),
    "repair-weights": (Stage.S4_SKINNING,),
    "apply-clip": (Stage.S5_ANIMATION,),
    "retarget-motion": (Stage.S5_ANIMATION,),
    "generate-motion": (Stage.S5_ANIMATION,),
}

# ---------------------------------------------------------------------------
# The bounded loop driver's stage table (bead hlxij)
#
# `next_action` says *what* to do; this table says *what that costs a GPU*.
# It is the inverse of ARTIFACT_STAGES above: which job kind produces the
# artifact a stage is evaluated from. Only stages this codebase can run
# head-lessly, from an input the run already holds, appear here.
#
# Deliberately absent, each for its own reason:
#  - S0/S1: their input is a *creative decision* (which reference image,
#    which prompt/seed to spend). A loop driver that re-generates geometry
#    on its own initiative is exactly the unattended GPU spend this whole
#    design refuses; those stages are `STAGE_MANUAL_INPUT` instead.
#  - S6: the operator-accept gate is a human decision by definition. There
#    is no job to submit -- somebody has to look at the asset.
#
# S5 is the interesting one: it HAS a job kind (`apply-clip`), but the job
# is meaningless without a clip name, and picking a clip is choosing what
# the character *does*. So it is both mapped and listed in
# `STAGE_CREATIVE_INPUT`: the driver may submit it, but only once a caller
# has named the clip. See `stage_step_kind`.
# ---------------------------------------------------------------------------

STAGE_JOB_KINDS: dict[Stage, str] = {
    Stage.S2_CANONICALIZATION: "canonicalize",
    # One auto-rig pass places the joints (S3) and solves the weights (S4),
    # which is why both stages map onto the same kind -- the mirror image of
    # ARTIFACT_STAGES mapping one "auto-rig" artifact onto both stages.
    Stage.S3_SKELETON_FIT: "auto-rig",
    Stage.S4_SKINNING: "auto-rig",
    Stage.S5_ANIMATION: "apply-clip",
}

# Stages whose next step is a human action, and what that action is.
STAGE_MANUAL_INPUT: dict[Stage, str] = {
    Stage.S0_SOURCE: "source-image",
    Stage.S1_GENERATION: "generation-request",
    Stage.S6_EXPORT: "operator-accept",
}

# Stages whose job kind exists but needs a creative choice named first, and
# the settings key that choice arrives under.
STAGE_CREATIVE_INPUT: dict[Stage, str] = {
    Stage.S5_ANIMATION: "clip",
}

# How many times one explicit invocation series may re-run the SAME stage
# for remediation before it stops asking the GPU and starts asking a human.
# Two is "the retry that fixes a flake, plus one" -- a third identical
# re-run of a deterministic stage is a machine repeating itself.
DEFAULT_STAGE_ATTEMPT_CAP = 2

# Key under a job's `settings` (and therefore under the run manifest's
# attempt settings, and the output artifact's `metadata["settings"]`) that
# carries a step's provenance. One key, three places, no extra schema.
PIPELINE_STEP_SETTINGS_KEY = "pipeline_step"


def stage_job_kind(stage: Stage) -> str | None:
    """The job kind that produces `stage`'s artifact, or None when the stage
    is a human action rather than a job."""
    return STAGE_JOB_KINDS.get(stage)


def stage_step_kind(stage: Stage) -> tuple[str | None, str | None]:
    """`(job_kind, input_required)` for one step at `stage`.

    Exactly one of the two is non-None for every stage the loop can reach:
    a stage either has a job to submit or a named human input it is waiting
    on. S5 reports both -- it has a job kind AND wants a clip named -- and
    the caller resolves that by supplying the clip.
    """
    return STAGE_JOB_KINDS.get(stage), (
        STAGE_MANUAL_INPUT.get(stage) or STAGE_CREATIVE_INPUT.get(stage)
    )


def step_provenance(
    *,
    mode: Literal["advance", "remediation"],
    stage: Stage,
    evaluated_at: str,
    policy_action: str,
    attempt: int,
    attempt_cap: int,
    source_artifact_id: str | None,
    triggering_stage: Stage | None = None,
    finding_codes: tuple[str, ...] | list[str] = (),
    remediates_attempt_id: str | None = None,
) -> dict[str, Any]:
    """The provenance block stamped on every job a pipeline step submits.

    `initiator` is always "api": a step only ever happens because somebody
    called the endpoint. `policy_evaluated_at` is the timestamp of the
    evaluation that *chose* this job, so a job can always be traced back to
    the exact policy verdict that asked for it -- the determinism principle
    applied to the loop itself, not just to the artifacts.
    """
    block: dict[str, Any] = {
        "initiator": "api",
        "mode": mode,
        "stage": stage.value,
        "policy_evaluated_at": evaluated_at,
        "policy_action": policy_action,
        "attempt": attempt,
        "attempt_cap": attempt_cap,
        "source_artifact_id": source_artifact_id,
    }
    if triggering_stage is not None:
        block["triggering_stage"] = triggering_stage.value
    if finding_codes:
        block["finding_codes"] = list(finding_codes)
    if remediates_attempt_id is not None:
        block["remediates_attempt_id"] = remediates_attempt_id
    return block


def step_settings_block(settings: dict[str, Any] | None) -> dict[str, Any]:
    """The `pipeline_step` provenance recorded on one attempt's settings, or
    `{}` for an attempt no pipeline step submitted."""
    if not isinstance(settings, dict):
        return {}
    block = settings.get(PIPELINE_STEP_SETTINGS_KEY)
    return block if isinstance(block, dict) else {}


def count_stage_remediations(
    attempt_settings: "list[dict[str, Any]] | tuple[dict[str, Any], ...]", stage: Stage
) -> int:
    """How many remediation jobs a pipeline step has already spent on
    `stage` in this run.

    Counted straight off the run manifest's own recorded attempt settings --
    the cap needs no side table and survives a backend restart, because the
    evidence it counts IS the run's history. Every submitted job counts,
    including one that failed or was cancelled: the GPU time was spent
    either way, and that is what the cap bounds.
    """
    return sum(
        1
        for settings in attempt_settings
        if step_settings_block(settings).get("mode") == "remediation"
        and step_settings_block(settings).get("stage") == stage.value
    )


# metric key -> where the adapter read (or derived) it from. Emitted with
# every evaluation so a value's provenance is never a code-reading exercise.
METRIC_SOURCES: dict[str, str] = {
    # S1
    "textures_present": "metadata.textured (TRELLIS worker result)",
    "source_views_recorded": "metadata.source_views / metadata.source_image_artifact_id",
    # S1 provenance
    "seed": "metadata.seed",
    "tool_versions": "metadata.{backend,model,resolution}",
    "source_artifact_id": "metadata.source_artifact_id",
    # S2
    "connected_components_welded": "metadata.attempts[-1].components (canonicalize gate report), else mesh_quality.components",
    "connected_components_unwelded": "metadata.attempts[-1].components_unwelded (canonicalize gate report)",
    "watertight": "metadata.attempts[-1].watertight, else mesh_quality.watertight",
    "non_manifold_edges": "metadata.attempts[-1].non_manifold_edges, else mesh_quality.non_manifold_edges",
    "triangle_count": "metadata.attempts[-1].faces, else mesh_quality.faces",
    "triangle_budget_band": "metadata.settings.family_band, else metadata.triangle_budget_band (the selected quality tier's band -- quality_tiers.resolve_quality_target)",
    "quality_tier": "metadata.quality_tier (the operator's selected mesh density; absent on pre-i63ky artifacts)",
    "target_triangles": "metadata.target_triangles (the tier's resolved triangle target)",
    "s1_s2_image_similarity_score": "metadata.similarity.s1_s2_image_similarity_score (native.evaluate_canonicalize_similarity)",
    "floor_contact_z_abs": f"|mesh_quality.bounds.minimum[{GLB_UP_AXIS_INDEX}]| (GLB is exported Y-up)",
    "uv_unwrapped": "metadata.uv_unwrapped, else geometry_contract.uv_unwrapped (measured UV area per triangle -- not 'a UV layer exists')",
    "uniform_scale": "metadata.uniform_scale_ok, else geometry_contract.uniform_scale_ok (per-object scale anisotropy + identity mesh-node transforms)",
    "up_axis": "geometry_contract.up_axis (Blender glTF import convention; the load-bearing evidence is geometry_contract.axis_convention_ok)",
    "projected_texel_fraction": "metadata.view_projection.projected_texel_fraction (project-views report, view_projection.projection_report)",
    "source_fidelity_silhouette_iou": "metadata.source_fidelity.silhouette_iou (native.measure_source_fidelity, S1 or S2)",
    "source_fidelity_face_ssim": "metadata.source_fidelity.face_ssim (native.measure_source_fidelity, S1 or S2)",
    # S3
    "bone_names": "rig_quality.joint_names",
    "template_bone_names": "rig_quality.joint_names minus unknown_bone_name findings, plus missing_required_bone findings (rig_quality.template_conformance)",
    "bone_parents": "rig_quality.template_conformance reparented_bone findings (actual parents)",
    "template_bone_parents": "rig_quality.template_conformance reparented_bone findings (template parents)",
    "max_rest_direction_deviation_degrees": "max rest_direction_deviation finding in rig_quality.template_conformance (floor)",
    "max_roll_deviation_degrees": "max roll_deviation finding in rig_quality.template_conformance (floor)",
    "symmetry_band_violation_count": "count of mirrored_left_right + mirrored_side_position findings in rig_quality.template_conformance",
    "fit_converged": "rig_quality.joints > 0 (the bind worker raises rather than emit a looser fit)",
    "joints_outside_mesh_count": "rig_quality.joints_outside_mesh_count (joint_containment, ray-blocking-5-axes -- shell-aware; see docs/PIPELINE_POLICY.md's S2 hollow-shell note)",
    "symmetry_mirror_error_p95": "rig_quality.bilateral_symmetry mirror_error_p95 (sagittal-mirror-bind-pose, in model heights; measured only when the rig's facing was detected confidently)",
    "symmetry_pairs_outside_band_count": "rig_quality.bilateral_symmetry pairs_outside_band_count (pairs whose mirror error exceeds bilateral_symmetry.SYMMETRY_MIRROR_BAND)",
    # S4
    "solver_used": "metadata.weighting",
    "max_influences_per_vertex": "rig_quality.maximum_influences",
    "max_weight_sum_deviation": "0.0 iff rig_quality.non_normalized_vertices == 0 (the magnitude itself is not recorded)",
    "coincident_weight_mismatch_count": "metadata.seam_stitched_vertices (coincident vertices that did NOT share weights)",
    "rom_edge_stretch_p99": "deformation_review.maximum_p99_edge_stretch - 1.0 (rest-length multiplier -> fractional stretch)",
    # S5
    "clip_bone_names_match_template": "rig_quality.animations[clip].invalid_targets == 0",
    "clip_p99_stretch": "deformation_review.maximum_p99_edge_stretch - 1.0 (rest-length multiplier -> fractional stretch)",
    "mechanism": "metadata.mechanism",
    "foot_contact_error": "deformation_review.motion_review.foot_slide_max_in_contact (contact-filtered stance-foot drift, in skeleton heights; populated only when contact_stance_episodes > 0, never from the over-counting raw foot_slide_max)",
    "static_tail_fraction": "deformation_review.motion_review.static_tail_fraction",
    "joint_motion_coverage": "deformation_review.motion_review.joint_motion_coverage",
    # S6
    "operator_accept_decision": "run manifest review_status",
    "operator_accept_artifact_id": "run manifest reviewed_artifact_id",
}

_REPARENTED_BONE_RE = re.compile(r"^bone '([^']+)' has parent ")
_MISSING_BONE_RE = re.compile(r"^'([^']+)' present$")


def _first_present(*candidates: tuple[dict[str, Any], str]) -> Any:
    """First non-None value of `source[key]` across the given (source, key)
    pairs, or None when none of them carries the key."""
    for source, key in candidates:
        if isinstance(source, dict) and source.get(key) is not None:
            return source[key]
    return None


def _fractional_stretch(deformation_review: dict[str, Any]) -> float | None:
    """`deformation_review.maximum_p99_edge_stretch` is a multiple of the
    rest edge length (1.0 == unchanged), while the policy's stretch checks
    threshold *fractional* stretch. Convert, and only when the review
    actually covered the whole clip -- a partial review is not evidence."""
    if not isinstance(deformation_review, dict):
        return None
    if deformation_review.get("complete_action_range") is not True:
        return None
    measured = deformation_review.get("maximum_p99_edge_stretch")
    if measured is None:
        return None
    return round(float(measured) - 1.0, 6)


def _s0_source_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Nothing in this codebase classifies a subject family, counts
    subjects, or records source resolution, so every S0 check is honestly
    not-evaluable. Kept as a named function so the gap is explicit rather
    than an omission."""
    return {}


def _source_fidelity_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """`source_fidelity_silhouette_iou`/`_face_ssim`, shared by S1 and S2 --
    the same `main._stamp_source_fidelity` report judges both stages'
    output against the same source photo. Absent on a skipped report."""
    report = metadata.get("source_fidelity")
    if not isinstance(report, dict) or "skipped" in report:
        return {}
    metrics: dict[str, Any] = {}
    if report.get("silhouette_iou") is not None:
        metrics["source_fidelity_silhouette_iou"] = float(report["silhouette_iou"])
    if report.get("face_ssim") is not None:
        metrics["source_fidelity_face_ssim"] = float(report["face_ssim"])
    return metrics


def _s1_generation_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if metadata.get("textured") is not None:
        metrics["textures_present"] = bool(metadata["textured"])
    # A generate job always records which image(s) conditioned it, so this
    # is genuinely measurable either way (True or False), never absent.
    metrics["source_views_recorded"] = bool(
        metadata.get("source_views") or metadata.get("source_image_artifact_id")
    )
    # `recognizable_subject` has no measurement -- deliberately unmapped.
    metrics.update(_source_fidelity_metrics(metadata))
    return metrics


def _s1_generation_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    if metadata.get("seed") is not None:
        provenance["seed"] = metadata["seed"]
    tool_versions = {
        key: metadata[key]
        for key in ("backend", "model", "resolution")
        if metadata.get(key) is not None
    }
    if tool_versions:
        provenance["tool_versions"] = tool_versions
    if metadata.get("source_artifact_id"):
        provenance["source_artifact_id"] = metadata["source_artifact_id"]
    return provenance


def _s2_canonicalization_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """S2 reads the canonicalize job's own gate report -- the last entry of
    `metadata["attempts"]`, which `main._execute_job` already feeds to
    `evaluate_canonicalize`. That report measures the OBJ that was actually
    repaired and baked; `mesh_quality` (a re-measurement of the exported
    GLB) is the fallback for the metrics the gate report does not carry."""
    attempts = metadata.get("attempts") or []
    gate = attempts[-1] if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else {}
    mesh = metadata.get("mesh_quality") or {}
    similarity = metadata.get("similarity") or {}

    metrics: dict[str, Any] = {}
    components = _first_present((gate, "components"), (mesh, "components"))
    if components is not None:
        metrics["connected_components_welded"] = int(components)
    if gate.get("components_unwelded") is not None:
        metrics["connected_components_unwelded"] = int(gate["components_unwelded"])
    watertight = _first_present((gate, "watertight"), (mesh, "watertight"))
    if watertight is not None:
        metrics["watertight"] = bool(watertight)
    non_manifold = _first_present((gate, "non_manifold_edges"), (mesh, "non_manifold_edges"))
    if non_manifold is not None:
        metrics["non_manifold_edges"] = int(non_manifold)
    faces = _first_present((gate, "faces"), (mesh, "faces"))
    if faces is not None:
        metrics["triangle_count"] = int(faces)
    # Band precedence mirrors `main`'s gate call: an explicit per-request
    # `settings.family_band` override first, then the band belonging to the
    # mesh density the operator selected (bead i63ky -- `quality_tiers`, which
    # `run_canonicalize` records and `main` writes back as the band the gate
    # actually used). An artifact carrying NEITHER is a pre-tier run, and the
    # check falls through to `DEFAULT_TRIANGLE_BUDGET` exactly as before --
    # a tier-less legacy artifact is never silently judged against a wider band.
    band = _first_present(
        ((metadata.get("settings") or {}), "family_band"),
        (metadata, "triangle_budget_band"),
    )
    if isinstance(band, (list, tuple)) and len(band) == 2:
        metrics["triangle_budget_band"] = (int(band[0]), int(band[1]))
    tier = metadata.get("quality_tier")
    if tier is not None:
        metrics["quality_tier"] = str(tier)
    if metadata.get("target_triangles") is not None:
        metrics["target_triangles"] = int(metadata["target_triangles"])

    score = _first_present(
        (metadata, "s1_s2_image_similarity_score"),
        (similarity, "s1_s2_image_similarity_score"),
    )
    if score is not None:
        metrics["s1_s2_image_similarity_score"] = float(score)

    minimum = ((mesh.get("bounds") or {}).get("minimum")) if isinstance(mesh, dict) else None
    if isinstance(minimum, (list, tuple)) and len(minimum) > GLB_UP_AXIS_INDEX:
        metrics["floor_contact_z_abs"] = round(abs(float(minimum[GLB_UP_AXIS_INDEX])), 6)

    # Recorded by `native.run_canonicalize` from `geometry_contract` and
    # mirrored at the top level of the canonicalize result -- the same
    # top-level-mirror pattern `s1_s2_image_similarity_score` uses. Absent on
    # artifacts canonicalized before that measurement landed, which is exactly
    # when these checks must stay not-evaluable instead of assuming a pass.
    contract = metadata.get("geometry_contract") or {}
    uv_unwrapped = _first_present((metadata, "uv_unwrapped"), (contract, "uv_unwrapped"))
    if uv_unwrapped is not None:
        metrics["uv_unwrapped"] = bool(uv_unwrapped)
    uniform_scale = _first_present(
        (metadata, "uniform_scale_ok"), (contract, "uniform_scale_ok")
    )
    if uniform_scale is not None:
        metrics["uniform_scale"] = bool(uniform_scale)
    if contract.get("up_axis") is not None:
        metrics["up_axis"] = str(contract["up_axis"])

    # `facing_axis` stays unmapped on purpose -- nothing measures a subject's
    # forward direction before a rig exists. See `s2.axis_convention`.
    metrics.update(_source_fidelity_metrics(metadata))
    return metrics


def _conformance_findings(metadata: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The recorded `skeleton_template.conformance_report` for this rig, or
    None when one was never produced (a chain rig, or a rig predating the
    template contract). None is the honest "cannot evaluate S3"."""
    rig_quality = metadata.get("rig_quality")
    if not isinstance(rig_quality, dict):
        return None
    findings = rig_quality.get("template_conformance")
    return findings if isinstance(findings, list) else None


def _max_finding_value(findings: list[dict[str, Any]], code: str) -> float:
    values = [
        float(item["actual"])
        for item in findings
        if item.get("code") == code and isinstance(item.get("actual"), (int, float))
    ]
    return max(values) if values else 0.0


def _s3_skeleton_fit_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    rig_quality = metadata.get("rig_quality") or {}
    metrics: dict[str, Any] = {}

    joints = rig_quality.get("joints")
    if isinstance(joints, int) and joints > 0:
        # The bind worker raises outright when a required bone has no
        # prediction and no synthesis rule ("cannot fit blames S2/S0, never
        # a looser fit"), so an artifact that exists at all is a fit that
        # converged.
        metrics["fit_converged"] = True

    # `rig_quality` mirrors the joint-containment count at its top level under
    # the exact key this check asks for. Read before the conformance
    # early-return below, so a chain rig (which produces no template
    # conformance report) still gets a containment verdict. Absent when
    # `joint_containment` was unmeasurable -- no joints, or no skinned
    # triangles to test against -- and absent on rigs that predate it.
    joints_outside = rig_quality.get("joints_outside_mesh_count")
    if isinstance(joints_outside, int) and not isinstance(joints_outside, bool):
        metrics["joints_outside_mesh_count"] = joints_outside

    # Same story for the measured bilateral symmetry: mirrored at the top
    # level of `rig_quality`, read before the conformance early-return, and
    # absent both on rigs that predate the measurement and on rigs whose
    # facing could not be detected confidently enough to define a plane.
    symmetry_p95 = rig_quality.get("symmetry_mirror_error_p95")
    if isinstance(symmetry_p95, (int, float)) and not isinstance(symmetry_p95, bool):
        metrics["symmetry_mirror_error_p95"] = float(symmetry_p95)
    symmetry_outside = rig_quality.get("symmetry_pairs_outside_band_count")
    if isinstance(symmetry_outside, int) and not isinstance(symmetry_outside, bool):
        metrics["symmetry_pairs_outside_band_count"] = symmetry_outside

    conformance = _conformance_findings(metadata)
    joint_names = rig_quality.get("joint_names")
    if conformance is None or not isinstance(joint_names, list):
        return metrics

    unknown = {
        item["actual"] for item in conformance
        if item.get("code") == "unknown_bone_name" and isinstance(item.get("actual"), str)
    }
    missing: set[str] = set()
    for item in conformance:
        if item.get("code") != "missing_required_bone":
            continue
        match = _MISSING_BONE_RE.match(str(item.get("expected") or ""))
        missing.add(match.group(1) if match else "<unnamed-missing-required-bone>")

    metrics["bone_names"] = sorted(str(name) for name in joint_names)
    metrics["template_bone_names"] = sorted(
        ({str(name) for name in joint_names} - unknown) | missing
    )

    reparented = [item for item in conformance if item.get("code") == "reparented_bone"]
    actual_parents: dict[str, Any] = {}
    template_parents: dict[str, Any] = {}
    for index, item in enumerate(reparented):
        match = _REPARENTED_BONE_RE.match(str(item.get("message") or ""))
        key = match.group(1) if match else f"bone[{index}]"
        actual_parents[key] = item.get("actual")
        template_parents[key] = item.get("expected")
    # Both empty when the recorded report found no reparenting -- which is
    # exactly "the hierarchy matches the template".
    metrics["bone_parents"] = actual_parents
    metrics["template_bone_parents"] = template_parents

    metrics["max_rest_direction_deviation_degrees"] = _max_finding_value(
        conformance, "rest_direction_deviation"
    )
    metrics["max_roll_deviation_degrees"] = _max_finding_value(conformance, "roll_deviation")
    metrics["symmetry_band_violation_count"] = sum(
        1 for item in conformance
        if item.get("code") in {"mirrored_left_right", "mirrored_side_position"}
    )
    # `medial_line_distance_violation_count` and `proportion_deviation_ratio`
    # have no measurement -- unmapped.
    return metrics


def _s4_skinning_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    rig_quality = metadata.get("rig_quality") or {}
    metrics: dict[str, Any] = {}

    weighting = metadata.get("weighting")
    if isinstance(weighting, str) and weighting:
        metrics["solver_used"] = "primary" if weighting == PRIMARY_WEIGHT_SOLVER else weighting

    if rig_quality.get("maximum_influences") is not None:
        metrics["max_influences_per_vertex"] = int(rig_quality["maximum_influences"])

    non_normalized = rig_quality.get("non_normalized_vertices")
    if non_normalized == 0:
        # `rig_quality` records the COUNT of vertices whose weights do not
        # sum to 1, not the worst deviation. A count of zero certifies the
        # deviation is within the analyzer's tolerance; a non-zero count
        # tells us the check fails but not by how much, so the magnitude
        # stays unmapped rather than invented, and the check reports
        # not-evaluable.
        metrics["max_weight_sum_deviation"] = 0.0

    if metadata.get("seam_stitched_vertices") is not None:
        metrics["coincident_weight_mismatch_count"] = int(metadata["seam_stitched_vertices"])

    stretch = _fractional_stretch(metadata.get("deformation_review") or {})
    if stretch is not None:
        metrics["rom_edge_stretch_p99"] = stretch

    # `joint_coverage_out_of_band_count`, `rom_self_intersection_ratio` and
    # `rom_seam_tearing_detected` have no measurement -- unmapped.
    return metrics


def _matching_animation(rig_quality: dict[str, Any], clip_name: str | None) -> dict[str, Any] | None:
    animations = rig_quality.get("animations")
    if not isinstance(animations, list) or not animations:
        return None
    if clip_name:
        for entry in animations:
            if isinstance(entry, dict) and str(entry.get("name") or "") == clip_name:
                return entry
    animated = [
        entry for entry in animations
        if isinstance(entry, dict) and int(entry.get("animated_joints") or 0) > 0
    ]
    return animated[-1] if animated else None


def _s5_animation_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    rig_quality = metadata.get("rig_quality") or {}
    metrics: dict[str, Any] = {}

    clip_name = metadata.get("clip") or (metadata.get("settings") or {}).get("animation")
    entry = _matching_animation(rig_quality, str(clip_name) if clip_name else None)
    if entry is not None and entry.get("invalid_targets") is not None:
        # Every animation channel resolving to a joint that exists on the
        # rig is exactly "the clip's bone names match the template" for a
        # direct-attach clip authored against that template.
        metrics["clip_bone_names_match_template"] = int(entry["invalid_targets"]) == 0

    if metadata.get("mechanism"):
        metrics["mechanism"] = str(metadata["mechanism"])

    stretch = _fractional_stretch(metadata.get("deformation_review") or {})
    if stretch is not None:
        metrics["clip_p99_stretch"] = stretch

    # Motion quality, sampled per clip by
    # `blender_worker.deformation_diagnostics` into the deformation report. A
    # synthetic rest-pose sway is not evidence about any clip, so it is
    # skipped outright rather than measured.
    motion = (metadata.get("deformation_review") or {}).get("motion_review") or {}
    if motion.get("measurable") is True and not motion.get("synthetic"):
        # The CONTACT-FILTERED figure, deliberately. The raw `foot_slide_max`
        # charges flight and stride to slide -- a roll is airborne for ~95% of
        # its frames and measures 0.83 with nothing wrong with it -- and is
        # never read here.
        #
        # A contact-filtered 0.0 is ambiguous on its own: it is what a
        # perfectly planted foot measures AND what a clip with no qualifying
        # contact episode measures. `contact_stance_episodes` separates them,
        # and with no episode there is no evidence, so `foot_contact_error`
        # stays absent and `s5.foot_contact_band` stays honestly
        # not-evaluable rather than passing on a fabricated zero.
        episodes = motion.get("contact_stance_episodes")
        slide = motion.get("foot_slide_max_in_contact")
        if isinstance(episodes, int) and episodes > 0 and slide is not None:
            metrics["foot_contact_error"] = float(slide)
        # Populated for the run's policy block and for any check that comes to
        # want them. `quality_policy` already raises its own advisory findings
        # on both; no S5 contract check requires them today, so populating
        # them changes no verdict -- it puts the numbers where a reader of the
        # policy payload can see them.
        if motion.get("static_tail_fraction") is not None:
            metrics["static_tail_fraction"] = float(motion["static_tail_fraction"])
        if motion.get("joint_motion_coverage") is not None:
            metrics["joint_motion_coverage"] = float(motion["joint_motion_coverage"])

    # `root_motion_policy_honored` and `hunch_detected` have no measurement
    # -- unmapped.
    return metrics


_STAGE_METRIC_ADAPTERS: dict[Stage, Callable[[dict[str, Any]], dict[str, Any]]] = {
    Stage.S0_SOURCE: _s0_source_metrics,
    Stage.S1_GENERATION: _s1_generation_metrics,
    Stage.S2_CANONICALIZATION: _s2_canonicalization_metrics,
    Stage.S3_SKELETON_FIT: _s3_skeleton_fit_metrics,
    Stage.S4_SKINNING: _s4_skinning_metrics,
    Stage.S5_ANIMATION: _s5_animation_metrics,
}


def artifact_stages(metadata: dict[str, Any], kind: str | None = None) -> tuple[Stage, ...]:
    """Which policy stage(s) this artifact's metadata can speak for."""
    stage_name = str((metadata or {}).get("stage") or "")
    if stage_name in ARTIFACT_STAGES:
        return ARTIFACT_STAGES[stage_name]
    if not stage_name and kind in {"image", "png", "jpg", "jpeg", "webp"}:
        # A raw uploaded reference image: the run's S0 source.
        return (Stage.S0_SOURCE,)
    return ()


def stage_artifact_from_metadata(
    stage: Stage, metadata: dict[str, Any], artifact_id: str | None = None
) -> StageArtifact:
    """One artifact's recorded job metadata, mapped onto one stage's
    `StageArtifact`. Metrics with no measurement are absent, never guessed."""
    adapter = _STAGE_METRIC_ADAPTERS.get(stage)
    metrics = adapter(metadata or {}) if adapter is not None else {}
    provenance = (
        _s1_generation_provenance(metadata or {}) if stage is Stage.S1_GENERATION else {}
    )
    return StageArtifact(
        stage=stage, artifact_id=artifact_id, metrics=metrics, provenance=provenance
    )


def s6_export_artifact(
    artifact_id: str | None,
    review_status: str,
    reviewed_artifact_id: str | None,
) -> StageArtifact:
    """The S6 operator-accept gate, read off the run manifest's review
    fields. Callers only build this once a review decision exists -- an
    unreviewed run has no S6 artifact at all, so the loop's advice is
    `Advance(S6)` ("the operator accept is what's next"), not a
    manufactured rejection."""
    return StageArtifact(
        stage=Stage.S6_EXPORT,
        artifact_id=artifact_id,
        metrics={
            "operator_accept_decision": "accept" if review_status == "accepted" else review_status,
            "operator_accept_artifact_id": reviewed_artifact_id,
        },
    )


# ---------------------------------------------------------------------------
# Run assembly + summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRecord:
    """One candidate artifact for a run's policy evaluation.

    `on_lineage` marks artifacts on the terminal artifact's own
    `source_artifact_id` chain -- the coherent journey of the asset the run
    currently points at. Off-lineage records (a sibling experiment inside
    the same run) are still used to fill stages the lineage does not cover,
    but are flagged so the reader can see the seam.

    `produced_in_run` marks artifacts this run actually produced (the
    output of one of its attempts), as opposed to lineage it inherited --
    an uploaded source asset and everything upstream of it. Only a run's
    own outputs can evidence that a stage was RE-RUN inside the run, which
    is what `detect_stale_stages` needs to tell "this stage was fixed and
    downstream work is now obsolete" apart from "this run started from an
    asset that already had a rig".
    """

    artifact_id: str | None
    metadata: dict[str, Any]
    kind: str | None = None
    on_lineage: bool = False
    produced_in_run: bool = False


def assemble_pipeline_run(
    records: "list[ArtifactRecord]",
    *,
    source: str,
    family: str | None = None,
) -> tuple[PipelineRun, dict[Stage, ArtifactRecord]]:
    """Build a `PipelineRun` from a run's artifacts.

    `records` are ordered oldest-first. Lineage records are expected to come
    first, then the run's other completed outputs; the last record claiming
    a stage wins, except that an on-lineage record is never displaced by an
    off-lineage one. Returns the run plus which record backs each stage.

    A remediation output is the one exception to the lineage preference,
    in both directions: it is the artifact that REPLACED its stage, so it
    takes the stage even from the lineage the run currently points at, and
    no ordinary sibling takes the stage back off it. Without that, a run
    that ran one more job on the pre-fix branch after a remediation would
    put the superseded mesh back at S2 and lose the fix entirely.
    """
    placements: dict[Stage, ArtifactRecord] = {}
    for record in records:
        for stage in artifact_stages(record.metadata, record.kind):
            existing = placements.get(stage)
            if existing is not None and not record_is_remediation(record):
                if existing.on_lineage and not record.on_lineage:
                    continue
                if record_is_remediation(existing):
                    continue
            placements[stage] = record

    run = PipelineRun(
        source=source,
        family=family or _infer_family(placements),
        seeds=_collect_seeds(records),
    )
    for stage, record in sorted(placements.items(), key=lambda item: stage_index(item[0])):
        run = run.with_stage_artifact(
            stage_artifact_from_metadata(stage, record.metadata, record.artifact_id)
        )
    return run, placements


def _infer_family(placements: dict[Stage, ArtifactRecord]) -> str:
    for record in placements.values():
        metadata = record.metadata or {}
        if metadata.get("family"):
            return str(metadata["family"])
        rig_quality = metadata.get("rig_quality")
        if isinstance(rig_quality, dict) and rig_quality.get("template_conformance_family"):
            return str(rig_quality["template_conformance_family"])
        certificate = (metadata.get("quality_decisions") or {}).get("rig-certificate") or {}
        if certificate.get("profile") == "humanoid":
            return "biped"
    return "unknown"


# ---------------------------------------------------------------------------
# Lineage staleness (bead 38nlk)
#
# PIPELINE_POLICY.md, Principle 2: "the loop after a remediation is
# mechanical: invalidate from the remediated stage, re-run downstream,
# re-evaluate contracts." Without this section the evaluation happily
# stepped OVER downstream work that was built on the pre-fix artifact:
# run 070772d8 remediated S2 into a new canonical mesh, and the loop still
# reported `advance -> S5` because S3/S4 held a rig -- a rig of the OLD
# mesh. The seam was visible (`off_lineage_stages`) but not acted on.
#
# Two things had to be told apart, and the difference is what this code
# is for:
#
#  - A stage was re-run inside this run, so its earlier output (and every
#    artifact descended from it) is superseded. That is staleness.
#  - A run holds two artifacts for a stage because it INHERITED one (it
#    started from an asset that already had a rig) and produced the other.
#    Sibling branches of that kind are already reported as
#    `off_lineage_stages`; they are not stale, and flagging them would
#    re-run a stage nothing invalidated -- real GPU spend on a false
#    positive.
#
# So a "fix point" is a stage whose CURRENT artifact either carries
# remediation provenance (the step driver stamps `settings.pipeline_step`)
# or supersedes an earlier artifact this same run produced for that stage.
# Everything downstream of a fix point that does not descend from the fix
# is stale.
# ---------------------------------------------------------------------------


# Same bound and rationale as main.py's lineage walk: a run's asset chain is
# short, and the bound stops a corrupted `source_artifact_id` cycle from
# walking the whole artifact store.
STALENESS_LINEAGE_DEPTH = 24


@dataclass(frozen=True)
class StaleStage:
    """One stage whose recorded artifact describes a superseded asset.

    `obsoleted_by_stage`/`obsoleted_by_artifact_id` name the upstream fix
    this artifact does not descend from -- the whole reason it is stale --
    so the report says which artifact obsoleted which, never just "stale".
    """

    stage: Stage
    artifact_id: str | None
    obsoleted_by_stage: Stage
    obsoleted_by_artifact_id: str | None
    trigger: Literal["remediation", "stage_rerun"]

    @property
    def reason(self) -> str:
        because = (
            "was remediated"
            if self.trigger == "remediation"
            else "was re-run in this run"
        )
        return (
            f"{self.obsoleted_by_stage.value} {because} into "
            f"{self.obsoleted_by_artifact_id or 'a pending artifact'}; this stage's "
            f"artifact does not descend from it"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "artifact_id": self.artifact_id,
            "obsoleted_by": {
                "stage": self.obsoleted_by_stage.value,
                "artifact_id": self.obsoleted_by_artifact_id,
            },
            "trigger": self.trigger,
            "reason": self.reason,
        }


def record_is_remediation(record: ArtifactRecord) -> bool:
    """True when this artifact was produced by a policy remediation step.

    The step driver stamps its provenance into the job's settings
    (`settings[PIPELINE_STEP_SETTINGS_KEY]`), and the workers copy the
    submitted settings into the output artifact's metadata, so the marker
    survives into the artifact. The un-nested form is accepted too, for a
    caller that hands the provenance block straight through.
    """
    metadata = record.metadata or {}
    settings = metadata.get("settings")
    block = settings.get(PIPELINE_STEP_SETTINGS_KEY) if isinstance(settings, dict) else None
    if not isinstance(block, dict):
        block = metadata.get(PIPELINE_STEP_SETTINGS_KEY)
    return isinstance(block, dict) and block.get("mode") == "remediation"


def _ancestor_ids(
    record: ArtifactRecord, by_id: dict[str, ArtifactRecord]
) -> set[str]:
    """Every artifact id `record` descends from, walking `source_artifact_id`.

    The walk stops at an id with no record (a deleted or external
    ancestor): that id is still counted -- it is a real link in the chain --
    but nothing beyond it can be known.
    """
    ancestors: set[str] = set()
    current = (record.metadata or {}).get("source_artifact_id")
    while (
        isinstance(current, str)
        and current
        and current not in ancestors
        and len(ancestors) < STALENESS_LINEAGE_DEPTH
    ):
        ancestors.add(current)
        parent = by_id.get(current)
        if parent is None:
            break
        current = (parent.metadata or {}).get("source_artifact_id")
    return ancestors


def _fix_points(
    records: "list[ArtifactRecord]", placements: dict[Stage, ArtifactRecord]
) -> list[tuple[Stage, ArtifactRecord, Literal["remediation", "stage_rerun"]]]:
    """The stages whose current artifact obsoletes downstream work."""
    produced_per_stage: dict[Stage, int] = {}
    for record in records:
        if not record.produced_in_run:
            continue
        for stage in artifact_stages(record.metadata, record.kind):
            produced_per_stage[stage] = produced_per_stage.get(stage, 0) + 1

    fixes: list[tuple[Stage, ArtifactRecord, Literal["remediation", "stage_rerun"]]] = []
    for stage, record in sorted(placements.items(), key=lambda item: stage_index(item[0])):
        if record_is_remediation(record):
            fixes.append((stage, record, "remediation"))
        elif record.produced_in_run and produced_per_stage.get(stage, 0) > 1:
            fixes.append((stage, record, "stage_rerun"))
    return fixes


def detect_stale_stages(
    records: "list[ArtifactRecord]", placements: dict[Stage, ArtifactRecord]
) -> dict[Stage, StaleStage]:
    """Which placed stages describe an asset that has since been superseded.

    A stage is stale when some EARLIER stage is a fix point (its current
    artifact is a remediation, or a re-run of a stage this run had already
    produced an artifact for) and this stage's artifact does not descend
    from that fix. Staleness propagates for free: a stage built on a stale
    stage does not descend from the fix either.

    One artifact can back two stages (an auto-rig output backs S3 and S4);
    the fix point's own stages are never stale against themselves.
    """
    by_id = {
        record.artifact_id: record for record in records if record.artifact_id
    }
    fixes = _fix_points(records, placements)
    if not fixes:
        return {}

    stale: dict[Stage, StaleStage] = {}
    for stage in STAGE_ORDER:
        record = placements.get(stage)
        if record is None:
            continue
        ancestors: set[str] | None = None
        for fix_stage, fix_record, trigger in fixes:
            if stage_index(fix_stage) >= stage_index(stage):
                continue
            if record is fix_record or (
                record.artifact_id is not None
                and record.artifact_id == fix_record.artifact_id
            ):
                continue
            if ancestors is None:
                ancestors = _ancestor_ids(record, by_id)
            if fix_record.artifact_id is not None and fix_record.artifact_id in ancestors:
                continue
            stale[stage] = StaleStage(
                stage=stage,
                artifact_id=record.artifact_id,
                obsoleted_by_stage=fix_stage,
                obsoleted_by_artifact_id=fix_record.artifact_id,
                trigger=trigger,
            )
            break
    return stale


def _collect_seeds(records: "list[ArtifactRecord]") -> tuple[int, ...]:
    seeds: list[int] = []
    for record in records:
        seed = (record.metadata or {}).get("seed")
        if isinstance(seed, int) and seed not in seeds:
            seeds.append(seed)
    return tuple(seeds)


def _compact_finding(item: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    trimmed = {
        "code": item["code"],
        "severity": item["severity"],
        "metric": item["metric"],
        "actual": item["actual"],
        "expected": item["expected"],
    }
    if not compact:
        trimmed["message"] = item["message"]
    if compact and isinstance(trimmed["actual"], (list, dict)) and len(trimmed["actual"]) > 8:
        trimmed["actual"] = f"<{len(trimmed['actual'])} entries>"
    return trimmed


def _blame_detail(finding_code: str, run: PipelineRun) -> dict[str, Any]:
    row = _BLAME_ROW_BY_FINDING_CODE.get(finding_code)
    detail: dict[str, Any] = {
        "finding_code": finding_code,
        "owning_stage": resolve_blame(finding_code, run).value,
    }
    if row is not None:
        detail.update({
            "symptom": row.symptom,
            "rationale": row.rationale,
            "incident_ref": row.incident_ref,
            "conditional": row.condition_description,
        })
    else:
        detail["rationale"] = "safe default: the stage that emitted the finding"
    return detail


def summarize_pipeline_policy(
    run: PipelineRun,
    placements: dict[Stage, ArtifactRecord] | None = None,
    *,
    records: "list[ArtifactRecord] | None" = None,
    compact: bool = False,
) -> dict[str, Any]:
    """The whole-run evaluation, as a JSON-serialisable block.

    `compact=True` drops finding messages and collapses long `actual`
    payloads -- that form is what gets stamped into job/artifact metadata;
    the endpoint serves the full form.

    `records` are the same records `assemble_pipeline_run` was given. They
    are what makes lineage staleness knowable (the `source_artifact_id`
    chains live in their metadata); without them the summary reports no
    stale stages, which is the old, seam-blind behaviour.
    """
    placements = placements or {}
    stale = detect_stale_stages(records or [], placements)
    stages: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    first_violated_stage: Stage | None = None

    for stage in STAGE_ORDER:
        artifact = run.stage_artifacts.get(stage)
        if artifact is None:
            stages.append({
                "stage": stage.value,
                "status": "not_run",
                "artifact_id": None,
                "on_lineage": None,
                "stale": False,
                "findings": [],
                "counts": {"error": 0, "warning": 0, "not_evaluable": 0},
                "metric_sources": {},
            })
            continue
        decision = evaluate_stage(stage, artifact)
        findings = decision["findings"]
        counts = {
            "error": sum(1 for item in findings if item["severity"] == "error"),
            "warning": sum(1 for item in findings if item["severity"] == "warning"),
            "not_evaluable": sum(1 for item in findings if item["severity"] == "not-evaluable"),
        }
        if counts["error"] and first_violated_stage is None:
            first_violated_stage = stage
        record = placements.get(stage)
        stages.append({
            "stage": stage.value,
            "status": decision["status"],
            "artifact_id": artifact.artifact_id,
            "on_lineage": record.on_lineage if record is not None else None,
            # A stale stage's verdict is real, but it describes an asset the
            # run has already replaced -- the loop re-runs the stage instead
            # of counting it. Kept here rather than dropped so nothing is
            # hidden: the reader still sees what the superseded work said.
            "stale": stage in stale,
            "findings": [_compact_finding(item, compact=compact) for item in findings],
            "counts": counts,
            "metric_sources": {
                key: METRIC_SOURCES[key]
                for key in sorted(artifact.metrics)
                if key in METRIC_SOURCES
            },
        })
        for item in findings:
            if item["severity"] == "not-evaluable":
                not_evaluable.append({
                    "stage": stage.value,
                    "check": item["code"],
                    "missing_metrics": [
                        part.strip() for part in str(item["metric"]).split(",") if part.strip()
                    ],
                })

    action = next_action(run, stale)
    if isinstance(action, Done):
        action_block: dict[str, Any] = {"kind": "done", "stage": None}
    elif isinstance(action, Advance):
        action_block = {"kind": "advance", "stage": action.stage.value}
    else:
        action_block = {
            "kind": "stop-at-stage",
            "stage": action.stage.value,
            "triggering_stage": action.triggering_stage.value,
            "findings": [_compact_finding(item, compact=compact) for item in action.findings],
            "blame": [_blame_detail(item["code"], run) for item in action.findings],
        }

    return {
        "policy_version": 1,
        "source": run.source,
        "family": run.family,
        "seeds": list(run.seeds),
        "stages": stages,
        "first_violated_stage": first_violated_stage.value if first_violated_stage else None,
        "next_action": action_block,
        "not_evaluable": not_evaluable,
        "stages_not_run": [
            item["stage"] for item in stages if item["status"] == "not_run"
        ],
        # Stages backed by an artifact that is NOT on the run's current
        # asset's lineage -- a sibling branch of the same run. The contract
        # verdict for such a stage is real, but it describes a different
        # asset than the one the run currently points at, so advice that
        # depends on it has to be read with that in mind.
        "off_lineage_stages": [
            item["stage"] for item in stages if item["on_lineage"] is False
        ],
        # Stages whose artifact was built on an upstream artifact this run
        # has since replaced (a remediation, or a re-run of the stage).
        # `next_action` treats these as not-run -- "invalidate from the
        # remediated stage, re-run downstream" -- and each entry names the
        # artifact that is stale and the upstream artifact that obsoleted
        # it, so re-running them is a stated decision rather than a silent
        # one.
        "stale_stages": [
            stale[stage].as_dict() for stage in STAGE_ORDER if stage in stale
        ],
    }
