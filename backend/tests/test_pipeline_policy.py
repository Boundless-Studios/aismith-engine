"""Tests for `backend/app/pipeline_policy.py` (workstream N6).

Coverage:
- Every contract check triggers on a crafted violating artifact and passes
  on a compliant one (parametrized per stage).
- Every check with a `requires` metric surfaces a "not-evaluable" finding
  -- never a silent pass -- when that metric is missing from the artifact.
- The 7 blame-table rows resolve to their documented owning stage,
  including the two rows whose blame is conditional on run history.
- The loop (`next_action`) advances stage by stage, stops at the right
  stage on a violation (resolving blame, including the upstream-earliest
  rule when multiple findings implicate different stages), and reaches
  `Done` once every stage evaluates clean.
- Lineage staleness: work built on an artifact a remediation replaced is
  invalidated and re-entered, while a run that merely inherited a second
  artifact for a stage is left alone.
- The measured-metric adapters: each newly-measured metric flips its check
  from not-evaluable to a real verdict when the data is present and leaves
  it not-evaluable when it is not, and the calibrated foot-contact band
  clears every clip the family's own library authored.
"""
from __future__ import annotations

import json
import unittest

from backend.app.pipeline_policy import (
    ALL_CONTRACTS,
    BLAME_TABLE,
    CONTRACTS_BY_STAGE,
    FOOT_CONTACT_BAND_THRESHOLD,
    METRIC_SOURCES,
    STAGE_ORDER,
    SYMMETRY_MIRROR_ERROR_P95_THRESHOLD,
    Advance,
    ArtifactRecord,
    Done,
    PipelineRun,
    Stage,
    StageArtifact,
    StopForRemediation,
    artifact_stages,
    assemble_pipeline_run,
    blame_stage,
    detect_stale_stages,
    evaluate_stage,
    next_action,
    provenance_complete,
    resolve_blame,
    stage_artifact_from_metadata,
    summarize_pipeline_policy,
)


# ---------------------------------------------------------------------------
# Compliant metrics fixtures, one per stage -- every check on that stage
# passes cleanly against these.
# ---------------------------------------------------------------------------

S0_METRICS = {
    "full_subject_visible": True,
    "subject_count": 1,
    "min_resolution_px": 1024,
    "subject_family": "biped",
    "family_confidence": 0.95,
}

S1_METRICS = {
    "recognizable_subject": True,
    "textures_present": True,
    "source_views_recorded": True,
    "source_fidelity_silhouette_iou": 0.95,
    "source_fidelity_face_ssim": 0.9,
}
S1_PROVENANCE = {
    "seed": 42,
    "tool_versions": {"trellis": "2.1.0"},
    "source_artifact_id": "src-abc123",
}

S2_METRICS = {
    "connected_components_welded": 1,
    "connected_components_unwelded": 1,
    "watertight": True,
    "non_manifold_edges": 0,
    "triangle_count": 25_000,
    "uv_unwrapped": True,
    "up_axis": "Z",
    "facing_axis": "-Y",
    "uniform_scale": True,
    "floor_contact_z_abs": 0.0,
    "s1_s2_image_similarity_score": 0.97,
    "source_fidelity_silhouette_iou": 0.95,
    "source_fidelity_face_ssim": 0.9,
}

S3_METRICS = {
    "bone_names": ["hips", "spine", "head"],
    "template_bone_names": ["hips", "spine", "head"],
    "bone_parents": {"hips": None, "spine": "hips", "head": "spine"},
    "template_bone_parents": {"hips": None, "spine": "hips", "head": "spine"},
    "max_rest_direction_deviation_degrees": 3.0,
    "max_roll_deviation_degrees": 2.0,
    "joints_outside_mesh_count": 0,
    "symmetry_band_violation_count": 0,
    "symmetry_mirror_error_p95": 0.0,
    "symmetry_pairs_outside_band_count": 0,
    "medial_line_distance_violation_count": 0,
    "proportion_deviation_ratio": 0.05,
    "fit_converged": True,
}

S4_METRICS = {
    "solver_used": "primary",
    "max_influences_per_vertex": 4,
    "max_weight_sum_deviation": 0.0,
    "coincident_weight_mismatch_count": 0,
    "joint_coverage_out_of_band_count": 0,
    "rom_edge_stretch_p99": 0.05,
    "rom_self_intersection_ratio": 0.0,
    "rom_seam_tearing_detected": False,
}

S5_METRICS = {
    "clip_bone_names_match_template": True,
    "root_motion_policy_honored": True,
    "foot_contact_error": 0.005,
    "clip_p99_stretch": 0.05,
    "hunch_detected": False,
    "mechanism": "direct-attach",
}

S6_METRICS = {
    "operator_accept_decision": "accept",
    "operator_accept_artifact_id": "art-final-1",
}


def _artifact(stage: Stage, metrics: dict, provenance: dict | None = None, artifact_id: str | None = None) -> StageArtifact:
    return StageArtifact(stage=stage, artifact_id=artifact_id, metrics=dict(metrics), provenance=dict(provenance or {}))


COMPLIANT_ARTIFACTS: dict[Stage, StageArtifact] = {
    Stage.S0_SOURCE: _artifact(Stage.S0_SOURCE, S0_METRICS),
    Stage.S1_GENERATION: _artifact(Stage.S1_GENERATION, S1_METRICS, S1_PROVENANCE),
    Stage.S2_CANONICALIZATION: _artifact(Stage.S2_CANONICALIZATION, S2_METRICS),
    Stage.S3_SKELETON_FIT: _artifact(Stage.S3_SKELETON_FIT, S3_METRICS),
    Stage.S4_SKINNING: _artifact(Stage.S4_SKINNING, S4_METRICS),
    Stage.S5_ANIMATION: _artifact(Stage.S5_ANIMATION, S5_METRICS),
    Stage.S6_EXPORT: _artifact(Stage.S6_EXPORT, S6_METRICS, artifact_id="art-final-1"),
}

# check_id -> metrics override that violates exactly that check, holding
# everything else on the stage's compliant fixture constant.
VIOLATIONS: dict[str, dict] = {
    # S0
    "s0.full_subject_visible": {"full_subject_visible": False},
    "s0.single_subject": {"subject_count": 2},
    "s0.resolution_floor": {"min_resolution_px": 200},
    "s0.family_classified": {"subject_family": ""},
    "s0.family_confidence_ambiguous": {"family_confidence": 0.3},
    # S1
    "s1.recognizable_subject": {"recognizable_subject": False},
    "s1.textures_present": {"textures_present": False},
    "s1.source_views_recorded": {"source_views_recorded": False},
    "s1.source_fidelity_silhouette": {"source_fidelity_silhouette_iou": 0.5},
    "s1.source_fidelity_face": {"source_fidelity_face_ssim": 0.1},
    # S2
    "s2.single_connected_component_welded": {"connected_components_welded": 2},
    "s2.single_connected_component_unwelded": {"connected_components_unwelded": 3},
    "s2.watertight": {"watertight": False},
    "s2.non_manifold_edges": {"non_manifold_edges": 5},
    "s2.triangle_budget": {"triangle_count": 500},
    "s2.uv_unwrapped": {"uv_unwrapped": False},
    "s2.axis_convention": {"up_axis": "Y"},
    "s2.uniform_scale": {"uniform_scale": False},
    "s2.floor_contact": {"floor_contact_z_abs": 0.5},
    "s2.appearance_preservation_bake_gate": {"s1_s2_image_similarity_score": 0.5},
    "s2.source_fidelity_silhouette": {"source_fidelity_silhouette_iou": 0.5},
    "s2.source_fidelity_face": {"source_fidelity_face_ssim": 0.1},
    # S3
    "s3.exact_bone_set": {"bone_names": ["hips", "spine"]},
    "s3.bone_hierarchy_matches_template": {
        "bone_parents": {"hips": None, "spine": "hips", "head": "hips"},
    },
    "s3.bone_rest_direction_matches_template": {"max_rest_direction_deviation_degrees": 30.0},
    "s3.bone_roll_matches_template": {"max_roll_deviation_degrees": 45.0},
    "s3.joints_inside_mesh": {"joints_outside_mesh_count": 2},
    "s3.left_right_symmetry": {"symmetry_band_violation_count": 1},
    "s3.bilateral_symmetry_p95": {"symmetry_mirror_error_p95": 0.12},
    "s3.medial_line_distance": {"medial_line_distance_violation_count": 1},
    "s3.proportion_sanity": {"proportion_deviation_ratio": 0.5},
    "s3.fit_converged": {"fit_converged": False},
    # S4
    "s4.primary_solver_engaged": {"solver_used": "fallback"},
    "s4.max_influences": {"max_influences_per_vertex": 6},
    "s4.weights_sum_to_one": {"max_weight_sum_deviation": 0.1},
    "s4.coincident_vertices_share_weights": {"coincident_weight_mismatch_count": 4},
    "s4.per_joint_coverage": {"joint_coverage_out_of_band_count": 2},
    "s4.rom_edge_stretch_p99": {"rom_edge_stretch_p99": 9.0},
    "s4.rom_self_intersection": {"rom_self_intersection_ratio": 0.2},
    "s4.rom_seam_tearing": {"rom_seam_tearing_detected": True},
    # S5
    "s5.exact_bone_name_match": {"clip_bone_names_match_template": False},
    "s5.root_motion_policy_honored": {"root_motion_policy_honored": False},
    "s5.foot_contact_band": {"foot_contact_error": 1.5},
    "s5.stretch_p99_under_threshold": {"clip_p99_stretch": 9.0},
    "s5.rest_pose_hunch_detected": {"hunch_detected": True},
    "s5.direct_clip_attach": {"mechanism": "bvh-retarget"},
    # S6
    "s6.operator_accept_recorded": {"operator_accept_decision": "reject"},
}

WARNING_CHECK_IDS = {
    "s0.family_confidence_ambiguous",
    "s1.source_fidelity_silhouette",
    "s1.source_fidelity_face",
    "s2.source_fidelity_silhouette",
    "s2.source_fidelity_face",
    "s3.proportion_sanity",
    # Advisory even after the 2026-09-04 `scripts/calibrate_symmetry.py`
    # calibration: one corpus pass is not FOOT_CONTACT_BAND_THRESHOLD's
    # multi-family guarantee, so a red here still recommends rather than
    # blocks.
    "s3.bilateral_symmetry_p95",
    "s4.per_joint_coverage",
    "s5.direct_clip_attach",
    # Advisory since the 2026-09-01 calibration: the contact-filtered slide
    # measure cannot separate an in-place walk's stride from real slip, so it
    # reports but never blocks. See FOOT_CONTACT_BAND_THRESHOLD.
    "s5.foot_contact_band",
}


class CompliantArtifactsPassCleanTests(unittest.TestCase):
    """Every stage's compliant fixture evaluates to a clean decision."""

    def test_every_stage_compliant_fixture_is_green(self):
        for stage, artifact in COMPLIANT_ARTIFACTS.items():
            with self.subTest(stage=stage):
                decision = evaluate_stage(stage, artifact)
                self.assertEqual(decision["status"], "green", decision["findings"])


class EveryCheckHasComplianceAndViolationCoverageTests(unittest.TestCase):
    """Sanity check on the test data itself: every registered check id has
    a violation fixture, and vice versa (nothing in the doc silently
    dropped, nothing tested that doesn't exist)."""

    def test_all_check_ids_covered_by_violations_map(self):
        all_ids = {check.id for contract in ALL_CONTRACTS for check in contract.checks}
        # s1.provenance_recorded is exercised separately (it has no single
        # `requires` metric -- it wraps provenance_complete()).
        expected = all_ids - {"s1.provenance_recorded"}
        self.assertEqual(expected, set(VIOLATIONS.keys()))


class ContractCheckViolationTests(unittest.TestCase):
    """Every check triggers on a crafted violating artifact and passes on
    the stage's compliant artifact."""

    def test_each_violation_triggers_only_its_own_check(self):
        for check_id, override in VIOLATIONS.items():
            check = next(
                c for contract in ALL_CONTRACTS for c in contract.checks if c.id == check_id
            )
            stage = check.stage
            base_artifact = COMPLIANT_ARTIFACTS[stage]
            metrics = dict(base_artifact.metrics)
            metrics.update(override)
            violating = StageArtifact(
                stage=stage, artifact_id=base_artifact.artifact_id,
                metrics=metrics, provenance=base_artifact.provenance,
            )
            with self.subTest(check_id=check_id):
                findings = check.run(violating)
                self.assertEqual(len(findings), 1, findings)
                self.assertEqual(findings[0]["code"], check_id)
                expected_severity = "warning" if check_id in WARNING_CHECK_IDS else "error"
                self.assertEqual(findings[0]["severity"], expected_severity)

                # And the compliant fixture itself never triggers this check.
                self.assertEqual(check.run(base_artifact), [])

    def test_violating_stage_decision_status_reflects_severity(self):
        # Error-severity violation -> stage decision is "blocked".
        metrics = dict(S2_METRICS)
        metrics.update(VIOLATIONS["s2.watertight"])
        artifact = StageArtifact(stage=Stage.S2_CANONICALIZATION, metrics=metrics)
        decision = evaluate_stage(Stage.S2_CANONICALIZATION, artifact)
        self.assertEqual(decision["status"], "blocked")

        # Warning-only violation -> "needs_review", not "blocked".
        metrics = dict(S3_METRICS)
        metrics.update(VIOLATIONS["s3.proportion_sanity"])
        artifact = StageArtifact(stage=Stage.S3_SKELETON_FIT, metrics=metrics)
        decision = evaluate_stage(Stage.S3_SKELETON_FIT, artifact)
        self.assertEqual(decision["status"], "needs_review")


class NotEvaluableTests(unittest.TestCase):
    """A check whose required metric(s) are missing surfaces a
    "not-evaluable" finding -- it never silently passes."""

    def test_every_requiring_check_is_not_evaluable_on_empty_metrics(self):
        for contract in ALL_CONTRACTS:
            for check in contract.checks:
                if not check.requires:
                    continue
                with self.subTest(check_id=check.id):
                    empty_artifact = StageArtifact(stage=check.stage, metrics={})
                    findings = check.run(empty_artifact)
                    self.assertEqual(len(findings), 1, findings)
                    self.assertEqual(findings[0]["severity"], "not-evaluable")
                    self.assertEqual(findings[0]["code"], check.id)

    def test_not_evaluable_does_not_silently_pass_as_stage_status(self):
        artifact = StageArtifact(stage=Stage.S2_CANONICALIZATION, metrics={})
        decision = evaluate_stage(Stage.S2_CANONICALIZATION, artifact)
        self.assertNotEqual(decision["status"], "green")
        self.assertTrue(any(f["severity"] == "not-evaluable" for f in decision["findings"]))

    def test_s2_bake_gate_is_not_evaluable_when_similarity_score_absent(self):
        # The running example from the task: nothing in the codebase
        # computes image-similarity yet, so this check is structurally
        # not-evaluable on any artifact that hasn't had that metric wired
        # in -- including an otherwise-fully-compliant S2 artifact.
        metrics = dict(S2_METRICS)
        del metrics["s1_s2_image_similarity_score"]
        artifact = StageArtifact(stage=Stage.S2_CANONICALIZATION, metrics=metrics)
        decision = evaluate_stage(Stage.S2_CANONICALIZATION, artifact)
        self.assertEqual(decision["status"], "not_evaluable")
        codes = {f["code"] for f in decision["findings"]}
        self.assertIn("s2.appearance_preservation_bake_gate", codes)


class ProvenanceCompleteTests(unittest.TestCase):
    def test_complete_provenance_has_no_findings(self):
        self.assertEqual(provenance_complete(dict(S1_PROVENANCE)), [])

    def test_missing_seed_is_flagged(self):
        meta = dict(S1_PROVENANCE)
        del meta["seed"]
        findings = provenance_complete(meta)
        self.assertEqual([f["code"] for f in findings], ["provenance.missing_seed"])

    def test_missing_tool_versions_is_flagged(self):
        meta = dict(S1_PROVENANCE)
        del meta["tool_versions"]
        findings = provenance_complete(meta)
        self.assertEqual([f["code"] for f in findings], ["provenance.missing_tool_versions"])

    def test_missing_source_artifact_id_is_flagged(self):
        meta = dict(S1_PROVENANCE)
        del meta["source_artifact_id"]
        findings = provenance_complete(meta)
        self.assertEqual([f["code"] for f in findings], ["provenance.missing_source_artifact_id"])

    def test_all_missing_reports_all_three(self):
        findings = provenance_complete({})
        codes = {f["code"] for f in findings}
        self.assertEqual(
            codes,
            {
                "provenance.missing_seed",
                "provenance.missing_tool_versions",
                "provenance.missing_source_artifact_id",
            },
        )
        self.assertTrue(all(f["severity"] == "error" for f in findings))

    def test_s1_provenance_recorded_check_wraps_provenance_complete(self):
        check = next(c for c in CONTRACTS_BY_STAGE[Stage.S1_GENERATION].checks
                     if c.id == "s1.provenance_recorded")
        clean = StageArtifact(stage=Stage.S1_GENERATION, metrics=S1_METRICS, provenance=S1_PROVENANCE)
        self.assertEqual(check.run(clean), [])

        dirty = StageArtifact(stage=Stage.S1_GENERATION, metrics=S1_METRICS, provenance={})
        findings = check.run(dirty)
        self.assertEqual(len(findings), 3)
        self.assertTrue(all(f["severity"] == "error" for f in findings))


class BlameTableTests(unittest.TestCase):
    def test_blame_table_has_seven_rows(self):
        self.assertEqual(len(BLAME_TABLE), 7)

    def test_seams_split_tearing_blames_s2_by_default(self):
        self.assertEqual(blame_stage("s4.rom_seam_tearing"), Stage.S2_CANONICALIZATION)

    def test_wrong_mirrored_limb_naming_blames_s3(self):
        self.assertEqual(blame_stage("s3.exact_bone_set"), Stage.S3_SKELETON_FIT)
        self.assertEqual(blame_stage("s3.bone_hierarchy_matches_template"), Stage.S3_SKELETON_FIT)

    def test_hunch_inherited_donor_posture_blames_s3_by_default(self):
        self.assertEqual(blame_stage("s5.rest_pose_hunch_detected"), Stage.S3_SKELETON_FIT)

    def test_limb_roll_twist_blames_s3(self):
        self.assertEqual(blame_stage("s3.bone_roll_matches_template"), Stage.S3_SKELETON_FIT)

    def test_face_texture_degraded_blames_s2(self):
        self.assertEqual(blame_stage("s2.appearance_preservation_bake_gate"), Stage.S2_CANONICALIZATION)

    def test_stretch_metric_red_visual_clean_blames_no_stage_by_default_context(self):
        # "The METRIC" row: not a re-runnable pipeline stage. Context-free
        # `blame_stage` falls back to the safe default (the emitting
        # stage) since there is nothing else it can hand back.
        self.assertEqual(blame_stage("s4.rom_edge_stretch_p99"), Stage.S4_SKINNING)
        self.assertEqual(blame_stage("s5.stretch_p99_under_threshold"), Stage.S5_ANIMATION)
        row = next(r for r in BLAME_TABLE if r.symptom_code == "stretch_metric_red_visual_clean")
        self.assertIsNone(row.owning_stage)

    def test_solver_fallback_engaged_blames_s2(self):
        self.assertEqual(blame_stage("s4.primary_solver_engaged"), Stage.S2_CANONICALIZATION)

    def test_unknown_code_falls_back_to_stage_prefix(self):
        self.assertEqual(blame_stage("s3.some_future_check"), Stage.S3_SKELETON_FIT)

    def test_totally_unrecognized_code_falls_back_to_s0_without_raising(self):
        self.assertEqual(blame_stage("not-a-real-prefix.whatever"), Stage.S0_SOURCE)

    def test_resolve_blame_without_run_matches_blame_stage(self):
        for row in BLAME_TABLE:
            for code in row.finding_codes:
                with self.subTest(code=code):
                    self.assertEqual(resolve_blame(code), blame_stage(code))

    def test_seams_conditional_blame_moves_to_s4_when_s2_was_green(self):
        run = PipelineRun(source="src", family="biped").with_stage_artifact(
            COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION]
        )
        self.assertEqual(
            resolve_blame("s4.rom_seam_tearing", run), Stage.S4_SKINNING,
        )

    def test_seams_conditional_blame_stays_s2_when_s2_not_green(self):
        dirty_metrics = dict(S2_METRICS)
        dirty_metrics.update(VIOLATIONS["s2.non_manifold_edges"])
        run = PipelineRun(source="src", family="biped").with_stage_artifact(
            StageArtifact(stage=Stage.S2_CANONICALIZATION, metrics=dirty_metrics)
        )
        self.assertEqual(
            resolve_blame("s4.rom_seam_tearing", run), Stage.S2_CANONICALIZATION,
        )

    def test_hunch_conditional_blame_moves_to_s5_when_s3_was_green(self):
        run = PipelineRun(source="src", family="biped").with_stage_artifact(
            COMPLIANT_ARTIFACTS[Stage.S3_SKELETON_FIT]
        )
        self.assertEqual(
            resolve_blame("s5.rest_pose_hunch_detected", run), Stage.S5_ANIMATION,
        )

    def test_hunch_conditional_blame_stays_s3_when_s3_not_green(self):
        dirty_metrics = dict(S3_METRICS)
        dirty_metrics.update(VIOLATIONS["s3.bone_roll_matches_template"])
        run = PipelineRun(source="src", family="biped").with_stage_artifact(
            StageArtifact(stage=Stage.S3_SKELETON_FIT, metrics=dirty_metrics)
        )
        self.assertEqual(
            resolve_blame("s5.rest_pose_hunch_detected", run), Stage.S3_SKELETON_FIT,
        )

    def test_cannot_fit_blames_s2_per_doc_prose_not_the_table(self):
        # "Cannot fit" is a prose blame rule (S2 bad mesh, or S0
        # misclassified family), not one of the 7 table rows.
        self.assertEqual(blame_stage("s3.fit_converged"), Stage.S2_CANONICALIZATION)


class PipelineRunLoopTests(unittest.TestCase):
    def test_empty_run_advances_from_s0(self):
        run = PipelineRun(source="src.png", family="biped")
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S0_SOURCE)

    def test_advances_through_clean_stages_in_order(self):
        run = PipelineRun(source="src.png", family="biped")
        run = run.with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S1_GENERATION)

        run = run.with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
        action = next_action(run)
        self.assertEqual(action.stage, Stage.S2_CANONICALIZATION)

    def test_stops_for_remediation_at_violating_stage_with_no_blame_indirection(self):
        # A wrong bone set is an S3 problem that blames S3 itself.
        metrics = dict(S3_METRICS)
        metrics.update(VIOLATIONS["s3.exact_bone_set"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(StageArtifact(stage=Stage.S3_SKELETON_FIT, metrics=metrics))
        )
        action = next_action(run)
        self.assertIsInstance(action, StopForRemediation)
        self.assertEqual(action.triggering_stage, Stage.S3_SKELETON_FIT)
        self.assertEqual(action.stage, Stage.S3_SKELETON_FIT)
        self.assertEqual(len(action.findings), 1)
        self.assertEqual(action.findings[0]["code"], "s3.exact_bone_set")

    def test_stops_for_remediation_and_blames_upstream_stage(self):
        # Solver fallback is observed at S4 but blames S2 (the mesh made
        # the primary solver fail) -- the run must stop for remediation
        # AT S2, not S4, even though S2's own artifact was recorded clean.
        metrics = dict(S4_METRICS)
        metrics.update(VIOLATIONS["s4.primary_solver_engaged"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S3_SKELETON_FIT])
            .with_stage_artifact(StageArtifact(stage=Stage.S4_SKINNING, metrics=metrics))
        )
        action = next_action(run)
        self.assertIsInstance(action, StopForRemediation)
        self.assertEqual(action.triggering_stage, Stage.S4_SKINNING)
        self.assertEqual(action.stage, Stage.S2_CANONICALIZATION)

    def test_earliest_stage_wins_when_findings_implicate_different_stages(self):
        # Solver fallback (blames S2) AND a max-influences violation
        # (blames S4 itself, the safe default) both fire at S4 -- the
        # remediation target is the earliest implicated stage, S2.
        metrics = dict(S4_METRICS)
        metrics.update(VIOLATIONS["s4.primary_solver_engaged"])
        metrics.update(VIOLATIONS["s4.max_influences"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S3_SKELETON_FIT])
            .with_stage_artifact(StageArtifact(stage=Stage.S4_SKINNING, metrics=metrics))
        )
        action = next_action(run)
        self.assertIsInstance(action, StopForRemediation)
        self.assertEqual(action.stage, Stage.S2_CANONICALIZATION)
        self.assertEqual(len(action.findings), 2)

    def test_warnings_do_not_stop_the_loop(self):
        metrics = dict(S3_METRICS)
        metrics.update(VIOLATIONS["s3.proportion_sanity"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(StageArtifact(stage=Stage.S3_SKELETON_FIT, metrics=metrics))
        )
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S4_SKINNING)

    def test_not_evaluable_does_not_stop_the_loop(self):
        metrics = dict(S2_METRICS)
        del metrics["s1_s2_image_similarity_score"]
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(StageArtifact(stage=Stage.S2_CANONICALIZATION, metrics=metrics))
        )
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S3_SKELETON_FIT)

    def test_done_once_every_stage_is_clean_through_s6(self):
        run = PipelineRun(source="src.png", family="biped")
        for stage in (
            Stage.S0_SOURCE, Stage.S1_GENERATION, Stage.S2_CANONICALIZATION,
            Stage.S3_SKELETON_FIT, Stage.S4_SKINNING, Stage.S5_ANIMATION, Stage.S6_EXPORT,
        ):
            run = run.with_stage_artifact(COMPLIANT_ARTIFACTS[stage])
        self.assertIsInstance(next_action(run), Done)

    def test_re_run_from_owning_stage_clears_the_stop(self):
        # After "remediation lands" (a fresh, compliant S2 artifact is
        # recorded) the run advances past S2 again -- re-running downstream
        # is then just advancing through the remaining stages.
        bad_metrics = dict(S4_METRICS)
        bad_metrics.update(VIOLATIONS["s4.primary_solver_engaged"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S3_SKELETON_FIT])
            .with_stage_artifact(StageArtifact(stage=Stage.S4_SKINNING, metrics=bad_metrics))
        )
        stop = next_action(run)
        self.assertIsInstance(stop, StopForRemediation)
        self.assertEqual(stop.stage, Stage.S2_CANONICALIZATION)

        # Remediation lands: S2 (and the now-invalidated S4) get fresh
        # artifacts recorded (same seeds, per the doc -- seeds aren't
        # touched by this test since `next_action` doesn't inspect them).
        remediated = run.with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
        remediated = remediated.with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S4_SKINNING])
        self.assertIsInstance(next_action(remediated), Advance)
        self.assertEqual(next_action(remediated).stage, Stage.S5_ANIMATION)

    def test_next_action_is_deterministic(self):
        metrics = dict(S3_METRICS)
        metrics.update(VIOLATIONS["s3.joints_inside_mesh"])
        run = (
            PipelineRun(source="src.png", family="biped")
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S1_GENERATION])
            .with_stage_artifact(COMPLIANT_ARTIFACTS[Stage.S2_CANONICALIZATION])
            .with_stage_artifact(StageArtifact(stage=Stage.S3_SKELETON_FIT, metrics=metrics))
        )
        first = next_action(run)
        second = next_action(run)
        self.assertEqual(first, second)


class StageArtifactMismatchTests(unittest.TestCase):
    def test_evaluate_rejects_artifact_for_wrong_stage(self):
        with self.assertRaises(ValueError):
            evaluate_stage(Stage.S2_CANONICALIZATION, COMPLIANT_ARTIFACTS[Stage.S0_SOURCE])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Metric adapters: real job metadata -> StageArtifact metrics (bead x8qhy)
#
# Fixtures below are trimmed copies of REAL artifact metadata from the box's
# `studio-data/artifacts/` (Mary's canonical run: generate 21aac4d4 ->
# canonicalize df6067df -> auto-rig 2215c51e -> apply-clip 5ee113cb), so the
# adapters are tested against the shapes the pipeline actually writes rather
# than against invented ones.
# ---------------------------------------------------------------------------

GENERATE_METADATA = {
    "stage": "textured-geometry",
    "backend": "trellis2-fp8",
    "model": "visualbruno/TRELLIS.2-4B-FP8",
    "resolution": 512,
    "seed": 1337,
    "textured": True,
    "source_artifact_id": "4268f9afb3704517b6daaf8be555d4dc",
    "source_image_artifact_id": "4268f9afb3704517b6daaf8be555d4dc",
    "mesh_quality": {"faces": 231576, "components": 1, "watertight": False},
}

CANONICALIZE_METADATA = {
    "stage": "canonicalize",
    "backend": "blender-voxel-remesh+blender-canonicalize-bake",
    "remesh_method": "voxel",
    "source_artifact_id": "21aac4d416144044ba4c5e8c5f6fc487",
    "s1_s2_image_similarity_score": 0.994851,
    "similarity": {"s1_s2_image_similarity_score": 0.994851, "aggregate_distance": 0.005149},
    "settings": {},
    "attempts": [{
        "attempt": 1,
        "method": "voxel",
        "components": 1,
        "components_unwelded": 1,
        "watertight": True,
        "non_manifold_edges": 0,
        "faces": 29992,
    }],
    "mesh_quality": {
        "faces": 29992,
        "components": 1,
        "watertight": True,
        "non_manifold_edges": 0,
        "bounds": {"minimum": [-0.50065, 0.0, -0.11208], "maximum": [0.5001, 0.99814, 0.11201]},
    },
}

AUTO_RIG_METADATA = {
    "stage": "auto-rig",
    "backend": "unirig-auto-rig",
    "source_artifact_id": "df6067dfcbc04ef7813aea834d02d352",
    "weighting": "blender-bone-heat",
    "rig_quality": {
        "joints": 21,
        "joint_names": ["head", "neck", "pelvis", "spine_1"],
        "maximum_influences": 4,
        "non_normalized_vertices": 0,
        "template_conformance": [],
        "animations": [{"name": "Rest", "animated_joints": 0, "invalid_targets": 0}],
    },
    "deformation_review": {
        "complete_action_range": True,
        "maximum_p99_edge_stretch": 1.0,
        "action": "Rest_UniRigArmature",
    },
    "quality_decisions": {"rig-certificate": {"status": "ready", "profile": "humanoid"}},
}

APPLY_CLIP_METADATA = {
    "stage": "apply-clip",
    "backend": "blender-direct-attach",
    "source_artifact_id": "2215c51e77ef4235ae8bf7a0a272590c",
    "clip": "Sword_Attack_Armature",
    "family": "biped",
    "mechanism": "direct-attach",
    "rig_quality": {
        "joints": 21,
        "animations": [{
            "name": "Sword_Attack_Armature",
            "animated_joints": 18,
            "invalid_targets": 0,
        }],
        "template_conformance": None,
    },
    "deformation_review": {"complete_action_range": True, "maximum_p99_edge_stretch": 4.1403},
}


class StageClassificationTests(unittest.TestCase):
    def test_generate_output_is_recognized_despite_its_rewritten_stage_label(self):
        # `native.run_trellis_fp8` overwrites metadata["stage"] with
        # "textured-geometry"; the classifier must not miss S1 because of it.
        self.assertEqual(
            artifact_stages(GENERATE_METADATA, "glb"), (Stage.S1_GENERATION,)
        )

    def test_a_rig_job_speaks_for_both_the_fit_and_the_skinning_stage(self):
        self.assertEqual(
            artifact_stages(AUTO_RIG_METADATA, "glb"),
            (Stage.S3_SKELETON_FIT, Stage.S4_SKINNING),
        )

    def test_a_raw_uploaded_image_is_the_runs_source_stage(self):
        self.assertEqual(artifact_stages({}, "image"), (Stage.S0_SOURCE,))

    def test_view_repaint_passes_are_s2_refinements_like_the_studio_says(self):
        # The studio files both under Clean (journey.ts / workspace.ts
        # CLEAN_OUTPUT_STAGES); the policy must agree or a rebuild-views
        # output claims no stage and is gated as a refinement of its input.
        self.assertEqual(
            artifact_stages({"stage": "project-views"}, "glb"), (Stage.S2_CANONICALIZATION,)
        )
        self.assertEqual(
            artifact_stages({"stage": "rebuild-views"}, "glb"), (Stage.S2_CANONICALIZATION,)
        )

    def test_retopo_is_not_mapped_onto_any_policy_stage(self):
        # Mapping a retopo (which records no seed or model) onto S1 would
        # mask the real generation artifact's provenance.
        self.assertEqual(artifact_stages({"stage": "retopo"}, "glb"), ())
        self.assertEqual(artifact_stages({"stage": "retopologize"}, "glb"), ())


class StageMetricAdapterTests(unittest.TestCase):
    def test_s1_maps_textures_views_and_provenance(self):
        artifact = stage_artifact_from_metadata(Stage.S1_GENERATION, GENERATE_METADATA, "gen-1")
        self.assertTrue(artifact.metrics["textures_present"])
        self.assertTrue(artifact.metrics["source_views_recorded"])
        self.assertEqual(artifact.provenance["seed"], 1337)
        self.assertEqual(artifact.provenance["source_artifact_id"], GENERATE_METADATA["source_artifact_id"])
        self.assertIn("model", artifact.provenance["tool_versions"])
        decision = evaluate_stage(Stage.S1_GENERATION, artifact)
        # Provenance is complete, so the remaining gaps are the un-measured
        # "recognizable subject" judgement and the source-fidelity checks --
        # this fixture predates that measurement.
        self.assertEqual(decision["status"], "not_evaluable")
        self.assertEqual(
            [f["code"] for f in decision["findings"]],
            [
                "s1.recognizable_subject",
                "s1.source_fidelity_silhouette",
                "s1.source_fidelity_face",
            ],
        )

    def test_s1_missing_seed_is_a_real_provenance_error_not_a_pass(self):
        metadata = dict(GENERATE_METADATA)
        metadata.pop("seed")
        artifact = stage_artifact_from_metadata(Stage.S1_GENERATION, metadata, "gen-1")
        codes = [f["code"] for f in evaluate_stage(Stage.S1_GENERATION, artifact)["findings"]]
        self.assertIn("provenance.missing_seed", codes)

    def test_s2_reads_the_canonicalize_gate_report_and_the_similarity_score(self):
        artifact = stage_artifact_from_metadata(
            Stage.S2_CANONICALIZATION, CANONICALIZE_METADATA, "canon-1"
        )
        self.assertEqual(artifact.metrics["connected_components_welded"], 1)
        self.assertEqual(artifact.metrics["connected_components_unwelded"], 1)
        self.assertTrue(artifact.metrics["watertight"])
        self.assertEqual(artifact.metrics["non_manifold_edges"], 0)
        self.assertEqual(artifact.metrics["triangle_count"], 29992)
        self.assertAlmostEqual(artifact.metrics["s1_s2_image_similarity_score"], 0.994851)
        # GLB is exported Y-up, so the floor axis in a recorded bounds
        # triple is index 1 -- canonicalize normalises it to exactly 0.
        self.assertEqual(artifact.metrics["floor_contact_z_abs"], 0.0)
        decision = evaluate_stage(Stage.S2_CANONICALIZATION, artifact)
        self.assertEqual(
            [f["severity"] for f in decision["findings"]], ["not-evaluable"] * 5
        )
        self.assertEqual(
            sorted(f["code"] for f in decision["findings"]),
            [
                "s2.axis_convention",
                "s2.source_fidelity_face",
                "s2.source_fidelity_silhouette",
                "s2.uniform_scale",
                "s2.uv_unwrapped",
            ],
        )

    def test_s2_prefers_the_gate_report_over_a_re_measured_glb(self):
        metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
        metadata["attempts"][-1]["components"] = 3
        metadata["mesh_quality"]["components"] = 1
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        self.assertEqual(artifact.metrics["connected_components_welded"], 3)

    def test_s2_without_a_similarity_measurement_is_not_evaluable_not_a_pass(self):
        metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
        metadata.pop("s1_s2_image_similarity_score")
        metadata.pop("similarity")
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        self.assertNotIn("s1_s2_image_similarity_score", artifact.metrics)
        finding = next(
            f for f in evaluate_stage(Stage.S2_CANONICALIZATION, artifact)["findings"]
            if f["code"] == "s2.appearance_preservation_bake_gate"
        )
        self.assertEqual(finding["severity"], "not-evaluable")

    def test_s3_reads_the_recorded_template_conformance_report(self):
        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA, "rig-1")
        self.assertTrue(artifact.metrics["fit_converged"])
        self.assertEqual(artifact.metrics["bone_names"], artifact.metrics["template_bone_names"])
        self.assertEqual(artifact.metrics["bone_parents"], {})
        self.assertEqual(artifact.metrics["max_rest_direction_deviation_degrees"], 0.0)
        self.assertEqual(artifact.metrics["symmetry_band_violation_count"], 0)
        codes = sorted(
            f["code"] for f in evaluate_stage(Stage.S3_SKELETON_FIT, artifact)["findings"]
        )
        self.assertEqual(codes, [
            "s3.bilateral_symmetry_p95",
            "s3.joints_inside_mesh",
            "s3.medial_line_distance",
            "s3.proportion_sanity",
        ])

    def test_s3_conformance_violations_become_contract_violations(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["rig_quality"]["template_conformance"] = [
            {"code": "unknown_bone_name", "severity": "error", "metric": "bone_name_known",
             "actual": "spine_1", "expected": "one of [...]",
             "message": "bone 'spine_1' is not part of the 'biped' template"},
            {"code": "reparented_bone", "severity": "error", "metric": "parent_name",
             "actual": "pelvis", "expected": "spine_1",
             "message": "bone 'neck' has parent 'pelvis', template expects 'spine_1'"},
            {"code": "roll_deviation", "severity": "error", "metric": "roll_angle_deg",
             "actual": 41.2, "expected": "<= 20.0", "message": "bone 'neck' roll is 41.2deg"},
            {"code": "mirrored_left_right", "severity": "error",
             "metric": "angle_to_own_side_vs_mirrored_side_deg",
             "actual": [110.0, 12.0], "expected": "...", "message": "bone 'upper_arm_l' ..."},
        ]
        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")
        self.assertNotIn("spine_1", artifact.metrics["template_bone_names"])
        self.assertEqual(artifact.metrics["bone_parents"], {"neck": "pelvis"})
        self.assertEqual(artifact.metrics["template_bone_parents"], {"neck": "spine_1"})
        self.assertEqual(artifact.metrics["max_roll_deviation_degrees"], 41.2)
        self.assertEqual(artifact.metrics["symmetry_band_violation_count"], 1)
        decision = evaluate_stage(Stage.S3_SKELETON_FIT, artifact)
        self.assertEqual(decision["status"], "blocked")
        self.assertLessEqual(
            {"s3.exact_bone_set", "s3.bone_hierarchy_matches_template",
             "s3.bone_roll_matches_template", "s3.left_right_symmetry"},
            {f["code"] for f in decision["findings"]},
        )

    def test_s3_without_a_conformance_report_is_not_evaluable(self):
        # A chain rig, or a rig predating the template contract: nothing was
        # measured, so nothing may be asserted.
        artifact = stage_artifact_from_metadata(
            Stage.S3_SKELETON_FIT, APPLY_CLIP_METADATA, "clip-1"
        )
        self.assertNotIn("bone_names", artifact.metrics)
        codes = {f["code"] for f in evaluate_stage(Stage.S3_SKELETON_FIT, artifact)["findings"]}
        self.assertIn("s3.exact_bone_set", codes)
        self.assertEqual(
            evaluate_stage(Stage.S3_SKELETON_FIT, artifact)["status"], "not_evaluable"
        )

    def test_s4_maps_solver_influences_and_rom_stretch(self):
        artifact = stage_artifact_from_metadata(Stage.S4_SKINNING, AUTO_RIG_METADATA, "rig-1")
        self.assertEqual(artifact.metrics["solver_used"], "primary")
        self.assertEqual(artifact.metrics["max_influences_per_vertex"], 4)
        self.assertEqual(artifact.metrics["max_weight_sum_deviation"], 0.0)
        # 1.0x rest edge length == zero fractional stretch.
        self.assertEqual(artifact.metrics["rom_edge_stretch_p99"], 0.0)
        codes = sorted(f["code"] for f in evaluate_stage(Stage.S4_SKINNING, artifact)["findings"])
        self.assertEqual(codes, [
            "s4.coincident_vertices_share_weights",
            "s4.per_joint_coverage",
            "s4.rom_seam_tearing",
            "s4.rom_self_intersection",
        ])

    def test_s4_envelope_fallback_is_reported_as_a_non_primary_solver(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["weighting"] = "blender-bone-envelope"
        artifact = stage_artifact_from_metadata(Stage.S4_SKINNING, metadata, "rig-1")
        self.assertEqual(artifact.metrics["solver_used"], "blender-bone-envelope")
        finding = next(
            f for f in evaluate_stage(Stage.S4_SKINNING, artifact)["findings"]
            if f["code"] == "s4.primary_solver_engaged"
        )
        self.assertEqual(finding["severity"], "error")

    def test_s4_seam_stitched_vertices_feed_the_coincident_weight_assertion(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["seam_stitched_vertices"] = 12
        artifact = stage_artifact_from_metadata(Stage.S4_SKINNING, metadata, "rig-1")
        self.assertEqual(artifact.metrics["coincident_weight_mismatch_count"], 12)
        finding = next(
            f for f in evaluate_stage(Stage.S4_SKINNING, artifact)["findings"]
            if f["code"] == "s4.coincident_vertices_share_weights"
        )
        self.assertEqual(finding["severity"], "error")

    def test_s4_non_normalized_weights_leave_the_magnitude_unmapped(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["rig_quality"]["non_normalized_vertices"] = 7
        artifact = stage_artifact_from_metadata(Stage.S4_SKINNING, metadata, "rig-1")
        # rig_quality records the COUNT, not the worst deviation; inventing a
        # magnitude is exactly what this layer must never do.
        self.assertNotIn("max_weight_sum_deviation", artifact.metrics)
        finding = next(
            f for f in evaluate_stage(Stage.S4_SKINNING, artifact)["findings"]
            if f["code"] == "s4.weights_sum_to_one"
        )
        self.assertEqual(finding["severity"], "not-evaluable")

    def test_s5_maps_clip_targets_stretch_and_mechanism(self):
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, APPLY_CLIP_METADATA, "clip-1")
        self.assertTrue(artifact.metrics["clip_bone_names_match_template"])
        self.assertEqual(artifact.metrics["mechanism"], "direct-attach")
        self.assertAlmostEqual(artifact.metrics["clip_p99_stretch"], 3.1403)
        codes = sorted(f["code"] for f in evaluate_stage(Stage.S5_ANIMATION, artifact)["findings"])
        self.assertEqual(codes, [
            "s5.foot_contact_band",
            "s5.rest_pose_hunch_detected",
            "s5.root_motion_policy_honored",
        ])

    def test_s5_retargeted_clip_is_flagged_as_off_the_direct_attach_contract(self):
        metadata = json.loads(json.dumps(APPLY_CLIP_METADATA))
        metadata["mechanism"] = "bvh-retarget"
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")
        finding = next(
            f for f in evaluate_stage(Stage.S5_ANIMATION, artifact)["findings"]
            if f["code"] == "s5.direct_clip_attach"
        )
        # Advisory, not blocking: a retargeted clip still plays.
        self.assertEqual(finding["severity"], "warning")

    def test_s5_channels_targeting_missing_joints_violate_the_bone_name_contract(self):
        metadata = json.loads(json.dumps(APPLY_CLIP_METADATA))
        metadata["rig_quality"]["animations"][0]["invalid_targets"] = 3
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")
        self.assertFalse(artifact.metrics["clip_bone_names_match_template"])
        finding = next(
            f for f in evaluate_stage(Stage.S5_ANIMATION, artifact)["findings"]
            if f["code"] == "s5.exact_bone_name_match"
        )
        self.assertEqual(finding["severity"], "error")

    def test_a_partial_deformation_review_is_not_stretch_evidence(self):
        metadata = json.loads(json.dumps(APPLY_CLIP_METADATA))
        metadata["deformation_review"]["complete_action_range"] = False
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")
        self.assertNotIn("clip_p99_stretch", artifact.metrics)

    def test_every_populated_metric_declares_where_it_came_from(self):
        for stage, metadata in (
            (Stage.S1_GENERATION, GENERATE_METADATA),
            (Stage.S2_CANONICALIZATION, CANONICALIZE_METADATA),
            (Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA),
            (Stage.S4_SKINNING, AUTO_RIG_METADATA),
            (Stage.S5_ANIMATION, APPLY_CLIP_METADATA),
        ):
            artifact = stage_artifact_from_metadata(stage, metadata, "a-1")
            for key in artifact.metrics:
                if key == "triangle_budget_band":
                    continue
                with self.subTest(stage=stage, metric=key):
                    self.assertIn(key, METRIC_SOURCES)


class RunAssemblyTests(unittest.TestCase):
    """Mary's canonical chain: image -> generate -> voxel canonicalize ->
    template rig -> direct clip attach."""

    def _records(self):
        return [
            ArtifactRecord("4268f9af", {}, "image", on_lineage=True),
            ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("df6067df", CANONICALIZE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("2215c51e", AUTO_RIG_METADATA, "glb", on_lineage=True),
            ArtifactRecord("5ee113cb", APPLY_CLIP_METADATA, "glb", on_lineage=True),
        ]

    def test_the_canonical_chain_places_every_stage_and_advances_to_export(self):
        run, placements = assemble_pipeline_run(self._records(), source="run-1")
        self.assertEqual(run.family, "biped")
        self.assertEqual(placements[Stage.S2_CANONICALIZATION].artifact_id, "df6067df")
        self.assertEqual(placements[Stage.S3_SKELETON_FIT].artifact_id, "2215c51e")
        self.assertEqual(placements[Stage.S4_SKINNING].artifact_id, "2215c51e")
        self.assertEqual(placements[Stage.S5_ANIMATION].artifact_id, "5ee113cb")
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S6_EXPORT)

    def test_an_on_lineage_artifact_is_never_displaced_by_an_off_lineage_one(self):
        stale = ArtifactRecord("stale-rig", AUTO_RIG_METADATA, "glb", on_lineage=False)
        run, placements = assemble_pipeline_run(self._records() + [stale], source="run-1")
        self.assertEqual(placements[Stage.S3_SKELETON_FIT].artifact_id, "2215c51e")

    def test_a_remediation_takes_its_stage_even_from_the_current_lineage(self):
        # After a remediation, one more job on the pre-fix branch makes the
        # SUPERSEDED mesh the run's lineage again. The stage still belongs
        # to the artifact that replaced it, or the fix would be lost and
        # everything built on the old mesh would read as current.
        remediated = ArtifactRecord(
            "canon-2", REMEDIATED_CANONICALIZE_METADATA, "glb", on_lineage=False
        )
        _, placements = assemble_pipeline_run(
            self._records() + [remediated], source="run-1"
        )
        self.assertEqual(placements[Stage.S2_CANONICALIZATION].artifact_id, "canon-2")

    def test_an_ordinary_sibling_never_takes_a_stage_back_off_a_remediation(self):
        remediated = ArtifactRecord(
            "canon-2", REMEDIATED_CANONICALIZE_METADATA, "glb", on_lineage=False
        )
        sibling = ArtifactRecord("canon-3", CANONICALIZE_METADATA, "glb", on_lineage=False)
        _, placements = assemble_pipeline_run(
            self._records() + [remediated, sibling], source="run-1"
        )
        self.assertEqual(placements[Stage.S2_CANONICALIZATION].artifact_id, "canon-2")

    def test_a_stage_the_lineage_never_reached_is_filled_from_a_sibling_branch(self):
        records = [r for r in self._records() if r.artifact_id != "5ee113cb"]
        sibling = ArtifactRecord("sibling-clip", APPLY_CLIP_METADATA, "glb", on_lineage=False)
        _, placements = assemble_pipeline_run(records + [sibling], source="run-1")
        self.assertEqual(placements[Stage.S5_ANIMATION].artifact_id, "sibling-clip")
        self.assertFalse(placements[Stage.S5_ANIMATION].on_lineage)

    def test_a_missing_canonicalize_stops_the_run_at_s2(self):
        records = [r for r in self._records() if r.artifact_id != "df6067df"]
        run, _ = assemble_pipeline_run(records, source="run-1")
        action = next_action(run)
        self.assertIsInstance(action, Advance)
        self.assertEqual(action.stage, Stage.S2_CANONICALIZATION)

    def test_an_unwelded_seam_split_blames_s2_and_stops_the_run_there(self):
        # The blame table's "08-30 Mary hips" row, driven end to end from
        # recorded canonicalize gate numbers.
        records = self._records()
        broken = json.loads(json.dumps(CANONICALIZE_METADATA))
        broken["attempts"][-1]["components_unwelded"] = 4
        records[2] = ArtifactRecord("df6067df", broken, "glb", on_lineage=True)
        run, _ = assemble_pipeline_run(records, source="run-1")
        action = next_action(run)
        self.assertIsInstance(action, StopForRemediation)
        self.assertEqual(action.stage, Stage.S2_CANONICALIZATION)
        self.assertEqual(action.triggering_stage, Stage.S2_CANONICALIZATION)


class SummaryTests(unittest.TestCase):
    def _run(self, records):
        run, placements = assemble_pipeline_run(records, source="run-1")
        return summarize_pipeline_policy(run, placements)

    def test_summary_reports_status_blame_and_the_not_evaluable_list(self):
        summary = self._run([
            ArtifactRecord("4268f9af", {}, "image", on_lineage=True),
            ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("df6067df", CANONICALIZE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("2215c51e", AUTO_RIG_METADATA, "glb", on_lineage=True),
            ArtifactRecord("5ee113cb", APPLY_CLIP_METADATA, "glb", on_lineage=True),
        ])
        self.assertEqual(summary["policy_version"], 1)
        self.assertEqual([item["stage"] for item in summary["stages"]],
                         [stage.value for stage in STAGE_ORDER])
        self.assertIsNone(summary["first_violated_stage"])
        self.assertEqual(summary["next_action"],
                         {"kind": "advance", "stage": "S6_EXPORT"})
        self.assertEqual(summary["stages_not_run"], ["S6_EXPORT"])
        self.assertEqual(summary["off_lineage_stages"], [])
        # Every not-evaluable entry names the check and the metric that is
        # missing, so the gaps are a work list rather than a silence.
        self.assertTrue(summary["not_evaluable"])
        for entry in summary["not_evaluable"]:
            self.assertTrue(entry["missing_metrics"])
        s2 = next(item for item in summary["stages"] if item["stage"] == "S2_CANONICALIZATION")
        self.assertIn("s1_s2_image_similarity_score", s2["metric_sources"])

    def test_a_violation_summary_carries_the_blame_table_rationale(self):
        broken = json.loads(json.dumps(CANONICALIZE_METADATA))
        broken["s1_s2_image_similarity_score"] = 0.4
        broken["similarity"]["s1_s2_image_similarity_score"] = 0.4
        summary = self._run([
            ArtifactRecord("4268f9af", {}, "image", on_lineage=True),
            ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("df6067df", broken, "glb", on_lineage=True),
        ])
        self.assertEqual(summary["first_violated_stage"], "S2_CANONICALIZATION")
        self.assertEqual(summary["next_action"]["kind"], "stop-at-stage")
        self.assertEqual(summary["next_action"]["stage"], "S2_CANONICALIZATION")
        blame = summary["next_action"]["blame"][0]
        self.assertEqual(blame["finding_code"], "s2.appearance_preservation_bake_gate")
        self.assertEqual(blame["rationale"], "bake gate")
        self.assertEqual(blame["incident_ref"], "08-30 AI-Refine regression")

    def test_compact_summaries_drop_messages_for_metadata_storage(self):
        run, placements = assemble_pipeline_run(
            [ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True)],
            source="run-1",
        )
        full = summarize_pipeline_policy(run, placements)
        compact = summarize_pipeline_policy(run, placements, compact=True)
        self.assertIn("message", full["stages"][1]["findings"][0])
        self.assertNotIn("message", compact["stages"][1]["findings"][0])


# ---------------------------------------------------------------------------
# Lineage staleness (bead 38nlk)
#
# The ids below are the real ones from the box, so these fixtures are the
# two runs the rule has to tell apart:
#
#   070772d8  S2 remediated into a NEW canonical mesh (ae580728) while the
#             rig (5610c84a) still descends from the pre-fix geometry. The
#             loop used to report `advance -> S5`, stepping straight over a
#             rig of a mesh that no longer exists.
#   a459f7b0  ("Mary") started from an already-rigged upload (2215c51e),
#             attached a clip to it, then produced her own rig. Two rigs,
#             no remediation, nothing invalidated -- and a rule that called
#             her clip stale would spend a GPU on a false positive.
# ---------------------------------------------------------------------------

REMEDIATED_CANONICALIZE_METADATA = {
    **json.loads(json.dumps(CANONICALIZE_METADATA)),
    "settings": {
        "pipeline_step": {
            "initiator": "api",
            "mode": "remediation",
            "stage": "S2_CANONICALIZATION",
            "policy_action": "stop-at-stage",
            "attempt": 1,
            "source_artifact_id": "21aac4d4",
        }
    },
}


class LineageStalenessTests(unittest.TestCase):
    """`detect_stale_stages`: which recorded work a remediation obsoleted."""

    def _post_remediation_records(self, *, remediation_provenance=True):
        """Run 070772d8's shape: one canonicalize, one rig built off the
        GENERATED mesh, then a second canonicalize that supersedes the
        first. The rig descends from neither canonical mesh."""
        canonical_new = (
            REMEDIATED_CANONICALIZE_METADATA
            if remediation_provenance
            else CANONICALIZE_METADATA
        )
        rig = {**AUTO_RIG_METADATA, "source_artifact_id": "21aac4d4"}
        return [
            ArtifactRecord("4268f9af", {}, "image", on_lineage=True),
            ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True),
            ArtifactRecord(
                "ae580728", canonical_new, "glb", on_lineage=True, produced_in_run=True
            ),
            ArtifactRecord(
                "57405dbb", CANONICALIZE_METADATA, "glb", produced_in_run=True
            ),
            ArtifactRecord("5610c84a", rig, "glb", produced_in_run=True),
        ]

    def _mary_records(self):
        """Run a459f7b0's shape: an inherited rig (2215c51e, the run's own
        source upload), a clip attached to it, and the run's own re-rig."""
        return [
            ArtifactRecord("4268f9af", {}, "image", on_lineage=True),
            ArtifactRecord("21aac4d4", GENERATE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("df6067df", CANONICALIZE_METADATA, "glb", on_lineage=True),
            ArtifactRecord("2215c51e", AUTO_RIG_METADATA, "glb", on_lineage=True),
            ArtifactRecord(
                "8c3a3ba7",
                {**AUTO_RIG_METADATA, "source_artifact_id": "2215c51e"},
                "glb",
                on_lineage=True,
                produced_in_run=True,
            ),
            ArtifactRecord(
                "5ee113cb", APPLY_CLIP_METADATA, "glb", produced_in_run=True
            ),
        ]

    def test_a_rig_that_predates_the_remediated_mesh_is_stale(self):
        records = self._post_remediation_records()
        _, placements = assemble_pipeline_run(records, source="run-1")
        stale = detect_stale_stages(records, placements)

        self.assertEqual(
            sorted(stage.value for stage in stale),
            ["S3_SKELETON_FIT", "S4_SKINNING"],
        )
        entry = stale[Stage.S3_SKELETON_FIT]
        self.assertEqual(entry.artifact_id, "5610c84a")
        self.assertEqual(entry.obsoleted_by_stage, Stage.S2_CANONICALIZATION)
        self.assertEqual(entry.obsoleted_by_artifact_id, "ae580728")
        self.assertEqual(entry.trigger, "remediation")

    def test_the_remediated_stage_itself_is_never_stale(self):
        records = self._post_remediation_records()
        _, placements = assemble_pipeline_run(records, source="run-1")
        stale = detect_stale_stages(records, placements)
        for stage in (Stage.S0_SOURCE, Stage.S1_GENERATION, Stage.S2_CANONICALIZATION):
            self.assertNotIn(stage, stale)

    def test_a_hand_rerun_stage_invalidates_downstream_without_step_provenance(self):
        # The "Clean (canonical)" button re-runs S2 without the step
        # driver's provenance block. A re-run is a re-run: what it
        # supersedes is superseded either way.
        records = self._post_remediation_records(remediation_provenance=False)
        _, placements = assemble_pipeline_run(records, source="run-1")
        stale = detect_stale_stages(records, placements)
        self.assertIn(Stage.S3_SKELETON_FIT, stale)
        self.assertEqual(stale[Stage.S3_SKELETON_FIT].trigger, "stage_rerun")

    def test_work_that_descends_from_the_remediation_is_fresh(self):
        records = self._post_remediation_records()
        records.append(
            ArtifactRecord(
                "rig-2",
                {**AUTO_RIG_METADATA, "source_artifact_id": "ae580728"},
                "glb",
                on_lineage=True,
                produced_in_run=True,
            )
        )
        _, placements = assemble_pipeline_run(records, source="run-1")
        self.assertEqual(placements[Stage.S3_SKELETON_FIT].artifact_id, "rig-2")
        self.assertEqual(detect_stale_stages(records, placements), {})

    def test_staleness_carries_through_to_work_built_on_a_stale_stage(self):
        records = self._post_remediation_records()
        records.append(
            ArtifactRecord(
                "clip-1",
                {**APPLY_CLIP_METADATA, "source_artifact_id": "5610c84a"},
                "glb",
                produced_in_run=True,
            )
        )
        _, placements = assemble_pipeline_run(records, source="run-1")
        stale = detect_stale_stages(records, placements)
        # The clip descends from the stale rig, so it does not descend from
        # the remediated mesh either -- and the loop re-enters at the
        # EARLIEST stale stage, not at the clip.
        self.assertIn(Stage.S5_ANIMATION, stale)
        self.assertEqual(stale[Stage.S5_ANIMATION].obsoleted_by_artifact_id, "ae580728")

    def test_an_inherited_rig_is_not_a_remediation_and_nothing_goes_stale(self):
        # The false-positive guard: Mary's run holds two rigs, but only one
        # of them is work SHE ran. Nothing was invalidated, so her clip
        # stands and her evaluation is unchanged.
        records = self._mary_records()
        _, placements = assemble_pipeline_run(records, source="run-2")
        self.assertEqual(detect_stale_stages(records, placements), {})

    def test_a_remediation_is_visible_from_the_placements_alone(self):
        # The provenance rides on the remediated artifact itself, so the
        # rule still fires for a caller that has only the placed stages.
        records = self._post_remediation_records()
        _, placements = assemble_pipeline_run(records, source="run-1")
        self.assertIn(Stage.S3_SKELETON_FIT, detect_stale_stages([], placements))

    def test_a_hand_rerun_is_only_knowable_from_the_runs_records(self):
        # Nothing on a hand-re-run artifact says "I superseded something";
        # only the run's own record of having produced two artifacts for
        # that stage does. Without the records, the summary reports what it
        # can prove -- and proves nothing.
        records = self._post_remediation_records(remediation_provenance=False)
        run, placements = assemble_pipeline_run(records, source="run-1")
        self.assertEqual(summarize_pipeline_policy(run, placements)["stale_stages"], [])
        self.assertTrue(
            summarize_pipeline_policy(run, placements, records=records)["stale_stages"]
        )


class StaleStageActionTests(unittest.TestCase):
    """What the loop DOES about a stale stage: re-enter it, and say so."""

    def setUp(self):
        self.records = LineageStalenessTests()._post_remediation_records()
        self.run, self.placements = assemble_pipeline_run(self.records, source="run-1")
        self.stale = detect_stale_stages(self.records, self.placements)

    def test_next_action_re_enters_the_stale_stage_instead_of_advancing_past_it(self):
        # Without the stale set this run reports "advance -> S5": it counts
        # a rig of the pre-fix mesh as S3/S4 done. That is the bug.
        self.assertEqual(next_action(self.run), Advance(stage=Stage.S5_ANIMATION))
        action = next_action(self.run, self.stale)
        self.assertEqual(action, Advance(stage=Stage.S3_SKELETON_FIT))

    def test_the_summary_reports_the_stale_stages_rather_than_hiding_them(self):
        summary = summarize_pipeline_policy(
            self.run, self.placements, records=self.records
        )
        self.assertEqual(
            summary["next_action"], {"kind": "advance", "stage": "S3_SKELETON_FIT"}
        )
        self.assertEqual(
            [entry["stage"] for entry in summary["stale_stages"]],
            ["S3_SKELETON_FIT", "S4_SKINNING"],
        )
        entry = summary["stale_stages"][0]
        self.assertEqual(entry["artifact_id"], "5610c84a")
        self.assertEqual(
            entry["obsoleted_by"],
            {"stage": "S2_CANONICALIZATION", "artifact_id": "ae580728"},
        )
        self.assertIn("does not descend", entry["reason"])
        # The stale stage keeps its own verdict -- flagged, not deleted.
        s3 = next(item for item in summary["stages"] if item["stage"] == "S3_SKELETON_FIT")
        self.assertTrue(s3["stale"])
        self.assertEqual(s3["artifact_id"], "5610c84a")
        self.assertNotEqual(s3["status"], "not_run")
        s2 = next(item for item in summary["stages"] if item["stage"] == "S2_CANONICALIZATION")
        self.assertFalse(s2["stale"])

    def test_a_clean_run_reports_an_empty_stale_list(self):
        records = LineageStalenessTests()._mary_records()
        run, placements = assemble_pipeline_run(records, source="run-2")
        summary = summarize_pipeline_policy(run, placements, records=records)
        self.assertEqual(summary["stale_stages"], [])
        self.assertEqual(summary["next_action"], {"kind": "advance", "stage": "S6_EXPORT"})
        self.assertFalse(any(item["stale"] for item in summary["stages"]))


# The 5jooo measurements, as the workers actually record them. Kept separate
# from the base fixtures above so the "an artifact that predates the
# measurement stays not-evaluable" half of every flip is testable against the
# same metadata the rest of this file uses.
GEOMETRY_CONTRACT = {
    "schema_version": 1,
    "uv_unwrapped": True,
    "uv_faces": 29992,
    "uv_total_area": 0.03653551,
    "uv_non_degenerate_face_fraction": 1.0,
    "uv_degenerate_faces": 0,
    "uniform_scale_ok": True,
    "identity_scale": True,
    "objects": 1,
    "max_scale_anisotropy": 0.0,
    "up_axis": "Z",
    "up_axis_source": "blender-gltf-import-convention",
    "extent_profile": [1.0, 0.22392, 0.997388],
    "mesh_nodes_identity_transforms": True,
    "axis_profile_preserved": True,
    "axis_convention_ok": True,
}

JOINT_CONTAINMENT = {
    "schema_version": 1,
    "measurable": True,
    "method": "ray-blocking-5-axes",
    "measured_joints": 21,
    "inside_joints": 21,
    "joints_outside_mesh_count": 0,
    "joints_inside_mesh_fraction": 1.0,
    "triangles": 29992,
    "enclosed_volume": 0.0040265,
    "bounding_box_volume": 0.22384191,
    "enclosed_volume_ratio": 0.017988,
    "worst_offenders": [],
}

BILATERAL_SYMMETRY = {
    "schema_version": 1,
    "measurable": True,
    "method": "sagittal-mirror-bind-pose",
    "pairs_measured": 8,
    "pairs_skipped": [],
    "pairs": [
        {"left": "upper_arm_l", "right": "upper_arm_r", "mirror_error_normalized": 0.0,
         "length_ratio": 1.0, "direction_angle_degrees": 0.0},
    ],
    "worst_offenders": [
        {"left": "upper_arm_l", "right": "upper_arm_r", "mirror_error_normalized": 0.0,
         "length_ratio": 1.0, "direction_angle_degrees": 0.0},
    ],
    "mirror_error_p95": 0.0,
    "mirror_error_max": 0.0,
    "pairs_outside_band_count": 0,
    "band": 0.03,
    "caveats": [],
}

# `Walk_Armature` as measured on the store's own template armature by
# `scripts/calibrate_foot_contact.py` (2026-09-01). The raw figure is the
# same here because a walk is grounded throughout; what matters is that the
# stride-driven 0.3055 is well inside the calibrated advisory band.
MOTION_REVIEW = {
    "schema_version": 1,
    "measurable": True,
    "synthetic": False,
    "source_action": "Walk_Armature",
    "sampled_frames": 41,
    "joint_motion_coverage": 0.952381,
    "foot_slide_max": 0.831331,
    "foot_slide_mean": 0.243657,
    "foot_slide_max_in_contact": 0.305542,
    "foot_slide_mean_in_contact": 0.201,
    "stance_frames_fraction": 1.0,
    "stance_switch_count": 2,
    "contact_stance_episodes": 3,
    "static_tail_fraction": 0.022727,
    "root_travel_path": 0.084123,
    "root_travel_net": 0.0,
}


def _with_geometry_contract(**overrides):
    metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
    contract = dict(GEOMETRY_CONTRACT)
    contract.update(overrides)
    metadata["geometry_contract"] = contract
    metadata["uv_unwrapped"] = contract["uv_unwrapped"]
    metadata["uniform_scale_ok"] = contract["uniform_scale_ok"]
    metadata["axis_convention_ok"] = contract["axis_convention_ok"]
    return metadata


def _with_containment(**overrides):
    metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
    containment = dict(JOINT_CONTAINMENT)
    containment.update(overrides)
    metadata["rig_quality"]["joint_containment"] = containment
    metadata["rig_quality"]["joints_outside_mesh_count"] = containment.get(
        "joints_outside_mesh_count"
    )
    return metadata


def _with_symmetry(**overrides):
    metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
    symmetry = dict(BILATERAL_SYMMETRY)
    symmetry.update(overrides)
    metadata["rig_quality"]["bilateral_symmetry"] = symmetry
    metadata["rig_quality"]["symmetry_mirror_error_p95"] = symmetry.get("mirror_error_p95")
    metadata["rig_quality"]["symmetry_pairs_outside_band_count"] = symmetry.get(
        "pairs_outside_band_count"
    )
    return metadata


def _with_motion_review(**overrides):
    metadata = json.loads(json.dumps(APPLY_CLIP_METADATA))
    motion = dict(MOTION_REVIEW)
    motion.update(overrides)
    metadata["deformation_review"]["motion_review"] = motion
    return metadata


def _codes(stage, metadata, severity=None):
    artifact = stage_artifact_from_metadata(stage, metadata, "a-1")
    findings = evaluate_stage(stage, artifact)["findings"]
    return sorted(
        f["code"] for f in findings if severity is None or f["severity"] == severity
    )


class MeasuredS2MetricsTests(unittest.TestCase):
    """`canonicalize.geometry_contract` turns three S2 checks from
    not-evaluable into real verdicts -- and leaves the fourth honestly
    unanswerable."""

    def test_geometry_contract_flips_uv_and_scale_to_evaluated_passes(self):
        metadata = _with_geometry_contract()
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")

        self.assertIs(artifact.metrics["uv_unwrapped"], True)
        self.assertIs(artifact.metrics["uniform_scale"], True)
        self.assertEqual(artifact.metrics["up_axis"], "Z")

        # `s2.uv_unwrapped` and `s2.uniform_scale` were not-evaluable before
        # the measurement and are clean passes now; `s2.axis_convention` and
        # the source-fidelity checks (this fixture predates that
        # measurement) are the remaining gaps on this artifact.
        self.assertEqual(
            _codes(Stage.S2_CANONICALIZATION, metadata, "not-evaluable"),
            ["s2.axis_convention", "s2.source_fidelity_face", "s2.source_fidelity_silhouette"],
        )
        self.assertEqual(_codes(Stage.S2_CANONICALIZATION, metadata, "error"), [])

    def test_an_artifact_predating_the_measurement_stays_not_evaluable(self):
        artifact = stage_artifact_from_metadata(
            Stage.S2_CANONICALIZATION, CANONICALIZE_METADATA, "canon-1"
        )

        for key in ("uv_unwrapped", "uniform_scale", "up_axis"):
            self.assertNotIn(key, artifact.metrics)
        self.assertEqual(
            _codes(Stage.S2_CANONICALIZATION, CANONICALIZE_METADATA, "not-evaluable"),
            [
                "s2.axis_convention",
                "s2.source_fidelity_face",
                "s2.source_fidelity_silhouette",
                "s2.uniform_scale",
                "s2.uv_unwrapped",
            ],
        )

    def test_a_measured_false_is_a_real_violation_not_a_gap(self):
        # The point of recording the booleans instead of only raising on
        # them: a future loosening of the raise shows up as a red check.
        metadata = _with_geometry_contract(uv_unwrapped=False, uniform_scale_ok=False)

        self.assertEqual(
            _codes(Stage.S2_CANONICALIZATION, metadata, "error"),
            ["s2.uniform_scale", "s2.uv_unwrapped"],
        )

    def test_the_nested_block_is_read_when_the_top_level_mirror_is_missing(self):
        metadata = _with_geometry_contract()
        for key in ("uv_unwrapped", "uniform_scale_ok", "axis_convention_ok"):
            metadata.pop(key)

        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")

        self.assertIs(artifact.metrics["uv_unwrapped"], True)
        self.assertIs(artifact.metrics["uniform_scale"], True)

    def test_axis_convention_reports_only_facing_axis_as_the_missing_half(self):
        metadata = _with_geometry_contract()
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        finding = next(
            f for f in evaluate_stage(Stage.S2_CANONICALIZATION, artifact)["findings"]
            if f["code"] == "s2.axis_convention"
        )

        # Nothing measures subject-forward before a rig exists, so the check
        # must stay not-evaluable rather than pass on the half it has.
        self.assertEqual(finding["severity"], "not-evaluable")
        self.assertEqual(finding["metric"], "facing_axis")
        self.assertNotIn("up_axis", finding["expected"])


class TierAwareTriangleBudgetAdapterTests(unittest.TestCase):
    """`_s2_canonicalization_metrics` picks the triangle band up from the
    quality tier the run selected (bead i63ky) -- and leaves a tier-less
    legacy artifact judged against the fixed family band."""

    @staticmethod
    def _canonicalize_metadata(faces, **extra):
        metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
        metadata["attempts"][-1]["faces"] = faces
        metadata["mesh_quality"]["faces"] = faces
        metadata.update(extra)
        return metadata

    def test_an_ultra_run_is_judged_against_the_tier_s_band(self):
        metadata = self._canonicalize_metadata(
            148_000, quality_tier="ultra", target_triangles=150_000,
            triangle_budget_band=[75_000, 225_000],
        )
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")

        self.assertEqual(artifact.metrics["triangle_budget_band"], (75_000, 225_000))
        self.assertEqual(artifact.metrics["quality_tier"], "ultra")
        self.assertNotIn("s2.triangle_budget", _codes(Stage.S2_CANONICALIZATION, metadata, "error"))

    def test_a_tier_less_legacy_artifact_keeps_the_fixed_family_band(self):
        # The band must NOT silently widen for a run that never asked for a
        # denser mesh: the same 148k triangles are a violation here.
        metadata = self._canonicalize_metadata(148_000)
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")

        self.assertNotIn("triangle_budget_band", artifact.metrics)
        self.assertNotIn("quality_tier", artifact.metrics)
        self.assertIn("s2.triangle_budget", _codes(Stage.S2_CANONICALIZATION, metadata, "error"))

    def test_an_explicit_family_band_override_still_wins(self):
        metadata = self._canonicalize_metadata(
            148_000, quality_tier="ultra", triangle_budget_band=[75_000, 225_000],
        )
        metadata["settings"] = {"family_band": [10_000, 50_000]}
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")

        self.assertEqual(artifact.metrics["triangle_budget_band"], (10_000, 50_000))

    def test_a_tier_run_outside_its_own_band_is_still_a_violation(self):
        metadata = self._canonicalize_metadata(
            300_000, quality_tier="ultra", triangle_budget_band=[75_000, 225_000],
        )

        self.assertIn("s2.triangle_budget", _codes(Stage.S2_CANONICALIZATION, metadata, "error"))

    def test_the_violation_finding_names_the_tier(self):
        metadata = self._canonicalize_metadata(
            300_000, quality_tier="ultra", triangle_budget_band=[75_000, 225_000],
        )
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        finding = next(
            f for f in evaluate_stage(Stage.S2_CANONICALIZATION, artifact)["findings"]
            if f["code"] == "s2.triangle_budget"
        )

        self.assertIn("ultra", finding["message"])
        self.assertEqual(finding["expected"], "[75000, 225000]")


class SourceFidelityMetricsAdapterTests(unittest.TestCase):
    """`_s1_generation_metrics`/`_s2_canonicalization_metrics` map
    `metadata["source_fidelity"]` onto the checks' metric names, and treat a
    `{"skipped": ...}` report the same as no report at all -- the checks
    stay honestly not-evaluable rather than reading a placeholder."""

    def test_s1_low_silhouette_iou_surfaces_the_warning(self):
        metadata = json.loads(json.dumps(GENERATE_METADATA))
        metadata["source_fidelity"] = {"silhouette_iou": 0.5, "face_ssim": 0.9}
        artifact = stage_artifact_from_metadata(Stage.S1_GENERATION, metadata, "gen-1")
        self.assertEqual(artifact.metrics["source_fidelity_silhouette_iou"], 0.5)
        codes = [f["code"] for f in evaluate_stage(Stage.S1_GENERATION, artifact)["findings"]]
        self.assertIn("s1.source_fidelity_silhouette", codes)

    def test_s2_low_face_ssim_surfaces_the_warning(self):
        metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
        metadata["source_fidelity"] = {"silhouette_iou": 0.95, "face_ssim": 0.1}
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        self.assertEqual(artifact.metrics["source_fidelity_face_ssim"], 0.1)
        codes = [f["code"] for f in evaluate_stage(Stage.S2_CANONICALIZATION, artifact)["findings"]]
        self.assertIn("s2.source_fidelity_face", codes)

    def test_the_metric_is_absent_without_a_report_and_the_check_is_not_evaluable(self):
        artifact = stage_artifact_from_metadata(Stage.S1_GENERATION, GENERATE_METADATA, "gen-1")
        self.assertNotIn("source_fidelity_silhouette_iou", artifact.metrics)
        codes = _codes(Stage.S1_GENERATION, GENERATE_METADATA, "not-evaluable")
        self.assertIn("s1.source_fidelity_silhouette", codes)
        self.assertIn("s1.source_fidelity_face", codes)

    def test_a_skipped_report_is_ignored_by_the_adapter(self):
        metadata = json.loads(json.dumps(CANONICALIZE_METADATA))
        metadata["source_fidelity"] = {"skipped": "no photo"}
        artifact = stage_artifact_from_metadata(Stage.S2_CANONICALIZATION, metadata, "canon-1")
        self.assertNotIn("source_fidelity_silhouette_iou", artifact.metrics)
        self.assertNotIn("source_fidelity_face_ssim", artifact.metrics)
        codes = _codes(Stage.S2_CANONICALIZATION, metadata, "not-evaluable")
        self.assertIn("s2.source_fidelity_silhouette", codes)
        self.assertIn("s2.source_fidelity_face", codes)


class MeasuredS3MetricsTests(unittest.TestCase):
    def test_joint_containment_flips_joints_inside_mesh_to_a_pass(self):
        metadata = _with_containment()
        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")

        self.assertEqual(artifact.metrics["joints_outside_mesh_count"], 0)
        self.assertNotIn(
            "s3.joints_inside_mesh",
            _codes(Stage.S3_SKELETON_FIT, metadata, "not-evaluable"),
        )
        self.assertEqual(_codes(Stage.S3_SKELETON_FIT, metadata, "error"), [])

    def test_a_joint_outside_the_mesh_is_a_real_s3_violation(self):
        metadata = _with_containment(
            joints_outside_mesh_count=2, inside_joints=19, joints_inside_mesh_fraction=0.905
        )

        self.assertEqual(
            _codes(Stage.S3_SKELETON_FIT, metadata, "error"), ["s3.joints_inside_mesh"]
        )

    def test_a_rig_predating_the_measurement_stays_not_evaluable(self):
        artifact = stage_artifact_from_metadata(
            Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA, "rig-1"
        )

        self.assertNotIn("joints_outside_mesh_count", artifact.metrics)
        self.assertIn(
            "s3.joints_inside_mesh",
            _codes(Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA, "not-evaluable"),
        )

    def test_containment_is_read_even_without_a_template_conformance_report(self):
        # A chain rig produces no conformance report, and the adapter returns
        # early on it -- containment must be read before that return.
        metadata = _with_containment()
        metadata["rig_quality"]["template_conformance"] = None

        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")

        self.assertEqual(artifact.metrics["joints_outside_mesh_count"], 0)
        self.assertNotIn("bone_names", artifact.metrics)

    def test_an_unmeasurable_containment_report_populates_nothing(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["rig_quality"]["joint_containment"] = {
            "schema_version": 1, "measurable": False, "reason": "no skinned triangles",
        }
        metadata["rig_quality"]["joints_outside_mesh_count"] = None

        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")

        self.assertNotIn("joints_outside_mesh_count", artifact.metrics)


class MeasuredS3SymmetryTests(unittest.TestCase):
    """`s3.bilateral_symmetry_p95` reads the mirror error `rig_quality`
    mirrors to its top level -- the geometric companion to the naming-only
    `s3.left_right_symmetry` check."""

    def test_a_clean_mirror_error_flips_the_check_to_a_pass(self):
        metadata = _with_symmetry()
        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")

        self.assertEqual(artifact.metrics["symmetry_mirror_error_p95"], 0.0)
        self.assertNotIn(
            "s3.bilateral_symmetry_p95",
            _codes(Stage.S3_SKELETON_FIT, metadata, "not-evaluable"),
        )
        self.assertEqual(_codes(Stage.S3_SKELETON_FIT, metadata, "warning"), [])

    def test_a_high_mirror_error_is_a_real_warning_not_an_error(self):
        metadata = _with_symmetry(mirror_error_p95=0.12, pairs_outside_band_count=1)

        self.assertEqual(
            _codes(Stage.S3_SKELETON_FIT, metadata, "warning"),
            ["s3.bilateral_symmetry_p95"],
        )
        # An uncalibrated placeholder threshold must never block a run.
        self.assertEqual(_codes(Stage.S3_SKELETON_FIT, metadata, "error"), [])

    def test_a_rig_predating_the_measurement_stays_not_evaluable(self):
        artifact = stage_artifact_from_metadata(
            Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA, "rig-1"
        )

        self.assertNotIn("symmetry_mirror_error_p95", artifact.metrics)
        self.assertIn(
            "s3.bilateral_symmetry_p95",
            _codes(Stage.S3_SKELETON_FIT, AUTO_RIG_METADATA, "not-evaluable"),
        )

    def test_an_unmeasurable_symmetry_report_populates_nothing(self):
        metadata = json.loads(json.dumps(AUTO_RIG_METADATA))
        metadata["rig_quality"]["bilateral_symmetry"] = {
            "schema_version": 1, "measurable": False,
            "caveats": ["facing not confident"],
        }
        metadata["rig_quality"]["symmetry_mirror_error_p95"] = None
        metadata["rig_quality"]["symmetry_pairs_outside_band_count"] = None

        artifact = stage_artifact_from_metadata(Stage.S3_SKELETON_FIT, metadata, "rig-1")

        self.assertNotIn("symmetry_mirror_error_p95", artifact.metrics)
        self.assertNotIn("symmetry_pairs_outside_band_count", artifact.metrics)
        self.assertIn(
            "s3.bilateral_symmetry_p95",
            _codes(Stage.S3_SKELETON_FIT, metadata, "not-evaluable"),
        )


class MeasuredS5MotionMetricsTests(unittest.TestCase):
    def test_foot_contact_error_is_the_contact_filtered_slide_not_the_raw_one(self):
        metadata = _with_motion_review()
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        # The raw 0.831 over-counts flight and stride; only the
        # contact-filtered 0.3055 is evidence about foot contact.
        self.assertAlmostEqual(artifact.metrics["foot_contact_error"], 0.305542)
        self.assertNotEqual(artifact.metrics["foot_contact_error"], 0.831331)
        self.assertNotIn(
            "s5.foot_contact_band", _codes(Stage.S5_ANIMATION, metadata, "not-evaluable")
        )

    def test_a_reference_walk_cycle_stride_does_not_trip_the_band(self):
        # `Walk_Armature` on the store's own armature -- the family's own
        # authored clip, and therefore by definition correct. A band it fails
        # would be a broken band, which is why the old 0.02 was replaced.
        metadata = _with_motion_review()

        self.assertEqual(_codes(Stage.S5_ANIMATION, metadata, "warning"), [])
        self.assertGreater(MOTION_REVIEW["foot_slide_max_in_contact"], 0.02)

    def test_gross_slip_past_every_reference_clip_is_an_advisory_finding(self):
        metadata = _with_motion_review(foot_slide_max_in_contact=1.5)
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")
        finding = next(
            f for f in evaluate_stage(Stage.S5_ANIMATION, artifact)["findings"]
            if f["code"] == "s5.foot_contact_band"
        )

        # Advisory, never blocking: the measure cannot separate stride from
        # slip, so it reports and lets the run continue.
        self.assertEqual(finding["severity"], "warning")
        self.assertEqual(finding["actual"], 1.5)

    def test_no_qualifying_contact_episode_is_not_evaluable_not_a_zero_pass(self):
        # An airborne clip (a roll) measures 0.0 contact slide because no
        # episode qualified. Deriving foot_contact_error from that would be a
        # fabricated pass on a check nothing measured.
        metadata = _with_motion_review(
            foot_slide_max_in_contact=0.0,
            foot_slide_mean_in_contact=0.0,
            contact_stance_episodes=0,
            stance_frames_fraction=0.044444,
        )
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        self.assertNotIn("foot_contact_error", artifact.metrics)
        self.assertIn(
            "s5.foot_contact_band", _codes(Stage.S5_ANIMATION, metadata, "not-evaluable")
        )

    def test_a_motion_review_without_the_episode_count_is_not_evaluable(self):
        # Clips measured before `contact_stance_episodes` existed cannot
        # disambiguate their own zero, so they stay not-evaluable.
        metadata = _with_motion_review()
        metadata["deformation_review"]["motion_review"].pop("contact_stance_episodes")
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        self.assertNotIn("foot_contact_error", artifact.metrics)

    def test_a_synthetic_rest_sway_is_not_evidence_about_any_clip(self):
        metadata = _with_motion_review(synthetic=True, source_action="UniRigRestSway")
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        for key in ("foot_contact_error", "static_tail_fraction", "joint_motion_coverage"):
            self.assertNotIn(key, artifact.metrics)

    def test_an_unmeasurable_motion_review_populates_nothing(self):
        metadata = _with_motion_review()
        metadata["deformation_review"]["motion_review"] = {
            "schema_version": 1, "measurable": False, "reason": "fewer than two sampled frames",
        }
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        self.assertNotIn("foot_contact_error", artifact.metrics)

    def test_static_tail_and_joint_coverage_are_exposed_on_the_stage(self):
        metadata = _with_motion_review()
        artifact = stage_artifact_from_metadata(Stage.S5_ANIMATION, metadata, "clip-1")

        self.assertAlmostEqual(artifact.metrics["static_tail_fraction"], 0.022727)
        self.assertAlmostEqual(artifact.metrics["joint_motion_coverage"], 0.952381)
        # No S5 contract check requires either one today, so exposing them
        # must not invent a verdict.
        self.assertEqual(_codes(Stage.S5_ANIMATION, metadata, "error"), [])

    def test_a_clip_predating_the_measurement_stays_not_evaluable(self):
        artifact = stage_artifact_from_metadata(
            Stage.S5_ANIMATION, APPLY_CLIP_METADATA, "clip-1"
        )

        self.assertNotIn("foot_contact_error", artifact.metrics)
        self.assertIn(
            "s5.foot_contact_band",
            _codes(Stage.S5_ANIMATION, APPLY_CLIP_METADATA, "not-evaluable"),
        )

    def test_every_newly_populated_metric_declares_its_source(self):
        for stage, metadata in (
            (Stage.S2_CANONICALIZATION, _with_geometry_contract()),
            (Stage.S3_SKELETON_FIT, _with_containment()),
            (Stage.S5_ANIMATION, _with_motion_review()),
        ):
            artifact = stage_artifact_from_metadata(stage, metadata, "a-1")
            for key in artifact.metrics:
                if key == "triangle_budget_band":
                    continue
                with self.subTest(stage=stage, metric=key):
                    self.assertIn(key, METRIC_SOURCES)


class FootContactBandCalibrationTests(unittest.TestCase):
    """The number itself. Measured by `scripts/calibrate_foot_contact.py`
    over all 87 clips of the biped store on its own template armature."""

    # family -> (worst contact-filtered slide observed, clip count)
    REFERENCE_MAXIMA = {
        "locomotion": (0.5984, 8),
        "attack": (1.3293, 28),
        "idle": (0.1897, 26),
        "acrobatic": (1.1037, 15),
        "other": (0.2674, 10),
    }

    def test_the_band_clears_every_authored_clip_in_every_family(self):
        self.assertEqual(sum(count for _, count in self.REFERENCE_MAXIMA.values()), 87)
        for family, (worst, _) in self.REFERENCE_MAXIMA.items():
            with self.subTest(family=family):
                self.assertGreater(FOOT_CONTACT_BAND_THRESHOLD, worst)

    def test_the_band_is_not_so_loose_that_nothing_could_ever_trip_it(self):
        # Room above the store's worst clip, but bounded: a clip sliding more
        # than ~1.5x the worst authored clip is still caught.
        worst = max(value for value, _ in self.REFERENCE_MAXIMA.values())
        self.assertLess(FOOT_CONTACT_BAND_THRESHOLD, worst * 1.5)

    def test_the_check_is_advisory_because_the_measure_conflates_stride(self):
        check = next(
            c for c in CONTRACTS_BY_STAGE[Stage.S5_ANIMATION].checks
            if c.id == "s5.foot_contact_band"
        )
        self.assertEqual(check.severity, "warning")
        self.assertEqual(check.requires, ("foot_contact_error",))


class BilateralSymmetryCalibrationTests(unittest.TestCase):
    """The number itself. Measured by `scripts/calibrate_symmetry.py` over
    the GPU box's full artifact corpus (75 auto-rig/rig-adjust GLBs, 59
    measurable) on 2026-09-04 -- see the calibration comment above
    `SYMMETRY_MIRROR_ERROR_P95_THRESHOLD`."""

    # artifact -> measured mirror_error_p95
    ACCEPTED_REFERENCES = {
        "edd0bb20 (punk v2)": 0.0064,
        "6cdce431 (mary)": 0.0043,
    }
    LOWEST_KNOWN_BAD = 0.0505  # 36bfef83, 2 pairs outside band

    def test_every_accepted_reference_clears_the_threshold(self):
        for artifact, p95 in self.ACCEPTED_REFERENCES.items():
            with self.subTest(artifact=artifact):
                self.assertLess(p95, SYMMETRY_MIRROR_ERROR_P95_THRESHOLD)

    def test_the_lowest_known_bad_rig_does_not_clear_the_threshold(self):
        self.assertGreater(self.LOWEST_KNOWN_BAD, SYMMETRY_MIRROR_ERROR_P95_THRESHOLD)

    def test_the_threshold_sits_in_the_measured_gap(self):
        # Highest good rig observed was 0.0235; the gap to the lowest known-bad
        # rig (0.0505) is where the threshold was chosen.
        highest_good = 0.0235
        self.assertGreater(SYMMETRY_MIRROR_ERROR_P95_THRESHOLD, highest_good)
        self.assertLess(SYMMETRY_MIRROR_ERROR_P95_THRESHOLD, self.LOWEST_KNOWN_BAD)

    def test_the_check_is_advisory_despite_being_calibrated(self):
        check = next(
            c for c in CONTRACTS_BY_STAGE[Stage.S3_SKELETON_FIT].checks
            if c.id == "s3.bilateral_symmetry_p95"
        )
        self.assertEqual(check.severity, "warning")
        self.assertEqual(check.requires, ("symmetry_mirror_error_p95",))
