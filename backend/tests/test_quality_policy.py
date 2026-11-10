import unittest

from backend.app.quality_policy import (
    assess_mesh_promotion,
    evaluate_for_animation,
    evaluate_for_rigging,
    evaluate_deformation_review,
    evaluate_rig_certificate,
    is_rig_producing_stage,
    rig_candidate_is_promotable,
)


class QualityPolicyTests(unittest.TestCase):
    def test_all_rig_producing_stage_names_share_the_promotion_gate(self):
        for stage in (
            "rig", "repair-weights", "rig-transfer", "target-proxy-rig", "retarget-motion",
        ):
            self.assertTrue(is_rig_producing_stage(stage))
        self.assertFalse(is_rig_producing_stage("retopologize"))

    def test_rig_candidate_promotion_requires_ready_topology(self):
        decisions = {
            "rig": {"status": "blocked"},
            "rig-certificate": {"status": "ready"},
            "animation-review": {"status": "ready"},
            "deformation-review": {"status": "ready"},
        }

        self.assertFalse(rig_candidate_is_promotable(decisions))
        decisions["rig"]["status"] = "ready"
        self.assertTrue(rig_candidate_is_promotable(decisions))
        del decisions["deformation-review"]
        self.assertFalse(rig_candidate_is_promotable(decisions))

    def test_deformation_review_separates_credible_and_catastrophic_mary_rigs(self):
        credible = evaluate_deformation_review({
            "complete_action_range": True,
            "frames": [{"frame": frame} for frame in range(5)],
            "maximum_p99_edge_stretch": 5.3757,
        })
        catastrophic = evaluate_deformation_review({
            "complete_action_range": True,
            "frames": [{"frame": frame} for frame in range(5)],
            "maximum_p99_edge_stretch": 12.033,
        })

        self.assertEqual(credible["status"], "needs_review")
        self.assertEqual(
            credible["findings"][0]["code"],
            "deformation.visual_review_required",
        )
        self.assertEqual(catastrophic["status"], "blocked")
        self.assertEqual(
            catastrophic["findings"][0]["code"],
            "deformation.edge_stretch",
        )

    def test_deformation_review_accepts_complete_short_action_evidence(self):
        decision = evaluate_deformation_review({
            "complete_action_range": True,
            "frames": [{"frame": frame} for frame in range(4)],
            "maximum_p99_edge_stretch": 2.0,
        })

        self.assertEqual(decision["status"], "needs_review")
        self.assertNotEqual(
            decision["findings"][0]["code"],
            "deformation.evidence_missing",
        )

    def test_deformation_review_rejects_legacy_sparse_evidence(self):
        decision = evaluate_deformation_review({
            "frames": [{"frame": frame} for frame in range(5)],
            "maximum_p99_edge_stretch": 2.0,
        })

        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(
            decision["findings"][0]["code"],
            "deformation.evidence_missing",
        )

    def _deformation_report(self, **motion):
        return {
            "complete_action_range": True,
            "frames": [{"frame": frame} for frame in range(5)],
            "maximum_p99_edge_stretch": 2.0,
            "motion_review": {
                "measurable": True,
                "synthetic": False,
                "joint_motion_coverage": 1.0,
                "static_tail_fraction": 0.0,
                "foot_slide_max": 0.0,
                **motion,
            },
        }

    def test_a_healthy_clip_adds_no_motion_findings(self):
        decision = evaluate_deformation_review(self._deformation_report())

        self.assertEqual(
            [item["code"] for item in decision["findings"]],
            ["deformation.visual_review_required"],
        )

    def test_a_clip_that_dies_partway_through_is_flagged(self):
        # The 2026-08-31 shape: edge stretch stays clean, but three quarters
        # of the clip carries no motion at all.
        decision = evaluate_deformation_review(
            self._deformation_report(static_tail_fraction=0.76)
        )

        codes = [item["code"] for item in decision["findings"]]
        self.assertIn("motion.static_tail", codes)
        self.assertEqual(decision["status"], "needs_review")
        self.assertIn("reattach-animation", decision["recommended_actions"])

    def test_the_static_tail_seen_on_healthy_donor_clips_stays_clean(self):
        # The donor BVH, donor source and curated source all measured <= 0.053.
        decision = evaluate_deformation_review(
            self._deformation_report(static_tail_fraction=0.053)
        )

        self.assertNotIn(
            "motion.static_tail", [item["code"] for item in decision["findings"]],
        )

    def test_a_clip_reaching_only_a_minority_of_bones_is_flagged(self):
        decision = evaluate_deformation_review(
            self._deformation_report(joint_motion_coverage=0.12)
        )

        self.assertIn(
            "motion.joint_coverage", [item["code"] for item in decision["findings"]],
        )

    def test_large_foot_slide_is_recorded_but_never_gated(self):
        # The stance-foot heuristic overcounts slide on stepping clips, so no
        # threshold exists yet; the number is carried without a finding.
        decision = evaluate_deformation_review(
            self._deformation_report(foot_slide_max=0.59)
        )

        self.assertEqual(
            [item["code"] for item in decision["findings"]],
            ["deformation.visual_review_required"],
        )

    def test_a_synthetic_rest_sway_produces_no_motion_findings(self):
        decision = evaluate_deformation_review(
            self._deformation_report(
                synthetic=True, static_tail_fraction=0.9, joint_motion_coverage=0.02,
            )
        )

        self.assertEqual(
            [item["code"] for item in decision["findings"]],
            ["deformation.visual_review_required"],
        )

    def test_a_report_without_motion_metrics_is_unchanged(self):
        decision = evaluate_deformation_review({
            "complete_action_range": True,
            "frames": [{"frame": frame} for frame in range(5)],
            "maximum_p99_edge_stretch": 2.0,
        })

        self.assertEqual(
            [item["code"] for item in decision["findings"]],
            ["deformation.visual_review_required"],
        )

    def test_a_catastrophic_stretch_still_blocks_alongside_motion_findings(self):
        report = self._deformation_report(static_tail_fraction=0.9)
        report["maximum_p99_edge_stretch"] = 12.0

        decision = evaluate_deformation_review(report)

        self.assertEqual(decision["status"], "blocked")
        self.assertEqual(decision["findings"][0]["code"], "deformation.edge_stretch")
        self.assertIn(
            "motion.static_tail", [item["code"] for item in decision["findings"]],
        )

    def test_rig_certificate_rejects_mostly_unskinned_character_geometry(self):
        joint_names = [
            "pelvis", "spine", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 1000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.5,
            "unused_joint_names": [],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 100, "total_weight": 1000 / len(joint_names)}
                for name in joint_names
            ],
        }
        report["unskinned_vertices"] = report["skinned_vertices"] * 2

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        self.assertIn("rig.unskinned_geometry", [item["code"] for item in decision["findings"]])
    def test_mesh_promotion_accepts_refinement_and_rejects_fragmentation_regression(self):
        original = {
            "faces": 243925,
            "largest_component_face_ratio": 0.916122,
            "non_manifold_edges": 6026,
            "degenerate_faces": 6,
            "finite": True,
        }
        refined = {
            "faces": 48974,
            "largest_component_face_ratio": 0.996263,
            "non_manifold_edges": 496,
            "degenerate_faces": 0,
            "finite": True,
        }
        fragmented = {
            "faces": 101643,
            "largest_component_face_ratio": 0.016637,
            "non_manifold_edges": 516,
            "degenerate_faces": 4,
            "finite": True,
        }

        accepted = assess_mesh_promotion(original, refined)
        rejected = assess_mesh_promotion(refined, fragmented)

        self.assertTrue(accepted["promote"])
        self.assertFalse(rejected["promote"])
        self.assertIn("primary component coverage regressed", rejected["reasons"])

    def test_fragmented_non_manifold_mary_mesh_is_blocked_before_rigging(self):
        report = {
            "faces": 243925,
            "components": 178,
            "largest_component_face_ratio": 0.916122,
            "boundary_edges": 5,
            "non_manifold_edges": 6026,
            "degenerate_faces": 6,
            "watertight": False,
            "finite": True,
        }

        decision = evaluate_for_rigging(report)

        self.assertEqual(decision["policy_version"], 1)
        self.assertEqual(decision["status"], "blocked")
        self.assertIn("mesh.fragmented", [item["code"] for item in decision["findings"]])
        self.assertIn("mesh.non_manifold", [item["code"] for item in decision["findings"]])
        self.assertIn("refine-reconstruct", decision["recommended_actions"])

    def test_structural_rig_with_weak_blending_requires_review(self):
        report = {
            "skins": 1,
            "joints": 28,
            "skinned_vertices": 245819,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "maximum_influences": 2,
            "rigid_vertex_ratio": 0.914,
            "unused_joint_names": ["root", "pelvis", "head_leaf"],
            "animations": [
                {
                    "name": "Sword_Attack",
                    "duration_seconds": 1.9167,
                    "animated_joint_coverage": 1.0,
                    "invalid_targets": 0,
                }
            ],
        }

        decision = evaluate_for_animation(report, required_animation="Sword Attack")

        self.assertEqual(decision["status"], "needs_review")
        self.assertIn("rig.rigid_weights", [item["code"] for item in decision["findings"]])
        self.assertIn("rig.core_joint_unused", [item["code"] for item in decision["findings"]])
        self.assertIn("inspect-weight-paint", decision["recommended_actions"])

    def test_mary_rig_is_blocked_by_unused_pelvis_and_head_weight_concentration(self):
        report = {
            "skins": 1,
            "joints": 28,
            "joint_names": [
                "root", "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head",
                "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
                "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
                "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
            ],
            "skinned_vertices": 245819,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.914,
            "unused_joint_names": ["root", "pelvis", "head_leaf"],
            "joint_weight_coverage": [
                {"joint": "pelvis", "vertices": 0, "total_weight": 0.0},
                {"joint": "spine_01", "vertices": 7572, "total_weight": 5524.5},
                {"joint": "head", "vertices": 157361, "total_weight": 153890.0},
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["target_stage"], "rig-certificate")
        self.assertEqual(decision["status"], "blocked")
        self.assertIn("rig.required_joint_unused", [item["code"] for item in decision["findings"]])
        self.assertIn("rig.weight_concentration", [item["code"] for item in decision["findings"]])
        self.assertIn("repair-weights", decision["recommended_actions"])

    def test_rig_is_blocked_when_bind_pose_misses_weighted_limb_landmarks(self):
        joint_names = [
            "pelvis", "spine", "head",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unskinned_vertices": 0,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.5,
            "unused_joint_names": [],
            "joint_weight_coverage": [
                {"joint_node": index, "joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for index, name in enumerate(joint_names)
            ],
            "bind_pose_landmark_errors": [
            {
                "joint_node": joint_names.index("hand_r"),
                "joint": "hand_r",
                "normalized_error": 0.31,
            }
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        self.assertIn(
            "rig.bind_pose_mismatch",
            [item["code"] for item in decision["findings"]],
        )
        self.assertIn("inspect-weight-paint", decision["recommended_actions"])
        self.assertNotIn("fit-bind-pose", decision["recommended_actions"])

    def test_bind_pose_gate_ignores_non_anatomical_attachment_joints(self):
        joint_names = [
            "pelvis", "spine", "head",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unskinned_vertices": 0,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.5,
            "unused_joint_names": [],
            "joint_weight_coverage": [
                {"joint_node": index, "joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for index, name in enumerate(joint_names)
            ] + [
                {"joint_node": 99, "joint": "hand_r", "vertices": 8, "total_weight": 8.0},
            ],
            "bind_pose_landmark_errors": [
                {"joint_node": 99, "joint": "hand_r", "normalized_error": 0.9},
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")

    def test_rig_certificate_uses_current_bind_pose_policy_version(self):
        joint_names = [
            "pelvis", "spine", "head",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unskinned_vertices": 0,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.5,
            "unused_joint_names": [],
            "joint_weight_coverage": [
                {"joint_node": index, "joint": name, "vertices": 700, "total_weight": 500.0}
                for index, name in enumerate(joint_names)
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["policy_version"], 2)

    def test_balanced_humanoid_rig_receives_ready_certificate(self):
        joint_names = [
            "head_leaf", "pelvis", "spine", "chest", "neck", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.45,
            "unused_joint_names": ["head_leaf"],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for name in joint_names
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["resolved_roles"]["head"], "head")

    def _balanced_humanoid_report(self, **overrides):
        joint_names = [
            "head_leaf", "pelvis", "spine", "chest", "neck", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.45,
            "unused_joint_names": ["head_leaf"],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for name in joint_names
            ],
        }
        report.update(overrides)
        return report

    def test_rig_certificate_blocks_on_template_conformance_error_finding(self):
        # N3: skeleton_template.conformance_report-shaped findings, threaded
        # through unchanged from unirig_bind_worker's bind-time computation
        # (backend.app.rig_quality._extract_template_conformance reads them
        # back from the exported GLB's node extras). A single error-severity
        # finding must block the certificate the same way any other
        # rig-certificate error does.
        report = self._balanced_humanoid_report(template_conformance=[{
            "code": "missing_required_bone",
            "severity": "error",
            "message": "required bone 'hand_l' is missing from the candidate rig",
            "metric": "required_bone_present",
            "actual": False,
            "expected": "'hand_l' present",
        }])

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        self.assertIn(
            {
                "code": "missing_required_bone",
                "severity": "error",
                "message": "required bone 'hand_l' is missing from the candidate rig",
                "metric": "required_bone_present",
                "actual": False,
                "expected": "'hand_l' present",
            },
            decision["findings"],
        )
        self.assertIn("rerig", decision["recommended_actions"])

    def test_rig_certificate_ignores_absent_template_conformance_key(self):
        # `template_conformance` is `None` for a rig with no N3 conformance
        # data at all (older GLB, or non-humanoid) -- must not be treated
        # as a blocking finding.
        report = self._balanced_humanoid_report()
        self.assertNotIn("template_conformance", report)

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")

    def test_rig_certificate_treats_empty_template_conformance_list_as_conformant(self):
        report = self._balanced_humanoid_report(template_conformance=[])

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])

    def test_rig_certificate_blocks_left_right_inverted_handedness(self):
        joint_names = [
            "head_leaf", "pelvis", "spine", "chest", "neck", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.45,
            "unused_joint_names": ["head_leaf"],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for name in joint_names
            ],
            "handedness": {
                "facing_forward": [0.0, -1.0, 0.0],
                "facing_source": "foot-toe-direction",
                "facing_warning": None,
                "mismatches": [
                    {
                        "joint": "upperarm_l", "role": "upper_arm_l",
                        "expected_side": "left", "actual_side": "right",
                        "position": [-0.09, 1.3, -0.1],
                    },
                    {
                        "joint": "thigh_l", "role": "thigh_l",
                        "expected_side": "left", "actual_side": "right",
                        "position": [-0.13, 0.57, -0.02],
                    },
                ],
            },
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        codes = [finding["code"] for finding in decision["findings"]]
        self.assertIn("rig.left_right_inverted", codes)
        finding = next(f for f in decision["findings"] if f["code"] == "rig.left_right_inverted")
        self.assertEqual(finding["severity"], "error")
        self.assertEqual(finding["actual"]["facing_source"], "foot-toe-direction")
        self.assertEqual(len(finding["actual"]["mismatches"]), 2)

    def test_rig_certificate_ready_when_handedness_has_no_mismatches(self):
        joint_names = [
            "head_leaf", "pelvis", "spine", "chest", "neck", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.45,
            "unused_joint_names": ["head_leaf"],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 700, "total_weight": 10000 / len(joint_names)}
                for name in joint_names
            ],
            "handedness": {
                "facing_forward": [0.0, -1.0, 0.0],
                "facing_source": "foot-toe-direction",
                "facing_warning": None,
                "mismatches": [],
            },
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")
        self.assertNotIn(
            "rig.left_right_inverted",
            [finding["code"] for finding in decision["findings"]],
        )

    def test_rig_certificate_blocks_unweighted_finger_segments(self):
        joint_names = [
            "pelvis", "spine", "head",
            "upperarm_l", "lowerarm_l", "hand_l", "index_01_l",
            "upperarm_r", "lowerarm_r", "hand_r", "index_01_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.45,
            "unused_joint_names": ["index_01_l", "index_01_r"],
            "joint_weight_coverage": [
                {"joint": name, "vertices": 700, "total_weight": 500.0}
                for name in joint_names
            ],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        self.assertIn(
            "rig.finger_joint_unused",
            [finding["code"] for finding in decision["findings"]],
        )

    def test_missing_requested_animation_is_blocked(self):
        report = {
            "skins": 1,
            "joints": 24,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "maximum_influences": 4,
            "rigid_vertex_ratio": 0.5,
            "unused_joint_names": [],
            "animations": [],
        }

        decision = evaluate_for_animation(report, required_animation="Sword Attack")

        self.assertEqual(decision["status"], "blocked")
        self.assertIn("animation.missing", [item["code"] for item in decision["findings"]])

    def test_authored_attack_coverage_uses_humanoid_roles_not_finger_joint_count(self):
        joint_names = [
            "pelvis", "spine_03", "head",
            "upperarm_l", "lowerarm_l", "hand_l",
            "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
            *[f"finger_{index}" for index in range(51)],
        ]
        report = {
            "skins": 1,
            "joints": len(joint_names),
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "rigid_vertex_ratio": 0.4,
            "unused_joint_names": [],
            "animations": [{
                "name": "Sword_Attack",
                "duration_seconds": 0.875,
                "animated_joint_coverage": 5 / 66,
                "animated_joint_names": [
                    "pelvis", "spine_03", "upperarm_r", "lowerarm_r", "hand_r",
                ],
                "invalid_targets": 0,
            }],
        }

        decision = evaluate_for_animation(report, required_animation="Sword Attack")

        self.assertEqual(decision["status"], "ready")


def _chain_joint_names(spine_count: int) -> list[str]:
    return [f"spine_{index:02d}" for index in range(spine_count)] + ["head"]


class ChainRigCertificateTests(unittest.TestCase):
    """Coverage for the non-humanoid "chain" rig certificate variant --
    a serpent or other limbless creature (see backend.app.skeleton_naming's
    chain profile)."""

    def _base_chain_report(self, spine_count: int = 10) -> dict:
        joint_names = _chain_joint_names(spine_count)
        return {
            "profile": "chain",
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unskinned_vertices": 0,
            "unweighted_vertices": 0,
            "chain_topology": {"is_single_chain": True, "branch_joints": []},
            "chain_axis_errors": [
                {"joint": name, "axis_deviation_ratio": 0.1} for name in joint_names
            ],
        }

    def test_ready_certificate_on_a_well_formed_synthetic_chain(self):
        report = self._base_chain_report(spine_count=10)

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])
        self.assertEqual(decision["profile"], "chain")
        self.assertEqual(decision["resolved_roles"]["head"], "head")
        self.assertEqual(decision["resolved_roles"]["spine"], "spine_00")

    def test_profile_is_auto_detected_from_joint_names_without_an_explicit_field(self):
        report = self._base_chain_report(spine_count=10)
        del report["profile"]

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["profile"], "chain")
        self.assertEqual(decision["status"], "ready")

    def test_too_short_on_a_five_joint_chain(self):
        # 4 spine segments + head = 5 joints total -- well under the >= 8
        # spine-bone READY threshold.
        report = self._base_chain_report(spine_count=4)
        report["chain_axis_errors"] = [
            {"joint": name, "axis_deviation_ratio": 0.1} for name in report["joint_names"]
        ]

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        codes = [finding["code"] for finding in decision["findings"]]
        self.assertIn("chain.too_short", codes)

    def test_branching_on_a_y_shaped_skeleton_via_topology(self):
        report = self._base_chain_report(spine_count=10)
        report["chain_topology"] = {
            "is_single_chain": False,
            "branch_joints": ["spine_05"],
        }

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        finding = next(f for f in decision["findings"] if f["code"] == "chain.branching")
        self.assertIn("spine_05", finding["actual"])

    def test_branching_on_bone_names_outside_the_chain_vocabulary(self):
        report = self._base_chain_report(spine_count=10)
        report["joint_names"].append("control_bone")

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        finding = next(f for f in decision["findings"] if f["code"] == "chain.branching")
        self.assertIn("control_bone", finding["actual"])

    def test_off_axis_bones_are_flagged(self):
        report = self._base_chain_report(spine_count=10)
        report["chain_axis_errors"][3] = {"joint": "spine_03", "axis_deviation_ratio": 0.6}

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        finding = next(f for f in decision["findings"] if f["code"] == "chain.off_axis")
        self.assertEqual(len(finding["actual"]), 1)
        self.assertEqual(finding["actual"][0]["joint"], "spine_03")

    def test_insufficiently_weighted_chain_mesh_is_blocked(self):
        report = self._base_chain_report(spine_count=10)
        report["skinned_vertices"] = 9000
        report["unweighted_vertices"] = 1000  # 10% unweighted, > 5% allowance

        decision = evaluate_rig_certificate(report)

        self.assertEqual(decision["status"], "blocked")
        codes = [finding["code"] for finding in decision["findings"]]
        self.assertIn("rig.invalid_weights", codes)

    def test_chain_certificate_skips_humanoid_role_and_handedness_findings(self):
        report = self._base_chain_report(spine_count=10)
        # A humanoid-only field that would otherwise trip
        # rig.left_right_inverted -- must be ignored entirely for a chain.
        report["handedness"] = {
            "facing_forward": [0.0, -1.0, 0.0],
            "facing_source": "foot-toe-direction",
            "facing_warning": None,
            "mismatches": [{"joint": "x", "role": "upper_arm_l",
                             "expected_side": "left", "actual_side": "right", "position": [0, 0, 0]}],
        }

        decision = evaluate_rig_certificate(report)

        codes = [finding["code"] for finding in decision["findings"]]
        self.assertNotIn("rig.left_right_inverted", codes)
        self.assertNotIn("rig.humanoid_roles_missing", codes)


class ChainAnimationReviewTests(unittest.TestCase):
    """Coverage for the chain-rig variant of the animation-review gate
    (evaluate_for_animation) -- a serpent must not be held to the humanoid
    15-joint floor or humanoid attack-role clip coverage."""

    def _base_chain_animation_report(self, spine_count: int = 10) -> dict:
        joint_names = _chain_joint_names(spine_count)
        return {
            "profile": "chain",
            "skins": 1,
            "joint_names": joint_names,
            "skinned_vertices": 10000,
            "unweighted_vertices": 0,
            "non_normalized_vertices": 0,
            "invalid_joint_references": 0,
            "rigid_vertex_ratio": 0.3,
            "unused_joint_names": [],
            "animations": [{
                "name": "Slither",
                "duration_seconds": 1.5,
                "animated_joint_coverage": 0.9,
                "animated_joint_names": joint_names,
                "invalid_targets": 0,
            }],
        }

    def test_ready_on_a_well_formed_chain_rig_with_a_slither_clip(self):
        report = self._base_chain_animation_report(spine_count=10)

        decision = evaluate_for_animation(report, required_animation="Slither")

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["findings"], [])

    def test_profile_auto_detected_without_explicit_field(self):
        report = self._base_chain_animation_report(spine_count=10)
        del report["profile"]

        decision = evaluate_for_animation(report, required_animation="Slither")

        self.assertEqual(decision["status"], "ready")

    def test_missing_slither_clip_is_still_flagged(self):
        report = self._base_chain_animation_report(spine_count=10)
        report["animations"] = [{"name": "Idle", "duration_seconds": 1.0,
                                  "animated_joint_coverage": 0.9, "animated_joint_names": [], "invalid_targets": 0}]

        decision = evaluate_for_animation(report, required_animation="Slither")

        self.assertEqual(decision["status"], "blocked")
        codes = [finding["code"] for finding in decision["findings"]]
        self.assertIn("animation.missing", codes)

    def test_chain_specific_too_few_joints_wording_not_humanoid_fifteen_joint_floor(self):
        # 4 spine segments + head -- fewer than the chain minimum, but this
        # must never be measured against the humanoid ">= 15 joints" floor
        # (previously firing before this gate was made profile-aware).
        report = self._base_chain_animation_report(spine_count=4)

        decision = evaluate_for_animation(report, required_animation="Slither")

        self.assertEqual(decision["status"], "blocked")
        finding = next(f for f in decision["findings"] if f["code"] == "rig.missing")
        self.assertNotIn("humanoid", finding["message"])
        self.assertIn("chain", finding["message"])

    def test_chain_rig_skips_humanoid_core_joint_unused_and_attack_role_coverage(self):
        report = self._base_chain_animation_report(spine_count=10)
        # "spine_00" contains the humanoid "spine" token; an unused chain
        # spine bone must not trip the humanoid-worded core-joint-unused
        # warning, and clip coverage must never be measured against
        # humanoid attack roles (pelvis/upper_arm_r/...) a chain has none of.
        report["unused_joint_names"] = ["spine_00"]

        decision = evaluate_for_animation(report, required_animation="Slither")

        codes = [finding["code"] for finding in decision["findings"]]
        self.assertNotIn("rig.core_joint_unused", codes)
        self.assertEqual(decision["status"], "ready")


# --- mrmmn: unweighted OPTIONAL-tier finger/toe joints are advisory ---
#
# The bind worker grows the biped template's 45 optional bones on every
# humanoid rig, because a clip's finger channels need them to exist. A
# stylized mesh will leave some of them with no vertices, and blocking
# approval on that would make such a character permanently unapprovable --
# S3 already treats the optional tier as optional. A finger bone that is NOT
# an optional template bone still hard-fails.


def _humanoid_rig_report(unused_joint_names):
    """The smallest rig_quality report that reaches the finger-weight rule
    with an otherwise clean humanoid certificate."""
    joint_names = [
        "root", "pelvis", "spine_1", "spine_2", "spine_3", "neck", "head",
        "clavicle_l", "upper_arm_l", "lower_arm_l", "hand_l",
        "clavicle_r", "upper_arm_r", "lower_arm_r", "hand_r",
        "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
    ]
    return {
        "skins": 1,
        "joints": len(joint_names) + len(unused_joint_names),
        "joint_names": joint_names + list(unused_joint_names),
        "skinned_vertices": 10000,
        "unskinned_vertices": 0,
        "unweighted_vertices": 0,
        "non_normalized_vertices": 0,
        "invalid_joint_references": 0,
        "max_influences": 4,
        "unused_joint_names": list(unused_joint_names),
        "rigid_vertex_ratio": 0.1,
        "profile": "humanoid",
        "animations": [],
    }


class OptionalTierFingerWeightTests(unittest.TestCase):
    def _codes(self, decision):
        return {finding["code"]: finding["severity"] for finding in decision["findings"]}

    def test_unweighted_optional_finger_joints_do_not_block_approval(self):
        decision = evaluate_rig_certificate(
            _humanoid_rig_report(["thumb_3_l", "thumb_tip_l", "pinky_tip_r"])
        )

        codes = self._codes(decision)
        self.assertNotIn("rig.finger_joint_unused", codes)
        self.assertEqual(codes.get("rig.optional_finger_joint_unused"), "warning")
        # "needs_review", not "blocked": the Skeleton card's Approve button
        # is gated on "blocked" alone.
        self.assertNotEqual(decision["status"], "blocked")

    def test_the_advisory_finding_names_the_bones_it_found(self):
        decision = evaluate_rig_certificate(
            _humanoid_rig_report(["thumb_3_l", "thumb_tip_l"])
        )

        finding = next(
            item for item in decision["findings"]
            if item["code"] == "rig.optional_finger_joint_unused"
        )
        self.assertEqual(finding["actual"], ["thumb_3_l", "thumb_tip_l"])

    def test_a_non_template_finger_bone_still_blocks(self):
        # The donor-named bones a proxy rig emits are not biped-template
        # optional bones, so an unweighted one is still a broken skin.
        decision = evaluate_rig_certificate(
            _humanoid_rig_report(["index_01_l", "middle_02_r"])
        )

        codes = self._codes(decision)
        self.assertEqual(codes.get("rig.finger_joint_unused"), "error")
        self.assertNotIn("rig.optional_finger_joint_unused", codes)
        self.assertEqual(decision["status"], "blocked")

    def test_a_mixed_rig_reports_both_tiers_separately(self):
        decision = evaluate_rig_certificate(
            _humanoid_rig_report(["thumb_tip_l", "index_01_r"])
        )

        by_code = {finding["code"]: finding for finding in decision["findings"]}
        self.assertEqual(by_code["rig.finger_joint_unused"]["actual"], ["index_01_r"])
        self.assertEqual(by_code["rig.optional_finger_joint_unused"]["actual"], ["thumb_tip_l"])
        self.assertEqual(decision["status"], "blocked")

    def test_a_fully_weighted_rig_reports_no_finger_finding_at_all(self):
        decision = evaluate_rig_certificate(_humanoid_rig_report([]))

        codes = self._codes(decision)
        self.assertNotIn("rig.finger_joint_unused", codes)
        self.assertNotIn("rig.optional_finger_joint_unused", codes)


if __name__ == "__main__":
    unittest.main()
