from __future__ import annotations

import re
from typing import Any, Literal

from functools import lru_cache

from .proxy_rig import is_finger_deform_bone
from .skeleton_naming import detect_profile_from_names
from .skeleton_template import load_template


Severity = Literal["warning", "error"]

RIG_PROMOTION_GATES = (
    "rig",
    "rig-certificate",
    "animation-review",
    "deformation-review",
)

RIG_PRODUCING_STAGES = frozenset({
    "rig", "repair-weights", "rig-transfer", "target-proxy-rig", "retarget-motion",
})


def is_rig_producing_stage(stage: str) -> bool:
    return stage in RIG_PRODUCING_STAGES


def rig_candidate_is_promotable(decisions: dict[str, Any]) -> bool:
    """Require every geometry, rig, animation, and deformation gate to pass."""
    return all(
        isinstance(decisions.get(name), dict)
        and decisions[name].get("status") == "ready"
        for name in RIG_PROMOTION_GATES
    )


def assess_mesh_promotion(input_report: dict[str, Any], output_report: dict[str, Any]) -> dict[str, Any]:
    """Reject only material, objectively measured mesh regressions."""
    reasons: list[str] = []
    input_faces = max(int(input_report.get("faces", 0)), 1)
    output_faces = max(int(output_report.get("faces", 0)), 1)
    input_primary = float(input_report.get("largest_component_face_ratio", 0.0))
    output_primary = float(output_report.get("largest_component_face_ratio", 0.0))
    input_non_manifold = int(input_report.get("non_manifold_edges", 0)) / input_faces
    output_non_manifold = int(output_report.get("non_manifold_edges", 0)) / output_faces

    if bool(input_report.get("finite", False)) and not bool(output_report.get("finite", False)):
        reasons.append("finite geometry regressed")
    if output_faces < 500:
        reasons.append("output has too few faces for humanoid deformation")
    if output_primary < input_primary - 0.05:
        reasons.append("primary component coverage regressed")
    if output_non_manifold > input_non_manifold + 0.01:
        reasons.append("non-manifold edge ratio regressed")
    if int(output_report.get("degenerate_faces", 0)) > int(input_report.get("degenerate_faces", 0)) + 100:
        reasons.append("degenerate face count regressed")
    return {
        "policy_version": 1,
        "promote": not reasons,
        "reasons": reasons,
        "input": {
            "largest_component_face_ratio": input_primary,
            "non_manifold_edges_per_face": input_non_manifold,
            "faces": input_faces,
        },
        "output": {
            "largest_component_face_ratio": output_primary,
            "non_manifold_edges_per_face": output_non_manifold,
            "faces": output_faces,
        },
    }


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


def evaluate_for_rigging(report: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    faces = int(report.get("faces", 0))
    if not report.get("finite", False):
        findings.append(
            _finding(
                "mesh.non_finite",
                "error",
                "The mesh contains non-finite vertex coordinates.",
                metric="finite",
                actual=report.get("finite"),
                expected="true",
            )
        )
        actions.append("retry-generation")
    if faces < 500:
        findings.append(
            _finding(
                "mesh.too_sparse",
                "error",
                "The mesh has too few faces for humanoid deformation.",
                metric="faces",
                actual=faces,
                expected=">= 500",
            )
        )
        actions.append("retry-generation")

    largest_ratio = float(report.get("largest_component_face_ratio", 0.0))
    if largest_ratio < 0.95:
        findings.append(
            _finding(
                "mesh.fragmented",
                "error",
                "Too much geometry lies outside the primary connected body.",
                metric="largest_component_face_ratio",
                actual=largest_ratio,
                expected=">= 0.95",
            )
        )
        actions.extend(("refine-reconstruct", "manual-blender-cleanup"))
    elif int(report.get("components", 0)) > 20:
        findings.append(
            _finding(
                "mesh.many_components",
                "warning",
                "The mesh has many disconnected pieces; verify hair, clothing, and accessories.",
                metric="components",
                actual=int(report.get("components", 0)),
                expected="<= 20",
            )
        )
        actions.append("inspect-components")

    non_manifold_edges = int(report.get("non_manifold_edges", 0))
    non_manifold_ratio = non_manifold_edges / max(faces, 1)
    if non_manifold_ratio > 0.01:
        findings.append(
            _finding(
                "mesh.non_manifold",
                "error",
                "Non-manifold topology is too extensive for reliable skin deformation.",
                metric="non_manifold_edges_per_face",
                actual=non_manifold_ratio,
                expected="<= 0.01",
            )
        )
        actions.extend(("refine-reconstruct", "repair-topology"))
    elif non_manifold_edges:
        findings.append(
            _finding(
                "mesh.non_manifold",
                "warning",
                "The mesh contains non-manifold edges that should be inspected before rigging.",
                metric="non_manifold_edges",
                actual=non_manifold_edges,
                expected="0",
            )
        )
        actions.append("repair-topology")

    if not report.get("watertight", False):
        findings.append(
            _finding(
                "mesh.not_watertight",
                "warning",
                "The mesh is not watertight; intentional clothing openings require review.",
                metric="watertight",
                actual=False,
                expected="true or reviewed exception",
            )
        )
        actions.append("inspect-boundaries")
    if faces > 200_000:
        findings.append(
            _finding(
                "mesh.high_density",
                "warning",
                "The mesh is unusually dense for interactive rigging and weight painting.",
                metric="faces",
                actual=faces,
                expected="<= 200000",
            )
        )
        actions.append("retopologize")
    return _decision("rig", findings, actions)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_REQUIRED_HUMANOID_ROLES = {
    "pelvis": ("pelvis", "hips", "hip"),
    "spine": ("spine",),
    "head": ("head",),
    "left_upper_arm": ("upperarml", "leftupperarm", "leftarm"),
    "left_lower_arm": ("lowerarml", "leftlowerarm", "leftforearm"),
    "left_hand": ("handl", "lefthand"),
    "right_upper_arm": ("upperarmr", "rightupperarm", "rightarm"),
    "right_lower_arm": ("lowerarmr", "rightlowerarm", "rightforearm"),
    "right_hand": ("handr", "righthand"),
    "left_upper_leg": ("thighl", "leftthigh", "leftupleg", "leftupperleg"),
    "left_lower_leg": ("calfl", "leftcalf", "leftleg", "leftlowerleg"),
    "left_foot": ("footl", "leftfoot"),
    "right_upper_leg": ("thighr", "rightthigh", "rightupleg", "rightupperleg"),
    "right_lower_leg": ("calfr", "rightcalf", "rightleg", "rightlowerleg"),
    "right_foot": ("footr", "rightfoot"),
}

_ATTACK_MOTION_ROLES = {
    "pelvis",
    "spine",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
}


def _resolve_humanoid_roles(joint_names: list[str]) -> dict[str, str]:
    normalized = [(name, _normalized_name(name)) for name in joint_names]
    resolved: dict[str, str] = {}
    for role, aliases in _REQUIRED_HUMANOID_ROLES.items():
        candidates = [
            (name, candidate)
            for name, candidate in normalized
            if any(alias in candidate for alias in aliases)
        ]
        if not candidates:
            continue
        name, _ = min(
            candidates,
            key=lambda item: (
                "leaf" in item[1] or "helper" in item[1] or "control" in item[1],
                not any(item[1] == alias for alias in aliases),
                len(item[1]),
            ),
        )
        resolved[role] = name
    return resolved


_CHAIN_SPINE_PATTERN = re.compile(r"^spine_(\d+)$")
_CHAIN_MIN_SPINE_BONES = 8
_CHAIN_AXIS_DEVIATION_RATIO = 0.35
_CHAIN_MIN_WEIGHTED_RATIO = 0.95


def _resolve_chain_roles(joint_names: list[str]) -> dict[str, str]:
    """Minimal role map for a chain rig: `head`, and `spine` standing in for
    the whole chain (its root segment, `spine_00`) -- enough for downstream
    code that reads `resolved_roles` (e.g. motion mapping in retarget.py) to
    look up a sane bone rather than crash on a missing humanoid role."""
    resolved: dict[str, str] = {}
    spine_candidates = sorted(
        (int(match.group(1)), name)
        for name in joint_names
        for match in [_CHAIN_SPINE_PATTERN.match(name.strip().casefold())]
        if match
    )
    if spine_candidates:
        resolved["spine"] = spine_candidates[0][1]
    head_candidates = [name for name in joint_names if name.strip().casefold() == "head"]
    if head_candidates:
        resolved["head"] = head_candidates[0]
    return resolved


def _evaluate_chain_rig_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Certify that a chain rig (a serpent or other limbless creature) is
    structurally safe to receive motion. Humanoid-specific role/handedness
    findings do not apply -- see `_evaluate_humanoid_rig_certificate` for
    those."""
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    joint_names = [str(name) for name in report.get("joint_names", [])]
    resolved_roles = _resolve_chain_roles(joint_names)

    has_head = "head" in resolved_roles
    spine_names = [
        name for name in joint_names
        if _CHAIN_SPINE_PATTERN.match(name.strip().casefold())
    ]
    if not has_head or len(spine_names) < _CHAIN_MIN_SPINE_BONES:
        findings.append(
            _finding(
                "chain.too_short",
                "error",
                "The chain rig does not have a head bone and enough spine "
                "segments for smooth full-body deformation.",
                metric="head_present/spine_bone_count",
                actual=[has_head, len(spine_names)],
                expected=f"head present, >= {_CHAIN_MIN_SPINE_BONES} spine_ bones",
            )
        )
        actions.append("rerig")

    # A chain rig should resolve entirely into the spine/head/spur
    # vocabulary skeleton_naming's chain namer produces; any other bone
    # name indicates branching structure this namer did not account for.
    unclassified = sorted(
        name for name in joint_names
        if not _CHAIN_SPINE_PATTERN.match(name.strip().casefold())
        and name.strip().casefold() != "head"
        and not name.strip().casefold().startswith("spur_")
    )
    topology = report.get("chain_topology")
    topology_branch_joints = (
        list(topology.get("branch_joints", []))
        if isinstance(topology, dict) and topology.get("is_single_chain") is False
        else []
    )
    branch_evidence = sorted(set(unclassified) | set(topology_branch_joints))
    if branch_evidence:
        findings.append(
            _finding(
                "chain.branching",
                "error",
                "The chain rig has branching structure -- bones outside the "
                "spine/head/spur vocabulary, or a spine/head bone with more "
                "than one deforming child -- rather than a single chain.",
                metric="branching_joints",
                actual=branch_evidence,
                expected="none",
            )
        )
        actions.append("rerig")

    axis_errors = report.get("chain_axis_errors", [])
    off_axis = [
        item for item in axis_errors
        if float(item.get("axis_deviation_ratio", 0.0)) > _CHAIN_AXIS_DEVIATION_RATIO
    ]
    if off_axis:
        findings.append(
            _finding(
                "chain.off_axis",
                "error",
                "Some chain bones sit off the mesh's medial axis, which "
                "will twist the surface unnaturally when animated.",
                metric="axis_deviation_ratio",
                actual=off_axis,
                expected=f"<= {_CHAIN_AXIS_DEVIATION_RATIO}",
            )
        )
        actions.append("rerig")

    skinned_vertices = int(report.get("skinned_vertices", 0))
    unskinned_vertices = int(report.get("unskinned_vertices", 0))
    unweighted_vertices = int(report.get("unweighted_vertices", 0))
    total_vertices = skinned_vertices + unskinned_vertices
    unweighted_effective = unweighted_vertices + unskinned_vertices
    weighted_ratio = 1.0 - (unweighted_effective / max(total_vertices, 1))
    if total_vertices == 0 or weighted_ratio <= _CHAIN_MIN_WEIGHTED_RATIO:
        findings.append(
            _finding(
                "rig.invalid_weights",
                "error",
                "Too much of the chain mesh is not bound to the skeleton.",
                metric="weighted_vertex_ratio",
                actual=round(weighted_ratio, 4) if total_vertices else 0.0,
                expected=f"> {_CHAIN_MIN_WEIGHTED_RATIO}",
            )
        )
        actions.append("repair-weights")

    decision = _decision("rig-certificate", findings, actions)
    decision["policy_version"] = 2
    decision["resolved_roles"] = resolved_roles
    decision["profile"] = "chain"
    return decision


def evaluate_rig_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Certify a rig is structurally safe to receive motion.

    Dispatches on the rig's profile: a humanoid gets full role resolution
    and handedness checking (`_evaluate_humanoid_rig_certificate`); a
    limbless chain creature (a serpent, tail, tentacle...) gets a much
    smaller set of chain-specific findings
    (`_evaluate_chain_rig_certificate`) -- humanoid role/handedness rules
    would either miss the rig's real problems or misfire on structure a
    chain is expected to have.
    """
    joint_names = [str(name) for name in report.get("joint_names", [])]
    profile = report.get("profile")
    if profile not in ("humanoid", "chain"):
        profile = detect_profile_from_names(joint_names)
    if profile == "chain":
        return _evaluate_chain_rig_certificate(report)
    return _evaluate_humanoid_rig_certificate(report)


def _evaluate_humanoid_rig_certificate(report: dict[str, Any]) -> dict[str, Any]:
    """Certify that a humanoid rig is structurally safe to receive motion."""
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    skins = int(report.get("skins", 0))
    joints = int(report.get("joints", 0))
    vertices = int(report.get("skinned_vertices", 0))
    if skins < 1 or joints < 15 or vertices == 0:
        findings.append(
            _finding(
                "rig.missing",
                "error",
                "A humanoid skin with at least 15 joints is required.",
                metric="skins/joints/skinned_vertices",
                actual=[skins, joints, vertices],
                expected=">= 1 skin, >= 15 joints, > 0 vertices",
            )
        )
        actions.append("rerig")

    unskinned_vertices = int(report.get("unskinned_vertices", 0))
    geometry_vertices = vertices + unskinned_vertices
    if unskinned_vertices / max(geometry_vertices, 1) > 0.01:
        findings.append(
            _finding(
                "rig.unskinned_geometry",
                "error",
                "Too much visible geometry is not bound to the humanoid skin.",
                metric="unskinned_vertex_ratio",
                actual=unskinned_vertices / max(geometry_vertices, 1),
                expected="<= 0.01",
            )
        )
        actions.append("rerig")

    joint_names = [str(name) for name in report.get("joint_names", [])]
    resolved_roles = _resolve_humanoid_roles(joint_names)
    missing_roles = sorted(set(_REQUIRED_HUMANOID_ROLES) - set(resolved_roles))
    if missing_roles:
        findings.append(
            _finding(
                "rig.humanoid_roles_missing",
                "error",
                "The skeleton is missing required humanoid roles for deterministic retargeting.",
                metric="missing_humanoid_roles",
                actual=missing_roles,
                expected="none",
            )
        )
        actions.append("rerig")

    unweighted = int(report.get("unweighted_vertices", 0))
    non_normalized = int(report.get("non_normalized_vertices", 0))
    invalid_references = int(report.get("invalid_joint_references", 0))
    if unweighted / max(vertices, 1) > 0.001 or non_normalized / max(vertices, 1) > 0.001:
        findings.append(
            _finding(
                "rig.invalid_weights",
                "error",
                "Too many vertices have missing or non-normalized skin weights.",
                metric="unweighted/non_normalized vertices",
                actual=[unweighted, non_normalized],
                expected="each <= 0.1% of skinned vertices",
            )
        )
        actions.append("repair-weights")
    if invalid_references:
        findings.append(
            _finding(
                "rig.invalid_joint_references",
                "error",
                "Skin weights reference joints outside the active skin.",
                metric="invalid_joint_references",
                actual=invalid_references,
                expected="0",
            )
        )
        actions.append("rerig")

    unused = {_normalized_name(str(name)) for name in report.get("unused_joint_names", [])}
    required_unused = sorted(
        role for role, name in resolved_roles.items() if _normalized_name(name) in unused
    )
    if required_unused:
        findings.append(
            _finding(
                "rig.required_joint_unused",
                "error",
                "Required deforming joints have no assigned vertex weights.",
                metric="unused_required_roles",
                actual=required_unused,
                expected="none",
            )
        )
        actions.append("repair-weights")

    # A finger joint with no weights is a broken skin ONLY when the rig was
    # supposed to deform with it. Since the bind worker grows the biped
    # template's optional tier (fingers, toes, head_tip) on every humanoid
    # rig -- they have to exist for a clip's finger channels to drive
    # anything -- a stylized mesh with no modelled finger geometry will
    # legitimately leave some of them unweighted. S3 already treats the
    # optional tier as optional; blocking approval on it would make every
    # such character unapprovable. Optional-tier gaps are therefore recorded
    # as an advisory warning (still surfaced on the certificate, still
    # keeping the candidate out of auto-promotion), while an unweighted
    # finger joint that is NOT an optional template bone -- a required bone,
    # or a bespoke one from another rigging path -- stays a hard error.
    unused_finger_names = sorted(
        str(name)
        for name in report.get("unused_joint_names", [])
        if is_finger_deform_bone(str(name))
    )
    optional_tier = _optional_template_bone_names("biped")
    unused_fingers = [name for name in unused_finger_names if name not in optional_tier]
    unused_optional_fingers = [name for name in unused_finger_names if name in optional_tier]
    if unused_fingers:
        findings.append(
            _finding(
                "rig.finger_joint_unused",
                "error",
                "Deforming finger joints have no assigned vertex weights.",
                metric="unused_finger_joints",
                actual=unused_fingers,
                expected="none",
            )
        )
        actions.append("repair-weights")
    if unused_optional_fingers:
        findings.append(
            _finding(
                "rig.optional_finger_joint_unused",
                "warning",
                "Optional-tier finger/toe joints carry no vertex weights: the bones are "
                "there so clip finger motion can drive them, but this mesh has no "
                "geometry close enough for the skinner to give them. Inspect the hands "
                "before relying on finger articulation.",
                metric="unused_optional_finger_joints",
                actual=unused_optional_fingers,
                expected="weighted wherever the mesh actually has finger geometry",
            )
        )
        actions.append("inspect-weight-paint")

    coverage = report.get("joint_weight_coverage", [])
    total_weight = sum(max(float(item.get("total_weight", 0.0)), 0.0) for item in coverage)
    concentrated = [
        {
            "joint": str(item.get("joint", "unknown")),
            "ratio": float(item.get("total_weight", 0.0)) / total_weight,
        }
        for item in coverage
        if total_weight and float(item.get("total_weight", 0.0)) / total_weight > 0.4
    ]
    if concentrated:
        findings.append(
            _finding(
                "rig.weight_concentration",
                "error",
                "A single joint controls too much of the character for credible full-body deformation.",
                metric="joint_total_weight_ratio",
                actual=concentrated,
                expected="each <= 0.4",
            )
        )
        actions.append("repair-weights")

    coverage_by_name: dict[str, list[dict[str, Any]]] = {}
    for item in coverage:
        coverage_by_name.setdefault(
            _normalized_name(str(item.get("joint", ""))),
            [],
        ).append(item)
    anatomical_joint_nodes = set()
    for name in resolved_roles.values():
        candidates = coverage_by_name.get(_normalized_name(name), [])
        if not candidates:
            continue
        landmark = max(
            candidates,
            key=lambda item: float(item.get("total_weight", 0.0)),
        )
        if "joint_node" in landmark:
            anatomical_joint_nodes.add(int(landmark["joint_node"]))
    bind_pose_mismatches = [
        item
        for item in report.get("bind_pose_landmark_errors", [])
        if int(item.get("joint_node", -1)) in anatomical_joint_nodes
        if float(item.get("normalized_error", 0.0)) > 0.2
    ]
    if bind_pose_mismatches:
        findings.append(
            _finding(
                "rig.bind_pose_mismatch",
                "error",
                "The skeleton rest pose does not match the geometry controlled by its joints.",
                metric="normalized_joint_landmark_error",
                actual=bind_pose_mismatches,
                expected="each <= 0.2 of character height",
            )
        )
        actions.append("inspect-weight-paint")

    rigid_ratio = float(report.get("rigid_vertex_ratio", 0.0))
    if vertices and rigid_ratio > 0.9:
        findings.append(
            _finding(
                "rig.rigid_weights",
                "warning",
                "Most vertices are assigned to only one joint; bends require visual inspection.",
                metric="rigid_vertex_ratio",
                actual=rigid_ratio,
                expected="<= 0.9 or visually approved",
            )
        )
        actions.append("inspect-weight-paint")

    handedness = report.get("handedness")
    if isinstance(handedness, dict) and handedness.get("mismatches"):
        findings.append(
            _finding(
                "rig.left_right_inverted",
                "error",
                "Bones named for one side of the body sit on the opposite side of the "
                "detected facing direction; motion retargeted onto this rig will swap "
                "hands and feet. This reuses the same facing detector the bind worker "
                "uses, so it flags any rig named this way — not only a bug in this "
                "repo's own namer.",
                metric="handedness_mismatches",
                actual={
                    "facing_forward": handedness.get("facing_forward"),
                    "facing_source": handedness.get("facing_source"),
                    "mismatches": handedness["mismatches"],
                },
                expected="named side matches the side implied by the detected facing direction",
            )
        )
        actions.append("rerig")

    # N3: biped-template conformance (PIPELINE_POLICY.md S3 -- "the
    # skeleton IS the family template"). Computed at bind time by
    # `unirig_bind_worker.unirig_bind` (skeleton_template.conformance_report
    # plus the position-based mirror gate,
    # skeleton_naming.template_side_conformance_findings) and threaded
    # through unchanged -- already shaped like this module's own findings
    # (code/severity/message/metric/actual/expected), so wiring it in is a
    # straight concatenation, not a reshape. `None` means the GLB carries
    # no template-conformance data at all (a non-humanoid rig, or one from
    # before this wiring existed) and adds nothing; `[]` is a real
    # "fully conformant" result, also adding nothing.
    template_conformance = report.get("template_conformance")
    if isinstance(template_conformance, list) and template_conformance:
        findings.extend(template_conformance)
        actions.append("rerig")

    decision = _decision("rig-certificate", findings, actions)
    decision["policy_version"] = 2
    decision["resolved_roles"] = resolved_roles
    decision["profile"] = "humanoid"
    return decision


@lru_cache(maxsize=4)
def _optional_template_bone_names(family: str) -> frozenset[str]:
    """Names of `family`'s OPTIONAL-tier template bones (biped: the 45
    finger/toe/head_tip bones). Empty when the family has no template, so a
    missing template can only make the gate stricter, never looser."""
    try:
        template = load_template(family)
    except (FileNotFoundError, KeyError, ValueError):
        return frozenset()
    return frozenset(bone.name for bone in template.optional_bones)


def _weight_and_rigidity_findings(
    vertices: int,
    unweighted: int,
    non_normalized: int,
    invalid_references: int,
    rigid_ratio: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Skin-weight validity checks shared by both the humanoid and chain
    animation-review variants -- these are generic (no role/joint-count
    assumptions), so profile has no bearing on them."""
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    if unweighted / max(vertices, 1) > 0.001 or non_normalized / max(vertices, 1) > 0.001:
        findings.append(
            _finding(
                "rig.invalid_weights",
                "error",
                "Too many vertices have missing or non-normalized skin weights.",
                metric="unweighted/non_normalized vertices",
                actual=[unweighted, non_normalized],
                expected="each <= 0.1% of skinned vertices",
            )
        )
        actions.append("repair-weights")
    if invalid_references:
        findings.append(
            _finding(
                "rig.invalid_joint_references",
                "error",
                "Skin weights reference joints outside the active skin.",
                metric="invalid_joint_references",
                actual=invalid_references,
                expected="0",
            )
        )
        actions.append("rerig")
    if vertices and rigid_ratio > 0.9:
        findings.append(
            _finding(
                "rig.rigid_weights",
                "warning",
                "Most vertices are assigned to only one joint; bends may look segmented.",
                metric="rigid_vertex_ratio",
                actual=rigid_ratio,
                expected="<= 0.9 or visually approved",
            )
        )
        actions.append("inspect-weight-paint")
    return findings, actions


def _match_required_animation(report: dict[str, Any], required_animation: str) -> dict[str, Any] | None:
    requested = _normalized_name(required_animation)
    return next(
        (
            animation
            for animation in report.get("animations", [])
            if _normalized_name(str(animation.get("name", ""))) == requested
        ),
        None,
    )


def _animation_missing_finding(report: dict[str, Any], required_animation: str) -> dict[str, Any]:
    return _finding(
        "animation.missing",
        "error",
        f"The required {required_animation} clip is not embedded.",
        metric="animation_names",
        actual=[item.get("name") for item in report.get("animations", [])],
        expected=required_animation,
    )


def _evaluate_chain_animation(
    report: dict[str, Any],
    *,
    required_animation: str,
) -> dict[str, Any]:
    """Chain-rig variant of the animation-review gate. Skips every
    humanoid role/joint-count assumption (see `_evaluate_humanoid_animation`
    for those): the joint-count floor matches the chain certificate's own
    (>= 8 spine_ bones + head), and clip coverage falls back straight to
    the generic `animated_joint_coverage` ratio since a chain has no
    "attack roles" to check against.
    """
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    skins = int(report.get("skins", 0))
    vertices = int(report.get("skinned_vertices", 0))
    joint_names = [str(name) for name in report.get("joint_names", [])]
    has_head = any(name.strip().casefold() == "head" for name in joint_names)
    spine_names = [name for name in joint_names if _CHAIN_SPINE_PATTERN.match(name.strip().casefold())]
    if skins < 1 or vertices == 0 or not has_head or len(spine_names) < _CHAIN_MIN_SPINE_BONES:
        findings.append(
            _finding(
                "rig.missing",
                "error",
                "A chain skin with a head bone and enough spine segments is required.",
                metric="skins/head_present/spine_bone_count/skinned_vertices",
                actual=[skins, has_head, len(spine_names), vertices],
                expected=f">= 1 skin, head present, >= {_CHAIN_MIN_SPINE_BONES} spine_ bones, > 0 vertices",
            )
        )
        actions.append("rerig")

    weight_findings, weight_actions = _weight_and_rigidity_findings(
        vertices,
        int(report.get("unweighted_vertices", 0)),
        int(report.get("non_normalized_vertices", 0)),
        int(report.get("invalid_joint_references", 0)),
        float(report.get("rigid_vertex_ratio", 0.0)),
    )
    findings.extend(weight_findings)
    actions.extend(weight_actions)

    matching = _match_required_animation(report, required_animation)
    if matching is None:
        findings.append(_animation_missing_finding(report, required_animation))
        actions.append("attach-animation")
    else:
        duration = float(matching.get("duration_seconds", 0.0))
        coverage = float(matching.get("animated_joint_coverage", 0.0))
        expected_coverage = 0.5
        invalid_targets = int(matching.get("invalid_targets", 0))
        if duration <= 0.1 or coverage < expected_coverage or invalid_targets:
            findings.append(
                _finding(
                    "animation.invalid",
                    "error",
                    "The required animation has insufficient duration, joint coverage, or valid targets.",
                    metric="duration/animated_joint_coverage/invalid_targets",
                    actual=[duration, coverage, invalid_targets],
                    expected=(
                        f"> 0.1s, >= {expected_coverage:.1f} animated_joint_coverage, "
                        "0 invalid targets"
                    ),
                )
            )
            actions.append("reattach-animation")
    return _decision("animation-review", findings, actions)


def evaluate_for_animation(
    report: dict[str, Any],
    *,
    required_animation: str,
) -> dict[str, Any]:
    """Gate a rig + embedded clip as ready for the final review stage.

    Dispatches on profile: a humanoid gets full role/joint-count checks
    (`_evaluate_humanoid_animation`); a chain rig gets the chain-appropriate
    variant (`_evaluate_chain_animation`) so a serpent isn't held to a
    15-joint humanoid floor or humanoid attack-role coverage it can never
    satisfy.
    """
    joint_names = [str(name) for name in report.get("joint_names", [])]
    profile = report.get("profile")
    if profile not in ("humanoid", "chain"):
        profile = detect_profile_from_names(joint_names)
    if profile == "chain":
        return _evaluate_chain_animation(report, required_animation=required_animation)
    return _evaluate_humanoid_animation(report, required_animation=required_animation)


def _evaluate_humanoid_animation(
    report: dict[str, Any],
    *,
    required_animation: str,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    skins = int(report.get("skins", 0))
    joints = int(report.get("joints", 0))
    vertices = int(report.get("skinned_vertices", 0))
    if skins < 1 or joints < 15 or vertices == 0:
        findings.append(
            _finding(
                "rig.missing",
                "error",
                "A humanoid skin with at least 15 joints is required.",
                metric="skins/joints/skinned_vertices",
                actual=[skins, joints, vertices],
                expected=">= 1 skin, >= 15 joints, > 0 vertices",
            )
        )
        actions.append("rerig")

    unweighted = int(report.get("unweighted_vertices", 0))
    non_normalized = int(report.get("non_normalized_vertices", 0))
    invalid_references = int(report.get("invalid_joint_references", 0))
    rigid_ratio = float(report.get("rigid_vertex_ratio", 0.0))
    weight_findings, weight_actions = _weight_and_rigidity_findings(
        vertices, unweighted, non_normalized, invalid_references, rigid_ratio,
    )
    findings.extend(weight_findings)
    actions.extend(weight_actions)

    unused_names = [str(name) for name in report.get("unused_joint_names", [])]
    core_unused = [
        name
        for name in unused_names
        if any(token in _normalized_name(name) for token in ("pelvis", "hips", "spine", "upperarm", "thigh"))
    ]
    if core_unused:
        findings.append(
            _finding(
                "rig.core_joint_unused",
                "warning",
                "Core deforming joints have no assigned vertex weights.",
                metric="unused_core_joints",
                actual=core_unused,
                expected="none",
            )
        )
        actions.append("inspect-weight-paint")

    matching = _match_required_animation(report, required_animation)
    if matching is None:
        findings.append(_animation_missing_finding(report, required_animation))
        actions.append("attach-animation")
    else:
        duration = float(matching.get("duration_seconds", 0.0))
        animated_names = {
            _normalized_name(str(name))
            for name in matching.get("animated_joint_names", [])
        }
        resolved_roles = _resolve_humanoid_roles(
            [str(name) for name in report.get("joint_names", [])]
        )
        attack_joints = {
            _normalized_name(resolved_roles[role])
            for role in _ATTACK_MOTION_ROLES
            if role in resolved_roles
        }
        if animated_names and len(attack_joints) == len(_ATTACK_MOTION_ROLES):
            coverage = len(animated_names & attack_joints) / len(attack_joints)
            expected_coverage = 0.8
            coverage_metric = "required_attack_role_coverage"
        else:
            coverage = float(matching.get("animated_joint_coverage", 0.0))
            expected_coverage = 0.5
            coverage_metric = "animated_joint_coverage"
        invalid_targets = int(matching.get("invalid_targets", 0))
        if duration <= 0.1 or coverage < expected_coverage or invalid_targets:
            findings.append(
                _finding(
                    "animation.invalid",
                    "error",
                    "The required animation has insufficient duration, joint coverage, or valid targets.",
                    metric=f"duration/{coverage_metric}/invalid_targets",
                    actual=[duration, coverage, invalid_targets],
                    expected=(
                        f"> 0.1s, >= {expected_coverage:.1f} {coverage_metric}, "
                        "0 invalid targets"
                    ),
                )
            )
            actions.append("reattach-animation")
    return _decision("animation-review", findings, actions)


def evaluate_deformation_review(report: dict[str, Any]) -> dict[str, Any]:
    """Gate animation candidates on measured surface stability across the clip."""
    findings: list[dict[str, Any]] = []
    actions: list[str] = []
    frame_count = len(report.get("frames", []))
    complete_action_range = report.get("complete_action_range") is True
    p99_stretch = float(report.get("maximum_p99_edge_stretch", 0.0))
    if not complete_action_range:
        findings.append(_finding(
            "deformation.evidence_missing",
            "error",
            "Full-clip deformation evidence is required before promotion.",
            metric="review_frames",
            actual=frame_count,
            expected=">= 5",
        ))
        actions.append("render-deformation-review")
    elif p99_stretch > 8.0:
        findings.append(_finding(
            "deformation.edge_stretch",
            "error",
            "The animated surface stretches catastrophically across the attack clip.",
            metric="maximum_p99_edge_stretch",
            actual=p99_stretch,
            expected="<= 8.0",
        ))
        actions.extend(["repair-weights", "inspect-weight-paint"])
    elif p99_stretch > 6.0:
        findings.append(_finding(
            "deformation.edge_stretch",
            "warning",
            "The animation contains elevated surface stretch that requires visual review.",
            metric="maximum_p99_edge_stretch",
            actual=p99_stretch,
            expected="<= 6.0 or visually approved",
        ))
        actions.append("inspect-weight-paint")
    else:
        findings.append(_finding(
            "deformation.visual_review_required",
            "warning",
            "Stable edge lengths do not prove anatomical motion or clean skin deformation; inspect the retained attack frames.",
            metric="maximum_p99_edge_stretch",
            actual=p99_stretch,
            expected="visual approval of the retained full-clip evidence",
        ))
        actions.append("inspect-deformation-review")
    findings.extend(_motion_review_findings(report.get("motion_review"), actions))
    return _decision("deformation-review", findings, actions)


# A clip whose trailing portion has no bone turning faster than
# motion_metrics.STATIC_FRAME_THRESHOLD_DEGREES_PER_FRAME has gone dead before
# it ended. Calibrated on the 2026-08-31 blame-split corpus: the donor BVH,
# the donor source and the curated source all measured <= 0.053, while the
# baked clip that visibly froze a quarter of the way through measured 0.76.
# 0.25 sits in the empty gap between those two populations.
STATIC_TAIL_FRACTION_THRESHOLD = 0.25

# Fraction of length-bearing bones that must show real direction change over
# the clip. Every healthy clip in that same corpus measured >= 0.98; the 0.5
# floor matches the coverage floor `evaluate_animation_review` already applies
# to animated-joint coverage, rather than inventing a second number.
JOINT_MOTION_COVERAGE_THRESHOLD = 0.5


def _motion_review_findings(
    motion: Any,
    actions: list[str],
) -> list[dict[str, Any]]:
    """Advisory findings from the measured motion metrics.

    Only the two metrics with a calibrated threshold produce a finding.
    `foot_slide_*` and `root_travel_*` are recorded by
    `motion_metrics.compute_motion_metrics` but deliberately NOT gated here:
    the stance-foot heuristic overcounts slide on genuinely stepping clips, so
    a threshold would have to be per-motion-family and no such calibration
    exists yet.
    """
    if not isinstance(motion, dict) or not motion.get("measurable"):
        return []
    if motion.get("synthetic"):
        # A rest-pose sway synthesized because the rig carries no clip yet is
        # not evidence about any clip's motion quality.
        return []
    findings: list[dict[str, Any]] = []
    static_tail = float(motion.get("static_tail_fraction", 0.0))
    coverage = float(motion.get("joint_motion_coverage", 0.0))
    if static_tail > STATIC_TAIL_FRACTION_THRESHOLD:
        findings.append(_finding(
            "motion.static_tail",
            "warning",
            "The clip stops moving before it ends; the trailing frames carry no "
            "bone motion, which is how a dropped animation bake looks.",
            metric="static_tail_fraction",
            actual=round(static_tail, 4),
            expected=f"<= {STATIC_TAIL_FRACTION_THRESHOLD}",
        ))
        actions.append("reattach-animation")
    if coverage < JOINT_MOTION_COVERAGE_THRESHOLD:
        findings.append(_finding(
            "motion.joint_coverage",
            "warning",
            "Most of the rig never changes direction during the clip; the "
            "animation is reaching only a minority of the bones.",
            metric="joint_motion_coverage",
            actual=round(coverage, 4),
            expected=f">= {JOINT_MOTION_COVERAGE_THRESHOLD}",
        ))
        actions.append("reattach-animation")
    return findings
