"""UniRig auto-rig Blender worker: strip embedded rigs, bind predicted skeletons.

Usage: blender --background --factory-startup --python unirig_bind_worker.py -- '<json spec>'

Ops:
  strip-rig    Remove armatures, armature modifiers, and skin parenting from a
               GLB so UniRig sees a clean mesh-only asset.
  unirig-bind  Import a clean mesh plus UniRig's predicted skeleton npz, undo
               the inference-side [-1, 1] cube normalization (AugmentAffine:
               v -> (v - bbox_center) / half_max_extent), build the armature,
               bind with Blender bone-heat weights, validate per-bone vertex
               assignment, and export a rigged GLB.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.skeleton_naming import (
    CONFIDENT_FACING_SOURCES,
    TorsoShellRegion,
    build_template_conformant_skeleton,
    chain_candidate_extent_ratio,
    densify_chain,
    bridge_blend_weights,
    detect_bridge_region,
    detect_hanging_region,
    detect_profile_from_names,
    detect_torso_shell,
    distal_refinement_json,
    hanging_region_blend_weights,
    left_from_forward,
    medial_axis_chain,
    mix_smoothed_weights,
    nearest_bone_blend_weights,
    refine_distal_anchors,
    remove_joints_reparent,
    resolve_position_overrides,
    score_chain_candidate,
    select_chain_continuation_child,
    semantic_joint_names as _semantic_joint_names,
    stitch_coincident_vertex_weights,
    template_side_conformance_findings,
    torso_shell_blend_weights,
)
from backend.app.joint_containment import measure_joint_containment
from backend.app.skeleton_naming import (
    _BRIDGE_BALANCE,
    _BRIDGE_CEILING,
    _BRIDGE_CLEARANCE,
    _BRIDGE_CORRIDOR,
    _BRIDGE_PELVIS_FLOOR,
    _TORSO_CEILING,
    _TORSO_RADIAL_GAP,
    _TORSO_SHOULDER_SHARE,
)
from backend.app.skeleton_template import conformance_report as template_conformance_report
from backend.app.skeleton_template import load_template
from backend.app.skin_weights import smooth_weight_maps

# Non-deform helper bones (chain rigs' "foot_extra_l_47" and similar) never
# carry mesh influence -- excluded from the nearest-bone-blend candidate
# set entirely, same as they were never usefully weighted by envelope.
_NON_DEFORM_NAME_MARKER = "_extra_"

# Topology-aware smoothing applied only to the nearest-bone-blend weighting
# path (see `apply_nearest_bone_blend_weights`): each vertex's weight
# vector is pulled halfway toward the mean of its mesh-edge neighbours',
# repeated this many iterations, to soften the hard bone-ownership
# boundaries a purely geometric nearest-segment assignment still leaves at
# multi-bone junctions (neck: head/neck/chest; shoulder: chest/upper-arm)
# -- the shoulder-clipping / stretched-neck skinning symptom. Bone-heat and
# distance-along-chain weighting are untouched by this.
_BLEND_SMOOTHING_FACTOR = 0.5
_BLEND_SMOOTHING_ITERATIONS = 8
_BLEND_SMOOTHING_MAX_INFLUENCES = 4

# Coincident-vertex position-match tolerance for seam stitching (see
# `apply_seam_stitching`) -- generous enough to catch float-precision drift
# between UV/normal-split duplicates of the same source vertex, tight
# enough to never merge two genuinely distinct, nearby vertices.
_SEAM_STITCH_TOLERANCE = 1e-6

# glTF's own skinning limit, and the cap the hanging-region blend
# renormalizes to (see `apply_hanging_region_blend`).
_HANGING_MAX_INFLUENCES = 4


def log(message: str) -> None:
    print(f"[unirig-bind-worker] {message}", flush=True)


def _names_json_path(spec: dict) -> Path | None:
    """Persistent name->joint-index map location for this bind's lineage.

    For a rig-adjust (rebind) call, `raw_npz` is always
    `<bind_inputs_dir>/raw_data.npz` (see `main.py`'s rig-adjust handler and
    `native.py:run_unirig_rebind`), so its parent directory is the durable,
    per-character `bind_inputs_dir` that persists across every rebind of the
    same character. Writing/reading `names.json` there lets a later rebind
    resolve a hand-adjustment map by the joint indices it was authored
    against, even if bone naming changes between binds.

    For the very first auto-rig bind, `raw_npz` is still inside a temporary
    extraction directory; `native.py:run_unirig_auto_rig` copies the file
    this function names into the durable `bind_inputs_dir` afterward,
    mirroring how it already copies `raw_data.npz` and the other bind inputs.
    """
    raw_npz = spec.get("raw_npz")
    if not raw_npz:
        return None
    return Path(raw_npz).parent / "names.json"


def _load_stored_names(names_path: Path | None) -> dict[str, int] | None:
    if names_path is None or not names_path.is_file():
        return None
    try:
        data = json.loads(names_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Current format nests the name->index map under "names" (alongside
    # "profile"); a names.json written before the chain-profile addition is
    # the flat map itself.
    mapping = data["names"] if isinstance(data.get("names"), dict) else data
    try:
        return {str(name): int(index) for name, index in mapping.items()}
    except (TypeError, ValueError):
        return None


def _write_stored_names(names_path: Path | None, names: list[str], profile: str) -> None:
    if names_path is None:
        return
    mapping = {name: index for index, name in enumerate(names)}
    payload = {"names": mapping, "profile": profile}
    names_path.parent.mkdir(parents=True, exist_ok=True)
    names_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_stored_profile(names_path: Path | None) -> str | None:
    """The `"profile"` a prior bind of this same lineage wrote alongside
    its name->index map (see `_write_stored_names`), if any."""
    if names_path is None or not names_path.is_file():
        return None
    try:
        data = json.loads(names_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    profile = data.get("profile") if isinstance(data, dict) else None
    return profile if profile in ("humanoid", "chain") else None


def _import_source_rig_skeleton(source_rig_path: str) -> tuple[np.ndarray, list[int | None], list[str], np.ndarray]:
    """Read a rigged GLB's own armature directly, rather than re-deriving a
    skeleton from a UniRig prediction: joint head positions, bone names,
    and parent indices, in the armature's `data.bones` order. Every object
    this imports (the armature and any skinned meshes) is deleted before
    returning, leaving the scene as it was found -- the caller imports the
    clean mesh separately afterward.

    The source rig's own rest-clip action (baked in by a prior bind's
    "First-time rigs carry no animation" fallback, see `unirig_bind`)
    imports into `bpy.data.actions` along with the armature but is *not*
    deleted by removing the armature object -- actions are independent
    datablocks. Left behind, it would make the later new-armature
    "carries no animation" check see a playable action in the global pool
    and skip creating a Rest clip for the *new* armature, exporting a GLB
    with no `animations` at all (this broke the deformation-review gate,
    which needs a posed frame to measure). Every action newly present
    after this import is removed before returning so the caller's scene
    is left exactly as it was found, actions included.
    """
    before_objects = set(bpy.context.scene.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=source_rig_path)
    new_objects = [o for o in bpy.context.scene.objects if o not in before_objects]
    armatures = [o for o in new_objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"source_rig GLB contains no armature: {source_rig_path}")
    arm = armatures[0]
    bones = list(arm.data.bones)
    name_to_index = {bone.name: index for index, bone in enumerate(bones)}
    joints = np.asarray(
        [np.asarray(arm.matrix_world @ bone.head_local, dtype=np.float64) for bone in bones],
        dtype=np.float64,
    )
    names = [bone.name for bone in bones]
    # Keep the source rig's own bone tails: leaf bones (toe/finger helpers) have no
    # child to rebuild a tail from, and the retarget worker reads bone directions
    # from tails -- a default +Z tail on a toe helper flips the foot direction and
    # folds the transferred pose.
    tails = np.asarray(
        [np.asarray(arm.matrix_world @ bone.tail_local, dtype=np.float64) for bone in bones],
        dtype=np.float64,
    )
    parents: list[int | None] = [
        name_to_index.get(bone.parent.name) if bone.parent is not None else None
        for bone in bones
    ]
    for obj in new_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        if action not in before_actions:
            bpy.data.actions.remove(action)
    log(f"source_rig armature: {len(bones)} bone(s) read from {source_rig_path}")
    return joints, parents, names, tails


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_meshes(glb_path: str) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    meshes = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB contains no mesh objects")
    return meshes


def strip_rig(spec: dict) -> None:
    clear_scene()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=spec["input"])
    new_objects = [o for o in bpy.context.scene.objects if o not in before]

    removed = 0
    for obj in new_objects:
        if obj.type == "ARMATURE":
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for modifier in list(obj.modifiers):
            if modifier.type == "ARMATURE":
                obj.modifiers.remove(modifier)
        if obj.parent is not None and obj.parent.type == "ARMATURE":
            obj.matrix_world = obj.parent.matrix_world @ obj.matrix_world
            obj.parent = None
    log(f"stripped {removed} armature(s)")

    out_path = Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB")
    log(f"stripped GLB: {out_path} bytes={out_path.stat().st_size}")


def semantic_joint_names(
    joints: np.ndarray,
    parents: list,
    mesh_vertices=None,
    profile_hint: str | None = None,
) -> tuple[list[str], dict]:
    """Derive semantic bone names, logging the detected profile and facing.

    The pure topology/handedness/profile logic lives in
    ``backend.app.skeleton_naming`` (bpy-free, unit tested); this wrapper
    adapts it to the worker's logging and validation-JSON conventions.
    Returns (names, facing_report) where facing_report is JSON-serializable
    for the bind validation output and carries "profile" (``"humanoid"`` or
    ``"chain"``), "profile_source" (``"hint"`` or ``"detected"``), alongside
    the facing/forward-direction fields.

    `profile_hint` ("humanoid"/"chain", from the run's subject_kind)
    overrides automatic detection -- see `semantic_joint_names` in
    skeleton_naming.py for exactly how (a "humanoid" hint always wins; a
    "chain" hint only wins if the topology actually supports it).
    """
    result = _semantic_joint_names(joints, parents, mesh_vertices=mesh_vertices, profile_hint=profile_hint)
    facing_report = {
        "forward": [round(float(v), 4) for v in result.forward],
        "source": result.facing_source,
        "warning": result.warning,
        "profile": result.profile,
        "profile_source": result.profile_source,
    }
    log(
        f"profile={result.profile} ({result.profile_source}) facing={facing_report['forward']} "
        f"source={facing_report['source']}"
    )
    if result.warning:
        log(f"WARNING: {result.warning}")
    return result.names, facing_report


def build_armature(
    joints: np.ndarray,
    tails: np.ndarray | None,
    parents: list,
    names: list[str],
    *,
    rolls: list[float] | None = None,
    trust_tails: bool = False,
) -> bpy.types.Object:
    """Build the armature from placed joint heads/tails/parents/names.

    `rolls`, when given, sets each bone's edit-bone roll directly (S3:
    "template rolls are law" -- this function previously set no roll at
    all). `trust_tails` skips the anatomical-tails child-selection pass
    below: a template-conformant build (see
    `skeleton_naming.build_template_conformant_skeleton`) already supplies
    an exact template rest-direction tail for every bone, which the
    child-guess heuristic would otherwise silently overwrite for any bone
    with more than zero children -- i.e. every non-leaf bone.
    """
    arm_data = bpy.data.armatures.new("UniRigArmature")
    arm_obj = bpy.data.objects.new("UniRigArmature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = []
    for i in range(len(joints)):
        bone = arm_data.edit_bones.new(names[i] or f"bone_{i}")
        bone.head = joints[i]
        tail = tails[i] if tails is not None else joints[i] + np.array([0.0, 0.0, 0.05])
        bone.tail = tail if (Vector(tail) - Vector(joints[i])).length > 1e-6 else joints[i] + Vector((0.0, 0.0, 0.05))
        edit_bones.append(bone)
    for i, bone in enumerate(edit_bones):
        parent = parents[i]
        if parent is not None and parent >= 0 and parent != i:
            bone.parent = edit_bones[int(parent)]
            bone.use_connect = False
    if not trust_tails:
        # Anatomical tails: point every bone at a deliberately chosen child so
        # bone directions are consistent across the hierarchy (UniRig's
        # predicted tails are unreliable — mixed conventions broke retargeting
        # alignment). A bone with several children (pelvis -> spine + two
        # thighs; the last spine bone -> neck + two upper arms) needs a
        # deliberate pick, not "whichever child has the lowest joint index" —
        # that could point the pelvis's tail down a leg and skew envelope-
        # fallback bone lengths, and the retarget pipeline relies on pelvis/spine
        # rest directions pointing up.
        children_by_parent: dict[str, list] = {}
        for bone in edit_bones:
            parent = bone.parent
            if parent is not None:
                children_by_parent.setdefault(parent.name, []).append(bone)
        for bone in edit_bones:
            children = children_by_parent.get(bone.name)
            if not children:
                continue
            if len(children) == 1:
                chosen = children[0]
            else:
                parent_head = np.asarray(bone.parent.head, dtype=np.float64) if bone.parent is not None else None
                chosen_index = select_chain_continuation_child(
                    bone.name,
                    np.asarray(bone.head, dtype=np.float64),
                    parent_head,
                    [child.name for child in children],
                    [np.asarray(child.head, dtype=np.float64) for child in children],
                )
                chosen = children[chosen_index]
            direction = chosen.head - bone.head
            if direction.length > 1e-6:
                bone.tail = bone.head + direction.normalized() * max(direction.length * 0.9, 0.02)
    if rolls is not None:
        # Set after tails are finalized above -- roll is only meaningful
        # relative to the bone's final head->tail direction.
        for bone, roll in zip(edit_bones, rolls):
            bone.roll = float(roll)
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _mesh_world_vertices(mesh_obj) -> np.ndarray:
    vertex_count = len(mesh_obj.data.vertices)
    local = np.empty(vertex_count * 3, dtype=np.float64)
    mesh_obj.data.vertices.foreach_get("co", local)
    local = local.reshape(vertex_count, 3)
    matrix = np.asarray(mesh_obj.matrix_world, dtype=np.float64)
    homogeneous = np.concatenate([local, np.ones((vertex_count, 1), dtype=np.float64)], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def _mesh_world_triangles(mesh_obj) -> np.ndarray:
    """The mesh's triangulated surface in world space, `(T, 3, 3)`.

    `joint_containment` casts rays against a triangle soup; Blender's mesh
    is polygonal, so the loop triangulation is computed first (`n`-gons and
    quads both fan out here).
    """
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    triangle_count = len(mesh.loop_triangles)
    if triangle_count == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    corner_indices = np.empty(triangle_count * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", corner_indices)
    vertices = _mesh_world_vertices(mesh_obj)
    return vertices[corner_indices.reshape(triangle_count, 3)]


def _mesh_edge_pairs(mesh_obj) -> list[tuple[int, int]]:
    """The mesh's own edge topology as `(vertex_index, vertex_index)`
    pairs -- used to smooth nearest-bone-blend weights only across actual
    mesh-surface connectivity, never across nearby-in-space but
    disconnected geometry (see `smooth_weight_maps`)."""
    edge_count = len(mesh_obj.data.edges)
    raw = np.empty(edge_count * 2, dtype=np.int64)
    mesh_obj.data.edges.foreach_get("vertices", raw)
    return [(int(a), int(b)) for a, b in raw.reshape(edge_count, 2)]


def _mesh_group_weights_sparse(mesh_obj) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read every vertex's current vertex-group weights into the sparse
    `(V, max_influences)` representation `stitch_coincident_vertex_weights`
    (and `nearest_bone_blend_weights`) use: `bone_indices` int with unused
    slots `-1`, `weights` float with unused slots `0.0`. `max_influences` is
    however many groups the most-influenced vertex in this mesh actually
    carries (never a fixed cap), so this covers bone-heat's own group
    counts as faithfully as the geometric weighting paths' fixed widths.
    `group_names` is the vertex-group list in index order, so slot values
    in `bone_indices` are indices into it.
    """
    group_names = [group.name for group in mesh_obj.vertex_groups]
    vertices = mesh_obj.data.vertices
    vertex_count = len(vertices)
    max_influences = 1
    for vertex in vertices:
        count = len(vertex.groups)
        if count > max_influences:
            max_influences = count

    bone_indices = np.full((vertex_count, max_influences), -1, dtype=np.int64)
    weights = np.zeros((vertex_count, max_influences), dtype=np.float64)
    for vertex_index, vertex in enumerate(vertices):
        for slot, element in enumerate(vertex.groups):
            bone_indices[vertex_index, slot] = element.group
            weights[vertex_index, slot] = element.weight
    return bone_indices, weights, group_names


def _rewrite_mesh_group_weights(
    mesh_obj, bone_indices: np.ndarray, weights: np.ndarray, group_names: list[str]
) -> None:
    """Replace every vertex group's weight assignment with `bone_indices`/
    `weights` (the sparse `(V, max_influences)` form `_mesh_group_weights_sparse`
    reads). Groups are cleared and recreated in the same name order so
    `group_names[bone_index]` stays the correct bone identity -- the same
    clear-then-`.add()` pattern `apply_distance_along_chain_weights` uses.
    """
    for group in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(group)
    groups = [mesh_obj.vertex_groups.new(name=name) for name in group_names]
    vertex_count, max_influences = bone_indices.shape
    for vertex_index in range(vertex_count):
        for slot in range(max_influences):
            bone_index = int(bone_indices[vertex_index, slot])
            weight = float(weights[vertex_index, slot])
            if bone_index < 0 or weight <= 1e-6:
                continue
            groups[bone_index].add([vertex_index], weight, "REPLACE")


def apply_seam_stitching(mesh_obj) -> np.ndarray:
    """Make every group of coincident vertices (duplicated at UV/normal
    seams -- see `stitch_coincident_vertex_weights`) share identical,
    weight-averaged bone influences, so a pose can never pull a seam apart.
    Zero geometry/texture change: only per-vertex bone weights are
    rewritten.

    Called after EVERY weighting path's final vertex-group state, right
    before export -- bone-heat, nearest-bone-blend (after its own
    smoothing pass), distance-along-chain, and again after the
    nearest-joint unweighted-vertex fallback, since that fallback can
    reintroduce a fresh mismatch at exactly the vertices it just filled
    (two coincident vertices independently assigned to their own nearest
    joint, which need not be the same joint).

    Returns the array of vertex indices whose weights were actually
    rewritten (empty if nothing needed stitching -- e.g. a mesh with no
    coincident vertices, or one already stitched by a prior call this run).
    """
    vertex_count = len(mesh_obj.data.vertices)
    if vertex_count == 0:
        return np.empty(0, dtype=np.int64)

    bone_indices, weights, group_names = _mesh_group_weights_sparse(mesh_obj)
    world_vertices = _mesh_world_vertices(mesh_obj)
    stitched_bones, stitched_weights = stitch_coincident_vertex_weights(
        world_vertices, bone_indices, weights, tolerance=_SEAM_STITCH_TOLERANCE
    )

    changed = np.any(stitched_bones != bone_indices, axis=1) | np.any(
        np.abs(stitched_weights - weights) > 1e-9, axis=1
    )
    changed_indices = np.nonzero(changed)[0]
    if changed_indices.size == 0:
        return changed_indices

    _rewrite_mesh_group_weights(mesh_obj, stitched_bones, stitched_weights, group_names)
    return changed_indices


def chain_end_weight_concentration(mesh_obj, ordered_chain_names: list[str]) -> float:
    """Fraction of total vertex weight held by the two end bones of a
    linear chain -- a symptom of bone-heat/envelope weighting collapsing
    onto the ends of a long, thin body instead of spreading along it."""
    if len(ordered_chain_names) < 2:
        return 0.0
    end_group_indices = {
        group.index for group in mesh_obj.vertex_groups
        if group.name in (ordered_chain_names[0], ordered_chain_names[-1])
    }
    if not end_group_indices:
        return 0.0
    end_weight = 0.0
    total_weight = 0.0
    for vertex in mesh_obj.data.vertices:
        for element in vertex.groups:
            total_weight += element.weight
            if element.group in end_group_indices:
                end_weight += element.weight
    return end_weight / total_weight if total_weight > 1e-9 else 0.0


def apply_distance_along_chain_weights(mesh_obj, ordered_chain_positions: np.ndarray, ordered_chain_names: list[str]) -> int:
    """Two-bone-blend skinning along a linear bone chain: each vertex is
    weighted by how far its projection onto the chain polyline falls
    between its two nearest bones. Robust where bone-heat/envelope
    weighting collapses onto the end bones on a long, thin body -- this
    scheme cannot collapse, since every vertex is anchored to whichever
    segment it is geometrically nearest.
    """
    for group in list(mesh_obj.vertex_groups):
        mesh_obj.vertex_groups.remove(group)
    groups = [mesh_obj.vertex_groups.new(name=name) for name in ordered_chain_names]

    segment_starts = ordered_chain_positions[:-1]
    segment_vectors = ordered_chain_positions[1:] - segment_starts
    segment_length_sq = np.sum(segment_vectors ** 2, axis=1)
    num_segments = len(segment_vectors)

    world_vertices = _mesh_world_vertices(mesh_obj)
    vertex_count = len(world_vertices)
    best_distance_sq = np.full(vertex_count, np.inf)
    best_segment = np.zeros(vertex_count, dtype=np.int64)
    best_t = np.zeros(vertex_count, dtype=np.float64)
    for segment_index in range(num_segments):
        offset = world_vertices - segment_starts[segment_index]
        length_sq = segment_length_sq[segment_index]
        if length_sq <= 1e-12:
            t = np.zeros(vertex_count)
        else:
            t = np.clip(offset @ segment_vectors[segment_index] / length_sq, 0.0, 1.0)
        projected = segment_starts[segment_index] + np.outer(t, segment_vectors[segment_index])
        distance_sq = np.sum((world_vertices - projected) ** 2, axis=1)
        better = distance_sq < best_distance_sq
        best_distance_sq = np.where(better, distance_sq, best_distance_sq)
        best_segment = np.where(better, segment_index, best_segment)
        best_t = np.where(better, t, best_t)

    assigned = 0
    for vertex_index in range(vertex_count):
        segment_index = int(best_segment[vertex_index])
        t = float(best_t[vertex_index])
        current_weight = 1.0 - t
        next_weight = t
        if current_weight > 1e-6:
            groups[segment_index].add([vertex_index], current_weight, "REPLACE")
        if next_weight > 1e-6:
            groups[segment_index + 1].add([vertex_index], next_weight, "ADD")
        if current_weight > 1e-6 or next_weight > 1e-6:
            assigned += 1
    return assigned


def apply_nearest_bone_blend_weights(mesh_obj, armature_obj) -> int:
    """Geometric nearest-bone-segment skinning: replaces the
    ARMATURE_ENVELOPE fallback used when bone-heat weighting fails.

    Blender's envelope weighting sizes each bone's capture radius from its
    own length (`bone.envelope_distance`), which on slender limbs reaches
    past the neighbouring joint into the next segment entirely -- a calf's
    envelope dominating vertices well above the knee, into the thigh. This
    scheme instead measures true point-to-bone-*segment* distance (see
    `nearest_bone_blend_weights`) with no radius involved, so a vertex can
    never be claimed by a bone whose segment it isn't actually near.

    Assumes vertex groups for every bone already exist and are empty (from
    `bpy.ops.object.parent_set(type="ARMATURE_NAME")`, called by the
    caller before this). Non-deform helper bones (name containing
    `_NON_DEFORM_NAME_MARKER`, e.g. a chain rig's `foot_extra_l_47`) are
    excluded from the candidate set entirely -- their groups exist but are
    left empty, same as before this change.

    After the raw geometric assignment, weights are smoothed across the
    mesh's own edge topology (`smooth_weight_maps`, `_BLEND_SMOOTHING_*`
    constants) -- a purely geometric nearest-segment pick still produces a
    hard ownership boundary right at a multi-bone junction (neck, shoulder)
    even once every nearby bone is a blend candidate; this pass softens
    that boundary over the mesh surface instead of over raw 3D distance,
    so it can't bleed across disconnected geometry the way a spatial blend
    could.
    """
    bones = list(armature_obj.data.bones)
    candidate_indices = [
        i for i, bone in enumerate(bones)
        if _NON_DEFORM_NAME_MARKER not in bone.name and bone.use_deform
    ]
    if not candidate_indices:
        return 0
    matrix = armature_obj.matrix_world
    candidate_names = [bones[i].name for i in candidate_indices]
    heads = np.asarray(
        [np.asarray(matrix @ bones[i].head_local, dtype=np.float64) for i in candidate_indices],
        dtype=np.float64,
    )
    tails = np.asarray(
        [np.asarray(matrix @ bones[i].tail_local, dtype=np.float64) for i in candidate_indices],
        dtype=np.float64,
    )

    world_vertices = _mesh_world_vertices(mesh_obj)
    bone_indices, weights = nearest_bone_blend_weights(world_vertices, heads, tails)

    vertex_count = bone_indices.shape[0]
    influence_slots = bone_indices.shape[1]
    weight_maps: list[dict[str, float]] = []
    for vertex_index in range(vertex_count):
        vertex_weights: dict[str, float] = {}
        for slot in range(influence_slots):
            candidate_index = int(bone_indices[vertex_index, slot])
            weight = float(weights[vertex_index, slot])
            if candidate_index < 0 or weight <= 1e-6:
                continue
            vertex_weights[candidate_names[candidate_index]] = weight
        weight_maps.append(vertex_weights)

    smoothed_maps = smooth_weight_maps(
        weight_maps,
        _mesh_edge_pairs(mesh_obj),
        factor=_BLEND_SMOOTHING_FACTOR,
        iterations=_BLEND_SMOOTHING_ITERATIONS,
        maximum_influences=_BLEND_SMOOTHING_MAX_INFLUENCES,
    )

    groups = {name: mesh_obj.vertex_groups[name] for name in candidate_names}
    assigned = 0
    for vertex_index, vertex_weights in enumerate(smoothed_maps):
        vertex_assigned = False
        for bone_name, weight in vertex_weights.items():
            if weight <= 1e-6:
                continue
            groups[bone_name].add([vertex_index], weight, "REPLACE")
            vertex_assigned = True
        if vertex_assigned:
            assigned += 1
    return assigned




# Leg chains used to measure "distance to a leg" in the hanging-region
# detector: the whole limb including the foot and toe segments, so a real
# foot reads as ON a leg rather than as material hanging between them.
_HANGING_LEFT_CHAIN = ("thigh_l", "calf_l", "foot_l", "toes_l")
_HANGING_RIGHT_CHAIN = ("thigh_r", "calf_r", "foot_r", "toes_r")
# Arm chains, for the torso-shell detector's arm exclusion: a T-posed
# character's outstretched arms are a huge outer radial cluster in the
# lateral sectors of every chest band, so without excluding them the
# detector fires on every character alike (measured: it did).
_TORSO_ARM_CHAINS = (
    ("upper_arm_l", "lower_arm_l", "hand_l"),
    ("upper_arm_r", "lower_arm_r", "hand_r"),
)
# The spine polyline the torso's cylindrical frame is measured about, and
# (minus `neck`) the bones a torso-shell vertex is re-anchored to by height.
_TORSO_SPINE_CENTRE = ("pelvis", "spine_1", "spine_2", "spine_3", "neck")
_TORSO_SPINE_BONES = ("pelvis", "spine_1", "spine_2", "spine_3")

# The biped template's `root` is a motion/control bone: it has no anatomy,
# it is the parent of `pelvis`, and the family's clips drive it with root
# motion only. It is also, in the template's rest layout, a LONG horizontal
# bone running forward from the pelvis at hip height -- straight through the
# character's abdomen. Blender's bone heat therefore floods the belly and
# the waist of any garment with root weight (measured on a real character:
# 6434 of 35552 vertices carried root weight, 1144 of them root-DOMINANT),
# and that material then follows root motion instead of the spine and
# shears apart in any twisting pose. `root` is not one of the certificate's
# required deforming roles, so leaving it unweighted is safe.
_CONTROL_BONE_NAMES = ("root",)


def disable_control_bone_deform(armature_obj) -> list[str]:
    """Take the rig's control bones out of the skinning candidate set.

    Only a bone that is genuinely a control bone qualifies: named in
    `_CONTROL_BONE_NAMES`, at the top of the hierarchy (no parent), and the
    parent of an anatomical bone (`pelvis`). A chain rig whose real first
    deform bone happens to be called `root` fails the `pelvis` test and is
    left alone.

    Returns the names actually disabled. Must run BEFORE the bind, since
    both binding paths read `use_deform`: Blender's bone heat skips
    non-deform bones, and `ARMATURE_NAME` does not even create vertex groups
    for them.
    """
    disabled: list[str] = []
    for name in _CONTROL_BONE_NAMES:
        bone = armature_obj.data.bones.get(name)
        if bone is None or bone.parent is not None:
            continue
        if not any(child.name == "pelvis" for child in bone.children):
            continue
        if not bone.use_deform:
            continue
        bone.use_deform = False
        disabled.append(name)
    return disabled


def _armature_bone_segment(armature_obj, name: str):
    """World-space (head, tail) for a bone, or None when the rig has no
    such bone (a required-tier-only rig has no `toes_*`, for instance)."""
    bone = armature_obj.data.bones.get(name)
    if bone is None:
        return None
    matrix = armature_obj.matrix_world
    return (
        np.asarray(matrix @ bone.head_local, dtype=np.float64),
        np.asarray(matrix @ bone.tail_local, dtype=np.float64),
    )


# Topology smoothing applied inside a detected region (the crotch bridge,
# the torso shell). Measured on a real crotch-web tear: 8 iterations at
# factor 0.5 took the region's worst edge stretch from 7.99 to 3.49 and its
# p99 from 2.82 to 1.92, where every re-targeting variant made both worse.
# Past ~8 iterations the curve flattens.
_REGION_SMOOTHING_ITERATIONS = 8
_REGION_SMOOTHING_FACTOR = 0.5


def _region_smoothed_weights(mesh_obj, bone_indices, weights, group_names):
    """Topology-smoothed copy of `(bone_indices, weights)`.

    Smoothing runs over the WHOLE mesh (it is a neighbourhood average, and a
    region vertex's neighbours are mostly outside the region -- that is the
    point: the region's ownership is blended continuously into the skin
    around it). The caller decides which rows to keep.
    """
    maps: list[dict[str, float]] = []
    for row in range(bone_indices.shape[0]):
        entry: dict[str, float] = {}
        for slot in range(bone_indices.shape[1]):
            index = int(bone_indices[row, slot])
            value = float(weights[row, slot])
            if index >= 0 and value > 1e-6:
                entry[group_names[index]] = entry.get(group_names[index], 0.0) + value
        maps.append(entry)
    smoothed = smooth_weight_maps(
        maps,
        _mesh_edge_pairs(mesh_obj),
        factor=_REGION_SMOOTHING_FACTOR,
        iterations=_REGION_SMOOTHING_ITERATIONS,
        maximum_influences=_HANGING_MAX_INFLUENCES,
    )
    index_of = {name: index for index, name in enumerate(group_names)}
    out_bones = np.full((len(smoothed), _HANGING_MAX_INFLUENCES), -1, dtype=np.int64)
    out_weights = np.zeros((len(smoothed), _HANGING_MAX_INFLUENCES), dtype=np.float64)
    for row, entry in enumerate(smoothed):
        items = sorted(entry.items(), key=lambda item: -item[1])[:_HANGING_MAX_INFLUENCES]
        total = sum(value for _, value in items)
        if total <= 0.0:
            continue
        for slot, (name, value) in enumerate(items):
            out_bones[row, slot] = index_of.get(name, -1)
            out_weights[row, slot] = value / total
    return out_bones, out_weights


def apply_hanging_region_blend(mesh_obj, armature_obj) -> dict:
    """Re-weight geometry that hangs below the hips instead of sitting on a
    leg -- a robe/skirt shell, a coat tail, the fabric web bridging the two
    legs at the crotch (see `skeleton_naming.detect_hanging_region`).

    Bone-heat gives such material a nearest-wins left/right partition with
    a knife-edge boundary down the midline, so any scissoring pose shears
    it into two rigid halves: a floor-length robe collapses into a flat
    vertical plank below the hem, a crotch web smears into a thin sheet
    trailing the swinging leg. This pass replaces that ownership with a
    smooth lateral blend across the pelvis span, pelvis-anchored at the top
    of the region and handed to the leg pair towards the hem.

    Runs AFTER the primary weighting and BEFORE the final seam stitching
    (the caller re-stitches once this returns, because averaging new
    weights into a coincident group's members can otherwise leave the
    group's rows disagreeing again).

    Humanoid rigs only -- a chain rig (serpent, tail) has no leg pair to
    blend between, and its long body is already handled by the
    distance-along-chain weighting fallback. Returns a provenance dict for
    the bind validation JSON; an empty-ish dict when nothing applied.
    """
    required = ("pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r", "head")
    missing = [name for name in required if armature_obj.data.bones.get(name) is None]
    if missing:
        return {"applied": False, "reason": f"missing bones: {missing}"}

    pelvis_head = _armature_bone_segment(armature_obj, "pelvis")[0]
    head_head = _armature_bone_segment(armature_obj, "head")[0]
    thigh_left_head = _armature_bone_segment(armature_obj, "thigh_l")[0]
    thigh_right_head = _armature_bone_segment(armature_obj, "thigh_r")[0]
    calf_left_head = _armature_bone_segment(armature_obj, "calf_l")[0]
    calf_right_head = _armature_bone_segment(armature_obj, "calf_r")[0]

    up_vector = head_head - pelvis_head
    up_axis = int(np.argmax(np.abs(up_vector)))
    up_sign = 1.0 if up_vector[up_axis] > 0 else -1.0
    lateral_vector = thigh_left_head - thigh_right_head
    lateral_axis = int(np.argmax(np.abs(lateral_vector)))
    if lateral_axis == up_axis:
        return {"applied": False, "reason": "degenerate hip span"}

    hip_height = max(thigh_left_head[up_axis] * up_sign, thigh_right_head[up_axis] * up_sign)
    lateral_left = float(thigh_left_head[lateral_axis])
    lateral_right = float(thigh_right_head[lateral_axis])
    hip_half_span = abs(lateral_left - lateral_right) / 2.0
    hip_midline = (lateral_left + lateral_right) / 2.0
    knee_height = (
        calf_left_head[up_axis] * up_sign + calf_right_head[up_axis] * up_sign
    ) / 2.0
    foot_left = _armature_bone_segment(armature_obj, "foot_l")
    foot_right = _armature_bone_segment(armature_obj, "foot_r")
    ankle_heads = [
        segment[0][up_axis] * up_sign
        for segment in (foot_left, foot_right) if segment is not None
    ]
    # No modelled ankle (a required-tier-only rig): put the handover at the
    # bottom of the leg chain so the foot level simply never activates.
    ankle_height = sum(ankle_heads) / len(ankle_heads) if ankle_heads else knee_height

    left_segments = [
        segment for segment in
        (_armature_bone_segment(armature_obj, name) for name in _HANGING_LEFT_CHAIN)
        if segment is not None
    ]
    right_segments = [
        segment for segment in
        (_armature_bone_segment(armature_obj, name) for name in _HANGING_RIGHT_CHAIN)
        if segment is not None
    ]

    # Torso frame: the spine polyline (for the cylindrical coordinates the
    # torso-shell detector measures in) and the arm chains it excludes.
    depth_axis = next(axis for axis in (0, 1, 2) if axis not in (up_axis, lateral_axis))
    spine_heads = [
        (name, _armature_bone_segment(armature_obj, name)) for name in _TORSO_SPINE_CENTRE
    ]
    spine_heads = [(name, segment[0]) for name, segment in spine_heads if segment is not None]
    spine_segments = [
        (spine_heads[index][1], spine_heads[index + 1][1]) for index in range(len(spine_heads) - 1)
    ]
    arm_segments = [
        segment
        for chain in _TORSO_ARM_CHAINS
        for segment in (_armature_bone_segment(armature_obj, name) for name in chain)
        if segment is not None
    ]
    clavicle_left = _armature_bone_segment(armature_obj, "clavicle_l")
    clavicle_right = _armature_bone_segment(armature_obj, "clavicle_r")
    upper_arm_left = _armature_bone_segment(armature_obj, "upper_arm_l")
    upper_arm_right = _armature_bone_segment(armature_obj, "upper_arm_r")
    shoulder_heights = [
        segment[0][up_axis] * up_sign
        for segment in (upper_arm_left, upper_arm_right) if segment is not None
    ]
    shoulder_height = min(shoulder_heights) if shoulder_heights else None
    clavicle_heights = [
        segment[0][up_axis] * up_sign
        for segment in (clavicle_left, clavicle_right) if segment is not None
    ]
    clavicle_height = sum(clavicle_heights) / len(clavicle_heights) if clavicle_heights else None
    # Signed (left - right), like `lateral_left - lateral_right` above: the
    # torso-shell blend splits its clavicle share by the sign, and l/r come
    # from the character's facing, so clavicle_l may sit at the lower
    # lateral coordinate.
    clavicle_span = (
        float(clavicle_left[0][lateral_axis]) - float(clavicle_right[0][lateral_axis])
        if clavicle_left is not None and clavicle_right is not None
        else lateral_left - lateral_right
    )

    world_vertices = _mesh_world_vertices(mesh_obj)
    region = detect_hanging_region(
        world_vertices,
        up_axis=up_axis,
        up_sign=up_sign,
        lateral_axis=lateral_axis,
        hip_height=float(hip_height),
        hip_half_span=float(hip_half_span),
        hip_midline=float(hip_midline),
        left_segments=left_segments,
        right_segments=right_segments,
    )
    bridge = detect_bridge_region(
        world_vertices,
        up_axis=up_axis,
        up_sign=up_sign,
        lateral_axis=lateral_axis,
        hip_height=float(hip_height),
        hip_half_span=float(hip_half_span),
        hip_midline=float(hip_midline),
        left_segments=left_segments,
        right_segments=right_segments,
    )
    torso = TorsoShellRegion(
        mask=np.zeros(world_vertices.shape[0], dtype=bool),
        strength=np.zeros(world_vertices.shape[0], dtype=np.float64),
        smoothing_strength=np.zeros(world_vertices.shape[0], dtype=np.float64),
        core_radius=0.0,
        region_top=float(hip_height),
        region_bottom=float(hip_height),
        band_shell_counts=[],
        widest_gap=0.0,
    )
    if shoulder_height is not None and len(spine_heads) >= 2:
        torso = detect_torso_shell(
            world_vertices,
            up_axis=up_axis,
            up_sign=up_sign,
            lateral_axis=lateral_axis,
            depth_axis=depth_axis,
            hip_height=float(hip_height),
            shoulder_height=float(shoulder_height),
            spine_heights=[float(head[up_axis] * up_sign) for _, head in spine_heads],
            spine_lateral=[float(head[lateral_axis]) for _, head in spine_heads],
            spine_depth=[float(head[depth_axis]) for _, head in spine_heads],
            spine_segments=spine_segments,
            arm_segments=arm_segments,
        )

    provenance = {
        "applied": False,
        "mode": region.mode,
        "enclosure_depth": round(region.enclosure_depth, 4),
        "band_gaps": region.band_gaps,
        "hip_height": round(float(hip_height), 5),
        "hip_half_span": round(float(hip_half_span), 5),
        "knee_height": round(float(knee_height), 5),
        "ankle_height": round(float(ankle_height), 5),
        "shoulder_height": round(float(shoulder_height), 5) if shoulder_height is not None else None,
        "region_top": round(region.region_top, 5),
        "region_bottom": round(region.region_bottom, 5),
        "vertices": int(np.count_nonzero(region.mask)),
        "torso_shell_vertices": int(np.count_nonzero(torso.mask)),
        "bridge_vertices": int(np.count_nonzero(bridge.mask)),
        "mesh_vertices": int(world_vertices.shape[0]),
        "torso_shell": {
            "vertices": int(np.count_nonzero(torso.mask)),
            "core_radius": round(float(torso.core_radius), 5),
            "widest_gap": float(torso.widest_gap),
            "band_shell_counts": list(torso.band_shell_counts),
            "region_top": round(float(torso.region_top), 5),
            "region_bottom": round(float(torso.region_bottom), 5),
            "radial_gap_threshold": _TORSO_RADIAL_GAP,
            "ceiling": _TORSO_CEILING,
            "shoulder_share": _TORSO_SHOULDER_SHARE,
            "spine_anchors": [name for name, _ in spine_heads if name in _TORSO_SPINE_BONES],
        },
        "bridge": {
            "vertices": int(np.count_nonzero(bridge.mask)),
            "region_top": round(float(bridge.region_top), 5),
            "region_bottom": round(float(bridge.region_bottom), 5),
            "corridor": _BRIDGE_CORRIDOR,
            "balance_threshold": _BRIDGE_BALANCE,
            "clearance_threshold": _BRIDGE_CLEARANCE,
            "pelvis_floor": _BRIDGE_PELVIS_FLOOR,
            "ceiling": _BRIDGE_CEILING,
            "smoothing_iterations": _REGION_SMOOTHING_ITERATIONS,
            "smoothing_factor": _REGION_SMOOTHING_FACTOR,
        },
    }
    if provenance["vertices"] == 0 and provenance["torso_shell_vertices"] == 0 \
            and provenance["bridge_vertices"] == 0:
        return provenance

    bone_indices, weights, group_names = _mesh_group_weights_sparse(mesh_obj)
    # The blended result can carry up to 4 influences even where the
    # incoming weighting carried fewer, so widen the sparse arrays first.
    width = bone_indices.shape[1]
    if width < _HANGING_MAX_INFLUENCES:
        pad = _HANGING_MAX_INFLUENCES - width
        bone_indices = np.concatenate(
            [bone_indices, np.full((bone_indices.shape[0], pad), -1, dtype=np.int64)], axis=1
        )
        weights = np.concatenate(
            [weights, np.zeros((weights.shape[0], pad), dtype=np.float64)], axis=1
        )

    def group_index(name: str) -> int:
        return group_names.index(name) if name in group_names else -1

    # The three passes run in sequence on the same arrays, each mixing its
    # own target shape into whatever the previous one left. Where they
    # overlap (a coat tail is both hanging and bridging) the later, more
    # specific rule is the one that gets the last word.
    blended_bones, blended_weights = hanging_region_blend_weights(
        world_vertices,
        region,
        bone_indices,
        weights,
        up_axis=up_axis,
        up_sign=up_sign,
        lateral_axis=lateral_axis,
        hip_height=float(hip_height),
        hip_half_span=float(hip_half_span),
        hip_midline=float(hip_midline),
        lateral_left=lateral_left,
        lateral_right=lateral_right,
        knee_height=float(knee_height),
        ankle_height=float(ankle_height),
        pelvis_bone=group_index("pelvis"),
        thigh_left_bone=group_index("thigh_l"),
        thigh_right_bone=group_index("thigh_r"),
        calf_left_bone=group_index("calf_l"),
        calf_right_bone=group_index("calf_r"),
        foot_left_bone=group_index("foot_l"),
        foot_right_bone=group_index("foot_r"),
        max_influences=_HANGING_MAX_INFLUENCES,
    )
    if int(np.count_nonzero(torso.mask)):
        anchors = [(name, head) for name, head in spine_heads if name in _TORSO_SPINE_BONES]
        blended_bones, blended_weights = torso_shell_blend_weights(
            world_vertices,
            torso,
            blended_bones,
            blended_weights,
            up_axis=up_axis,
            up_sign=up_sign,
            lateral_axis=lateral_axis,
            spine_bones=[group_index(name) for name, _ in anchors],
            spine_bone_heights=[float(head[up_axis] * up_sign) for _, head in anchors],
            clavicle_left_bone=group_index("clavicle_l"),
            clavicle_right_bone=group_index("clavicle_r"),
            clavicle_height=clavicle_height,
            clavicle_span=float(clavicle_span),
            lateral_midline=float(hip_midline),
            max_influences=_HANGING_MAX_INFLUENCES,
        )
        smoothed_bones, smoothed_weights = _region_smoothed_weights(
            mesh_obj, blended_bones, blended_weights, group_names
        )
        rows = np.nonzero(torso.mask)[0]
        blended_bones, blended_weights = mix_smoothed_weights(
            blended_bones,
            blended_weights,
            smoothed_bones,
            smoothed_weights,
            rows,
            torso.smoothing_strength[rows],
            max_influences=_HANGING_MAX_INFLUENCES,
        )
    if int(np.count_nonzero(bridge.mask)):
        blended_bones, blended_weights = bridge_blend_weights(
            world_vertices,
            bridge,
            blended_bones,
            blended_weights,
            up_axis=up_axis,
            up_sign=up_sign,
            lateral_axis=lateral_axis,
            hip_height=float(hip_height),
            hip_half_span=float(hip_half_span),
            hip_midline=float(hip_midline),
            pelvis_bone=group_index("pelvis"),
            thigh_left_bone=group_index("thigh_l"),
            thigh_right_bone=group_index("thigh_r"),
            lateral_left=lateral_left,
            lateral_right=lateral_right,
            max_influences=_HANGING_MAX_INFLUENCES,
        )
        # ... and then flatten the weight GRADIENT across the bridge, which
        # is what the sheet actually tears along. Re-targeting decides WHO
        # drives a vertex; this decides how fast that answer is allowed to
        # change from one vertex to the next, and on a real crotch web it is
        # worth roughly twice as much as the re-target (see
        # `mix_smoothed_weights`).
        smoothed_bones, smoothed_weights = _region_smoothed_weights(
            mesh_obj, blended_bones, blended_weights, group_names
        )
        rows = np.nonzero(bridge.mask)[0]
        blended_bones, blended_weights = mix_smoothed_weights(
            blended_bones,
            blended_weights,
            smoothed_bones,
            smoothed_weights,
            rows,
            bridge.smoothing_strength[rows],
            max_influences=_HANGING_MAX_INFLUENCES,
        )

    changed = np.any(blended_bones != bone_indices, axis=1) | np.any(
        np.abs(blended_weights - weights) > 1e-9, axis=1
    )
    changed_indices = np.nonzero(changed)[0]
    if changed_indices.size == 0:
        return provenance

    _rewrite_mesh_group_weights(mesh_obj, blended_bones, blended_weights, group_names)
    moved = np.sum(np.abs(blended_weights - weights), axis=1)[changed_indices]
    provenance.update({
        "applied": True,
        "rewritten_vertices": int(changed_indices.size),
        "mean_weight_delta": round(float(np.mean(moved)), 4),
        "maximum_weight_delta": round(float(np.max(moved)), 4),
    })
    if np.any(region.mask):
        provenance["mean_strength"] = round(float(np.mean(region.strength[region.mask])), 4)
    if np.any(torso.mask):
        provenance["torso_shell"]["mean_strength"] = round(
            float(np.mean(torso.strength[torso.mask])), 4
        )
    if np.any(bridge.mask):
        provenance["bridge"]["mean_strength"] = round(
            float(np.mean(bridge.strength[bridge.mask])), 4
        )
    return provenance
def _unirig_bind_from_prediction(spec: dict, meshes: list) -> tuple[np.ndarray, list, list[str], np.ndarray | None, dict, bool]:
    """Legacy path: derive the skeleton from a UniRig prediction npz
    (`spec["skeleton_npz"]`/`spec["raw_npz"]`) -- semantic naming, the
    medial-axis/densify chain fallbacks, and stored-name-indirected
    position overrides all apply here. Unchanged behaviourally from before
    `source_rig` existed.

    Returns (joints, parents, names, tails, facing_report, clean_linear_chain).
    """
    skeleton = np.load(spec["skeleton_npz"], allow_pickle=True)
    raw = np.load(spec["raw_npz"], allow_pickle=True) if spec.get("raw_npz") else skeleton

    # Undo the inference-side cube normalization exactly as AugmentAffine
    # applied it: bounds over the mesh the model actually saw.
    raw_vertices = np.asarray(raw["vertices"], dtype=np.float64)
    bmin = raw_vertices.min(axis=0)
    bmax = raw_vertices.max(axis=0)
    scale = float(np.max((bmax - bmin) / 2.0))
    center = (bmax + bmin) / 2.0

    joints = np.asarray(skeleton["joints"], dtype=np.float64) * scale + center
    tails = None
    if "tails" in skeleton.files:
        raw_tails = np.asarray(skeleton["tails"], dtype=np.float64)
        if raw_tails.size:
            tails = raw_tails * scale + center
    parents = list(skeleton["parents"])

    def mesh_vertices_lazy() -> np.ndarray:
        # Only called when no foot has a toe child (see
        # detect_forward_from_feet's fallback path) -- bulk vertex extraction
        # is expensive for dense meshes, so build it lazily and pull raw
        # local coordinates in one call instead of a per-vertex Python loop.
        all_world_vertices = []
        for mesh_obj in meshes:
            vertex_count = len(mesh_obj.data.vertices)
            if vertex_count == 0:
                continue
            local = np.empty(vertex_count * 3, dtype=np.float64)
            mesh_obj.data.vertices.foreach_get("co", local)
            local = local.reshape(vertex_count, 3)
            matrix = np.asarray(mesh_obj.matrix_world, dtype=np.float64)
            homogeneous = np.concatenate(
                [local, np.ones((vertex_count, 1), dtype=np.float64)], axis=1
            )
            all_world_vertices.append((matrix @ homogeneous.T).T[:, :3])
        if not all_world_vertices:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(all_world_vertices, axis=0)

    profile_hint = spec.get("profile_hint")
    names, facing_report = semantic_joint_names(joints, parents, mesh_vertices_lazy, profile_hint=profile_hint)

    resolved_mesh_vertices: np.ndarray | None = None

    def get_mesh_vertices() -> np.ndarray:
        nonlocal resolved_mesh_vertices
        if resolved_mesh_vertices is None:
            resolved_mesh_vertices = mesh_vertices_lazy()
        return resolved_mesh_vertices

    def refresh_facing_after_skeleton_replacement(new_names: list[str], source: str) -> dict:
        head_index = new_names.index("head")
        tail_index = 0 if head_index != 0 else len(new_names) - 1
        forward_vector = joints[head_index] - joints[tail_index]
        forward_norm = float(np.linalg.norm(forward_vector))
        return {
            **facing_report,
            "forward": [round(float(v), 4) for v in (
                forward_vector / forward_norm if forward_norm > 1e-9 else forward_vector
            )],
            "profile": "chain",
            "skeleton_source": source,
        }

    chain_joints_setting = spec.get("chain_joints")
    target_joint_count = int(chain_joints_setting) if chain_joints_setting else 12
    skeleton_source = "unirig"
    # True once `names`/`joints` are a plain linear chain (index 0 =
    # spine_00/tail ... last index = head, no interleaved spurs) -- needed
    # below to know the distance-along-chain weighting fallback's ordered
    # bone list is simply `names` itself.
    clean_linear_chain = False

    # A predicted skeleton can be too degenerate for densify_chain to fix:
    # densify_chain only resamples *between* a candidate's existing
    # joints, so a UniRig prediction bunched into a short cluster (a
    # "star" of spurs spanning a fraction of a long body) stays short no
    # matter how densely it's resampled. Detect that case and build the
    # chain from mesh geometry alone instead of trusting the prediction's
    # topology at all.
    #
    # This only ever runs for an *already chain-profiled* skeleton (a hint
    # of "chain", or automatic detection finding a genuinely limbless
    # topology) -- never merely because the mesh looks elongated. A
    # T-posed humanoid's arm span routinely exceeds its height, which is
    # normal geometry, not evidence of a chain body plan; mesh elongation
    # alone used to trigger this fallback even when the skeleton had
    # already been correctly classified "humanoid" (real arm/leg
    # branches), silently discarding a good rig. detect_rig_profile
    # already checks topology (real branches) before elongation, and a
    # profile_hint of "humanoid" overrides detection entirely -- so a
    # "humanoid" profile here means the skeleton genuinely has limbs (or
    # was explicitly told to), and must never be replaced.
    use_geometric_fallback = False
    if facing_report["profile"] == "chain":
        candidate_score = score_chain_candidate(joints, parents)
        extent_ratio = chain_candidate_extent_ratio(joints, parents, get_mesh_vertices())
        if candidate_score.main_chain_length < 6 or extent_ratio < 0.6:
            use_geometric_fallback = True
            log(
                f"predicted chain too degenerate to densify (main_chain_length="
                f"{candidate_score.main_chain_length}, extent_ratio={extent_ratio:.2f} of mesh "
                "principal-axis extent); falling back to a mesh-only medial-axis chain"
            )

    if use_geometric_fallback:
        try:
            fallback = medial_axis_chain(get_mesh_vertices(), target_joint_count)
        except ValueError as fallback_error:
            log(f"WARNING: medial-axis fallback failed ({fallback_error}); binding the raw prediction instead")
        else:
            log(f"medial-axis chain: {len(fallback.joints)} joints built from mesh geometry alone")
            joints = fallback.joints
            parents = list(fallback.parents)
            names = fallback.names
            tails = None
            skeleton_source = "medial-axis"
            clean_linear_chain = True
            facing_report = refresh_facing_after_skeleton_replacement(names, skeleton_source)
    elif facing_report["profile"] == "chain" and chain_joints_setting:
        # Densify a sparse-but-usable chain-profile prediction (a serpent,
        # tail, tentacle, ...): UniRig's raw joint count is often too low,
        # and the joints it does predict aren't always on the mesh's true
        # medial line. Resample the main chain via Catmull-Rom, snap each
        # new joint to its local mesh cross-section centroid, and replace
        # the predicted skeleton with this denser, cleaner one before
        # binding. Spurs and the original tails are dropped.
        try:
            densified = densify_chain(joints, parents, target_joint_count, mesh_vertices_lazy)
        except ValueError as densify_error:
            log(f"WARNING: could not densify chain ({densify_error}); binding the raw prediction instead")
        else:
            log(
                f"densified chain: {len(joints)} predicted joints -> "
                f"{len(densified.joints)} joints"
            )
            joints = densified.joints
            parents = list(densified.parents)
            names = densified.names
            tails = None
            clean_linear_chain = True
            facing_report = refresh_facing_after_skeleton_replacement(names, skeleton_source)

    facing_report.setdefault("skeleton_source", skeleton_source)
    # Overrides are authored against a prior bind's own joint
    # indices/names; densifying or replacing the skeleton entirely changes
    # the joint set (different count, different identities), so there is
    # no sound way to carry them forward onto it.
    if facing_report["skeleton_source"] != "unirig" and spec.get("apply_positions"):
        log(
            "WARNING: ignoring apply_positions overrides -- they were authored "
            "against a different (unmodified UniRig) joint set"
        )
        spec = {**spec, "apply_positions": None}

    # Hand adjustments from the review viewport: absolute Blender-space joint
    # positions keyed by semantic bone name, authored against whichever
    # bind's names the review UI last displayed. Resolve them through that
    # prior bind's stored name->index map (persisted alongside this
    # lineage's bind inputs) rather than this bind's freshly derived names,
    # which can differ from the prior names whenever the naming algorithm
    # changes between binds (see resolve_position_overrides). Undriven
    # joints keep the predicted positions.
    overrides = spec.get("apply_positions") or {}
    names_path = _names_json_path(spec)
    if overrides:
        stored_names = _load_stored_names(names_path)
        resolution = resolve_position_overrides(overrides, names, stored_names)
        if resolution.used_fallback:
            log(
                "WARNING: no stored joint-name map found for this bind's inputs; "
                "matching position overrides by freshly derived names, which is "
                "only correct if bone naming has not changed since the override "
                "was authored"
            )
        if resolution.unresolved:
            log(f"WARNING: ignoring overrides for unresolved joint name(s): {resolution.unresolved}")
        original_heads = np.asarray(skeleton["joints"], dtype=np.float64) * scale + center
        for index, new_head in resolution.resolved.items():
            if tails is not None:
                # carry the bone tail along so direction/length is kept
                tails[index] = tails[index] + (new_head - original_heads[index])
            joints[index] = new_head
        log(f"applied {len(resolution.resolved)} hand-adjusted joint position(s)")

    # Persist this bind's own names for the next rebind of the same
    # character to resolve its overrides against.
    _write_stored_names(names_path, names, facing_report["profile"])

    return joints, parents, names, tails, facing_report, clean_linear_chain



def _extrapolate_chain_leaf_tails(joints: np.ndarray, parents: list, tails: np.ndarray | None) -> np.ndarray:
    """For every leaf bone (no children) with a parent, point its tail along
    parent -> leaf, continuing the chain. A chain's free end has no child to
    derive a tail from; the glTF importer's guess for it is a +Z stub, which
    slither_bake rightly rejects as "too vertical to bend about world-up"."""
    tails = np.array(tails, dtype=np.float64) if tails is not None else np.array(joints, dtype=np.float64) + np.array([0.0, 0.0, 0.05])
    has_child = {int(parent) for parent in parents if parent is not None and parent >= 0}
    for index, parent in enumerate(parents):
        if index in has_child or parent is None or parent < 0:
            continue
        direction = joints[index] - joints[int(parent)]
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        tails[index] = joints[index] + direction / length * max(length * 0.9, 0.02)
    return tails


def _unirig_bind_from_source_rig(
    spec: dict, source_rig: str
) -> tuple[np.ndarray, list, list[str], np.ndarray | None, dict, bool, list[str], list[str]]:
    """Rig-adjust path: rebind starting from the prior rig's own armature
    (`source_rig`) instead of re-deriving a skeleton from a UniRig
    prediction. Semantic naming, the medial-axis/densify fallbacks, and
    the stored-name-indirected override resolution used by the prediction
    path are all irrelevant here -- this is a real, previously-bound
    skeleton, not a fresh guess, so its own names are used directly and
    unconditionally.

    Returns (joints, parents, names, tails, facing_report, clean_linear_chain,
    removed_joints, refused_joints).
    """
    joints, parents, names, tails = _import_source_rig_skeleton(source_rig)

    names_path = _names_json_path(spec)
    stored_profile = _load_stored_profile(names_path)
    profile = stored_profile or detect_profile_from_names(names)
    clean_linear_chain = (
        profile == "chain"
        and (not parents or parents[0] is None)
        and all(parents[i] == i - 1 for i in range(1, len(parents)))
    )
    facing_report = {
        "profile": profile,
        "profile_source": "source-rig",
        "skeleton_source": "source-rig",
        "forward": [0.0, 0.0, 0.0],
        "source": "source-rig",
        "warning": None,
    }
    log(
        f"source_rig skeleton: profile={profile} bones={len(names)} "
        f"clean_linear_chain={clean_linear_chain}"
    )

    # Hand adjustments are authored by name directly against this same
    # source rig's own bone names -- there is no separate "freshly
    # derived" naming pass here that could disagree with what the review
    # UI displayed, so no stored-names indirection is needed (contrast
    # `resolve_position_overrides`, used by the prediction path above).
    overrides = spec.get("apply_positions") or {}
    if overrides:
        applied = 0
        unresolved: list[str] = []
        for name, value in overrides.items():
            if name in names:
                joints[names.index(name)] = np.asarray(value, dtype=np.float64)
                applied += 1
            else:
                unresolved.append(name)
        if unresolved:
            log(f"WARNING: ignoring overrides for unresolved joint name(s): {unresolved}")
        log(f"applied {applied} hand-adjusted joint position(s)")

    removed_joints: list[str] = []
    refused_joints: list[str] = []
    remove_requested = [str(name) for name in (spec.get("remove_joints") or [])]
    if remove_requested:
        removal = remove_joints_reparent(joints, parents, names, tails, remove_requested)
        joints, parents, names, tails = removal.joints, removal.parents, removal.names, removal.tails
        removed_joints, refused_joints = removal.removed, removal.refused
        log(f"removed {len(removed_joints)} joint(s): {removed_joints}; refused: {refused_joints}")

    if profile == "chain":
        tails = _extrapolate_chain_leaf_tails(joints, parents, tails)

    # Persist this bind's own names for the next rebind of the same
    # character to resolve its overrides against.
    _write_stored_names(names_path, names, facing_report["profile"])

    return joints, parents, names, tails, facing_report, clean_linear_chain, removed_joints, refused_joints


def unirig_bind(spec: dict) -> None:
    clear_scene()

    source_rig = spec.get("source_rig")
    removed_joints: list[str] = []
    refused_joints: list[str] = []

    if source_rig:
        # The source rig's armature must be read from an otherwise-empty
        # scene, and its objects deleted again, before the clean mesh is
        # imported -- see `_import_source_rig_skeleton`.
        (
            joints, parents, names, tails, facing_report, clean_linear_chain,
            removed_joints, refused_joints,
        ) = _unirig_bind_from_source_rig(spec, source_rig)
        meshes = import_meshes(spec["input"])
    else:
        meshes = import_meshes(spec["input"])
        joints, parents, names, tails, facing_report, clean_linear_chain = _unirig_bind_from_prediction(spec, meshes)

    # N3: humanoid-only template-conformance build (PIPELINE_POLICY.md S3).
    # The chain (serpent) path is untouched -- it never enters this branch.
    # This maps the resolved semantic role names onto the biped family
    # template's exact bone set/hierarchy/rest-direction/roll, synthesizing
    # `root`/`clavicle_l`/`clavicle_r` (see
    # skeleton_naming.build_template_conformant_skeleton). A required bone
    # that cannot be placed at all (no UniRig prediction, no synthesis
    # rule) blames upstream (S2/S0) rather than emitting a looser fit: the
    # bind fails outright, after best-effort recording the finding in the
    # validation JSON so the failure is diagnosable.
    #
    # The template's OPTIONAL tier (45 bones: fingers, toes, head_tip) is
    # grown from the placed hands/feet/head rather than predicted -- UniRig
    # emits no finger or toe joints at all, so without this the character
    # gets mitts and every clip's finger motion is discarded. The mesh is
    # handed in so each chain is scaled to this character's own hand/foot
    # extent (see `skeleton_naming.optional_chain_scale`).
    # `settings.skip_optional_bones` on the auto-rig / rig-adjust job turns
    # this off and restores the 21-bone behaviour.
    template_conformance: list[dict] = []
    template_synthesized_bones: list[str] = []
    template_synthesized_optional: list[str] = []
    template_optional_chain_scales: list[dict] = []
    optional_joint_containment: dict | None = None
    distal_refinement: list[dict] = []
    skip_optional_bones = bool(spec.get("skip_optional_bones", False))
    skip_distal_refinement = bool(spec.get("skip_distal_refinement", False))
    # `settings.unirig.skip_hanging_blend` turns off the hanging-region
    # weighting post-pass, restoring the pure nearest-wins weights.
    skip_hanging_blend = bool(spec.get("skip_hanging_blend", False))
    rolls: list[float] | None = None
    trust_tails = False
    if facing_report.get("profile") == "humanoid":
        template = load_template("biped")
        mesh_vertices = _mesh_world_vertices(meshes[0])

        # S3 joint PLACEMENT, before anything is built on top of it. UniRig
        # fits the wrist about halfway up the forearm (measured on zix:
        # `hand_l` at x=0.237 against a mesh arm that runs to x=0.497 with a
        # modelled hand at x 0.39-0.50), so the anchor is snapped onto the
        # mesh's own distal anatomy first -- see
        # `skeleton_naming.refine_distal_anchor`. This runs BEFORE the
        # template build so the tails, the finger/toe chains and their
        # per-chain mesh measurement all derive from the corrected anchor;
        # doing it afterwards would grow every finger from the wrong place
        # and then move the palm out from under them.
        #
        # Independent of `skip_optional_bones`: a wrist in the right place
        # matters to the bone-heat weights whether or not fingers are grown.
        # A rig-adjust rebind's hand-placed positions are protected -- the
        # detector must never argue with a human who moved that joint.
        if not skip_distal_refinement:
            joints, refinements = refine_distal_anchors(
                joints, names, mesh_vertices,
                protected=set((spec.get("apply_positions") or {}).keys()),
            )
            distal_refinement = [distal_refinement_json(report) for report in refinements]
            log(
                "distal refinement: "
                + " ".join(
                    f"{report.anchor}={report.reason}"
                    f"(moved={report.moved:.4f} extent={report.limb_extent:.4f} "
                    f"bulge={report.bulge_ratio:.2f} conf={report.confidence:.2f})"
                    for report in refinements
                )
            )

        build = build_template_conformant_skeleton(
            names, joints, template,
            skip_optional_bones=skip_optional_bones,
            mesh_vertices=None if skip_optional_bones else mesh_vertices,
        )
        if build.missing_required:
            if spec.get("validation"):
                validation_path = Path(spec["validation"])
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                validation_path.write_text(json.dumps({
                    "backend": "unirig-auto-rig",
                    "profile": "humanoid",
                    "template_family": template.family,
                    "template_conformance_error": "missing_required_bones",
                    "template_missing_required_bones": build.missing_required,
                    "distal_refinement": distal_refinement,
                    "facing": facing_report,
                }, indent=2), encoding="utf-8")
            raise RuntimeError(
                "cannot fit biped template: required bone(s) "
                f"{build.missing_required} have no UniRig prediction and no synthesis "
                "rule (PIPELINE_POLICY.md S3: 'cannot fit' blames S2/S0, never a looser fit)"
            )
        parent_names = build.parents
        names, joints, tails, rolls = build.names, build.heads, build.tails, build.rolls
        name_to_index = {name: index for index, name in enumerate(names)}
        parents = [name_to_index[parent] if parent is not None else None for parent in parent_names]
        trust_tails = True
        template_synthesized_bones = build.synthesized
        template_synthesized_optional = list(build.synthesized_optional)
        template_optional_chain_scales = [
            {"anchor": anchor, "scale": round(float(scale), 6), "source": source}
            for anchor, scale, source in build.optional_chain_scales
        ]

        template_conformance = template_conformance_report(
            names, parent_names, joints, tails, rolls,
            family=template.family, template=template,
        )
        # The N2 caveat: leg rest directions are side-identical in the
        # template, so the direction-based mirrored_left_right check above
        # cannot catch a leg-only mirror. Extend with a position-based
        # check -- but only when facing was actually measured (a
        # rig-adjust rebind's facing_report carries no measurement at all;
        # see _unirig_bind_from_source_rig).
        if facing_report.get("source") in CONFIDENT_FACING_SOURCES:
            forward = np.asarray(facing_report.get("forward", [0.0, 0.0, 0.0]), dtype=np.float64)
            left = left_from_forward(forward)
            reference_index = name_to_index.get("pelvis", name_to_index.get("root"))
            reference_point = joints[reference_index] if reference_index is not None else np.zeros(3)
            template_conformance = template_conformance + template_side_conformance_findings(
                names, joints, template, left, reference_point,
            )
        log(
            f"template conformance: family={template.family} bones={len(names)} "
            f"synthesized={template_synthesized_bones} findings={len(template_conformance)}"
        )

        # Where did the grown optional joints actually land? A stylized mesh
        # with mitt hands has no finger geometry to sit inside, so a poor
        # containment fraction here is a fact to record, not a failure to
        # raise on: the bones are what let a clip's finger channels drive
        # anything at all, and the optional tier is explicitly allowed to be
        # imperfect (PIPELINE_POLICY.md S3 treats the tier as optional).
        if template_synthesized_optional:
            optional_index = {name: index for index, name in enumerate(names)}
            optional_names = [
                name for name in template_synthesized_optional if name in optional_index
            ]
            triangles = _mesh_world_triangles(meshes[0])
            optional_joint_containment = measure_joint_containment(
                optional_names,
                joints[[optional_index[name] for name in optional_names]],
                triangles,
                model_height=build.skeleton_height,
            )
            log(
                "optional-tier synthesis: "
                f"bones={len(optional_names)} scales={template_optional_chain_scales} "
                f"inside_fraction={optional_joint_containment.get('joints_inside_mesh_fraction')} "
                f"outside={optional_joint_containment.get('joints_outside_mesh_count')}"
            )

    armature = build_armature(joints, tails, parents, names, rolls=rolls, trust_tails=trust_tails)
    if facing_report.get("profile") == "humanoid":
        # Carried through as glTF node `extras` on export below so the
        # rig-certificate gate can read it back from the exported GLB alone
        # (backend.app.rig_quality/quality_policy), with no re-derivation
        # (and no risk of re-deriving it wrong from re-parsed geometry).
        armature["template_conformance"] = template_conformance
        armature["template_conformance_family"] = "biped"
        if template_synthesized_bones:
            armature["template_synthesized_bones"] = template_synthesized_bones
        if distal_refinement:
            armature["distal_refinement"] = distal_refinement
    mesh = meshes[0]
    log(f"armature bones={len(armature.data.bones)} mesh verts={len(mesh.data.vertices)}")

    # Blender background-mode quirk: the mesh must be the active object when
    # selected, otherwise select_set silently fails.
    bpy.context.view_layer.objects.active = mesh
    for obj in meshes:
        obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bone_count = len(armature.data.bones)

    def assigned_vertex_count(mesh_obj) -> int:
        return sum(1 for v in mesh_obj.data.vertices if len(v.groups) > 0)

    def clear_groups(mesh_obj) -> None:
        for group in list(mesh_obj.vertex_groups):
            mesh_obj.vertex_groups.remove(group)

    # Vertex indices whose weights seam-stitching has rewritten so far this
    # bind -- a running union across every stitching call below, since a
    # later path (the chain fallback, the nearest-joint fallback) can touch
    # vertices an earlier call already stitched, or leave others alone.
    seam_stitched_indices: set[int] = set()

    def stitch_and_log(stage: str) -> None:
        changed = apply_seam_stitching(mesh)
        if changed.size:
            seam_stitched_indices.update(int(i) for i in changed)
            log(f"seam-stitched {changed.size} coincident vertex(es) after {stage}")

    # Control bones never carry skin. `root` in particular is a long
    # horizontal bone running forward through the abdomen, so leaving it in
    # the candidate set hands the belly and the waist of any garment to a
    # bone that carries root motion instead of spine motion -- and that
    # material then shears apart in every twisting pose (see
    # `disable_control_bone_deform`).
    control_bones_disabled = disable_control_bone_deform(armature)
    if control_bones_disabled:
        log(f"control bones excluded from skinning: {control_bones_disabled}")

    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    assigned = assigned_vertex_count(mesh)
    if assigned > len(mesh.data.vertices) * 0.1:
        weighting = "blender-bone-heat"
        stitch_and_log(weighting)
    else:
        log("bone-heat produced no weights; falling back to nearest-bone-blend weighting")
        clear_groups(mesh)
        bpy.context.view_layer.objects.active = armature
        # ARMATURE_NAME creates one empty vertex group per bone plus the
        # armature modifier, with no weights of its own -- weights are
        # filled below by apply_nearest_bone_blend_weights, a geometric
        # point-to-bone-segment scheme that cannot reach past a
        # neighbouring joint the way ARMATURE_ENVELOPE's capsule radius
        # can on slender limbs (see that function's docstring).
        bpy.ops.object.parent_set(type="ARMATURE_NAME")
        assigned = apply_nearest_bone_blend_weights(mesh, armature)
        weighting = "nearest-bone-blend"
        stitch_and_log(weighting)  # after apply_nearest_bone_blend_weights' own smoothing
    if assigned == 0:
        raise RuntimeError("weight binding produced no vertex weights")

    # Long hanging geometry (a robe/skirt shell, a coat tail, the crotch
    # web) has no skeletal representation, so every nearest-wins weighting
    # scheme partitions it rigidly left/right and any scissoring pose
    # shears it into a plank or a smeared sheet. Re-weight it to a smooth
    # lateral blend before the final stitching pass.
    hanging_blend: dict = {"applied": False, "reason": "skipped"}
    if skip_hanging_blend:
        log("skip_hanging_blend: leaving nearest-wins weights on hanging geometry")
    elif facing_report.get("profile") == "humanoid":
        hanging_blend = apply_hanging_region_blend(mesh, armature)
        log(
            f"hanging-region blend: mode={hanging_blend.get('mode')} "
            f"enclosure_depth={hanging_blend.get('enclosure_depth')} "
            f"vertices={hanging_blend.get('vertices')}/{hanging_blend.get('mesh_vertices')} "
            f"torso_shell={hanging_blend.get('torso_shell_vertices')} "
            f"bridge={hanging_blend.get('bridge_vertices')} "
            f"applied={hanging_blend.get('applied')} "
            f"mean_delta={hanging_blend.get('mean_weight_delta')}"
        )
        if hanging_blend.get("applied"):
            # Averaging new weights into individual members of a
            # coincident group can leave that group disagreeing again.
            stitch_and_log("hanging-region blend")
    else:
        hanging_blend = {"applied": False, "reason": "non-humanoid profile"}

    end_weight_concentration = 0.0
    if clean_linear_chain:
        end_weight_concentration = chain_end_weight_concentration(mesh, names)
        if end_weight_concentration > 0.6:
            log(
                f"WARNING: {weighting} weighting concentrated "
                f"{end_weight_concentration:.0%} of vertex weight on the two end bones "
                "(a long, thin body collapsing onto its ends); falling back to "
                "distance-along-chain two-bone-blend weights"
            )
            assigned = apply_distance_along_chain_weights(mesh, joints, names)
            weighting = "distance-along-chain"
            if assigned == 0:
                raise RuntimeError("distance-along-chain weighting produced no vertex weights")
            stitch_and_log(weighting)

    # Unweighted vertices tear the mesh when posed (they stay at bind while
    # neighbours move). Assign each to its nearest joint head with full weight.
    unweighted_before = len(mesh.data.vertices) - assigned
    if unweighted_before > 0:
        import numpy as _np
        joint_positions = _np.array(
            [armature.matrix_world @ b.head_local for b in armature.data.bones],
            dtype=_np.float64,
        )
        world_verts = _np.array(
            [mesh.matrix_world @ v.co for v in mesh.data.vertices], dtype=_np.float64
        )
        distances = _np.linalg.norm(
            world_verts[:, None, :] - joint_positions[None, :, :], axis=2
        )
        nearest = _np.argmin(distances, axis=1)
        bone_names = [b.name for b in armature.data.bones]
        for vertex_index, bone_index in enumerate(nearest):
            mesh.vertex_groups[bone_names[bone_index]].add([vertex_index], 1.0, "ADD")
        assigned = len(mesh.data.vertices)
        log(f"nearest-joint fallback: assigned {unweighted_before} unweighted vertex(es)")
        # The fallback assigns each unweighted vertex independently to its
        # own nearest joint -- two coincident vertices can land on
        # different joints here, reintroducing exactly the tearing risk
        # stitching just closed, so stitch again.
        stitch_and_log("nearest-joint fallback")

    seam_stitched_vertices = len(seam_stitched_indices)

    name_to_group = {g.name: g for g in mesh.vertex_groups}
    group_index_to_name = {g.index: g.name for g in mesh.vertex_groups}
    per_bone_members = {name: 0 for name in name_to_group}
    for vertex in mesh.data.vertices:
        for element in vertex.groups:
            name = group_index_to_name.get(element.group)
            if name is not None:
                per_bone_members[name] += 1
    empty = sorted(name for name, n in per_bone_members.items() if n == 0)
    weighted = bone_count - len(empty)
    log(f"weighting={weighting} vertices_assigned={assigned}/{len(mesh.data.vertices)} "
        f"deform_bones_with_vertices={weighted}/{bone_count} empty={empty[:8]} "
        f"seam_stitched_vertices={seam_stitched_vertices}")

    # Did the normal bind path actually give the grown optional bones any
    # mesh to move? Bone-heat sees them like any other bone, so on a mesh
    # with modelled fingers they pick up their own vertices; on a mitt they
    # may pick up none and the hand simply moves as a block. Both are
    # recorded, neither is fatal.
    optional_weighted_bones = sorted(
        name for name in template_synthesized_optional if per_bone_members.get(name, 0) > 0
    )
    optional_empty_bones = sorted(
        name for name in template_synthesized_optional if per_bone_members.get(name, 0) == 0
    )
    if template_synthesized_optional:
        log(
            f"optional-tier weighting: weighted={len(optional_weighted_bones)}/"
            f"{len(template_synthesized_optional)} empty={optional_empty_bones[:8]}"
        )

    # First-time rigs carry no animation. A subtle rest clip gives downstream
    # consumers (and Blender's glTF importer) an armature with an action to
    # measure deformation against. Checked against this *armature's own*
    # animation_data/action rather than the global `bpy.data.actions` pool:
    # an unrelated playable action left over in the pool (e.g. a donor-
    # library action, or -- before the source_rig import fix -- a prior
    # bind's own leftover Rest clip) would satisfy a pool-wide check
    # without this armature ever having been assigned an action, silently
    # exporting a GLB with no `animations` at all.
    existing_action = armature.animation_data.action if armature.animation_data else None
    if existing_action is None or existing_action.frame_range[1] <= existing_action.frame_range[0]:
        armature.animation_data_create()
        rest_action = bpy.data.actions.new("Rest")
        armature.animation_data.action = rest_action
        pose_bone = armature.pose.bones[0]
        curve = rest_action.fcurves.new(
            data_path=f'pose.bones["{pose_bone.name}"].rotation_euler', index=1
        )
        for frame, angle in ((1, 0.0), (12, 0.05), (24, 0.0)):
            curve.keyframe_points.insert(frame, angle)

    out_path = Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        export_skins=True,
        export_animations=True,
        # Custom properties set on `armature` above (template_conformance
        # and friends) only reach the exported GLB's node `extras` when
        # this is on -- that's how the rig-certificate gate reads N3
        # template-conformance findings back out of the final artifact.
        export_extras=True,
    )
    log(f"rigged GLB: {out_path} bytes={out_path.stat().st_size}")

    if spec.get("validation"):
        validation_path = Path(spec["validation"])
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(json.dumps({
            "backend": "unirig-auto-rig",
            "weighting": weighting,
            "vertices_assigned": assigned,
            "seam_stitched_vertices": seam_stitched_vertices,
            "hanging_region_vertices": int(hanging_blend.get("vertices") or 0),
            "hanging_region_torso_shell_vertices": int(
                hanging_blend.get("torso_shell_vertices") or 0
            ),
            "hanging_region_bridge_vertices": int(hanging_blend.get("bridge_vertices") or 0),
            "hanging_region_blend": hanging_blend,
            "skip_hanging_blend": skip_hanging_blend,
            "control_bones_excluded_from_skinning": control_bones_disabled,
            "bones": len(armature.data.bones),
            "weighted_bones": weighted,
            "empty_bones": empty,
            "vertices": len(mesh.data.vertices),
            "profile": facing_report["profile"],
            "profile_source": facing_report.get("profile_source", "detected"),
            "skeleton_source": facing_report.get("skeleton_source", "unirig"),
            "facing": facing_report,
            "removed_joints": removed_joints,
            "refused_joints": refused_joints,
            "template_conformance": template_conformance,
            "template_conformance_family": "biped" if facing_report.get("profile") == "humanoid" else None,
            "template_synthesized_bones": template_synthesized_bones,
            "skip_optional_bones": skip_optional_bones,
            "synthesized_optional_bones": template_synthesized_optional,
            "optional_chain_scales": template_optional_chain_scales,
            "optional_joint_containment": optional_joint_containment,
            "optional_containment_fraction": (
                optional_joint_containment.get("joints_inside_mesh_fraction")
                if optional_joint_containment else None
            ),
            "optional_weighted_bones": optional_weighted_bones,
            "optional_empty_bones": optional_empty_bones,
            "skip_distal_refinement": skip_distal_refinement,
            "distal_refinement": distal_refinement,
        }, indent=2), encoding="utf-8")


# mesh2motion rig name -> SOMA name (inverse of the app's SOMA_TARGET_FALLBACKS)
DONOR_TO_SOMA = {
    "root": "Root", "pelvis": "Hips", "spine_01": "Spine1", "spine_02": "Spine2",
    "spine_03": "Chest", "neck_01": "Neck1", "head": "Head",
    "clavicle_l": "LeftShoulder", "upperarm_l": "LeftArm", "lowerarm_l": "LeftForeArm",
    "hand_l": "LeftHand", "clavicle_r": "RightShoulder", "upperarm_r": "RightArm",
    "lowerarm_r": "RightForeArm", "hand_r": "RightHand",
    "thigh_l": "LeftLeg", "calf_l": "LeftShin", "foot_l": "LeftFoot",
    "ball_l": "LeftToeBase", "thigh_r": "RightLeg", "calf_r": "RightShin",
    "foot_r": "RightFoot", "ball_r": "RightToeBase",
}

def export_soma_bvh(spec: dict) -> None:
    """Export a donor-library action as a SOMA-named BVH motion clip."""
    clear_scene()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=spec["input"])
    new_objects = [o for o in bpy.context.scene.objects if o not in before]
    armatures = [o for o in new_objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("donor library contains no armature")
    armature = armatures[0]

    wanted = spec.get("animation")
    action = bpy.data.actions.get(wanted) if wanted else None
    if wanted and action is None:
        # Silently falling back to "the longest clip" produced a 5 s crouch
        # labelled Sword_Attack (the donor's action is Sword_Attack_Armature).
        available = sorted(a.name for a in bpy.data.actions if a.frame_range[1] > a.frame_range[0])
        raise RuntimeError(f"donor library has no action named {wanted!r}; available: {available}")
    if action is None:
        playable = [a for a in bpy.data.actions if a.frame_range[1] > a.frame_range[0]]
        if not playable:
            raise RuntimeError("donor library contains no animation actions")
        action = max(playable, key=lambda a: a.frame_range[1] - a.frame_range[0])
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action

    if armature.name != "Root" and "Root" in DONOR_TO_SOMA.values():
        pass
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="OBJECT")
    renamed = 0
    for bone in armature.data.bones:
        soma = DONOR_TO_SOMA.get(bone.name)
        if soma:
            bone.name = soma
            renamed += 1
    log(f"renamed {renamed} bones to SOMA convention")

    start, end = (int(v) for v in action.frame_range)
    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = end
    scene.render.fps = 30

    out_path = Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # BVH in centimetres: the retarget worker imports with global_scale=0.01
    bpy.ops.export_anim.bvh(filepath=str(out_path), global_scale=100.0)
    log(f"exported BVH {out_path} bytes={out_path.stat().st_size}")


def main() -> None:
    argv = sys.argv
    raw = argv[argv.index("--") + 1] if "--" in argv else argv[1]
    spec = json.loads(raw)
    op = spec["op"]
    if op == "strip-rig":
        strip_rig(spec)
    elif op == "unirig-bind":
        unirig_bind(spec)
    elif op == "export-soma-bvh":
        export_soma_bvh(spec)
    else:
        raise RuntimeError(f"Unknown UniRig worker operation: {op}")
    # bpy segfaults during interpreter teardown after glTF export; exit
    # explicitly once the work is flushed and validated on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
