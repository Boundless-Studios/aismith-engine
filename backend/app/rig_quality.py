from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path
from collections.abc import Collection
from typing import Any, Callable

import numpy as np

from . import skeleton_template
from .bilateral_symmetry import (
    SYMMETRY_MIRROR_BAND,
    measure_bilateral_symmetry,
    unmeasurable_report,
)
from .joint_containment import measure_joint_containment
from .skeleton_naming import (
    CONFIDENT_FACING_SOURCES,
    detect_forward_from_feet,
    detect_profile_from_names,
    left_from_forward,
    slab_axis_deviation,
)

# Matches this pipeline's own chain-naming convention (see
# skeleton_naming._name_chain_profile): "spine_00", "spine_01", ... in
# tail-to-head order. Used only once a rig has already been classified
# "chain" (via detect_profile_from_names), so -- unlike that classifier's
# own stricter zero-padded pattern -- this can safely accept any digit
# count when ordering/counting the chain's own bones.
_CHAIN_SPINE_ORDER_PATTERN = re.compile(r"^spine_(\d+)$")


AccessorReader = Callable[[int], np.ndarray]

# glTF is Y-up by convention, and the Blender exporter this pipeline runs
# through (backend/workers/unirig_bind_worker.py, default "+Y Up" settings)
# writes every rig in that convention. Unlike the pre-export Blender-native
# joint arrays in skeleton_naming.py (vertical axis = Z), exported GLBs read
# here have vertical axis = Y.
_EXPORTED_VERTICAL_AXIS = 1

# The only family this repo fits today. Kept as a named default rather than
# a literal so the day a second family ships, the call sites that need to
# thread a real family through are greppable.
DEFAULT_SKELETON_FAMILY = "biped"

# Same alias tuples quality_policy._REQUIRED_HUMANOID_ROLES uses for these
# roles — kept local (rather than imported) so this geometry module does not
# depend on the policy module that in turn depends on it indirectly.
_HANDEDNESS_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "pelvis": ("pelvis", "hips", "hip"),
    "upper_arm_l": ("upperarml", "leftupperarm", "leftarm"),
    "upper_arm_r": ("upperarmr", "rightupperarm", "rightarm"),
    "thigh_l": ("thighl", "leftthigh", "leftupleg", "leftupperleg"),
    "thigh_r": ("thighr", "rightthigh", "rightupleg", "rightupperleg"),
    "foot_l": ("footl", "leftfoot"),
    "foot_r": ("footr", "rightfoot"),
}


def _normalize_joint_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _resolve_handedness_roles(joint_names_by_index: dict[int, str]) -> dict[str, int]:
    normalized = {index: _normalize_joint_name(name) for index, name in joint_names_by_index.items()}
    resolved: dict[str, int] = {}
    for role, aliases in _HANDEDNESS_ROLE_ALIASES.items():
        for index in sorted(normalized):
            if any(alias in normalized[index] for alias in aliases):
                resolved[role] = index
                break
    return resolved


def _detect_rig_handedness(
    nodes: list[dict[str, Any]],
    node_world_matrices: list[np.ndarray],
    joint_nodes: set[int],
    mesh_vertices: np.ndarray | None,
) -> dict[str, Any] | None:
    """Detect facing + left/right bone-position mismatches on an exported rig.

    Reuses the exact same facing detector as the bind worker
    (`backend.app.skeleton_naming.detect_forward_from_feet`), just against
    the exported glTF's own node positions. That means this check is
    independent of which pipeline produced the rig's names — it catches any
    humanoid rig whose named sides disagree with its measured facing, not
    only a bug in this repo's own namer.

    Returns None when the rig doesn't expose enough named joints (pelvis,
    both feet) to measure anything.
    """
    joint_names_by_index = {
        index: str(nodes[index].get("name", f"joint_{index}"))
        for index in joint_nodes
        if 0 <= index < len(nodes)
    }
    roles = _resolve_handedness_roles(joint_names_by_index)
    if "pelvis" not in roles or "foot_l" not in roles or "foot_r" not in roles:
        return None

    children_by_node: dict[int, list[int]] = {
        index: [c for c in node.get("children", []) if isinstance(c, int)]
        for index, node in enumerate(nodes)
    }
    joints = np.asarray([matrix[:3, 3] for matrix in node_world_matrices], dtype=np.float64)
    foot_indices = [roles["foot_l"], roles["foot_r"]]
    foot_children = [
        [c for c in children_by_node.get(index, []) if c in joint_nodes]
        for index in foot_indices
    ]
    facing = detect_forward_from_feet(
        joints,
        foot_indices,
        foot_children,
        mesh_vertices,
        vertical_axis=_EXPORTED_VERTICAL_AXIS,
    )
    left = left_from_forward(facing.forward, vertical_axis=_EXPORTED_VERTICAL_AXIS)

    pelvis_position = joints[roles["pelvis"]]
    mismatches = []
    # Only trust the measurement enough to flag mismatches when a confident
    # facing signal was found (agreeing, non-degenerate toe directions, or a
    # mesh-vertex measurement); a low-confidence toe reading or an assumed
    # default forward is not grounds to block a certificate.
    if facing.source in CONFIDENT_FACING_SOURCES:
        for role in ("upper_arm_l", "upper_arm_r", "thigh_l", "thigh_r"):
            if role not in roles:
                continue
            index = roles[role]
            expected_side = "left" if role.endswith("_l") else "right"
            side_value = float(np.dot(joints[index] - pelvis_position, left))
            actual_side = "left" if side_value > 0.0 else "right"
            if actual_side != expected_side:
                mismatches.append({
                    "joint": joint_names_by_index[index],
                    "role": role,
                    "expected_side": expected_side,
                    "actual_side": actual_side,
                    "position": [round(float(v), 4) for v in joints[index]],
                })

    return {
        "facing_forward": [round(float(v), 4) for v in facing.forward],
        "facing_source": facing.source,
        "facing_warning": facing.warning,
        "mismatches": mismatches,
    }

_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is too small")
    magic, version, declared_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError("Invalid GLB 2.0 header")

    document: dict[str, Any] | None = None
    binary = b""
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise ValueError("GLB chunk exceeds declared file length")
        chunk = payload[offset:end]
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset = end
    if document is None:
        raise ValueError("GLB has no JSON document")
    return document, binary


def _accessor_reader(document: dict[str, Any], binary: bytes) -> AccessorReader:
    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])

    def read(index: int) -> np.ndarray:
        accessor = accessors[index]
        count = int(accessor.get("count", 0))
        width = _TYPE_WIDTHS.get(accessor.get("type"))
        dtype = _COMPONENT_DTYPES.get(accessor.get("componentType"))
        if width is None or dtype is None:
            raise ValueError(f"Unsupported accessor {index}")
        if "bufferView" in accessor:
            view = buffer_views[accessor["bufferView"]]
            if int(view.get("buffer", 0)) != 0:
                raise ValueError("External GLB buffers are not supported by rig diagnostics")
            byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
            item_size = dtype.itemsize * width
            stride = int(view.get("byteStride", item_size))
            end = byte_offset + max(0, count - 1) * stride + item_size
            if end > len(binary):
                raise ValueError(f"Accessor {index} exceeds the GLB binary chunk")
            values = np.ndarray(
                shape=(count, width),
                dtype=dtype,
                buffer=binary,
                offset=byte_offset,
                strides=(stride, dtype.itemsize),
            ).copy()
        else:
            values = np.zeros((count, width), dtype=dtype)
        sparse = accessor.get("sparse")
        if sparse:
            sparse_count = int(sparse.get("count", 0))
            indices_spec = sparse.get("indices", {})
            index_dtype = _COMPONENT_DTYPES.get(indices_spec.get("componentType"))
            if index_dtype not in {_COMPONENT_DTYPES[5121], _COMPONENT_DTYPES[5123], _COMPONENT_DTYPES[5125]}:
                raise ValueError(f"Unsupported sparse index type for accessor {index}")
            index_view = buffer_views[indices_spec["bufferView"]]
            value_spec = sparse.get("values", {})
            value_view = buffer_views[value_spec["bufferView"]]
            if int(index_view.get("buffer", 0)) != 0 or int(value_view.get("buffer", 0)) != 0:
                raise ValueError("External GLB buffers are not supported by rig diagnostics")
            index_offset = int(index_view.get("byteOffset", 0)) + int(indices_spec.get("byteOffset", 0))
            value_offset = int(value_view.get("byteOffset", 0)) + int(value_spec.get("byteOffset", 0))
            indices = np.frombuffer(binary, dtype=index_dtype, count=sparse_count, offset=index_offset)
            sparse_values = np.frombuffer(
                binary,
                dtype=dtype,
                count=sparse_count * width,
                offset=value_offset,
            ).reshape((sparse_count, width))
            if np.any(indices >= count):
                raise ValueError(f"Sparse accessor {index} contains an out-of-range index")
            values[indices] = sparse_values
        if accessor.get("normalized") and np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if np.issubdtype(dtype, np.signedinteger):
                values = np.maximum(values.astype(np.float64) / info.max, -1.0)
            else:
                values = values.astype(np.float64) / info.max
        return values[:, 0] if width == 1 else values

    return read


def analyze_rig_path(
    path: Path,
    preferred_joint_names: Collection[str] | None = None,
) -> dict[str, Any]:
    if path.suffix.lower() != ".glb":
        raise ValueError("Rig diagnostics currently require a binary GLB")
    document, binary = _read_glb(path)
    return analyze_rig_document(
        document,
        _accessor_reader(document, binary),
        preferred_joint_names=preferred_joint_names,
    )


def _channel_has_motion(values: np.ndarray, path_name: str, interpolation: str) -> bool:
    if len(values) <= 1:
        return False
    if interpolation == "CUBICSPLINE" and len(values) % 3 == 0:
        key_values = values[1::3]
        used_tangents = (
            (values[2::3][:-1], key_values[:-1]),
            (values[0::3][1:], key_values[1:]),
        )
        if path_name == "rotation" and key_values.ndim == 2 and key_values.shape[1] == 4:
            norms = np.linalg.norm(key_values, axis=1)
            if np.any(norms <= 1e-12):
                tangents = np.concatenate([item[0] for item in used_tangents], axis=0)
                return bool(np.any(np.abs(tangents) > 1e-6))
            for tangents, tangent_keys in used_tangents:
                normalized_keys = tangent_keys / np.linalg.norm(tangent_keys, axis=1)[:, None]
                projected = tangents - np.sum(tangents * normalized_keys, axis=1)[:, None] * normalized_keys
                if np.any(np.abs(projected) > 1e-6):
                    return True
        else:
            tangents = np.concatenate([item[0] for item in used_tangents], axis=0)
            if np.any(np.abs(tangents) > 1e-6):
                return True
        values = key_values
    if len(values) <= 1:
        return False
    if path_name == "rotation" and values.ndim == 2 and values.shape[1] == 4:
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 1e-12):
            return bool(np.any(np.ptp(values, axis=0) > 1e-6))
        normalized = values / norms[:, None]
        alignment = np.abs(normalized @ normalized[0])
        return bool(np.any(1.0 - np.clip(alignment, 0.0, 1.0) > 1e-6))
    return bool(np.any(np.ptp(values, axis=0) > 1e-6))


def _node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    x, y, z, w = np.asarray(
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _node_world_matrices(nodes: list[dict[str, Any]]) -> list[np.ndarray]:
    parents = {
        child: parent
        for parent, node in enumerate(nodes)
        for child in node.get("children", [])
        if isinstance(child, int)
    }
    resolved: dict[int, np.ndarray] = {}

    def resolve(index: int) -> np.ndarray:
        if index in resolved:
            return resolved[index]
        local = _node_local_matrix(nodes[index])
        parent = parents.get(index)
        resolved[index] = resolve(parent) @ local if parent is not None else local
        return resolved[index]

    return [resolve(index) for index in range(len(nodes))]


def _active_scene_nodes(
    document: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> set[int]:
    scenes = document.get("scenes", [])
    if not scenes:
        return set(range(len(nodes)))
    scene_index = int(document.get("scene", 0))
    if not (0 <= scene_index < len(scenes)):
        raise ValueError("The default glTF scene index is invalid")
    active = set()
    pending = list(scenes[scene_index].get("nodes", []))
    while pending:
        index = pending.pop()
        if not isinstance(index, int) or not (0 <= index < len(nodes)):
            raise ValueError("The default glTF scene references an invalid node")
        if index in active:
            continue
        active.add(index)
        pending.extend(nodes[index].get("children", []))
    return active


def _is_anatomical_joint_name(name: str) -> bool:
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    if any(
        token in normalized
        for token in ("weapon", "socket", "attachment", "helper", "control")
    ):
        return False
    return any(
        token in normalized
        for token in (
            "pelvis", "hips", "spine", "chest", "neck", "head",
            "upperarm", "lowerarm", "forearm", "hand",
            "thigh", "upleg", "calf", "shin", "foot",
        )
    ) and not any(
        token in normalized
        for token in ("finger", "thumb", "index", "middle", "ring", "pinky", "leaf")
    )


def _point_set_diameter(points: np.ndarray) -> float:
    """Estimate diameter from deterministic directional extrema."""
    if len(points) < 2:
        return 0.0
    directions = np.asarray([
        (x, y, z)
        for x in range(-2, 3)
        for y in range(-2, 3)
        for z in range(-2, 3)
        if (x, y, z) != (0, 0, 0)
    ], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    candidates = np.unique(
        points[np.argmax(points @ directions.T, axis=0)],
        axis=0,
    )
    if len(candidates) < 2:
        return 0.0
    return max(
        float(np.linalg.norm(first - second))
        for offset, first in enumerate(candidates)
        for second in candidates[offset + 1:]
    )


def joint_rest_positions(
    nodes: list[dict[str, Any]],
    joint_nodes: Collection[int],
    joint_names_by_index: dict[int, str],
    node_world_matrices: list[np.ndarray],
    instance_bind_positions: dict[tuple[int, int], np.ndarray],
    rigid_attachment_nodes: Collection[int],
) -> dict[str, np.ndarray]:
    """Every joint's bind-pose head in world space, keyed by bone name.

    Bind-matrix first, node world transform as the fallback: the GLB's
    inverse bind matrices are what the SKIN was authored against, so a rig
    whose node transforms have drifted from its bind pose is still measured
    where the mesh thinks its joints are. A skin instance that is itself a
    rigid attachment (a weapon parented under a hand) is not allowed to
    supply the bind position -- it carries the prop's transform, not the
    body's.

    Shared by the containment and bilateral-symmetry measurements so both
    read the same positions; disagreeing on where a joint IS would make
    their verdicts impossible to compare.
    """
    positions: dict[str, np.ndarray] = {}
    for joint_index in sorted(joint_nodes):
        if not (0 <= joint_index < len(nodes)):
            continue
        name = joint_names_by_index.get(joint_index, f"joint_{joint_index}")
        bind = [
            position
            for (mesh_node, bound_joint), position in sorted(instance_bind_positions.items())
            if bound_joint == joint_index and mesh_node not in rigid_attachment_nodes
        ]
        positions[name] = bind[0] if bind else node_world_matrices[joint_index][:3, 3]
    return positions


def joint_rest_directions(
    nodes: list[dict[str, Any]],
    joint_nodes: Collection[int],
    joint_names_by_index: dict[int, str],
    rest_positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray | None]:
    """Each joint's head->tail direction, inferred from its child joints.

    glTF has no tail: a bone is a node, and its "length" is only visible as
    the offset to whatever hangs off it. The mean of the child heads is used
    rather than the first child so a branching joint (a hand with five
    fingers) gets an average aim instead of an arbitrary one. A leaf joint
    has no tail to infer and is reported as None rather than as a zero
    vector, which a caller would otherwise measure an angle against.
    """
    directions: dict[str, np.ndarray | None] = {}
    joint_set = set(joint_nodes)
    for joint_index in sorted(joint_set):
        if not (0 <= joint_index < len(nodes)):
            continue
        name = joint_names_by_index.get(joint_index, f"joint_{joint_index}")
        head = rest_positions.get(name)
        child_heads = [
            rest_positions[joint_names_by_index.get(child, f"joint_{child}")]
            for child in nodes[joint_index].get("children", [])
            if isinstance(child, int)
            and child in joint_set
            and joint_names_by_index.get(child, f"joint_{child}") in rest_positions
        ]
        if head is None or not child_heads:
            directions[name] = None
            continue
        directions[name] = np.mean(np.asarray(child_heads, dtype=np.float64), axis=0) - head
    return directions


def _measure_joint_containment(
    nodes: list[dict[str, Any]],
    joint_nodes: Collection[int],
    joint_names_by_index: dict[int, str],
    node_world_matrices: list[np.ndarray],
    instance_bind_positions: dict[tuple[int, int], np.ndarray],
    rigid_attachment_nodes: Collection[int],
    skinned_triangle_sets: list[np.ndarray],
    model_height: float,
) -> dict[str, Any]:
    """S3's "all placed joints are inside the mesh", as a measured fraction.

    Joint heads are read from the skin's inverse bind matrices where the GLB
    carries them, so the joint and the mesh are compared in the same bind
    pose; a rig without inverse bind matrices falls back to node world
    transforms. Prop/weapon sockets are excluded by name -- a sword socket
    sits outside the body on purpose and is not an S3 fit failure.
    """
    rest_positions = joint_rest_positions(
        nodes,
        joint_nodes,
        joint_names_by_index,
        node_world_matrices,
        instance_bind_positions,
        rigid_attachment_nodes,
    )
    excluded: list[str] = []
    names: list[str] = []
    positions: list[np.ndarray] = []
    for name, position in rest_positions.items():
        if _is_rigid_attachment_name(name):
            excluded.append(name)
            continue
        names.append(name)
        positions.append(position)

    triangles = (
        np.concatenate(skinned_triangle_sets, axis=0)
        if skinned_triangle_sets
        else np.empty((0, 3, 3), dtype=np.float64)
    )
    report = measure_joint_containment(
        names,
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        triangles,
        model_height=model_height,
    )
    report["excluded_joints"] = excluded
    return report


def _measure_bilateral_symmetry(
    nodes: list[dict[str, Any]],
    joint_nodes: Collection[int],
    joint_names_by_index: dict[int, str],
    node_world_matrices: list[np.ndarray],
    instance_bind_positions: dict[tuple[int, int], np.ndarray],
    rigid_attachment_nodes: Collection[int],
    handedness: dict[str, Any] | None,
    model_height: float,
    family: str = DEFAULT_SKELETON_FAMILY,
) -> dict[str, Any]:
    """S3's "left/right placement is within the family's symmetry bands", as
    a measured mirror error.

    Gated on a CONFIDENT facing detection for the same reason
    `_detect_rig_handedness` gates its mismatch list: the sagittal plane is
    derived from the measured forward vector, so an assumed or low-confidence
    facing would mirror across the wrong plane and invent errors on a rig
    that is fine. "We could not tell which way this character faces" is
    reported as unmeasurable, never as asymmetry.
    """
    if handedness is None:
        return unmeasurable_report("no handedness measurement (pelvis and both feet required)")
    if handedness.get("facing_source") not in CONFIDENT_FACING_SOURCES:
        return unmeasurable_report(
            f"facing not confident (source {handedness.get('facing_source')!r}); "
            "the sagittal plane would be a guess"
        )
    try:
        template = skeleton_template.load_template(family)
    except (FileNotFoundError, KeyError, ValueError):
        return unmeasurable_report(f"no skeleton template for family {family!r}")

    rest_positions = joint_rest_positions(
        nodes,
        joint_nodes,
        joint_names_by_index,
        node_world_matrices,
        instance_bind_positions,
        rigid_attachment_nodes,
    )
    plane_joint = next(
        (name for name in ("pelvis", "root") if name in rest_positions), None
    )
    if plane_joint is None:
        return unmeasurable_report(
            "no 'pelvis' or 'root' joint to anchor the sagittal plane on"
        )

    return measure_bilateral_symmetry(
        joint_positions=rest_positions,
        joint_directions=joint_rest_directions(
            nodes, joint_nodes, joint_names_by_index, rest_positions,
        ),
        pairs=[
            (bone.name, template.mirror_of(bone.name))
            for bone in template.bones
            if bone.name.endswith("_l") and template.mirror_of(bone.name)
        ],
        left=left_from_forward(
            np.asarray(handedness["facing_forward"], dtype=np.float64),
            vertical_axis=_EXPORTED_VERTICAL_AXIS,
        ),
        plane_point=rest_positions[plane_joint],
        model_height=model_height,
        band=SYMMETRY_MIRROR_BAND,
    )


def _is_rigid_attachment_name(name: str) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in (
        "weapon", "sword", "blade", "guard", "grip", "shield", "prop",
    ))


def _order_chain_bones(joint_names_by_index: dict[int, str]) -> list[int]:
    """Order a chain rig's `spine_NN` bones tail-to-head, then `head` last --
    matching skeleton_naming's own chain-naming convention."""
    spine_entries = []
    head_index: int | None = None
    for index, name in joint_names_by_index.items():
        normalized = name.strip().casefold()
        match = _CHAIN_SPINE_ORDER_PATTERN.match(normalized)
        if match:
            spine_entries.append((int(match.group(1)), index))
        elif normalized == "head":
            head_index = index
    spine_entries.sort(key=lambda item: item[0])
    ordered = [index for _, index in spine_entries]
    if head_index is not None:
        ordered.append(head_index)
    return ordered


def _chain_topology(
    chain_indices: set[int],
    children_by_node: dict[int, list[int]],
    parent_by_node: dict[int, int],
    joint_names_by_index: dict[int, str],
) -> dict[str, Any]:
    """Whether the chain-vocabulary (spine + head) bones form a single
    unbranched path, using the exported rig's own node hierarchy -- a
    finding a namer bug (or a hand-edited rig) could otherwise slip past
    name-pattern checks alone.

    Uses *undirected* degree within the chain-vocabulary subgraph (a
    node's parent counts as a neighbor too, not just its children): a real
    UniRig root often sits mid-body (see skeleton_naming's tree-diameter
    profile detection), so a root with exactly two chain-vocabulary
    children is just the "elbow" of a single valid path, not a branch --
    only a node with *three or more* such neighbors is a genuine fork a
    simple path can't have.
    """
    branch_joints = []
    for index in chain_indices:
        neighbors = {
            child for child in children_by_node.get(index, []) if child in chain_indices
        }
        parent = parent_by_node.get(index)
        if parent is not None and parent in chain_indices:
            neighbors.add(parent)
        if len(neighbors) > 2:
            branch_joints.append(joint_names_by_index[index])
    branch_joints.sort()
    return {"is_single_chain": not branch_joints, "branch_joints": branch_joints}


def _nearest_chain_segment(chain_positions: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Index of the chain polyline segment (i -> i+1) each vertex is
    closest to, by perpendicular distance to the clamped segment."""
    vertex_count = len(vertices)
    if len(chain_positions) < 2 or vertex_count == 0:
        return np.zeros(vertex_count, dtype=np.int64)
    starts = chain_positions[:-1]
    vectors = chain_positions[1:] - starts
    lengths_sq = np.sum(vectors ** 2, axis=1)
    best_distance_sq = np.full(vertex_count, np.inf)
    best_segment = np.zeros(vertex_count, dtype=np.int64)
    for segment_index in range(len(vectors)):
        offset = vertices - starts[segment_index]
        length_sq = lengths_sq[segment_index]
        if length_sq <= 1e-12:
            t = np.zeros(vertex_count)
        else:
            t = np.clip(offset @ vectors[segment_index] / length_sq, 0.0, 1.0)
        projected = starts[segment_index] + np.outer(t, vectors[segment_index])
        distance_sq = np.sum((vertices - projected) ** 2, axis=1)
        better = distance_sq < best_distance_sq
        best_distance_sq = np.where(better, distance_sq, best_distance_sq)
        best_segment = np.where(better, segment_index, best_segment)
    return best_segment


def _chain_axis_errors(
    ordered_chain: list[int],
    node_world_matrices: list[np.ndarray],
    joint_names_by_index: dict[int, str],
    mesh_vertices: np.ndarray,
) -> list[dict[str, Any]]:
    """For each chain bone, how far a genuine local *geometric* slab of
    mesh vertices sits off the local chain axis, relative to the slab's
    own spread (an approximation of "local cross-section radius") -- a
    bone sitting off the mesh's medial line will show a large offset
    relative to that local spread, even though a snake's local radius
    itself varies along its length.

    Deliberately spatial (a slab of vertices near the joint's own
    *position*), not skinning-weight-based (vertices bone-heat/envelope
    happened to weight *to* that joint): the two are not the same thing.
    A long chain's weight functions can be smooth and wide, especially
    near the ends where fewer joints compete for influence, so "the
    joint's dominantly-weighted vertices" can span a stretch far longer
    than its true local neighborhood -- inflating this measurement with
    every vertex's spread along the *whole* stretch, not just close by.
    """
    if len(ordered_chain) < 2 or mesh_vertices is None or len(mesh_vertices) == 0:
        return []
    positions = {index: node_world_matrices[index][:3, 3] for index in ordered_chain}
    # Per-segment lengths, not a single global average: a raw (non-
    # densified) UniRig prediction's joint spacing is often highly uneven,
    # so a chain-wide average half-thickness can be wildly wrong at any
    # given joint -- too wide where the true local spacing is much
    # smaller, capturing vertices spanning several real segments (and any
    # curvature between them) into what should be one local cross-section.
    segment_lengths = [
        float(np.linalg.norm(positions[ordered_chain[i + 1]] - positions[ordered_chain[i]]))
        for i in range(len(ordered_chain) - 1)
    ]
    fallback_half_thickness = max(
        (float(np.mean(segment_lengths)) if segment_lengths else 0.0) * 0.5, 1e-9,
    )
    # A coiled body (a serpent in an S or a loop) brings *other* stretches
    # of itself within a slab's reach: a slab perpendicular to the local
    # axis at one joint cuts straight through the neighbouring coil, whose
    # vertices then dominate both the "local radius" and the centroid
    # offset even though the joint sits dead on its own medial line (seen
    # live: local_radius 0.28 on a body whose true half-thickness is 0.046,
    # ratio 0.96 on joints with bind-pose landmark error < 3%). Assign
    # every vertex to its nearest chain segment first, and let each joint
    # measure only the vertices that belong to its own two segments.
    nearest_segment = _nearest_chain_segment(
        np.asarray([positions[index] for index in ordered_chain], dtype=np.float64),
        mesh_vertices,
    )
    errors: list[dict[str, Any]] = []
    for position_in_chain, joint_index in enumerate(ordered_chain):
        own_segments = [
            index for index in (position_in_chain - 1, position_in_chain)
            if 0 <= index < len(segment_lengths)
        ]
        local_vertices = mesh_vertices[np.isin(nearest_segment, own_segments)]
        if position_in_chain == 0:
            neighbor_a, neighbor_b = ordered_chain[0], ordered_chain[1]
        elif position_in_chain == len(ordered_chain) - 1:
            neighbor_a, neighbor_b = ordered_chain[-2], ordered_chain[-1]
        else:
            neighbor_a, neighbor_b = ordered_chain[position_in_chain - 1], ordered_chain[position_in_chain + 1]
        direction = positions[neighbor_b] - positions[neighbor_a]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        direction = direction / norm
        joint_position = positions[joint_index]
        local_segments = [
            segment_lengths[index]
            for index in (position_in_chain - 1, position_in_chain)
            if 0 <= index < len(segment_lengths)
        ]
        half_thickness = (
            max(float(np.mean(local_segments)) * 0.5, 1e-9)
            if local_segments
            else fallback_half_thickness
        )
        result = slab_axis_deviation(joint_position, direction, local_vertices, half_thickness)
        if result is None:
            continue
        radius, deviation = result
        deviation_ratio = deviation / radius if radius > 1e-9 else 0.0
        errors.append({
            "joint": joint_names_by_index.get(joint_index, f"joint_{joint_index}"),
            "joint_node": joint_index,
            "local_radius": round(radius, 4),
            "axis_deviation": round(deviation, 4),
            "axis_deviation_ratio": round(deviation_ratio, 4),
        })
    return errors


def _extract_template_conformance(nodes: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The N3 bind worker (`unirig_bind_worker.unirig_bind`) computes
    biped-template conformance findings at bind time, with exact bpy
    edit-bone data (heads/tails/rolls in the family's own house frame) --
    full precision no re-derivation from this re-parsed, Y-up-converted
    GLB could match. It stashes them as a custom property on the armature
    object, which the glTF exporter (`export_extras=True`) carries through
    as that node's `extras.template_conformance`. Read straight back here
    rather than re-computing anything geometric.

    Returns `None` when no node carries the key at all (a non-humanoid rig,
    or a GLB from before this wiring existed) -- `[]` is a real, meaningful
    "conformant" result and must stay distinguishable from "absent".
    """
    for node in nodes:
        extras = node.get("extras")
        if isinstance(extras, dict) and "template_conformance" in extras:
            findings = extras["template_conformance"]
            return list(findings) if isinstance(findings, list) else []
    return None


def analyze_rig_document(
    document: dict[str, Any],
    read_accessor: AccessorReader,
    preferred_joint_names: Collection[str] | None = None,
) -> dict[str, Any]:
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    active_nodes = _active_scene_nodes(document, nodes)
    if document.get("scenes"):
        active_skin_indices = {
            nodes[index].get("skin")
            for index in active_nodes
            if isinstance(nodes[index].get("skin"), int)
        }
        active_skins = [
            skin
            for index, skin in enumerate(skins)
            if index in active_skin_indices
        ]
    else:
        active_skins = skins
    joint_nodes = {joint for skin in active_skins for joint in skin.get("joints", [])}
    joint_names = [
        str(nodes[index].get("name", f"joint_{index}"))
        for index in sorted(joint_nodes)
        if 0 <= index < len(nodes)
    ]
    joint_names_by_index = {
        index: name
        for index, name in zip(sorted(i for i in joint_nodes if 0 <= i < len(nodes)), joint_names)
    }
    profile = detect_profile_from_names(joint_names)
    animation_joint_nodes = joint_nodes
    preferred_names = {name.casefold() for name in preferred_joint_names or () if name}
    if preferred_names:
        skin_joint_sets = [
            {joint for joint in skin.get("joints", []) if isinstance(joint, int)}
            for skin in active_skins
        ]
        best_match = max(
            skin_joint_sets,
            key=lambda joints: sum(
                1
                for index in joints
                if 0 <= index < len(nodes)
                and str(nodes[index].get("name", "")).casefold() in preferred_names
            ),
            default=set(),
        )
        overlap = {
            index
            for index in best_match
            if 0 <= index < len(nodes)
            and str(nodes[index].get("name", "")).casefold() in preferred_names
        }
        if overlap:
            animation_joint_nodes = overlap

    skinned_primitives = 0
    unskinned_primitives = 0
    skinned_vertices = 0
    unskinned_vertices = 0
    rigid_attachment_primitives = 0
    rigid_attachment_vertices = 0
    unweighted_vertices = 0
    non_normalized_vertices = 0
    invalid_joint_references = 0
    maximum_influences = 0
    total_influences = 0
    rigid_vertices = 0
    blended_vertices = 0
    equal_blend_vertices = 0
    used_joint_nodes: set[int] = set()
    joint_vertex_counts: dict[int, int] = {}
    joint_weight_totals: dict[int, float] = {}
    joint_weighted_positions: dict[int, np.ndarray] = {}
    joint_landmark_weight_totals: dict[int, float] = {}
    joint_landmark_positions: dict[int, np.ndarray] = {}
    instance_landmark_weight_totals: dict[tuple[int, int], float] = {}
    instance_landmark_positions: dict[tuple[int, int], np.ndarray] = {}
    instance_bind_positions: dict[tuple[int, int], np.ndarray] = {}
    rigid_attachment_nodes: set[int] = set()
    skinned_position_sets: list[np.ndarray] = []
    skinned_triangle_sets: list[np.ndarray] = []
    anatomical_position_sets: list[tuple[int, np.ndarray]] = []
    node_world_matrices = _node_world_matrices(nodes)

    parent_nodes = {
        child: parent
        for parent, node in enumerate(nodes)
        for child in node.get("children", [])
        if isinstance(child, int)
    }

    def attached_to_joint(node_index: int) -> bool:
        parent = parent_nodes.get(node_index)
        while parent is not None:
            if parent in joint_nodes:
                return True
            parent = parent_nodes.get(parent)
        return False

    for node_index, node in enumerate(nodes):
        if node_index not in active_nodes:
            continue
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if not isinstance(mesh_index, int) or not (0 <= mesh_index < len(meshes)):
            continue
        primitives = meshes[mesh_index].get("primitives", [])
        if not isinstance(skin_index, int) or not (0 <= skin_index < len(skins)):
            is_rigid_attachment = attached_to_joint(node_index)
            if is_rigid_attachment:
                rigid_attachment_primitives += len(primitives)
            else:
                unskinned_primitives += len(primitives)
            for primitive in primitives:
                position_index = primitive.get("attributes", {}).get("POSITION")
                if isinstance(position_index, int):
                    vertex_count = len(read_accessor(position_index))
                    if is_rigid_attachment:
                        rigid_attachment_vertices += vertex_count
                    else:
                        unskinned_vertices += vertex_count
            continue
        skin_joints = skins[skin_index].get("joints", [])
        inverse_bind_index = skins[skin_index].get("inverseBindMatrices")
        if isinstance(inverse_bind_index, int):
            inverse_bind_matrices = np.asarray(
                read_accessor(inverse_bind_index),
                dtype=np.float64,
            )
            for local_index, joint_node in enumerate(skin_joints):
                if local_index >= len(inverse_bind_matrices):
                    break
                inverse_bind = inverse_bind_matrices[local_index].reshape(
                    (4, 4),
                    order="F",
                )
                instance_bind_positions[(node_index, joint_node)] = (
                    node_world_matrices[node_index]
                    @ np.linalg.inv(inverse_bind)
                )[:3, 3]
        else:
            mesh_origin = node_world_matrices[node_index][:3, 3]
            for joint_node in skin_joints:
                instance_bind_positions[(node_index, joint_node)] = mesh_origin.copy()
        for primitive in primitives:
            attributes = primitive.get("attributes", {})
            position_index = attributes.get("POSITION")
            influence_sets = [
                (attributes.get(f"JOINTS_{suffix}"), weights_index)
                for key, weights_index in attributes.items()
                if key.startswith("WEIGHTS_")
                for suffix in [key.removeprefix("WEIGHTS_")]
                if isinstance(attributes.get(f"JOINTS_{suffix}"), int)
                and isinstance(weights_index, int)
            ]
            if not isinstance(position_index, int) or not influence_sets:
                unskinned_primitives += 1
                if isinstance(position_index, int):
                    unskinned_vertices += len(read_accessor(position_index))
                continue
            positions = np.asarray(read_accessor(position_index), dtype=np.float64).copy()
            morph_weights = node.get("weights", meshes[mesh_index].get("weights", []))
            for target, weight in zip(
                primitive.get("targets", []),
                morph_weights,
            ):
                target_position = target.get("POSITION")
                if not isinstance(target_position, int) or float(weight) == 0.0:
                    continue
                deltas = np.asarray(read_accessor(target_position), dtype=np.float64)
                if len(deltas) != len(positions):
                    raise ValueError("Morph target positions have mismatched vertex counts")
                positions[:, :3] += deltas[:, :3] * float(weight)
            joints = np.concatenate(
                [np.asarray(read_accessor(joints_index)) for joints_index, _ in influence_sets],
                axis=1,
            )
            weights = np.concatenate(
                [np.asarray(read_accessor(weights_index), dtype=np.float64) for _, weights_index in influence_sets],
                axis=1,
            )
            if len(positions) != len(joints) or len(positions) != len(weights):
                raise ValueError("Skin attributes have mismatched vertex counts")
            surface_area_weights = np.ones(len(positions), dtype=np.float64)
            index_accessor = primitive.get("indices")
            if isinstance(index_accessor, int):
                raw_indices = np.asarray(
                    read_accessor(index_accessor),
                    dtype=np.int64,
                ).reshape(-1)
                referenced = np.unique(raw_indices)
                if np.any(referenced < 0) or np.any(referenced >= len(positions)):
                    raise ValueError("Primitive indices reference vertices outside its attributes")
                if primitive.get("mode", 4) == 4 and len(raw_indices) % 3 == 0:
                    triangles = raw_indices.reshape((-1, 3))
                    triangle_positions = positions[triangles, :3]
                    triangle_areas = np.linalg.norm(
                        np.cross(
                            triangle_positions[:, 1] - triangle_positions[:, 0],
                            triangle_positions[:, 2] - triangle_positions[:, 0],
                        ),
                        axis=1,
                    ) * 0.5
                    surface_area_weights = np.zeros(len(positions), dtype=np.float64)
                    np.add.at(
                        surface_area_weights,
                        triangles.reshape(-1),
                        np.repeat(triangle_areas / 3.0, 3),
                    )
                    # Keep the skinned triangle soup in world space: it is the
                    # mesh volume `joint_containment` casts parity rays
                    # against. Collected here, before `positions` is
                    # re-indexed onto the referenced subset below, so the
                    # triangle indices still address `positions`.
                    if len(triangles):
                        world = node_world_matrices[node_index]
                        skinned_triangle_sets.append(
                            (triangle_positions.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
                            .reshape(-1, 3, 3)
                        )
                positions = positions[referenced]
                joints = joints[referenced]
                weights = weights[referenced]
                surface_area_weights = surface_area_weights[referenced]
            if not np.any(surface_area_weights > 0):
                surface_area_weights.fill(1.0)
            homogeneous = np.concatenate(
                [positions[:, :3], np.ones((len(positions), 1), dtype=np.float64)],
                axis=1,
            )
            world_positions = (node_world_matrices[node_index] @ homogeneous.T).T[:, :3]
            if len(world_positions):
                skinned_position_sets.append(world_positions)
            anatomical_local_indices = {
                local_index
                for local_index, joint_node in enumerate(skin_joints)
                if 0 <= joint_node < len(nodes)
                and _is_anatomical_joint_name(
                    str(nodes[joint_node].get("name", f"joint_{joint_node}"))
                )
            }
            joint_totals = np.zeros((len(joints), len(skin_joints)), dtype=np.float64)
            rows = np.repeat(np.arange(len(joints)), joints.shape[1])
            flattened_joints = joints.reshape(-1).astype(np.int64)
            flattened_weights = weights.reshape(-1)
            valid_slots = (
                (flattened_joints >= 0)
                & (flattened_joints < len(skin_joints))
                & (flattened_weights > 0)
            )
            np.add.at(
                joint_totals,
                (rows[valid_slots], flattened_joints[valid_slots]),
                flattened_weights[valid_slots],
            )
            dominant_ranking = joint_totals.copy()
            if anatomical_local_indices:
                dominant_ranking[:, sorted(anatomical_local_indices)] += 1e-12
            dominant_joints = np.argmax(dominant_ranking, axis=1)
            influence_counts = np.count_nonzero(joint_totals > 1e-4, axis=1)
            weighted_vertices = np.sum(weights, axis=1) > 1e-4
            rigid_skin_attachment = (
                np.any(weighted_vertices)
                and np.all(influence_counts[weighted_vertices] == 1)
                and len(np.unique(dominant_joints[weighted_vertices])) == 1
            )
            node_name = str(node.get("name", ""))
            mesh_name = str(meshes[mesh_index].get("name", ""))
            if rigid_skin_attachment and (
                _is_rigid_attachment_name(node_name)
                or _is_rigid_attachment_name(mesh_name)
            ):
                rigid_attachment_nodes.add(node_index)
            anatomical_vertices = np.isin(
                dominant_joints,
                list(anatomical_local_indices),
            ) & (np.max(weights, axis=1) > 1e-4)
            if np.any(anatomical_vertices):
                anatomical_positions = world_positions[anatomical_vertices]
                anatomical_position_sets.append((node_index, anatomical_positions))
            skinned_primitives += 1
            skinned_vertices += len(positions)
            weight_sums = weights.sum(axis=1)
            unweighted_vertices += int(np.count_nonzero(weight_sums <= 1e-6))
            non_normalized_vertices += int(np.count_nonzero(np.abs(weight_sums - 1.0) > 0.01))
            maximum_influences = max(
                maximum_influences,
                int(np.max(influence_counts, initial=0)),
            )
            total_influences += int(np.sum(influence_counts))
            rigid_vertices += int(np.count_nonzero(influence_counts == 1))
            blended_vertices += int(np.count_nonzero(influence_counts > 1))
            sorted_weights = np.sort(weights, axis=1)
            equal_blend_vertices += int(
                np.count_nonzero(
                    (influence_counts == 2)
                    & (np.abs(sorted_weights[:, -1] - sorted_weights[:, -2]) <= 0.01)
                )
            )
            active = weights > 1e-4
            invalid_joint_references += int(
                np.count_nonzero(active & ((joints < 0) | (joints >= len(skin_joints))))
            )
            valid = active & (joints >= 0) & (joints < len(skin_joints))
            for local_index in np.unique(joints[valid]):
                local_index = int(local_index)
                joint_node = skin_joints[local_index]
                mask = valid & (joints == local_index)
                used_joint_nodes.add(joint_node)
                joint_vertex_counts[joint_node] = joint_vertex_counts.get(joint_node, 0) + int(
                    np.count_nonzero(np.any(mask, axis=1))
                )
                joint_weight_totals[joint_node] = joint_weight_totals.get(joint_node, 0.0) + float(
                    np.sum(weights[mask])
                )
                vertex_weights = joint_totals[:, local_index]
                joint_weighted_positions[joint_node] = (
                    joint_weighted_positions.get(joint_node, np.zeros(3, dtype=np.float64))
                    + np.sum(
                        world_positions
                        * vertex_weights[:, None],
                        axis=0,
                    )
                )
                landmark_vertices = (
                    anatomical_vertices
                    if local_index in anatomical_local_indices
                    else dominant_joints == local_index
                )
                landmark_weights = np.sum(
                    np.where(mask & landmark_vertices[:, None], weights, 0.0),
                    axis=1,
                ) * surface_area_weights
                landmark_total = float(np.sum(landmark_weights))
                if landmark_total > 0:
                    instance_key = (node_index, joint_node)
                    joint_landmark_weight_totals[joint_node] = (
                        joint_landmark_weight_totals.get(joint_node, 0.0)
                        + landmark_total
                    )
                    joint_landmark_positions[joint_node] = (
                        joint_landmark_positions.get(
                            joint_node,
                            np.zeros(3, dtype=np.float64),
                        )
                        + np.sum(
                            world_positions * landmark_weights[:, None],
                            axis=0,
                        )
                    )
                    instance_landmark_weight_totals[instance_key] = (
                        instance_landmark_weight_totals.get(instance_key, 0.0)
                        + landmark_total
                    )
                    instance_landmark_positions[instance_key] = (
                        instance_landmark_positions.get(
                            instance_key,
                            np.zeros(3, dtype=np.float64),
                        )
                        + np.sum(
                            world_positions * landmark_weights[:, None],
                            axis=0,
                        )
                    )

    animations: list[dict[str, Any]] = []
    for animation_index, animation in enumerate(document.get("animations", [])):
        samplers = animation.get("samplers", [])
        durations: list[float] = []
        for sampler in samplers:
            input_index = sampler.get("input")
            if isinstance(input_index, int):
                times = np.asarray(read_accessor(input_index), dtype=np.float64)
                if times.size:
                    durations.append(float(np.max(times) - np.min(times)))
        animated_joints: set[int] = set()
        path_counts: dict[str, int] = {}
        invalid_targets = 0
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            node_index = target.get("node")
            path_name = str(target.get("path", "unknown"))
            path_counts[path_name] = path_counts.get(path_name, 0) + 1
            if not isinstance(node_index, int) or not (0 <= node_index < len(nodes)):
                invalid_targets += 1
            elif node_index in animation_joint_nodes:
                sampler_index = channel.get("sampler")
                if not isinstance(sampler_index, int) or not (0 <= sampler_index < len(samplers)):
                    invalid_targets += 1
                    continue
                sampler = samplers[sampler_index]
                output_index = sampler.get("output")
                if not isinstance(output_index, int):
                    invalid_targets += 1
                    continue
                values = np.asarray(read_accessor(output_index), dtype=np.float64)
                if _channel_has_motion(
                    values,
                    path_name,
                    str(sampler.get("interpolation", "LINEAR")),
                ):
                    animated_joints.add(node_index)
        animations.append(
            {
                "name": str(animation.get("name") or f"Animation {animation_index + 1}"),
                "duration_seconds": max(durations, default=0.0),
                "channels": len(animation.get("channels", [])),
                "channel_paths": path_counts,
                "animated_joints": len(animated_joints),
                "animated_joint_names": [
                    str(nodes[index].get("name", f"joint_{index}"))
                    for index in sorted(animated_joints)
                ],
                "animated_joint_coverage": (
                    len(animated_joints) / len(animation_joint_nodes)
                    if animation_joint_nodes
                    else 0.0
                ),
                "animation_joint_scope": len(animation_joint_nodes),
                "invalid_targets": invalid_targets,
            }
        )

    joint_weight_coverage = [
        {
            "joint_node": index,
            "joint": str(nodes[index].get("name", f"joint_{index}")),
            "vertices": joint_vertex_counts.get(index, 0),
            "total_weight": round(joint_weight_totals.get(index, 0.0), 4),
            "weight_centroid": [
                round(float(value), 4)
                for value in (
                    joint_landmark_positions.get(index, np.zeros(3, dtype=np.float64))
                    / max(joint_landmark_weight_totals.get(index, 0.0), 1e-12)
                )
            ],
        }
        for index in sorted(joint_nodes)
        if 0 <= index < len(nodes)
    ]
    unused_joint_names = [
        str(nodes[index].get("name", f"joint_{index}"))
        for index in sorted(joint_nodes - used_joint_nodes)
        if 0 <= index < len(nodes)
    ]
    normalization_sets = [
        positions
        for node_index, positions in anatomical_position_sets
        if node_index not in rigid_attachment_nodes
    ] or skinned_position_sets
    if normalization_sets:
        normalization_positions = np.concatenate(normalization_sets, axis=0)
        model_height = _point_set_diameter(normalization_positions)
    else:
        normalization_positions = None
        model_height = 0.0
    handedness = _detect_rig_handedness(
        nodes, node_world_matrices, joint_nodes, normalization_positions,
    )
    chain_topology: dict[str, Any] | None = None
    chain_axis_errors: list[dict[str, Any]] = []
    if profile == "chain":
        chain_indices = {
            index for index, name in joint_names_by_index.items()
            if _CHAIN_SPINE_ORDER_PATTERN.match(name.strip().casefold())
            or name.strip().casefold() == "head"
        }
        children_by_node = {
            index: [child for child in node.get("children", []) if isinstance(child, int)]
            for index, node in enumerate(nodes)
        }
        chain_topology = _chain_topology(chain_indices, children_by_node, parent_nodes, joint_names_by_index)
        ordered_chain = _order_chain_bones(joint_names_by_index)
        all_skinned_vertices = (
            np.concatenate(skinned_position_sets, axis=0)
            if skinned_position_sets
            else np.empty((0, 3), dtype=np.float64)
        )
        chain_axis_errors = _chain_axis_errors(
            ordered_chain, node_world_matrices, joint_names_by_index, all_skinned_vertices,
        )
    bind_pose_landmark_errors = []
    for (node_index, joint_index), weighted_position in sorted(
        instance_landmark_positions.items()
    ):
        if node_index in rigid_attachment_nodes:
            continue
        landmark_total = instance_landmark_weight_totals[(node_index, joint_index)]
        joint_position = instance_bind_positions.get(
            (node_index, joint_index),
            node_world_matrices[joint_index][:3, 3],
        )
        centroid = weighted_position / landmark_total
        distance = float(np.linalg.norm(joint_position - centroid))
        bind_pose_landmark_errors.append({
            "joint_node": joint_index,
            "joint": str(nodes[joint_index].get("name", f"joint_{joint_index}")),
            "distance": round(distance, 4),
            "normalized_error": round(distance / max(model_height, 1e-12), 4),
            "joint_position": [round(float(value), 4) for value in joint_position],
            "weight_centroid": [round(float(value), 4) for value in centroid],
        })

    joint_containment = _measure_joint_containment(
        nodes,
        joint_nodes,
        joint_names_by_index,
        node_world_matrices,
        instance_bind_positions,
        rigid_attachment_nodes,
        skinned_triangle_sets,
        model_height,
    )
    bilateral_symmetry = _measure_bilateral_symmetry(
        nodes,
        joint_nodes,
        joint_names_by_index,
        node_world_matrices,
        instance_bind_positions,
        rigid_attachment_nodes,
        handedness,
        model_height,
    )

    return {
        "schema_version": 1,
        "skins": len(active_skins),
        "joints": len(joint_nodes),
        "joint_names": joint_names,
        "skinned_primitives": skinned_primitives,
        "unskinned_primitives": unskinned_primitives,
        "skinned_vertices": skinned_vertices,
        "unskinned_vertices": unskinned_vertices,
        "rigid_attachment_primitives": rigid_attachment_primitives,
        "rigid_attachment_vertices": rigid_attachment_vertices,
        "unweighted_vertices": unweighted_vertices,
        "non_normalized_vertices": non_normalized_vertices,
        "invalid_joint_references": invalid_joint_references,
        "maximum_influences": maximum_influences,
        "average_influences": total_influences / skinned_vertices if skinned_vertices else 0.0,
        "rigid_vertices": rigid_vertices,
        "rigid_vertex_ratio": rigid_vertices / skinned_vertices if skinned_vertices else 0.0,
        "blended_vertices": blended_vertices,
        "equal_blend_vertices": equal_blend_vertices,
        "equal_blend_vertex_ratio": (
            equal_blend_vertices / skinned_vertices if skinned_vertices else 0.0
        ),
        "unused_joints": len(unused_joint_names),
        "unused_joint_names": unused_joint_names,
        "joint_weight_coverage": joint_weight_coverage,
        "model_height": round(model_height, 4),
        "bind_pose_landmark_errors": bind_pose_landmark_errors,
        "animations": animations,
        "handedness": handedness,
        "profile": profile,
        "chain_topology": chain_topology,
        "chain_axis_errors": chain_axis_errors,
        "template_conformance": _extract_template_conformance(nodes),
        "joint_containment": joint_containment,
        # Mirrored at top level under the exact key name pipeline_policy.py's
        # `s3.joints_inside_mesh` check asks for, following the same pattern
        # `native.run_canonicalize` uses for `s1_s2_image_similarity_score`.
        "joints_outside_mesh_count": joint_containment.get("joints_outside_mesh_count"),
        "bilateral_symmetry": bilateral_symmetry,
        # Same mirroring convention: the scalars `pipeline_policy`'s
        # `s3.bilateral_symmetry_p95` check reads, lifted to the top level.
        "symmetry_mirror_error_p95": bilateral_symmetry.get("mirror_error_p95"),
        "symmetry_pairs_outside_band_count": bilateral_symmetry.get(
            "pairs_outside_band_count"
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m backend.app.rig_quality <model.glb>")
    print(json.dumps(analyze_rig_path(Path(sys.argv[1])), indent=2))
