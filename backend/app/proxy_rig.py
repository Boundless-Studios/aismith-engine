"""Pure policy helpers for target-derived proxy skinning."""

from collections.abc import Hashable, Iterable
import re

FINGER_BONE_TOKENS = (
    "finger",
    "index",
    "middle",
    "pinky",
    "ring",
    "thumb",
)

LIMB_CHAIN_ALIASES = (
    (
        ("upperarm_l", "leftarm", "upper_arm.l"),
        ("lowerarm_l", "leftforearm", "forearm.l"),
        ("hand_l", "lefthand", "hand.l"),
    ),
    (
        ("upperarm_r", "rightarm", "upper_arm.r"),
        ("lowerarm_r", "rightforearm", "forearm.r"),
        ("hand_r", "righthand", "hand.r"),
    ),
    (
        ("thigh_l", "leftupleg", "thigh.l"),
        ("calf_l", "leftleg", "shin.l"),
        ("foot_l", "leftfoot", "foot.l"),
    ),
    (
        ("thigh_r", "rightupleg", "thigh.r"),
        ("calf_r", "rightleg", "shin.r"),
        ("foot_r", "rightfoot", "foot.r"),
    ),
)

DEFORMATION_SURFACES = frozenset({"visible-original", "voxel-proxy"})
DEFORMATION_PROXY_TARGET_FACES = 25_000
DEFORMATION_PROXY_BAKE_RAY_MULTIPLIER = 1.0


def select_deformation_surface(value: str) -> str:
    """Validate the visible surface used by the target-proxy rig experiment."""
    selected = value.strip()
    if selected not in DEFORMATION_SURFACES:
        raise ValueError(f"Unsupported deformation surface: {selected}")
    return selected


def deformation_surface_can_promote(value: str) -> bool:
    """Only the fidelity-preserving visible surface may replace run state."""
    return select_deformation_surface(value) == "visible-original"


def largest_face_component(neighbors: Iterable[Iterable[int]]) -> set[int]:
    """Return face indices in the largest edge-connected surface component."""
    adjacency = [set(items) for items in neighbors]
    unseen = set(range(len(adjacency)))
    largest: set[int] = set()
    while unseen:
        seed = min(unseen)
        component = {seed}
        frontier = [seed]
        unseen.remove(seed)
        while frontier:
            current = frontier.pop()
            connected = adjacency[current] & unseen
            unseen.difference_update(connected)
            component.update(connected)
            frontier.extend(connected)
        if len(component) > len(largest):
            largest = component
    return largest


def deformation_proxy_decimation_ratio(
    face_count: int,
    target_faces: int = DEFORMATION_PROXY_TARGET_FACES,
) -> float:
    """Return a Blender decimation ratio capped at the proxy face budget."""
    if face_count <= 0 or target_faces <= 0:
        raise ValueError("Deformation proxy requires a positive face count")
    return min(1.0, target_faces / face_count)


def _normalized_bone_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.casefold()).removeprefix("mixamorig")
    return normalized.removeprefix("def")


def resolve_limb_chains(available: Iterable[str]) -> list[tuple[str, str, str]]:
    """Resolve major humanoid limb chains across common donor conventions."""
    by_normalized = {
        _normalized_bone_name(name): name
        for name in available
    }
    resolved = []
    for chain_aliases in LIMB_CHAIN_ALIASES:
        chain = []
        for aliases in chain_aliases:
            bone = next(
                (
                    by_normalized[_normalized_bone_name(alias)]
                    for alias in aliases
                    if _normalized_bone_name(alias) in by_normalized
                ),
                None,
            )
            if bone is None:
                break
            chain.append(bone)
        if len(chain) == 3:
            resolved.append(tuple(chain))
    return resolved


def descendant_names(parent_by_name: dict[str, str | None], ancestor: str) -> list[str]:
    """Return descendants in stable parent-before-child order."""
    result = []
    frontier = [ancestor]
    while frontier:
        parent = frontier.pop(0)
        children = sorted(
            name for name, candidate in parent_by_name.items()
            if candidate == parent
        )
        result.extend(children)
        frontier.extend(children)
    return result


def select_action_name(available: Iterable[str], requested: str) -> str:
    """Prefer an exact donor action, then one unique importer-suffixed variant."""
    names = list(available)
    if requested in names:
        return requested
    importer_name = f"{requested}_Armature"
    if importer_name in names:
        return importer_name
    importer_collisions = [
        name for name in names
        if re.fullmatch(rf"{re.escape(importer_name)}\.\d+", name)
    ]
    if len(importer_collisions) == 1:
        return importer_collisions[0]
    suffixed = [name for name in names if name.startswith(f"{requested}_")]
    if len(suffixed) != 1:
        raise ValueError(
            f"Donor action '{requested}' was not found exactly once; "
            f"available actions: {', '.join(sorted(names))}"
        )
    return suffixed[0]


def resolve_right_hand_bone(available: Iterable[str]) -> str:
    """Resolve the concrete right-hand bone used by common humanoid rigs."""
    names = list(available)
    candidates = []
    for name in names:
        normalized = re.sub(r"[^a-z0-9]+", "", name.casefold())
        if normalized in {"handr", "righthand"} or normalized.endswith("righthand"):
            candidates.append(name)
    if not candidates:
        raise ValueError("The donor skeleton has no supported right-hand bone")
    return min(
        candidates,
        key=lambda name: (
            name != "hand_r",
            any(token in name.casefold() for token in ("helper", "control", "leaf")),
            len(name),
        ),
    )


def is_proxy_deform_bone(name: str) -> bool:
    """Return whether a coarse generated proxy should receive this bone's weights."""
    normalized = name.casefold()
    if normalized == "root" or "leaf" in normalized:
        return False
    return not any(token in normalized for token in FINGER_BONE_TOKENS)


def is_finger_deform_bone(name: str) -> bool:
    """Return whether a bone is a deforming finger segment, excluding tips."""
    normalized = name.casefold()
    return (
        "leaf" not in normalized
        and any(token in normalized for token in FINGER_BONE_TOKENS)
    )


def blend_local_finger_weights(
    current: dict[str, float],
    finger_distances: dict[str, float],
    *,
    hand_bone: str,
    maximum_distance: float,
) -> dict[str, float]:
    """Move local hand influence to the nearest finger without leaking elsewhere."""
    if maximum_distance <= 0:
        raise ValueError("Finger weighting requires a positive distance")
    nearest = min(finger_distances.items(), key=lambda item: item[1], default=None)
    if nearest is None or nearest[1] >= maximum_distance:
        return dict(current)
    hand_weight = current.get(hand_bone, 0.0)
    if hand_weight <= 0:
        return dict(current)
    finger_share = hand_weight * 0.9 * (1.0 - nearest[1] / maximum_distance)
    result = dict(current)
    result[hand_bone] = hand_weight - finger_share
    result[nearest[0]] = result.get(nearest[0], 0.0) + finger_share
    return {name: round(weight, 12) for name, weight in result.items() if weight > 0}


def add_local_finger_coverage(
    current: dict[str, float],
    *,
    target_bone: str,
    side: str,
    share: float = 0.15,
) -> dict[str, float]:
    """Give an unused finger segment bounded influence from its nearest local donor."""
    if side not in {"l", "r"} or not 0 < share < 1:
        raise ValueError("Finger coverage requires a side and fractional share")
    donors = [
        (name, weight)
        for name, weight in current.items()
        if name != target_bone
        and weight > 0
        and (name.casefold() == f"hand_{side}" or (
            is_finger_deform_bone(name)
            and name.casefold().endswith(f"_{side}")
        ))
    ]
    if not donors:
        return dict(current)
    donor, donor_weight = max(donors, key=lambda item: item[1])
    transferred = donor_weight * share
    result = dict(current)
    result[donor] = donor_weight - transferred
    result[target_bone] = result.get(target_bone, 0.0) + transferred
    return {name: round(weight, 12) for name, weight in result.items() if weight > 0}


def weighted_centroid(
    samples: Iterable[tuple[tuple[float, float, float], float]],
) -> tuple[float, float, float]:
    """Return the centroid used to anchor a prop to transferred skin weights."""
    totals = [0.0, 0.0, 0.0]
    total_weight = 0.0
    for point, weight in samples:
        if weight <= 0:
            continue
        total_weight += weight
        for axis in range(3):
            totals[axis] += point[axis] * weight
    if total_weight <= 0:
        raise ValueError("A weighted centroid requires at least one positive sample")
    return tuple(value / total_weight for value in totals)


def sword_component_offsets(
    *,
    blade_length: float,
    grip_length: float,
) -> dict[str, float]:
    """Center the handle on the hand and start the blade beyond its guard."""
    if blade_length <= 0 or grip_length <= 0:
        raise ValueError("Sword components require positive lengths")
    guard = grip_length * 0.5
    return {
        "blade": guard + blade_length * 0.5,
        "guard": guard,
        "grip": 0.0,
    }


def chain_endpoints_from_centroids(
    centroids: Iterable[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Estimate connected joint positions around ordered weighted regions."""
    points = [tuple(float(value) for value in point) for point in centroids]
    if len(points) < 2:
        raise ValueError("A fitted bone chain requires at least two centroids")

    def midpoint(first, second):
        return tuple((first[axis] + second[axis]) * 0.5 for axis in range(3))

    boundaries = [midpoint(first, second) for first, second in zip(points, points[1:])]
    first = tuple(2 * points[0][axis] - boundaries[0][axis] for axis in range(3))
    last = tuple(2 * points[-1][axis] - boundaries[-1][axis] for axis in range(3))
    return [first, *boundaries, last]


def unique_face_indices(faces: Iterable[Iterable[int]]) -> list[int]:
    """Keep one triangle for each coincident surface, regardless of winding."""
    seen: set[tuple[int, ...]] = set()
    kept: list[int] = []
    for index, face in enumerate(faces):
        key = tuple(sorted(face))
        if key in seen:
            continue
        seen.add(key)
        kept.append(index)
    return kept


def unique_position_face_indices(
    vertices: Iterable[Iterable[float]],
    faces: Iterable[Iterable[int]],
    *,
    tolerance: float = 1e-7,
    face_groups: Iterable[Hashable] | None = None,
    preserve_opposite_winding_groups: set[Hashable] | None = None,
    seen_keys: set[tuple[Hashable, tuple, tuple]] | None = None,
) -> list[int]:
    """Keep one face per material surface while respecting culled card winding."""
    if tolerance <= 0:
        raise ValueError("Position tolerance must be positive")
    quantized = [
        tuple(round(float(value) / tolerance) for value in vertex)
        for vertex in vertices
    ]
    position_faces = [tuple(quantized[index] for index in face) for face in faces]
    groups = list(face_groups) if face_groups is not None else [0] * len(position_faces)
    if len(groups) != len(position_faces):
        raise ValueError("Face groups must match the face count")
    winding_groups = preserve_opposite_winding_groups or set()
    seen = seen_keys if seen_keys is not None else set()
    kept = []
    for index, (face, group) in enumerate(zip(position_faces, groups, strict=True)):
        if len(face) == 3:
            surface = tuple(sorted(face))
        else:
            rotations = [face[offset:] + face[:offset] for offset in range(len(face))]
            if group not in winding_groups:
                reversed_face = tuple(reversed(face))
                rotations.extend(
                    reversed_face[offset:] + reversed_face[:offset]
                    for offset in range(len(reversed_face))
                )
            surface = min(rotations)
        if group in winding_groups and len(face) == 3:
            rotations = [face[offset:] + face[:offset] for offset in range(len(face))]
            winding = min(rotations)
        else:
            winding = ()
        key = (group, surface, winding)
        if key in seen:
            continue
        seen.add(key)
        kept.append(index)
    return kept


def face_render_signature(
    positions: Iterable[Iterable[float]],
    corner_attributes: Iterable[Iterable[float]],
    *,
    tolerance: float = 1e-7,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Describe a rendered face independent of corner order."""
    if tolerance <= 0:
        raise ValueError("Render-signature tolerance must be positive")
    corners = list(zip(positions, corner_attributes, strict=True))
    return tuple(sorted(
        (
            tuple(round(float(value) / tolerance) for value in position),
            tuple(round(float(value) / tolerance) for value in attributes),
        )
        for position, attributes in corners
    ))
