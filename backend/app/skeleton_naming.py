"""Facing-aware semantic bone naming for UniRig-predicted skeletons.

Pure numpy logic (no bpy) so it can be unit tested on the host and imported
by the bpy-hosted bind worker (`backend/workers/unirig_bind_worker.py`).

Coordinate frame: UniRig's extract stage imports meshes via Blender, so the
skeleton/joint arrays this module receives are in Blender's Z-up convention:
vertical axis = Z (index 2), side axis = X (index 0), the remaining axis
(index 1, "Y") is the character's forward/depth axis.

Handedness bug this module fixes: naming used to assign the "r" suffix to
+X unconditionally, which is only correct for a +Y-facing character. A
character facing -Y has its left side at +X. This module derives the
character's actual facing direction from its feet and assigns l/r from
`left = up x forward` instead of from a fixed axis sign.
"""
from __future__ import annotations

import math
import re
from collections import deque
from typing import Callable, Collection, NamedTuple, Sequence

import numpy as np

# N3: template-conformant humanoid skeleton building (see the bottom of this
# module) reuses the biped family template's bone list/hierarchy/rest
# geometry. `skeleton_template.py` carries the same bpy-free, no-scipy
# constraint this module does (both are imported inside UniRig's bundled
# bpy venv), so importing it here does not add a new deployment risk.
from .skeleton_template import Template

VERTICAL_AXIS = 2
SIDE_AXIS = 0
_DEFAULT_FORWARD = np.array([0.0, -1.0, 0.0])
_EPS = 1e-9

# --- Chain-rig profile (non-humanoid creatures: serpents, tails, tentacles) ---
#
# A branch hanging off the skeleton's main path with fewer than this many
# joints is a prunable "spur" (a fin, whisker, or forked tail tip); a branch
# at or above this length is a real limb, which disqualifies "chain"
# classification entirely (see detect_rig_profile).
_MIN_SPUR_LENGTH = 3
# The skeleton's longest principal axis must have at least this many times
# the extent of its second-longest axis to be considered "elongated" (a
# necessary chain trait: humanoids are not this lopsided along any axis).
_CHAIN_ELONGATION_RATIO = 1.5
# This module's own chain-naming convention zero-pads to (at least) two
# digits ("spine_00", "spine_01", ...), which never collides with the
# humanoid path's "spine_1"/"spine_2"/"spine_3" (see _CHAIN_CONTINUATIONS
# below) -- that's what lets detect_profile_from_names tell a chain rig's
# exported bone names apart from a humanoid's.
_CHAIN_SPINE_NAME_PATTERN = re.compile(r"^spine_(\d{2,})$")
_HUMANOID_LIMB_NAME_TOKENS = (
    "upperarm", "lowerarm", "forearm", "hand",
    "thigh", "upleg", "calf", "shin", "foot",
)

# Confidence thresholds for the toe-direction facing measurement (see
# detect_forward_from_feet). Two feet whose toe directions disagree by more
# than this many degrees, or whose combined (averaged) horizontal magnitude
# is too small relative to either foot's own segment length or the
# skeleton's height, are not trusted as a confident facing measurement.
_MAX_FOOT_DISAGREEMENT_DEGREES = 90.0
_MIN_RELATIVE_TO_FOOT_LENGTH = 0.25
_MIN_RELATIVE_TO_SKELETON_HEIGHT = 0.02

# A bone with more than one child needs a deliberate choice of which child
# its tail should point at (see select_chain_continuation_child): prefer the
# child that continues the named anatomical chain.
_CHAIN_CONTINUATIONS = {
    "pelvis": "spine_1",
    "spine_1": "spine_2",
    "spine_2": "spine_3",
    "spine_3": "neck",
    "neck": "head",
}


class FacingResult(NamedTuple):
    forward: np.ndarray
    source: str
    warning: str | None


class SkeletonNaming(NamedTuple):
    names: list[str]
    forward: np.ndarray
    facing_source: str
    warning: str | None
    profile: str = "humanoid"
    profile_source: str = "detected"


class ResolvedOverrides(NamedTuple):
    resolved: dict[int, np.ndarray]
    unresolved: list[str]
    used_fallback: bool


# A rig's naming can change between two binds of the same skeleton (e.g. this
# facing fix). Confident sources are trustworthy enough to gate a
# certificate on; low-confidence/assumed ones are measurements of last
# resort and should not block anything downstream.
CONFIDENT_FACING_SOURCES = frozenset({"foot-toe-direction", "foot-region-vertices"})


def _horizontal(vector: np.ndarray, vertical_axis: int) -> np.ndarray:
    horizontal = vector.copy()
    horizontal[vertical_axis] = 0.0
    return horizontal


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return None
    return vector / norm


def detect_forward_from_feet(
    joints: np.ndarray,
    foot_indices: Sequence[int],
    foot_children: Sequence[Sequence[int]],
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None = None,
    *,
    vertical_axis: int = VERTICAL_AXIS,
    default_forward: np.ndarray = _DEFAULT_FORWARD,
) -> FacingResult:
    """Detect a humanoid's forward direction from its feet.

    Axis-parameterized so it can be reused against skeletons in different
    "up" conventions (Blender Z-up joint arrays here, glTF Y-up exported
    rigs in `rig_quality.py`).

    Primary: horizontal mean, over both legs, of (toe joint - foot joint)
    when a foot has a child joint. This is only trusted as confident when
    the two feet roughly agree (within `_MAX_FOOT_DISAGREEMENT_DEGREES`) and
    the averaged vector isn't a near-cancellation of disagreeing feet (its
    length is at least `_MIN_RELATIVE_TO_FOOT_LENGTH` of the mean per-foot
    segment length, or `_MIN_RELATIVE_TO_SKELETON_HEIGHT` of the skeleton's
    height) — otherwise two feet pointing ~180 degrees apart would average
    to a near-zero vector that normalization turns into a confident but
    meaningless direction.
    Fallback: mesh vertex centroid below ankle height, minus mean ankle
    position, horizontal component. `mesh_vertices` may be a plain array or
    a zero-arg callable that builds one lazily — only invoked when the toe
    measurement above isn't confident, since building a vertex cloud can be
    expensive for dense meshes.
    Final fallback: `default_forward`, with a warning — callers that need to
    distinguish a real measurement from this fallback should check
    `.source in CONFIDENT_FACING_SOURCES`.
    """
    per_foot_directions = []
    for foot_index, children in zip(foot_indices, foot_children):
        if not children:
            continue
        toe_index = children[0]
        per_foot_directions.append(_horizontal(joints[toe_index] - joints[foot_index], vertical_axis))

    weak_result: FacingResult | None = None
    if per_foot_directions:
        mean_direction = np.mean(per_foot_directions, axis=0)
        normalized = _normalize(mean_direction)
        if normalized is not None:
            per_foot_lengths = [float(np.linalg.norm(d)) for d in per_foot_directions]
            mean_length = float(np.linalg.norm(mean_direction))
            mean_foot_length = float(np.mean(per_foot_lengths))
            skeleton_height = float(np.ptp(joints[:, vertical_axis]))
            magnitude_ok = (
                (mean_foot_length > _EPS and mean_length >= _MIN_RELATIVE_TO_FOOT_LENGTH * mean_foot_length)
                or (skeleton_height > _EPS and mean_length >= _MIN_RELATIVE_TO_SKELETON_HEIGHT * skeleton_height)
            )
            agreement_ok = True
            if len(per_foot_directions) >= 2:
                unit_directions = [d for d in (_normalize(v) for v in per_foot_directions) if d is not None]
                if len(unit_directions) >= 2:
                    min_cosine = min(
                        float(np.dot(unit_directions[i], unit_directions[j]))
                        for i in range(len(unit_directions))
                        for j in range(i + 1, len(unit_directions))
                    )
                    agreement_ok = min_cosine >= math.cos(math.radians(_MAX_FOOT_DISAGREEMENT_DEGREES))
            if magnitude_ok and agreement_ok:
                return FacingResult(normalized, "foot-toe-direction", None)
            weak_result = FacingResult(
                normalized,
                "foot-toe-direction-low-confidence",
                "feet toe directions disagree or are too small relative to foot/skeleton "
                "size to trust confidently; falling back to a weaker facing signal",
            )

    resolved_mesh_vertices = mesh_vertices() if callable(mesh_vertices) else mesh_vertices
    if resolved_mesh_vertices is not None and len(resolved_mesh_vertices) and len(foot_indices):
        ankle_positions = joints[list(foot_indices)]
        ankle_mean = np.mean(ankle_positions, axis=0)
        ankle_vertical = float(np.mean(ankle_positions[:, vertical_axis]))
        below_ankle = resolved_mesh_vertices[resolved_mesh_vertices[:, vertical_axis] < ankle_vertical]
        if len(below_ankle):
            centroid = np.mean(below_ankle, axis=0)
            direction = _horizontal(centroid - ankle_mean, vertical_axis)
            normalized = _normalize(direction)
            if normalized is not None:
                return FacingResult(normalized, "foot-region-vertices", None)

    if weak_result is not None:
        return weak_result

    warning = (
        "could not detect character facing from toe joints or mesh vertices; "
        f"assuming {default_forward.tolist()} forward"
    )
    return FacingResult(default_forward.copy(), "assumed", warning)


def left_from_forward(
    forward: np.ndarray,
    *,
    vertical_axis: int = VERTICAL_AXIS,
    fallback_left: np.ndarray | None = None,
) -> np.ndarray:
    """`left = up x forward`, degenerate-safe.

    `up` is the unit vector along `vertical_axis`. When `forward` is
    (anti)parallel to `up` the cross product is zero; fall back to
    `fallback_left` (default: +X, or +Y if `vertical_axis` is itself X)
    rather than dividing by zero.
    """
    up = np.zeros(3)
    up[vertical_axis] = 1.0
    left = _normalize(np.cross(up, forward))
    if left is not None:
        return left
    if fallback_left is not None:
        return fallback_left
    default = np.array([1.0, 0.0, 0.0])
    default[vertical_axis] = 0.0
    normalized_default = _normalize(default)
    if normalized_default is not None:
        return normalized_default
    # vertical_axis == 0 (X is "up"): the +X default above is degenerate
    # too; +Y is never parallel to an X-axis "up".
    return np.array([0.0, 1.0, 0.0])


def _subtree_size(node: int, children: dict[int, list[int]]) -> int:
    """Count of `node` plus every descendant, following `children`."""
    stack = [node]
    size = 0
    while stack:
        current = stack.pop()
        size += 1
        stack.extend(children[current])
    return size


def _dfs_order(node: int, children: dict[int, list[int]]) -> list[int]:
    """`node` plus every descendant, in a deterministic top-down order."""
    order: list[int] = []
    stack = [node]
    while stack:
        current = stack.pop(0)
        order.append(current)
        stack.extend(children[current])
    return order


def _build_adjacency(count: int, parents: Sequence[int | None]) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for child, parent in enumerate(parents):
        if parent is None or parent < 0 or parent == child:
            continue
        parent = int(parent)
        adjacency[parent].append(child)
        adjacency[child].append(parent)
    return adjacency


def _bfs_distances(start: int, adjacency: dict[int, list[int]]) -> tuple[int, dict[int, int], dict[int, int | None]]:
    """BFS from `start`. Returns (farthest_node, distances, predecessors)."""
    distance: dict[int, int] = {start: 0}
    predecessor: dict[int, int | None] = {start: None}
    queue: deque[int] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                predecessor[neighbor] = node
                queue.append(neighbor)
                if distance[neighbor] > distance[farthest]:
                    farthest = neighbor
    return farthest, distance, predecessor


def _tree_diameter_path(count: int, parents: Sequence[int | None]) -> list[int]:
    """The longest path through the skeleton, treated as an undirected tree.

    UniRig's designated root is not necessarily either anatomical end of a
    chain creature -- it can sit mid-body, with the true head/tail chain
    passing *through* it via two of its branches. The standard two-BFS
    algorithm for a tree's diameter finds that path regardless of where the
    root sits; a root at an actual end (the common case for hand-built or
    synthetic chains) is just the degenerate case where the path never
    "turns" through a branch point.
    """
    if count == 0:
        return []
    adjacency = _build_adjacency(count, parents)
    one_end, _, _ = _bfs_distances(0, adjacency)
    other_end, _, predecessor = _bfs_distances(one_end, adjacency)
    path = [other_end]
    node = other_end
    while predecessor[node] is not None:
        node = predecessor[node]
        path.append(node)
    path.reverse()
    return path


def _component_excluding_edge(start: int, blocked: int, adjacency: dict[int, list[int]]) -> list[int]:
    """Every node reachable from `start` without crossing the edge back to
    `blocked` -- i.e. the whole branch/spur component on `start`'s side
    once that single edge is cut. Deterministic BFS order, `start` first.
    """
    visited = {blocked, start}
    order = [start]
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                order.append(neighbor)
                queue.append(neighbor)
    return order


def _chain_spurs(
    main_path: list[int],
    adjacency: dict[int, list[int]],
    min_spur_length: int,
) -> tuple[bool, dict[int, tuple[int, list[int]]]]:
    """Classify every branch hanging off `main_path` (the tree's diameter).

    A branch is a prunable "spur" when it has fewer than `min_spur_length`
    joints; a branch at or above that length is a real limb, which
    disqualifies chain classification entirely (`is_chain` becomes False).

    Checks each path node's *full* neighbor set (via undirected
    `adjacency`), not just its rooted children: an interior path node's
    two path-adjacent neighbors (prev and next) already account for both
    directions the path can go, whichever is parent and whichever is
    child -- but a path *endpoint* (`main_path[0]` or `main_path[-1]`) has
    only one path-adjacent neighbor, so its other tree-neighbor (if any --
    e.g. the skeleton's actual root sitting one step further up, off the
    diameter entirely) would otherwise go unclassified forever, exported
    as a stray un-renamed bone. Returns spur_root -> (the main_path node it
    attaches to, its component nodes in BFS order) so callers can name it
    relative to that attachment point regardless of which direction
    (parent or child) the branch happens to leave the path from.
    """
    spurs: dict[int, tuple[int, list[int]]] = {}
    for position, node in enumerate(main_path):
        prev_node = main_path[position - 1] if position > 0 else None
        next_node = main_path[position + 1] if position < len(main_path) - 1 else None
        for neighbor in adjacency[node]:
            if neighbor == prev_node or neighbor == next_node:
                continue
            component = _component_excluding_edge(neighbor, node, adjacency)
            if len(component) >= min_spur_length:
                return False, {}
            spurs[neighbor] = (node, component)
    return True, spurs


def detect_rig_profile(
    joints: np.ndarray,
    parents: Sequence[int | None],
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None = None,
) -> str:
    """Classify a predicted skeleton as `"humanoid"` or `"chain"`.

    A chain (e.g. a serpent) has no real limb branches: once branches
    shorter than `_MIN_SPUR_LENGTH` joints are pruned as spurs (a fin,
    whisker, or forked tail tip), what remains is a single path (the
    skeleton's tree diameter -- see `_tree_diameter_path`), and the
    creature is elongated: its extent along its longest principal axis
    dominates the other two, unlike a humanoid's proportions.

    Elongation is measured from the joints first (cheap, always
    available); a handful of predicted joints can understate a mesh's true
    elongation when spur branches (pruned above, but still part of the
    joint cloud) pull spread into a second direction, so `mesh_vertices` --
    resolved lazily, only when the joint-based measurement is inconclusive,
    since building a vertex cloud can be expensive -- is tried as a second,
    more reliable opinion before giving up on "chain".
    """
    count = len(joints)
    if count < 4:
        return "humanoid"
    roots = [i for i, parent in enumerate(parents) if parent is None or parent < 0 or parent == i]
    if len(roots) != 1:
        return "humanoid"
    main_path = _tree_diameter_path(count, parents)
    if len(main_path) < 4:
        return "humanoid"
    adjacency = _build_adjacency(count, parents)
    is_chain, _spurs = _chain_spurs(main_path, adjacency, _MIN_SPUR_LENGTH)
    if not is_chain:
        return "humanoid"
    if _is_elongated(joints):
        return "chain"
    resolved_vertices = _resolve_lazy_vertices(mesh_vertices)
    if resolved_vertices is not None and _is_elongated(resolved_vertices):
        return "chain"
    return "humanoid"


def _is_elongated(points: np.ndarray) -> bool:
    if len(points) < 2:
        return False
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) < 2:
        return False
    if singular_values[1] <= _EPS:
        return bool(singular_values[0] > _EPS)
    return bool(singular_values[0] >= _CHAIN_ELONGATION_RATIO * singular_values[1])


class ChainCandidateScore(NamedTuple):
    """How good a single UniRig prediction (one seed) is as a chain
    skeleton -- used to pick the best of several candidate seeds before
    committing to a naming/bind. Sort candidates by
    `(not disqualified, main_chain_length, -spur_count)` descending: never
    prefer a disqualified (genuinely branching) candidate, then prefer the
    longest unbranched main chain, then fewest spurs among ties.
    """
    main_chain_length: int
    spur_count: int
    disqualified: bool


def score_chain_candidate(
    joints: np.ndarray,
    parents: Sequence[int | None],
    *,
    min_spur_length: int = _MIN_SPUR_LENGTH,
) -> ChainCandidateScore:
    """Score one predicted skeleton as a chain candidate, independent of
    the elongation check `detect_rig_profile` also applies -- callers
    comparing several seeds of the *same* mesh already know it's chain-
    shaped and only need to compare topology quality across candidates.
    """
    count = len(joints)
    if count < 2:
        return ChainCandidateScore(0, 0, True)
    roots = [i for i, parent in enumerate(parents) if parent is None or parent < 0 or parent == i]
    if len(roots) != 1:
        return ChainCandidateScore(0, 0, True)
    main_path = _tree_diameter_path(count, parents)
    adjacency = _build_adjacency(count, parents)
    is_chain, spurs = _chain_spurs(main_path, adjacency, min_spur_length)
    return ChainCandidateScore(len(main_path), len(spurs), not is_chain)


def _catmull_rom_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _catmull_rom_resample(
    control_points: np.ndarray,
    target_count: int,
    samples_per_segment: int = 50,
) -> np.ndarray:
    """Resample a polyline through `control_points` via a uniform
    Catmull-Rom spline, at `target_count` points evenly spaced by *arc
    length* along the curve (not by spline parameter -- segments between
    unevenly-spaced control points would otherwise bunch samples). The
    first and last output points exactly match `control_points[0]` and
    `control_points[-1]`.
    """
    count = len(control_points)
    if count < 2:
        raise ValueError("need at least 2 control points to resample a chain")
    if target_count < 2:
        raise ValueError("target_count must be at least 2 (a head and a tail)")
    if count == 2:
        dense = np.linspace(control_points[0], control_points[1], samples_per_segment + 1)
    else:
        dense_points = [control_points[0]]
        for i in range(count - 1):
            p0 = control_points[i - 1] if i - 1 >= 0 else control_points[0]
            p1 = control_points[i]
            p2 = control_points[i + 1]
            p3 = control_points[i + 2] if i + 2 < count else control_points[-1]
            # t=0 duplicates the previous segment's endpoint; skip it.
            for t in np.linspace(0.0, 1.0, samples_per_segment + 1)[1:]:
                dense_points.append(_catmull_rom_point(p0, p1, p2, p3, float(t)))
        dense = np.asarray(dense_points)
    deltas = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = float(cumulative[-1])
    if total_length <= _EPS:
        return np.tile(control_points[0], (target_count, 1))
    targets = np.linspace(0.0, total_length, target_count)
    resampled = np.empty((target_count, 3))
    for axis in range(3):
        resampled[:, axis] = np.interp(targets, cumulative, dense[:, axis])
    # Arc-length interpolation can accumulate floating point drift at the
    # exact endpoints; pin them to the true control points.
    resampled[0] = control_points[0]
    resampled[-1] = control_points[-1]
    return resampled


def _slab_centroid(
    position: np.ndarray,
    axis_direction: np.ndarray,
    vertices: np.ndarray,
    half_thickness: float,
) -> np.ndarray | None:
    """Centroid of mesh vertices within `half_thickness` of `position`
    along `axis_direction` -- a slab perpendicular to the local chain
    direction, approximating the mesh's true medial-axis point there.
    Returns None when no vertices fall in the slab (position is kept as-is)."""
    if vertices is None or len(vertices) == 0 or half_thickness <= _EPS:
        return None
    offsets = vertices - position
    along = offsets @ axis_direction
    slab = vertices[np.abs(along) <= half_thickness]
    if len(slab) == 0:
        return None
    return np.mean(slab, axis=0)


def slab_axis_deviation(
    position: np.ndarray,
    axis_direction: np.ndarray,
    vertices: np.ndarray,
    half_thickness: float,
) -> tuple[float, float] | None:
    """(local_radius, axis_deviation) for a genuine *geometric* slab of
    mesh vertices around `position`, perpendicular to `axis_direction` --
    the public form of the same slab math `densify_chain`/
    `medial_axis_chain` use, for callers like the rig certificate's
    off-axis check that need it against an *exported* rig's own mesh.

    Deliberately geometric, not skinning-weight-based: which vertices a
    bone-heat/envelope solver decided to *weight* to a joint is not the
    same as which vertices sit *near* it in space -- weight functions can
    be smooth and wide, especially on a long thin chain where few joints
    compete for influence, so "the joint's dominantly-weighted vertices"
    can span a stretch far longer than the joint's true local
    neighborhood. A spatial slab has no such conflation.

    `local_radius` is the mean perpendicular-to-axis distance of slab
    vertices from `position` (the local cross-section radius); axis_
    deviation is the magnitude of their mean perpendicular offset (how far
    off-axis the slab's own centroid sits from `position`). Returns None
    when no vertices fall in the slab.
    """
    if vertices is None or len(vertices) == 0 or half_thickness <= _EPS:
        return None
    offsets = vertices - position
    along = offsets @ axis_direction
    slab_mask = np.abs(along) <= half_thickness
    if not np.any(slab_mask):
        return None
    perpendicular = offsets[slab_mask] - np.outer(along[slab_mask], axis_direction)
    radius = float(np.mean(np.linalg.norm(perpendicular, axis=1)))
    deviation = float(np.linalg.norm(np.mean(perpendicular, axis=0)))
    return radius, deviation


class DensifiedChain(NamedTuple):
    joints: np.ndarray
    parents: list[int | None]
    names: list[str]


def densify_chain(
    joints: np.ndarray,
    parents: Sequence[int | None],
    target_joint_count: int,
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None = None,
    *,
    min_spur_length: int = _MIN_SPUR_LENGTH,
) -> DensifiedChain:
    """Resample a predicted chain skeleton's main path into
    `target_joint_count` evenly-by-arc-length spaced joints.

    UniRig's raw prediction is often too sparse (few joints) and locally
    noisy (a joint's position doesn't sit exactly on the mesh's medial
    line) for smooth chain deformation. This fits a Catmull-Rom spline
    through the original predicted main-path joints (spurs dropped
    entirely), resamples it evenly by arc length into `target_joint_count`
    points, then snaps each point to the centroid of the mesh's local
    cross-section slab there (reusing the same slab-around-a-joint idea as
    the certificate's `chain_axis_errors`) -- the true medial-axis
    position, rather than trusting the spline (or the original noisy
    joints) directly. `head` is kept at the head end (the same thickness
    heuristic `_name_chain_profile` uses); the new skeleton is a plain
    linear chain, tail (`spine_00`) to `head`.
    """
    if target_joint_count < 2:
        raise ValueError("target_joint_count must be at least 2 (a head and a tail)")
    count = len(joints)
    roots = [i for i, parent in enumerate(parents) if parent is None or parent < 0 or parent == i]
    if len(roots) != 1:
        raise ValueError("densify_chain requires a single-root skeleton")
    main_path = _tree_diameter_path(count, parents)
    if len(main_path) < 2:
        raise ValueError("densify_chain requires a main chain of at least 2 joints")
    adjacency = _build_adjacency(count, parents)
    is_chain, _spurs = _chain_spurs(main_path, adjacency, min_spur_length)
    if not is_chain:
        raise ValueError("densify_chain requires an unbranched chain topology")

    end_a, end_b = main_path[0], main_path[-1]
    _, distances_from_root, _ = _bfs_distances(roots[0], adjacency)
    head_index = end_a if distances_from_root.get(end_a, 0) <= distances_from_root.get(end_b, 0) else end_b

    resolved_vertices = _resolve_lazy_vertices(mesh_vertices)
    if resolved_vertices is not None and len(main_path) >= 2:
        segment_lengths = [
            float(np.linalg.norm(joints[main_path[i + 1]] - joints[main_path[i]]))
            for i in range(len(main_path) - 1)
        ]
        mean_segment = float(np.mean(segment_lengths)) if segment_lengths else 0.0
        thickness_radius = max(mean_segment * 2.0, _EPS)
        direction_a = _normalize(joints[main_path[1]] - joints[end_a])
        direction_b = _normalize(joints[end_b] - joints[main_path[-2]])
        if direction_a is not None and direction_b is not None:
            radius_a = _local_cross_section_radius(joints[end_a], direction_a, resolved_vertices, thickness_radius)
            radius_b = _local_cross_section_radius(joints[end_b], direction_b, resolved_vertices, thickness_radius)
            if radius_a != radius_b:
                head_index = end_a if radius_a > radius_b else end_b

    ordered_from_tail = main_path if head_index == end_b else list(reversed(main_path))
    control_points = joints[ordered_from_tail]

    resampled = _catmull_rom_resample(control_points, target_joint_count)

    if resolved_vertices is not None and len(resolved_vertices):
        total_length = float(np.linalg.norm(np.diff(resampled, axis=0), axis=1).sum())
        slab_half_thickness = max(
            total_length / max(target_joint_count - 1, 1) * 0.5,
            _EPS,
        )
        for index in range(target_joint_count):
            if target_joint_count == 1:
                direction = None
            elif index == 0:
                direction = _normalize(resampled[1] - resampled[0])
            elif index == target_joint_count - 1:
                direction = _normalize(resampled[-1] - resampled[-2])
            else:
                direction = _normalize(resampled[index + 1] - resampled[index - 1])
            if direction is None:
                continue
            centroid = _slab_centroid(resampled[index], direction, resolved_vertices, slab_half_thickness)
            if centroid is not None:
                resampled[index] = centroid

    new_parents: list[int | None] = [None] + list(range(target_joint_count - 1))
    names = [f"spine_{i:02d}" for i in range(target_joint_count - 1)] + ["head"]
    return DensifiedChain(resampled, new_parents, names)


def _principal_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mesh PCA: (centroid, unit first-principal-component direction,
    extent -- the range of vertex projections onto that direction)."""
    mean = vertices.mean(axis=0)
    centered = vertices - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projections = centered @ direction
    extent = float(projections.max() - projections.min())
    return mean, direction, extent


def mesh_is_elongated(vertices: np.ndarray) -> bool:
    """Public wrapper on the same elongation test `detect_rig_profile` uses
    internally, for callers (e.g. the bind worker) deciding whether a mesh
    alone -- independent of any predicted skeleton -- looks chain-shaped."""
    return _is_elongated(vertices)


def chain_candidate_extent_ratio(
    joints: np.ndarray,
    parents: Sequence[int | None],
    mesh_vertices: np.ndarray,
    *,
    min_spur_length: int = _MIN_SPUR_LENGTH,
) -> float:
    """Ratio of a predicted chain candidate's main-chain endpoint-to-
    endpoint span to the mesh's own principal-axis extent.

    `densify_chain` can only resample *between* a candidate's existing
    joints -- it cannot extend a chain that UniRig bunched into a short
    cluster (e.g. a small "star" of spurs at mid-body spanning a fraction
    of a long body). This ratio is how a caller detects that case before
    committing to densify_chain, rather than after finding out its result
    still only covers a fraction of the mesh.
    """
    count = len(joints)
    if count < 2:
        return 0.0
    main_path = _tree_diameter_path(count, parents)
    if len(main_path) < 2:
        return 0.0
    span = float(np.linalg.norm(joints[main_path[-1]] - joints[main_path[0]]))
    _, _, mesh_extent = _principal_axis(mesh_vertices)
    if mesh_extent <= _EPS:
        return 0.0
    return span / mesh_extent


class MedialAxisChain(NamedTuple):
    joints: np.ndarray
    parents: list[int | None]
    names: list[str]


def medial_axis_chain(
    mesh_vertices: np.ndarray | Callable[[], np.ndarray],
    target_joint_count: int,
    *,
    slab_thickness_ratio: float = 0.5,
) -> MedialAxisChain:
    """Build a chain skeleton directly from mesh geometry, independent of
    any predicted skeleton -- for when a prediction is too degenerate to
    densify (see `chain_candidate_extent_ratio`).

    Marches along the body via curved-slab sweeps: starting at one end
    (by PCA), each step advances along the *current* local direction,
    snaps to the nearest mesh vertex (so it can't wander into empty space
    across a curve), takes the centroid of vertices in a slab around that
    point perpendicular to the current direction, then updates the
    direction from the last two centroids -- so the walk follows an
    S-curve instead of the straight PCA axis. The resulting centerline is
    resampled evenly by arc length into `target_joint_count` joints (reusing
    the same Catmull-Rom resampling as `densify_chain`). Head is the
    thicker end (larger mean local cross-section radius over the last 15%
    of arc length at each end); `spine_00..` runs tail to `head`.
    """
    if target_joint_count < 2:
        raise ValueError("target_joint_count must be at least 2 (a head and a tail)")
    resolved_vertices = _resolve_lazy_vertices(mesh_vertices)
    if resolved_vertices is None or len(resolved_vertices) < 4:
        raise ValueError("medial_axis_chain requires a non-trivial vertex cloud")

    mean, principal_direction, extent = _principal_axis(resolved_vertices)
    if extent <= _EPS:
        raise ValueError("mesh has no discernible principal axis (degenerate extent)")

    projections = (resolved_vertices - mean) @ principal_direction
    start_vertex = resolved_vertices[int(np.argmin(projections))]

    step = extent / max(target_joint_count * 2, 1)
    slab_half_thickness = max(step * slab_thickness_ratio, _EPS)

    def slab_centroid_at(position: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        return _slab_centroid(position, direction, resolved_vertices, slab_half_thickness)

    current_direction = principal_direction
    current_position = slab_centroid_at(start_vertex, current_direction)
    if current_position is None:
        current_position = start_vertex.copy()
    centerline = [current_position]

    max_steps = target_joint_count * 4  # generous cap; arc-length resample trims to N
    total_travelled = 0.0
    for _ in range(max_steps):
        candidate_position = current_position + current_direction * step
        # Snap to the nearest actual mesh vertex before slabbing: the raw
        # advance can drift off the body across a curve, but the nearest
        # vertex is always on the surface.
        distances = np.linalg.norm(resolved_vertices - candidate_position, axis=1)
        anchor = resolved_vertices[int(np.argmin(distances))]
        next_centroid = slab_centroid_at(anchor, current_direction)
        if next_centroid is None:
            break
        step_vector = next_centroid - current_position
        step_length = float(np.linalg.norm(step_vector))
        if step_length <= _EPS:
            break
        total_travelled += step_length
        centerline.append(next_centroid)
        current_direction = step_vector / step_length
        current_position = next_centroid
        if total_travelled >= extent * 1.05:
            break

    if len(centerline) < 2:
        raise ValueError("medial-axis march produced too short a centerline to build a chain")

    resampled = _catmull_rom_resample(np.asarray(centerline), target_joint_count)

    # Head = thicker end: mean local cross-section radius over the last
    # 15% of arc length at each end of the resampled polyline.
    deltas = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = float(cumulative[-1]) if cumulative[-1] > _EPS else 1.0
    start_window = cumulative <= 0.15 * total_length
    end_window = cumulative >= 0.85 * total_length

    def _mean_end_radius(mask: np.ndarray) -> float:
        radii = []
        for index in np.nonzero(mask)[0]:
            if index == 0:
                direction = _normalize(resampled[1] - resampled[0])
            elif index == len(resampled) - 1:
                direction = _normalize(resampled[-1] - resampled[-2])
            else:
                direction = _normalize(resampled[index + 1] - resampled[index - 1])
            if direction is None:
                continue
            radii.append(_local_cross_section_radius(
                resampled[index], direction, resolved_vertices, slab_half_thickness * 3.0,
            ))
        return float(np.mean(radii)) if radii else 0.0

    if _mean_end_radius(start_window) > _mean_end_radius(end_window):
        # The starting tip is thicker -- that's the head; reverse so it
        # ends up last, matching this module's tail-to-head convention.
        resampled = resampled[::-1].copy()

    new_parents: list[int | None] = [None] + list(range(target_joint_count - 1))
    names = [f"spine_{i:02d}" for i in range(target_joint_count - 1)] + ["head"]
    return MedialAxisChain(resampled, new_parents, names)


def detect_profile_from_names(joint_names: Sequence[str]) -> str:
    """Classify an *exported* rig's bone names as `"humanoid"` or `"chain"`.

    Used where only bone names are available (a GLB read back after
    export), rather than the joint/parent arrays `detect_rig_profile`
    classifies from. Chain: at least one zero-padded `spine_NN` bone (this
    module's chain-naming convention -- see `_CHAIN_SPINE_NAME_PATTERN`)
    plus a `head` bone, and no humanoid limb-role bone names.
    """
    normalized = [str(name).strip().casefold() for name in joint_names]
    has_head = "head" in normalized
    has_chain_spine = any(_CHAIN_SPINE_NAME_PATTERN.match(name) for name in normalized)
    # Limb-role tokens are matched against the alnum-only form (dropping
    # underscores/spaces) since real rigs spell them either way
    # ("upperarm_l", "upper_arm_l", "UpperArm_L", ...).
    alnum_only = ["".join(character for character in name if character.isalnum()) for name in normalized]
    has_limb_roles = any(
        any(token in name for token in _HUMANOID_LIMB_NAME_TOKENS)
        for name in alnum_only
    )
    if has_head and has_chain_spine and not has_limb_roles:
        return "chain"
    return "humanoid"


def _local_cross_section_radius(
    joint: np.ndarray,
    axis_direction: np.ndarray,
    vertices: np.ndarray,
    radius: float,
) -> float:
    """Mean perpendicular-to-axis distance of mesh vertices within `radius`
    of `joint` -- an approximation of the mesh's local thickness there."""
    if vertices is None or len(vertices) == 0 or radius <= _EPS:
        return 0.0
    offsets = vertices - joint
    distances = np.linalg.norm(offsets, axis=1)
    nearby = offsets[distances <= radius]
    if len(nearby) == 0:
        return 0.0
    along = nearby @ axis_direction
    perpendicular = nearby - np.outer(along, axis_direction)
    return float(np.mean(np.linalg.norm(perpendicular, axis=1)))


def _resolve_lazy_vertices(
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None,
) -> np.ndarray | None:
    resolved = mesh_vertices() if callable(mesh_vertices) else mesh_vertices
    if resolved is None or len(resolved) == 0:
        return None
    return resolved


def _name_chain_profile(
    joints: np.ndarray,
    parents: Sequence[int | None],
    main_path: list[int],
    spurs: dict[int, tuple[int, list[int]]],
    root_index: int,
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None,
) -> list[str]:
    """Name a chain skeleton's bones: `spine_00..NN` tail-to-head, `head` at
    the head end, spurs as `spur_<parent>_<i>`.

    Head-end heuristic: the end whose local mesh cross-section (vertices
    within a radius of the end joint, projected perpendicular to the local
    chain direction) is larger -- a head is thicker than a tail tip.
    Fallback (no mesh evidence, or an inconclusive reading): the end
    nearest UniRig's own root joint by graph distance -- for the common
    case of a root placed at one actual end, that end.
    """
    count = len(joints)
    names = [f"bone_{i}" for i in range(count)]
    end_a, end_b = main_path[0], main_path[-1]

    adjacency = _build_adjacency(count, parents)
    _, distances_from_root, _ = _bfs_distances(root_index, adjacency)
    head_index = end_a if distances_from_root.get(end_a, 0) <= distances_from_root.get(end_b, 0) else end_b

    resolved_vertices = _resolve_lazy_vertices(mesh_vertices)
    if resolved_vertices is not None and len(main_path) >= 2:
        segment_lengths = [
            float(np.linalg.norm(joints[main_path[i + 1]] - joints[main_path[i]]))
            for i in range(len(main_path) - 1)
        ]
        mean_segment = float(np.mean(segment_lengths)) if segment_lengths else 0.0
        radius = max(mean_segment * 2.0, _EPS)
        direction_a = _normalize(joints[main_path[1]] - joints[end_a])
        direction_b = _normalize(joints[end_b] - joints[main_path[-2]])
        if direction_a is not None and direction_b is not None:
            radius_a = _local_cross_section_radius(joints[end_a], direction_a, resolved_vertices, radius)
            radius_b = _local_cross_section_radius(joints[end_b], direction_b, resolved_vertices, radius)
            if radius_a != radius_b:
                head_index = end_a if radius_a > radius_b else end_b

    ordered_from_tail = main_path if head_index == end_b else list(reversed(main_path))
    for position, joint_index in enumerate(ordered_from_tail[:-1]):
        names[joint_index] = f"spine_{position:02d}"
    names[head_index] = "head"

    spur_index_by_parent: dict[str, int] = {}
    for spur_root in sorted(spurs):
        attached_node, component = spurs[spur_root]
        # Named for the main-path node this branch attaches to -- true
        # regardless of whether the branch leaves that node via a rooted
        # child edge or (a path endpoint's) parent edge; `parents[]` alone
        # only reflects the latter, which is why `_chain_spurs` reports
        # the attachment point explicitly instead.
        parent_name = names[attached_node]
        for node in component:
            i = spur_index_by_parent.get(parent_name, 0)
            names[node] = f"spur_{parent_name}_{i}"
            spur_index_by_parent[parent_name] = i + 1
    return names


def select_chain_continuation_child(
    bone_name: str,
    head: np.ndarray,
    parent_head: np.ndarray | None,
    child_names: Sequence[str],
    child_heads: Sequence[np.ndarray],
    *,
    vertical_axis: int = VERTICAL_AXIS,
) -> int:
    """Pick which child a bone with more than one child should point its
    tail at (a branch point: pelvis -> spine + two thighs, or the last spine
    bone -> neck + two upper arms).

    Prefers a child whose name is the semantic chain continuation (e.g.
    pelvis -> spine_1, spine_3 -> neck, from `_CHAIN_CONTINUATIONS`);
    otherwise the child whose direction is most parallel to this bone's own
    parent->bone direction (continuing "straight" through the bone); for an
    unparented (root) bone with no name match, the child that climbs most
    steeply along `vertical_axis`.
    """
    count = len(child_names)
    if count <= 1:
        return 0
    preferred_name = _CHAIN_CONTINUATIONS.get(bone_name)
    if preferred_name is not None and preferred_name in child_names:
        return child_names.index(preferred_name)
    if parent_head is not None:
        parent_direction = _normalize(np.asarray(head) - np.asarray(parent_head))
        if parent_direction is not None:
            best_index, best_score = 0, -math.inf
            for index, child_head in enumerate(child_heads):
                child_direction = _normalize(np.asarray(child_head) - np.asarray(head))
                if child_direction is None:
                    continue
                score = float(np.dot(parent_direction, child_direction))
                if score > best_score:
                    best_index, best_score = index, score
            return best_index
    return max(
        range(count),
        key=lambda index: child_heads[index][vertical_axis] - head[vertical_axis],
    )


def resolve_position_overrides(
    overrides: dict[str, Sequence[float]],
    fresh_names: list[str],
    stored_names: dict[str, int] | None,
) -> ResolvedOverrides:
    """Resolve a `{semantic_name: [x, y, z]}` hand-adjustment map to joint
    indices.

    Overrides are authored against whatever names a *prior* bind's review
    UI displayed. If the naming algorithm changes between that bind and this
    one (exactly what this facing fix does), `fresh_names` can disagree with
    those names — the same joint index can get renamed, or worse, the same
    name can land on the mirror-image joint. Resolve through `stored_names`
    (the name -> index map captured at the time the override was authored)
    when available; only fall back to matching against `fresh_names`
    (the pre-fix behavior) when no stored map exists, since that fallback is
    only correct if naming hasn't changed since the override was made.
    """
    if stored_names is not None:
        resolve_map = stored_names
        used_fallback = False
    else:
        resolve_map = {name: index for index, name in enumerate(fresh_names)}
        used_fallback = True
    resolved: dict[int, np.ndarray] = {}
    unresolved: list[str] = []
    for name, value in overrides.items():
        index = resolve_map.get(name)
        if index is None or not (0 <= index < len(fresh_names)):
            unresolved.append(name)
            continue
        resolved[index] = np.asarray(value, dtype=np.float64)
    return ResolvedOverrides(resolved, unresolved, used_fallback)


# Structural roles a rig-adjust removal must never touch: the skeleton
# root and the bones every other bone's naming/weighting logic in this
# module (and the bind worker) assumes exist. Exact, case-insensitive
# match only -- a chain rig's `spine_00`/`spine_1`/... bones are *not*
# named exactly "spine" and stay removable.
_PROTECTED_REMOVAL_NAMES = frozenset({"root", "hips", "pelvis", "spine"})


class RemovedJoints(NamedTuple):
    joints: np.ndarray
    parents: list[int | None]
    names: list[str]
    tails: np.ndarray | None
    removed: list[str]
    refused: list[str]


def remove_joints_reparent(
    joints: np.ndarray,
    parents: Sequence[int | None],
    names: Sequence[str],
    tails: np.ndarray | None,
    remove: Sequence[str],
) -> RemovedJoints:
    """Splice named joints out of a skeleton, one at a time.

    Removing a joint reparents each of its own children onto *its* parent
    (bypassing the removed joint) rather than deleting the subtree, so a
    rig-adjust "remove this bone" request thins the skeleton without
    dropping mesh regions that were weighted to it or its descendants.
    Indices are compacted after every single removal and `parents` is
    rewritten to the new indices, so removals interact correctly when
    several are requested in one call (e.g. removing two adjacent chain
    joints, or a joint and then one of its now-reparented former
    grandchildren).

    A name is refused (left in place, reported in `refused` instead of
    `removed`) rather than removed when it: does not match any current
    joint name; names the skeleton root (a joint with no parent); or
    case-insensitively exact-matches a protected structural role (see
    `_PROTECTED_REMOVAL_NAMES`). The order of the joints that remain is
    otherwise preserved.
    """
    cur_joints = [np.asarray(row, dtype=np.float64) for row in joints]
    cur_parents: list[int | None] = [None if p is None or p < 0 else int(p) for p in parents]
    cur_names = [str(name) for name in names]
    cur_tails = None if tails is None else [np.asarray(row, dtype=np.float64) for row in tails]

    removed: list[str] = []
    refused: list[str] = []

    for raw_name in remove:
        name = str(raw_name)
        if name not in cur_names:
            refused.append(name)
            continue
        index = cur_names.index(name)
        if cur_parents[index] is None:
            refused.append(name)
            continue
        if name.strip().casefold() in _PROTECTED_REMOVAL_NAMES:
            refused.append(name)
            continue

        parent_of_removed = cur_parents[index]
        for child_index, parent_index in enumerate(cur_parents):
            if parent_index == index:
                cur_parents[child_index] = parent_of_removed

        del cur_joints[index]
        del cur_names[index]
        if cur_tails is not None:
            del cur_tails[index]
        del cur_parents[index]
        # Every surviving parent reference past the removed slot shifts
        # down by one; references before it are untouched. Reparented
        # children no longer point at `index` (handled above), so this is
        # a pure compaction, not a second reparent pass.
        cur_parents = [
            None if parent_index is None else (parent_index - 1 if parent_index > index else parent_index)
            for parent_index in cur_parents
        ]
        removed.append(name)

    result_joints = np.asarray(cur_joints, dtype=np.float64) if cur_joints else np.zeros((0, 3), dtype=np.float64)
    if cur_tails is None:
        result_tails = None
    else:
        result_tails = np.asarray(cur_tails, dtype=np.float64) if cur_tails else np.zeros((0, 3), dtype=np.float64)
    return RemovedJoints(result_joints, cur_parents, cur_names, result_tails, removed, refused)


def _side_prefix(joint_index: int, root_index: int, joints: np.ndarray, left: np.ndarray) -> str:
    offset = joints[joint_index] - joints[root_index]
    return "l" if float(np.dot(offset, left)) > 0.0 else "r"


def semantic_joint_names(
    joints: np.ndarray,
    parents: Sequence[int | None],
    mesh_vertices: np.ndarray | Callable[[], np.ndarray] | None = None,
    *,
    profile_hint: str | None = None,
) -> SkeletonNaming:
    """Derive semantic bone names from skeleton topology.

    UniRig emits generic bone_N names. Two body plans are handled:

    Humanoid: the downstream rig certificate resolves humanoid roles
    (pelvis, spine, upper_arm_r, hand_l, ...) from name aliases, so rename
    by structure: vertical axis = Z, spine chain climbs it, arm chains
    extend sideways, leg chains descend. Left/right is resolved from the
    detected facing direction rather than a fixed axis sign (see module
    docstring).

    Chain (a serpent or other limbless creature -- see `detect_rig_profile`):
    no vertical-climb spine or sideways arms exist, so this path is skipped
    entirely in favor of `_name_chain_profile`.

    `profile_hint` ("humanoid" or "chain", from the run's subject_kind --
    see the bind worker) overrides automatic detection when given: a
    "humanoid" hint always takes the humanoid path below, regardless of
    mesh elongation (a T-posed humanoid's arm span can exceed its height,
    which is not evidence of a chain body plan). A "chain" hint is only
    honored if the topology genuinely supports it (no disqualifying
    branch) -- a mismatched hint falls back to automatic detection rather
    than forcing a chain naming onto a real limbed skeleton. `None` (no
    hint) is current behaviour: pure automatic detection.
    """
    count = len(joints)
    names = [f"bone_{i}" for i in range(count)]
    children: dict[int, list[int]] = {i: [] for i in range(count)}
    roots = []
    for i, parent in enumerate(parents):
        if parent is None or parent < 0 or parent == i:
            roots.append(i)
        else:
            children[int(parent)].append(i)
    if not roots:
        return SkeletonNaming(names, _DEFAULT_FORWARD.copy(), "no-root", "skeleton has no root joint", "humanoid")

    profile_source = "hint" if profile_hint in ("humanoid", "chain") else "detected"
    if profile_hint == "humanoid":
        profile = "humanoid"
    elif profile_hint == "chain" and len(roots) == 1:
        main_path = _tree_diameter_path(count, parents)
        adjacency = _build_adjacency(count, parents)
        is_chain, _spurs = _chain_spurs(main_path, adjacency, _MIN_SPUR_LENGTH) if len(main_path) >= 2 else (False, {})
        profile = "chain" if is_chain and len(main_path) >= 2 else "humanoid"
        if profile != "chain":
            profile_source = "detected"  # hint didn't match the topology; fell back
    else:
        profile = detect_rig_profile(joints, parents, mesh_vertices)

    if profile == "chain":
        main_path = _tree_diameter_path(count, parents)
        adjacency = _build_adjacency(count, parents)
        is_chain, spurs = _chain_spurs(main_path, adjacency, _MIN_SPUR_LENGTH)
        # Either detect_rig_profile already confirmed this forms a chain,
        # or (profile_hint == "chain" branch above) it was just checked
        # directly; recomputed here (rather than threading state through
        # detect_rig_profile's public signature) so that function stays a
        # pure classifier.
        assert is_chain, "resolved profile is chain but the topology does not form one"
        chain_names = _name_chain_profile(joints, parents, main_path, spurs, roots[0], mesh_vertices)
        head_index = chain_names.index("head")
        tail_index = main_path[0] if head_index == main_path[-1] else main_path[-1]
        forward = _normalize(joints[head_index] - joints[tail_index])
        if forward is None:
            forward = _DEFAULT_FORWARD.copy()
        return SkeletonNaming(chain_names, forward, "chain-axis", None, "chain", profile_source)

    vertical = VERTICAL_AXIS
    side_axis = SIDE_AXIS
    spread = np.ptp(joints, axis=0)
    root = roots[0]

    # spine chain: follow the child chain climbing +vertical
    spine_roles = ["pelvis", "spine_1", "spine_2", "spine_3"]
    node = root
    names[root] = "pelvis"
    role_index = 1
    while True:
        climbs = [c for c in children[node] if joints[c, vertical] > joints[node, vertical]]
        if not climbs:
            break
        node = max(climbs, key=lambda c: joints[c, vertical])
        if role_index < len(spine_roles):
            names[node] = spine_roles[role_index]
            role_index += 1
        else:
            names[node] = f"spine_extra_{role_index}"
            role_index += 1
    head_name = "head"
    neck_index = int(parents[node]) if parents[node] is not None and parents[node] >= 0 else None
    if names[node].startswith("spine_extra"):
        names[node] = head_name
        if neck_index is not None and names[neck_index].startswith("spine_extra"):
            names[neck_index] = "neck"
    else:
        # topmost chain bone above the last spine gets neck/head
        climbing = [c for c in children[node] if joints[c, vertical] > joints[node, vertical]]
        if climbing:
            neck = max(climbing, key=lambda c: joints[c, vertical])
            if not names[neck].startswith("spine"):
                names[neck] = "neck"
            above = [c for c in children[neck] if joints[c, vertical] > joints[neck, vertical]]
            if above:
                names[max(above, key=lambda c: joints[c, vertical])] = head_name

    # arm chains: from spine bones, subtrees extending furthest along side_axis
    def subtree_extreme(node: int) -> float:
        stack = [node]
        extreme = 0.0
        while stack:
            current = stack.pop()
            extreme = max(extreme, abs(joints[current, side_axis] - joints[root, side_axis]))
            stack.extend(children[current])
        return extreme

    arm_roots = []
    for i in range(count):
        if i == root or parents[i] is None or parents[i] < 0:
            continue
        parent_index = int(parents[i])
        if names[i].startswith("spine") or names[i] == "neck":
            continue
        parent_is_spine = names[parent_index].startswith("spine") or names[parent_index] == "neck" or parent_index == root
        if parent_is_spine and subtree_extreme(i) > spread[side_axis] * 0.3:
            arm_roots.append(i)

    def walk_arm_chain(chain_root: int) -> list[int]:
        chain = [chain_root]
        current = chain_root
        while True:
            next_children = [c for c in children[current]
                             if abs(joints[c, side_axis] - joints[current, side_axis])
                             > abs(joints[c, vertical] - joints[current, vertical]) * 0.5]
            if not next_children:
                break
            current = max(next_children, key=lambda c: abs(joints[c, side_axis] - joints[root, side_axis]))
            chain.append(current)
        return chain

    def name_side_chain(chain: list[int], prefix: str) -> None:
        roles = [f"upper_arm_{prefix}", f"lower_arm_{prefix}", f"hand_{prefix}"]
        for index, role in zip(chain, roles):
            names[index] = role
        for index in chain[3:]:
            names[index] = f"hand_extra_{prefix}_{index}"

    arm_chains = {arm_root: walk_arm_chain(arm_root) for arm_root in arm_roots}

    # leg chains: chains descending from the root along -vertical
    leg_roots = [c for c in children[root] if joints[c, vertical] < joints[root, vertical]]

    def walk_leg_chain(leg_root: int) -> list[int]:
        chain = [leg_root]
        current = leg_root
        while True:
            descends = [c for c in children[current] if joints[c, vertical] <= joints[current, vertical]]
            if not descends:
                break
            current = min(descends, key=lambda c: joints[c, vertical])
            chain.append(current)
        return chain

    leg_chains = {leg_root: walk_leg_chain(leg_root) for leg_root in leg_roots}

    def name_leg_chain(chain: list[int], prefix: str) -> None:
        roles = [f"thigh_{prefix}", f"calf_{prefix}", f"foot_{prefix}"]
        for index, role in zip(chain, roles):
            names[index] = role
        for index in chain[3:]:
            names[index] = f"foot_extra_{prefix}_{index}"

    # Facing detection needs the (unnamed) foot joint and its child (toe),
    # for every leg chain, before any side prefix is assigned.
    foot_indices = [chain[2] if len(chain) >= 3 else chain[-1] for chain in leg_chains.values()]
    foot_children = [children[foot_index] for foot_index in foot_indices]
    facing = detect_forward_from_feet(joints, foot_indices, foot_children, mesh_vertices)
    left = left_from_forward(facing.forward)

    for arm_root, chain in arm_chains.items():
        prefix = _side_prefix(arm_root, root, joints, left)
        name_side_chain(chain, prefix)
    for leg_root, chain in leg_chains.items():
        prefix = _side_prefix(leg_root, root, joints, left)
        name_leg_chain(chain, prefix)

    # de-duplicate: suffix repeats (fingers keep generic names untouched)
    seen: dict[str, int] = {}
    for i in range(count):
        base = names[i]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[i] = f"{base}_{seen[base]}"

    return SkeletonNaming(names, facing.forward, facing.source, facing.warning, "humanoid", profile_source)


# --- Nearest-bone-blend weighting (envelope-weighting replacement) ---
#
# A candidate bone is folded into a vertex's weights only when its distance
# is within this multiple of the nearest bone's own distance -- i.e. the
# candidates are genuinely close together (near a joint), not merely "close
# enough" the way an envelope-distance capsule radius would treat a whole
# neighbouring segment as reachable.
_DEFAULT_BLEND_RATIO = 1.35


def nearest_bone_blend_weights(
    vertices: np.ndarray,
    heads: np.ndarray,
    tails: np.ndarray,
    *,
    blend_ratio: float = _DEFAULT_BLEND_RATIO,
    max_influences: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric bone-segment weighting: replaces Blender's
    `ARMATURE_ENVELOPE` fallback, which uses a per-bone envelope-distance
    capsule radius that, on slender limbs, reaches past the neighbouring
    joint into the next segment entirely (a calf's envelope dominating
    vertices above the knee, into the thigh).

    For every vertex, this measures distance to every bone's *segment*
    (point-to-capsule-axis: the vertex's projection onto `head -> tail`,
    clamped to `t in [0, 1]`, never a radius) and blends over *every* bone
    within `blend_ratio` times the nearest bone's own distance -- not just
    the second-nearest -- up to `max_influences` candidates (the closest
    `max_influences` bones are considered; any of those still farther than
    `blend_ratio * nearest_distance` are dropped). Blending over every
    nearby bone, rather than only the runner-up, matters at three-way (and
    higher) junctions -- a neck sitting between head/neck/chest, or a
    shoulder between chest/upper-arm -- where a strict nearest-two blend
    still leaves a hard boundary one bone short of the joint. Blend
    weights are inverse-distance-*squared*, normalized to sum to 1, so a
    bone twice as far contributes a quarter as much rather than half.

    This cannot reach past a neighbouring joint the way an envelope radius
    can, because "nearest segment" is a purely geometric, unbounded-radius
    comparison -- a vertex beside a limb's mid-segment is never closer to
    any other bone's axis than to that limb's own.

    A degenerate bone (`|tail - head| < 1e-9`) is treated as a point at
    `head`.

    Returns `(bone_indices, weights)`, each shaped `(len(vertices),
    max_influences)`: `bone_indices` is `int`, unused influence slots are
    `-1`; `weights` is `float`, unused slots are `0.0`, and each row's
    non-zero weights sum to 1 (a vertex nearer to no bone at all -- an
    empty `heads`/`tails` -- would be a caller error, not handled here).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    heads = np.asarray(heads, dtype=np.float64)
    tails = np.asarray(tails, dtype=np.float64)
    num_vertices = vertices.shape[0]
    num_bones = heads.shape[0]

    bone_indices = np.full((num_vertices, max_influences), -1, dtype=np.int64)
    weights = np.zeros((num_vertices, max_influences), dtype=np.float64)
    if num_vertices == 0 or num_bones == 0:
        return bone_indices, weights

    segment_vectors = tails - heads  # (B, 3)
    segment_length_sq = np.sum(segment_vectors ** 2, axis=1)  # (B,)
    degenerate = segment_length_sq <= 1e-18  # |tail-head| < 1e-9, squared

    # (V, B, 3) vertex-to-head offsets, reduced to (V, B) projection
    # parameters along each bone's segment.
    offsets = vertices[:, None, :] - heads[None, :, :]
    safe_length_sq = np.where(segment_length_sq > 0.0, segment_length_sq, 1.0)
    t = np.einsum("vbc,bc->vb", offsets, segment_vectors) / safe_length_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    t = np.where(degenerate[None, :], 0.0, t)

    projected = heads[None, :, :] + t[:, :, None] * segment_vectors[None, :, :]  # (V, B, 3)
    distance = np.sqrt(np.sum((vertices[:, None, :] - projected) ** 2, axis=2))  # (V, B)

    order = np.argsort(distance, axis=1)  # (V, B), ascending
    vertex_rows = np.arange(num_vertices)
    nearest_distance = distance[vertex_rows, order[:, 0]]

    # Only the closest `max_influences` bones are ever candidates -- if a
    # bone beyond that many is still within `blend_ratio`, it loses to
    # the influence-count cap, not the ratio.
    slots = min(max_influences, num_bones)
    slot_bones = order[:, :slots]  # (V, slots), nearest-first
    slot_distance = np.take_along_axis(distance, slot_bones, axis=1)  # (V, slots)

    within = slot_distance <= nearest_distance[:, None] * blend_ratio
    within[:, 0] = True  # the nearest bone is always included

    # Inverse-distance-squared blend, epsilon-padded so candidates at (or
    # extremely near) equal zero distance -- e.g. a vertex sitting exactly
    # on a joint shared by several segments -- blend evenly rather than
    # dividing by zero.
    eps = 1e-9
    inverse_square = np.where(within, 1.0 / (slot_distance + eps) ** 2, 0.0)
    total = np.sum(inverse_square, axis=1, keepdims=True)
    slot_weight = inverse_square / total

    bone_indices[:, :slots] = np.where(within, slot_bones, -1)
    weights[:, :slots] = slot_weight

    return bone_indices, weights


def stitch_coincident_vertex_weights(
    vertices: np.ndarray,
    bone_indices: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Force every group of coincident vertices to share identical bone
    weights, appearance-preserving fix for mesh tearing at UV/normal seams.

    A weld-then-count of the pre-rig mesh shows it is a single connected
    component -- the "thousands of components" `mesh_quality.py` reports
    pre-weld are duplicated seam vertices: several mesh vertices sitting at
    the exact same 3D position, split apart only so each copy can carry its
    own UV coordinate or normal. Any per-vertex weighting scheme (bone-heat,
    nearest-bone-blend, distance-along-chain, the nearest-joint fallback)
    treats these as independent vertices and can assign them different bone
    weights; when the rig poses, two vertices that used to coincide move
    apart, tearing the mesh open at the seam. Averaging the weights of every
    vertex in a coincident group and writing that same average back to each
    member makes it geometrically impossible for a seam to split under any
    pose -- no vertex position or UV/normal is touched, only the bone
    weights the caller already computed.

    `vertices` is `(V, 3)` world-space positions. `bone_indices`/`weights`
    are the same sparse `(V, max_influences)` representation produced by
    `nearest_bone_blend_weights` (`bone_indices` int, unused slots `-1`;
    `weights` float, unused slots `0.0`, valid rows summing to 1) -- this
    lets a caller run the same stitching pass over bone-heat or
    distance-along-chain output too, by first reading Blender's own vertex
    groups into this shape.

    Vertices are grouped by true Euclidean distance (`scipy.spatial.cKDTree
    .query_pairs`, no per-vertex Python loop, so this stays fast at 170k+
    vertices), not by quantizing position to a fixed grid and hashing the
    quantized cell: grid quantization has a boundary-snap failure mode --
    two vertices closer together than `tolerance` can still round into
    *different* grid cells when their true (tiny) offset straddles a cell
    edge, silently leaving that pair unstitched. A KD-tree pair query has
    no such edge case. The pairwise "within `tolerance`" graph is then
    resolved into connected components (`scipy.sparse.csgraph
    .connected_components`) so a group can have more than 2 members (a
    chain of near-duplicates, each within `tolerance` of a neighbour but
    not of every other member, still collapses to one group). Groups of
    size 1 are returned untouched. For a group of size > 1, each bone's
    weight is *summed* across the group's members then divided by the
    member count (a per-bone average, not a per-member renormalization), so
    a bone only one member carries still pulls the group average down
    rather than being treated as if every member carried it. The averaged
    result is then capped to the input arrays' influence width
    (`bone_indices.shape[1]`), keeping the strongest bones and dropping the
    rest, and renormalized so each stitched row again sums to 1. The same
    final row is written to every member of the group.

    Memory: this never materializes a dense `(V, num_bones)` or
    `(num_groups, num_bones)` array. Only vertices belonging to a group of
    size > 1 are considered, and the group/bone accumulation is done over
    the sparse (group, bone) pairs actually present (bounded by
    `stitched_vertices * max_influences`), not the full group-count x
    bone-count cross product.

    Returns `(bone_indices, weights)` in the same shape and dtype as the
    inputs; both are copies, so it is safe to compare against the arrays
    passed in to see what changed.
    """
    # Imported lazily: this module is also imported by blender_worker.py,
    # which runs inside Blender's own bundled Python (no scipy installed
    # there) -- a module-level scipy import would break every caller of
    # this file, not just this one function. The bind worker that actually
    # calls this (unirig_bind_worker.py) runs under the UniRig venv, which
    # does have scipy.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    bone_indices = np.asarray(bone_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    out_bone_indices = bone_indices.copy()
    out_weights = weights.copy()

    num_vertices = vertices.shape[0]
    if num_vertices == 0 or bone_indices.shape[1] == 0:
        return out_bone_indices, out_weights
    max_influences = bone_indices.shape[1]

    # Exact-distance grouping: every pair within `tolerance` of each other
    # becomes an edge, and connected components of that graph are the
    # coincident-vertex groups (see the docstring for why this replaces
    # grid quantization). A vertex with no neighbour within `tolerance` is
    # its own singleton component.
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=tolerance, output_type="ndarray")
    if pairs.size:
        graph = coo_matrix(
            (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
            shape=(num_vertices, num_vertices),
        )
        _, group_of = connected_components(graph, directed=False)
    else:
        group_of = np.arange(num_vertices, dtype=np.int64)
    group_counts = np.bincount(group_of)  # num_vertices > 0 here (checked above)

    duplicated_group = group_counts > 1
    if not np.any(duplicated_group):
        return out_bone_indices, out_weights

    member_mask = duplicated_group[group_of]  # (V,) bool: in a group of size > 1
    member_indices = np.nonzero(member_mask)[0]
    if member_indices.size == 0:
        return out_bone_indices, out_weights

    member_group_ids = group_of[member_indices]
    member_bones = bone_indices[member_indices]  # (M, max_influences)
    member_weights = weights[member_indices]  # (M, max_influences)
    valid = (member_bones >= 0) & (member_weights > 0.0)

    flat_group_ids = np.repeat(member_group_ids, max_influences)[valid.reshape(-1)]
    flat_bones = member_bones.reshape(-1)[valid.reshape(-1)]
    flat_weights = member_weights.reshape(-1)[valid.reshape(-1)]
    if flat_group_ids.size == 0:
        return out_bone_indices, out_weights

    # Densify the group ids and bone ids that actually appear here, so the
    # accumulation array below is sized by what is actually present, never
    # by the full vertex count or the full bone count.
    present_groups, dense_group = np.unique(flat_group_ids, return_inverse=True)
    present_bones, dense_bone = np.unique(flat_bones, return_inverse=True)
    dense_group = dense_group.reshape(-1)
    dense_bone = dense_bone.reshape(-1)
    num_present_groups = present_groups.shape[0]
    num_present_bones = present_bones.shape[0]

    pair_key = dense_group.astype(np.int64) * num_present_bones + dense_bone
    pair_sum = np.bincount(
        pair_key, weights=flat_weights, minlength=num_present_groups * num_present_bones
    )
    pair_sum = pair_sum.reshape(num_present_groups, num_present_bones)

    member_counts = group_counts[present_groups].astype(np.float64)
    averaged = pair_sum / member_counts[:, None]  # sum-per-bone / group size

    # Cap to the caller's influence width, keeping the strongest bones.
    keep = min(max_influences, num_present_bones)
    order = np.argsort(-averaged, axis=1, kind="stable")[:, :keep]
    row_index = np.arange(num_present_groups)[:, None]
    top_weight = averaged[row_index, order]
    top_bone = present_bones[order]

    nonzero = top_weight > 1e-12
    top_bone = np.where(nonzero, top_bone, -1)
    top_weight = np.where(nonzero, top_weight, 0.0)
    row_total = np.sum(top_weight, axis=1, keepdims=True)
    row_total = np.where(row_total > 1e-12, row_total, 1.0)
    top_weight = top_weight / row_total

    if keep < max_influences:
        pad = max_influences - keep
        top_bone = np.concatenate(
            [top_bone, np.full((num_present_groups, pad), -1, dtype=np.int64)], axis=1
        )
        top_weight = np.concatenate(
            [top_weight, np.zeros((num_present_groups, pad), dtype=np.float64)], axis=1
        )

    # Write the group's stitched row to every member. `present_groups` is
    # sorted (np.unique), so searchsorted maps each member's group id to
    # its row; a member whose group had no valid (bone, weight) at all
    # (every member fully unweighted) has no row here and is left as-is.
    member_row = np.searchsorted(present_groups, member_group_ids)
    member_row = np.clip(member_row, 0, num_present_groups - 1)
    has_row = present_groups[member_row] == member_group_ids
    target_indices = member_indices[has_row]
    if target_indices.size:
        out_bone_indices[target_indices] = top_bone[member_row[has_row]]
        out_weights[target_indices] = top_weight[member_row[has_row]]

    return out_bone_indices, out_weights




# --- Hanging-region weighting (skirts, robes, coat tails, crotch webs) ---
#
# Bone-heat (and the nearest-bone-blend fallback) are both nearest-wins
# schemes: every vertex ends up owned by whichever bone is geometrically
# closest. That is right for material that sits ON a limb, and wrong for
# material that merely HANGS between or around the limbs and has no
# skeletal representation of its own -- a floor-length robe, a skirt, a
# coat tail, or the fabric web bridging the two legs at the crotch.
#
# Nearest-wins partitions such material into rigid left/right halves with a
# knife-edge boundary down the midline. Under any scissoring pose (every
# walk cycle) the two halves shear in opposite directions: a robe collapses
# into a flat vertical plank below the hem, a crotch web is stretched into
# a thin smeared sheet trailing the swinging leg. The material has no
# skeletal degree of freedom that could hold it together, so no amount of
# smoothing over the mesh topology fixes it -- the OWNERSHIP is wrong, not
# the smoothness of the boundary.
#
# `detect_hanging_region` finds that material and `hanging_region_blend`
# replaces its nearest-wins ownership with a smooth lateral blend across
# the pelvis span (so the midline material is driven by BOTH legs equally
# and simply stays put when they scissor), anchored to the pelvis at the
# top of the region and handed over to the leg pair towards the hem.
#
# No new bones: v1 is a pure weighting post-pass, applied after the primary
# weighting and before the final seam stitching.

_HANGING_BANDS = 12
# A horizontal slice's widest empty lateral interval straddling the midline,
# as a multiple of the hip HALF-span. Two separated legs leave a real gap
# here (measured 1.0-2.0 on fitted characters); a shell that encloses both
# legs leaves none (measured 0.00 on a floor-length robe, at every height).
_HANGING_GAP_THRESHOLD = 0.35
# How far below the hips the contiguous gapless run must reach, as a
# fraction of leg length, before the lower body is treated as one enclosing
# shell rather than two limbs. Fitted characters measure ~0.25 (only the
# pelvis/crotch bands are gapless); a floor-length robe measures 1.0.
_HANGING_SHELL_DEPTH = 0.5
# Strength ramp below the hips, as a fraction of leg length: the region
# fades in over this distance so the pelvis/upper-crotch band -- where even
# a fitted character's cross-section is legitimately gapless -- is barely
# touched.
_HANGING_CROTCH_MARGIN = 0.22
# Per-vertex acceptance on a FITTED character (crotch-web material only):
# `balance` is min/max distance to the two leg chains (1.0 = exactly
# between them), `clearance` is the distance to the nearer leg chain as a
# multiple of the hip half-span (>1 means the vertex is farther from its
# own leg than the legs are from each other -- it is bridging, not skin).
_HANGING_BALANCE_THRESHOLD = 0.45
_HANGING_CLEARANCE_THRESHOLD = 0.9
# Width of the fade above `_HANGING_CLEARANCE_THRESHOLD` over which fitted
# strength reaches full. Clearance on real crotch-web material measures
# 1.0-1.5 hip-half-spans, so a fade of ~1.0 would leave the correction an
# order of magnitude too weak to change anything.
_HANGING_CLEARANCE_FADE = 0.35
# A fitted character's bridging material sits AT the hip line (it is the
# crotch), so the height ramp that protects the pelvis band on a shell must
# be much shorter here -- the balance/clearance gates already do the
# protecting, and a long ramp simply cancels the fix.
_HANGING_FITTED_CROTCH_MARGIN = 0.06
# ... and fitted-mode blending stays a correction rather than a takeover:
# the leg surface around the crotch is real skin whose bone-heat weights
# are mostly right, so the target weights never fully replace them.
_HANGING_FITTED_CEILING = 0.75
# The lateral blend ramps over this multiple of the full hip span, so the
# handover from one leg to the other is spread across roughly the whole
# width of a skirt instead of collapsing to a seam at the midline.
_HANGING_LATERAL_SOFTNESS = 2.0
# Share of the region's lowest material handed to the foot pair. A
# floor-length hem hangs past the ankle, and leaving the feet with no
# vertices at all fails the rig certificate's required-joint check -- but
# the feet swing hardest of any bone in a stride, so the hem takes a
# minority share of them and stays calf-driven.
_HANGING_FOOT_SHARE = 0.35
_HANGING_MINIMUM_REGION_VERTICES = 64


class HangingRegion(NamedTuple):
    """Result of `detect_hanging_region`.

    `strength` is the per-vertex blend strength in [0, 1] (0 = leave the
    existing weights alone), `mask` is `strength > 0`. `mode` is "shell"
    (the lower body is one enclosing shell -- a robe/skirt), "fitted" (two
    separated limbs; only bridging material qualifies) or "none".
    """

    mask: np.ndarray
    strength: np.ndarray
    mode: str
    enclosure_depth: float
    region_top: float
    region_bottom: float
    band_gaps: list[float]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def point_segment_distance(points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Distance from each point to the SEGMENT head->tail (not the infinite
    line, and with no capsule radius): the same measure
    `nearest_bone_blend_weights` uses, exposed for the hanging-region
    detector so both agree on what "distance to a bone" means."""
    points = np.asarray(points, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)
    axis = tail - head
    length_squared = float(axis @ axis)
    if length_squared <= 1e-18:
        return np.linalg.norm(points - head, axis=1)
    t = np.clip(((points - head) @ axis) / length_squared, 0.0, 1.0)
    return np.linalg.norm(points - (head + t[:, None] * axis), axis=1)


def chain_segment_distance(points: np.ndarray, segments: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Distance to the nearest segment of a bone chain (thigh -> calf ->
    foot -> toes). `segments` is a sequence of (head, tail) pairs. An empty
    chain yields +inf, so a caller missing one leg entirely degrades to
    "everything is far from that leg" rather than crashing."""
    points = np.asarray(points, dtype=np.float64)
    best = np.full(points.shape[0], np.inf, dtype=np.float64)
    for head, tail in segments:
        best = np.minimum(best, point_segment_distance(points, head, tail))
    return best


def _mix_blend_targets(
    bone_indices: np.ndarray,
    weights: np.ndarray,
    rows: np.ndarray,
    strength: np.ndarray,
    target_bones: np.ndarray,
    target_weights: np.ndarray,
    max_influences: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(1 - strength) * existing + strength * target` per vertex, capped to
    `max_influences` (strongest kept) and renormalized.

    Shared by every weighting post-pass in this module (the hanging region,
    the torso shell, the inter-leg bridge). Rows whose target weights sum to
    nothing -- every target bone unresolved -- are returned byte-identical,
    as is every row outside `rows`.

    `target_bones` are indices into the caller's own bone list; -1 means
    "the caller could not resolve this bone", and that target's share is
    handed to the remaining targets by the renormalization.
    """
    out_bone_indices = np.asarray(bone_indices, dtype=np.int64).copy()
    out_weights = np.asarray(weights, dtype=np.float64).copy()
    if rows.size == 0:
        return out_bone_indices, out_weights

    target_bones = np.asarray(target_bones, dtype=np.int64)
    target_weights = np.where(target_bones[None, :] >= 0, target_weights, 0.0)
    totals = np.sum(target_weights, axis=1, keepdims=True)
    usable = totals[:, 0] > _EPS
    if not np.any(usable):
        return out_bone_indices, out_weights
    target_weights = np.where(usable[:, None], target_weights / np.where(totals > _EPS, totals, 1.0), 0.0)

    width = out_bone_indices.shape[1]
    for offset, row in enumerate(rows):
        if not usable[offset]:
            continue
        blend = float(strength[offset])
        accumulated: dict[int, float] = {}
        for slot in range(width):
            bone = int(bone_indices[row, slot])
            value = float(weights[row, slot])
            if bone < 0 or value <= 0.0:
                continue
            accumulated[bone] = accumulated.get(bone, 0.0) + value * (1.0 - blend)
        for slot in range(target_bones.shape[0]):
            bone = int(target_bones[slot])
            value = float(target_weights[offset, slot])
            if bone < 0 or value <= 0.0:
                continue
            accumulated[bone] = accumulated.get(bone, 0.0) + value * blend
        if not accumulated:
            continue
        ordered = sorted(accumulated.items(), key=lambda item: -item[1])[:max_influences]
        total = sum(value for _, value in ordered)
        if total <= _EPS:
            continue
        out_bone_indices[row, :] = -1
        out_weights[row, :] = 0.0
        for slot, (bone, value) in enumerate(ordered):
            out_bone_indices[row, slot] = bone
            out_weights[row, slot] = value / total
    return out_bone_indices, out_weights


def midline_lateral_gap(
    lateral: np.ndarray, hip_half_span: float, *, bins: int = 48, midline: float = 0.0
) -> float:
    """Widest empty lateral interval whose centre lies within
    `0.6 * hip_half_span` of `midline`, as a multiple of `hip_half_span`.

    This is the structural signal that separates "two limbs" from "one
    enclosing shell" at a given height, and it is the only signal in this
    module that does not degrade into circularity on a mesh whose legs are
    not modelled at all. A pair of trouser legs leaves a genuine empty
    corridor between them; a robe's cross-section is a closed ring whose
    front and back walls both cross the midline, so it leaves none.

    Occupancy uses a small absolute-plus-relative floor rather than "any
    vertex at all", so a handful of stray interior vertices (a seam, a
    stray shell the canonicalize pass left inside) cannot fill the corridor
    and mask a genuinely separated pair of legs.

    The measure assumes a mesh-density cross-section: at `bins` bins it
    needs a few hundred vertices in the slice before empty bins mean
    "corridor" rather than "sparse sampling". Every mesh reaching this
    stage is a generated/canonicalized surface with thousands of vertices
    per height band, so that holds; `minimum_region_vertices` in
    `detect_hanging_region` is the backstop for anything far below it.
    """
    lateral = np.asarray(lateral, dtype=np.float64)
    if lateral.size < 8 or hip_half_span <= _EPS:
        return 0.0
    low, high = float(lateral.min()), float(lateral.max())
    if high - low <= _EPS:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    counts, _ = np.histogram(lateral, bins=edges)
    occupied = counts > max(1, int(0.002 * lateral.size))
    widest = 0.0
    index = 0
    while index < bins:
        if occupied[index]:
            index += 1
            continue
        end = index
        while end < bins and not occupied[end]:
            end += 1
        centre = (edges[index] + edges[end]) / 2.0
        if abs(centre - midline) < 0.6 * hip_half_span:
            widest = max(widest, (edges[end] - edges[index]) / hip_half_span)
        index = end
    return float(widest)


def detect_hanging_region(
    vertices: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    hip_height: float,
    hip_half_span: float,
    hip_midline: float,
    left_segments: Sequence[Sequence[np.ndarray]],
    right_segments: Sequence[Sequence[np.ndarray]],
    bands: int = _HANGING_BANDS,
    gap_threshold: float = _HANGING_GAP_THRESHOLD,
    shell_depth: float = _HANGING_SHELL_DEPTH,
    crotch_margin: float = _HANGING_CROTCH_MARGIN,
    balance_threshold: float = _HANGING_BALANCE_THRESHOLD,
    clearance_threshold: float = _HANGING_CLEARANCE_THRESHOLD,
    clearance_fade: float = _HANGING_CLEARANCE_FADE,
    fitted_crotch_margin: float = _HANGING_FITTED_CROTCH_MARGIN,
    fitted_ceiling: float = _HANGING_FITTED_CEILING,
    minimum_region_vertices: int = _HANGING_MINIMUM_REGION_VERTICES,
) -> HangingRegion:
    """Find below-the-hips geometry that hangs rather than sitting on a leg.

    Two regimes, decided by how far below the hips the mesh's horizontal
    cross-sections stay gapless across the midline
    (`midline_lateral_gap`):

    * **shell** -- the gapless run reaches at least `shell_depth` of the
      leg length (a skirt/robe: the legs are not modelled at all, the whole
      lower body is one cone). Every vertex in the run is hanging; there is
      nothing else down there to protect.
    * **fitted** -- the run stops near the crotch, so the legs ARE
      modelled. Only material that is roughly equidistant from both leg
      chains (`balance_threshold`) AND well clear of either
      (`clearance_threshold`, in hip-half-span units) qualifies: the crotch
      web, a coat tail hanging between the legs. An ordinary inner-thigh
      vertex fails `clearance` -- it is skin, sitting at its own leg's
      radius.

    Strength fades in over `crotch_margin` of leg length below the hips, so
    the pelvis band (legitimately gapless on every character) is barely
    touched, and the caller's blend degrades continuously to a no-op there
    rather than stepping.

    Pure numpy; no scipy, no bpy.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    count = vertices.shape[0]
    empty = HangingRegion(
        mask=np.zeros(count, dtype=bool),
        strength=np.zeros(count, dtype=np.float64),
        mode="none",
        enclosure_depth=0.0,
        region_top=float(hip_height),
        region_bottom=float(hip_height),
        band_gaps=[],
    )
    if count == 0 or hip_half_span <= _EPS or not left_segments or not right_segments:
        return empty

    height = vertices[:, up_axis] * up_sign
    lateral = vertices[:, lateral_axis]
    below = height < hip_height
    if int(np.count_nonzero(below)) < minimum_region_vertices:
        return empty

    bottom = float(height[below].min())
    leg_length = float(hip_height) - bottom
    if leg_length <= _EPS:
        return empty

    # Band the leg span and measure each band's midline corridor.
    edges = np.linspace(bottom, float(hip_height), bands + 1)
    band_gaps: list[float] = []
    for index in range(bands):
        in_band = below & (height >= edges[index]) & (height < edges[index + 1])
        band_lateral = lateral[in_band]
        band_gaps.append(midline_lateral_gap(band_lateral, hip_half_span, midline=hip_midline))

    # Contiguous gapless run measured DOWNWARD from the hips: a knee-length
    # skirt's run stops at its hem (bare legs below it are separated
    # again), which is exactly the region that should be re-weighted.
    run_bottom_index = bands
    for index in range(bands - 1, -1, -1):
        if band_gaps[index] >= gap_threshold:
            break
        run_bottom_index = index
    if run_bottom_index >= bands:
        return empty
    run_bottom = float(edges[run_bottom_index])
    enclosure_depth = (float(hip_height) - run_bottom) / leg_length
    mode = "shell" if enclosure_depth >= shell_depth else "fitted"

    distance_left = chain_segment_distance(vertices, left_segments)
    distance_right = chain_segment_distance(vertices, right_segments)
    nearer = np.minimum(distance_left, distance_right)
    farther = np.maximum(distance_left, distance_right)
    balance = nearer / np.maximum(farther, _EPS)
    clearance = nearer / hip_half_span

    candidate = below & (height >= run_bottom - _EPS)
    if mode == "fitted":
        candidate &= (balance > balance_threshold) & (clearance > clearance_threshold)

    # Height ramp: 0 at the hips, 1 by `crotch_margin` of the leg length
    # below them.
    margin = crotch_margin if mode == "shell" else fitted_crotch_margin
    margin = max(margin * leg_length, _EPS)
    ramp = _smoothstep((float(hip_height) - height) / margin)
    strength = np.where(candidate, ramp, 0.0)
    if mode == "fitted":
        # Fade in with how bridging the vertex actually is, so a marginal
        # inner-thigh vertex that scrapes past both thresholds still moves
        # almost not at all.
        strength = strength * _smoothstep((balance - balance_threshold) / max(1.0 - balance_threshold, _EPS))
        strength = strength * _smoothstep((clearance - clearance_threshold) / max(clearance_fade, _EPS))
        strength = strength * fitted_ceiling

    mask = strength > 1e-6
    if int(np.count_nonzero(mask)) < minimum_region_vertices:
        return empty
    region_heights = height[mask]
    return HangingRegion(
        mask=mask,
        strength=strength,
        mode=mode,
        enclosure_depth=float(enclosure_depth),
        region_top=float(region_heights.max()),
        region_bottom=float(region_heights.min()),
        band_gaps=[round(float(g), 4) for g in band_gaps],
    )


def hanging_region_blend_weights(
    vertices: np.ndarray,
    region: HangingRegion,
    bone_indices: np.ndarray,
    weights: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    hip_height: float,
    hip_half_span: float,
    hip_midline: float,
    lateral_left: float,
    lateral_right: float,
    knee_height: float,
    ankle_height: float,
    pelvis_bone: int,
    thigh_left_bone: int,
    thigh_right_bone: int,
    calf_left_bone: int,
    calf_right_bone: int,
    foot_left_bone: int = -1,
    foot_right_bone: int = -1,
    max_influences: int = 4,
    lateral_softness: float = _HANGING_LATERAL_SOFTNESS,
    foot_share: float = _HANGING_FOOT_SHARE,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace nearest-wins ownership in the hanging region with a smooth
    lateral blend, mixed into the existing weights by `region.strength`.

    Target weights for a hanging vertex:

    * `u` -- normalized lateral coordinate across the pelvis span, ramped
      over `lateral_softness` times the FULL hip span and smoothstepped, so
      the midline of a skirt is driven 50/50 by both legs (and therefore
      stays put when they scissor) while its outer edges still follow the
      leg on their own side. This is the whole fix: nearest-wins gives the
      midline a knife-edge 100/0 boundary, which is what shears the shell
      into two rigid halves.
    * height blend -- pelvis-dominant at the top of the region, handed over
      to the leg pair towards the hem, so the waist of a skirt rides the
      pelvis (as it physically does) instead of the knees.
    * within the leg pair, thigh above the knee handing over to calf below
      it and then to a minority share of the foot below the ankle, so a
      hip-level panel is driven by the thighs, a floor-length hem mostly by
      the calves, and the feet keep the vertices the rig certificate
      requires every required bone to have.

    The result is mixed with the incoming weights as
    `(1 - strength) * existing + strength * target`, then capped to
    `max_influences` (strongest kept) and renormalized. A vertex outside
    the region, or with zero strength, is returned byte-identical.

    Bone indices are indices into the caller's own bone list; a bone the
    caller could not resolve is passed as -1 and its share is redistributed
    to the remaining targets by renormalization.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    bone_indices = np.asarray(bone_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    out_bone_indices = bone_indices.copy()
    out_weights = weights.copy()

    rows = np.nonzero(region.mask)[0]
    if rows.size == 0:
        return out_bone_indices, out_weights
    # Never write more influence slots than the caller's arrays carry.
    max_influences = max(1, min(max_influences, bone_indices.shape[1]))

    height = vertices[rows, up_axis] * up_sign
    lateral = vertices[rows, lateral_axis]
    strength = region.strength[rows]

    span = float(lateral_left) - float(lateral_right)
    if abs(span) <= _EPS:
        span = 2.0 * hip_half_span if hip_half_span > _EPS else 1.0
    u = _smoothstep(0.5 + (lateral - hip_midline) / (lateral_softness * span))

    # Pelvis at the top of the region, legs at the hem.
    hem = region.region_bottom
    top = float(hip_height)
    denominator = max(top - hem, _EPS)
    down = np.clip((top - height) / denominator, 0.0, 1.0)
    pelvis_share = 1.0 - _smoothstep(down)
    leg_share = 1.0 - pelvis_share

    # Thigh above the knee, calf below it, foot below the ankle.
    knee_denominator = max(float(knee_height) - float(ankle_height), _EPS)
    below_knee = _smoothstep((float(knee_height) - height) / knee_denominator)
    ankle_denominator = max(float(ankle_height) - hem, _EPS)
    below_ankle = _smoothstep((float(ankle_height) - height) / ankle_denominator)
    foot_fraction = np.clip(foot_share, 0.0, 1.0)
    foot_level = below_knee * below_ankle * foot_fraction
    calf_share = below_knee - foot_level
    thigh_share = 1.0 - below_knee

    target_bones = np.array(
        [
            pelvis_bone,
            thigh_left_bone, thigh_right_bone,
            calf_left_bone, calf_right_bone,
            foot_left_bone, foot_right_bone,
        ],
        dtype=np.int64,
    )
    target_weights = np.stack(
        [
            pelvis_share,
            leg_share * thigh_share * u,
            leg_share * thigh_share * (1.0 - u),
            leg_share * calf_share * u,
            leg_share * calf_share * (1.0 - u),
            leg_share * foot_level * u,
            leg_share * foot_level * (1.0 - u),
        ],
        axis=1,
    )
    # Unresolved bones contribute nothing; the shared mixer's
    # renormalization hands their share to the rest.
    return _mix_blend_targets(
        bone_indices, weights, rows, strength, target_bones, target_weights, max_influences
    )
# --- torso shell + inter-leg bridge (c5fbq) --------------------------------
#
# `detect_hanging_region` above is below-the-hips only. Two failure modes it
# cannot reach were left open by it:
#
#   * A loose garment ABOVE the hips -- an open vest, a jacket, a coat shell
#     standing off the ribcage. Nearest-wins hands its flaps to whichever
#     bone happens to be closest, which near the shoulders is an ARM bone,
#     so half a vest ends up glued to a swinging arm and tears off the trunk
#     in any wide swing. The structural signal is the same corridor idea one
#     dimension over: a fitted torso's horizontal cross-section is a single
#     closed ring, so along any radial direction from the spine the surface
#     is crossed ONCE; a shell offset from the body crosses it TWICE, with a
#     measurable empty radial gap (air) between the two crossings.
#
#   * Material bridging BETWEEN the legs (the crotch web). The fitted branch
#     of `detect_hanging_region` blends it laterally, which halved the tear
#     but did not close it: the lateral blend still hands the vertex mostly
#     to one leg near the hem of the region, and a stride then drags the
#     sheet along with that leg. Bridging material needs the two legs to
#     SHARE the stride tension symmetrically, with the pelvis holding the
#     middle, which is a different target shape -- hence a dedicated rule
#     rather than another knob on the lateral blend.
#
# Both are pure weighting post-passes like the hanging-region blend, run in
# the same place in the bind and covered by the same `skip_hanging_blend`
# escape hatch.

_TORSO_BANDS = 10
_TORSO_SECTORS = 16
# An empty radial interval inside one (height band, angular sector) cell,
# as a multiple of the torso's own core radius, before the material beyond
# it counts as a separate shell rather than surface roughness. Measured on
# real characters: a fitted bodysuit (zix) and a fitted dress (Mary) produce
# NO cell above 0.13 once the arms are excluded; an open jacket (the punk)
# produces cells at 0.13-0.55 all round the waist and lower chest.
_TORSO_RADIAL_GAP = 0.15
# Vertices required on each side of the split before a radial gap is a gap
# between two SURFACES rather than one stray vertex sitting off the mesh.
_TORSO_CLUSTER_MINIMUM = 3
# Below this a cell is too sparsely sampled for "the radii form two
# clusters" to mean anything (see `midline_lateral_gap`'s same caveat).
_TORSO_SECTOR_MINIMUM = 12
_TORSO_BAND_MINIMUM = 40
# The gap has to be comfortably clear of `_TORSO_RADIAL_GAP` before the
# correction reaches full strength, so a cell that only just qualifies moves
# almost nothing -- the same "fade in with how sure you are" discipline the
# fitted branch of `detect_hanging_region` uses.
_TORSO_GAP_FADE = 0.15
# Blur passes over the (height band x angular sector) grid. A shell is a
# CONTIGUOUS patch of cells; a single cell firing on its own is noise, and
# acting on it hands one speckle of vertices to a different bone than its
# own mesh neighbours -- a fresh knife-edge boundary, which measured worse
# than doing nothing at all.
_TORSO_CELL_BLUR = 2
# Radial ramp beyond a cell's split radius, in core-radius units: the
# strength rises over this distance rather than stepping at the split, so
# two vertices either side of it are treated almost identically.
_TORSO_RADIAL_FADE = 0.25
# Torso-shell blending stays a correction, not a takeover: the garment is
# still sewn to the body somewhere, and bone-heat got much of it right.
_TORSO_CEILING = 0.7
# ... and, as in the bridge, the second lever: flattening the weight
# gradient across the flap, which cannot hand a vertex to a bone its own
# neighbourhood does not already use and so runs at full amplitude.
_TORSO_SMOOTHING_CEILING = 1.0
# Strength ramp above the hip line, as a fraction of the hip->shoulder span,
# so the pass degrades continuously into `detect_hanging_region`'s territory
# instead of stepping at the hip line.
_TORSO_HIP_MARGIN = 0.12
# Share of a torso-shell vertex's weight that may go to the clavicle pair,
# reached only at the top of the torso band and only for material out at the
# sides. A flap over the sternum stays 100% spine-driven at every height:
# lateral spread towards the shoulders is what lets a shoulder panel follow
# the shoulder, and handing a front flap to a clavicle would recreate the
# bug in a milder form.
_TORSO_SHOULDER_SHARE = 0.35
# Lateral ramp for the clavicle left/right split, as a multiple of the
# clavicle span -- the same softness idea as `_HANGING_LATERAL_SOFTNESS`.
_TORSO_SHOULDER_SOFTNESS = 1.5
_TORSO_MINIMUM_REGION_VERTICES = 48

# Inter-leg bridging. A vertex qualifies when it is between the legs
# LATERALLY (inside the corridor), roughly equidistant from both leg chains,
# and well clear of either. Measured on the worst-stretched below-hip
# vertices of a real crotch-web tear (zix, Walk): |lateral| 0.2-0.5 hip
# half-spans, distance to the nearer leg 0.75-1.6 half-spans.
_BRIDGE_CORRIDOR = 0.65
_BRIDGE_BALANCE = 0.55
_BRIDGE_CLEARANCE = 0.75
_BRIDGE_CLEARANCE_FADE = 0.35
_BRIDGE_HIP_MARGIN = 0.05
# The pelvis stabilizer floor: bridging material never drops below this much
# pelvis weight however far down the region it sits. Without it the two
# thigh weights sum to ~1 and the sheet is stretched between two bones that
# scissor in opposite directions -- sharing the tension symmetrically is not
# enough on its own, something has to hold the middle still.
_BRIDGE_PELVIS_FLOOR = 0.3
_BRIDGE_LATERAL_SOFTNESS = 1.0
# The re-target's amplitude. Deliberately small: measured against a real
# crotch web, re-targeting is the WEAKER of the two levers and a strong one
# actively hurts (the sheet stops being dragged by one leg, and its boundary
# with the leg skin is stretched instead). It earns its place by making the
# ownership symmetric, not by taking over.
_BRIDGE_CEILING = 0.3
# The gradient-flattening amplitude (see `BridgeRegion`).
_BRIDGE_SMOOTHING_CEILING = 1.0
_BRIDGE_MINIMUM_REGION_VERTICES = 32


class TorsoShellRegion(NamedTuple):
    """Result of `detect_torso_shell`.

    `strength` is the per-vertex blend strength in [0, 1], `mask` is
    `strength > 0`. `core_radius` is the median distance from the spine axis
    over the (arms-excluded) torso band, the unit every radial threshold is
    expressed in.
    """

    mask: np.ndarray
    strength: np.ndarray
    smoothing_strength: np.ndarray
    core_radius: float
    region_top: float
    region_bottom: float
    band_shell_counts: list[int]
    widest_gap: float


class BridgeRegion(NamedTuple):
    """Result of `detect_bridge_region`: material bridging the two legs.

    Two strength fields, because the region gets two different treatments
    with different appetites for risk. `strength` drives the re-target
    (ceilinged, conservative -- it changes WHO owns a vertex).
    `smoothing_strength` drives the gradient flattening (the same fade
    shape without the ceiling, so the core of a real bridge is smoothed
    properly while the boundary still fades to nothing); flattening a
    gradient cannot hand a vertex to a bone its neighbourhood does not
    already use, so it earns the larger amplitude.
    """

    mask: np.ndarray
    strength: np.ndarray
    smoothing_strength: np.ndarray
    region_top: float
    region_bottom: float

def radial_shell_split(radii: np.ndarray, core_radius: float, *, cluster_minimum: int = _TORSO_CLUSTER_MINIMUM
                       ) -> tuple[float, float]:
    """Widest empty radial interval inside one angular sector, as
    `(gap, split_radius)` in `core_radius` units.

    This is `midline_lateral_gap`'s idea rotated into the radial direction.
    A fitted torso's cross-section is a single closed ring, so the radii
    seen along one direction from the spine form ONE cluster (the surface
    crossing) and the widest interior gap is surface roughness. A garment
    standing off the body is crossed twice, with air between, so the radii
    form two clusters separated by a real gap.

    `cluster_minimum` vertices are required on each side of the split, so a
    single stray vertex floating off the mesh cannot pass as a shell.
    Returns `(0.0, 0.0)` when the sector is too small to say anything.
    """
    radii = np.sort(np.asarray(radii, dtype=np.float64))
    if radii.size < 2 * cluster_minimum + 2 or core_radius <= _EPS:
        return 0.0, 0.0
    gaps = np.diff(radii)
    low = cluster_minimum - 1
    high = radii.size - cluster_minimum
    if high <= low:
        return 0.0, 0.0
    index = low + int(np.argmax(gaps[low:high]))
    return float(gaps[index]) / core_radius, float(radii[index])


def detect_torso_shell(
    vertices: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    depth_axis: int,
    hip_height: float,
    shoulder_height: float,
    spine_heights: Sequence[float],
    spine_lateral: Sequence[float],
    spine_depth: Sequence[float],
    spine_segments: Sequence[Sequence[np.ndarray]],
    arm_segments: Sequence[Sequence[np.ndarray]],
    bands: int = _TORSO_BANDS,
    sectors: int = _TORSO_SECTORS,
    radial_gap: float = _TORSO_RADIAL_GAP,
    gap_fade: float = _TORSO_GAP_FADE,
    cluster_minimum: int = _TORSO_CLUSTER_MINIMUM,
    sector_minimum: int = _TORSO_SECTOR_MINIMUM,
    band_minimum: int = _TORSO_BAND_MINIMUM,
    cell_blur: int = _TORSO_CELL_BLUR,
    radial_fade: float = _TORSO_RADIAL_FADE,
    hip_margin: float = _TORSO_HIP_MARGIN,
    ceiling: float = _TORSO_CEILING,
    smoothing_ceiling: float = _TORSO_SMOOTHING_CEILING,
    minimum_region_vertices: int = _TORSO_MINIMUM_REGION_VERTICES,
) -> TorsoShellRegion:
    """Find garment material standing off the torso between hips and
    shoulders.

    The mesh's torso band is expressed in cylindrical coordinates about the
    spine polyline and cut into (height band, angular sector) cells. In each
    cell `radial_shell_split` looks for two radial clusters separated by air;
    everything beyond the split is shell.

    **Arms are excluded before anything is measured**, by keeping only
    vertices closer to the spine chain than to either arm chain. Without
    that, a T-posed character's outstretched arms read as a huge outer
    cluster in the lateral sectors on EVERY character and the detector fires
    on all of them. The exclusion is deliberately conservative at the
    armpit: material there is left to the ordinary weighting rather than
    risked.

    Strength fades in with how far past `radial_gap` the cell's gap actually
    is, and with height above the hip line, and is ceilinged at `ceiling` --
    a fitted shirt must come out a no-op, so every gate errs towards doing
    nothing.

    Pure numpy; no scipy, no bpy.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    count = vertices.shape[0]
    empty = TorsoShellRegion(
        mask=np.zeros(count, dtype=bool),
        strength=np.zeros(count, dtype=np.float64),
        smoothing_strength=np.zeros(count, dtype=np.float64),
        core_radius=0.0,
        region_top=float(shoulder_height),
        region_bottom=float(hip_height),
        band_shell_counts=[],
        widest_gap=0.0,
    )
    span = float(shoulder_height) - float(hip_height)
    if count == 0 or span <= _EPS or not spine_segments or len(spine_heights) < 2:
        return empty

    height = vertices[:, up_axis] * up_sign
    band = (height >= float(hip_height)) & (height <= float(shoulder_height))
    if int(np.count_nonzero(band)) < band_minimum:
        return empty

    order = np.argsort(np.asarray(spine_heights, dtype=np.float64))
    knot_height = np.asarray(spine_heights, dtype=np.float64)[order]
    knot_lateral = np.asarray(spine_lateral, dtype=np.float64)[order]
    knot_depth = np.asarray(spine_depth, dtype=np.float64)[order]
    centre_lateral = np.interp(height, knot_height, knot_lateral)
    centre_depth = np.interp(height, knot_height, knot_depth)
    offset_lateral = vertices[:, lateral_axis] - centre_lateral
    offset_depth = vertices[:, depth_axis] - centre_depth
    radius = np.hypot(offset_lateral, offset_depth)
    angle = np.arctan2(offset_depth, offset_lateral)

    distance_spine = chain_segment_distance(vertices, spine_segments)
    distance_arm = (
        chain_segment_distance(vertices, arm_segments)
        if arm_segments else np.full(count, np.inf)
    )
    torso = band & (distance_spine < distance_arm)
    if int(np.count_nonzero(torso)) < band_minimum:
        return empty
    core_radius = float(np.median(radius[torso]))
    if core_radius <= _EPS:
        return empty

    # Per-cell measurement, then a CONTINUOUS field. Classifying vertex by
    # vertex ("is it past this cell's split radius?") produces a speckled
    # mask -- one sector splits, its neighbour does not, and two adjacent
    # vertices end up owned by different bones. That is the same knife-edge
    # boundary this whole module exists to remove, just rotated, and
    # measured on a real character it made the tear WORSE. So the cell grid
    # carries two smooth quantities instead -- a confidence and a split
    # radius -- which are blurred across neighbouring cells, interpolated at
    # each vertex, and turned into strength by a radial ramp.
    edges = np.linspace(float(hip_height), float(shoulder_height), bands + 1)
    confidence = np.zeros((bands, sectors), dtype=np.float64)
    # No detection in a cell means "the shell starts beyond anything here".
    split_radius = np.full((bands, sectors), np.inf, dtype=np.float64)
    band_shell_counts: list[int] = []
    widest_gap = 0.0
    for index in range(bands):
        in_band = torso & (height >= edges[index]) & (height < edges[index + 1])
        rows = np.nonzero(in_band)[0]
        if rows.size < band_minimum:
            band_shell_counts.append(0)
            continue
        sector = ((angle[rows] + np.pi) / (2.0 * np.pi) * sectors).astype(np.int64) % sectors
        found = 0
        for cell in range(sectors):
            members = rows[sector == cell]
            if members.size < sector_minimum:
                continue
            gap, split = radial_shell_split(radius[members], core_radius, cluster_minimum=cluster_minimum)
            widest_gap = max(widest_gap, gap)
            if gap < radial_gap:
                continue
            confidence[index, cell] = _smoothstep(
                np.asarray((gap - radial_gap) / max(gap_fade, _EPS))
            )
            split_radius[index, cell] = split / core_radius
            found += int(np.count_nonzero(radius[members] > split + _EPS))
        band_shell_counts.append(found)

    if not np.any(confidence > 0.0):
        return empty
    # Blur both grids over neighbouring cells (wrapping around the sectors,
    # clamped across the height bands) so an isolated cell is damped towards
    # nothing while a contiguous patch -- a real flap -- keeps its strength,
    # and so the field has no cell-boundary steps left in it.
    kernel = np.array([0.25, 0.5, 0.25])
    finite = np.where(np.isfinite(split_radius), split_radius, 0.0)
    weight = np.where(np.isfinite(split_radius), 1.0, 0.0) * confidence
    numerator = finite * weight
    for _ in range(max(int(cell_blur), 0)):
        confidence = (
            kernel[0] * np.roll(confidence, 1, axis=1)
            + kernel[1] * confidence
            + kernel[2] * np.roll(confidence, -1, axis=1)
        )
        confidence = (
            kernel[0] * np.vstack([confidence[:1], confidence[:-1]])
            + kernel[1] * confidence
            + kernel[2] * np.vstack([confidence[1:], confidence[-1:]])
        )
        for grid_name in ("numerator", "weight"):
            grid = numerator if grid_name == "numerator" else weight
            grid = (
                kernel[0] * np.roll(grid, 1, axis=1)
                + kernel[1] * grid
                + kernel[2] * np.roll(grid, -1, axis=1)
            )
            grid = (
                kernel[0] * np.vstack([grid[:1], grid[:-1]])
                + kernel[1] * grid
                + kernel[2] * np.vstack([grid[1:], grid[-1:]])
            )
            if grid_name == "numerator":
                numerator = grid
            else:
                weight = grid
    smoothed_split = np.where(weight > _EPS, numerator / np.where(weight > _EPS, weight, 1.0), np.inf)

    # Bilinear sample of both grids at every torso vertex: cell centres in
    # height, wrapping cell centres in angle.
    band_position = np.clip(
        (height - float(hip_height)) / max(span / bands, _EPS) - 0.5, 0.0, bands - 1.0
    )
    lower_band = np.floor(band_position).astype(np.int64)
    upper_band = np.minimum(lower_band + 1, bands - 1)
    band_fraction = band_position - lower_band
    sector_position = (angle + np.pi) / (2.0 * np.pi) * sectors - 0.5
    lower_sector = np.floor(sector_position).astype(np.int64) % sectors
    upper_sector = (lower_sector + 1) % sectors
    sector_fraction = sector_position - np.floor(sector_position)

    def sample(grid: np.ndarray) -> np.ndarray:
        top_left = grid[lower_band, lower_sector]
        top_right = grid[lower_band, upper_sector]
        bottom_left = grid[upper_band, lower_sector]
        bottom_right = grid[upper_band, upper_sector]
        top = top_left * (1.0 - sector_fraction) + top_right * sector_fraction
        bottom = bottom_left * (1.0 - sector_fraction) + bottom_right * sector_fraction
        return top * (1.0 - band_fraction) + bottom * band_fraction

    sampled_confidence = sample(confidence)
    sampled_split = sample(np.where(np.isfinite(smoothed_split), smoothed_split, 1e9))
    beyond = _smoothstep((radius / core_radius - sampled_split) / max(radial_fade, _EPS))
    ramp = _smoothstep((height - float(hip_height)) / max(hip_margin * span, _EPS))
    shape = np.where(torso, sampled_confidence * beyond * ramp, 0.0)
    strength = shape * float(np.clip(ceiling, 0.0, 1.0))
    smoothing_strength = shape * float(np.clip(smoothing_ceiling, 0.0, 1.0))
    mask = shape > 1e-6
    if int(np.count_nonzero(mask)) < minimum_region_vertices:
        return empty
    region_heights = height[mask]
    return TorsoShellRegion(
        mask=mask,
        strength=strength,
        smoothing_strength=smoothing_strength,
        core_radius=core_radius,
        region_top=float(region_heights.max()),
        region_bottom=float(region_heights.min()),
        band_shell_counts=band_shell_counts,
        widest_gap=round(float(widest_gap), 4),
    )


def torso_shell_blend_weights(
    vertices: np.ndarray,
    region: TorsoShellRegion,
    bone_indices: np.ndarray,
    weights: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    spine_bones: Sequence[int],
    spine_bone_heights: Sequence[float],
    clavicle_left_bone: int = -1,
    clavicle_right_bone: int = -1,
    clavicle_height: float | None = None,
    clavicle_span: float = 0.0,
    lateral_midline: float = 0.0,
    max_influences: int = 4,
    shoulder_share: float = _TORSO_SHOULDER_SHARE,
    shoulder_softness: float = _TORSO_SHOULDER_SOFTNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-anchor torso-shell vertices to the spine trunk.

    Target weights for a shell vertex:

    * **spine by height** -- a partition of unity over the spine bones'
      own head heights (`pelvis`, `spine_1`, `spine_2`, `spine_3`), so a
      waist flap rides the lower spine and a chest panel the upper one. This
      is the whole point: a flap follows the TRUNK, at its own height, and
      never snaps to whichever arm bone happens to be nearest.
    * **clavicle pair near the shoulders only** -- up to `shoulder_share`,
      ramped in over the last stretch below the clavicle line AND with how
      far out to the side the vertex sits, split left/right by lateral
      position. A shoulder panel has to follow the shoulder; a flap over the
      sternum, at the same height, gets none of it because its lateral term
      is zero.

    `clavicle_span` is SIGNED: the clavicle_l head's lateral coordinate
    minus the clavicle_r head's, the same `left - right` convention as
    `hanging_region_blend_weights`. Its magnitude sets the shoulder reach
    and softness; its sign says which lateral direction is the rig's left,
    so the clavicle split lands on the bone actually on that side. The l/r
    suffixes come from the character's facing (module docstring), so a
    negative span is a normal rig, not an error. A zero span (unknown)
    falls back to +lateral = left.

    The clavicle, not the upper arm, is the shoulder target on purpose: the
    clavicle carries the shoulder's own motion while the upper arm carries
    the swing, and a vest sleeve-hole panel that follows a swinging arm is
    the failure this pass exists to remove.

    Mixed into the existing weights by `region.strength` (see
    `_mix_blend_targets`); a vertex outside the region is byte-identical.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    bone_indices = np.asarray(bone_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    rows = np.nonzero(region.mask)[0]
    if rows.size == 0:
        return bone_indices.copy(), weights.copy()
    max_influences = max(1, min(max_influences, bone_indices.shape[1]))

    height = vertices[rows, up_axis] * up_sign
    lateral = vertices[rows, lateral_axis]
    strength = region.strength[rows]

    knots = np.asarray(spine_bone_heights, dtype=np.float64)
    bones = list(spine_bones)
    order = np.argsort(knots)
    knots = knots[order]
    bones = [bones[i] for i in order]
    # Piecewise-linear hat functions over the spine knots: a partition of
    # unity, clamped flat outside the outermost knots.
    spine_shares = []
    for index in range(len(bones)):
        share = np.ones_like(height)
        if index > 0:
            lower = max(knots[index] - knots[index - 1], _EPS)
            share = np.minimum(share, (height - knots[index - 1]) / lower)
        if index < len(bones) - 1:
            upper = max(knots[index + 1] - knots[index], _EPS)
            share = np.minimum(share, (knots[index + 1] - height) / upper)
        spine_shares.append(np.clip(share, 0.0, 1.0))
    spine_stack = np.stack(spine_shares, axis=1)
    totals = np.sum(spine_stack, axis=1, keepdims=True)
    spine_stack = spine_stack / np.where(totals > _EPS, totals, 1.0)

    shoulder = np.zeros_like(height)
    if clavicle_left_bone >= 0 or clavicle_right_bone >= 0:
        if clavicle_height is not None and len(knots) >= 2:
            start = float(knots[-1])
            denominator = max(float(clavicle_height) - start, _EPS)
            vertical = _smoothstep((height - start) / denominator)
        else:
            vertical = np.zeros_like(height)
        reach = max(abs(float(clavicle_span)) / 2.0, _EPS)
        sideways = _smoothstep(np.abs(lateral - float(lateral_midline)) / reach)
        shoulder = np.clip(shoulder_share, 0.0, 1.0) * vertical * sideways

    # Signed (left - right): a negative span ramps `u` towards the clavicle
    # at the lower lateral coordinate, which is then clavicle_l.
    span = float(clavicle_span)
    if abs(span) <= _EPS:
        span = 1.0
    u = _smoothstep(0.5 + (lateral - float(lateral_midline)) / (max(shoulder_softness, _EPS) * span))

    target_bones = np.array(list(bones) + [clavicle_left_bone, clavicle_right_bone], dtype=np.int64)
    columns = [spine_stack[:, index] * (1.0 - shoulder) for index in range(len(bones))]
    columns.append(shoulder * u)
    columns.append(shoulder * (1.0 - u))
    target_weights = np.stack(columns, axis=1)
    return _mix_blend_targets(
        bone_indices, weights, rows, strength, target_bones, target_weights, max_influences
    )


def mix_smoothed_weights(
    bone_indices: np.ndarray,
    weights: np.ndarray,
    smoothed_bone_indices: np.ndarray,
    smoothed_weights: np.ndarray,
    rows: np.ndarray,
    strength: np.ndarray,
    max_influences: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a topology-smoothed weighting into the existing one, per vertex,
    by `strength` -- the same `(1 - s) * existing + s * replacement` shape as
    `_mix_blend_targets`, but with a per-ROW replacement (each vertex's own
    smoothed weight vector) instead of one shared target bone set.

    Why smoothing at all: the visible tear in a sheet of material is a
    function of the weight GRADIENT across it, not of the weight values. Two
    adjacent vertices driven by different bones pull apart under any pose
    that separates those bones, however sensible each vertex's own weights
    look in isolation. Re-targeting alone cannot fix that -- it just moves
    the gradient somewhere else, which is exactly what a measured
    symmetric-thigh re-target did (it lowered nothing and raised the
    boundary). Flattening the gradient over the region's own mesh
    neighbourhood does fix it.

    Mixing by `strength` rather than switching on the region mask matters:
    a hard switch leaves a smoothed patch abutting an unsmoothed one, which
    is a new gradient at the region boundary.
    """
    out_bone_indices = np.asarray(bone_indices, dtype=np.int64).copy()
    out_weights = np.asarray(weights, dtype=np.float64).copy()
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return out_bone_indices, out_weights
    max_influences = max(1, min(max_influences, out_bone_indices.shape[1]))

    width = out_bone_indices.shape[1]
    smoothed_width = smoothed_bone_indices.shape[1]
    for offset, row in enumerate(rows):
        blend = float(strength[offset])
        if blend <= 0.0:
            continue
        accumulated: dict[int, float] = {}
        for slot in range(width):
            bone = int(bone_indices[row, slot])
            value = float(weights[row, slot])
            if bone < 0 or value <= 0.0:
                continue
            accumulated[bone] = accumulated.get(bone, 0.0) + value * (1.0 - blend)
        for slot in range(smoothed_width):
            bone = int(smoothed_bone_indices[row, slot])
            value = float(smoothed_weights[row, slot])
            if bone < 0 or value <= 0.0:
                continue
            accumulated[bone] = accumulated.get(bone, 0.0) + value * blend
        if not accumulated:
            continue
        ordered = sorted(accumulated.items(), key=lambda item: -item[1])[:max_influences]
        total = sum(value for _, value in ordered)
        if total <= _EPS:
            continue
        out_bone_indices[row, :] = -1
        out_weights[row, :] = 0.0
        for slot, (bone, value) in enumerate(ordered):
            out_bone_indices[row, slot] = bone
            out_weights[row, slot] = value / total
    return out_bone_indices, out_weights


def detect_bridge_region(
    vertices: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    hip_height: float,
    hip_half_span: float,
    hip_midline: float,
    left_segments: Sequence[Sequence[np.ndarray]],
    right_segments: Sequence[Sequence[np.ndarray]],
    corridor: float = _BRIDGE_CORRIDOR,
    balance_threshold: float = _BRIDGE_BALANCE,
    clearance_threshold: float = _BRIDGE_CLEARANCE,
    clearance_fade: float = _BRIDGE_CLEARANCE_FADE,
    hip_margin: float = _BRIDGE_HIP_MARGIN,
    ceiling: float = _BRIDGE_CEILING,
    smoothing_ceiling: float = _BRIDGE_SMOOTHING_CEILING,
    minimum_region_vertices: int = _BRIDGE_MINIMUM_REGION_VERTICES,
) -> BridgeRegion:
    """Find material spanning the gap BETWEEN the two legs.

    Three conditions, all required:

    * **in the corridor** -- laterally between the legs, within `corridor`
      hip half-spans of the midline. This is what makes the rule
      bridging-specific rather than a second lateral blend: material outside
      the corridor is on a leg, whatever its distances say.
    * **balanced** -- comparably far from both leg chains.
    * **clear** -- far enough from the nearer leg that it cannot be that
      leg's own skin.

    Measured on the worst-stretched below-hip vertices of a real crotch-web
    tear: lateral offset 0.2-0.5 half-spans, nearer-leg distance 0.75-1.6
    half-spans. An ordinary inner-thigh vertex sits at its own leg's radius
    and fails `clearance`; a fitted character with no bridging fabric
    produces no region at all.

    Pure numpy; no scipy, no bpy.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    count = vertices.shape[0]
    empty = BridgeRegion(
        mask=np.zeros(count, dtype=bool),
        strength=np.zeros(count, dtype=np.float64),
        smoothing_strength=np.zeros(count, dtype=np.float64),
        region_top=float(hip_height),
        region_bottom=float(hip_height),
    )
    if count == 0 or hip_half_span <= _EPS or not left_segments or not right_segments:
        return empty

    height = vertices[:, up_axis] * up_sign
    lateral_offset = np.abs(vertices[:, lateral_axis] - float(hip_midline)) / float(hip_half_span)
    distance_left = chain_segment_distance(vertices, left_segments)
    distance_right = chain_segment_distance(vertices, right_segments)
    nearer = np.minimum(distance_left, distance_right)
    farther = np.maximum(distance_left, distance_right)
    balance = nearer / np.maximum(farther, _EPS)
    clearance = nearer / float(hip_half_span)

    below = height < float(hip_height)
    if int(np.count_nonzero(below)) < minimum_region_vertices:
        return empty
    leg_length = float(hip_height) - float(height[below].min())
    if leg_length <= _EPS:
        return empty

    candidate = (
        below
        & (lateral_offset <= corridor)
        & (balance > balance_threshold)
        & (clearance > clearance_threshold)
    )
    if not np.any(candidate):
        return empty

    ramp = _smoothstep((float(hip_height) - height) / max(hip_margin * leg_length, _EPS))
    corridor_fade = _smoothstep((corridor - lateral_offset) / max(corridor, _EPS))
    balance_fade = _smoothstep((balance - balance_threshold) / max(1.0 - balance_threshold, _EPS))
    clearance_fade_term = _smoothstep((clearance - clearance_threshold) / max(clearance_fade, _EPS))
    shape = np.where(candidate, ramp * corridor_fade * balance_fade * clearance_fade_term, 0.0)
    strength = shape * float(np.clip(ceiling, 0.0, 1.0))
    smoothing_strength = shape * float(np.clip(smoothing_ceiling, 0.0, 1.0))

    mask = shape > 1e-6
    if int(np.count_nonzero(mask)) < minimum_region_vertices:
        return empty
    region_heights = height[mask]
    return BridgeRegion(
        mask=mask,
        strength=strength,
        smoothing_strength=smoothing_strength,
        region_top=float(region_heights.max()),
        region_bottom=float(region_heights.min()),
    )


def bridge_blend_weights(
    vertices: np.ndarray,
    region: BridgeRegion,
    bone_indices: np.ndarray,
    weights: np.ndarray,
    *,
    up_axis: int,
    up_sign: float,
    lateral_axis: int,
    hip_height: float,
    hip_half_span: float,
    hip_midline: float,
    pelvis_bone: int,
    thigh_left_bone: int,
    thigh_right_bone: int,
    lateral_left: float | None = None,
    lateral_right: float | None = None,
    max_influences: int = 4,
    pelvis_floor: float = _BRIDGE_PELVIS_FLOOR,
    lateral_softness: float = _BRIDGE_LATERAL_SOFTNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Share a bridging vertex's stride tension between both thighs, with
    the pelvis holding the middle.

    Target weights:

    * `pelvis` -- never below `pelvis_floor`, rising back towards 1 at the
      hip line. This is the stabilizer: two thighs alone scissor in opposite
      directions and stretch the sheet between them however symmetric the
      split is, so a floor of pelvis weight has to hold the middle still.
    * `thigh_l` / `thigh_r` -- the remainder, split by lateral position
      through a smoothstep across the hip span, so a vertex ON the midline
      is driven exactly 50/50 and simply stays put when the legs scissor,
      while material nearer one leg still follows it.

    Symmetric by construction: mirroring a vertex across the midline
    mirrors its two thigh weights and leaves its pelvis weight unchanged.

    Which side is "left" follows the rig, not the axis: `lateral_left` /
    `lateral_right` are the thigh_l / thigh_r head coordinates on
    `lateral_axis`, and the split ramps from `thigh_right_bone` at
    `lateral_right` to `thigh_left_bone` at `lateral_left` -- the same
    signed `left - right` span `hanging_region_blend_weights` uses. The l/r
    suffixes come from the character's facing (module docstring), so
    `thigh_l` may sit at the LOWER lateral coordinate; a split hard-wired to
    +lateral would then hand the left leg's web to `thigh_r`. Without the
    two coordinates the split falls back to +lateral = left.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    bone_indices = np.asarray(bone_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    rows = np.nonzero(region.mask)[0]
    if rows.size == 0:
        return bone_indices.copy(), weights.copy()
    max_influences = max(1, min(max_influences, bone_indices.shape[1]))

    height = vertices[rows, up_axis] * up_sign
    lateral = vertices[rows, lateral_axis]
    strength = region.strength[rows]

    hem = float(region.region_bottom)
    denominator = max(float(hip_height) - hem, _EPS)
    down = np.clip((float(hip_height) - height) / denominator, 0.0, 1.0)
    floor = float(np.clip(pelvis_floor, 0.0, 1.0))
    pelvis_share = floor + (1.0 - floor) * (1.0 - _smoothstep(down))
    leg_share = 1.0 - pelvis_share

    # Signed: positive when thigh_l sits at the higher lateral coordinate,
    # negative when the rig's left is at the lower one.
    span = 0.0
    if lateral_left is not None and lateral_right is not None:
        span = float(lateral_left) - float(lateral_right)
    if abs(span) <= _EPS:
        span = 2.0 * float(hip_half_span) if hip_half_span > _EPS else 1.0
    u = _smoothstep(0.5 + (lateral - float(hip_midline)) / (max(lateral_softness, _EPS) * span))

    target_bones = np.array([pelvis_bone, thigh_left_bone, thigh_right_bone], dtype=np.int64)
    target_weights = np.stack([pelvis_share, leg_share * u, leg_share * (1.0 - u)], axis=1)
    return _mix_blend_targets(
        bone_indices, weights, rows, strength, target_bones, target_weights, max_influences
    )


# --- Template-conformant humanoid skeleton building (N3) ---
#
# PIPELINE_POLICY.md's S3 contract: the fitter (UniRig, here) only PLACES
# the family template's joints; it never emits bespoke topology, extra
# bones, or renamed bones (see backend/app/skeleton_template.py, N2). This
# section maps UniRig's semantic role names (this module's own vocabulary --
# "pelvis", "upper_arm_l", ... -- already matches the biped template's
# convention, see assets/skeletons/biped/README.md) onto the template's
# exact bone set/hierarchy/orientation, synthesizing the small set of
# required bones UniRig never predicts directly (`root`, `clavicle_l/r`).
#
# Bone HEAD positions come from whatever UniRig actually predicted (real
# mesh-fitted geometry); bone TAIL directions and lengths come from the
# template's rest_direction/relative_length, scaled to this character's own
# skeleton height, so every emitted bone's orientation matches the family
# convention regardless of what UniRig's own (unreliable, mixed-convention)
# tails looked like -- the same "anatomical tails" motivation
# unirig_bind_worker.build_armature already documents, extended to also fix
# roll (build_armature previously set no roll at all).


# --- optional-tier synthesis (fingers, toes, head_tip) ---------------------
#
# UniRig never predicts finger or toe joints, so a template-conformant build
# from its prediction alone stops at the 21 required bones and the character
# gets mitts. The template's other 45 bones (5 fingers x [3 segments + tip]
# per hand, toes + toes_tip per foot, head_tip) are fully specified --
# parent, rest_direction, relative_length, roll -- so they can be GROWN from
# the hand/foot/head the fitter did place, rather than predicted. Every clip
# in the family store animates all 66 bones by exact name, so once these
# bones exist, direct-attach drives them with no further change.
#
# Two things are deliberately anchored locally rather than globally:
#
#   * Chain root. Each optional chain starts exactly at its anchor bone's
#     TAIL (hand_l's tail for every left finger, foot_l's tail for the left
#     toes, head's tail for head_tip) -- so no gap opens at the wrist/ankle
#     no matter how the chain is scaled, and the anchor's own required-bone
#     geometry is left untouched.
#
#   * Chain length. The template's relative_lengths are fractions of the
#     DONOR's height, so scaling them by this character's height alone
#     assumes donor-proportioned hands. Stylized characters routinely have
#     oversized mitts or stubby feet, so when the bind mesh is available the
#     chain is instead scaled by THIS character's own measured hand/foot
#     extent (see `measure_distal_span`): the fingers span the mesh hand,
#     not a donor-proportioned fraction of the body. With no mesh in hand
#     the scale falls back to 1.0, i.e. exact template proportion.
#
# The template records no per-knuckle head offsets (only a direction and a
# length per bone), so the five finger chains all start at the same point
# and separate only by their own differing rest_directions -- a narrow fan
# for index/middle/ring/pinky (which are within ~14deg of each other in the
# donor) and a wide one for the thumb (~35deg off the hand axis and out of
# the hand plane). That is the entire fan signal the template carries.

# Keep the last joint of a synthesized chain just inside the mesh surface
# rather than exactly at it: a joint head sitting on the fingertip surface
# reads as "outside" to `joint_containment` half the time, and skinning a
# vertex to a bone whose head is at the surface is worse than one slightly
# inside.
_DISTAL_SPAN_MARGIN = 0.9
# Below this many mesh vertices in the hand/foot region the measurement is
# noise, not anatomy -- fall back to template proportion.
_MIN_DISTAL_SAMPLES = 8
# A measured span this far from the template's own proportion is more likely
# a measurement escaping the hand (an arm held against a cape, a mesh whose
# hand region merges into the torso) than a real anatomy signal.
_MIN_DISTAL_SCALE = 0.35
_MAX_DISTAL_SCALE = 2.5
# The hand/foot region is a CYLINDER around the chain's axis, not a sphere
# around the anchor head: a sphere sized from the template's own expected
# span assumes the anchor joint is correctly placed, and when it is not (a
# wrist fitted halfway up the forearm, which UniRig does produce) the sphere
# ends before the mesh does and the "measurement" is just the sphere's own
# radius -- a saturated, meaningless number that pins every chain to the
# clamp. Bounding sideways instead keeps the arm's own tube in and the torso
# out, and lets the measurement run to wherever the limb actually ends.
# Both limits are fractions of the character's own skeleton height.
_DISTAL_LATERAL_LIMIT_FACTOR = 0.15
_DISTAL_ALONG_LIMIT_FACTOR = 0.35
# The measured span is a high quantile, not the max: one stray vertex (a
# nail, a spike, a stray weld) must not stretch the whole chain.
_DISTAL_SPAN_QUANTILE = 0.99
# Below this the template's own chain span is degenerate and nothing can be
# scaled against it.
_EPS_LENGTH = 1e-9


def template_anchor_bone(template: Template, name: str) -> str | None:
    """The nearest REQUIRED ancestor of `name` (itself, if required).

    This is what an optional chain is grown from and measured against:
    every left-hand finger bone resolves to `hand_l`, the toes to `foot_l`
    / `foot_r`, `head_tip` to `head`.
    """
    try:
        bone = template.bone(name)
    except KeyError:
        return None
    while bone.tier == "optional":
        if bone.parent is None:
            return None
        bone = template.bone(bone.parent)
    return bone.name


def optional_chain_root(template: Template, name: str) -> str | None:
    """The first optional bone in `name`'s chain -- the one whose own parent
    is required. `index_3_l` and `index_tip_l` both resolve to `index_1_l`;
    `toes_tip_r` to `toes_r`; `head_tip` to itself. None for a required bone
    or a name the template does not carry.

    Scaling is keyed by CHAIN, not by anchor: the thumb leaves the palm
    plane at ~35deg, so measuring the mesh hand along the thumb's own axis
    is a different (and much shorter, on a mitt) question than measuring it
    along the four straight fingers'. Sharing one hand-wide scale between
    them pushes the thumb straight out through the surface.
    """
    try:
        bone = template.bone(name)
    except KeyError:
        return None
    if bone.tier != "optional":
        return None
    while bone.parent is not None:
        parent = template.bone(bone.parent)
        if parent.tier == "required":
            return bone.name
        bone = parent
    return None


def optional_chain_layout(template: Template, root: str) -> dict[str, np.ndarray]:
    """Head offsets (in units of the donor's height) of `root` and every
    optional bone below it, relative to the head of the REQUIRED bone the
    chain hangs off.

    The template stores only per-bone direction and length, so a bone's head
    is its parent's tail -- which is exactly how the donor's own finger
    chains are reconstructed here. `root`'s own offset is therefore the
    anchor bone's rest direction times its relative length: the wrist-to-
    knuckle (or ankle-to-ball) step the chain starts from.
    """
    root_spec = template.bone(root)
    if root_spec.tier != "optional" or root_spec.parent is None:
        return {}
    anchor = template.bone(root_spec.parent)
    offsets: dict[str, np.ndarray] = {
        root: np.asarray(anchor.rest_direction, dtype=np.float64) * anchor.relative_length
    }
    grew = True
    while grew:
        grew = False
        for bone in template.bones:
            if bone.tier != "optional" or bone.parent is None:
                continue
            if bone.name in offsets or bone.parent not in offsets:
                continue
            parent = template.bone(bone.parent)
            offsets[bone.name] = offsets[bone.parent] + np.asarray(
                parent.rest_direction, dtype=np.float64
            ) * parent.relative_length
            grew = True
    return offsets


def measure_distal_span(
    anchor_head: np.ndarray,
    axis: np.ndarray,
    vertices: np.ndarray,
    lateral_limit: float,
    along_limit: float,
) -> tuple[float, bool] | None:
    """How far the mesh actually extends past `anchor_head` along `axis`,
    and whether that measurement ran into `along_limit` (saturated).

    The region is a cylinder: every vertex on the positive side of
    `anchor_head` along `axis`, no further sideways from the axis than
    `lateral_limit` and no further along it than `along_limit`. For a hand
    in the fitter's canonical T-pose that is the forearm-and-hand tube and
    nothing else -- the upper arm and torso project negative, and anything
    off to the side is outside the cylinder.

    Returns None when too few vertices qualify to call it a measurement.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) == 0 or lateral_limit <= 0.0 or along_limit <= 0.0:
        return None
    axis_unit = _normalize(np.asarray(axis, dtype=np.float64))
    if axis_unit is None:
        return None
    relative = vertices - np.asarray(anchor_head, dtype=np.float64)
    projection = relative @ axis_unit
    lateral = np.linalg.norm(relative - np.outer(projection, axis_unit), axis=1)
    selected = projection[
        (projection > 0.0) & (projection <= along_limit) & (lateral <= lateral_limit)
    ]
    if len(selected) < _MIN_DISTAL_SAMPLES:
        return None
    span = float(np.quantile(selected, _DISTAL_SPAN_QUANTILE))
    return span, bool(span >= along_limit * 0.98)


def optional_chain_scale(
    template: Template,
    root: str,
    anchor_head: np.ndarray,
    skeleton_height: float,
    mesh_vertices: np.ndarray | None,
) -> tuple[float, str]:
    """Length multiplier for the optional chain rooted at `root`, and where
    it came from. `anchor_head` is the head of the required bone the chain
    hangs off (`hand_l` for a left finger, `foot_r` for the right toes).

    The chain's own segments are scaled; the anchor bone's length (the
    wrist-to-knuckle / ankle-to-ball offset the chain starts at) is not,
    since it is a required bone whose geometry this build does not touch.
    So the multiplier solves

        margin * measured_span = anchor_length + scale * template_chain_span

    for `scale`, all along the chain's own axis. Falls back to 1.0 (exact
    template proportion) whenever the mesh cannot answer -- no mesh, a
    degenerate skeleton, or too few vertices in the region.
    """
    if skeleton_height <= 0.0:
        return 1.0, "template-proportion-degenerate-height"
    offsets = optional_chain_layout(template, root)
    if not offsets:
        return 1.0, "template-proportion-no-chain"
    # The direction this one chain points: the mean of its own head offsets,
    # so the mesh is measured along the finger being grown rather than along
    # some hand-wide average. That is what keeps a thumb inside a mitt that
    # has no thumb bump while the four straight fingers still reach the
    # fingertips.
    axis = _normalize(np.mean(np.stack(list(offsets.values())), axis=0))
    if axis is None:
        return 1.0, "template-proportion-degenerate-chain"
    anchor_spec = template.bone(template.bone(root).parent)
    anchor_projection = float(
        np.dot(np.asarray(anchor_spec.rest_direction, dtype=np.float64), axis)
        * anchor_spec.relative_length
    )
    template_span = max(float(np.dot(offset, axis)) for offset in offsets.values())
    chain_span = template_span - anchor_projection
    if chain_span <= _EPS_LENGTH:
        return 1.0, "template-proportion-degenerate-chain"
    if mesh_vertices is None:
        return 1.0, "template-proportion-no-mesh"
    measurement = measure_distal_span(
        anchor_head,
        axis,
        mesh_vertices,
        lateral_limit=_DISTAL_LATERAL_LIMIT_FACTOR * skeleton_height,
        along_limit=_DISTAL_ALONG_LIMIT_FACTOR * skeleton_height,
    )
    if measurement is None:
        return 1.0, "template-proportion-no-region-vertices"
    measured, saturated = measurement
    scale = (_DISTAL_SPAN_MARGIN * measured - anchor_projection * skeleton_height) / (
        chain_span * skeleton_height
    )
    if not np.isfinite(scale):
        return 1.0, "template-proportion-non-finite"
    if saturated:
        # The limb kept going past the region's own bound, so the number is
        # a floor, not a measurement. Growing the chain to a floor is how a
        # wrist stuck up the forearm turns into finger bones several times
        # their proper length, so this falls back to the family template's
        # own proportion rather than passing the floor off as a length.
        return 1.0, "template-proportion-region-saturated"
    if scale > _MAX_DISTAL_SCALE:
        # Overwhelmingly this means the anchor joint is misfitted (a wrist
        # placed up the forearm makes the "hand" measure several times a
        # donor-proportioned one), not that the character has a giant hand.
        # Measured on the punk character: with `hand_l` left at x=0.229 on
        # an arm reaching x=0.492, every one of the ten finger chains hit
        # the clamp and the thumb chains left the arm entirely -- long
        # diagonal bones hanging below the forearm, visible in the rig
        # overlay. A clamp is not a measurement: when the number is out of
        # range the mesh has not answered the question, so answer it the
        # same way `template-proportion-no-mesh` does. (Characters whose
        # wrist snap succeeded -- zix, Mary -- measure 1.06-1.76 and never
        # reach this branch, so they are unaffected.)
        return 1.0, "template-proportion-implausible-measure"
    if scale < _MIN_DISTAL_SCALE:
        # The other end of the clamp is the SAFE direction and keeps its
        # old behaviour: a mitt hand with no modelled fingers measures
        # almost no span, and clamping the chain short keeps the grown
        # bones inside the mesh instead of poking out of it.
        return _MIN_DISTAL_SCALE, "measured-mesh-clamped"
    return float(scale), "measured-mesh"


# --- distal anchor refinement (S3 joint PLACEMENT) --------------------------
#
# UniRig fits `hand_l`/`hand_r` roughly HALFWAY UP THE FOREARM. Measured on
# zix: `lower_arm_l` head x=0.0998, `hand_l` head x=0.2366, while the mesh arm
# runs to x=0.497 and the modelled hand (a clear cross-section bulge) spans
# x 0.39-0.50. Every finger chain grown from that anchor therefore articulates
# the distal FOREARM, and no chain-growth strategy anchored there can do
# better: raising `_MAX_DISTAL_SCALE` to push the tips into the real fingers
# made containment WORSE (0.867 -> 0.800), because the thumb leaves the mesh
# before the fingers arrive. The fix has to be the anchor's PLACEMENT.
#
# The signal is the limb's own cross-section radius profile, walked along the
# elbow -> predicted-hand axis: a forearm that narrows to a WRIST, then a
# palm BULGE, then the fingers, then the mesh ends. The anatomical wrist is
# where the radius leaves the forearm baseline on its way up to the bulge.
# The same shape reads an ankle (calf belly -> ankle narrow -> foot bulge),
# which is why feet run through the identical detector -- on much tighter
# guard rails, since UniRig already places feet well and a foot that moves is
# far more likely to be a misread than a fix.
#
# Everything here is a pure function of joints + mesh vertices. A refinement
# that cannot be made confidently is SKIPPED, never approximated: a skipped
# limb leaves the rig byte-for-byte what it was before this code existed.

# The measurement region runs from the PROXIMAL joint (elbow/knee), not from
# the anchor, so it has to be long enough to contain the whole segment plus
# the hand -- on zix that is 0.40 against a 0.86 skeleton height. The old
# `_DISTAL_ALONG_LIMIT_FACTOR` (0.35) is measured from the wrist and is far
# too short here: it stops 0.1 short of the fingertips and the profile never
# sees the bulge at all.
_REFINE_ALONG_LIMIT_FACTOR = 0.6
# Sideways bound on the region, as a fraction of skeleton height -- keeps the
# torso out of an arm's tube.
_REFINE_LATERAL_LIMIT_FACTOR = 0.15
# ...and never more than this fraction of the way to the MIRROR anchor. A
# vertical knee->ankle axis with a height-derived lateral bound swallows the
# opposite leg (measured on zix: p90 radius 0.11 instead of the leg's real
# 0.04, i.e. half the stance width), and the "ankle" it then finds is an
# artefact of the other leg's silhouette.
_REFINE_MIRROR_LATERAL_FRACTION = 0.45
_REFINE_BIN_COUNT = 64
# A bin thinner than this is noise, not a cross-section.
_REFINE_MIN_BIN_SAMPLES = 6
# Fewer usable bins than this is not a profile worth reading a wrist out of.
_REFINE_MIN_USABLE_BINS = 12
# The bulge peak must lie in the distal part of the limb; a "peak" in the
# proximal half is the elbow or the calf belly.
_REFINE_PEAK_SEARCH_FRACTION = 0.45
# The forearm/shin baseline is the median radius over this fraction band of
# the distance to the peak -- clear of the elbow flare at one end and of the
# bulge's own rise at the other.
_REFINE_BASELINE_BAND = (0.15, 0.65)
# The wrist is the last bin, walking back from the peak, still within this
# fraction of the way from baseline to peak. Calibrated on zix: 0.35 lands
# the wrist at x=0.385 against a modelled palm starting at x=0.39.
_REFINE_WRIST_LEVEL_FRACTION = 0.35
# Radius per bin is a high quantile of lateral distance, not the max: one
# stray vertex must not invent a bulge.
_REFINE_RADIUS_QUANTILE = 0.90
# Below this fraction of the limb extent -- or below this many bins,
# whichever is larger -- the anchor is already where the detector would put
# it: report it and change nothing. The bin floor is not cosmetic. A p90
# radius flips to the WIDER value as soon as a tenth of its slab is wide, so
# the detected wrist can sit up to a bin proximal of the true step; the
# estimate's own resolution is therefore about a bin either way, and a move
# smaller than two of them is inside that noise. (It errs proximally, into
# the forearm, which is the safe direction: an anchor slightly short of the
# wrist is still in the arm.)
_REFINE_MIN_MOVE_FRACTION = 0.02
_REFINE_MIN_MOVE_BINS = 2.0
# Guard rails on where a refined anchor may land, as fractions of the limb's
# own MEASURED extent (proximal joint -> mesh tip along the axis). Anchoring
# the move cap to the rig's own elbow->hand segment instead would be
# self-defeating: the error being corrected is exactly that this segment is
# too short, so the cap would shrink with the mistake (on zix the required
# move is 109% of the fitted segment but only 37% of the measured extent).
_REFINE_MAX_MOVE_FRACTION = 0.40
# ...unless the read is strongly evidenced, in which case it may move
# further. The base cap was calibrated on characters whose wrist snap
# already succeeded (zix 0.368/0.380, Mary 0.366/0.377 of limb extent) and
# so sat only 5% above the working set -- tight enough that the punk
# character, whose UniRig wrist landed even further up the forearm, needed
# 0.425/0.434 and was refused. That refusal is what left its finger chains
# growing off a mid-forearm anchor. The wider cap is gated on the two
# independent signals that say the profile really did find a hand, so a
# MARGINAL read is capped exactly as before: only a confident, strongly
# bulged one is allowed the longer move.
_REFINE_CONFIDENT_MAX_MOVE_FRACTION = 0.50
_REFINE_CONFIDENT_MOVE_MIN_CONFIDENCE = 0.70
_REFINE_CONFIDENT_MOVE_MIN_BULGE = 1.60
_REFINE_MIN_WRIST_FRACTION = 0.10
_REFINE_MAX_WRIST_FRACTION = 0.90
# How much bulge counts as a hand/foot rather than a lumpy forearm.
_REFINE_MIN_BULGE_RATIO = 1.25
_REFINE_MIN_CONFIDENCE = 0.35

# Per-kind overrides. Feet are deliberately conservative: they are usually
# well placed already, the foot's own extent is short, and a mesh whose legs
# nearly touch or whose skirt/base geometry hangs below the sole gives a
# far noisier profile than an arm held out in a T-pose.
_REFINE_PRESETS: dict[str, dict[str, float]] = {
    "hand": {},
    "foot": {
        "min_bulge_ratio": 1.35,
        "max_move_fraction": 0.15,
        "confident_max_move_fraction": 0.15,
        "min_confidence": 0.50,
    },
}

# (proximal, anchor, mirror anchor, preset). The mirror is used only to bound
# the measurement region sideways; it is never moved by its partner's pass.
DEFAULT_DISTAL_CHAINS: tuple[tuple[str, str, str, str], ...] = (
    ("lower_arm_l", "hand_l", "hand_r", "hand"),
    ("lower_arm_r", "hand_r", "hand_l", "hand"),
    ("calf_l", "foot_l", "foot_r", "foot"),
    ("calf_r", "foot_r", "foot_l", "foot"),
)


class LimbRadiusProfile(NamedTuple):
    """Binned cross-section radius of a limb along its own axis.

    `counts[i]`/`radii[i]` describe the slab `[i*width, (i+1)*width)` measured
    from the proximal joint; `usable` is the length of the CONTIGUOUS run of
    populated bins starting at bin 0 (a limb is connected -- a gap means the
    region has left the limb), so `extent = usable * width` is where the mesh
    limb ends. `saturated` says that run reached the region's own outer bound,
    which makes `extent` a floor rather than a measurement.
    """

    width: float
    counts: tuple[int, ...]
    radii: tuple[float, ...]
    usable: int
    extent: float
    saturated: bool


class DistalRefinement(NamedTuple):
    """What `refine_distal_anchor` decided about one limb.

    `applied` false means the rig is left exactly as it was; `reason` then
    carries `refine_skipped:<why>`. `wrist_to` is populated whenever the
    detector produced a candidate at all -- including for a candidate that a
    guard rail then rejected, so the skip is diagnosable rather than silent.
    """

    proximal: str
    anchor: str
    applied: bool
    reason: str
    wrist_from: tuple[float, float, float] | None = None
    wrist_to: tuple[float, float, float] | None = None
    # `moved` is the ALONG-limb correction (the wrist station); the sideways
    # correction onto the limb's medial point is reported separately.
    moved: float = 0.0
    medial_offset: float = 0.0
    limb_extent: float = 0.0
    hand_extent: float = 0.0
    bulge_ratio: float = 0.0
    confidence: float = 0.0


def _refine_skip(proximal: str, anchor: str, why: str, **extra) -> DistalRefinement:
    return DistalRefinement(
        proximal=proximal, anchor=anchor, applied=False,
        reason=f"refine_skipped:{why}", **extra,
    )


def limb_radius_profile(
    proximal_head: np.ndarray,
    axis: np.ndarray,
    vertices: np.ndarray,
    lateral_limit: float,
    along_limit: float,
    *,
    bins: int = _REFINE_BIN_COUNT,
    min_bin_samples: int = _REFINE_MIN_BIN_SAMPLES,
) -> LimbRadiusProfile | None:
    """Cross-section radius of the mesh tube around `axis`, bin by bin.

    Same cylinder selection as `measure_distal_span` (positive projection,
    bounded sideways and along), but resolved into a PROFILE rather than
    collapsed to a single span -- the shape is the whole signal here.
    Returns None when the region holds too little to bin.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) == 0 or lateral_limit <= 0.0 or along_limit <= 0.0 or bins <= 0:
        return None
    axis_unit = _normalize(np.asarray(axis, dtype=np.float64))
    if axis_unit is None:
        return None
    relative = vertices - np.asarray(proximal_head, dtype=np.float64)
    projection = relative @ axis_unit
    lateral = np.linalg.norm(relative - np.outer(projection, axis_unit), axis=1)
    keep = (projection > 0.0) & (projection <= along_limit) & (lateral <= lateral_limit)
    if int(keep.sum()) < _MIN_DISTAL_SAMPLES:
        return None
    projection = projection[keep]
    lateral = lateral[keep]
    width = along_limit / bins
    index = np.minimum((projection / width).astype(np.int64), bins - 1)
    counts: list[int] = []
    radii: list[float] = []
    for bin_index in range(bins):
        selected = index == bin_index
        count = int(selected.sum())
        counts.append(count)
        radii.append(
            float(np.quantile(lateral[selected], _REFINE_RADIUS_QUANTILE)) if count else 0.0
        )
    usable = 0
    for count in counts:
        if count < min_bin_samples:
            break
        usable += 1
    return LimbRadiusProfile(
        width=float(width),
        counts=tuple(counts),
        radii=tuple(radii),
        usable=usable,
        extent=float(usable * width),
        saturated=usable >= bins,
    )


def limb_medial_point(
    proximal_head: np.ndarray,
    axis: np.ndarray,
    vertices: np.ndarray,
    station: float,
    lateral_limit: float,
    half_thickness: float,
) -> np.ndarray | None:
    """Centroid of the limb's cross-section at `station` along `axis`.

    Same cylinder as `limb_radius_profile`, one slab of it. This is the
    limb's medial point there, which is NOT generally on the proximal ->
    anchor line: that line was drawn through a mis-fitted anchor. Measured
    on zix's left hand it runs 0.025 to one side and 0.017 below the mesh
    hand's own centre -- most of a finger's thickness, which is the
    difference between a finger joint inside the mesh and one just outside
    it. Returns None when the slab is too thin to average.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    axis_unit = _normalize(np.asarray(axis, dtype=np.float64))
    if axis_unit is None or len(vertices) == 0 or half_thickness <= 0.0:
        return None
    relative = vertices - np.asarray(proximal_head, dtype=np.float64)
    projection = relative @ axis_unit
    lateral = np.linalg.norm(relative - np.outer(projection, axis_unit), axis=1)
    slab = vertices[
        (np.abs(projection - station) <= half_thickness) & (lateral <= lateral_limit)
    ]
    if len(slab) < _MIN_DISTAL_SAMPLES:
        return None
    return np.mean(slab, axis=0)


def refine_distal_anchor(
    joints: np.ndarray,
    names: Sequence[str],
    mesh_vertices: np.ndarray | None,
    chain: tuple[str, str] = ("lower_arm_l", "hand_l"),
    *,
    mirror_anchor: str | None = None,
    skeleton_height: float | None = None,
    min_bulge_ratio: float = _REFINE_MIN_BULGE_RATIO,
    max_move_fraction: float = _REFINE_MAX_MOVE_FRACTION,
    confident_max_move_fraction: float = _REFINE_CONFIDENT_MAX_MOVE_FRACTION,
    confident_move_min_confidence: float = _REFINE_CONFIDENT_MOVE_MIN_CONFIDENCE,
    confident_move_min_bulge: float = _REFINE_CONFIDENT_MOVE_MIN_BULGE,
    min_confidence: float = _REFINE_MIN_CONFIDENCE,
    bins: int = _REFINE_BIN_COUNT,
) -> DistalRefinement:
    """Where the distal anchor of `chain` SHOULD sit, given this mesh.

    `chain` is `(proximal, anchor)` -- `("lower_arm_l", "hand_l")` for a left
    wrist, `("calf_r", "foot_r")` for a right ankle. `joints[i]` is
    `names[i]`'s head position, in the same house frame everything else in
    this module uses.

    Walks the proximal -> anchor axis, bins the mesh's cross-section radius
    along it (`limb_radius_profile`), finds the distal BULGE (the hand or the
    foot) and puts the anchor at the narrow immediately before it -- the
    anatomical wrist/ankle. Reports the limb's measured extent, the extent
    left past the refined anchor (`hand_extent`: what a finger chain then has
    to grow into), and a confidence.

    Refuses rather than guesses. Every early return names its own reason, and
    a refusal means the caller must leave the joint untouched.
    """
    proximal_name, anchor_name = chain
    joints = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    by_name: dict[str, np.ndarray] = {}
    for name, position in zip(names, joints):
        if name not in by_name:
            by_name[name] = position
    if proximal_name not in by_name or anchor_name not in by_name:
        return _refine_skip(proximal_name, anchor_name, "missing-joint")
    proximal_head = by_name[proximal_name]
    anchor_head = by_name[anchor_name]
    anchor_tuple = tuple(float(v) for v in anchor_head)

    axis = _normalize(anchor_head - proximal_head)
    segment_length = float(np.linalg.norm(anchor_head - proximal_head))
    if axis is None or segment_length <= _EPS_LENGTH:
        return _refine_skip(proximal_name, anchor_name, "degenerate-axis",
                            wrist_from=anchor_tuple)
    if skeleton_height is None:
        skeleton_height = float(joints[:, VERTICAL_AXIS].max() - joints[:, VERTICAL_AXIS].min())
    if skeleton_height <= 0.0:
        return _refine_skip(proximal_name, anchor_name, "degenerate-height",
                            wrist_from=anchor_tuple)
    if mesh_vertices is None or len(np.asarray(mesh_vertices).reshape(-1, 3)) == 0:
        return _refine_skip(proximal_name, anchor_name, "no-mesh", wrist_from=anchor_tuple)

    lateral_limit = _REFINE_LATERAL_LIMIT_FACTOR * skeleton_height
    if mirror_anchor is not None and mirror_anchor in by_name:
        separation = float(np.linalg.norm(by_name[mirror_anchor] - anchor_head))
        if separation > 0.0:
            lateral_limit = min(lateral_limit, _REFINE_MIRROR_LATERAL_FRACTION * separation)

    profile = limb_radius_profile(
        proximal_head, axis, mesh_vertices,
        lateral_limit=lateral_limit,
        along_limit=_REFINE_ALONG_LIMIT_FACTOR * skeleton_height,
        bins=bins,
    )
    if profile is None:
        return _refine_skip(proximal_name, anchor_name, "no-region-vertices",
                            wrist_from=anchor_tuple)
    if profile.usable < _REFINE_MIN_USABLE_BINS:
        return _refine_skip(proximal_name, anchor_name, "profile-too-short",
                            wrist_from=anchor_tuple, limb_extent=profile.extent)
    if profile.saturated:
        # The mesh kept going past the region's outer bound, so `extent` is a
        # floor and every fraction-of-extent guard rail below would be scaled
        # off a number that means nothing. On zix that is exactly what the
        # legs do -- geometry hangs a full body-height below the soles.
        return _refine_skip(proximal_name, anchor_name, "region-saturated",
                            wrist_from=anchor_tuple, limb_extent=profile.extent)

    extent = profile.extent
    radii = np.asarray(profile.radii[:profile.usable], dtype=np.float64)
    counts = np.asarray(profile.counts[:profile.usable], dtype=np.int64)
    centres = (np.arange(profile.usable, dtype=np.float64) + 0.5) * profile.width

    search = centres >= _REFINE_PEAK_SEARCH_FRACTION * extent
    if not search.any():
        return _refine_skip(proximal_name, anchor_name, "no-peak-band",
                            wrist_from=anchor_tuple, limb_extent=extent)
    peak_index = int(np.flatnonzero(search)[np.argmax(radii[search])])
    peak_centre = float(centres[peak_index])
    peak_radius = float(radii[peak_index])

    low, high = _REFINE_BASELINE_BAND
    band = (centres >= low * peak_centre) & (centres <= high * peak_centre)
    if int(band.sum()) < 3:
        band = centres < peak_centre
    if int(band.sum()) < 3:
        return _refine_skip(proximal_name, anchor_name, "no-baseline",
                            wrist_from=anchor_tuple, limb_extent=extent)
    baseline = float(np.median(radii[band]))
    if baseline <= _EPS_LENGTH:
        return _refine_skip(proximal_name, anchor_name, "degenerate-baseline",
                            wrist_from=anchor_tuple, limb_extent=extent)

    bulge_ratio = peak_radius / baseline
    if bulge_ratio < min_bulge_ratio:
        # A mitt, a sleeve, a mesh whose hand merges into the forearm: there
        # is no distal anatomy to snap to, so nothing is snapped.
        return _refine_skip(proximal_name, anchor_name, "no-bulge",
                            wrist_from=anchor_tuple, limb_extent=extent,
                            bulge_ratio=bulge_ratio)

    threshold = baseline + _REFINE_WRIST_LEVEL_FRACTION * (peak_radius - baseline)
    below = np.flatnonzero((radii[:peak_index] <= threshold))
    if not len(below):
        return _refine_skip(proximal_name, anchor_name, "no-wrist-narrow",
                            wrist_from=anchor_tuple, limb_extent=extent,
                            bulge_ratio=bulge_ratio)
    wrist_index = int(below[-1])
    # Interpolate the threshold crossing between this bin and the next
    # rather than snapping to a bin centre: half a bin of quantization is
    # the same order as the "did it move at all?" test below, and a wrist
    # that jitters by a bin between two runs of the same character is not a
    # measurement anyone should act on.
    wrist_centre = float(centres[wrist_index])
    next_radius = float(radii[wrist_index + 1]) if wrist_index + 1 < len(radii) else None
    if next_radius is not None and next_radius > radii[wrist_index]:
        fraction = (threshold - float(radii[wrist_index])) / (next_radius - float(radii[wrist_index]))
        wrist_centre += float(np.clip(fraction, 0.0, 1.0)) * profile.width
    wrist_to = proximal_head + axis * wrist_centre
    # ...and off the proximal->anchor line onto the limb's own medial point
    # there. The line was drawn through the joint being corrected, so it is
    # only as centred as that joint was; every synthesized finger then fans
    # out from wherever it left the hand's centre, which is what puts them
    # through the surface of a thin, separated-finger hand. Bounded by the
    # slab's own radius -- a medial point cannot be further off-axis than
    # the tube it was measured in.
    medial_offset = 0.0
    # Measured at the WRIST slab, not averaged over the whole hand: the hand's
    # own medial line curves away from the arm axis, and centring the anchor
    # on that average tilts the (straight, template-directed) finger fan off
    # the wrist it starts at. Tried both live -- the wrist slab keeps zix at
    # parity, the hand-region average costs it two joints.
    medial = limb_medial_point(
        proximal_head, axis, mesh_vertices, wrist_centre,
        lateral_limit, 0.5 * profile.width,
    )
    if medial is not None:
        offset = medial - wrist_to
        offset = offset - float(np.dot(offset, axis)) * axis
        magnitude = float(np.linalg.norm(offset))
        limit = float(radii[wrist_index])
        if magnitude > limit > 0.0:
            offset = offset * (limit / magnitude)
            magnitude = limit
        wrist_to = wrist_to + offset
        medial_offset = magnitude
    wrist_tuple = tuple(float(v) for v in wrist_to)
    hand_extent = extent - wrist_centre
    moved = wrist_centre - segment_length

    support = counts[wrist_index:peak_index + 1]
    density = float(np.clip(
        (float(support.min()) if len(support) else 0.0) / (3.0 * _REFINE_MIN_BIN_SAMPLES),
        0.0, 1.0,
    ))
    shape = float(np.clip(
        (bulge_ratio - 1.0) / max(2.0 * (min_bulge_ratio - 1.0), _EPS_LENGTH), 0.0, 1.0,
    ))
    confidence = round(shape * density, 4)
    common = dict(
        wrist_from=anchor_tuple, wrist_to=wrist_tuple, moved=round(moved, 6),
        medial_offset=round(medial_offset, 6),
        limb_extent=round(extent, 6), hand_extent=round(hand_extent, 6),
        bulge_ratio=round(bulge_ratio, 4), confidence=confidence,
    )

    if not (_REFINE_MIN_WRIST_FRACTION * extent <= wrist_centre <= _REFINE_MAX_WRIST_FRACTION * extent):
        return _refine_skip(proximal_name, anchor_name, "wrist-out-of-band", **common)
    if abs(moved) < max(_REFINE_MIN_MOVE_FRACTION * extent, _REFINE_MIN_MOVE_BINS * profile.width):
        # The fitter already had it right -- which is what the feet do, and
        # worth recording as a measurement rather than as a non-event.
        return _refine_skip(proximal_name, anchor_name, "already-placed", **common)
    move_cap = max_move_fraction
    if (
        confident_max_move_fraction > max_move_fraction
        and confidence >= confident_move_min_confidence
        and bulge_ratio >= confident_move_min_bulge
    ):
        move_cap = confident_max_move_fraction
    if abs(moved) > move_cap * extent:
        # Clamping here would land the anchor somewhere that is neither the
        # fitter's answer nor the detector's -- strictly worse than both.
        return _refine_skip(proximal_name, anchor_name, "move-exceeds-guard", **common)
    if confidence < min_confidence:
        return _refine_skip(proximal_name, anchor_name, "low-confidence", **common)

    return DistalRefinement(
        proximal=proximal_name, anchor=anchor_name, applied=True, reason="refined", **common,
    )


def refine_distal_anchors(
    joints: np.ndarray,
    names: Sequence[str],
    mesh_vertices: np.ndarray | None,
    *,
    chains: Sequence[tuple[str, str, str, str]] = DEFAULT_DISTAL_CHAINS,
    protected: Collection[str] = (),
    skeleton_height: float | None = None,
) -> tuple[np.ndarray, list[DistalRefinement]]:
    """Run `refine_distal_anchor` over every limb and apply what it accepts.

    Returns a COPY of `joints` with each accepted anchor moved (every entry
    carrying that name, not just the first) and the per-limb reports. Joints
    named in `protected` are never moved -- that is how a rig-adjust rebind
    keeps a position a human placed by hand.

    Each limb is measured against the ORIGINAL joints, so the four passes are
    independent of each other and of their order; a mirror anchor is only
    ever read (to bound the region sideways), never written.
    """
    joints = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    refined = joints.copy()
    if skeleton_height is None and len(joints):
        skeleton_height = float(joints[:, VERTICAL_AXIS].max() - joints[:, VERTICAL_AXIS].min())
    reports: list[DistalRefinement] = []
    for proximal_name, anchor_name, mirror_anchor, preset in chains:
        if anchor_name in protected:
            reports.append(_refine_skip(proximal_name, anchor_name, "manual-override"))
            continue
        report = refine_distal_anchor(
            joints, names, mesh_vertices, (proximal_name, anchor_name),
            mirror_anchor=mirror_anchor,
            skeleton_height=skeleton_height,
            **_REFINE_PRESETS.get(preset, {}),
        )
        reports.append(report)
        if report.applied and report.wrist_to is not None:
            target = np.asarray(report.wrist_to, dtype=np.float64)
            for index, name in enumerate(names):
                if name == anchor_name:
                    refined[index] = target
    return refined, reports


def distal_refinement_json(report: DistalRefinement) -> dict:
    """`DistalRefinement` as a plain JSON-able dict, for the bind-validation
    report and the exported GLB's `extras`."""
    return {
        "proximal": report.proximal,
        "anchor": report.anchor,
        "applied": report.applied,
        "status": report.reason,
        "wrist_from": list(report.wrist_from) if report.wrist_from else None,
        "wrist_to": list(report.wrist_to) if report.wrist_to else None,
        "moved": round(float(report.moved), 6),
        "medial_offset": round(float(report.medial_offset), 6),
        "limb_extent": round(float(report.limb_extent), 6),
        "hand_extent": round(float(report.hand_extent), 6),
        "bulge_ratio": round(float(report.bulge_ratio), 4),
        "confidence": round(float(report.confidence), 4),
    }


class TemplateSkeletonBuild(NamedTuple):
    """Result of mapping a UniRig-predicted, semantically-named skeleton
    onto a skeleton family template (see `build_template_conformant_skeleton`).

    `names`/`parents`/`heads`/`tails`/`rolls` are aligned by index and
    describe only the bones that could actually be placed -- a template
    bone with no UniRig prediction and no synthesis rule is left out
    entirely (see `missing_required`), never emitted with a made-up
    position. `parents` holds *names* (template convention), not indices --
    same as `skeleton_template.conformance_report` expects; a caller that
    needs integer parent indices (e.g.
    `unirig_bind_worker.build_armature`) builds its own index space from
    `names` separately.
    """

    names: list[str]
    parents: list[str | None]
    heads: np.ndarray
    tails: np.ndarray
    rolls: list[float]
    missing_required: list[str]
    synthesized: list[str]
    skeleton_height: float
    # Optional-tier (finger/toe/head_tip) bones this build GREW rather than
    # took from a prediction, and the per-anchor length multiplier + its
    # provenance that grew them (see `optional_chain_scale`). Empty when
    # optional synthesis was skipped or nothing could be anchored.
    # Keyed by CHAIN ROOT (`index_1_l`, `thumb_1_r`, `toes_l`, ...), not by
    # anchor: each finger is measured against the mesh along its own axis.
    synthesized_optional: tuple[str, ...] = ()
    optional_chain_scales: tuple[tuple[str, float, str], ...] = ()


def build_template_conformant_skeleton(
    resolved_names: Sequence[str],
    joints: np.ndarray,
    template: Template,
    *,
    skip_optional_bones: bool = False,
    mesh_vertices: np.ndarray | None = None,
) -> TemplateSkeletonBuild:
    """Map a UniRig prediction's semantic role names onto `template`'s exact
    bone set, hierarchy, and rest orientation (PIPELINE_POLICY.md S3).

    `resolved_names[i]` is `joints[i]`'s role name from `semantic_joint_names`
    (or, for a rig-adjust rebind, the source rig's own bone name -- both use
    the same vocabulary the biped template's bone names already follow, so
    matching is a direct name lookup, no adapter layer). `joints` are
    world-space head positions in the family's house frame (the same frame
    `semantic_joint_names` operates in: Z-up, side axis X -- see
    `VERTICAL_AXIS`/`SIDE_AXIS`).

    For every template bone:
      - if `resolved_names` has a matching entry, its UniRig-predicted head
        position is used directly (the fitter's real, mesh-fitted geometry);
      - else if it's an OPTIONAL bone (a finger segment, a toe, head_tip --
        UniRig predicts none of them, ever) and `skip_optional_bones` is
        false, it is GROWN from its anchor: head at its parent's tail, with
        the chain's segment lengths scaled to this character's own measured
        hand/foot extent when `mesh_vertices` is given. See the
        "optional-tier synthesis" block above for why that is anchored
        locally rather than to the character's height. Passing
        `skip_optional_bones=True` restores the pre-mrmmn behaviour: an
        optional bone is emitted only when something predicted it.
      - else if it's one of the small set of required bones UniRig never
        predicts directly, its head is synthesized:
          - `root`: no anatomical UniRig counterpart exists (the donor's own
            root is a flat, horizontal reference/scale bone, not a spinal
            segment -- see assets/skeletons/biped/README.md's "Surprises"
            section -- so there is no verticality assumption to make here).
            Anchored at the predicted `pelvis` head.
          - `clavicle_l`/`clavicle_r`: no UniRig counterpart either;
            synthesized as the midpoint between the predicted `spine_3` and
            `upper_arm_{l,r}` heads.
      - else it is left unplaced; a *required* unplaced bone is recorded in
        `missing_required` rather than emitted with a guessed position (S3:
        "cannot fit" blames S2/S0, never a looser fit) -- and so is any
        bone (required or optional) whose *parent* could not be placed,
        since a bone's position is meaningless without its parent chain.

    Every placed bone's tail is `head + template_rest_direction *
    (template_relative_length * skeleton_height)`, where `skeleton_height`
    is this character's own predicted vertical extent (max - min along the
    vertical axis, over every raw UniRig-predicted joint) -- the same
    scale-independence the template itself uses (`relative_length` is a
    fraction of the donor's own height) -- with the one exception that a
    grown optional chain's segments (not the required anchor they hang off)
    additionally carry that chain's own measured scale, so a bone's tail
    always lands on its child's head. Roll is copied from the template
    unchanged ("template rolls are law", per PIPELINE_POLICY.md's blame
    table) -- this module makes no attempt to reconcile it against anything
    UniRig predicted, since UniRig does not predict roll at all.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if len(resolved_names) != len(joints):
        raise ValueError("resolved_names and joints must have the same length")

    by_name: dict[str, np.ndarray] = {}
    for name, position in zip(resolved_names, joints):
        if name not in by_name:
            by_name[name] = position

    if len(joints):
        vertical = joints[:, VERTICAL_AXIS]
        skeleton_height = float(vertical.max() - vertical.min())
    else:
        skeleton_height = 0.0

    placed: dict[str, np.ndarray] = {}
    synthesized: list[str] = []

    # `root` has to be resolved before the main per-template-bone loop
    # below (rather than inline, in template order) because it is
    # `pelvis`'s *parent* -- the loop gates every bone on its parent
    # already being placed, and `pelvis` is (virtually always) a direct
    # UniRig prediction, not itself something that waits on `root`.
    if "root" in by_name:
        placed["root"] = by_name["root"]
    elif "pelvis" in by_name:
        placed["root"] = by_name["pelvis"].copy()
        synthesized.append("root")

    for bone in template.bones:
        if bone.name == "root":
            continue
        if bone.parent is not None and bone.parent not in placed:
            continue
        if bone.name in by_name:
            placed[bone.name] = by_name[bone.name]
            continue
        if bone.name in ("clavicle_l", "clavicle_r"):
            side = bone.name.rsplit("_", 1)[-1]
            spine_3 = by_name.get("spine_3")
            upper_arm = by_name.get(f"upper_arm_{side}")
            if spine_3 is not None and upper_arm is not None:
                placed[bone.name] = (spine_3 + upper_arm) / 2.0
                synthesized.append(bone.name)

    # Second pass: grow the optional tier (fingers, toes, head_tip) from the
    # hands/feet/head the first pass placed. Separate from the loop above
    # because a chain's scale needs its anchor already placed, and run to a
    # fixpoint rather than in template order so a chain is grown correctly
    # however the template happens to list its bones.
    synthesized_optional: list[str] = []
    chain_scales: dict[str, tuple[float, str]] = {}
    if not skip_optional_bones and skeleton_height > 0.0:
        grew = True
        while grew:
            grew = False
            for bone in template.bones:
                if bone.tier != "optional" or bone.name in placed:
                    continue
                if bone.parent is None or bone.parent not in placed:
                    continue
                anchor = template_anchor_bone(template, bone.name)
                root = optional_chain_root(template, bone.name)
                if anchor is None or root is None or anchor not in placed:
                    continue
                parent_spec = template.bone(bone.parent)
                if parent_spec.name == anchor:
                    # The chain starts exactly at its anchor's tail: the
                    # anchor is a required bone whose own geometry this
                    # build does not touch, so no gap can open at the
                    # wrist/ankle however the chain is scaled.
                    segment_scale = 1.0
                else:
                    if root not in chain_scales:
                        chain_scales[root] = optional_chain_scale(
                            template, root, placed[anchor], skeleton_height, mesh_vertices,
                        )
                    segment_scale = chain_scales[root][0]
                placed[bone.name] = placed[bone.parent] + np.asarray(
                    parent_spec.rest_direction, dtype=np.float64
                ) * (parent_spec.relative_length * skeleton_height * segment_scale)
                synthesized.append(bone.name)
                synthesized_optional.append(bone.name)
                grew = True

    missing_required = [bone.name for bone in template.required_bones if bone.name not in placed]

    names_out: list[str] = []
    parents_out: list[str | None] = []
    heads_out: list[np.ndarray] = []
    tails_out: list[np.ndarray] = []
    rolls_out: list[float] = []
    for bone in template.bones:
        if bone.name not in placed:
            continue
        head = placed[bone.name]
        direction = np.asarray(bone.rest_direction, dtype=np.float64)
        # An optional bone in a scaled chain must carry that chain's scale in
        # its own length too, or its tail stops short of (or overshoots) the
        # child head placed above and the chain shows gaps between segments.
        # Required bones -- including a chain's anchor, whose tail is exactly
        # where the chain starts -- are never rescaled.
        length_scale = 1.0
        if bone.tier == "optional":
            root = optional_chain_root(template, bone.name)
            if root is not None and root in chain_scales:
                length_scale = chain_scales[root][0]
        tail = head + direction * (bone.relative_length * skeleton_height * length_scale)
        names_out.append(bone.name)
        parents_out.append(bone.parent)
        heads_out.append(head)
        tails_out.append(tail)
        rolls_out.append(bone.roll)

    return TemplateSkeletonBuild(
        names=names_out,
        parents=parents_out,
        heads=np.asarray(heads_out, dtype=np.float64).reshape(-1, 3),
        tails=np.asarray(tails_out, dtype=np.float64).reshape(-1, 3),
        rolls=rolls_out,
        missing_required=missing_required,
        synthesized=synthesized,
        skeleton_height=skeleton_height,
        synthesized_optional=tuple(synthesized_optional),
        optional_chain_scales=tuple(
            (anchor, scale, source) for anchor, (scale, source) in sorted(chain_scales.items())
        ),
    )


def template_side_conformance_findings(
    names: Sequence[str],
    heads: np.ndarray,
    template: Template,
    left: np.ndarray,
    reference_point: np.ndarray,
) -> list[dict]:
    """Catch a left/right-swapped rig that direction-only checking misses.

    `skeleton_template.conformance_report`'s own `mirrored_left_right`
    finding compares each bone's rest *direction* against its own template
    side vs. the mirrored side -- but the biped template's leg bones are
    side-identical in rest direction (thigh_l and thigh_r both point
    straight down; see PIPELINE_POLICY.md S3 / the N2 template build), so a
    leg-only mirror (both legs physically on the wrong side, everything
    else correct) produces no direction mismatch at all and passes that
    check silently.

    This checks absolute head *position* instead, for every `_l`/`_r`
    template bone: an `_l` bone must sit on the positive side of `left`
    (relative to `reference_point`, e.g. the pelvis), an `_r` bone the
    negative side. This is direction-independent, so it catches a leg-only
    mirror the same way it catches a full-body mirror (which the
    direction-based check already catches too, for the arm/clavicle bones,
    where left and right genuinely point different ways).

    `left` should only be trusted when it came from a confident facing
    measurement (`CONFIDENT_FACING_SOURCES`) -- callers with a
    low-confidence or assumed facing (e.g. a rig-adjust rebind, which has
    no measured facing at all) should skip calling this rather than pass an
    unmeasured `left` guess.
    """
    left_unit = _normalize(np.asarray(left, dtype=np.float64))
    if left_unit is None:
        return []
    reference_point = np.asarray(reference_point, dtype=np.float64)
    by_name = {name: index for index, name in enumerate(names)}
    findings: list[dict] = []
    for bone in template.bones:
        if not (bone.name.endswith("_l") or bone.name.endswith("_r")):
            continue
        index = by_name.get(bone.name)
        if index is None:
            continue
        side_value = float(np.dot(np.asarray(heads[index], dtype=np.float64) - reference_point, left_unit))
        expected_left = bone.name.endswith("_l")
        actual_left = side_value > 0.0
        if actual_left == expected_left:
            continue
        findings.append({
            "code": "mirrored_side_position",
            "severity": "error",
            "message": (
                f"bone {bone.name!r} sits on the "
                f"{'right' if expected_left else 'left'} side of the detected facing "
                "direction, not its own named side; left/right sides look swapped "
                "(position-based check -- catches a leg-only mirror that a "
                "rest-direction check cannot, since the template's leg bones are "
                "side-identical in direction)"
            ),
            "metric": "side_position_sign",
            "actual": round(side_value, 4),
            "expected": "> 0" if expected_left else "<= 0",
        })
    return findings
