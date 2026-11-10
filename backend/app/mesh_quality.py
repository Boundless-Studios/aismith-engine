from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _as_mesh(value: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    if not isinstance(value, trimesh.Scene):
        raise TypeError(f"Unsupported mesh value: {type(value).__name__}")

    mesh = value.to_mesh()
    if len(mesh.faces) == 0:
        raise ValueError("The asset does not contain mesh geometry")
    return mesh


def analyze_path(path: Path) -> dict[str, Any]:
    loaded = trimesh.load(path, force=None, process=False)
    return analyze_mesh(_as_mesh(loaded))


def _component_face_counts(faces: np.ndarray) -> list[int]:
    face_count = len(faces)
    parents = np.arange(face_count)
    sizes = np.ones(face_count, dtype=np.int64)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    edge_face_ids = np.tile(np.arange(face_count), 3)
    _, inverse = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_edges = inverse[order]
    sorted_faces = edge_face_ids[order]
    for position in range(1, len(order)):
        if sorted_edges[position] == sorted_edges[position - 1]:
            union(int(sorted_faces[position]), int(sorted_faces[position - 1]))

    counts: dict[int, int] = {}
    for face_index in range(face_count):
        root = find(face_index)
        counts[root] = counts.get(root, 0) + 1
    return list(counts.values())


def analyze_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("The asset does not contain a non-empty triangle mesh")

    # GLB exporters duplicate vertex indices at UV and normal seams. Those are
    # rendering boundaries, not physical gaps, so measure topology on a copy
    # welded by position while retaining original vertex/face counts.
    topology_mesh = mesh.copy()
    topology_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    topology_faces = np.asarray(topology_mesh.faces)
    edge_counts = np.unique(
        np.sort(
            np.concatenate(
                (
                    topology_faces[:, [0, 1]],
                    topology_faces[:, [1, 2]],
                    topology_faces[:, [2, 0]],
                )
            ),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )[1]
    component_face_counts = _component_face_counts(topology_faces)
    largest_component_face_ratio = (
        max(component_face_counts) / len(faces) if component_face_counts else 0.0
    )
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    bounds = np.asarray(mesh.bounds)
    extents = np.asarray(mesh.extents)

    return {
        "schema_version": 1,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "components": len(component_face_counts),
        "largest_component_face_ratio": round(float(largest_component_face_ratio), 6),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "degenerate_faces": int(np.count_nonzero(doubled_areas <= 1e-12)),
        "watertight": bool(topology_mesh.is_watertight),
        "winding_consistent": bool(topology_mesh.is_winding_consistent),
        "finite": bool(np.isfinite(vertices).all()),
        "bounds": {
            "minimum": bounds[0].tolist(),
            "maximum": bounds[1].tolist(),
            "extents": extents.tolist(),
            "diagonal": float(np.linalg.norm(extents)),
        },
    }
