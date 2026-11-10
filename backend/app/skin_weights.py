from __future__ import annotations

from collections.abc import Iterable, Mapping


def blend_projected_weights(
    neighbors: Iterable[tuple[float, Mapping[str, float]]],
    *,
    maximum_influences: int = 4,
) -> dict[str, float]:
    """Blend donor skin weights from nearby vertices into a normalized result."""
    accumulated: dict[str, float] = {}
    for distance, weights in neighbors:
        spatial_weight = 1.0 / max(float(distance) ** 2, 1e-12)
        for bone_name, bone_weight in weights.items():
            if bone_weight > 0:
                accumulated[bone_name] = (
                    accumulated.get(bone_name, 0.0)
                    + spatial_weight * float(bone_weight)
                )
    strongest = sorted(
        accumulated.items(),
        key=lambda item: (-item[1], item[0]),
    )[:maximum_influences]
    total = sum(weight for _, weight in strongest)
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in strongest}


def combine_weight_maps(
    weighted_maps: Iterable[tuple[float, Mapping[str, float]]],
    *,
    maximum_influences: int = 4,
) -> dict[str, float]:
    """Interpolate weight maps using explicit non-negative blend factors."""
    accumulated: dict[str, float] = {}
    for factor, weights in weighted_maps:
        if factor <= 0:
            continue
        for bone_name, bone_weight in weights.items():
            if bone_weight > 0:
                accumulated[bone_name] = (
                    accumulated.get(bone_name, 0.0)
                    + float(factor) * float(bone_weight)
                )
    strongest = sorted(
        accumulated.items(),
        key=lambda item: (-item[1], item[0]),
    )[:maximum_influences]
    total = sum(weight for _, weight in strongest)
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in strongest}


def smooth_weight_maps(
    weight_maps: list[Mapping[str, float]],
    edges: Iterable[tuple[int, int]],
    *,
    factor: float = 0.35,
    iterations: int = 2,
    maximum_influences: int = 4,
) -> list[dict[str, float]]:
    """Smooth weights only across target topology, never across nearby surfaces."""
    if not 0.0 <= factor <= 1.0:
        raise ValueError("Smoothing factor must be between zero and one")
    adjacency = [set() for _ in weight_maps]
    for first, second in edges:
        if not (0 <= first < len(weight_maps) and 0 <= second < len(weight_maps)):
            raise ValueError("Weight-smoothing edge references an unknown vertex")
        adjacency[first].add(second)
        adjacency[second].add(first)

    current = [dict(weights) for weights in weight_maps]
    for _ in range(max(0, iterations)):
        next_maps = []
        for index, own in enumerate(current):
            neighbors = adjacency[index]
            if not neighbors:
                next_maps.append(dict(own))
                continue
            neighborhood_influence_count = sum(
                len(current[neighbor]) for neighbor in neighbors
            )
            neighborhood = combine_weight_maps(
                ((1.0, current[neighbor]) for neighbor in sorted(neighbors)),
                maximum_influences=max(1, neighborhood_influence_count),
            )
            next_maps.append(combine_weight_maps(
                ((1.0 - factor, own), (factor, neighborhood)),
                maximum_influences=maximum_influences,
            ))
        current = next_maps
    return current
