"""Operator-selected mesh density for S2 canonicalization (bead i63ky).

Before this module the canonical clean had exactly one density: a 30k-triangle
decimation target hardcoded in `native.DEFAULT_CANONICALIZE_TARGET_FACES`,
with a raw voxel-search band (150k-350k) hand-tuned around it and a fixed
10-50k gate band that only happened to contain it. An operator who wanted a
denser mesh had no knob at all, and the two knobs that did exist
(`settings.target_vertices`, `settings.family_band`) moved independently --
so raising one without the other produced a candidate the gate then blocked
for being outside a band nobody had updated.

This module is the single place that turns an operator's INTENT ("Ultra") into
every number the rest of the pipeline needs to agree on:

  tier/custom target -> triangle target
                     -> gate band          (canonicalize_policy.evaluate_canonicalize)
                     -> raw voxel band     (voxel_remesh.voxel_raw_face_band)
                     -> bake atlas floor   (texture_bake.bake_texture_size)
                     -> instant-meshes vertex target (the legacy comparison path)

It is deliberately pure (no I/O, no Blender, no request objects) so the
resolution and band arithmetic is unit-testable, and so `main` can validate an
operator's settings and return a 400 using exactly the code that will later
run inside the job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QualityTierName = Literal["draft", "standard", "high", "ultra"]

# Triangle target per tier. `standard` is the historical hardcoded default
# (native.DEFAULT_CANONICALIZE_TARGET_FACES), so an operator who never touches
# the selector gets exactly the mesh they got before this module existed.
QUALITY_TIER_TARGETS: dict[str, int] = {
    "draft": 10_000,
    "standard": 30_000,
    "high": 80_000,
    "ultra": 150_000,
}
QUALITY_TIER_ORDER: tuple[str, ...] = ("draft", "standard", "high", "ultra")
DEFAULT_QUALITY_TIER = "standard"

# The gate band around a selected target, as a fraction of it. The S2 contract
# has always been "inside a band", never "exactly N triangles" -- voxel remesh
# plus decimation lands close to the target, not on it. 0.5x-1.5x is the same
# shape as the legacy fixed band (10-50k around a 30k target) expressed as a
# ratio, so `standard` reproduces a band the historical one contains.
QUALITY_BAND_LOW_RATIO = 0.5
QUALITY_BAND_HIGH_RATIO = 1.5

# A closed triangle mesh has roughly twice as many faces as vertices (Euler),
# which is the same relationship the old
# `DEFAULT_CANONICALIZE_TARGET_VERTICES = 15_000` comment recorded for its
# ~30k-triangle intent. Used only to map between an operator's vertex-flavoured
# custom number and the triangle counts the gate actually measures, and to feed
# the legacy Instant Meshes path (whose `--vertices` flag is a vertex target).
TRIANGLES_PER_VERTEX = 2.0

# Bounds on a custom numeric target. The floor is where a decimated humanoid
# stops having a face at all; the ceiling is where a bake atlas and a rig's
# weight solve stop being interactive on this box.
CUSTOM_TARGET_TRIANGLES_MINIMUM = 1_000
CUSTOM_TARGET_TRIANGLES_MAXIMUM = 500_000

# Bake-atlas FLOOR by density, as `(triangle target at or above which, floor)`,
# highest first. The atlas edge length is derived from the source's largest
# texture and capped at 4K by `texture_bake.bake_texture_size`; a tier never
# clamps it DOWN (a 4K-textured source still bakes 4K at Draft), it only raises
# the floor. Keyed off the resolved triangle target rather than the tier name,
# so a custom target gets the same treatment as the tier nearest it.
#
# The floor is not about the source's texel count -- it is about ISLAND COUNT.
# The bake atlas has to hold one packed UV island per smart-project face patch
# plus a 4-texel gutter around each, and a denser mesh produces proportionally
# more islands. Measured on the alien chain 2026-09-01 with a 2048 source
# texture: Standard (30k tris) packs to 41.8% atlas utilisation, Ultra (150k
# tris) to 9.4% on the same 2048 atlas -- and its detail ratio fell to 0.565,
# BELOW the 0.60 threshold, i.e. Ultra shipped a blurrier face than Standard
# while carrying five times the geometry. Raising Ultra's floor to 4096
# restores the texel budget the extra islands need.
ATLAS_MINIMUM_LADDER: tuple[tuple[int, int], ...] = (
    (QUALITY_TIER_TARGETS["ultra"], 4096),
    (QUALITY_TIER_TARGETS["high"], 2048),
    (0, 1024),
)
ATLAS_MINIMUM_BASE = 1024


@dataclass(frozen=True)
class QualityTarget:
    """The resolved density intent for one canonicalize run.

    `tier` is what the artifact will be labelled with -- one of the four tier
    names, or "custom" when a numeric target overrode the tier. `requested_tier`
    keeps the tier the operator had selected even when a custom number won, so
    provenance can say "custom 120,000 (tier ultra was selected)".
    """

    tier: str
    requested_tier: str | None
    target_triangles: int
    band: tuple[int, int]
    target_vertices: int
    atlas_minimum: int
    source: str

    def as_provenance(self) -> dict[str, Any]:
        """The report fields every consumer of a canonical artifact reads."""
        return {
            "quality_tier": self.tier,
            "quality_tier_requested": self.requested_tier,
            "quality_target_source": self.source,
            "target_triangles": self.target_triangles,
            "triangle_budget_band": [self.band[0], self.band[1]],
            "bake_atlas_minimum": self.atlas_minimum,
        }


def tier_triangle_target(tier: str) -> int:
    """Triangle target for a tier name, raising on an unknown one."""
    key = str(tier or "").strip().lower()
    if key not in QUALITY_TIER_TARGETS:
        raise ValueError(
            f"canonicalize settings.quality_tier must be one of {QUALITY_TIER_ORDER}, got {tier!r}"
        )
    return QUALITY_TIER_TARGETS[key]


def triangle_band(target_triangles: int) -> tuple[int, int]:
    """The gate's `[low, high]` triangle band around a resolved target.

    Tier and custom targets use the SAME ratio, so "the gate understands what
    I asked for" is one rule rather than a table with a hole in it.
    """
    if target_triangles <= 0:
        raise ValueError(
            f"triangle_band needs a positive triangle target, got {target_triangles}"
        )
    low = int(round(target_triangles * QUALITY_BAND_LOW_RATIO))
    high = int(round(target_triangles * QUALITY_BAND_HIGH_RATIO))
    return max(1, low), max(low + 1, high)


def atlas_minimum_for_target(target_triangles: int) -> int:
    """Bake-atlas floor for a density. Never a cap -- see ATLAS_MINIMUM_LADDER."""
    for threshold, floor in ATLAS_MINIMUM_LADDER:
        if target_triangles >= threshold:
            return floor
    return ATLAS_MINIMUM_BASE


def _coerce_positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canonicalize settings.{field} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"canonicalize settings.{field} must be positive, got {number}")
    return number


def _validate_custom_triangles(target_triangles: int, field: str) -> int:
    if not (
        CUSTOM_TARGET_TRIANGLES_MINIMUM
        <= target_triangles
        <= CUSTOM_TARGET_TRIANGLES_MAXIMUM
    ):
        raise ValueError(
            f"canonicalize settings.{field} resolves to {target_triangles} triangles, "
            f"outside the supported {CUSTOM_TARGET_TRIANGLES_MINIMUM}-"
            f"{CUSTOM_TARGET_TRIANGLES_MAXIMUM} range"
        )
    return target_triangles


def resolve_quality_target(
    settings: dict[str, Any] | None = None,
    *,
    request_target_vertices: int | None = None,
) -> QualityTarget:
    """Resolve one canonicalize job's density from its settings.

    Precedence, highest first:

    1. `settings.target_triangles` -- a custom target in the units the gate
       measures. This is the field the UI's custom box sends.
    2. `settings.target_vertices` (or the request's top-level `target_vertices`)
       -- the pre-existing knob, kept working. It is a VERTEX target, so it is
       converted at `TRIANGLES_PER_VERTEX`; a custom number of either flavour
       overrides a selected tier, per the operator's "custom overrides tier".
    3. `settings.quality_tier` -- one of draft/standard/high/ultra.
    4. `DEFAULT_QUALITY_TIER` -- standard, i.e. the historical 30k behaviour.

    An unknown tier, a non-integer target, or a target outside the supported
    range raises `ValueError` so the API answers 400 rather than starting a
    twenty-minute job that cannot pass its own gate.
    """
    job_settings = settings or {}

    requested_tier_raw = job_settings.get("quality_tier")
    requested_tier: str | None = None
    if requested_tier_raw is not None:
        requested_tier = str(requested_tier_raw).strip().lower()
        tier_triangle_target(requested_tier)  # validate eagerly, even if overridden

    custom_triangles_raw = job_settings.get("target_triangles")
    custom_vertices_raw = job_settings.get("target_vertices")
    if custom_vertices_raw is None and request_target_vertices is not None:
        custom_vertices_raw = request_target_vertices

    if custom_triangles_raw is not None:
        target_triangles = _validate_custom_triangles(
            _coerce_positive_int(custom_triangles_raw, "target_triangles"),
            "target_triangles",
        )
        tier = "custom"
        source = "custom-triangles"
    elif custom_vertices_raw is not None:
        target_vertices = _coerce_positive_int(custom_vertices_raw, "target_vertices")
        target_triangles = _validate_custom_triangles(
            int(round(target_vertices * TRIANGLES_PER_VERTEX)), "target_vertices"
        )
        tier = "custom"
        source = "custom-vertices"
    else:
        tier = requested_tier or DEFAULT_QUALITY_TIER
        target_triangles = tier_triangle_target(tier)
        source = "tier" if requested_tier else "default-tier"

    return QualityTarget(
        tier=tier,
        requested_tier=requested_tier,
        target_triangles=target_triangles,
        band=triangle_band(target_triangles),
        target_vertices=max(500, int(round(target_triangles / TRIANGLES_PER_VERTEX))),
        atlas_minimum=atlas_minimum_for_target(target_triangles),
        source=source,
    )
