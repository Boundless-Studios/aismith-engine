"""Stage implementations behind `aismith-engine run`.

Each stage calls the same native functions and gates as the app's job runner
(`backend/app/main.py`), so the engine and the app produce the same output for
the same inputs. Stages never download: the CLI runs them with
AISMITH_OFFLINE=1, and they check their weights against the lock first.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from backend.app import native
from backend.app.canonicalize_policy import decide_canonicalize
from backend.app.mesh_quality import analyze_path
from backend.app.quality_tiers import resolve_quality_target

from .contract import StageOutput, StageRejected, StageResult, StageSpec
from .inspect import check_stage_weights, identity

GLB = "model/gltf-binary"
VIEW_ROLES = ("front", "back", "left", "right")
MESH_RESOLUTIONS = (512, 768, 1024)
CLEAN_SETTINGS = {"quality_tier", "target_triangles", "skip_palette_transfer", "palette_transfer_strength", "project_views"}
IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}

Progress = Callable[[str], None]


def _require_image(name: str, path: Path) -> None:
    try:
        with Image.open(path) as image:
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise StageRejected(f"input {name!r} is not an image") from exc
    if image_format not in IMAGE_FORMATS:
        raise StageRejected(f"input {name!r} must be PNG, JPEG or WebP, not {image_format}")


GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942


def _require_glb(name: str, path: Path) -> None:
    """Reject a GLB that cannot be loaded before any native stage starts.

    Checks the binary glTF 2.0 container (header, declared length, a JSON
    first chunk, chunk bounds), then loads it and requires at least one
    triangle, so a truncated or corrupt upload is `rejected`, not a Blender
    `failed`.
    """
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise StageRejected(f"input {name!r} is not a binary glTF (GLB)")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise StageRejected(f"input {name!r} is glTF version {version}; only 2 is supported")
    if length != len(data):
        raise StageRejected(f"input {name!r} declares {length} bytes but has {len(data)}; it is truncated or padded")
    offset, chunks = 12, []
    while offset < length:
        if offset + 8 > length:
            raise StageRejected(f"input {name!r} has a truncated chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        if offset + 8 + chunk_length > length:
            raise StageRejected(f"input {name!r} has a chunk that runs past the end of the file")
        chunks.append((chunk_type, data[offset + 8: offset + 8 + chunk_length]))
        offset += 8 + chunk_length
    if not chunks or chunks[0][0] != GLB_JSON_CHUNK:
        raise StageRejected(f"input {name!r} does not start with a JSON chunk")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \x00"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StageRejected(f"input {name!r} has an unreadable JSON chunk") from exc
    if not isinstance(document, dict) or not document.get("meshes"):
        raise StageRejected(f"input {name!r} contains no meshes")
    try:
        measured = analyze_path(path)
    except Exception as exc:  # trimesh raises many types for corrupt buffers
        raise StageRejected(f"input {name!r} could not be loaded: {type(exc).__name__}") from exc
    if not measured.get("faces"):
        raise StageRejected(f"input {name!r} has no triangles")


def _int_setting(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageRejected(f"settings.{key} must be an integer")
    return value


def _weights_problem(stage: str) -> tuple[str | None, dict[str, Any] | None]:
    report = check_stage_weights(stage)
    if report is None:
        return None, None
    failing = [c for c in report.checks if c.status not in ("ok", "unpinned", "unverified")]
    if failing:
        return "weights: " + ", ".join(f"{c.dest} {c.status}" for c in failing), report.as_dict()
    return None, report.as_dict()


def _finish(result: StageResult, output: Path, started: float) -> StageResult:
    if not output.is_file() or output.stat().st_size < 64:
        result.status, result.error = "failed", "the stage did not produce a GLB"
        return result
    result.output = StageOutput.of(output, GLB)
    result.timings_s["total"] = round(time.monotonic() - started, 3)
    return result


def run_mesh(spec: StageSpec, workdir: Path, progress: Progress | None = None) -> StageResult:
    """Image (and optional back/left/right views) to a textured GLB with TRELLIS.2 FP8."""
    unknown = set(spec.inputs) - set(VIEW_ROLES)
    if "front" not in spec.inputs or unknown:
        raise StageRejected("mesh inputs are front (required) and optional back, left, right")
    for name, path in spec.inputs.items():
        _require_image(name, path)
    extra = set(spec.settings) - {"resolution", "seed"}
    if extra:
        raise StageRejected(f"unsupported mesh settings: {sorted(extra)}")
    resolution = _int_setting(spec.settings, "resolution", 768)
    if resolution not in MESH_RESOLUTIONS:
        raise StageRejected(f"settings.resolution must be one of {MESH_RESOLUTIONS}")
    seed = _int_setting(spec.settings, "seed", 1337)

    started = time.monotonic()
    result = StageResult(stage="mesh", status="ok", engine=identity())
    problem, result.weights = _weights_problem("mesh")
    if problem:
        result.status, result.error = "failed", problem
        return result
    output = workdir / "mesh.glb"
    views = dict(spec.inputs) if len(spec.inputs) > 1 else None
    metadata = native.run_trellis_fp8(spec.inputs["front"], output, resolution, progress, views, seed=seed)
    result.effective_inputs = {
        "resolution": metadata["resolution"],
        "seed": metadata["seed"],
        "views": sorted(spec.inputs),
        "model": metadata["model"],
    }
    result.diagnostics = {"metadata": metadata}
    result = _finish(result, output, started)
    if result.status == "ok":
        result.measured = analyze_path(output)
    return result


def run_clean(spec: StageSpec, workdir: Path, progress: Progress | None = None) -> StageResult:
    """S2 clean: voxel remesh, bake and floor-align one watertight body, then gate it."""
    views = {name: path for name, path in spec.inputs.items() if name in VIEW_ROLES}
    if "mesh" not in spec.inputs or set(spec.inputs) - {"mesh", *VIEW_ROLES}:
        raise StageRejected("clean inputs are mesh (required) and optional front, back, left, right views")
    _require_glb("mesh", spec.inputs["mesh"])
    for name, path in views.items():
        _require_image(name, path)
    extra = set(spec.settings) - CLEAN_SETTINGS
    if extra:
        raise StageRejected(f"unsupported clean settings: {sorted(extra)}")
    try:
        quality = resolve_quality_target(spec.settings)
    except ValueError as exc:
        raise StageRejected(str(exc)) from exc

    started = time.monotonic()
    result = StageResult(stage="clean", status="ok", engine=identity())
    output = workdir / "clean.glb"
    metadata = native.run_canonicalize(
        spec.inputs["mesh"],
        output,
        # Only the legacy Instant Meshes knob; the voxel path re-resolves the
        # same density from settings, as the app's job runner relies on.
        quality.target_vertices,
        native.DEFAULT_CANONICALIZE_VIEWS,
        native.DEFAULT_CANONICALIZE_RENDER_SIZE,
        progress,
        method="voxel",
        settings=dict(spec.settings),
        view_images=views if len(views) >= 2 else None,
    )
    result = _finish(result, output, started)
    if result.status != "ok":
        return result
    metadata["mesh_quality"] = result.measured = analyze_path(output)
    decision = decide_canonicalize(metadata)
    result.gate = {"name": "s2-canonicalize", "passed": decision["status"] == "ready", **decision}
    result.effective_inputs = {
        "method": "voxel",
        "quality_tier": quality.tier,
        "target_triangles": quality.target_triangles,
        "triangle_band": metadata["triangle_budget_band"],
        "views": sorted(views),
    }
    result.diagnostics = {"metadata": metadata}
    return result


RUNNERS = {"mesh": run_mesh, "clean": run_clean}
