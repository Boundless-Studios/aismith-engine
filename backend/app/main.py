from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import re
import requests
import shutil
import struct
import subprocess
import sys
import time
import uuid
import urllib.request
import venv
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from weakref import WeakValueDictionary

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator

from . import native
from ..engine import sources as engine_sources
from ..engine import weights as engine_weights
from .canonicalize_policy import decide_canonicalize
from .clip_store import find_clip, load_manifest
from .projection_policy import evaluate_projection
from .source_fidelity_policy import evaluate_source_fidelity
from .curated_motion import curated_clips
from .mesh_quality import analyze_path
from . import pipeline_policy
from .quality_policy import (
    assess_mesh_promotion,
    evaluate_deformation_review,
    evaluate_for_animation,
    evaluate_for_rigging,
    evaluate_rig_certificate,
    is_rig_producing_stage,
    rig_candidate_is_promotable,
)
from .proxy_rig import deformation_surface_can_promote, select_deformation_surface
from .quality_tiers import (
    CUSTOM_TARGET_TRIANGLES_MAXIMUM,
    TRIANGLES_PER_VERTEX,
    resolve_quality_target,
)
from .image_preparation import (
    VIEW_SYNTHESIS_PROFILE_FAMILIES,
    build_view_synthesis_metadata,
    resolve_preparation_profile,
    resolve_view_synthesis_profile,
    run_comfyui_image_edit,
)
from .joint_recipe import JointRecipe
from .rig_quality import analyze_rig_path
from .rig_quality import _extract_template_conformance, _read_glb
from .run_store import (
    PREPARED_FRONT_STAGES,
    ReviewState,
    RunStore,
    SubjectKind,
    select_view_synthesis_source,
    summarize_run,
)
from .view_consistency import (
    blocking_view_conflicts,
    compare_view_to_front,
    job_warnings as view_consistency_warnings,
    measure_silhouette,
)


ROOT = Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    configured = os.getenv("AISMITH_DATA_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else ROOT / "studio-data"


DATA = _data_root()
UPLOADS = DATA / "uploads"
ARTIFACTS = DATA / "artifacts"
JOBS_DIR = DATA / "jobs"
DOWNLOADS_DIR = DATA / "downloads"
RUNS_DIR = DATA / "runs"
SETTINGS_PATH = DATA / "settings.json"
for directory in (UPLOADS, ARTIFACTS, JOBS_DIR, DOWNLOADS_DIR, RUNS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

run_store = RunStore(RUNS_DIR)

ALLOWED_UPLOADS = {"png", "jpg", "jpeg", "webp", "glb", "gltf", "fbx", "obj", "bvh"}
# Generated GLBs commonly exceed 20 MB once they carry PBR textures. Keep a
# bounded local-import limit, but allow normal studio assets to round-trip.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
BATCH_ONE_ATTACK = "Sword Attack"
JobRequestKind = Literal["prepare-image", "synthesize-views", "create-tpose-set", "generate", "retopologize", "retopo", "canonicalize", "paint-bake", "project-views", "rebuild-views", "repair-weights", "auto-rig", "rig-adjust", "rig-transfer", "target-proxy-rig", "generate-motion", "retarget-motion", "apply-clip"]
AUTOMATED_QUALITY_ACTIONS: dict[str, JobRequestKind] = {
    "retry-generation": "generate",
    "refine-reconstruct": "retopologize",
    "repair-topology": "retopo",
    "retopologize": "retopo",
    "repair-weights": "repair-weights",
}
MANUAL_QUALITY_ACTIONS = {
    "manual-blender-cleanup",
    "inspect-boundaries",
    "inspect-components",
    "inspect-weight-paint",
    "rerig",
    "attach-animation",
    "reattach-animation",
}


SOURCE_VIEW_LABELS = {"front", "back", "left", "right"}
# Auto-generate-views (E3): the labels the prepare stage can synthesize from
# one approved front image via the ComfyUI Flux.2 Klein path.
SYNTHESIZED_VIEW_LABELS = {"back", "left", "right"}
DEFAULT_SYNTHESIZED_VIEWS = ("back", "left", "right")

# Bead 4vquq -- the canonical T-pose set.
#
# `create-tpose-set` collapses what used to be two operator steps (Prepare,
# then Auto-generate views) into one composite job: it draws the T-pose FRONT
# from the raw reference, gates that front, then fans the same three
# synthesized views off the T-pose front it just made. The four artifacts are
# registered as one SET, and from then on the raw upload is reference-only --
# it never conditions a biped generate again.
#
# Both `prepare-image` and `create-tpose-set` produce an approvable T-pose
# front, so every "has this run produced a front the operator can approve /
# must generate from?" question reads `run_store.PREPARED_FRONT_STAGES`
# (imported above) rather than one literal stage name.
#
# The T-pose front's own role inside a set, and the three roles fanned off it.
TPOSE_SET_FRONT_ROLE = "front"
TPOSE_SET_VIEW_ROLES = ("back", "left", "right")
TPOSE_SET_ROLES = (TPOSE_SET_FRONT_ROLE, *TPOSE_SET_VIEW_ROLES)
# The composite is a biped affordance: an object has no T-pose and a
# chain-profile creature has no shoulders to hold one.
TPOSE_SET_SUBJECT_KINDS = {"auto", "biped"}


class JobRequest(BaseModel):
    kind: JobRequestKind
    submission_id: str = Field(min_length=1, max_length=120)
    source_artifact_id: str | None = None
    # Multi-view conditioning for Generate: keys are restricted to
    # front/back/left/right image artifact ids, "front" is mandatory, and
    # when source_artifact_id is also set it must agree with the front view.
    source_views: dict[str, str] | None = None
    # 768 is AISmith 3D's balanced export preset: it runs the native 1024
    # cascade, then creates a viewport-friendly PBR asset.
    resolution: Literal[512, 768, 1024] = 768
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    target_faces: int = Field(default=10000, ge=500, le=200000)
    target_vertices: int | None = Field(default=None, ge=500, le=1_000_000)
    quality: Literal["game", "balanced", "high"] = "game"
    settings: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    run_name: str | None = Field(default=None, max_length=120)
    retry_of_attempt_id: str | None = None
    # Only consulted when this job creates a new run (no run_id supplied);
    # ignored when appending to an existing run, which already carries its
    # own subject_kind. Defaults to "auto" in RunStore.create when unset.
    subject_kind: SubjectKind | None = None

    @field_validator("source_views")
    @classmethod
    def _validate_source_views(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return value
        unknown = sorted(set(value) - SOURCE_VIEW_LABELS)
        if unknown:
            raise ValueError(f"Unknown source_views keys: {unknown}")
        if "front" not in value:
            raise ValueError("source_views requires a 'front' image")
        duplicate_ids = sorted(
            {
                artifact_id
                for artifact_id in value.values()
                if list(value.values()).count(artifact_id) > 1
            }
        )
        if duplicate_ids:
            raise ValueError(
                f"source_views must reference a distinct artifact per view; "
                f"duplicated ids: {duplicate_ids}"
            )
        return value

    @model_validator(mode="after")
    def _validate_source_views_agree_with_source_artifact_id(self) -> "JobRequest":
        if (
            self.source_views is not None
            and self.source_artifact_id is not None
            and self.source_artifact_id != self.source_views["front"]
        ):
            raise ValueError(
                "source_artifact_id must match source_views['front'] when both are provided"
            )
        return self


class JobRetryRequest(BaseModel):
    submission_id: str = Field(min_length=1, max_length=120)
    resolution: Literal[512, 768, 1024] | None = None
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    target_faces: int | None = Field(default=None, ge=500, le=200000)
    target_vertices: int | None = Field(default=None, ge=500, le=1_000_000)
    quality: Literal["game", "balanced", "high"] | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class PipelineStepRequest(BaseModel):
    """Body for POST /api/runs/{run_id}/pipeline/step (bead hlxij).

    Every field is optional: a bare `{}` is the normal call, and means
    "evaluate this run and take exactly one step". `clip` answers the one
    question the driver refuses to answer for itself (see the endpoint's
    `needs_input` branch); `attempt_cap` lets a caller tighten -- or, for a
    deliberate third try, loosen -- the per-stage remediation bound.
    """

    submission_id: str | None = Field(default=None, min_length=1, max_length=120)
    clip: str | None = Field(default=None, min_length=1, max_length=200)
    family: str | None = Field(default=None, min_length=1, max_length=80)
    attempt_cap: int = Field(
        default=pipeline_policy.DEFAULT_STAGE_ATTEMPT_CAP, ge=1, le=10
    )


class RunCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_artifact_id: str
    subject_kind: SubjectKind = "auto"


class RunReviewRequest(BaseModel):
    """Body for PATCH /api/runs/{run_id}.

    Exactly one of two shapes: the final-review triple (review_status,
    review_note, artifact_id), or a bare subject_kind update. They are
    mutually exclusive so a single PATCH endpoint can serve both without
    ambiguity about which action was intended.
    """

    review_status: ReviewState | None = None
    review_note: str | None = Field(default=None, min_length=1, max_length=2000)
    artifact_id: str | None = Field(default=None, min_length=1)
    subject_kind: SubjectKind | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "RunReviewRequest":
        review_fields = (self.review_status, self.review_note, self.artifact_id)
        review_fields_present = any(value is not None for value in review_fields)
        if self.subject_kind is not None and review_fields_present:
            raise ValueError("Provide either review fields or subject_kind, not both")
        if review_fields_present and not all(value is not None for value in review_fields):
            raise ValueError("review_status, review_note, and artifact_id are required together")
        if self.subject_kind is None and not review_fields_present:
            raise ValueError("Provide either review fields or subject_kind")
        return self


class CheckpointHandoffRequest(BaseModel):
    note: str = Field(min_length=1, max_length=2000)
    artifact_id: str = Field(min_length=1)


class ArtifactPromotionRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class TposeSetMemberRequest(BaseModel):
    """Body for replacing one member of a run's canonical T-pose set."""

    artifact_id: str = Field(min_length=1, max_length=120)
    reason: str = Field(default="", max_length=2000)


class MeshDiagnosticsRequest(BaseModel):
    run_id: str | None = None


class AttemptCreateRequest(BaseModel):
    stage: str = Field(min_length=1, max_length=80)
    input_artifact_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


class AttemptFinishRequest(BaseModel):
    status: Literal["completed", "failed", "cancelled"]
    output_artifact_id: str | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArtifactStageRequest(BaseModel):
    submission_id: str = Field(min_length=1, max_length=120)
    stage: str = Field(min_length=1, max_length=80)
    input_artifact_id: str
    output_artifact_id: str
    run_id: str | None = None
    run_name: str | None = Field(default=None, max_length=120)
    settings: dict[str, Any] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    id: str
    name: str
    kind: str
    path: str
    size_bytes: int
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class JobState:
    id: str
    kind: str
    submission_id: str = ""
    source_artifact_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    retry_of_attempt_id: str | None = None
    promoted_output: bool = True
    request_payload: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_artifact_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    # Advisory PIPELINE_POLICY.md evaluation of the run this job extended
    # (bead x8qhy). Never gates promotion; see `_evaluate_job_policy`.
    pipeline_policy: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DownloadState:
    id: str
    model: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    file_name: str | None = None
    file_index: int = 0
    file_count: int = 0
    file_bytes: int = 0
    file_total_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned[:100] or "asset"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile_interrupted_jobs(jobs_dir: Path, store: RunStore) -> dict[str, JobState]:
    recovered: dict[str, JobState] = {}
    error = "Interrupted by backend restart"
    for path in jobs_dir.glob("*.json"):
        try:
            job = JobState(**_read_json(path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            try:
                path.replace(path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex}"))
            except OSError:
                pass
            continue
        recovered[job.id] = job
        attempt = None
        if job.run_id and job.attempt_id:
            try:
                run = store.get(job.run_id)
                attempt = next(item for item in run.attempts if item.id == job.attempt_id)
            except (KeyError, StopIteration):
                job.warnings.append("The associated run attempt could not be reconciled")
        if attempt is not None and attempt.status not in {"queued", "running"} and job.status in {"queued", "running"}:
            job.status = attempt.status
            job.output_artifact_id = attempt.output_artifact_id
            job.error = attempt.error
            job.logs = attempt.logs
            job.warnings = attempt.warnings
            job.promoted_output = attempt.promoted
        elif attempt is not None and attempt.status in {"queued", "running"} and job.status not in {"queued", "running"}:
            store.finish_attempt(
                job.run_id,
                job.attempt_id,
                job.status,
                output_artifact_id=job.output_artifact_id,
                error=job.error,
                logs=job.logs,
                warnings=job.warnings,
                promote_output=job.promoted_output,
            )
        elif job.status in {"queued", "running"}:
            job.status = "failed"
            job.error = error
            job.logs.append(error)
            if attempt is not None:
                store.finish_attempt(
                    job.run_id,
                    job.attempt_id,
                    "failed",
                    error=error,
                    logs=job.logs,
                    warnings=job.warnings,
                )
        _write_json(path, job.as_dict())

    for run in store.list():
        for attempt in run.attempts:
            if attempt.job_id is None or attempt.status not in {"queued", "running"}:
                continue
            store.finish_attempt(
                run.id,
                attempt.id,
                "failed",
                error=error,
                logs=[*attempt.logs, error],
                warnings=attempt.warnings,
            )
    return recovered


def output_directory() -> Path:
    """The user-selected export directory, defaulting to AISmith 3D's artifact cache."""
    if SETTINGS_PATH.exists():
        try:
            selected = str(_read_json(SETTINGS_PATH).get("output_directory", "")).strip()
            if selected:
                path = Path(selected).expanduser().resolve()
                if path.exists() and path.is_dir():
                    return path
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return ARTIFACTS.resolve()


def set_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("Selected output directory does not exist")
    _write_json(SETTINGS_PATH, {"output_directory": str(resolved)})
    return resolved


def _artifact_path(artifact_id: str) -> Path:
    metadata = ARTIFACTS / f"{artifact_id}.json"
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact = _read_json(metadata)
    path = Path(artifact["path"]).resolve()
    allowed_roots = {ARTIFACTS.resolve(), output_directory().resolve()}
    if not any(path.parent == root or root in path.parents for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    return path


def save_artifact(source: Path, *, name: str, kind: str, metadata: dict[str, Any] | None = None) -> Artifact:
    artifact_id = uuid.uuid4().hex
    suffix = source.suffix.lower() or ".bin"
    payload_suffix = f".payload{suffix}" if suffix == ".json" else suffix
    destination = output_directory() / f"{artifact_id}{payload_suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    artifact = Artifact(
        id=artifact_id,
        name=name,
        kind=kind,
        path=str(destination),
        size_bytes=destination.stat().st_size,
        created_at=time.time(),
        metadata=metadata or {},
    )
    _write_json(ARTIFACTS / f"{artifact_id}.json", artifact.model_dump())
    return artifact


SOURCE_FIDELITY_STAGES = {"generate", "canonicalize", "project-views", "rebuild-views"}


def _stamp_source_fidelity(
    job: JobState,
    source_artifact_id: str,
    output: Path,
    metadata: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    """Advisory-only: score `output` against the run's ROOT source photo and
    stamp the report + its verdict onto `metadata`. Never raises, never
    touches `job.promoted_output` -- a drifted mesh is still a complete,
    shippable asset."""
    if settings.get("skip_source_fidelity"):
        metadata["source_fidelity"] = {"skipped": "disabled by request settings"}
    else:
        photo_path, photo_id, reason = native.resolve_source_photo(source_artifact_id)
        if photo_path is None:
            metadata["source_fidelity"] = {"skipped": reason}
        else:
            _job_log(job, "Measuring source fidelity", 97)
            try:
                report = native.measure_source_fidelity(
                    output, photo_path, ARTIFACTS / f"job-{job.id}-source-fidelity",
                    fov_x_degrees=settings.get("source_fov_x_degrees"),
                    yaw_degrees=float(settings.get("source_yaw_degrees") or 0.0),
                    pitch_degrees=float(settings.get("source_pitch_degrees") or 0.0),
                    samples=int(settings.get("source_fidelity_samples") or 32),
                )
            except Exception as exc:
                metadata["source_fidelity"] = {"skipped": f"measurement failed: {exc}"}
                job.warnings.append(f"Source fidelity measurement failed: {exc}")
                _job_log(job, f"Source fidelity measurement failed: {exc}")
            else:
                side = save_artifact(
                    Path(report["side_by_side"]), name=f"Source fidelity {job.id}.png", kind="image",
                    metadata={
                        "stage": "source-fidelity", "for_job_id": job.id,
                        "photo_artifact_id": photo_id, "source_artifact_id": source_artifact_id,
                    },
                )
                report["side_by_side_artifact_id"] = side.id
                report["photo_artifact_id"] = photo_id
                metadata["source_fidelity"] = report

    decision = evaluate_source_fidelity(metadata["source_fidelity"])
    metadata.setdefault("quality_decisions", {})["source_fidelity"] = decision
    if decision["status"] == "needs_review":
        job.warnings.append(
            f"Source fidelity needs review ({len(decision['findings'])} finding(s))"
        )


def persist_deformation_review_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    review_files = report.pop("_review_files", [])
    references = []
    try:
        for item in review_files:
            path = Path(str(item["path"]))
            artifact = save_artifact(
                path,
                name=f"Deformation review - {item['label']}.png",
                kind="image",
                metadata={
                    "stage": "deformation-review",
                    "label": item["label"],
                    "frame_ratio": item["ratio"],
                    "rest_pose": item["rest_pose"],
                    "focus_bone": item.get("focus_bone"),
                },
            )
            references.append({
                "artifact_id": artifact.id,
                "label": item["label"],
                "frame_ratio": item["ratio"],
                "rest_pose": item["rest_pose"],
                "focus_bone": item.get("focus_bone"),
            })
    finally:
        for item in review_files:
            Path(str(item["path"])).unlink(missing_ok=True)
    report["review_artifacts"] = references
    return report


def persist_skeleton_review_artifacts(report: dict[str, Any]) -> dict[str, Any]:
    review_files = report.pop("_review_files", [])
    references = []
    try:
        for item in review_files:
            path = Path(str(item["path"]))
            artifact = save_artifact(
                path,
                name=f"Skeleton review - {item['label']}.png",
                kind="image",
                metadata={"stage": "skeleton-review", "label": item["label"]},
            )
            references.append({"artifact_id": artifact.id, "label": item["label"]})
    finally:
        for item in review_files:
            Path(str(item["path"])).unlink(missing_ok=True)
    report["review_artifacts"] = references
    return report


def read_artifact(artifact_id: str) -> Artifact:
    metadata = ARTIFACTS / f"{artifact_id}.json"
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Artifact.model_validate(_read_json(metadata))


def _attach_rig_diagnostics(artifact: Artifact) -> Artifact:
    path = _artifact_path(artifact.id)
    if path.suffix.lower() != ".glb":
        raise ValueError("Rig diagnostics currently require a binary GLB asset")
    if "mesh_quality" not in artifact.metadata:
        artifact.metadata["mesh_quality"] = analyze_path(path)
    artifact.metadata["rig_quality"] = analyze_rig_path(path)
    decisions = dict(artifact.metadata.get("quality_decisions", {}))
    decisions.update({
        "rig": evaluate_for_rigging(artifact.metadata["mesh_quality"]),
        "rig-certificate": evaluate_rig_certificate(artifact.metadata["rig_quality"]),
        "animation-review": evaluate_for_animation(
            artifact.metadata["rig_quality"],
            required_animation=BATCH_ONE_ATTACK,
        ),
    })
    artifact.metadata["quality_decisions"] = decisions
    if (
        artifact.metadata.get("stage") in {"auto-rig", "rig-adjust"}
        and "skeleton_review" not in artifact.metadata
    ):
        # Only these two stages produce a fresh skeleton -- a contact sheet
        # is the visual check a naming-only symmetry check cannot give a
        # human, so it is attached at the same point the numeric checks are.
        # Rendered once: re-diagnosing an artifact that already carries its
        # sheet must not re-render and orphan the previous sheet artifacts.
        # Advisory, like the job path: a sheet that cannot render (no
        # Blender on this box, say) loses the reviewer the picture, not the
        # numeric diagnostics this call exists for.
        try:
            sheet_report = native.run_skeleton_sheet(path, lambda _message: None)
            artifact.metadata["skeleton_review"] = persist_skeleton_review_artifacts(sheet_report)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            artifact.metadata["skeleton_review_error"] = (
                f"Skeleton contact sheet could not be rendered: {exc}"
            )
    _write_json(ARTIFACTS / f"{artifact.id}.json", artifact.model_dump())
    return artifact


def _attach_rig_intake_diagnostics(artifact: Artifact) -> Artifact:
    artifact = _attach_rig_diagnostics(artifact)
    path = _artifact_path(artifact.id)
    with TemporaryDirectory(prefix="aismith-rig-intake-") as temporary:
        deformation_path = Path(temporary) / "deformation.json"
        deformation_report = native.run_deformation_diagnostics(
            path,
            deformation_path,
            lambda _message: None,
        )
        deformation_report = persist_deformation_review_artifacts(deformation_report)
    artifact.metadata["deformation_review"] = deformation_report
    artifact.metadata["quality_decisions"]["deformation-review"] = evaluate_deformation_review(
        deformation_report
    )
    _write_json(ARTIFACTS / f"{artifact.id}.json", artifact.model_dump())
    return artifact


async def _attach_serialized_rig_intake_diagnostics(artifact: Artifact) -> Artifact:
    await model_manager.acquire("blender-rig-intake")
    try:
        return await asyncio.to_thread(_attach_rig_intake_diagnostics, artifact)
    finally:
        await model_manager.release()


def _nvidia_smi_command() -> str:
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return discovered
    wsl_command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if "microsoft" in platform.release().lower() and wsl_command.exists():
        return str(wsl_command)
    return "nvidia-smi"


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "name": "CPU / undetected", "vram_total_mb": 0, "vram_used_mb": 0, "vram_free_mb": 0, "busy_external": False, "cuda": False, "torch_cuda_available": False, "compute_capability": None}
    try:
        result = subprocess.run([_nvidia_smi_command(), "--query-gpu=name,memory.total,memory.used,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            name, total, used = fields[:3]
            compute = fields[3] if len(fields) > 3 else None
            info.update({"available": True, "name": name, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used)), "cuda": True, "compute_capability": compute})
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update({"available": True, "name": properties.name, "vram_total_mb": round(properties.total_memory / 1024 / 1024), "cuda": True, "torch_cuda_available": True, "compute_capability": f"{properties.major}.{properties.minor}"})
    except Exception:
        pass
    info["model_variant"] = "C++ Generate + TRELLIS.2 FP8 Refine"
    info["recommended_resolution"] = 512 if info["vram_total_mb"] <= 8192 else 1024
    info["native_inference_ready"] = bool(info["available"] and info["torch_cuda_available"] and info["compute_capability"])
    # Ground-truth free/busy from the same nvidia-smi snapshot the GPU
    # preflight uses, so the UI's picture of contention matches what the
    # native workers actually see when they decide whether to wait.
    snapshot = native.gpu_memory_snapshot()
    if snapshot.available:
        info["vram_free_mb"] = snapshot.free_mb
        info["busy_external"] = snapshot.busy_external
    else:
        info["vram_free_mb"] = max(0, info["vram_total_mb"] - info["vram_used_mb"])
    return info


def _ram_info() -> dict[str, int]:
    """System RAM snapshot for the local-only status bar."""
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total_mb = round(status.ullTotalPhys / 1024 / 1024)
            available_mb = round(status.ullAvailPhys / 1024 / 1024)
            return {"total_mb": total_mb, "used_mb": max(0, total_mb - available_mb)}
    except Exception:
        pass
    if platform.system() == "Linux":
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value, *_ = line.split()
                values[key.rstrip(":")] = int(value)
            total_mb = round(values["MemTotal"] / 1024)
            available_mb = round(values["MemAvailable"] / 1024)
            return {"total_mb": total_mb, "used_mb": max(0, total_mb - available_mb)}
        except (OSError, KeyError, ValueError):
            pass
    return {"total_mb": 0, "used_mb": 0}


class ModelManager:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.loaded: str | None = None
        self.loaded_at: float | None = None
        self.active_job_id: str | None = None

    async def acquire(self, model_name: str, job_id: str | None = None) -> None:
        await self.lock.acquire()
        self.loaded = model_name
        self.loaded_at = time.time()
        self.active_job_id = job_id

    async def release(self) -> None:
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            self.loaded = None
            self.loaded_at = None
            self.active_job_id = None
        finally:
            if self.lock.locked():
                self.lock.release()


def _cors_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    configured = [
        origin.strip()
        for origin in os.getenv("AISMITH_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys([*defaults, *configured]))


app = FastAPI(title="Local 3D Studio API", version="0.1.0")
app.add_middleware(GZipMiddleware, minimum_size=1_000)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"], allow_headers=["*"])
model_manager = ModelManager()
jobs: dict[str, JobState] = {}
downloads: dict[str, DownloadState] = {}
record_submission_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
attempt_finish_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


@app.on_event("startup")
async def reconcile_state_on_startup() -> None:
    jobs.clear()
    jobs.update(reconcile_interrupted_jobs(JOBS_DIR, run_store))


def _model_readiness() -> dict[str, bool]:
    return {
        # The trellis2cpp GGUF fallback's required geometry models are on disk.
        "trellis_gguf_ready": native.trellis_gguf_ready(),
        "instant_meshes_ready": native.instant_meshes_exe() is not None,
        "trellis_refiner_ready": native.trellis_refiner_ready(),
        "kimodo_ready": native.kimodo_executable() is not None,
        "unirig_ready": native.unirig_ready(),
        "rebuild_views_ready": native.rebuild_views_available(),
    }


def _conditioning_view_consistency(
    front: Artifact,
    view_artifacts: dict[str, Artifact],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Check every conditioning view against the front image a generate will
    ACTUALLY be conditioned on (bead 8c9gf).

    `synthesize-views` already checks each view as it draws it, but that only
    covers views this backend just synthesized against the front it happened
    to synthesize them from. A `generate` submission can carry an
    operator-uploaded view, a view synthesized from a DIFFERENT front (the
    lineage-advance bug `select_view_synthesis_source` fixes), or a stale set
    left in the slots from an earlier reference image. The pairing that
    matters to TRELLIS.2 is the one it is handed, so it is re-measured here.

    Returns `(results by view label, labels whose image could not be read)`.
    An unreadable image is reported, never fatal: a gate that turns a missing
    file into a blocked generate would be a worse failure than the one it
    guards. Runs synchronously on purpose -- the caller holds a blocking
    `FileLock`, and awaiting under one would let a second request block the
    event loop while this one waits to resume.
    """
    results: dict[str, dict[str, Any]] = {}
    unreadable: list[str] = []
    try:
        front_bytes = Path(front.path).read_bytes()
    except OSError:
        return results, sorted(label for label in view_artifacts if label != "front")
    for label in sorted(view_artifacts):
        if label == "front":
            continue
        try:
            view_bytes = Path(view_artifacts[label].path).read_bytes()
        except OSError:
            unreadable.append(label)
            continue
        results[label] = compare_view_to_front(front_bytes, view_bytes, view=label)
    return results, unreadable


# ---------------------------------------------------------------------------
# Canonical T-pose set (bead 4vquq)
# ---------------------------------------------------------------------------


def _stamp_artifact_metadata(artifact_id: str, updates: dict[str, Any]) -> None:
    """Merge `updates` into an already-saved artifact's metadata sidecar.

    The set's `members` map cannot be known when the first member is saved --
    the front exists before the views it will be grouped with do. Rather than
    inventing a second registry that could disagree with the artifacts, each
    member is stamped with the completed map once the whole set is known, the
    same way `_attach_rig_diagnostics` re-writes a sidecar after the fact.
    """
    path = ARTIFACTS / f"{artifact_id}.json"
    payload = _read_json(path)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(updates)
    payload["metadata"] = metadata
    _write_json(path, payload)


def tpose_set_membership(artifact: Artifact) -> dict[str, Any] | None:
    """The `tpose_set` block an artifact carries, or None when it is not a
    member of one. Never raises on a malformed block: a hand-edited or
    partially-written sidecar reads as "not a set member" rather than
    breaking every generate on the run."""
    membership = artifact.metadata.get("tpose_set")
    if not isinstance(membership, dict):
        return None
    if not isinstance(membership.get("set_id"), str) or not membership["set_id"]:
        return None
    return membership


def tpose_set_conditioning_views(front: Artifact) -> dict[str, str]:
    """The `source_views` payload a generate conditioned on this set must use.

    Reads the SET FRONT's own membership block. Only roles whose artifact id
    is a non-empty string are included, so a set whose left view failed
    (`create-tpose-set` completes partially by design) conditions on the three
    members it actually has instead of refusing to generate at all.
    """
    membership = tpose_set_membership(front)
    if membership is None:
        return {}
    members = membership.get("members")
    if not isinstance(members, dict):
        return {}
    views: dict[str, str] = {TPOSE_SET_FRONT_ROLE: front.id}
    for role in TPOSE_SET_VIEW_ROLES:
        member_id = members.get(role)
        if isinstance(member_id, str) and member_id and member_id != front.id:
            views[role] = member_id
    return views


def latest_tpose_set_front(run: Any) -> str | None:
    """The artifact id of the `create-tpose-set` front a run's generate
    should be conditioned on, or None when the run has never built one.

    The APPROVED set wins: a `create-tpose-set` completes unpromoted by
    design (the operator reviews the four tiles, then promotes the front),
    so a run that approved set A and later drew a fresh, still-unreviewed set
    B must keep generating from A -- picking the newest set would demand B
    be promoted and make the approved A ungeneratable. Preference order: the
    run's `current_artifact_id` when it is a set front, then the newest
    promoted set front, then the newest completed set (nothing approved yet).
    """
    completed = [
        item
        for item in run.attempts
        if item.stage == "create-tpose-set"
        and item.status == "completed"
        and item.output_artifact_id
    ]
    if not completed:
        return None
    current_id = getattr(run, "current_artifact_id", None)
    if current_id and any(item.output_artifact_id == current_id for item in completed):
        return current_id
    promoted = next((item for item in reversed(completed) if item.promoted), None)
    if promoted is not None:
        return promoted.output_artifact_id
    return completed[-1].output_artifact_id


def _resolve_conditioning_view_paths(source_artifact: Artifact, run: Any) -> dict[str, Path]:
    """The T-pose conditioning view PNGs `project-views` should bake onto
    `source_artifact`'s texture.

    Walks the `source_artifact_id` lineage starting at `source_artifact`
    itself (bounded by `POLICY_LINEAGE_DEPTH`), looking for the first
    artifact -- typically the `generate` output -- that recorded
    `metadata["source_views"]` (the {role: artifact_id} payload a multi-view
    generate was actually conditioned on). Falls back to the run's own
    canonical T-pose set (`latest_tpose_set_front` ->
    `tpose_set_conditioning_views`, which already includes "front") when the
    lineage carries no recorded views at all. Returns every resolvable role
    mapped to its artifact's file path; an artifact that has since been
    deleted is silently dropped rather than raising.
    """
    seen: set[str] = set()
    source_views: dict[str, str] = {}
    candidate: Artifact | None = source_artifact
    depth = 0
    while candidate is not None and candidate.id not in seen and depth < POLICY_LINEAGE_DEPTH:
        seen.add(candidate.id)
        recorded = candidate.metadata.get("source_views")
        if isinstance(recorded, dict) and recorded:
            source_views = {str(role): str(value) for role, value in recorded.items()}
            break
        parent_id = candidate.metadata.get("source_artifact_id")
        candidate = None
        if parent_id:
            try:
                candidate = read_artifact(parent_id)
            except HTTPException:
                candidate = None
        depth += 1

    if not source_views and run is not None:
        front_id = latest_tpose_set_front(run)
        if front_id:
            try:
                front = read_artifact(front_id)
            except HTTPException:
                front = None
            if front is not None:
                source_views = tpose_set_conditioning_views(front)

    resolved: dict[str, Path] = {}
    for role, artifact_id in source_views.items():
        try:
            resolved[role] = Path(read_artifact(artifact_id).path)
        except HTTPException:
            continue
    return resolved


# THE RULE for `rebuild-views` -- approach9.sh's one agent-in-the-loop step
# (which reference views are true same-angle identity for their role) encoded
# as the default. The T-pose set's left/right are three-quarter views that
# repaint the face from the side; dropping them measured front SSIM
# 0.485 -> 0.544 and back 0.605 -> 0.672 (2026-09-05).
REBUILD_VIEWS_DEFAULT_ROLES = ("front", "back")


def _select_rebuild_view_references(
    resolved: dict[str, Path], settings: dict[str, Any]
) -> dict[str, Path]:
    """The reference views a `rebuild-views` job actually paints with:
    front (+ back when it resolved), or exactly the roles
    `settings["reference_roles"]` names when the operator overrides THE RULE.
    """
    requested = settings.get("reference_roles")
    roles = (
        [str(role) for role in requested]
        if isinstance(requested, list) and requested
        else list(REBUILD_VIEWS_DEFAULT_ROLES)
    )
    selected = {role: resolved[role] for role in roles if role in resolved}
    if "front" not in selected:
        raise ValueError(
            "Rebuilding from views requires a resolvable front reference view; "
            f"resolved {sorted(resolved)}, selected {sorted(selected)}"
        )
    return selected


def evaluate_tpose_front(source_bytes: bytes, front_bytes: bytes) -> dict[str, Any]:
    """The composite's cheap auto-gate on the T-pose front it just drew.

    Deliberately built out of the checks that already exist rather than new
    ones (bead 4vquq's instruction): `view_consistency.compare_view_to_front`
    measures the candidate against the raw reference on the front plane --
    silhouette, leg separation, lower-body correlation, and the palette
    distance it publishes but never gates -- and
    `view_consistency.measure_silhouette` answers "is there a keyable subject
    here at all".

    The gate FAILS the job on exactly two conditions:

      * the candidate has no measurable subject, i.e. the composite is about
        to fan three expensive redraws off an image nothing can be measured
        in; and
      * the existing check returned an ERROR-grade finding, the same bar
        `blocking_view_conflicts` already applies to a conditioning view.

    Warnings and refusals-to-score never fail it, for the reason
    `view_consistency` states in its own docstring: a check that blocks on
    its own uncertainty is a check operators learn to route around. Colour
    distance is recorded, never gated -- the live "robed witch" failure wore
    exactly the same purple as the jumpsuit it replaced.
    """
    result = compare_view_to_front(source_bytes, front_bytes, view=TPOSE_SET_FRONT_ROLE)
    measurable = measure_silhouette(front_bytes) is not None
    reasons: list[str] = []
    if not measurable:
        reasons.append(
            "The generated T-pose front has no measurable subject silhouette "
            "(empty or frame-filling mask, or a background that cannot be "
            "keyed), so the three views would be fanned off an image nothing "
            "can be checked against."
        )
    # Same error bar `blocking_view_conflicts` applies to a conditioning view,
    # re-worded for this pairing: here the "front" being compared against is
    # the operator's raw reference, not another view's front.
    errors = [item for item in result["findings"] if item["severity"] == "error"]
    if errors:
        reasons.append(
            "The generated T-pose front contradicts the reference image it was "
            "drawn from. " + " ".join(item["message"] for item in errors)
        )
    return {
        "check_version": result["check_version"],
        "status": "failed" if reasons else "passed",
        "front_measurable": measurable,
        "palette_distance": result["measurements"].get("lower_color_distance"),
        "reasons": reasons,
        "comparison": result,
    }


async def _synthesize_view_into(
    job: JobState,
    *,
    view: str,
    source: Artifact,
    source_reference_bytes: bytes,
    base_metadata: dict[str, Any],
    seed: int,
    comfyui_url: str,
    destination: Path,
    progress: int,
    family: str = "tpose",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw ONE synthesized view into `destination` and measure it.

    Shared by `synthesize-views` (which lets a failure fail the whole job) and
    `create-tpose-set` (which catches it and completes the set partially), so
    both paths run identical prompts, identical provenance, and identical
    consistency checks. Returns `(view metadata, consistency result)`.
    """
    profile = resolve_view_synthesis_profile(view, family=family)
    _job_log(job, f"Synthesizing the {view} view", progress)
    try:
        comfy_result = await asyncio.to_thread(
            run_comfyui_image_edit,
            Path(source.path),
            destination,
            profile=profile,
            seed=seed,
            base_url=comfyui_url,
            on_accepted=lambda prompt_id, view=view: _job_log(
                job, f"ComfyUI accepted the {view} prompt {prompt_id}", job.progress,
            ),
            cancelled=lambda: job.status == "cancelled",
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"ComfyUI unavailable at {comfyui_url}. Confirm the local ComfyUI "
            "service is running before synthesizing views."
        ) from exc
    view_metadata = build_view_synthesis_metadata(
        base_metadata=base_metadata,
        comfy_result=comfy_result,
        view=view,
        source_artifact_id=source.id,
        profile=profile,
        seed=seed,
    )
    consistency = await asyncio.to_thread(
        compare_view_to_front,
        source_reference_bytes,
        destination.read_bytes(),
        view=view,
    )
    view_metadata["view_consistency"] = consistency
    if consistency["status"] != "clean":
        _job_log(
            job,
            f"View consistency for the {view} view: {consistency['status']}",
            job.progress,
        )
    return view_metadata, consistency


def _has_certified_rig_quality(artifact: Artifact) -> bool:
    """Certified rigs are retarget-ready: role resolution is what motion needs."""
    decisions = artifact.metadata.get("quality_decisions", {})
    certificate = decisions.get("rig-certificate", {}) if isinstance(decisions, dict) else {}
    return (
        isinstance(certificate, dict)
        and certificate.get("status") == "ready"
        and int(certificate.get("policy_version", 0)) >= 2
    )


def _has_ready_rig_quality(artifact: Artifact) -> bool:
    decisions = artifact.metadata.get("quality_decisions", {})
    certificate = decisions.get("rig-certificate", {}) if isinstance(decisions, dict) else {}
    return (
        isinstance(decisions, dict)
        and rig_candidate_is_promotable(decisions)
        and isinstance(certificate, dict)
        and certificate.get("status") == "ready"
        and int(certificate.get("policy_version", 0)) >= 2
    )


def _is_template_conformant_rig(artifact: Artifact) -> bool:
    """N5's direct-attach bar: certified AND the family-template conformance
    check (N3's skeleton_template.conformance_report, threaded through
    rig_quality.analyze_rig_path -- see quality_policy.evaluate_rig_certificate)
    came back completely clean.

    `_has_certified_rig_quality` alone is not quite enough here: a non-empty
    template_conformance list already blocks certification (it is folded
    into the rig-certificate findings), so a READY certificate implies a
    clean list -- EXCEPT `template_conformance` can also be `None` (no
    conformance data at all: a non-humanoid rig, or one bound before N3),
    which must not be treated as conformant even though it adds no
    findings. Checking both keeps that distinction explicit rather than
    relying on an implication baked into a different module.
    """
    if not _has_certified_rig_quality(artifact):
        return False
    rig_quality = artifact.metadata.get("rig_quality")
    conformance = rig_quality.get("template_conformance") if isinstance(rig_quality, dict) else None
    if conformance is None:
        # Cached metadata predating this field (an artifact bound before
        # N3's rig_quality wiring landed, or re-exported by a worker that
        # doesn't recompute rig_quality) is not the same as "no conformance
        # data" -- the GLB's own extras (S3's ground truth, baked at bind
        # time by unirig_bind_worker and durable across metadata staleness)
        # are the authoritative fallback.
        try:
            document, _binary = _read_glb(Path(artifact.path))
            conformance = _extract_template_conformance(document.get("nodes", []))
        except (OSError, ValueError, KeyError, struct.error):
            return False
    return isinstance(conformance, list) and len(conformance) == 0


def _biped_or_auto_humanoid(subject_kind: str, certificate: dict[str, Any]) -> bool:
    """SOMA paths (retarget-motion, and generate-motion's non-Slither branch)."""
    if subject_kind == "biped":
        return True
    return subject_kind == "auto" and certificate.get("profile") == "humanoid"


def _creature_or_auto_chain(subject_kind: str, certificate: dict[str, Any]) -> bool:
    """generate-motion's Slither branch (chain-profile creatures)."""
    if subject_kind == "creature":
        return True
    return subject_kind == "auto" and certificate.get("profile") == "chain"


def validate_final_review_candidate(artifact: Artifact) -> None:
    """Require candidate-specific evidence before final acceptance."""
    if artifact.kind != "glb":
        raise ValueError("Final review requires an animated GLB artifact")
    decisions = artifact.metadata.get("quality_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("Final review requires candidate-specific quality decisions")
    blocked = [
        str(stage)
        for stage, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("status") == "blocked"
    ]
    if blocked:
        raise ValueError(f"Final review is blocked by: {', '.join(sorted(blocked))}")
    deformation = artifact.metadata.get("deformation_review")
    evidence = deformation.get("review_artifacts") if isinstance(deformation, dict) else None
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Final review requires deformation evidence for this animated artifact")


def _rig_candidate_can_promote(
    stage: str,
    settings: dict[str, Any],
    artifact: Artifact,
) -> bool:
    """Apply rig quality and experiment-containment policy on every intake path."""
    if not rig_candidate_is_promotable(
        artifact.metadata.get("quality_decisions", {})
    ):
        return False
    if stage != "target-proxy-rig":
        return True
    surface = settings.get(
        "deformation_surface",
        artifact.metadata.get("deformation_surface", "visible-original"),
    )
    return deformation_surface_can_promote(str(surface))


def _download_log(state: DownloadState, message: str, progress: int | None = None) -> None:
    state.message = message
    if progress is not None:
        state.progress = progress
    if message and (not state.logs or state.logs[-1] != message):
        state.logs.append(message[-500:])
    _write_json(DOWNLOADS_DIR / f"{state.id}.json", state.as_dict())


def _run_model_download(state: DownloadState) -> None:
    if state.model == "instant-meshes":
        _run_instant_meshes_download(state)
        return
    if state.model in {"trellis2", "trellis-refiner"}:
        _run_trellis_refiner_download(state)
        return
    state.status = "failed"
    state.error = f"{state.model} has no downloadable model package"
    _download_log(state, f"Download failed: {state.error}", state.progress)


def _instant_meshes_download_spec() -> tuple[str, str, str]:
    if sys.platform == "win32":
        return (
            "https://instant-meshes.s3.eu-central-1.amazonaws.com/Release/instant-meshes-windows.zip",
            "instant-meshes-windows.zip",
            "Windows",
        )
    if sys.platform.startswith("linux"):
        return (
            "https://instant-meshes.s3.eu-central-1.amazonaws.com/instant-meshes-linux.zip",
            "instant-meshes-linux.zip",
            "Linux",
        )
    raise RuntimeError(f"Instant Meshes automatic installation is unsupported on {sys.platform}")


def _run_instant_meshes_download(state: DownloadState) -> None:
    """Fetch the upstream host bundle and safely unpack it into app data."""
    url, file_name, platform_name = _instant_meshes_download_spec()
    archive = DOWNLOADS_DIR / f"{state.id}-instant-meshes.zip"
    install = native.instant_meshes_dir()
    state.status = "running"
    _download_log(state, f"Downloading Instant Meshes for {platform_name}", 3)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AISmith-3D/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            state.file_name = file_name
            state.file_index = state.file_count = 1
            state.file_total_bytes = total
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                state.file_bytes += len(chunk)
                fraction = state.file_bytes / total if total else 0
                _download_log(state, f"Downloading Instant Meshes for {platform_name}", min(88, max(4, round(fraction * 88))))
        _download_log(state, "Extracting Instant Meshes executable", 90)
        install.mkdir(parents=True, exist_ok=True)
        destination_root = install.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (destination_root / member.filename).resolve()
                if destination != destination_root and destination_root not in destination.parents:
                    raise RuntimeError("Instant Meshes archive contains an unsafe file path")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if sys.platform.startswith("linux") and destination.name in {"Instant Meshes", "InstantMeshes"}:
                    destination.chmod(destination.stat().st_mode | 0o111)
        if native.instant_meshes_exe() is None:
            raise RuntimeError("The download completed but a native Instant Meshes executable was not found in the archive")
        state.status = "completed"
        _download_log(state, "Instant Meshes is ready", 100)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        _download_log(state, f"Download failed: {exc}", state.progress)
    finally:
        archive.unlink(missing_ok=True)
def _trellis_refiner_runtime_packages() -> list[str]:
    packages = ["huggingface_hub", "rembg[cpu]==2.0.81"]
    if os.name != "nt":
        packages.extend([
            "spconv-cu121==2.3.8",
            "nvidia-cuda-nvcc==13.0.48",
            "nvidia-nvvm==13.0.48",
            "nvidia-cuda-crt==13.0.48",
            "nvidia-cuda-cccl",
            "ninja",
        ])
    return packages



def _trellis_native_install_command(python: Path, runtime: Path) -> list[str]:
    return [
        str(python),
        str(ROOT / "backend" / "workers" / "install_trellis2_native.py"),
        "--marker",
        str(runtime / "native-contract.json"),
    ]



def _run_trellis_refiner_download(state: DownloadState) -> None:
    """Install the isolated ComfyUI runtime and FP8 refiner checkpoints.

    It deliberately uses the app's CUDA Python environment rather than the
    generation C++ runtime.  The two stacks must never share a process.
    """
    runtime = native.trellis_refiner_dir()
    comfy = runtime / "ComfyUI"
    state.status = "running"
    _download_log(state, "Preparing isolated TRELLIS.2 FP8 runtime", 2)
    try:
        if native.offline_mode():
            raise RuntimeError("AISMITH_OFFLINE is set, so nothing is downloaded here. Install weights with `python -m backend.engine.weights install`.")
        wrapper_source = engine_sources.require_vendored("comfyui-trellis2")
        if not (comfy / "folder_paths.py").exists():
            # Pinned by commit (vendor/sources.lock.json), never a moving branch.
            pinned = engine_sources.source("comfyui")
            state.file_name, state.file_index, state.file_count = f"ComfyUI {pinned.ref or pinned.commit[:12]}", 1, 5
            _download_log(state, "Downloading the local CUDA refiner runtime", 5)
            staging = runtime / f"ComfyUI-{state.id}"
            shutil.rmtree(staging, ignore_errors=True)
            engine_sources.clone_pinned("comfyui", staging)
            staging.rename(comfy)
        _download_log(state, "Installing the TRELLIS.2 refiner CUDA extensions", 25)
        target = comfy / "custom_nodes" / "ComfyUI-Trellis2"
        if not (target / "nodes.py").is_file():
            shutil.copytree(wrapper_source, target, dirs_exist_ok=True)
        system = "Windows" if os.name == "nt" else "Linux"
        torch_release = "Torch280" if os.name == "nt" else "Torch291"
        wheel_root = wrapper_source / "wheels" / system / torch_release
        wheels = sorted(wheel_root.glob("*cp312*.whl"))
        if os.name != "nt":
            wheels = [wheel for wheel in wheels if not wheel.name.startswith(("cumesh-", "o_voxel-"))]
        if not wheels:
            raise RuntimeError(f"No CUDA extension wheels found for {system} Python 3.12 / {torch_release}")

        refiner_python = native.trellis_refiner_python()
        if not refiner_python.exists():
            venv.EnvBuilder(with_pip=True).create(refiner_python.parents[1])
        if os.name != "nt":
            torch_command = [
                str(refiner_python), "-m", "pip", "install",
                "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
                "--index-url", "https://download.pytorch.org/whl/cu130",
            ]
            completed = subprocess.run(torch_command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
            if completed.returncode != 0:
                raise RuntimeError(f"Could not install the Linux PyTorch runtime: {(completed.stdout + completed.stderr)[-1200:]}")
        command = [str(refiner_python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "-r", str(comfy / "requirements.txt"), *_trellis_refiner_runtime_packages()]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0:
            raise RuntimeError(f"Could not install TRELLIS.2 refiner dependencies: {(completed.stdout + completed.stderr)[-1200:]}")
        _download_log(state, "Downloading TRELLIS.2 FP8 refinement and PBR models", 35)
        if os.name != "nt":
            native_command = _trellis_native_install_command(refiner_python, runtime)
            completed = subprocess.run(native_command, cwd=ROOT, capture_output=True, text=True, timeout=3600)
            if completed.returncode != 0:
                raise RuntimeError(f"Could not build GPU-native TRELLIS.2 extensions: {(completed.stdout + completed.stderr)[-1200:]}")
        # Pinned revisions and hashes (vendor/weights.lock.json). DINOv3 is
        # gated: the Hugging Face login on this machine must have accepted it.
        state.file_name, state.file_index, state.file_count = "TRELLIS.2 FP8, DINOv3 and the sparse-structure decoder", 3, 5
        _download_log(state, "Downloading pinned TRELLIS.2 FP8 refinement and PBR models", 40)
        engine_weights.install(
            native.engine_weight_roots(),
            ["trellis2-4b-fp8", "dinov3-vitl16", "trellis-image-large-ss-decoder"],
            token=os.getenv("HF_TOKEN") or None,
            cache_path=engine_weights.default_cache_path(native.engine_weight_roots()),
        )
        if not native.trellis_refiner_ready():
            raise RuntimeError("The TRELLIS.2 refiner download did not contain all required model files")
        state.status = "completed"
        _download_log(state, "TRELLIS.2 FP8 Refine is ready", 100)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        _download_log(state, f"Download failed: {exc}", state.progress)


async def _execute_download(state: DownloadState) -> None:
    await asyncio.to_thread(_run_model_download, state)


def _job_log(job: JobState, message: str, progress: int | None = None) -> None:
    job.message = message
    if progress is not None:
        job.progress = progress
    job.logs.append(message)
    _write_json(JOBS_DIR / f"{job.id}.json", job.as_dict())


def _trellis_job_progress(message: str, current: int) -> int:
    """Map one-shot FP8 worker logs to useful, non-jumping progress."""
    xatlas = re.search(r"xatlas: Computing charts:\s+(\d{1,3})%", message, re.IGNORECASE)
    if xatlas:
        # UV charting is the long CPU-bound section after the model stages.
        # Surface its own progress instead of pinning the UI at 78%.
        return min(92, 78 + round(int(xatlas.group(1)) * 14 / 100))
    stage_values = {"Loading TRELLIS": 14, "Generating": 30, "Refining geometry": 30, "Reconstructing": 55, "Simplifying": 65, "Filling holes": 70, "Generating PBR": 78, "Exporting": 94, "job complete": 96}
    for marker, value in stage_values.items():
        if marker in message:
            return value
    return min(78, max(18, current + 1))


def _parse_rig_adjust_settings(settings: dict[str, Any]) -> tuple[dict[str, list[float]], list[str]]:
    """Parse and validate a rig-adjust job's settings: hand-adjusted joint
    positions and/or joints to remove. A rig-adjust job must request at
    least one of the two; raises `ValueError` otherwise."""
    positions = settings.get("positions") or {}
    remove_joints = [str(name) for name in (settings.get("remove_joints") or [])]
    if not positions and not remove_joints:
        raise ValueError("Rig adjust requires settings.positions or settings.remove_joints")
    return positions, remove_joints


# The largest vertex target canonicalize's own gate accepts: quality_tiers
# converts vertices to triangles at TRIANGLES_PER_VERTEX and refuses anything
# past CUSTOM_TARGET_TRIANGLES_MAXIMUM, so a request above this would pass the
# schema and then fail at execution time.
CANONICALIZE_TARGET_VERTICES_MAXIMUM = int(CUSTOM_TARGET_TRIANGLES_MAXIMUM // TRIANGLES_PER_VERTEX)


def _parse_canonicalize_settings(request: JobRequest) -> tuple[int, tuple[str, ...], str]:
    """Validate a canonicalize job's settings: the operator's mesh density
    (settings.quality_tier, or a custom settings.target_triangles /
    settings.target_vertices -- resolved by
    `quality_tiers.resolve_quality_target`, which is also what
    `native.run_canonicalize` calls, so a 400 here means the job would have
    been rejected there), optional views (defaults to the S2 gate's four
    matched viewpoints), and optional method (settings.method; defaults to
    native's voxel remesh -- see native.DEFAULT_CANONICALIZE_METHOD).

    The returned vertex target is only the legacy Instant Meshes path's
    `--vertices` knob; the voxel path re-resolves the same intent into a
    TRIANGLE target inside `run_canonicalize`."""
    settings = request.settings
    target_vertices_raw = settings.get("target_vertices")
    if target_vertices_raw is not None:
        try:
            explicit_vertices = int(target_vertices_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonicalize settings.target_vertices must be an integer") from exc
        if not (500 <= explicit_vertices <= CANONICALIZE_TARGET_VERTICES_MAXIMUM):
            raise ValueError(
                "canonicalize settings.target_vertices must be between 500 and "
                f"{CANONICALIZE_TARGET_VERTICES_MAXIMUM:,}"
            )
    if request.target_vertices is not None and request.target_vertices > CANONICALIZE_TARGET_VERTICES_MAXIMUM:
        # The pydantic bound (le=1_000_000) is the retopologize knob's range;
        # canonicalize's gate tops out at CUSTOM_TARGET_TRIANGLES_MAXIMUM
        # triangles, so reject here (-> 400) rather than 202 then fail.
        raise ValueError(
            "canonicalize target_vertices must be between 500 and "
            f"{CANONICALIZE_TARGET_VERTICES_MAXIMUM:,}"
        )
    # Raises ValueError (-> 400) on an unknown tier or an out-of-range custom
    # target, rather than starting a job that cannot pass its own gate.
    quality = resolve_quality_target(settings, request_target_vertices=request.target_vertices)
    target_vertices = quality.target_vertices

    views_raw = settings.get("views")
    if views_raw is None:
        views = native.DEFAULT_CANONICALIZE_VIEWS
    else:
        if not isinstance(views_raw, list) or not views_raw:
            raise ValueError("canonicalize settings.views must be a non-empty list")
        views = tuple(str(view) for view in views_raw)
        unknown_views = sorted(set(views) - {"front", "back", "left", "right"})
        if unknown_views:
            raise ValueError(f"canonicalize settings.views has unsupported view(s): {unknown_views}")
        if len(set(views)) != len(views):
            raise ValueError("canonicalize settings.views must not repeat a view")

    method_raw = settings.get("method")
    method = native.DEFAULT_CANONICALIZE_METHOD if method_raw is None else str(method_raw).strip().lower()
    if method not in native.CANONICALIZE_METHODS:
        raise ValueError(f"canonicalize settings.method must be one of {native.CANONICALIZE_METHODS}")

    return target_vertices, views, method


async def _execute_job(job: JobState, request: JobRequest) -> None:
    source: Artifact | None = None
    model_name: str | None = None
    # `create-tpose-set` has to save its front artifact BEFORE it draws the
    # three views, because each view records the front's artifact id as its
    # own provenance. The shared tail below stamps this artifact rather than
    # saving a second copy of the same file. None for every other job kind.
    prebuilt_artifact: Artifact | None = None
    try:
        source = read_artifact(request.source_artifact_id) if request.source_artifact_id else None
        if job.status == "cancelled":
            return
        if job.run_id and job.attempt_id:
            run_store.mark_attempt_running(job.run_id, job.attempt_id, job.id)
        job.status = "running"
        _job_log(job, "Preparing native local worker", 5)
        # Chain-rigged creatures (serpents, tails -- see skeleton_naming.py's
        # chain profile) never carry a humanoid SOMA skeleton, so a
        # `generate-motion` request against one bakes a procedural Slither
        # cycle directly with Blender instead of asking NVIDIA Kimodo for a
        # BVH clip it could never retarget onto them (task F3).
        is_slither = (
            request.kind == "generate-motion"
            and source is not None
            and source.kind == "glb"
            and (source.metadata.get("quality_decisions") or {}).get("rig-certificate", {}).get("profile")
            == "chain"
        )
        model_name = {
            "generate": "trellis2",
            "retopologize": "trellis2-fp8",
            "retopo": "instant-meshes",
            # Canonicalize defaults to the Blender voxel-remesh backend
            # (native.DEFAULT_CANONICALIZE_METHOD); it only loads Instant
            # Meshes when settings.method="instant-meshes" explicitly asks
            # for the comparison path (see the request-validation gate
            # below, which mirrors this same method check).
            "canonicalize": (
                "instant-meshes"
                if str(request.settings.get("method") or native.DEFAULT_CANONICALIZE_METHOD).strip().lower() == "instant-meshes"
                else "blender"
            ),
            "paint-bake": "blender",
            "project-views": "blender",
            "rebuild-views": "blender",
            "repair-weights": "blender",
            "auto-rig": "unirig",
            "rig-adjust": "unirig",
            "rig-transfer": "blender",
            "target-proxy-rig": "blender",
            "generate-motion": "blender" if is_slither else "kimodo",
            "retarget-motion": "blender",
            "apply-clip": "blender",
            # Always ComfyUI: there is no "upload" pass-through for synthesis.
            "synthesize-views": "flux2-klein",
            # The composite (bead 4vquq) holds the GPU for its whole staged
            # run -- one acquire covers the front and all three views, so no
            # other job can slip between the front and the views it must be
            # fanned from.
            "create-tpose-set": "flux2-klein",
        }.get(request.kind)
        if request.kind == "prepare-image" and str(
            request.settings.get("adapter") or "upload"
        ).strip() == "comfyui":
            model_name = "flux2-klein"
        if model_name:
            await model_manager.acquire(model_name, job.id)
        if job.status == "cancelled":
            return

        is_motion = request.kind == "generate-motion" and not is_slither
        is_image = request.kind in {"prepare-image", "synthesize-views", "create-tpose-set"}
        if is_motion:
            output_suffix = ".bvh"
        elif request.kind in {"synthesize-views", "create-tpose-set"}:
            output_suffix = ".png"
        elif is_image and source is not None:
            adapter = str(request.settings.get("adapter") or "upload").strip()
            output_suffix = ".png" if adapter == "comfyui" else Path(source.path).suffix.lower() or ".png"
        else:
            output_suffix = ".glb"
        output = ARTIFACTS / f"job-{job.id}{output_suffix}"
        metadata = {"stage": request.kind, "settings": request.settings, "native": True, "worker_isolated": True}
        # Keep provenance with every generated asset.  The optional image-guided
        # TRELLIS refiner needs the exact original reference image; it must not
        # guess one from the texture baked into a GLB.
        if source is not None:
            metadata["source_artifact_id"] = source.id
        if request.kind == "generate" and source is not None:
            metadata["source_image_artifact_id"] = source.id
        if request.kind == "generate" and request.settings.get("tpose_set_id"):
            # Bead 4vquq: the auditable record that this generate was
            # conditioned on the run's canonical T-pose set rather than on a
            # raw upload, plus any per-role operator override applied on top.
            metadata["conditioning_source"] = "tpose-set"
            metadata["tpose_set_id"] = request.settings["tpose_set_id"]
            metadata["tpose_set_overrides"] = request.settings.get("tpose_set_overrides") or {}
        if request.kind == "prepare-image":
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("Image preparation requires one source image")
            adapter = str(request.settings.get("adapter") or "upload").strip()
            if adapter not in {"upload", "comfyui"}:
                raise ValueError(f"Unsupported image preparation adapter: {adapter}")
            profile = resolve_preparation_profile(str(
                request.settings.get("profile") or "humanoid-t-pose-rig-v1"
            ))
            if adapter == "comfyui":
                metadata.update({"native": False, "worker_isolated": False, "execution": "external-comfyui"})
                _job_log(job, "Generating a reviewable rig-pose candidate with local ComfyUI", 20)
                comfyui_url = os.getenv("AISMITH_COMFYUI_URL", "http://127.0.0.1:8188")
                metadata.update(await asyncio.to_thread(
                    run_comfyui_image_edit,
                    Path(source.path),
                    output,
                    profile=profile,
                    seed=request.seed,
                    base_url=comfyui_url,
                    on_accepted=lambda prompt_id: _job_log(
                        job,
                        f"ComfyUI accepted prompt {prompt_id} with flux2-klein-image-edit-v1",
                        35,
                    ),
                    cancelled=lambda: job.status == "cancelled",
                ))
                _job_log(job, "Local ComfyUI candidate retained for visual review", 80)
            else:
                _job_log(job, "Retaining uploaded preparation candidate with versioned profile", 60)
                shutil.copy2(source.path, output)
            metadata.update({
                "adapter": adapter,
                "preparation_profile": profile.as_dict(),
                "seed": request.seed,
            })
            job.promoted_output = False
            job.warnings.append(
                "Prepared image candidate requires explicit visual review before Generate"
            )
        elif request.kind == "generate-motion":
            if source is None or source.kind != "glb":
                raise ValueError("Motion generation requires the rigged GLB it will be retargeted onto")
            certificate = (source.metadata.get("quality_decisions") or {}).get("rig-certificate", {})
            certified = certificate.get("status") == "ready" and int(certificate.get("policy_version", 0)) >= 2
            if not (certified or _has_ready_rig_quality(source)):
                raise ValueError("Motion generation requires a rig with a ready rig certificate")
            if is_slither:
                _job_log(job, "Baking a procedural Slither cycle with Blender", 12)
                metadata.update(await asyncio.to_thread(
                    native.run_slither_bake,
                    Path(source.path),
                    output,
                    "Slither",
                    request.settings,
                    progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                ))
                # Overwritten below by the shared rig/animation/deformation
                # gate block (same promotion policy as retarget-motion).
                job.promoted_output = False
            else:
                prompt = str(request.settings.get("prompt") or "A stylized female sword fighter performs one decisive slash and recovers to a combat-ready stance").strip()
                duration = float(request.settings.get("duration_seconds", 2.0))
                if not prompt:
                    raise ValueError("Motion generation requires a text prompt")
                if not 0.5 <= duration <= 10.0:
                    raise ValueError("Motion duration must be between 0.5 and 10 seconds")
                _job_log(job, "Generating a reproducible SOMA humanoid motion with NVIDIA Kimodo", 12)
                metadata.update(await asyncio.to_thread(
                    native.run_kimodo_motion,
                    prompt,
                    output,
                    duration=duration,
                    seed=request.seed,
                    progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 3))),
                ))
                # BVH is a child motion artifact. It must remain attached to the
                # character lifecycle without replacing the current playable GLB.
                job.promoted_output = False
        elif request.kind == "generate":
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("Generate requires a single uploaded image")
            mode_name = "fast" if request.resolution == 512 else "high-detail"
            view_images: dict[str, Path] | None = None
            resolved_source_views: dict[str, str] | None = None
            if request.source_views and len(request.source_views) > 1:
                view_images = {}
                resolved_source_views = {}
                for label, artifact_id in request.source_views.items():
                    view_artifact = read_artifact(artifact_id)
                    view_images[label] = Path(view_artifact.path)
                    resolved_source_views[label] = view_artifact.id
            if native.trellis_refiner_ready():
                _job_log(job, f"Selected official TRELLIS.2 FP8 validated-geometry workflow in {mode_name} mode", 12)
                if view_images:
                    result = await asyncio.to_thread(
                        native.run_trellis_fp8,
                        Path(source.path),
                        output,
                        request.resolution,
                        lambda message: _job_log(job, message, _trellis_job_progress(message, job.progress)),
                        view_images,
                        cancelled=lambda: job.status == "cancelled",
                    )
                else:
                    result = await asyncio.to_thread(
                        native.run_trellis_fp8,
                        Path(source.path),
                        output,
                        request.resolution,
                        lambda message: _job_log(job, message, _trellis_job_progress(message, job.progress)),
                        cancelled=lambda: job.status == "cancelled",
                    )
                metadata.update(result)
                # Multi-view conditioning is FP8-only; only stamp provenance
                # here where the views were actually used.
                if resolved_source_views:
                    metadata["source_views"] = resolved_source_views
            else:
                _job_log(job, f"Selected isolated TRELLIS.2 GGUF geometry + PBR workflow in {mode_name} mode", 12)
                if request.source_views:
                    _job_log(
                        job,
                        "source_views were supplied but the GGUF fallback path does not "
                        "support multi-view conditioning; generating from the front image only",
                        job.progress,
                    )
                result = await asyncio.to_thread(native.run_trellis_gguf, Path(source.path), output, request.resolution, lambda message: _job_log(job, message, _trellis_job_progress(message, job.progress)))
                metadata.update(result)
        elif request.kind == "auto-rig":
            if source is None or source.kind != "glb":
                raise ValueError("Auto-rig requires a single GLB mesh artifact")
            if not native.unirig_ready():
                raise RuntimeError(
                    "UniRig runtime is not installed. Run scripts/setup_unirig.sh "
                    "or set UNIRIG_DIR to a prepared UniRig checkout."
                )
            profile_hint = request.settings.get("profile_hint")
            if profile_hint:
                metadata["profile_hint"] = profile_hint
            _job_log(job, "Predicting a skeleton with UniRig", 12)
            metadata.update(await asyncio.to_thread(
                native.run_unirig_auto_rig,
                Path(source.path),
                output,
                progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 6))),
                unirig_settings=request.settings.get("unirig"),
                chain_joints=request.settings.get("chain_joints"),
                profile_hint=profile_hint,
                skip_optional_bones=bool(request.settings.get("skip_optional_bones", False)),
                skip_distal_refinement=bool(request.settings.get("skip_distal_refinement", False)),
            ))
        elif request.kind == "rig-adjust":
            if source is None or source.kind != "glb":
                raise ValueError("Rig adjust requires the auto-rig GLB artifact")
            positions, remove_joints = _parse_rig_adjust_settings(request.settings)
            inputs_dir = (source.metadata or {}).get("bind_inputs_dir")
            if not inputs_dir or not Path(inputs_dir).is_dir():
                raise ValueError("This artifact has no stored bind inputs; re-run auto-rig first")
            clean_glb = Path(inputs_dir) / "clean.glb"
            skeleton_npz = Path(inputs_dir) / "predict_skeleton.npz"
            raw_npz = Path(inputs_dir) / "raw_data.npz"
            _job_log(job, f"Rebinding with {len(positions)} adjusted joint(s), {len(remove_joints)} removal(s)", 12)
            rebind_result = await asyncio.to_thread(
                native.run_unirig_rebind,
                clean_glb,
                skeleton_npz,
                raw_npz,
                output,
                positions,
                source_rig=Path(source.path),
                remove_joints=remove_joints,
                skip_optional_bones=bool(request.settings.get("skip_optional_bones", False)),
                skip_distal_refinement=bool(request.settings.get("skip_distal_refinement", False)),
                progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 8))),
            )
            metadata.update(rebind_result)
            metadata["adjusted_bones"] = sorted(positions.keys())
            metadata["removed_bones"] = remove_joints
            # Carry the lineage's bind inputs forward so an adjusted rig can be adjusted again.
            metadata["bind_inputs_dir"] = str(inputs_dir)
            refused_joints = rebind_result.get("refused_joints") or []
            if refused_joints:
                job.warnings.append(f"Could not remove joint(s): {', '.join(refused_joints)}")
        elif request.kind == "synthesize-views":
            # Auto-generate-views (E3): synthesize back/left/right images from
            # one approved front photo with the same image-conditioned Flux.2
            # Klein path the T-pose candidate uses, so identity, outfit, and
            # lighting survive the redraw. A mirrored copy of the front image
            # is explicitly the wrong answer here (project log, 08-30) -- each
            # view is a genuine redraw from its own prompt, not a flip.
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("View synthesis requires one source image")
            requested_views = [
                str(view) for view in (request.settings.get("views") or DEFAULT_SYNTHESIZED_VIEWS)
            ]
            profile_family = str(request.settings.get("profile_family") or "tpose")
            metadata["view_synthesis_family"] = profile_family
            # Provenance for the choice `select_view_synthesis_source` made:
            # the tag AND the reason, so an operator inspecting a view months
            # later can see which image it was redrawn from and why that one.
            metadata["view_synthesis_source"] = request.settings.get("view_synthesis_source") or "front-image"
            metadata["view_synthesis_source_reason"] = request.settings.get("view_synthesis_source_reason") or (
                "this job recorded no source selection (submitted directly, or "
                "before the source was resolved), so the caller's source image was "
                "used as-is."
            )
            metadata.update({"native": False, "worker_isolated": False, "execution": "external-comfyui"})
            base_metadata = dict(metadata)
            comfyui_url = os.getenv("AISMITH_COMFYUI_URL", "http://127.0.0.1:8188")
            _job_log(job, f"Synthesizing {len(requested_views)} view(s) with local ComfyUI", 10)
            sibling_artifacts: dict[str, str] = {}
            # Bead 0wcdg: every synthesized view is checked against the front
            # reference it was drawn from before the operator can spend a
            # generate on it. The live failure this guards -- a left view that
            # reinvented a jumpsuited character as a robed witch, which the
            # multi-view bbox-height check let through because the heights
            # matched -- is documented in `view_consistency`.
            source_reference_bytes = Path(source.path).read_bytes()
            view_consistency: dict[str, dict[str, Any]] = {}
            primary_view = requested_views[0]
            view_span = max(1, 60 // max(len(requested_views), 1))
            with TemporaryDirectory(prefix="aismith-view-synthesis-") as temporary:
                temporary_root = Path(temporary)
                for index, view in enumerate(requested_views):
                    if job.status == "cancelled":
                        return
                    view_seed = request.seed + index + 1
                    view_output = temporary_root / f"{view}.png"
                    view_metadata, consistency = await _synthesize_view_into(
                        job,
                        view=view,
                        source=source,
                        source_reference_bytes=source_reference_bytes,
                        base_metadata=base_metadata,
                        seed=view_seed,
                        comfyui_url=comfyui_url,
                        destination=view_output,
                        progress=min(78, 10 + index * view_span),
                        family=profile_family,
                    )
                    view_consistency[view] = consistency
                    if view == primary_view:
                        shutil.copy2(view_output, output)
                        metadata.update(view_metadata)
                    else:
                        sibling = save_artifact(
                            view_output,
                            name=f"Synthesized {view} view.png",
                            kind="image",
                            metadata=view_metadata,
                        )
                        sibling_artifacts[view] = sibling.id
            if sibling_artifacts:
                metadata["sibling_views"] = sibling_artifacts
            if view_consistency:
                # Per-view detail already rides on each view artifact's own
                # metadata; this is the job-level roll-up the studio reads to
                # flag the offending tile.
                metadata["view_consistency_summary"] = {
                    label: {
                        "status": result["status"],
                        "codes": [item["code"] for item in result["findings"]],
                    }
                    for label, result in view_consistency.items()
                }
                job.warnings.extend(view_consistency_warnings(view_consistency))
            job.promoted_output = False
            job.warnings.append(
                "Synthesized views require explicit visual review before Generate"
            )
        elif request.kind == "create-tpose-set":
            # Bead 4vquq. One operator step, four staged phases:
            #   1/4  draw the T-pose FRONT from the raw reference (the exact
            #        `prepare-image` ComfyUI path and profile),
            #        then auto-gate it -- a bad front short-circuits here
            #        rather than fanning three expensive redraws off it,
            #   2-4/4 synthesize back/left/right FROM THAT FRONT (the exact
            #        `synthesize-views` path, every guard live).
            # The four artifacts are then registered as one SET. A single
            # view failing is NOT fatal: the set completes partially filled
            # with a warning naming the missing role, and per-tile regenerate
            # is the remedy -- losing an already-paid-for front and two good
            # views to one bad redraw is the worse outcome.
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("Creating a T-pose set requires one source image")
            profile = resolve_preparation_profile(str(
                request.settings.get("profile") or "humanoid-t-pose-rig-v1"
            ))
            requested_views = [
                str(view) for view in (request.settings.get("views") or TPOSE_SET_VIEW_ROLES)
            ]
            set_id = uuid.uuid4().hex
            total_members = 1 + len(requested_views)
            comfyui_url = os.getenv("AISMITH_COMFYUI_URL", "http://127.0.0.1:8188")
            metadata.update({
                "native": False,
                "worker_isolated": False,
                "execution": "external-comfyui",
                "adapter": "comfyui",
                "preparation_profile": profile.as_dict(),
                "seed": request.seed,
                # The raw upload stays in the run as the palette/identity
                # reference; from here on it is never a conditioning view.
                "reference_artifact_id": source.id,
            })
            _job_log(job, f"Generating the T-pose front… 1/{total_members}", 8)
            front_result = await asyncio.to_thread(
                run_comfyui_image_edit,
                Path(source.path),
                output,
                profile=profile,
                seed=request.seed,
                base_url=comfyui_url,
                on_accepted=lambda prompt_id: _job_log(
                    job,
                    f"ComfyUI accepted the T-pose front prompt {prompt_id}",
                    18,
                ),
                cancelled=lambda: job.status == "cancelled",
            )
            if job.status == "cancelled":
                output.unlink(missing_ok=True)
                return
            metadata.update(front_result)
            _job_log(job, "Checking the T-pose front against the reference image", 30)
            reference_bytes = Path(source.path).read_bytes()
            front_bytes = output.read_bytes()
            gate = await asyncio.to_thread(evaluate_tpose_front, reference_bytes, front_bytes)
            metadata["tpose_front_gate"] = gate
            if gate["status"] == "failed":
                output.unlink(missing_ok=True)
                raise ValueError(
                    "The T-pose front did not pass its own check, so no views were "
                    "drawn from it: " + " ".join(gate["reasons"])
                )
            _job_log(job, "T-pose front accepted; fanning the remaining views from it", 34)
            # Every view is drawn from, and measured against, the T-POSE
            # FRONT -- never the raw upload. That is the whole point of the
            # composite: `synthesize-views` had to pick a reference at
            # submission time (`select_view_synthesis_source`), and this path
            # removes the choice by construction.
            front_artifact = save_artifact(
                output,
                name=f"T-pose front{output_suffix}",
                kind="image",
                metadata=dict(metadata),
            )
            prebuilt_artifact = front_artifact
            base_metadata = {
                "stage": request.kind,
                "settings": request.settings,
                "native": False,
                "worker_isolated": False,
                "execution": "external-comfyui",
                "view_synthesis_source": "tpose-set-front",
                "view_synthesis_source_reason": (
                    f"{front_artifact.id} is the T-pose front this same "
                    f"create-tpose-set job just generated and gated, so every "
                    f"view in the set is redrawn from the reviewed pose and "
                    f"outfit rather than from the raw reference image "
                    f"{source.id}."
                ),
            }
            members: dict[str, str] = {TPOSE_SET_FRONT_ROLE: front_artifact.id}
            failed_members: dict[str, str] = {}
            view_consistency: dict[str, dict[str, Any]] = {}
            view_span = max(1, 50 // max(len(requested_views), 1))
            # A cancel mid-fan does NOT `return` out of here: the front is
            # already a saved artifact and every finished view already points
            # at it, so the loop breaks and the set-registration below stamps
            # what exists (marked incomplete, the unrun roles named) before
            # the shared tail's cancel check unlinks the working file. Bailing
            # early orphaned a front with no `tpose_set` block plus views
            # nothing referenced.
            with TemporaryDirectory(prefix="aismith-tpose-set-") as temporary:
                temporary_root = Path(temporary)
                for index, view in enumerate(requested_views):
                    if job.status == "cancelled":
                        break
                    view_output = temporary_root / f"{view}.png"
                    _job_log(
                        job,
                        f"Generating the {view} view… {index + 2}/{total_members}",
                        min(88, 38 + index * view_span),
                    )
                    try:
                        view_metadata, consistency = await _synthesize_view_into(
                            job,
                            view=view,
                            source=front_artifact,
                            source_reference_bytes=front_bytes,
                            base_metadata=base_metadata,
                            seed=request.seed + index + 1,
                            comfyui_url=comfyui_url,
                            destination=view_output,
                            progress=min(88, 38 + index * view_span),
                        )
                    except InterruptedError:
                        # `run_comfyui_image_edit` raises this only when the
                        # job itself was cancelled mid-prompt.
                        break
                    except Exception as exc:
                        failed_members[view] = str(exc)
                        job.warnings.append(
                            f"{view} view could not be generated ({exc}); the T-pose "
                            f"set is complete without it -- regenerate that tile to "
                            f"fill it in."
                        )
                        _job_log(job, f"The {view} view failed: {exc}", job.progress)
                        continue
                    view_metadata["tpose_set"] = {
                        "set_id": set_id,
                        "role": view,
                        "front_artifact_id": front_artifact.id,
                        "reference_artifact_id": source.id,
                        "job_id": job.id,
                    }
                    member = save_artifact(
                        view_output,
                        name=f"T-pose {view} view.png",
                        kind="image",
                        metadata=view_metadata,
                    )
                    members[view] = member.id
                    view_consistency[view] = consistency
            if view_consistency:
                metadata["view_consistency_summary"] = {
                    label: {
                        "status": result["status"],
                        "codes": [item["code"] for item in result["findings"]],
                    }
                    for label, result in view_consistency.items()
                }
                job.warnings.extend(view_consistency_warnings(view_consistency))
            if job.status == "cancelled":
                for view in requested_views:
                    if view not in members and view not in failed_members:
                        failed_members[view] = "cancelled before this view was drawn"
            membership = {
                "set_id": set_id,
                "role": TPOSE_SET_FRONT_ROLE,
                "front_artifact_id": front_artifact.id,
                "reference_artifact_id": source.id,
                "job_id": job.id,
                "members": members,
                "failed_members": failed_members,
                "complete": not failed_members,
            }
            metadata["tpose_set"] = membership
            # `sibling_views` is the shape the studio's view-grid rehydration
            # already reads off a completed synthesis output; keeping it means
            # the grid fills from a set without a second code path.
            metadata["sibling_views"] = {
                role: artifact_id for role, artifact_id in members.items()
                if role != TPOSE_SET_FRONT_ROLE
            }
            _job_log(
                job,
                f"Registering the canonical T-pose set ({len(members)}/{total_members} members)",
                92,
            )
            for role, member_id in members.items():
                _stamp_artifact_metadata(member_id, {"tpose_set": {
                    **membership,
                    "role": role,
                }})
            job.promoted_output = False
            job.warnings.append(
                "The T-pose set requires explicit visual review before Generate"
            )
        elif source:
            _job_log(job, f"Loading {source.name} into the {request.kind} worker", 15)
            if request.kind == "retopologize":
                target_vertices = request.target_vertices or request.target_faces
                repair_holes = bool(request.settings.get("repair_holes", False))
                gpu_first = bool(request.settings.get("gpu_first", False))
                # An explicitly attached image is required for imported GLBs and
                # takes precedence over Generate provenance.
                image_id = str(request.settings.get("refine_image_artifact_id") or source.metadata.get("source_image_artifact_id") or "")
                if not image_id:
                    parent_id = str(source.metadata.get("source_artifact_id") or "")
                    if parent_id:
                        parent = read_artifact(parent_id)
                        image_id = str(parent.metadata.get("source_image_artifact_id") or (parent.id if parent.kind == "image" else ""))
                if not image_id:
                    raise ValueError("Refine & Reconstruct requires a reference image. Attach one for imported GLBs.")
                image = read_artifact(image_id)
                if image.kind != "image":
                    raise ValueError("The retained refinement reference is not an image artifact.")
                metadata["source_image_artifact_id"] = image.id
                metadata.update(await asyncio.to_thread(
                    native.run_trellis_ai_refine,
                    Path(source.path),
                    Path(image.path),
                    output,
                    target_vertices,
                    repair_holes,
                    gpu_first,
                    lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                    cancelled=lambda: job.status == "cancelled",
                ))
            elif request.kind == "retopo":
                target_vertices = request.target_vertices or request.target_faces
                repair_holes = bool(request.settings.get("repair_holes", False))
                metadata.update(await asyncio.to_thread(native.run_instant_meshes, Path(source.path), output, target_vertices, repair_holes, lambda message: _job_log(job, message, min(94, max(18, job.progress + 4)))))
                if (w := (metadata.get("meshfix_repair") or {}).get("warning")):
                    job.warnings.append(w)
            elif request.kind == "canonicalize":
                canonicalize_target_vertices, canonicalize_views, canonicalize_method = _parse_canonicalize_settings(request)
                # `run_canonicalize` re-resolves the operator's density intent
                # from this settings dict, so a top-level `target_vertices`
                # (the pre-settings request shape) has to be folded in here or
                # the two resolutions would disagree about the tier.
                canonicalize_settings = dict(request.settings)
                if canonicalize_settings.get("target_vertices") is None and request.target_vertices:
                    canonicalize_settings["target_vertices"] = request.target_vertices
                canonicalize_run = run_store.get(job.run_id) if job.run_id else None
                canonicalize_view_images = _resolve_conditioning_view_paths(source, canonicalize_run)
                metadata.update(await asyncio.to_thread(
                    native.run_canonicalize,
                    Path(source.path),
                    output,
                    canonicalize_target_vertices,
                    canonicalize_views,
                    native.DEFAULT_CANONICALIZE_RENDER_SIZE,
                    lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                    method=canonicalize_method,
                    # Free-form passthrough for the palette-fidelity knobs
                    # (`skip_palette_transfer`, `palette_transfer_strength`,
                    # `palette_reference_artifact_id`) and the quality-tier
                    # knobs (`quality_tier`, `target_triangles`);
                    # native.run_canonicalize validates them, so this needs no
                    # new API surface.
                    settings=canonicalize_settings,
                    view_images=canonicalize_view_images or None,
                ))
                if (w := metadata.get("warnings")):
                    job.warnings.extend(w)
            elif request.kind == "paint-bake":
                metadata.update(await asyncio.to_thread(native.run_paint, Path(source.path), output, request.settings, lambda message: _job_log(job, message, min(78, max(18, job.progress + 3)))))
            elif request.kind == "project-views":
                project_views_run = run_store.get(job.run_id) if job.run_id else None
                project_views_images = _resolve_conditioning_view_paths(source, project_views_run)
                if len(project_views_images) < 2:
                    raise ValueError(
                        "Projecting conditioning views onto the canonical texture requires "
                        f"at least 2 resolvable views; found {sorted(project_views_images)}"
                    )
                metadata["conditioning_views_resolved"] = sorted(project_views_images)
                metadata.update(await asyncio.to_thread(
                    native.run_project_views,
                    Path(source.path),
                    output,
                    project_views_images,
                    request.settings,
                    lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                ))
            elif request.kind == "rebuild-views":
                rebuild_views_run = run_store.get(job.run_id) if job.run_id else None
                rebuild_views_images = _select_rebuild_view_references(
                    _resolve_conditioning_view_paths(source, rebuild_views_run),
                    request.settings,
                )
                metadata["references"] = {role: str(path) for role, path in rebuild_views_images.items()}
                metadata["reference_roles"] = list(rebuild_views_images)
                metadata.update(await asyncio.to_thread(
                    native.run_rebuild_views,
                    Path(source.path),
                    output,
                    rebuild_views_images,
                    request.settings,
                    lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                ))
                # The Pixal3D rebuild the deliverable was painted over, kept as
                # its own artifact so it can be inspected next to the overlay.
                pixal_mv_glb = Path(str(metadata["pixal_mv_glb"]))
                if pixal_mv_glb.is_file():
                    metadata["pixal_mv_artifact_id"] = save_artifact(
                        pixal_mv_glb,
                        name=f"Pixal3D mv {job.id}.glb",
                        kind="glb",
                        metadata={"stage": "rebuild-views", "role": "pixal-mv", "for_job_id": job.id},
                    ).id
            elif request.kind == "repair-weights":
                metadata.update(await asyncio.to_thread(native.run_repair_weights, Path(source.path), output, lambda message: _job_log(job, message, min(94, max(18, job.progress + 4)))))
            elif request.kind == "rig-transfer":
                donor_id = str(request.settings.get("donor_artifact_id") or "")
                if not donor_id:
                    raise ValueError("Rig transfer requires donor_artifact_id")
                donor = read_artifact(donor_id)
                if donor.kind != "glb":
                    raise ValueError("Rig transfer donor must be a GLB artifact")
                metadata["donor_artifact_id"] = donor.id
                metadata.update(await asyncio.to_thread(native.run_rig_transfer, Path(source.path), Path(donor.path), output, lambda message: _job_log(job, message, min(94, max(18, job.progress + 4)))))
            elif request.kind == "target-proxy-rig":
                curated_motion = str(request.settings.get("curated_motion") or "").strip()
                donor_id = str(request.settings.get("donor_artifact_id") or "")
                attach_sword = bool(request.settings.get("attach_sword", True))
                deformation_surface = select_deformation_surface(str(
                    request.settings.get("deformation_surface") or "visible-original"
                ))
                if curated_motion and donor_id:
                    raise ValueError(
                        "Target-proxy rig cannot specify both curated_motion and donor_artifact_id"
                    )
                if curated_motion == "grounded-sword-v1":
                    curated_donor = native.curated_sword_donor()
                    donor_path = curated_donor.path
                    action_name = curated_donor.action_name
                    metadata.update({
                        "motion_source": "curated-grounded-sword-v1",
                        "donor_sha256": curated_donor.sha256,
                        "donor_path": str(curated_donor.path.relative_to(native.ROOT)),
                    })
                else:
                    if curated_motion:
                        raise ValueError(
                            f"Unsupported target-proxy curated_motion: {curated_motion}"
                        )
                    if not donor_id:
                        raise ValueError("Target-proxy rig requires donor_artifact_id")
                    donor = read_artifact(donor_id)
                    if donor.kind != "glb":
                        raise ValueError("Target-proxy rig donor must be a GLB artifact")
                    donor_path = Path(donor.path)
                    action_name = str(request.settings.get("action_name") or "Sword_Attack")
                    metadata["donor_artifact_id"] = donor.id
                metadata.update(await asyncio.to_thread(
                    native.run_target_proxy_rig,
                    Path(source.path),
                    donor_path,
                    output,
                    lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                    action_name=action_name,
                    attach_sword=attach_sword,
                    deformation_surface=deformation_surface,
                ))
            elif request.kind == "retarget-motion":
                motion_id = str(request.settings.get("motion_artifact_id") or "")
                if not motion_id:
                    raise ValueError("Motion retarget requires motion_artifact_id")
                motion = read_artifact(motion_id)
                if motion.kind != "bvh":
                    raise ValueError("Motion retarget requires a BVH motion artifact")
                metadata["motion_artifact_id"] = motion.id
                certificate = source.metadata.get("quality_decisions", {}).get("rig-certificate", {})
                target_bones = certificate.get("resolved_roles", {})
                # The clip actually selected in the Animate panel (App.tsx's
                # attachCuratedMotion -> curatedActionLabel) -- this is what
                # Blender bakes as the output's own action name, so the
                # viewport's clip dropdown reads correctly instead of always
                # showing the old hardcoded "Sword Attack".
                retarget_animation_label = str(request.settings.get("animation") or "Sword_Attack")
                metadata.update(await asyncio.to_thread(
                    native.run_bvh_retarget,
                    Path(source.path),
                    Path(motion.path),
                    output,
                    target_bones=target_bones,
                    animation=retarget_animation_label,
                    progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                ))
            elif request.kind == "apply-clip":
                # N5: the payoff of N2 (fixed skeleton contract) + N3 (rigs
                # conform to the template) + N4 (clips pre-ported onto that
                # same template) -- direct action-copy by exact bone name,
                # no per-asset retarget math. Validated already at request
                # time (source certified + template-conformant, clip exists
                # in the family manifest); re-resolved here too since a job
                # can be re-run standalone.
                apply_clip_family = str(request.settings.get("family") or "biped")
                apply_clip_name = str(request.settings.get("clip") or "").strip()
                if not apply_clip_name:
                    raise ValueError("Direct clip attach requires settings.clip")
                clip_entry = find_clip(apply_clip_family, apply_clip_name)
                if clip_entry is None:
                    raise ValueError(
                        f"Clip {apply_clip_name!r} is not in the {apply_clip_family!r} family clip store"
                    )
                metadata["clip"] = apply_clip_name
                metadata["family"] = apply_clip_family
                metadata["mechanism"] = "direct-attach"
                metadata.update(await asyncio.to_thread(
                    native.run_apply_clip,
                    Path(source.path),
                    output,
                    family=apply_clip_family,
                    clip_name=apply_clip_name,
                    progress=lambda message: _job_log(job, message, min(94, max(18, job.progress + 4))),
                ))
            else:
                raise ValueError(f"Unsupported job kind: {request.kind}")
        else:
            raise ValueError(f"{request.kind} requires a source artifact")

        if job.status == "cancelled":
            output.unlink(missing_ok=True)
            return
        _job_log(job, f"Validating {'BVH motion' if is_motion else 'GLB artifact'}", 82)
        if not output.exists() or output.stat().st_size < 64:
            raise ValueError(f"Adapter did not produce a valid {'BVH' if is_motion else 'GLB'}")
        if not is_motion and not is_image:
            _job_log(job, "Measuring mesh quality", 96)
            metadata["mesh_quality"] = await asyncio.to_thread(analyze_path, output)
            metadata["quality_decisions"] = {
                "rig": evaluate_for_rigging(metadata["mesh_quality"]),
            }
        if request.kind == "canonicalize":
            # S2's mandatory contract gate (docs/PIPELINE_POLICY.md). Uses
            # the canonicalize pipeline's own component/watertight/non-
            # manifold measurement on the OBJ that was actually repaired
            # and baked (native.run_canonicalize's last attempt), not a
            # fresh re-measurement of the exported GLB -- the GLB's own
            # "components" can differ slightly after export/floor-normalize
            # round-tripping, and the gate must judge what was actually
            # iterated on.
            # The shared S2 gate (canonicalize_policy.decide_canonicalize):
            # the engine's `clean` stage makes the same decision.
            canonicalize_decision = decide_canonicalize(metadata, request.settings.get("family_band"))
            metadata["quality_decisions"]["canonicalize"] = canonicalize_decision
            job.promoted_output = canonicalize_decision["status"] == "ready"
            if not job.promoted_output:
                job.warnings.append(
                    f"Canonicalize candidate retained: {canonicalize_decision['status']} "
                    f"({len(canonicalize_decision['findings'])} finding(s))"
                )
            # `native.run_canonicalize` ran its own S2 view-projection pass
            # in place when it was given resolvable conditioning views; carry
            # its verdict alongside the canonicalize gate's rather than
            # letting it change canonicalize's own promotion outcome.
            canonicalize_view_projection_report = metadata.get("view_projection")
            if isinstance(canonicalize_view_projection_report, dict) and "skipped" not in canonicalize_view_projection_report:
                metadata["quality_decisions"]["project_views"] = evaluate_projection(
                    canonicalize_view_projection_report
                )
        if request.kind == "project-views":
            project_views_decision = evaluate_projection(metadata)
            metadata["quality_decisions"]["project_views"] = project_views_decision
            job.promoted_output = project_views_decision["status"] in {"green", "needs_review"}
            if not job.promoted_output:
                job.warnings.append(
                    f"Projection candidate retained: {project_views_decision['status']} "
                    f"({len(project_views_decision['findings'])} finding(s))"
                )
        if request.kind in SOURCE_FIDELITY_STAGES and source is not None and not is_motion and not is_image:
            await asyncio.to_thread(_stamp_source_fidelity, job, source.id, output, metadata, request.settings)
        if request.kind in {"auto-rig", "rig-adjust", "repair-weights", "rig-transfer", "target-proxy-rig", "retarget-motion", "apply-clip"} or is_slither:
            preferred_joint_names = (
                target_bones.values() if request.kind == "retarget-motion" else None
            )
            rig_report = await asyncio.to_thread(
                analyze_rig_path,
                output,
                preferred_joint_names,
            )
            certificate = evaluate_rig_certificate(rig_report)
            metadata["rig_quality"] = rig_report
            metadata["quality_decisions"]["rig-certificate"] = certificate
            if is_slither:
                required_animation = "Slither"
            elif request.kind == "retarget-motion":
                # Must track whatever clip was actually baked onto the
                # output above (retarget_animation_label) -- this used to be
                # hardcoded to BATCH_ONE_ATTACK ("Sword Attack") back when
                # that was the only curated clip, which silently blocked
                # promotion for every other clip once the label plumbing
                # started baking the real action name instead of always
                # lying and calling it "Sword_Attack".
                required_animation = request.settings.get("animation") or BATCH_ONE_ATTACK
            elif request.kind == "apply-clip":
                # The baked output's action is named exactly the clip's
                # manifest clip_name (blender_worker.py's apply_clip op
                # assigns the store's own action unrenamed -- direct-attach
                # has no reason to rename it the way retarget-motion's bake
                # does), so that is exactly what the animation-review gate
                # must look for.
                required_animation = request.settings.get("clip") or BATCH_ONE_ATTACK
            else:
                required_animation = BATCH_ONE_ATTACK
            metadata["quality_decisions"]["animation-review"] = evaluate_for_animation(
                rig_report,
                required_animation=required_animation,
            )
            animation_review = metadata["quality_decisions"]["animation-review"]
            deformation_path = output.with_suffix(".deformation.json")
            try:
                deformation_report = await asyncio.to_thread(
                    native.run_deformation_diagnostics,
                    output,
                    deformation_path,
                    lambda message: _job_log(
                        job,
                        message,
                        min(98, max(96, job.progress + 1)),
                    ),
                )
            finally:
                deformation_path.unlink(missing_ok=True)
            deformation_report = persist_deformation_review_artifacts(deformation_report)
            deformation_review = evaluate_deformation_review(deformation_report)
            metadata["deformation_review"] = deformation_report
            metadata["quality_decisions"]["deformation-review"] = deformation_review
            job.promoted_output = rig_candidate_is_promotable(
                metadata["quality_decisions"]
            )
            if request.kind == "target-proxy-rig":
                job.promoted_output = (
                    job.promoted_output
                    and deformation_surface_can_promote(deformation_surface)
                )
            if request.kind in {"auto-rig", "rig-adjust"}:
                # a fresh rig has no animation yet; the rig certificate is the
                # meaningful promotion bar here — the Animate stage attaches
                # motion and the full four-gate promotion applies to that
                certificate_ready = (
                    metadata["quality_decisions"]
                    .get("rig-certificate", {})
                    .get("status")
                    == "ready"
                )
                job.promoted_output = certificate_ready
                if not certificate_ready:
                    job.warnings.append(
                        "Auto-rig candidate retained: rig certificate did not pass"
                    )
            if request.kind in {"auto-rig", "rig-adjust"}:
                # Advisory visual evidence, not a promotion gate: a sheet
                # that fails to render must not fail the job that just
                # produced a valid rig -- it only means the human reviewer
                # loses the picture, not the rig.
                try:
                    sheet_report = await asyncio.to_thread(
                        native.run_skeleton_sheet,
                        output,
                        lambda message: _job_log(
                            job, message, min(98, max(96, job.progress + 1))
                        ),
                    )
                    metadata["skeleton_review"] = persist_skeleton_review_artifacts(sheet_report)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    job.warnings.append(f"Skeleton contact sheet could not be rendered: {exc}")
            if not job.promoted_output:
                job.warnings.append(
                    f"{request.kind.title()} candidate retained until its rig, "
                    "animation, and deformation checks pass"
                )
        input_quality = source.metadata.get("mesh_quality") if source is not None else None
        if isinstance(input_quality, dict) and request.kind not in {"repair-weights", "rig-transfer", "target-proxy-rig", "generate-motion", "retarget-motion", "apply-clip", "auto-rig", "rig-adjust", "canonicalize", "project-views", "rebuild-views"}:
            promotion = assess_mesh_promotion(input_quality, metadata["mesh_quality"])
            metadata["promotion_assessment"] = promotion
            job.promoted_output = bool(promotion["promote"])
            if not job.promoted_output:
                job.warnings.append(f"Candidate retained without promotion: {', '.join(promotion['reasons'])}")
        # PIPELINE_POLICY.md evaluation, immediately after every quality gate
        # above has attached its decision (bead x8qhy). Advisory: it reads the
        # gates' output, it never changes promotion or fails the job.
        artifact_kind = "image" if is_image else ("bvh" if is_motion else "glb")
        try:
            policy_block = _evaluate_job_policy(job, metadata, artifact_kind)
        except Exception as exc:
            policy_block = None
            _job_log(job, f"Pipeline policy evaluation skipped: {exc}")
        if policy_block is not None:
            metadata["pipeline_policy"] = policy_block
            job.pipeline_policy = policy_block
            _job_log(
                job,
                "Pipeline policy: next action "
                f"{policy_block['next_action']['kind']}"
                f"{' at ' + policy_block['next_action']['stage'] if policy_block['next_action']['stage'] else ''}"
                f" ({len(policy_block['not_evaluable'])} check(s) not evaluable)",
                job.progress,
            )
        if job.status == "cancelled":
            output.unlink(missing_ok=True)
            return
        if prebuilt_artifact is not None:
            # `create-tpose-set` already saved this file as the set's front,
            # because the three views it fans off it carry the front's
            # artifact id as their provenance. Stamp the metadata the shared
            # tail accumulated afterwards (the set block, the consistency
            # roll-up, the policy evaluation) rather than writing a second
            # artifact for the same image.
            _stamp_artifact_metadata(prebuilt_artifact.id, metadata)
            artifact = read_artifact(prebuilt_artifact.id)
        else:
            artifact = save_artifact(
                output,
                name=(
                    f"Synthesized {metadata.get('synthesized_view', 'view')} view{output_suffix}"
                    if request.kind == "synthesize-views"
                    else f"Prepared image{output_suffix}"
                    if is_image
                    else f"{request.kind.title()} result{'.bvh' if is_motion else '.glb'}"
                ),
                kind="image" if is_image else ("bvh" if is_motion else "glb"),
                metadata=metadata,
            )
        output.unlink(missing_ok=True)
        job.output_artifact_id = artifact.id
        job.status = "completed"
        _job_log(job, "Ready to pass to the next studio tab", 100)
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "Job cancelled"
        _job_log(job, job.error)
    except Exception as exc:
        if job.status == "cancelled":
            _job_log(job, "Cancelled")
            return
        job.status = "failed"
        job.error = str(exc)
        _job_log(job, f"Failed: {exc}")
    finally:
        if model_name:
            # Safety net for every path that leaves this job "cancelled": the
            # explicit /cancel endpoint already tries to kill the worker
            # directly, but a job can also end up here via asyncio.CancelledError
            # (e.g. the request/connection backing this BackgroundTask going
            # away) without that endpoint ever running -- and asyncio task
            # cancellation does NOT stop the background thread doing the
            # actual subprocess.wait(), so without this the worker just keeps
            # running, orphaned, with the GPU lock about to be released right
            # out from under it (bead xckqr).
            if job.status == "cancelled" and model_manager.active_job_id == job.id:
                killed = native.cancel_active_process(progress=lambda message: _job_log(job, message))
                _job_log(
                    job,
                    "Worker process confirmed terminated"
                    if killed
                    else "No worker process was found still tracked as active for this job",
                )
            await model_manager.release()
        if job.run_id and job.attempt_id and job.status in {"completed", "failed", "cancelled"}:
            try:
                run_store.finish_attempt(
                    job.run_id,
                    job.attempt_id,
                    job.status,
                    output_artifact_id=job.output_artifact_id,
                    error=job.error,
                    logs=job.logs,
                    warnings=job.warnings,
                    promote_output=job.promoted_output,
                )
            except KeyError:
                _job_log(job, "Run history association no longer exists")
            except ValueError as exc:
                _job_log(job, f"Run history already finalized: {exc}")


# ---------------------------------------------------------------------------
# PIPELINE_POLICY.md evaluation (bead x8qhy)
#
# Advisory only, by design: nothing below changes `job.promoted_output`,
# blocks a job, or remediates anything. It assembles the artifacts a run
# actually produced into a `pipeline_policy.PipelineRun` and records what
# the contract says about it.
# ---------------------------------------------------------------------------

# A run's asset lineage is short (image -> generate -> canonicalize -> rig ->
# clip); the bound stops a corrupted `source_artifact_id` cycle from walking
# the whole artifact store.
POLICY_LINEAGE_DEPTH = 24


def _policy_artifact_record(
    artifact_id: str, *, on_lineage: bool = False, produced_in_run: bool = False
) -> pipeline_policy.ArtifactRecord | None:
    """Read one artifact's stored metadata. Returns None (rather than
    raising) for an artifact that has been deleted or whose sidecar is
    unreadable -- a partially-cleaned run must still evaluate."""
    metadata_path = ARTIFACTS / f"{artifact_id}.json"
    if not metadata_path.exists():
        return None
    try:
        payload = _read_json(metadata_path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return pipeline_policy.ArtifactRecord(
        artifact_id=str(payload.get("id") or artifact_id),
        metadata=payload.get("metadata") or {},
        kind=payload.get("kind"),
        on_lineage=on_lineage,
        produced_in_run=produced_in_run,
    )


def _run_produced_artifact_ids(run: Any) -> set[str]:
    """The artifacts THIS run produced -- one per completed attempt.

    Deliberately excludes `run.source_artifact_id` and everything upstream
    of it: inherited lineage. A run that starts from an already-rigged
    upload holds two rigs without ever having re-run the rigging stage, and
    `pipeline_policy.detect_stale_stages` must not read that as a fix that
    invalidates downstream work.
    """
    return {
        attempt.output_artifact_id
        for attempt in run.attempts
        if attempt.output_artifact_id
    }


def _policy_lineage_records(
    terminal_artifact_id: str | None,
    *,
    pending: pipeline_policy.ArtifactRecord | None = None,
    produced_ids: set[str] | None = None,
) -> list[pipeline_policy.ArtifactRecord]:
    """The `source_artifact_id` chain ending at `terminal_artifact_id`,
    oldest first. `pending` is an artifact this job is about to save (it has
    no id and no sidecar yet) and, when given, is the chain's newest link.
    `produced_ids` names the artifacts the run produced itself, so records
    on the inherited part of the chain are not mistaken for its own work."""
    produced_ids = produced_ids or set()
    chain: list[pipeline_policy.ArtifactRecord] = []
    seen: set[str] = set()
    current = terminal_artifact_id
    if pending is not None:
        chain.append(pending)
        current = (pending.metadata or {}).get("source_artifact_id")
    while current and current not in seen and len(chain) < POLICY_LINEAGE_DEPTH:
        seen.add(current)
        record = _policy_artifact_record(
            current, on_lineage=True, produced_in_run=current in produced_ids
        )
        if record is None:
            break
        chain.append(record)
        current = (record.metadata or {}).get("source_artifact_id")
    chain.reverse()
    return chain


def _policy_run_records(
    run: Any, *, pending: pipeline_policy.ArtifactRecord | None = None
) -> list[pipeline_policy.ArtifactRecord]:
    """Every artifact this run can be evaluated from, oldest first.

    The terminal artifact's own lineage comes first -- that is the coherent
    journey of the asset the run currently points at. The run's other
    completed outputs follow, and only fill stages the lineage does not
    cover (`assemble_pipeline_run` never lets an off-lineage record displace
    an on-lineage one), which is what lets a run whose "current" artifact is
    a re-rig still be told about the clip attached in an earlier attempt.
    """
    produced_ids = _run_produced_artifact_ids(run)
    terminal = run.current_artifact_id or run.source_artifact_id
    records = _policy_lineage_records(
        terminal, pending=pending, produced_ids=produced_ids
    )
    seen = {record.artifact_id for record in records if record.artifact_id}

    source = _policy_artifact_record(run.source_artifact_id)
    extra: list[pipeline_policy.ArtifactRecord] = []
    if source is not None and source.artifact_id not in seen:
        seen.add(source.artifact_id)
        extra.append(source)
    for attempt in run.attempts:
        if attempt.status != "completed" or not attempt.output_artifact_id:
            continue
        if attempt.output_artifact_id in seen:
            continue
        record = _policy_artifact_record(
            attempt.output_artifact_id, produced_in_run=True
        )
        if record is None:
            continue
        seen.add(record.artifact_id)
        extra.append(record)
    return records + extra


def evaluate_run_policy(
    run: Any,
    *,
    pending: pipeline_policy.ArtifactRecord | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """The whole-run PIPELINE_POLICY.md evaluation for one run manifest."""
    records = _policy_run_records(run, pending=pending)
    policy_run, placements = pipeline_policy.assemble_pipeline_run(
        records, source=run.id,
    )
    # S6 is the operator-accept gate. An unreviewed run gets NO S6 artifact,
    # so the loop's advice is "advance to S6" (the accept is what's next)
    # rather than a manufactured rejection.
    if run.review_status != ReviewState.UNREVIEWED and run.reviewed_artifact_id:
        policy_run = policy_run.with_stage_artifact(
            pipeline_policy.s6_export_artifact(
                run.reviewed_artifact_id,
                run.review_status.value,
                run.reviewed_artifact_id,
            )
        )
    return pipeline_policy.summarize_pipeline_policy(
        policy_run, placements, records=records, compact=compact
    )


def evaluate_artifact_policy(
    pending: pipeline_policy.ArtifactRecord, *, compact: bool = False
) -> dict[str, Any]:
    """Evaluate a standalone job (no run association) from its artifact
    lineage alone."""
    records = _policy_lineage_records(None, pending=pending)
    policy_run, placements = pipeline_policy.assemble_pipeline_run(
        records, source=pending.artifact_id or "pending",
    )
    return pipeline_policy.summarize_pipeline_policy(
        policy_run, placements, records=records, compact=compact
    )


def _evaluate_job_policy(
    job: JobState, metadata: dict[str, Any], kind: str
) -> dict[str, Any] | None:
    """Evaluate the run this job just extended, using the not-yet-saved
    output artifact as the newest link. Advisory: any failure is logged and
    swallowed, because a policy evaluation must never fail a job that
    otherwise succeeded."""
    pending = pipeline_policy.ArtifactRecord(
        artifact_id=None, metadata=metadata, kind=kind, on_lineage=True,
        produced_in_run=True,
    )
    if job.run_id:
        return evaluate_run_policy(run_store.get(job.run_id), pending=pending, compact=True)
    return evaluate_artifact_policy(pending, compact=True)


@app.get("/api/system/capabilities")
async def capabilities() -> dict[str, Any]:
    gpu = _gpu_info()
    blender_path = native.find_blender()
    readiness = _model_readiness()
    diagnostics: list[str] = []
    if blender_path is None:
        diagnostics.append("Blender 4.2+ was not found; GLB conversion and texture baking are unavailable.")
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "environment": {"python_executable": sys.executable, "dedicated": str(ROOT / ".venv") in sys.executable},
        "gpu": gpu,
        "ram": _ram_info(),
        "blender": {"command": str(blender_path or "blender"), "available": blender_path is not None, "minimum_version": "4.2"},
        "models": {
            "trellis_gguf_ready": readiness["trellis_gguf_ready"],
            "instant_meshes_native": native.instant_meshes_exe() is not None,
            "trellis_refiner_ready": readiness["trellis_refiner_ready"],
            "kimodo_ready": readiness["kimodo_ready"],
            "unirig_ready": readiness["unirig_ready"],
            "rebuild_views_ready": readiness["rebuild_views_ready"],
        },
        "diagnostics": diagnostics,
        "downloads": {download_id: state.as_dict() for download_id, state in downloads.items() if state.status in {"queued", "running"}},
        "memory": {"loaded_model": model_manager.loaded, "loaded_at": model_manager.loaded_at},
        "output_directory": str(output_directory()),
    }


@app.post("/api/assets/upload", response_model=Artifact)
async def upload_asset(file: UploadFile = File(...)) -> Artifact:
    name = _safe_name(file.filename or "upload")
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_UPLOADS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_UPLOADS))}")
    destination = UPLOADS / f"{uuid.uuid4().hex}-{name}"
    total = 0
    exceeded_limit = False
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                exceeded_limit = True
                break
            handle.write(chunk)
    if exceeded_limit:
        # On Windows the file cannot be removed from inside its active ``with``
        # block; close it first so a size rejection remains a clean 413 error.
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Upload exceeds 100 MB")
    return save_artifact(destination, name=name, kind="image" if suffix in {"png", "jpg", "jpeg", "webp"} else suffix, metadata={"source": "upload"})


@app.post("/api/runs")
async def create_run(request: RunCreateRequest) -> Any:
    source = read_artifact(request.source_artifact_id)
    return run_store.create(request.name, source.id, source.kind, request.subject_kind)


def _find_submission_receipt(submission_id: str) -> JobState | None:
    receipt = next((item for item in jobs.values() if item.submission_id == submission_id), None)
    if receipt is not None:
        return receipt
    for stored in JOBS_DIR.glob("*.json"):
        try:
            payload = _read_json(stored)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("submission_id") == submission_id:
            receipt = JobState(**payload)
            jobs[receipt.id] = receipt
            return receipt
    return None


def _validate_submission_receipt(
    receipt: JobState | None,
    request_payload: dict[str, Any],
) -> Any | None:
    if receipt is None:
        return None
    if receipt.request_payload != request_payload:
        raise HTTPException(status_code=409, detail="submission_id was already used for another operation")
    if receipt.status == "completed" and receipt.run_id:
        return run_store.get(receipt.run_id)
    return None


@app.post("/api/runs/record-artifact")
async def record_artifact_stage(request: ArtifactStageRequest) -> Any:
    lock = record_submission_locks.setdefault(request.submission_id, asyncio.Lock())
    async with lock:
        return await _record_artifact_stage(request)


async def _record_artifact_stage(request: ArtifactStageRequest) -> Any:
    request_payload = request.model_dump(mode="json")
    with FileLock(JOBS_DIR / ".submissions.lock"):
        receipt = _find_submission_receipt(request.submission_id)
        completed = _validate_submission_receipt(receipt, request_payload)
        if completed is not None:
            return completed

    source = read_artifact(request.input_artifact_id)
    output = read_artifact(request.output_artifact_id)
    if request.run_id:
        try:
            target_run = run_store.get(request.run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        run_artifact_ids = {
            target_run.source_artifact_id,
            target_run.current_artifact_id,
            *(
                artifact_id
                for attempt in target_run.attempts
                for artifact_id in (attempt.input_artifact_id, attempt.output_artifact_id)
            ),
        }
        if source.id not in run_artifact_ids:
            raise HTTPException(
                status_code=409,
                detail="Input artifact does not belong to the target run",
            )
    if is_rig_producing_stage(request.stage):
        try:
            output = await _attach_serialized_rig_intake_diagnostics(output)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not validate the rigged GLB: {exc}",
            ) from exc

    with FileLock(JOBS_DIR / ".submissions.lock"):
        receipt = _find_submission_receipt(request.submission_id)
        completed = _validate_submission_receipt(receipt, request_payload)
        if completed is not None:
            return completed
        if receipt is None:
            receipt = JobState(
                id=uuid.uuid4().hex,
                kind=request.stage,
                submission_id=request.submission_id,
                source_artifact_id=source.id,
                run_id=request.run_id,
                request_payload=request_payload,
            )
            jobs[receipt.id] = receipt
            _write_json(JOBS_DIR / f"{receipt.id}.json", receipt.as_dict())
        if receipt.run_id:
            try:
                run = run_store.get(receipt.run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Run not found") from exc
        else:
            run = run_store.create(request.run_name or f"{source.name} experiment", source.id, source.kind)
            receipt.run_id = run.id
            _write_json(JOBS_DIR / f"{receipt.id}.json", receipt.as_dict())
        try:
            if receipt.attempt_id:
                current = run_store.get(run.id)
                attempt = next(item for item in current.attempts if item.id == receipt.attempt_id)
            else:
                attempt = run_store.reserve_attempt(
                    run.id,
                    receipt.id,
                    request.stage,
                    source.id,
                    request.settings,
                )
                receipt.attempt_id = attempt.id
                _write_json(JOBS_DIR / f"{receipt.id}.json", receipt.as_dict())
            if attempt.status in {"queued", "running"}:
                promote_output = output.kind == "glb"
                if is_rig_producing_stage(request.stage):
                    promote_output = _rig_candidate_can_promote(
                        request.stage,
                        request.settings,
                        output,
                    )
                attempt_warnings = list(request.warnings)
                if is_rig_producing_stage(request.stage) and not promote_output:
                    attempt_warnings.append(
                        "Rig export retained until topology, rig, animation, "
                        "and deformation checks pass"
                    )
                updated = run_store.finish_attempt(
                    run.id,
                    attempt.id,
                    "completed",
                    output_artifact_id=output.id,
                    logs=request.logs,
                    warnings=attempt_warnings,
                    promote_output=promote_output,
                )
                receipt.promoted_output = promote_output
                receipt.warnings = attempt_warnings
            elif attempt.status == "completed" and attempt.output_artifact_id == output.id:
                updated = run_store.get(run.id)
                receipt.promoted_output = attempt.promoted
            else:
                raise ValueError(f"Attempt {attempt.id} is already {attempt.status}")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        receipt.status = "completed"
        receipt.progress = 100
        receipt.message = "Artifact recorded"
        receipt.error = None
        receipt.logs = request.logs
        if not receipt.warnings:
            receipt.warnings = request.warnings
        receipt.output_artifact_id = output.id
        _write_json(JOBS_DIR / f"{receipt.id}.json", receipt.as_dict())
        return updated


@app.get("/api/runs")
async def list_runs() -> Any:
    return [summarize_run(run) for run in run_store.list()]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Any:
    try:
        return run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.get("/api/runs/{run_id}/policy")
async def get_run_policy(run_id: str) -> Any:
    """The whole-run PIPELINE_POLICY.md evaluation (bead x8qhy).

    Read-only and advisory: per-stage contract status, the first stage whose
    contract is violated, the blame-table resolution of what to do next
    (advance / stop-at-stage / done), and every check that could not be
    evaluated because nothing in the pipeline measures its metric yet.
    """
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return {"run_id": run.id, "run_name": run.name, **evaluate_run_policy(run)}


# ---------------------------------------------------------------------------
# The bounded pipeline-loop driver (bead hlxij)
#
# The policy module can already say advance / stop-at-stage / done. This is
# the part that gives that verdict hands -- under three rules that are the
# entire safety argument, and are enforced here rather than trusted:
#
#  1. ONE explicit invocation performs at most ONE step. There is no daemon,
#     no background re-run, no chaining: the endpoint submits a single job
#     and returns. Advancing a whole run is N deliberate calls by a human or
#     a script somebody is watching.
#  2. It never runs alongside a live job (409). A run with a queued or
#     running attempt is already spending the GPU; a second submission would
#     race it for the model lock and for the run's own `current_artifact_id`.
#  3. It never picks creative content. A stage whose input is a choice about
#     what the asset IS (S0/S1's image and prompt, S5's clip, S6's accept)
#     returns `needs_input` with the options, and spends nothing.
#
# Determinism (PIPELINE_POLICY.md principle 3) is the fourth rule and the
# reason remediation is not simply "run the stage again with today's
# defaults": a remediation re-runs the blamed stage from its ORIGINAL input
# artifact with its ORIGINAL recorded settings and seed, so the re-run is
# the same experiment, not a new one. And every submitted job carries a
# `pipeline_step` provenance block naming the evaluation that asked for it.
# ---------------------------------------------------------------------------

# Settings keys that are really JobRequest fields; recorded on every attempt
# by `create_job`, so they must be lifted back out of recorded settings
# rather than passed through as opaque settings.
_JOB_REQUEST_SETTING_FIELDS = ("resolution", "seed", "target_faces", "target_vertices", "quality")

# Settings keys a step must never inherit from the attempt it is reproducing:
# they describe how THAT job was requested, not what it computed.
_NON_INHERITED_SETTINGS = (
    pipeline_policy.PIPELINE_STEP_SETTINGS_KEY,
    "quality_action",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_attempt_producing(run: Any, artifact_id: str | None) -> Any | None:
    """The run attempt whose output is `artifact_id` -- i.e. the job that
    actually produced a stage's artifact, and therefore the settings and
    input a faithful re-run has to reuse."""
    if not artifact_id:
        return None
    return next(
        (
            attempt
            for attempt in reversed(run.attempts)
            if attempt.output_artifact_id == artifact_id
        ),
        None,
    )


def _latest_attempt_of_kind(run: Any, kind: str) -> Any | None:
    """The most recent attempt of one job kind in this run, completed or
    not -- what an `advance` step copies its settings defaults from, so a
    run that was tuned by hand keeps its tuning."""
    return next(
        (attempt for attempt in reversed(run.attempts) if attempt.stage == kind), None
    )


def _step_source_and_settings(
    run: Any, stage_artifact_id: str | None
) -> tuple[str | None, dict[str, Any], str | None]:
    """`(input_artifact_id, recorded_settings, attempt_id)` for re-running the
    stage that produced `stage_artifact_id`.

    Prefers the run's own attempt record (authoritative: it holds the exact
    settings the job was submitted with). Falls back to the artifact's stored
    `source_artifact_id` for a stage whose artifact came from outside this
    run's attempt history -- lineage the run inherited rather than ran.
    """
    attempt = _run_attempt_producing(run, stage_artifact_id)
    if attempt is not None:
        return attempt.input_artifact_id, dict(attempt.settings or {}), attempt.id
    record = _policy_artifact_record(stage_artifact_id) if stage_artifact_id else None
    if record is None:
        return None, {}, None
    metadata = record.metadata or {}
    settings = metadata.get("settings")
    return (
        metadata.get("source_artifact_id"),
        dict(settings) if isinstance(settings, dict) else {},
        None,
    )


def _advance_source_artifact_id(
    run: Any, policy: dict[str, Any], stage: pipeline_policy.Stage
) -> str | None:
    """The artifact an `advance` into `stage` should be run FROM.

    Normally the run's current artifact -- the newest thing it made. The
    exception is re-entry into a stale stage: after an upstream
    remediation, the run's current artifact can still be the superseded
    downstream output (the old rig), and re-running the rigger from the old
    rig would rebuild exactly the work the remediation invalidated. In that
    case the input is the newest artifact upstream of `stage` that the
    policy does NOT consider stale -- the remediated artifact itself, which
    is the whole point of re-running downstream.
    """
    candidate = run.current_artifact_id or run.source_artifact_id
    stale_ids = {
        entry["artifact_id"]
        for entry in policy.get("stale_stages", [])
        if entry.get("artifact_id")
    }
    if candidate not in stale_ids:
        return candidate
    for entry in reversed(policy["stages"]):
        entry_stage = pipeline_policy.Stage(entry["stage"])
        if pipeline_policy.stage_index(entry_stage) >= pipeline_policy.stage_index(stage):
            continue
        if entry.get("stale") or not entry.get("artifact_id"):
            continue
        return str(entry["artifact_id"])
    return candidate


def _step_job_request(
    *,
    kind: str,
    source_artifact_id: str,
    run_id: str,
    submission_id: str,
    recorded_settings: dict[str, Any],
    provenance: dict[str, Any],
    extra_settings: dict[str, Any] | None = None,
) -> JobRequest:
    """Rebuild one stage's job request from what the run recorded.

    The recorded settings are an attempt's settings dict, which mixes real
    job settings with the five JobRequest fields `create_job` folds into it;
    this splits them back apart so seed/resolution/target_faces/quality are
    reproduced as the typed fields they are, not smuggled through settings
    where nothing would read them.
    """
    recorded = dict(recorded_settings)
    # Lifted out unconditionally (so they never linger in settings where
    # nothing reads them), but only re-supplied when a value was actually
    # recorded -- a recorded `None` means "the field's own default", which
    # JobRequest already knows.
    lifted = {key: recorded.pop(key) for key in _JOB_REQUEST_SETTING_FIELDS if key in recorded}
    fields = {key: value for key, value in lifted.items() if value is not None}
    for key in _NON_INHERITED_SETTINGS:
        recorded.pop(key, None)
    settings = {
        **recorded,
        **(extra_settings or {}),
        pipeline_policy.PIPELINE_STEP_SETTINGS_KEY: provenance,
    }
    try:
        return JobRequest(
            kind=kind,
            submission_id=submission_id,
            source_artifact_id=source_artifact_id,
            run_id=run_id,
            settings=settings,
            **fields,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"The recorded settings for a {kind} step cannot be replayed: {exc}",
        ) from exc


def _manifest_clip_options(family: str) -> dict[str, Any]:
    """The family's clip manifest, as the options an S5 step needs named.

    A missing clip store is reported, not raised: "this family has no clips"
    is a real answer to "which clip should I attach?".
    """
    try:
        manifest = load_manifest(family)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {"family": family, "clips": [], "error": str(exc)}
    return {
        "family": family,
        "clips": [
            {
                "clip": clip.get("clip_name"),
                "display_name": clip.get("display_name"),
                "duration_seconds": clip.get("duration_seconds"),
            }
            for clip in manifest.get("clips", [])
        ],
    }


def _step_response(
    run: Any,
    policy: dict[str, Any],
    evaluated_at: str,
    action: str,
    why: str,
    *,
    stage: str | None = None,
    job: JobState | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One step's answer. Always the same five things -- what was done, at
    which stage, which job it cost (if any), why, and the evaluation that
    decided it -- so a caller never has to guess whether a job was spent."""
    return {
        "run_id": run.id,
        "run_name": run.name,
        "action": action,
        "stage": stage,
        "why": why,
        "job_id": job.id if job is not None else None,
        "job_kind": job.kind if job is not None else None,
        "attempt_id": job.attempt_id if job is not None else None,
        "policy_evaluated_at": evaluated_at,
        "policy": policy,
        **extra,
    }


@app.post("/api/runs/{run_id}/pipeline/step")
async def pipeline_step(
    run_id: str,
    background_tasks: BackgroundTasks,
    request: PipelineStepRequest | None = None,
) -> Any:
    """Perform exactly ONE step of the PIPELINE_POLICY.md loop (bead hlxij).

    Evaluates the run, then does at most one of:

    - `advanced`   -- submitted one job for the next stage that has not run,
                      or has run but is stale: an upstream remediation
                      replaced the artifact it was built on, so the policy
                      invalidated it and the loop re-enters the stage from
                      the remediated artifact.
    - `remediated` -- re-ran the blamed stage from its original input and
                      settings, tagged as a remediation and counted against
                      the per-stage cap.
    - `capped`     -- the blamed stage has already had `attempt_cap`
                      remediations. Nothing is submitted; the findings and
                      the stage are returned so a human can take over.
    - `needs_input`-- the next stage is a creative or human decision
                      (an image, a prompt, a clip, an operator accept).
                      Nothing is submitted; the options are returned.
    - `done`       -- S6 is green.

    It never chains: one call, at most one job, and never while a job for
    this run is already in flight (409).
    """
    request = request or PipelineStepRequest()
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc

    live = next(
        (attempt for attempt in run.attempts if attempt.status in {"queued", "running"}),
        None,
    )
    if live is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run already has a {live.status} {live.stage} attempt ({live.id}); "
                "a pipeline step never runs alongside a live job"
            ),
        )

    evaluated_at = _utc_now_iso()
    policy = evaluate_run_policy(run)
    next_action = policy["next_action"]
    submission_id = request.submission_id or f"pipeline-step-{uuid.uuid4().hex}"

    if next_action["kind"] == "done":
        return _step_response(
            run, policy, evaluated_at, "done",
            "Every stage through S6 evaluates clean; there is nothing left to run.",
        )

    stage = pipeline_policy.Stage(next_action["stage"])
    job_kind, input_required = pipeline_policy.stage_step_kind(stage)

    if job_kind is None:
        return _step_response(
            run, policy, evaluated_at, "needs_input",
            f"{stage.value} is a human decision, not a job: "
            f"this step is waiting on {input_required}.",
            stage=stage.value,
            input_required=input_required,
        )

    is_remediation = next_action["kind"] == "stop-at-stage"
    findings = tuple(next_action.get("findings", ()))
    finding_codes = tuple(item["code"] for item in findings)

    if is_remediation:
        stage_entry = next(
            (item for item in policy["stages"] if item["stage"] == stage.value), None
        )
        stage_artifact_id = stage_entry["artifact_id"] if stage_entry else None
        source_artifact_id, recorded_settings, remediates_attempt_id = (
            _step_source_and_settings(run, stage_artifact_id)
        )
        spent = pipeline_policy.count_stage_remediations(
            [attempt.settings for attempt in run.attempts], stage
        )
        if spent >= request.attempt_cap:
            return _step_response(
                run, policy, evaluated_at, "capped",
                f"{stage.value} has already been remediated {spent} time(s), which "
                f"meets this run's cap of {request.attempt_cap}. Nothing was submitted; "
                "the same job would spend the GPU on the same experiment again.",
                stage=stage.value,
                job_kind=job_kind,
                attempts_used=spent,
                attempt_cap=request.attempt_cap,
                findings=list(findings),
            )
    else:
        source_artifact_id = _advance_source_artifact_id(run, policy, stage)
        latest = _latest_attempt_of_kind(run, job_kind)
        recorded_settings = dict(latest.settings or {}) if latest is not None else {}
        remediates_attempt_id = None
        spent = 0

    if source_artifact_id is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No input artifact is recorded for a {job_kind} step at {stage.value}; "
                "this run cannot be advanced automatically"
            ),
        )

    extra_settings: dict[str, Any] = {}
    if stage in pipeline_policy.STAGE_CREATIVE_INPUT:
        creative_key = pipeline_policy.STAGE_CREATIVE_INPUT[stage]
        # Never let a creative choice ride in on inherited settings: the
        # advance path strips it, so an S5 advance can only ever run a clip
        # the CALLER named. A remediation is the one exception, and not an
        # exception to the rule at all -- re-running the exact clip the failed
        # attempt recorded is reproducing an experiment, not choosing one.
        inherited = recorded_settings.pop(creative_key, None)
        inherited_family = recorded_settings.pop("family", None)
        clip = request.clip or (inherited if is_remediation else None)
        family = request.family or (inherited_family if is_remediation else None)
        family = family or policy["family"]
        if family == "unknown":
            family = "biped"
        if not clip:
            return _step_response(
                run, policy, evaluated_at, "needs_input",
                f"{stage.value} attaches a clip, and choosing which motion a "
                "character performs is a creative decision this driver will not "
                "make on its own. Re-call this step with one of the clips below.",
                stage=stage.value,
                input_required=creative_key,
                **_manifest_clip_options(family),
            )
        extra_settings = {"clip": clip, "family": family}

    provenance = pipeline_policy.step_provenance(
        mode="remediation" if is_remediation else "advance",
        stage=stage,
        evaluated_at=evaluated_at,
        policy_action=next_action["kind"],
        attempt=spent + 1,
        attempt_cap=request.attempt_cap,
        source_artifact_id=source_artifact_id,
        triggering_stage=(
            pipeline_policy.Stage(next_action["triggering_stage"]) if is_remediation else None
        ),
        finding_codes=finding_codes,
        remediates_attempt_id=remediates_attempt_id,
    )
    job = await create_job(
        _step_job_request(
            kind=job_kind,
            source_artifact_id=source_artifact_id,
            run_id=run.id,
            submission_id=submission_id,
            recorded_settings=recorded_settings,
            provenance=provenance,
            extra_settings=extra_settings,
        ),
        background_tasks,
    )
    if is_remediation:
        why = (
            f"{stage.value}'s contract is violated "
            f"({', '.join(finding_codes) or 'no finding codes'}); re-ran it as "
            f"one {job_kind} job from its original input {source_artifact_id} with "
            f"its recorded settings. Remediation {spent + 1} of {request.attempt_cap}."
        )
    else:
        stale_entry = next(
            (
                entry
                for entry in policy.get("stale_stages", [])
                if entry["stage"] == stage.value
            ),
            None,
        )
        if stale_entry is not None:
            why = (
                f"{stage.value}'s recorded artifact ({stale_entry['artifact_id']}) is "
                f"stale: {stale_entry['reason']}. PIPELINE_POLICY.md re-runs downstream "
                f"from a remediated stage, so one {job_kind} job was submitted from "
                f"{source_artifact_id}."
            )
        else:
            why = (
                f"Every stage before {stage.value} evaluates clean enough to proceed and "
                f"{stage.value} has not run, so one {job_kind} job was submitted from the "
                f"run's current artifact {source_artifact_id}."
            )
    return _step_response(
        run, policy, evaluated_at, "remediated" if is_remediation else "advanced", why,
        stage=stage.value,
        job=job,
        attempts_used=spent + 1 if is_remediation else None,
        attempt_cap=request.attempt_cap if is_remediation else None,
        pipeline_step=provenance,
    )


@app.put("/api/runs/{run_id}/joint-recipe")
async def save_joint_recipe(run_id: str, recipe: JointRecipe) -> Any:
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    owned = {run.source_artifact_id}
    owned.update(
        attempt.output_artifact_id
        for attempt in run.attempts
        if attempt.output_artifact_id
    )
    if recipe.clean_mesh_artifact_id not in owned:
        raise HTTPException(status_code=409, detail="Joint recipe clean mesh does not belong to this run")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "joint-recipe.json"
        _write_json(path, recipe.model_dump(mode="json"))
        artifact = save_artifact(
            path,
            name="joint-recipe.json",
            kind="joint-recipe",
            metadata={
                "stage": "joint-recipe",
                "clean_mesh_artifact_id": recipe.clean_mesh_artifact_id,
                "editor_version": recipe.editor_version,
            },
        )
    attempt = run_store.start_attempt(
        run_id,
        "joint-recipe",
        recipe.clean_mesh_artifact_id,
        {"editor_version": recipe.editor_version},
    )
    updated = run_store.finish_attempt(
        run_id,
        attempt.id,
        "completed",
        output_artifact_id=artifact.id,
        promote_output=False,
    )
    return {"run": updated, "recipe": recipe, "artifact_id": artifact.id}


@app.get("/api/runs/{run_id}/joint-recipe")
async def get_joint_recipe(run_id: str) -> Any:
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    for attempt in reversed(run.attempts):
        if attempt.stage != "joint-recipe" or attempt.status != "completed" or not attempt.output_artifact_id:
            continue
        try:
            recipe = JointRecipe.model_validate(_read_json(_artifact_path(attempt.output_artifact_id)))
        except (HTTPException, OSError, ValueError):
            continue
        return {"recipe": recipe, "artifact_id": attempt.output_artifact_id}
    raise HTTPException(status_code=404, detail="Joint recipe not found")


@app.patch("/api/runs/{run_id}")
async def update_run(run_id: str, request: RunReviewRequest) -> Any:
    try:
        if request.subject_kind is not None:
            return run_store.set_subject_kind(run_id, request.subject_kind)
        assert request.review_status is not None
        assert request.review_note is not None
        assert request.artifact_id is not None
        if request.review_status == ReviewState.ACCEPTED:
            validate_final_review_candidate(read_artifact(request.artifact_id))
        return run_store.review(run_id, request.review_status, request.review_note, request.artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/checkpoints/{checkpoint}")
async def hand_back_checkpoint(run_id: str, checkpoint: Literal["skeleton"], request: CheckpointHandoffRequest) -> Any:
    """Record the operator approval of the exact rig artifact the review card showed.

    Deliberately identity-scoped, not pointer-scoped: the run's
    ``current_artifact_id`` is NOT required to already name
    ``request.artifact_id``. A fresh auto-rig whose certificate lands
    ``needs_review`` is retained rather than promoted, so it never becomes
    current -- and it is precisely that rig the human is being asked to
    judge. ``run_store.hand_back`` keeps the strict half of the check (a
    completed rig-stage output of THIS run) and pins the approved rig as the
    run's current artifact once the approval lands.
    """
    try:
        return run_store.hand_back(run_id, checkpoint, request.note, request.artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc




@app.get("/api/animations/curated")
async def list_curated_animations() -> Any:
    """Every playable clip in the curated donor library, for the Animate panel's picker."""
    return await asyncio.to_thread(curated_clips)


class CuratedBvhRequest(BaseModel):
    animation: str | None = None


@app.post("/api/animations/export-curated-bvh")
async def export_curated_bvh(request: CuratedBvhRequest) -> Any:
    """Export a curated library clip as a SOMA BVH motion artifact."""
    work = native.durable_data_path("tmp")
    work.mkdir(parents=True, exist_ok=True)
    stem = (request.animation or "curated").replace(" ", "_").replace("/", "-")
    output_bvh = work / f"curated-{stem}.bvh"
    result = await asyncio.to_thread(
        native.run_export_soma_bvh, output_bvh, request.animation, None
    )
    artifact = save_artifact(
        output_bvh,
        name=f"Curated motion - {request.animation or 'library'}.bvh",
        kind="bvh",
        metadata={
            "stage": "motion-source",
            "motion_source": "curated-library",
            **result,
        },
    )
    output_bvh.unlink(missing_ok=True)
    return artifact

@app.post("/api/runs/{run_id}/artifacts/{artifact_id}/promote")
async def promote_run_artifact(run_id: str, artifact_id: str, request: ArtifactPromotionRequest) -> Any:
    try:
        artifact = read_artifact(artifact_id)
        run = run_store.get(run_id)
        # Approving a T-pose SET is approving its front (bead 4vquq): the
        # checkpoint binds to one artifact, the set travels with it through
        # that artifact's `tpose_set` membership block, and the studio
        # presents the single approval as approving the quad.
        prepared_image = artifact.kind == "image" and any(
            attempt.stage in PREPARED_FRONT_STAGES
            and attempt.status == "completed"
            and attempt.output_artifact_id == artifact.id
            for attempt in run.attempts
        )
        if artifact.kind != "glb" and not prepared_image:
            raise ValueError(
                "Only playable GLBs or completed image-preparation candidates can become current"
            )
        return run_store.promote_artifact(run_id, artifact.id, request.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run or artifact not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/tpose-set/members/{role}")
async def replace_tpose_set_member(
    run_id: str, role: str, request: TposeSetMemberRequest,
) -> Any:
    """Swap one back/left/right member of the run's canonical T-pose set.

    The set is what a biped generate is conditioned on (bead 4vquq), so the
    two ways an operator changes a tile -- regenerating it, or uploading
    their own image over it -- have to land IN the set rather than only in
    the browser's slot state, or the next generate would silently condition
    on the member they just replaced.

    The set id never changes: replacing a tile edits the existing set, it
    does not fork a new one. The swap is recorded in the set's own
    `replacements` list so the conditioning of any past generate can still be
    reconstructed. The FRONT is deliberately not replaceable here -- the
    front is what the other three views were drawn from and gated against,
    so changing it means building a new set.
    """
    if role not in TPOSE_SET_VIEW_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"A T-pose set member must be one of {list(TPOSE_SET_VIEW_ROLES)}",
        )
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    front_id = latest_tpose_set_front(run)
    if not front_id:
        raise HTTPException(status_code=409, detail="This run has no canonical T-pose set")
    front = read_artifact(front_id)
    membership = tpose_set_membership(front)
    if membership is None:
        raise HTTPException(status_code=409, detail="This run's T-pose set is not registered")
    replacement = read_artifact(request.artifact_id)
    if replacement.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
        raise HTTPException(
            status_code=422, detail="A T-pose set member must be an image artifact",
        )
    members = {**(membership.get("members") or {}), role: replacement.id}
    failed_members = {
        key: value
        for key, value in (membership.get("failed_members") or {}).items()
        if key != role
    }
    replacements = list(membership.get("replacements") or [])
    replacements.append({
        "role": role,
        "artifact_id": replacement.id,
        "replaced_artifact_id": (membership.get("members") or {}).get(role),
        "reason": request.reason,
        "at": time.time(),
    })
    updated = {
        **membership,
        "members": members,
        "failed_members": failed_members,
        "complete": not failed_members,
        "replacements": replacements,
    }
    for member_role, member_id in members.items():
        _stamp_artifact_metadata(member_id, {"tpose_set": {**updated, "role": member_role}})
    return read_artifact(front.id)


@app.post("/api/runs/{run_id}/attempts")
async def create_run_attempt(run_id: str, request: AttemptCreateRequest) -> Any:
    try:
        return run_store.start_attempt(
            run_id,
            request.stage,
            request.input_artifact_id,
            request.settings,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/attempts/{attempt_id}/finish")
async def finish_run_attempt(run_id: str, attempt_id: str, request: AttemptFinishRequest) -> Any:
    lock_key = f"{run_id}:{attempt_id}"
    lock = attempt_finish_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        return await _finish_run_attempt(run_id, attempt_id, request)


async def _finish_run_attempt(run_id: str, attempt_id: str, request: AttemptFinishRequest) -> Any:
    try:
        run = run_store.get(run_id)
        attempt = next((item for item in run.attempts if item.id == attempt_id), None)
        if attempt is None:
            raise KeyError(attempt_id)
        if attempt.job_id is not None:
            raise HTTPException(status_code=409, detail="Backend-owned attempts must be finalized by their job")
        if attempt.status not in {"queued", "running"}:
            if (
                attempt.status == request.status
                and attempt.output_artifact_id == request.output_artifact_id
                and attempt.error == request.error
                and attempt.logs == request.logs
                and attempt.warnings == request.warnings
            ):
                return run
            raise ValueError(f"Attempt {attempt.id} is already {attempt.status}")
        output = (
            read_artifact(request.output_artifact_id)
            if request.status == "completed" and request.output_artifact_id
            else None
        )
        rig_completion = (
            request.status == "completed"
            and output is not None
            and is_rig_producing_stage(attempt.stage)
        )
        if rig_completion:
            try:
                output = await _attach_serialized_rig_intake_diagnostics(output)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                message = f"Could not validate the rigged GLB: {exc}"
                run_store.finish_attempt(
                    run_id,
                    attempt_id,
                    "failed",
                    output_artifact_id=request.output_artifact_id,
                    error=message,
                    logs=request.logs,
                    warnings=request.warnings,
                    promote_output=False,
                )
                raise HTTPException(status_code=422, detail=message) from exc
        promote_output = (
            request.status == "completed"
            and output is not None
            and output.kind == "glb"
        )
        if rig_completion:
            promote_output = _rig_candidate_can_promote(
                attempt.stage,
                attempt.settings,
                output,
            )
        return run_store.finish_attempt(
            run_id,
            attempt_id,
            request.status,
            output_artifact_id=request.output_artifact_id,
            error=request.error,
            logs=request.logs,
            warnings=request.warnings,
            promote_output=promote_output,
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run or attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs", response_model=JobState)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks) -> JobState:
    with FileLock(JOBS_DIR / ".submissions.lock"):
        if request.kind == "prepare-image":
            adapter = str(request.settings.get("adapter") or "upload").strip()
            if adapter not in {"upload", "comfyui"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image preparation adapter: {adapter}",
                )
            try:
                profile = resolve_preparation_profile(str(
                request.settings.get("profile") or "humanoid-t-pose-rig-v1"
                ))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            request.settings["adapter"] = adapter
            request.settings["profile"] = profile.id
        if request.kind == "create-tpose-set":
            # The composite is always the ComfyUI path -- there is no
            # "upload" pass-through for a T-pose set, because the set's whole
            # premise is that the front was GENERATED into the canonical pose.
            try:
                profile = resolve_preparation_profile(str(
                    request.settings.get("profile") or "humanoid-t-pose-rig-v1"
                ))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            views = request.settings.get("views")
            if views is None:
                views = list(TPOSE_SET_VIEW_ROLES)
            elif not isinstance(views, list) or not views:
                raise HTTPException(
                    status_code=400,
                    detail="create-tpose-set requires settings.views to be a non-empty list",
                )
            else:
                unknown = sorted({str(view) for view in views} - set(TPOSE_SET_VIEW_ROLES))
                if unknown:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown create-tpose-set views: {unknown}",
                    )
                if len(set(views)) != len(views):
                    raise HTTPException(
                        status_code=400,
                        detail="create-tpose-set views must not repeat",
                    )
            request.settings["adapter"] = "comfyui"
            request.settings["profile"] = profile.id
            request.settings["views"] = list(views)
        if request.kind == "target-proxy-rig":
            try:
                request.settings["deformation_surface"] = select_deformation_surface(str(
                    request.settings.get("deformation_surface") or "visible-original"
                ))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.kind == "synthesize-views":
            views = request.settings.get("views")
            if views is None:
                views = list(DEFAULT_SYNTHESIZED_VIEWS)
            elif not isinstance(views, list) or not views:
                raise HTTPException(
                    status_code=400,
                    detail="synthesize-views requires settings.views to be a non-empty list",
                )
            else:
                unknown = sorted({str(view) for view in views} - SYNTHESIZED_VIEW_LABELS)
                if unknown:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown synthesize-views views: {unknown}",
                    )
                if len(set(views)) != len(views):
                    raise HTTPException(
                        status_code=400,
                        detail="synthesize-views views must not repeat",
                    )
            request.settings["views"] = list(views)
            profile_family = str(request.settings.get("profile_family") or "tpose")
            if profile_family not in VIEW_SYNTHESIS_PROFILE_FAMILIES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown synthesize-views profile_family: {profile_family!r}; "
                        f"valid families: {sorted(VIEW_SYNTHESIS_PROFILE_FAMILIES)}"
                    ),
                )
            request.settings["profile_family"] = profile_family
        existing = next((item for item in jobs.values() if item.submission_id == request.submission_id), None)
        if existing is None:
            for stored in JOBS_DIR.glob("*.json"):
                payload = _read_json(stored)
                if payload.get("submission_id") == request.submission_id:
                    existing = JobState(**payload)
                    jobs[existing.id] = existing
                    break
        if existing is not None:
            request_payload = request.model_dump(mode="json")
            if existing.request_payload and existing.request_payload != request_payload:
                raise HTTPException(status_code=409, detail="submission_id was already used for another job")
            if not existing.request_payload and (
                existing.kind != request.kind or existing.source_artifact_id != request.source_artifact_id
            ):
                raise HTTPException(status_code=409, detail="submission_id was already used for another job")
            return existing
        # Advisory view-consistency findings raised by the pre-generate gate
        # below; attached to the job once it exists so the studio can show
        # them beside the run.
        pre_generate_warnings: list[str] = []
        # Bead 4vquq: once a biped run owns a canonical T-pose set, that set IS
        # the conditioning. The front is the gated T-pose front, never the raw
        # upload -- which stays in the run as the palette/identity reference
        # and nothing else. Resolved BEFORE the pre-generate consistency gate
        # below on purpose: the gate must measure the views against the SET
        # front, which means the set has to be substituted in first.
        if request.kind == "generate" and request.run_id and not request.settings.get("skip_tpose_set"):
            try:
                conditioning_run = run_store.get(request.run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Run not found") from exc
            set_front_id = (
                latest_tpose_set_front(conditioning_run)
                if conditioning_run.subject_kind in TPOSE_SET_SUBJECT_KINDS
                else None
            )
            if set_front_id:
                set_front = read_artifact(set_front_id)
                membership = tpose_set_membership(set_front)
                set_views = tpose_set_conditioning_views(set_front)
                supplied_front = (request.source_views or {}).get("front") or request.source_artifact_id
                reference_id = (membership or {}).get("reference_artifact_id")
                if supplied_front not in {None, set_front.id, reference_id}:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"This run has a canonical T-pose set, so its generate is "
                            f"conditioned on that set's front ({set_front.id}), not on "
                            f"{supplied_front}. Set settings.skip_tpose_set to true to "
                            f"generate from a different front anyway."
                        ),
                    )
                # An operator-uploaded (or per-tile regenerated) view REPLACES
                # the set member for its role -- the set supplies the default,
                # the operator's own image wins, and the swap is recorded on
                # the generate output so the conditioning can be audited later.
                resolved_views = dict(set_views)
                overrides: dict[str, str] = {}
                for label, artifact_id in (request.source_views or {}).items():
                    if label == "front" or not artifact_id:
                        continue
                    if resolved_views.get(label) != artifact_id:
                        overrides[label] = artifact_id
                    resolved_views[label] = artifact_id
                request.source_views = resolved_views
                request.source_artifact_id = set_front.id
                request.settings["tpose_set_id"] = (membership or {}).get("set_id")
                request.settings["tpose_set_overrides"] = overrides
                pre_generate_warnings.append(
                    "Conditioned on this run's canonical T-pose set; the uploaded "
                    "reference image is retained for palette and identity only."
                )
        if request.kind == "generate" and request.source_views:
            resolved_view_artifacts = {
                label: read_artifact(artifact_id)
                for label, artifact_id in request.source_views.items()
            }
            for label, view_artifact in resolved_view_artifacts.items():
                if view_artifact.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                    raise HTTPException(
                        status_code=422,
                        detail=f"source_views['{label}'] must be an image artifact",
                    )
            # JobRequest's model_validator already rejects a source_artifact_id
            # that disagrees with source_views["front"] before this handler ever
            # runs, so there is nothing left to reconcile here -- the run's
            # source artifact stays the front image; source_artifact_id keeps
            # doubling as the front view for every downstream consumer.
            request.source_artifact_id = resolved_view_artifacts["front"].id
            # Bead 8c9gf: refuse to spend a multi-view generate on a set that
            # contradicts its own front image. Only error-grade findings block;
            # warnings and refusals-to-score ride along as job warnings.
            if not request.settings.get("skip_view_consistency"):
                view_consistency_results, unreadable_views = _conditioning_view_consistency(
                    resolved_view_artifacts["front"], resolved_view_artifacts,
                )
                blocking = blocking_view_conflicts(view_consistency_results)
                if blocking:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            " ".join(blocking)
                            + " Set settings.skip_view_consistency to true to "
                            "generate with it anyway."
                        ),
                    )
                pre_generate_warnings.extend(view_consistency_warnings(view_consistency_results))
                pre_generate_warnings.extend(
                    f"{label} view: could not be read for the view-consistency check; "
                    f"generating without verifying it against the front image."
                    for label in unreadable_views
                )
        source = read_artifact(request.source_artifact_id) if request.source_artifact_id else None
        if request.kind == "prepare-image" and (
            source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}
        ):
            raise HTTPException(status_code=400, detail="Image preparation requires one source image")
        if request.kind == "synthesize-views" and (
            source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}
        ):
            raise HTTPException(status_code=400, detail="View synthesis requires one source image")
        if request.kind == "create-tpose-set" and (
            source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}
        ):
            raise HTTPException(
                status_code=400, detail="Creating a T-pose set requires one source image"
            )
        if request.kind == "target-proxy-rig":
            curated_motion = str(request.settings.get("curated_motion") or "").strip()
            donor_id = str(request.settings.get("donor_artifact_id") or "")
            if curated_motion and donor_id:
                raise HTTPException(
                    status_code=400,
                    detail="Target-proxy rig cannot specify both curated_motion and donor_artifact_id",
                )
        readiness = _model_readiness()
        if request.kind == "generate" and not (
            readiness["trellis_refiner_ready"] or readiness["trellis_gguf_ready"]
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "No geometry workflow is downloaded. Open Generate and download "
                    "one before starting inference."
                ),
            )
        if request.kind == "rebuild-views" and not readiness.get("rebuild_views_ready"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Rebuild from views needs the optional Pixal3D multi-view scripts, "
                    "which this installation does not include."
                ),
            )
        if request.kind == "retopologize" and not readiness["trellis_refiner_ready"]:
            raise HTTPException(status_code=409, detail="TRELLIS.2 FP8 Refine is not downloaded. Open Refine and download it before starting.")
        if request.kind == "retopo" and not readiness["instant_meshes_ready"]:
            raise HTTPException(status_code=409, detail="Instant Meshes is not downloaded. Open Retopology and download it before starting.")
        if request.kind == "canonicalize":
            # Density/views/method validation at submission time: an
            # out-of-range custom target (say 600,000 vertices, inside the
            # schema bound but past the gate's triangle ceiling) is a 400
            # here, not a job that 202s and then fails at execution.
            try:
                _parse_canonicalize_settings(request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if (
            request.kind == "canonicalize"
            and str(request.settings.get("method") or native.DEFAULT_CANONICALIZE_METHOD).strip().lower() == "instant-meshes"
            and not readiness["instant_meshes_ready"]
        ):
            raise HTTPException(status_code=409, detail="Instant Meshes is not downloaded. Canonicalize needs it for remeshing before starting.")
        # Chain-rigged creatures (serpents, tails -- see skeleton_naming.py's
        # chain profile) get a procedural Slither bake with Blender instead
        # of NVIDIA Kimodo's humanoid SOMA motion, and can never go through
        # SOMA retarget-motion at all (task F3).
        source_certificate = (
            (source.metadata.get("quality_decisions") or {}).get("rig-certificate", {})
            if source is not None
            else {}
        )
        is_chain_source = source_certificate.get("profile") == "chain"
        if request.kind == "generate-motion" and not is_chain_source and not readiness["kimodo_ready"]:
            raise HTTPException(status_code=409, detail="NVIDIA Kimodo is not installed in the managed tools directory.")
        if request.kind in {"target-proxy-rig", "generate-motion", "retarget-motion", "apply-clip"} and native.find_blender() is None:
            detail = "Blender 4.2+ is required for automated rigging." if request.kind == "target-proxy-rig" else "Blender 4.2+ is required to bake motion onto the character rig."
            raise HTTPException(status_code=409, detail=detail)
        if request.kind == "generate-motion" and (source is None or not (_has_certified_rig_quality(source) or _has_ready_rig_quality(source))):
            raise HTTPException(status_code=409, detail="Motion generation requires a GLB that passes every rig quality gate.")
        if request.kind == "retarget-motion":
            if source is None or source.kind != "glb" or not (_has_certified_rig_quality(source) or _has_ready_rig_quality(source)):
                raise HTTPException(status_code=409, detail="Motion retarget requires a GLB that passes every rig quality gate.")
            if is_chain_source:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Chain-rigged creatures (serpents, tails) cannot use SOMA "
                        "retarget-motion; use generate-motion for the Slither animation instead."
                    ),
                )
            motion_id = str(request.settings.get("motion_artifact_id") or "")
            if not motion_id:
                raise HTTPException(status_code=400, detail="Motion retarget requires motion_artifact_id.")
            motion = read_artifact(motion_id)
            if motion.kind != "bvh":
                raise HTTPException(status_code=409, detail="Motion retarget requires a BVH motion artifact.")
        if request.kind == "apply-clip":
            # N5: direct clip application. No BVH, no retarget math -- the
            # only requirements are a template-conformant certified rig and
            # a clip that exists in the requested family's manifest.
            if source is None or source.kind != "glb":
                raise HTTPException(status_code=400, detail="Direct clip attach requires a rigged GLB artifact.")
            if not _is_template_conformant_rig(source):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Direct clip attach requires a rig with a ready rig certificate and "
                        "clean template conformance (build one with UniRig auto-rig)."
                    ),
                )
            apply_clip_family = str(request.settings.get("family") or "biped")
            apply_clip_name = str(request.settings.get("clip") or "").strip()
            if not apply_clip_name:
                raise HTTPException(status_code=400, detail="Direct clip attach requires settings.clip.")
            try:
                clip_entry = find_clip(apply_clip_family, apply_clip_name)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=f"Unknown clip family: {apply_clip_family!r}") from exc
            if clip_entry is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Clip {apply_clip_name!r} is not in the {apply_clip_family!r} family clip store.",
                )
        if request.kind != "generate" and not request.source_artifact_id:
            raise HTTPException(status_code=400, detail=f"{request.kind} requires source_artifact_id")
        if request.run_id:
            try:
                run = run_store.get(request.run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Run not found") from exc
        elif source:
            run = run_store.create(
                request.run_name or f"{source.name} experiment",
                source.id,
                source.kind,
                request.subject_kind or "auto",
            )
        else:
            raise HTTPException(status_code=400, detail="A source artifact is required to create a run")
        if request.kind == "synthesize-views":
            # t0z17 + 8c9gf: an approved T-pose candidate (this run's promoted
            # prepare-image output) is a far better synthesis reference than
            # the raw front photo -- it already holds the exact pose and the
            # reviewed outfit the back/left/right views need to match. Prefer
            # it unless the caller named an explicit source via
            # settings.source_artifact_id, and prefer it whether or not the
            # run's lineage has since advanced past it (see the function's
            # docstring for the live "witch" incident that rule fixes).
            selected_source = select_view_synthesis_source(
                run, str(request.source_artifact_id), request.settings,
            )
            request.settings["view_synthesis_source"] = selected_source.label
            request.settings["view_synthesis_source_reason"] = selected_source.reason
            resolved_source_id = selected_source.artifact_id
            if resolved_source_id != request.source_artifact_id:
                resolved_source = read_artifact(resolved_source_id)
                if resolved_source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                    raise HTTPException(
                        status_code=422,
                        detail="View synthesis source override must be an image artifact",
                    )
                request.source_artifact_id = resolved_source_id
                source = resolved_source
        # Subject-kind semantics: the general 2D->3D pipeline has an optional
        # biped-rigging path. Generate never requires any of this (the
        # direct-generate gate above already covers it).
        if request.kind == "prepare-image" and run.subject_kind not in {"auto", "biped"}:
            raise HTTPException(
                status_code=409,
                detail=(
                    "T-pose preparation applies to biped characters; set the "
                    "run's subject kind to biped to use it"
                ),
            )
        if (
            request.kind == "create-tpose-set"
            and run.subject_kind not in TPOSE_SET_SUBJECT_KINDS
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "A canonical T-pose set applies to biped characters; set the "
                    "run's subject kind to biped to use it (objects and creatures "
                    "keep the single-image flow)"
                ),
            )
        if request.kind == "auto-rig":
            if run.subject_kind == "object":
                raise HTTPException(
                    status_code=409,
                    detail="Rigging is not available for objects; set the subject kind to creature or biped",
                )
            profile_hint = {"biped": "humanoid", "creature": "chain"}.get(run.subject_kind)
            if profile_hint:
                # skeleton_naming does not read this yet; carried through job
                # settings and artifact metadata for the rig owner to consume.
                request.settings["profile_hint"] = profile_hint
        if request.kind in {"retarget-motion", "apply-clip"} or (request.kind == "generate-motion" and not is_chain_source):
            if not _biped_or_auto_humanoid(run.subject_kind, source_certificate):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "SOMA motion requires a biped run; set the run's subject "
                        "kind to biped to use retarget-motion, apply-clip, or "
                        "humanoid generate-motion"
                    ),
                )
        if request.kind == "generate-motion" and is_chain_source:
            if not _creature_or_auto_chain(run.subject_kind, source_certificate):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Slither motion generation requires a creature run; set "
                        "the run's subject kind to creature to use it"
                    ),
                )
        if request.kind == "generate" and run.source_artifact_kind == "image":
            # prepare is optional: an uploaded/generated image may be sent
            # straight to 3D. When the run DID prepare, require the approved
            # prepared candidate as the generate source.
            prepared_in_run = any(
                attempt.stage in PREPARED_FRONT_STAGES and attempt.status == "completed"
                for attempt in run.attempts
            )
            if prepared_in_run:
                approved_preparation = next(
                    (
                        attempt
                        for attempt in run.attempts
                        if attempt.stage in PREPARED_FRONT_STAGES
                        and attempt.status == "completed"
                        and attempt.promoted
                        and attempt.output_artifact_id == request.source_artifact_id
                    ),
                    None,
                )
                if approved_preparation is None or run.current_artifact_id != request.source_artifact_id:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Generate requires the run's explicitly approved prepared "
                            "image. Approve the prepared image on the Prepare step first."
                        ),
                    )
        if request.kind == "project-views":
            if source is None or source.kind != "glb":
                raise HTTPException(
                    status_code=400,
                    detail="Projecting conditioning views onto the canonical texture requires a GLB source artifact",
                )
            if native.find_blender() is None:
                raise HTTPException(
                    status_code=409,
                    detail="Blender 4.2+ is required to project conditioning views onto the canonical texture",
                )
            resolvable_project_views = _resolve_conditioning_view_paths(source, run)
            if len(resolvable_project_views) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Projecting conditioning views onto the canonical texture requires "
                        "at least 2 resolvable T-pose conditioning views; found "
                        f"{sorted(resolvable_project_views)}"
                    ),
                )
        if request.retry_of_attempt_id:
            parent = next((item for item in run.attempts if item.id == request.retry_of_attempt_id), None)
            if parent is None:
                raise HTTPException(status_code=404, detail="Retry parent attempt was not found in this run")
            if parent.status in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="A running attempt cannot be retried")
            if parent.job_id is None:
                raise HTTPException(status_code=409, detail="Only backend-owned attempts can be retried")
            if parent.stage != request.kind or parent.input_artifact_id != request.source_artifact_id:
                raise HTTPException(status_code=409, detail="Retry stage and input must match the parent attempt")
        job = JobState(
            id=uuid.uuid4().hex,
            kind=request.kind,
            submission_id=request.submission_id,
            source_artifact_id=request.source_artifact_id,
            run_id=run.id,
            retry_of_attempt_id=request.retry_of_attempt_id,
            request_payload=request.model_dump(mode="json"),
            warnings=list(pre_generate_warnings),
        )
        try:
            attempt_settings = {
                **request.settings,
                "resolution": request.resolution,
                "seed": request.seed,
                "target_faces": request.target_faces,
                "target_vertices": request.target_vertices,
                "quality": request.quality,
            }
            attempt = run_store.reserve_attempt(
                run.id,
                job.id,
                request.kind,
                request.source_artifact_id or run.current_artifact_id or run.source_artifact_id,
                attempt_settings,
                retry_of_attempt_id=request.retry_of_attempt_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        job.attempt_id = attempt.id
        jobs[job.id] = job
        _write_json(JOBS_DIR / f"{job.id}.json", job.as_dict())
    background_tasks.add_task(_execute_job, job, request)
    return job


@app.post("/api/runs/{run_id}/attempts/{attempt_id}/retry", response_model=JobState)
async def retry_run_attempt(
    run_id: str,
    attempt_id: str,
    request: JobRetryRequest,
    background_tasks: BackgroundTasks,
) -> JobState:
    try:
        run = run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    attempt = next((item for item in run.attempts if item.id == attempt_id), None)
    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt.job_id is None:
        raise HTTPException(status_code=409, detail="Only backend-owned attempts can be retried")
    if attempt.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="A running attempt cannot be retried")

    original_job = jobs.get(attempt.job_id)
    if original_job is None:
        receipt_path = JOBS_DIR / f"{attempt.job_id}.json"
        if not receipt_path.exists():
            raise HTTPException(status_code=409, detail="The original job receipt is unavailable")
        try:
            original_job = JobState(**_read_json(receipt_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail="The original job receipt is invalid") from exc
        jobs[original_job.id] = original_job
    if not original_job.request_payload:
        raise HTTPException(status_code=409, detail="The original job predates reproducible request capture")

    try:
        payload = JobRequest.model_validate(original_job.request_payload).model_dump(mode="json")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="The original job request cannot be replayed") from exc
    overrides = request.model_dump(mode="json", exclude_none=True, exclude={"submission_id", "settings"})
    payload.update(overrides)
    payload["settings"] = {**payload.get("settings", {}), **request.settings}
    payload.update(
        submission_id=request.submission_id,
        source_artifact_id=attempt.input_artifact_id,
        run_id=run.id,
        run_name=None,
        retry_of_attempt_id=attempt.id,
    )
    return await create_job(JobRequest.model_validate(payload), background_tasks)


@app.post("/api/runs/{run_id}/quality-actions/{action}", response_model=JobState)
async def execute_quality_action(
    run_id: str,
    action: str,
    request: JobRetryRequest,
    background_tasks: BackgroundTasks,
) -> JobState:
    if action in MANUAL_QUALITY_ACTIONS:
        raise HTTPException(status_code=409, detail=f"Quality action '{action}' requires manual or agent-driven editing")
    kind = AUTOMATED_QUALITY_ACTIONS.get(action)
    if kind is None:
        raise HTTPException(status_code=404, detail="Unknown quality action")
    try:
        run = run_store.get(run_id)
        current = read_artifact(run.current_artifact_id or run.source_artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run or current artifact not found") from exc

    decisions = current.metadata.get("quality_decisions", {})
    recommended = {
        recommended_action
        for decision in decisions.values()
        if isinstance(decision, dict)
        for recommended_action in decision.get("recommended_actions", [])
    }
    if action not in recommended:
        raise HTTPException(status_code=409, detail=f"Quality action '{action}' is not recommended for the current artifact")

    source = current
    settings = {**request.settings, "quality_action": action}
    if kind == "generate":
        source = read_artifact(run.source_artifact_id)
        if source.kind != "image":
            raise HTTPException(status_code=409, detail="Generation recovery requires the run's retained source image")
    elif kind == "retopologize":
        image = read_artifact(run.source_artifact_id)
        if image.kind != "image":
            raise HTTPException(status_code=409, detail="Reconstruction recovery requires the run's retained source image")
        settings["refine_image_artifact_id"] = image.id
    overrides = request.model_dump(mode="json", exclude_none=True, exclude={"submission_id", "settings"})
    return await create_job(
        JobRequest(
            kind=kind,
            submission_id=request.submission_id,
            source_artifact_id=source.id,
            run_id=run.id,
            settings=settings,
            **overrides,
        ),
        background_tasks,
    )


@app.get("/api/downloads")
async def list_downloads() -> list[dict[str, Any]]:
    return [state.as_dict() for state in downloads.values() if state.status in {"queued", "running"}]


@app.post("/api/models/{name}/download", response_model=DownloadState)
async def download_model(name: str, background_tasks: BackgroundTasks) -> DownloadState:
    if name not in {"trellis2", "instant-meshes", "trellis-refiner"}:
        raise HTTPException(status_code=404, detail="This tab has no downloadable model package")
    gpu = _gpu_info()
    readiness = _model_readiness()
    if name in {"trellis2", "trellis-refiner"}:
        ready = readiness["trellis_refiner_ready"]
    else:
        ready = readiness["instant_meshes_ready"]
    if ready:
        state = DownloadState(id=uuid.uuid4().hex, model=name, status="completed", progress=100, message="Already downloaded")
        return state
    existing = next((state for state in downloads.values() if state.model == name and state.status in {"queued", "running"}), None)
    if existing:
        return existing
    state = DownloadState(id=uuid.uuid4().hex, model=name)
    downloads[state.id] = state
    _write_json(DOWNLOADS_DIR / f"{state.id}.json", state.as_dict())
    background_tasks.add_task(_execute_download, state)
    return state


@app.get("/api/downloads/{download_id}", response_model=DownloadState)
async def get_download(download_id: str) -> DownloadState:
    state = downloads.get(download_id)
    if state is None:
        stored = DOWNLOADS_DIR / f"{download_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Download not found")
        state = DownloadState(**_read_json(stored))
        downloads[download_id] = state
    return state


@app.get("/api/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str) -> JobState:
    if job_id not in jobs:
        stored = JOBS_DIR / f"{job_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        jobs[job_id] = JobState(**_read_json(stored))
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel", response_model=JobState)
async def cancel_job(job_id: str) -> JobState:
    job = jobs.get(job_id)
    if job is None:
        stored = JOBS_DIR / f"{job_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        job = JobState(**_read_json(stored))
        jobs[job_id] = job
    if job.status not in {"queued", "running"}:
        return job
    job.status = "cancelled"
    job.error = "Cancelled by user"
    _job_log(job, "Cancelling task")
    if model_manager.active_job_id == job.id:
        # cancel_active_process() blocks (SIGTERM, a grace period, and a
        # possible SIGKILL escalation, each confirmed) -- run it off the
        # event loop so cancelling one job doesn't stall every other request.
        killed = await asyncio.to_thread(
            native.cancel_active_process, lambda message: _job_log(job, message)
        )
        _job_log(
            job,
            "Worker process confirmed terminated"
            if killed
            else "No worker process was found still tracked as active for this job",
        )
    return job


@app.post("/api/models/{name}/unload")
async def unload_model(name: str) -> dict[str, Any]:
    if model_manager.loaded and (name == "all" or name == model_manager.loaded):
        await model_manager.release()
    return {"ok": True, "loaded_model": model_manager.loaded}


@app.post("/api/system/clear-memory")
async def clear_memory() -> dict[str, Any]:
    if model_manager.lock.locked():
        return {"ok": False, "message": "A job is running; memory will be released when it completes.", "gpu": _gpu_info()}
    if model_manager.loaded:
        await model_manager.release()
    else:
        gc.collect()
    return {"ok": True, "message": "GPU cache and Python references cleared.", "gpu": _gpu_info()}


@app.post("/api/system/clear-ram")
async def clear_ram() -> dict[str, Any]:
    if model_manager.lock.locked():
        return {"ok": False, "message": "A job is running; RAM is released when it completes.", "ram": _ram_info()}
    gc.collect()
    return {"ok": True, "message": "AISmith idle Python memory cleared.", "ram": _ram_info()}


@app.post("/api/system/output-location/select")
async def select_output_location() -> dict[str, str]:
    """Open the native folder picker; the selected folder becomes future output storage."""
    def choose() -> str:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=str(output_directory()), title="Choose AISmith 3D output folder")
        root.destroy()
        return selected

    selected = await asyncio.to_thread(choose)
    if not selected:
        return {"path": str(output_directory()), "changed": "false"}
    return {"path": str(set_output_directory(Path(selected))), "changed": "true"}


@app.get("/api/assets/{artifact_id}", response_model=Artifact)
async def get_artifact(artifact_id: str) -> Artifact:
    return read_artifact(artifact_id)


@app.get("/api/assets/{artifact_id}/download")
async def download_artifact(artifact_id: str) -> FileResponse:
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    default_suffix = ".bvh" if artifact.kind == "bvh" else ".png" if artifact.kind == "image" else ".json" if artifact.kind == "deformation-review" else ".glb"
    media_type = "text/plain" if artifact.kind == "bvh" else "image/png" if artifact.kind == "image" else "application/json" if artifact.kind == "deformation-review" else "model/gltf-binary"
    return FileResponse(path, filename=artifact.name if Path(artifact.name).suffix else f"{artifact.name}{default_suffix}", media_type=media_type)


@app.get("/api/assets/{artifact_id}/details")
async def asset_details(artifact_id: str) -> dict[str, Any]:
    """Read actual mesh counts from generated and uploaded GLB artifacts."""
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    if path.suffix.lower() not in {".glb", ".gltf"}:
        raise HTTPException(status_code=415, detail="Model details are available for GLB and glTF assets")
    try:
        import trimesh

        scene = trimesh.load(path, force="scene", process=False)
        geometries = list(scene.geometry.values())
        materials = {
            id(getattr(getattr(mesh, "visual", None), "material", None))
            for mesh in geometries
            if getattr(getattr(mesh, "visual", None), "material", None) is not None
        }
        return {
            "vertices": sum(len(mesh.vertices) for mesh in geometries),
            "triangles": sum(len(mesh.faces) for mesh in geometries),
            "materials": len(materials),
            "meshes": len(geometries),
            "format": path.suffix.lstrip(".").upper(),
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not inspect this model: {exc}") from exc


@app.post("/api/assets/{artifact_id}/diagnose", response_model=Artifact)
async def diagnose_artifact(
    artifact_id: str,
    request: MeshDiagnosticsRequest,
) -> Artifact:
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    if path.suffix.lower() not in {".glb", ".gltf", ".obj"}:
        raise HTTPException(
            status_code=415,
            detail="Mesh diagnostics are available for GLB, glTF, and OBJ assets",
        )

    attempt = None
    if request.run_id:
        try:
            attempt = run_store.start_attempt(
                request.run_id,
                "mesh-diagnostics",
                artifact.id,
                {"schema_version": 1},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        report = await asyncio.to_thread(analyze_path, path)
        artifact.metadata["mesh_quality"] = report
        artifact.metadata.setdefault("quality_decisions", {})["rig"] = evaluate_for_rigging(report)
        _write_json(ARTIFACTS / f"{artifact.id}.json", artifact.model_dump())
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "completed",
                output_artifact_id=artifact.id,
                logs=[
                    f"Inspected {report['vertices']} vertices and {report['faces']} faces",
                    f"Found {report['components']} connected components",
                ],
            )
        return artifact
    except Exception as exc:
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "failed",
                error=str(exc),
                logs=["Mesh diagnostics failed"],
            )
        raise HTTPException(
            status_code=422,
            detail=f"Could not diagnose this model: {exc}",
        ) from exc


@app.post("/api/assets/{artifact_id}/diagnose-rig", response_model=Artifact)
async def diagnose_rig_artifact(
    artifact_id: str,
    request: MeshDiagnosticsRequest,
) -> Artifact:
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    if path.suffix.lower() != ".glb":
        raise HTTPException(
            status_code=415,
            detail="Rig diagnostics currently require a binary GLB asset",
        )

    attempt = None
    if request.run_id:
        try:
            attempt = run_store.start_attempt(
                request.run_id,
                "rig-diagnostics",
                artifact.id,
                {"schema_version": 1},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        artifact = await asyncio.to_thread(_attach_rig_diagnostics, artifact)
        report = artifact.metadata["rig_quality"]
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "completed",
                output_artifact_id=artifact.id,
                logs=[
                    f"Found {report['skins']} skins and {report['joints']} joints",
                    f"Found {len(report['animations'])} embedded animations",
                ],
            )
        return artifact
    except Exception as exc:
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "failed",
                error=str(exc),
                logs=["Rig diagnostics failed"],
            )
        raise HTTPException(
            status_code=422,
            detail=f"Could not diagnose this rig: {exc}",
        ) from exc


@app.post("/api/assets/{artifact_id}/diagnose-deformation", response_model=Artifact)
async def diagnose_deformation_artifact(
    artifact_id: str,
    request: MeshDiagnosticsRequest,
) -> Artifact:
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    if path.suffix.lower() != ".glb":
        raise HTTPException(
            status_code=415,
            detail="Deformation diagnostics currently require a binary GLB asset",
        )

    attempt = None
    if request.run_id:
        try:
            attempt = run_store.start_attempt(
                request.run_id,
                "deformation-review",
                artifact.id,
                {"schema_version": 1},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        with TemporaryDirectory(prefix=f"aismith-deformation-{artifact.id}-") as temporary:
            output_path = Path(temporary) / "report.json"
            await model_manager.acquire("blender-deformation-review")
            try:
                report = await asyncio.to_thread(
                    native.run_deformation_diagnostics,
                    path,
                    output_path,
                )
            finally:
                await model_manager.release()
            report = persist_deformation_review_artifacts(report)
            decision = evaluate_deformation_review(report)
            artifact.metadata["deformation_review"] = report
            artifact.metadata.setdefault("quality_decisions", {})["deformation-review"] = decision
            _write_json(ARTIFACTS / f"{artifact.id}.json", artifact.model_dump())

            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            evidence = save_artifact(
                output_path,
                name=f"{artifact.name} deformation review.json",
                kind="deformation-review",
                metadata={
                    "stage": "deformation-review",
                    "source_artifact_id": artifact.id,
                    "deformation_review": report,
                    "quality_decisions": {"deformation-review": decision},
                },
            )
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "completed",
                output_artifact_id=evidence.id,
                logs=[
                    f"Reviewed {len(report.get('frames', []))} Sword Attack frames",
                    f"Deformation verdict: {decision['status']}",
                ],
                promote_output=False,
            )
        return artifact
    except Exception as exc:
        if attempt and request.run_id:
            run_store.finish_attempt(
                request.run_id,
                attempt.id,
                "failed",
                error=str(exc),
                logs=["Deformation review failed"],
            )
        raise HTTPException(
            status_code=422,
            detail=f"Could not review this animation: {exc}",
        ) from exc


@app.post("/api/assets/{artifact_id}/open-location")
async def open_location(artifact_id: str) -> dict[str, Any]:
    artifact = read_artifact(artifact_id)
    folder = str(Path(artifact.path).parent.resolve())
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
    return {"ok": True, "folder": folder}


class ExportArtifactRequest(BaseModel):
    # A caller-supplied filename base (e.g. the current run's name) — the
    # server still owns sanitization, the `-<artifact id>` suffix, and
    # collision handling below.
    name_hint: str | None = Field(default=None, max_length=120)


_EXPORT_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _sanitize_export_stem(raw: str) -> str:
    """A safe, human-readable filename stem with no path separators or extension."""
    stem = Path(raw).stem if Path(raw).suffix else raw
    cleaned = _EXPORT_FILENAME_UNSAFE_RE.sub("_", stem).strip(" ._-")
    return cleaned[:80] or "model"


def configured_output_directory() -> Path | None:
    """The explicitly user-selected output directory, or None if never configured.

    Unlike `output_directory()`, this never falls back to the internal
    artifacts cache — an export needs a real, deliberately chosen destination.
    """
    if not SETTINGS_PATH.exists():
        return None
    try:
        selected = str(_read_json(SETTINGS_PATH).get("output_directory", "")).strip()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not selected:
        return None
    path = Path(selected).expanduser().resolve()
    return path if path.exists() and path.is_dir() else None


@app.post("/api/assets/{artifact_id}/export")
async def export_artifact(artifact_id: str, request: ExportArtifactRequest = ExportArtifactRequest()) -> dict[str, Any]:
    """Copy an artifact into the operator's configured output folder.

    This is distinct from `/download` (a browser-side file download of
    whatever origin the frontend runs on) — this writes the file onto disk
    where the operator told the studio to keep finished exports.
    """
    artifact = read_artifact(artifact_id)
    source = _artifact_path(artifact_id)
    directory = configured_output_directory()
    if directory is None:
        raise HTTPException(
            status_code=409,
            detail="No output folder is configured. Choose an output folder in Generate before exporting.",
        )
    if not os.access(directory, os.W_OK):
        raise HTTPException(
            status_code=409,
            detail=f"The configured output folder is not writable: {directory}",
        )
    stem = _sanitize_export_stem(request.name_hint or artifact.name)
    base_name = f"{stem}-{artifact_id[:8]}"
    suffix = source.suffix.lower() or ".glb"
    destination = directory / f"{base_name}{suffix}"
    counter = 2
    while destination.exists():
        destination = directory / f"{base_name}-{counter}{suffix}"
        counter += 1
    shutil.copy2(source, destination)
    return {"path": str(destination), "size_bytes": destination.stat().st_size}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
