"""stdio MCP server: an external agent drives the AISmith pipeline as the
operator through this server rather than by calling the HTTP API directly.

Every tool is a thin wrapper over `backend/app/main.py`'s routes. Guardrails
read the same tables `backend/app/pipeline_policy.py` uses to drive the
pipeline loop itself, so this server cannot enforce a policy that has
drifted from the one the API evaluates against.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from backend.app import pipeline_policy

AISMITH_API_URL = os.environ.get("AISMITH_API_URL", "http://127.0.0.1:8100")
DOWNLOAD_DIR = Path(
    os.environ.get("AISMITH_MCP_DOWNLOAD_DIR", tempfile.gettempdir())
).resolve()

# Job kinds that spend GPU time on a creative decision rather than
# reproducing one already made -- these never run unattended. `rebuild-views`
# is a multi-hour Blender + ComfyUI repaint driven by agent-picked views.
GENERATION_SPEND_KINDS = {
    "generate",
    "create-tpose-set",
    "synthesize-views",
    "prepare-image",
    "rebuild-views",
}

# Artifact `metadata` blobs can carry whole contract reports; keep only the
# top-level shape plus the fields an operator actually reasons about.
_METADATA_SUMMARY_KEYS = ("stage", "quality_decisions", "pipeline_policy")

_RUN_SUMMARY_FIELDS = (
    "id", "name", "status", "subject_kind", "source_artifact_id",
    "current_artifact_id", "updated_at",
)
_ATTEMPT_FIELDS = (
    "stage", "status", "input_artifact_id", "output_artifact_id", "promoted", "error",
)
_JOB_FIELDS = ("status", "progress", "message", "output_artifact_id", "error")

mcp = FastMCP("aismith-operator")

client = httpx.Client(base_url=AISMITH_API_URL, timeout=30.0)


def _get(path: str) -> Any:
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def _post(path: str, body: dict[str, Any] | None = None) -> Any:
    if body is None:
        response = client.post(path)
    else:
        response = client.post(path, json=body)
    response.raise_for_status()
    return response.json()


def _trim_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    trimmed: dict[str, Any] = {"metadata_keys": sorted(metadata.keys())}
    for key in _METADATA_SUMMARY_KEYS:
        if key in metadata:
            trimmed[key] = metadata[key]
    return trimmed


def _trim_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(artifact)
    trimmed["metadata"] = _trim_metadata(artifact.get("metadata"))
    return trimmed


def _trim_run(run: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(run)
    trimmed["attempts"] = [
        {field: attempt.get(field) for field in _ATTEMPT_FIELDS}
        for attempt in run.get("attempts", [])
    ]
    return trimmed


def _trim_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    trimmed = {field: run.get(field) for field in _RUN_SUMMARY_FIELDS}
    trimmed["attempt_count"] = len(run.get("attempts", []))
    return trimmed


def _trim_job(job: dict[str, Any]) -> dict[str, Any]:
    return {field: job.get(field) for field in _JOB_FIELDS}


@mcp.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    runs = _get("/api/runs")
    return [_trim_run_summary(run) for run in runs[:limit]]


@mcp.tool()
def get_run(run_id: str) -> dict[str, Any]:
    return _trim_run(_get(f"/api/runs/{run_id}"))


@mcp.tool()
def get_run_policy(run_id: str) -> dict[str, Any]:
    return _get(f"/api/runs/{run_id}/policy")


@mcp.tool()
def pipeline_step(
    run_id: str,
    settings: dict[str, Any] | None = None,
    operator_confirmed: bool = False,
) -> dict[str, Any]:
    settings = settings or {}
    policy = get_run_policy(run_id)
    stage_value = policy.get("next_action", {}).get("stage")
    if stage_value is not None:
        stage = pipeline_policy.Stage(stage_value)
        needed = pipeline_policy.STAGE_MANUAL_INPUT.get(stage)
        if needed is None:
            needed = pipeline_policy.STAGE_CREATIVE_INPUT.get(stage)
        if needed is not None and needed not in settings:
            return {"refused": True, "needs_input": needed, "stage": stage.value}

    body: dict[str, Any] = {}
    if settings.get("submission_id"):
        body["submission_id"] = settings["submission_id"]
    if settings.get("clip"):
        body["clip"] = settings["clip"]
    if settings.get("family"):
        body["family"] = settings["family"]
    # An `attempt_cap` override is itself unattended remediation spend --
    # never let the loop widen its own cap without an explicit human say-so.
    if operator_confirmed and "attempt_cap" in settings:
        body["attempt_cap"] = settings["attempt_cap"]

    return _post(f"/api/runs/{run_id}/pipeline/step", body or None)


@mcp.tool()
def submit_job(
    kind: str,
    source_artifact_id: str,
    run_id: str | None = None,
    settings: dict[str, Any] | None = None,
    operator_confirmed: bool = False,
) -> dict[str, Any]:
    if kind in GENERATION_SPEND_KINDS and not operator_confirmed:
        return {
            "refused": True,
            "reason": "unattended generation spend requires operator_confirmed",
        }
    body: dict[str, Any] = {
        "kind": kind,
        "submission_id": f"mcp-{uuid.uuid4().hex}",
        "source_artifact_id": source_artifact_id,
        "settings": settings or {},
    }
    if run_id is not None:
        body["run_id"] = run_id
    return _post("/api/jobs", body)


@mcp.tool()
def job_status(job_id: str) -> dict[str, Any]:
    return _trim_job(_get(f"/api/jobs/{job_id}"))


@mcp.tool()
def cancel_job(job_id: str) -> dict[str, Any]:
    return _trim_job(_post(f"/api/jobs/{job_id}/cancel"))


@mcp.tool()
def get_artifact(artifact_id: str) -> dict[str, Any]:
    return _trim_artifact(_get(f"/api/assets/{artifact_id}"))


def _resolve_dest(dest_path: str) -> Path | None:
    dest = Path(dest_path)
    if not dest.is_absolute():
        return None
    resolved = dest.resolve()
    try:
        resolved.relative_to(DOWNLOAD_DIR)
    except ValueError:
        return None
    return resolved


@mcp.tool()
def download_artifact(artifact_id: str, dest_path: str) -> dict[str, Any]:
    resolved = _resolve_dest(dest_path)
    if resolved is None:
        return {
            "refused": True,
            "reason": f"dest_path must be an absolute path under {DOWNLOAD_DIR}",
        }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", f"/api/assets/{artifact_id}/download") as response:
        response.raise_for_status()
        with open(resolved, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return {
        "artifact_id": artifact_id,
        "dest_path": str(resolved),
        "bytes_written": resolved.stat().st_size,
    }


@mcp.tool()
def diagnose_rig(artifact_id: str, run_id: str | None = None) -> dict[str, Any]:
    return _trim_artifact(_post(f"/api/assets/{artifact_id}/diagnose-rig", {"run_id": run_id}))


@mcp.tool()
def record_checkpoint(run_id: str, checkpoint: str, note: str, artifact_id: str) -> dict[str, Any]:
    body = {"note": note, "artifact_id": artifact_id}
    return _trim_run(_post(f"/api/runs/{run_id}/checkpoints/{checkpoint}", body))


@mcp.tool()
def promote_artifact(run_id: str, artifact_id: str, reason: str = "promoted via MCP operator") -> dict[str, Any]:
    return _trim_run(_post(f"/api/runs/{run_id}/artifacts/{artifact_id}/promote", {"reason": reason}))


@mcp.tool()
def swap_tpose_member(run_id: str, role: str, artifact_id: str, reason: str = "") -> dict[str, Any]:
    body = {"artifact_id": artifact_id, "reason": reason}
    return _trim_artifact(_post(f"/api/runs/{run_id}/tpose-set/members/{role}", body))


@mcp.tool()
def health() -> dict[str, Any]:
    return _get("/api/health")


if __name__ == "__main__":
    mcp.run("stdio")
