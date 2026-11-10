"""The stdio MCP operator server (backend/mcp_server.py).

Every test mocks the HTTP transport (`httpx.MockTransport`) and calls the
tool functions directly -- there is no real API and no stdio round trip.
What is asserted: each tool hits the expected method + path, large fields
get trimmed, and the guardrails refuse locally (no HTTP call at all) rather
than trusting the server to say no.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from backend import mcp_server
from backend.app import pipeline_policy


def _json_response(payload, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


class RecordingRouter:
    """Maps (method, path) -> response and records every call it served."""

    def __init__(self):
        self.routes: dict[tuple[str, str], httpx.Response] = {}
        self.calls: list[tuple[str, str]] = []

    def add(self, method: str, path: str, response: httpx.Response) -> None:
        self.routes[(method, path)] = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        self.calls.append(key)
        if key not in self.routes:
            raise AssertionError(f"unexpected request: {key}")
        return self.routes[key]


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.router = RecordingRouter()
        mcp_server.client = httpx.Client(
            transport=httpx.MockTransport(self.router), base_url="http://testserver"
        )

    def test_health_passes_through(self):
        self.router.add("GET", "/api/health", _json_response({"status": "ok"}))
        self.assertEqual(mcp_server.health(), {"status": "ok"})
        self.assertEqual(self.router.calls, [("GET", "/api/health")])

    def test_list_runs_trims_and_limits(self):
        runs = [
            {
                "id": f"run-{i}",
                "name": f"Run {i}",
                "status": "in_progress",
                "subject_kind": "biped",
                "source_artifact_id": "art-src",
                "current_artifact_id": "art-cur",
                "updated_at": 1000 + i,
                "attempts": [{"stage": "S2_CANONICALIZATION"}] * 2,
                "review_note": "should not appear",
            }
            for i in range(5)
        ]
        self.router.add("GET", "/api/runs", _json_response(runs))
        result = mcp_server.list_runs(limit=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0],
            {
                "id": "run-0",
                "name": "Run 0",
                "status": "in_progress",
                "subject_kind": "biped",
                "source_artifact_id": "art-src",
                "current_artifact_id": "art-cur",
                "updated_at": 1000,
                "attempt_count": 2,
            },
        )
        self.assertNotIn("review_note", result[0])

    def test_get_run_trims_attempts(self):
        run = {
            "id": "run-1",
            "name": "Run 1",
            "status": "in_progress",
            "attempts": [
                {
                    "id": "attempt-1",
                    "stage": "S2_CANONICALIZATION",
                    "status": "completed",
                    "input_artifact_id": "art-in",
                    "output_artifact_id": "art-out",
                    "promoted": True,
                    "error": None,
                    "settings": {"huge": "blob" * 1000},
                    "logs": ["a very long log line"] * 50,
                    "warnings": [],
                }
            ],
        }
        self.router.add("GET", "/api/runs/run-1", _json_response(run))
        result = mcp_server.get_run("run-1")
        self.assertEqual(
            result["attempts"],
            [
                {
                    "stage": "S2_CANONICALIZATION",
                    "status": "completed",
                    "input_artifact_id": "art-in",
                    "output_artifact_id": "art-out",
                    "promoted": True,
                    "error": None,
                }
            ],
        )
        self.assertNotIn("settings", result["attempts"][0])
        self.assertEqual(self.router.calls, [("GET", "/api/runs/run-1")])

    def test_get_run_policy_returns_as_is(self):
        policy = {"policy_version": 1, "next_action": {"kind": "done", "stage": None}}
        self.router.add("GET", "/api/runs/run-1/policy", _json_response(policy))
        self.assertEqual(mcp_server.get_run_policy("run-1"), policy)

    def test_pipeline_step_refuses_manual_input_without_settings(self):
        policy = {
            "next_action": {"kind": "stop-at-stage", "stage": pipeline_policy.Stage.S0_SOURCE.value}
        }
        self.router.add("GET", "/api/runs/run-1/policy", _json_response(policy))
        result = mcp_server.pipeline_step("run-1")
        self.assertEqual(
            result, {"refused": True, "needs_input": "source-image", "stage": "S0_SOURCE"}
        )
        # No POST was ever attempted -- the guardrail must not call the API.
        self.assertEqual(self.router.calls, [("GET", "/api/runs/run-1/policy")])

    def test_pipeline_step_refuses_creative_input_without_clip(self):
        policy = {
            "next_action": {"kind": "advance", "stage": pipeline_policy.Stage.S5_ANIMATION.value}
        }
        self.router.add("GET", "/api/runs/run-1/policy", _json_response(policy))
        result = mcp_server.pipeline_step("run-1", settings={})
        self.assertEqual(
            result, {"refused": True, "needs_input": "clip", "stage": "S5_ANIMATION"}
        )
        self.assertEqual(self.router.calls, [("GET", "/api/runs/run-1/policy")])

    def test_pipeline_step_forwards_when_clip_is_supplied(self):
        policy = {
            "next_action": {"kind": "advance", "stage": pipeline_policy.Stage.S5_ANIMATION.value}
        }
        step_response = {"run_id": "run-1", "action": "advanced", "stage": "S5_ANIMATION"}
        self.router.add("GET", "/api/runs/run-1/policy", _json_response(policy))
        self.router.add("POST", "/api/runs/run-1/pipeline/step", _json_response(step_response))
        result = mcp_server.pipeline_step("run-1", settings={"clip": "wave"})
        self.assertEqual(result, step_response)
        self.assertEqual(
            self.router.calls,
            [("GET", "/api/runs/run-1/policy"), ("POST", "/api/runs/run-1/pipeline/step")],
        )

    def test_pipeline_step_proceeds_for_non_gated_stage(self):
        policy = {
            "next_action": {"kind": "advance", "stage": pipeline_policy.Stage.S2_CANONICALIZATION.value}
        }
        step_response = {"action": "advanced"}
        self.router.add("GET", "/api/runs/run-1/policy", _json_response(policy))
        self.router.add("POST", "/api/runs/run-1/pipeline/step", _json_response(step_response))
        result = mcp_server.pipeline_step("run-1")
        self.assertEqual(result, step_response)

    def test_submit_job_refuses_generate_without_confirmation(self):
        result = mcp_server.submit_job(kind="generate", source_artifact_id="art-1")
        self.assertEqual(
            result,
            {"refused": True, "reason": "unattended generation spend requires operator_confirmed"},
        )
        self.assertEqual(self.router.calls, [])

    def test_submit_job_refuses_rebuild_views_without_confirmation(self):
        # A rebuild is a multi-hour Blender + ComfyUI repaint: generation
        # spend, never started unattended.
        result = mcp_server.submit_job(kind="rebuild-views", source_artifact_id="art-1")
        self.assertEqual(
            result,
            {"refused": True, "reason": "unattended generation spend requires operator_confirmed"},
        )
        self.assertEqual(self.router.calls, [])

    def test_submit_job_allows_generate_with_confirmation(self):
        job = {"id": "job-1", "kind": "generate"}
        self.router.add("POST", "/api/jobs", _json_response(job))
        result = mcp_server.submit_job(
            kind="generate", source_artifact_id="art-1", operator_confirmed=True
        )
        self.assertEqual(result, job)
        self.assertEqual(self.router.calls, [("POST", "/api/jobs")])

    def test_submit_job_allows_non_spend_kinds_without_confirmation(self):
        job = {"id": "job-2", "kind": "canonicalize"}
        self.router.add("POST", "/api/jobs", _json_response(job))
        result = mcp_server.submit_job(kind="canonicalize", source_artifact_id="art-1")
        self.assertEqual(result, job)

    def test_job_status_trims(self):
        job = {
            "id": "job-1",
            "kind": "generate",
            "status": "running",
            "progress": 40,
            "message": "Reconstructing geometry",
            "output_artifact_id": None,
            "error": None,
            "logs": ["line"] * 100,
            "request_payload": {"huge": True},
        }
        self.router.add("GET", "/api/jobs/job-1", _json_response(job))
        result = mcp_server.job_status("job-1")
        self.assertEqual(
            result,
            {
                "status": "running",
                "progress": 40,
                "message": "Reconstructing geometry",
                "output_artifact_id": None,
                "error": None,
            },
        )

    def test_cancel_job_trims(self):
        job = {"id": "job-1", "status": "cancelled", "progress": 40, "message": "Cancelling task",
               "output_artifact_id": None, "error": "Cancelled by user"}
        self.router.add("POST", "/api/jobs/job-1/cancel", _json_response(job))
        result = mcp_server.cancel_job("job-1")
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.router.calls, [("POST", "/api/jobs/job-1/cancel")])

    def test_get_artifact_trims_metadata(self):
        artifact = {
            "id": "art-1",
            "name": "art.glb",
            "kind": "glb",
            "path": "/data/art-1.glb",
            "size_bytes": 123,
            "created_at": 1.0,
            "metadata": {
                "stage": "S3_SKELETON_FIT",
                "quality_decisions": {"promotable": True},
                "pipeline_policy": {"policy_version": 1},
                "raw_contract_report": {"huge": "blob" * 2000},
            },
        }
        self.router.add("GET", "/api/assets/art-1", _json_response(artifact))
        result = mcp_server.get_artifact("art-1")
        self.assertEqual(
            sorted(result["metadata"]["metadata_keys"]),
            sorted(["stage", "quality_decisions", "pipeline_policy", "raw_contract_report"]),
        )
        self.assertEqual(result["metadata"]["stage"], "S3_SKELETON_FIT")
        self.assertEqual(result["metadata"]["quality_decisions"], {"promotable": True})
        self.assertNotIn("raw_contract_report", result["metadata"])

    def test_diagnose_rig_posts_and_trims(self):
        artifact = {
            "id": "art-1", "name": "art.glb", "kind": "glb", "path": "/x", "size_bytes": 1,
            "created_at": 1.0, "metadata": {"rig_quality": {"joints": 40}},
        }
        self.router.add("POST", "/api/assets/art-1/diagnose-rig", _json_response(artifact))
        result = mcp_server.diagnose_rig("art-1")
        self.assertEqual(result["id"], "art-1")
        self.assertEqual(self.router.calls, [("POST", "/api/assets/art-1/diagnose-rig")])

    def test_record_checkpoint_posts_note_and_artifact(self):
        run = {"id": "run-1", "attempts": []}
        self.router.add("POST", "/api/runs/run-1/checkpoints/skeleton", _json_response(run))
        result = mcp_server.record_checkpoint("run-1", "skeleton", "looks right", "art-1")
        self.assertEqual(result["id"], "run-1")
        self.assertEqual(self.router.calls, [("POST", "/api/runs/run-1/checkpoints/skeleton")])

    def test_promote_artifact_posts(self):
        run = {"id": "run-1", "attempts": []}
        self.router.add("POST", "/api/runs/run-1/artifacts/art-1/promote", _json_response(run))
        result = mcp_server.promote_artifact("run-1", "art-1")
        self.assertEqual(result["id"], "run-1")

    def test_swap_tpose_member_posts(self):
        artifact = {
            "id": "art-1", "name": "front.png", "kind": "image", "path": "/x", "size_bytes": 1,
            "created_at": 1.0, "metadata": {},
        }
        self.router.add("POST", "/api/runs/run-1/tpose-set/members/back", _json_response(artifact))
        result = mcp_server.swap_tpose_member("run-1", "back", "art-2")
        self.assertEqual(result["id"], "art-1")

    def test_download_artifact_refuses_path_outside_allowed_dir(self):
        with TemporaryDirectory() as allowed_dir:
            mcp_server.DOWNLOAD_DIR = Path(allowed_dir).resolve()
            outside = str((Path(allowed_dir).parent / "escaped.glb"))
            result = mcp_server.download_artifact("art-1", outside)
            self.assertTrue(result["refused"])
            self.assertEqual(self.router.calls, [])

    def test_download_artifact_refuses_relative_path(self):
        with TemporaryDirectory() as allowed_dir:
            mcp_server.DOWNLOAD_DIR = Path(allowed_dir).resolve()
            result = mcp_server.download_artifact("art-1", "relative/path.glb")
            self.assertTrue(result["refused"])
            self.assertEqual(self.router.calls, [])

    def test_download_artifact_streams_into_allowed_dir(self):
        with TemporaryDirectory() as allowed_dir:
            mcp_server.DOWNLOAD_DIR = Path(allowed_dir).resolve()
            dest = str(Path(allowed_dir) / "sub" / "art-1.glb")
            self.router.add(
                "GET", "/api/assets/art-1/download",
                httpx.Response(200, content=b"glb-bytes"),
            )
            result = mcp_server.download_artifact("art-1", dest)
            self.assertEqual(result["bytes_written"], len(b"glb-bytes"))
            self.assertEqual(Path(dest).read_bytes(), b"glb-bytes")


if __name__ == "__main__":
    unittest.main()
