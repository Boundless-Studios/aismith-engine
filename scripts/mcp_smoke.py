"""Read-only smoke test for backend/mcp_server.py against a live API.

Calls the MCP tool functions directly (imports backend.mcp_server; no stdio
round trip, no MCP client): health, list_runs, get_run/get_run_policy for
the first run, and get_artifact for its current artifact. Prints a compact
JSON summary and exits 1 on any HTTP error.

Usage:
    python3 scripts/mcp_smoke.py --api-url http://127.0.0.1:8100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", default=os.environ.get("AISMITH_API_URL", "http://127.0.0.1:8100")
    )
    args = parser.parse_args()

    os.environ["AISMITH_API_URL"] = args.api_url
    sys.path.insert(0, str(REPO_ROOT))
    from backend import mcp_server

    summary: dict[str, object] = {"api_url": args.api_url}
    try:
        summary["health"] = mcp_server.health()

        runs = mcp_server.list_runs(limit=3)
        summary["runs"] = runs

        if runs:
            first_run_id = runs[0]["id"]
            summary["run"] = mcp_server.get_run(first_run_id)
            summary["policy"] = mcp_server.get_run_policy(first_run_id)

            current_artifact_id = runs[0].get("current_artifact_id")
            if current_artifact_id:
                summary["artifact"] = mcp_server.get_artifact(current_artifact_id)
    except httpx.HTTPError as exc:
        print(json.dumps({"error": str(exc), "summary_so_far": summary}, indent=2))
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
