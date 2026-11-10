"""Shared pytest fixtures for the backend suite.

Every test that drives `main._run_job` (or anything else that persists a
JobState) must not touch the live `studio-data/jobs` directory: a test's
fake "running" job left there (`paint-job.json`, `job-1.json`,
`job-proxy.json` were seen after a suite run) is served by the real API as
a live job and blocks operational restarts that wait for the box to go
idle. Most tests already patch `main.JOBS_DIR`; this autouse fixture makes
the redirection unconditional so no new test can regress it.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_jobs_dir(tmp_path):
    from backend.app import main

    jobs_dir = tmp_path / "jobs-autouse"
    jobs_dir.mkdir()
    with patch.object(main, "JOBS_DIR", jobs_dir):
        yield jobs_dir
