"""Optional fresh-process JSON transport for host-owned judge providers."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import tempfile
from .models import Identity, Request, Result


class CommandJudge:
    """Explicit argv, no shell. Command reads stdin JSON and writes one Result JSON.

    The executable is operator-configured, never copied from suggested actions.
    Provider wrapper owns credentials, model API, evidence resolution and isolation.
    A fresh process is necessary but not proof that a provider has no saved context.
    """
    def __init__(self, identity: Identity, argv: tuple[str, ...], *, timeout: float = 120):
        if not argv or timeout <= 0 or timeout > 600:
            raise ValueError("explicit command and timeout in (0,600] required")
        self.identity, self.argv, self.timeout = identity, argv, timeout

    def _call(self, payload):
        # Files avoid an unbounded in-memory PIPE buffer from a misbehaving adapter.
        with tempfile.TemporaryDirectory(prefix="asset-judge-") as directory:
            output = Path(directory) / "result.json"
            error = Path(directory) / "stderr.txt"
            with output.open("wb") as stdout, error.open("wb") as stderr:
                subprocess.run(self.argv, input=json.dumps(payload).encode(), stdout=stdout,
                               stderr=stderr, timeout=self.timeout, check=True)
            if output.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("judge output exceeds 2 MiB")
            return Result.model_validate_json(output.read_bytes())

    def evaluate(self, request: Request) -> Result:
        return self._call({"schema_version": 1, "mode": "judge", "request": request.model_dump(mode="json"),
                           "request_sha256": request.fingerprint()})

    def adjudicate(self, request: Request, primary: Result, secondary: Result) -> Result:
        return self._call({"schema_version": 1, "mode": "adjudicate", "request": request.model_dump(mode="json"),
                           "request_sha256": request.fingerprint(),
                           "opinions": [r.model_dump(mode="json") for r in (primary, secondary)]})
