"""The stage contract: one JSON spec in, one output file plus one JSON result out.

This is the interface Magic Studios' pool plugin calls (`aismith-engine run`),
so it is versioned: a breaking change bumps SCHEMA_VERSION and the engine's
major version.

Statuses:
  ok        an output was produced. A failed gate is still `ok`, with
            `gate.passed` false, so the caller can show why.
  rejected  the spec or its inputs are invalid. Terminal; retrying the same
            spec cannot succeed.
  failed    anything else (runtime, weights, GPU, crash). Never replayed by
            the engine; the caller decides whether to resubmit.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STAGES = ("mesh", "clean")
EXIT_CODES = {"ok": 0, "failed": 1, "rejected": 2}


class StageRejected(ValueError):
    """The spec or an input is invalid; terminal."""


@dataclass(frozen=True)
class StageSpec:
    stage: str
    inputs: dict[str, Path]
    settings: dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_json(cls, data: Any) -> "StageSpec":
        if not isinstance(data, dict):
            raise StageRejected("spec must be a JSON object")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise StageRejected(f"spec schema_version must be {SCHEMA_VERSION}")
        stage = data.get("stage")
        if stage not in STAGES:
            raise StageRejected(f"stage must be one of {', '.join(STAGES)}")
        inputs = data.get("inputs") or {}
        settings = data.get("settings") or {}
        if not isinstance(inputs, dict) or not all(isinstance(v, str) for v in inputs.values()):
            raise StageRejected("inputs must map names to file paths")
        if not isinstance(settings, dict):
            raise StageRejected("settings must be an object")
        paths = {name: Path(value) for name, value in inputs.items()}
        for name, path in paths.items():
            if not path.is_file():
                raise StageRejected(f"input {name!r} is not a file: {path}")
        return cls(stage=stage, inputs=paths, settings=dict(settings))


@dataclass
class StageOutput:
    path: str
    content_type: str
    sha256: str
    bytes: int

    @classmethod
    def of(cls, path: Path, content_type: str) -> "StageOutput":
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return cls(str(path), content_type, digest.hexdigest(), path.stat().st_size)


@dataclass
class StageResult:
    stage: str
    status: str
    schema_version: int = SCHEMA_VERSION
    output: StageOutput | None = None
    gate: dict[str, Any] | None = None
    effective_inputs: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    engine: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]
