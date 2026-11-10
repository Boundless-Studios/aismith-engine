"""`aismith-engine`: the per-stage entry point Magic Studios' pool runs.

    aismith-engine run <stage> --spec spec.json --workdir DIR
    aismith-engine inspect [--full]
    aismith-engine weights install|verify ...
    aismith-engine serve [--host 127.0.0.1] [--port 8190] [--work-root DIR] [--allow-private]

`run` writes `result.json` into the workdir, prints it, and exits 0 (ok),
1 (failed) or 2 (rejected). It always runs offline: weights are verified
against the lock and nothing is downloaded. It never waits for a busy GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .contract import STAGES, StageRejected, StageResult, StageSpec


def _offline() -> None:
    os.environ["AISMITH_OFFLINE"] = "1"
    # The pool hands the GPU over before a job; a busy GPU here is a failure,
    # not something to wait out.
    os.environ.setdefault("AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS", "0")


def run_stage(stage: str, spec_path: Path, workdir: Path) -> StageResult:
    _offline()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("stage") not in (None, stage):
            raise StageRejected(f"spec is for stage {data.get('stage')!r}, not {stage!r}")
        spec = StageSpec.from_json({**data, "stage": stage} if isinstance(data, dict) else data)
        from .stages import RUNNERS

        result = RUNNERS[stage](spec, workdir, lambda message: print(message, file=sys.stderr, flush=True))
    except StageRejected as exc:
        result = StageResult(stage=stage, status="rejected", error=str(exc))
    except (OSError, ValueError) as exc:
        if isinstance(exc, json.JSONDecodeError) or not spec_path.is_file():
            result = StageResult(stage=stage, status="rejected", error=f"unreadable spec: {exc}")
        else:
            result = StageResult(stage=stage, status="failed", error=str(exc))
            result.diagnostics["traceback"] = traceback.format_exc()[-4000:]
    except Exception as exc:  # a stage crash is a failed job, reported, never re-run
        result = StageResult(stage=stage, status="failed", error=f"{type(exc).__name__}: {exc}")
        result.diagnostics["traceback"] = traceback.format_exc()[-4000:]
    if not result.engine:
        from .inspect import identity

        result.engine = identity()
    (workdir / "result.json").write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aismith-engine", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one stage")
    run.add_argument("stage", choices=STAGES)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--workdir", type=Path, required=True)
    inspect_parser = sub.add_parser("inspect", help="report engine, runtimes, GPU and weights")
    inspect_parser.add_argument("--full", action="store_true", help="rehash every weight file")
    sub.add_parser("weights", help="install or verify pinned weights", add_help=False)
    serve_parser = sub.add_parser("serve", help="run stages for a loopback HTTP client (the engine sidecar)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="loopback unless --allow-private")
    serve_parser.add_argument("--port", type=int, default=8190)
    serve_parser.add_argument("--work-root", type=Path, default=None,
                              help="per-job work directories; its contents are removed at startup")
    serve_parser.add_argument("--allow-private", action="store_true",
                              help="also allow private, link-local or unspecified bind addresses")
    args, rest = parser.parse_known_args(argv)

    if args.command == "weights":
        from .weights import main as weights_main

        return weights_main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "serve":
        from .serve import serve

        return serve(args.host, args.port, args.work_root, args.allow_private)
    if args.command == "inspect":
        os.environ["AISMITH_OFFLINE"] = "1"
        from .inspect import inspect

        print(json.dumps(inspect(full=args.full), indent=2, default=str))
        return 0
    result = run_stage(args.stage, args.spec, args.workdir)
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
