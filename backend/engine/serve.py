"""`aismith-engine serve`: a loopback HTTP runner for the engine sidecar.

The engine container holds the GPU and the read-only model stores but no
credentials; the worker's agent container holds the credential but no GPU.
They share only a loopback network, and the agent drives the engine through
this runner (contract v1, documented in docs/engineering/engine-runtime.md).

Each job runs as its own `aismith-engine run <stage>` subprocess in a new
process group, so a hang or crash is isolated and killable. One job runs at a
time. Every job has a work directory under the work root; it is removed on
DELETE, and every stale one is removed at startup.

No auth and no secrets: the server binds loopback unless `--allow-private`
is given, and never a public address.
"""
from __future__ import annotations

import fcntl
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from backend.app import process_tree

from .contract import EXIT_CODES, SCHEMA_VERSION, STAGES, StageResult
from .sources import ROOT

DEFAULT_PORT = 8190
DEFAULT_WORK_ROOT = Path(tempfile.gettempdir()) / "aismith-engine-serve"
MARKER = ".aismith-engine-serve"
UPLOADS = ".uploads"
MiB = 1024 * 1024
INPUT_LIMITS = {"front": 20 * MiB, "back": 20 * MiB, "left": 20 * MiB, "right": 20 * MiB, "mesh": 256 * MiB}
RUN_BODY_LIMIT = 64 * 1024
MAX_DEADLINE_SECONDS = 24 * 3600
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CHUNK = MiB
GLB = "model/gltf-binary"
KILL_GRACE_SECONDS = 10.0
INSPECT_TIMEOUTS = {False: 300.0, True: 1800.0}
LOCK_FILE = ".lock"
TREE_SNAPSHOT_SECONDS = 0.5
# AISMITH_JOB_PROCESS_GROUP=1 keeps native workers (native.run_process) in the
# job's process group instead of their own session, so the group kill reaches
# them. It is native.JOB_PROCESS_GROUP_ENV; spelled out to keep this import light.
JOB_ENV = {"AISMITH_OFFLINE": "1", "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS": "0", "AISMITH_JOB_PROCESS_GROUP": "1"}


class RunnerError(Exception):
    """An HTTP error response: status plus a short message."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def check_host(host: str, *, allow_private: bool) -> str:
    """The address to bind, or ValueError.

    Loopback always; private, link-local and unspecified (0.0.0.0, ::) only
    with allow_private; public addresses never. Hostnames other than
    `localhost` are refused so the check covers what is actually bound.
    """
    if host == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"--host must be an IP address or localhost, not {host!r}") from None
    if address.is_loopback:
        return host
    if not allow_private:
        raise ValueError(f"refusing to bind {host}: the runner binds loopback only unless --allow-private is given")
    if address.is_unspecified or address.is_private or address.is_link_local:
        return host
    raise ValueError(f"refusing to bind public address {host}")


def _sniff_extension(name: str, head: bytes) -> str:
    """A file extension from the content; trimesh loads by extension."""
    if name == "mesh":
        return ".glb"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    return ".bin"  # the stage rejects it as not an image


def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate(process: subprocess.Popen, grace: float, tracked: dict[int, str] | None = None) -> None:
    """SIGTERM the process group and any tracked descendant, SIGKILL after the grace period."""
    _kill_group(process.pid, signal.SIGTERM)
    if tracked:
        process_tree.signal_all(tracked, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(process.pid, signal.SIGKILL)
    process.wait()


def _kill_survivors(tracked: dict[int, str], timeout: float = 10.0) -> None:
    """SIGKILL every recorded descendant still running, and reap the ones we inherited.

    A worker that called setsid is outside the job's process group, and once
    the stage exits it is reparented away, so it is found by the pids
    recorded while the stage ran. Start tokens guard against pid reuse.
    """
    if not tracked:
        return
    process_tree.signal_all(tracked, signal.SIGKILL)
    deadline = time.monotonic() + timeout
    while True:
        process_tree.reap(tracked)  # when we are PID 1 or a subreaper, they are ours to reap
        if not process_tree.alive(tracked) or time.monotonic() >= deadline:
            return
        time.sleep(0.05)


@dataclass
class Job:
    job_id: str
    directory: Path
    running: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    output: Path | None = None
    sha256: str | None = None


class Runner:
    """Job state, work directories and the stage subprocesses."""

    def __init__(self, work_root: Path, *, engine_command: list[str] | None = None,
                 kill_grace: float = KILL_GRACE_SECONDS, cwd: Path | None = None):
        self.work_root = Path(work_root).resolve()
        self.engine_command = list(engine_command or [sys.executable, "-m", "backend.engine"])
        self.kill_grace = kill_grace
        self.cwd = cwd or ROOT
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.active: Job | None = None
        self.process: subprocess.Popen | None = None
        self.tracked: dict[int, str] = {}
        self._lock_fd: int | None = None
        self._prepare_work_root()

    # -- work directories -------------------------------------------------
    def _prepare_work_root(self) -> None:
        """Lock the work root for this runner's lifetime, then clean it.

        The exclusive lock comes first, so a second runner on the same root
        refuses to start instead of wiping the first one's jobs.
        """
        root = self.work_root
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / LOCK_FILE
        lock_existed = lock_path.exists()
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ValueError(f"{root} is in use by another aismith-engine serve; refusing to start") from None
        entries = [entry for entry in root.iterdir() if entry.name != LOCK_FILE]
        if entries and not (root / MARKER).is_file():
            if not lock_existed:
                lock_path.unlink(missing_ok=True)
            os.close(fd)
            raise ValueError(f"{root} is not empty and is not a runner work root; refusing to clean it")
        self._lock_fd = fd
        for entry in entries:
            if entry.name == MARKER:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        (root / MARKER).write_text("aismith-engine serve work root; contents are removed at startup\n")
        (root / UPLOADS).mkdir()

    def close(self) -> None:
        """Release the work root lock."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # closing the descriptor drops the flock
            self._lock_fd = None

    def _job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            job = self.jobs[job_id] = Job(job_id, self.work_root / job_id)
        return job

    @staticmethod
    def _input_file(job: Job, name: str) -> Path | None:
        directory = job.directory / "inputs"
        if not directory.is_dir():
            return None
        return next((p for p in directory.iterdir() if p.stem == name and p.is_file()), None)

    # -- uploads ------------------------------------------------------------
    def put_input(self, job_id: str, name: str, length: int, body) -> None:
        upload = self.work_root / UPLOADS / f"{job_id}.{name}.{uuid.uuid4().hex}"
        try:
            remaining, head = length, b""
            with upload.open("wb") as handle:
                while remaining:
                    chunk = body.read(min(CHUNK, remaining))
                    if not chunk:
                        raise RunnerError(HTTPStatus.BAD_REQUEST, "request body is shorter than Content-Length")
                    if len(head) < 16:
                        head += chunk[:16]
                    handle.write(chunk)
                    remaining -= len(chunk)
            with self.lock:
                job = self._job(job_id)
                if job.running:
                    raise RunnerError(HTTPStatus.CONFLICT, "job is running")
                directory = job.directory / "inputs"
                directory.mkdir(parents=True, exist_ok=True)
                previous = self._input_file(job, name)
                if previous is not None:
                    previous.unlink()
                os.replace(upload, directory / f"{name}{_sniff_extension(name, head)}")
        finally:
            upload.unlink(missing_ok=True)

    # -- running ------------------------------------------------------------
    def run(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        stage, names, settings, deadline = _parse_run(request)
        with self.lock:
            if self.active is not None:
                raise RunnerError(HTTPStatus.CONFLICT, "busy")
            job = self._job(job_id)
            inputs = {}
            for name in names:
                path = self._input_file(job, name)
                if path is None:
                    raise RunnerError(HTTPStatus.NOT_FOUND, f"input {name!r} has not been uploaded")
                inputs[name] = str(path.resolve())
            job.running, job.result, job.output, job.sha256 = True, None, None, None
            job.cancel.clear()
            job.finished.clear()
            self.active = job
        result: dict[str, Any] | None = None
        output: Path | None = None
        try:
            result, output = self._execute(job, stage, inputs, settings, deadline)
        except Exception as exc:  # the runner itself failed; report it like a stage crash
            result = self._synthetic(stage, f"runner error: {type(exc).__name__}: {exc}", None,
                                     job.directory / "engine.log")
        finally:
            with self.lock:
                job.result, job.output = result, output
                job.sha256 = (result.get("output") or {}).get("sha256") if output and result else None
                job.running = False
                self.active = None
                self.process = None
                job.finished.set()
        return result

    def _execute(self, job: Job, stage: str, inputs: dict[str, str], settings: dict[str, Any],
                 deadline: int) -> tuple[dict[str, Any], Path | None]:
        workdir = job.directory / "run"
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        spec = job.directory / "spec.json"
        spec.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "stage": stage, "inputs": inputs,
                                    "settings": settings}), encoding="utf-8")
        log = job.directory / "engine.log"
        command = [*self.engine_command, "run", stage, "--spec", str(spec), "--workdir", str(workdir)]
        started = time.monotonic()
        reason = None
        with log.open("wb") as log_handle:
            with self.lock:
                if job.cancel.is_set():
                    return self._synthetic(stage, "cancelled", None, log), None
                process = subprocess.Popen(command, cwd=self.cwd, env={**os.environ, **JOB_ENV},
                                           stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.process = process
                tracked = self.tracked = {}
            next_snapshot = 0.0
            while True:
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if job.cancel.is_set():
                    reason = "cancelled"
                elif time.monotonic() - started >= deadline:
                    reason = "deadline exceeded"
                if reason or time.monotonic() >= next_snapshot:
                    # Record the whole tree while the stage is alive: a worker in its own
                    # session is out of reach of the group kill and is orphaned once the
                    # stage dies.
                    tracked.update(process_tree.descendants(process.pid))
                    next_snapshot = time.monotonic() + TREE_SNAPSHOT_SECONDS
                if reason:
                    _terminate(process, self.kill_grace, tracked)
                    break
            # Whatever the stage left behind, in its group or not, goes with it.
            _kill_group(process.pid, signal.SIGKILL)
            _kill_survivors(tracked)
        if reason:
            return self._synthetic(stage, reason, process.returncode, log), None
        try:
            result = json.loads((workdir / "result.json").read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("status") not in EXIT_CODES:
                raise ValueError("result.json has no valid status")
        except (OSError, ValueError) as exc:
            return self._synthetic(stage, f"the engine exited with code {process.returncode} and no valid "
                                          f"result ({type(exc).__name__})", process.returncode, log), None
        output = None
        if isinstance(result.get("output"), dict):
            entry = dict(result["output"])
            path = Path(str(entry.pop("path", ""))).resolve()
            if path.is_file() and path.is_relative_to(workdir.resolve()):
                output = path
                entry["url"] = f"/jobs/{job.job_id}/output"
                result["output"] = entry
            else:
                return self._synthetic(stage, "the engine's output is missing or outside its work directory",
                                       process.returncode, log), None
        return result, output

    @staticmethod
    def _synthetic(stage: str, error: str, returncode: int | None, log: Path) -> dict[str, Any]:
        from .inspect import identity

        result = StageResult(stage=stage, status="failed", error=error, engine=identity())
        tail = log.read_bytes()[-4000:].decode("utf-8", "replace") if log.is_file() else ""
        result.diagnostics = {"runner": {"returncode": returncode, "log_tail": tail}}
        return result.as_dict()

    def cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None and job.running:
                job.cancel.set()

    def delete(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            running = job is not None and job.running
            if running:
                job.cancel.set()
        if running:
            job.finished.wait(timeout=self.kill_grace + 30)
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None and job.running:
                raise RunnerError(HTTPStatus.CONFLICT, "job is still stopping")
            self.jobs.pop(job_id, None)
            shutil.rmtree(self.work_root / job_id, ignore_errors=True)

    def open_output(self, job_id: str) -> tuple[Any, int, str]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.running or job.output is None:
                raise RunnerError(HTTPStatus.NOT_FOUND, "no output")
            try:
                handle = job.output.open("rb")
            except OSError:
                raise RunnerError(HTTPStatus.NOT_FOUND, "no output") from None
            return handle, os.fstat(handle.fileno()).st_size, job.sha256 or ""

    @property
    def busy(self) -> bool:
        return self.active is not None

    def inspect(self, full: bool) -> dict[str, Any]:
        command = [*self.engine_command, "inspect", *(["--full"] if full else [])]
        process = subprocess.Popen(command, cwd=self.cwd, env={**os.environ, **JOB_ENV},
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=INSPECT_TIMEOUTS[full])
        except subprocess.TimeoutExpired:
            _terminate(process, self.kill_grace)
            raise RunnerError(HTTPStatus.GATEWAY_TIMEOUT, "inspect timed out") from None
        finally:
            _kill_group(process.pid, signal.SIGKILL)
        if process.returncode != 0:
            raise RunnerError(HTTPStatus.INTERNAL_SERVER_ERROR,
                              f"inspect exited with code {process.returncode}: {stderr.decode('utf-8', 'replace')[-2000:]}")
        try:
            return json.loads(stdout)
        except ValueError:
            raise RunnerError(HTTPStatus.INTERNAL_SERVER_ERROR, "inspect printed no JSON") from None

    def shutdown(self) -> None:
        """Stop the active stage's process group; the server is exiting."""
        with self.lock:
            if self.active is not None:
                self.active.cancel.set()
            process = self.process
            tracked = self.tracked
        if process is not None:
            if process.poll() is None:
                tracked.update(process_tree.descendants(process.pid))
            _terminate(process, min(self.kill_grace, 5.0), tracked)
            _kill_survivors(tracked)
        self.close()


def _parse_run(request: Any) -> tuple[str, list[str], dict[str, Any], int]:
    def bad(message: str) -> RunnerError:
        return RunnerError(HTTPStatus.BAD_REQUEST, message)

    if not isinstance(request, dict):
        raise bad("the body must be a JSON object")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise bad(f"schema_version must be {SCHEMA_VERSION}")
    stage = request.get("stage")
    if stage not in STAGES:
        raise bad(f"stage must be one of {', '.join(STAGES)}")
    names = request.get("inputs")
    if (not isinstance(names, list) or not names or len(set(map(str, names))) != len(names)
            or not all(isinstance(n, str) and n in INPUT_LIMITS for n in names)):
        raise bad(f"inputs must be a non-empty list of distinct names from {', '.join(INPUT_LIMITS)}")
    settings = request.get("settings", {})
    if not isinstance(settings, dict):
        raise bad("settings must be an object")
    deadline = request.get("deadline_seconds")
    if isinstance(deadline, bool) or not isinstance(deadline, int) or not 0 < deadline <= MAX_DEADLINE_SECONDS:
        raise bad(f"deadline_seconds must be an integer from 1 to {MAX_DEADLINE_SECONDS}")
    unknown = set(request) - {"schema_version", "stage", "inputs", "settings", "deadline_seconds"}
    if unknown:
        raise bad(f"unknown fields: {sorted(unknown)}")
    return stage, names, settings, deadline


JOB_PATH = re.compile(r"^/jobs/([^/]*)(?:/(.*))?$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "aismith-engine-serve"
    server: "RunnerServer"

    # -- plumbing -------------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler's name
        if not self.server.quiet:
            super().log_message(format, *args)

    def _send(self, status: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        body = b"" if payload is None else json.dumps(payload, default=str).encode()
        self.send_response(status)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        if status != HTTPStatus.NO_CONTENT:
            self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _content_length(self) -> int:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise RunnerError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required; chunked bodies are not accepted")
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise RunnerError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        if not raw.strip().isdigit():
            self.close_connection = True
            raise RunnerError(HTTPStatus.BAD_REQUEST, "Content-Length must be a non-negative integer")
        return int(raw)

    def _discard_body(self) -> None:
        """Requests whose body is not read cannot keep the connection alive."""
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0").strip() not in ("", "0"):
            self.close_connection = True

    def _dispatch(self, method: str) -> None:
        self.body_read = False
        try:
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            if not (method == "PUT" or (method == "POST" and path.endswith("/run"))):
                self._discard_body()
            if method == "GET" and path == "/health":
                return self._send(HTTPStatus.OK, {"ok": True, "busy": self.server.runner.busy,
                                                  "schema_version": SCHEMA_VERSION})
            if method == "GET" and path == "/inspect":
                full = query.get("full", ["0"])[-1]
                if full not in ("0", "1"):
                    raise RunnerError(HTTPStatus.BAD_REQUEST, "full must be 0 or 1")
                return self._send(HTTPStatus.OK, self.server.runner.inspect(full == "1"))
            match = JOB_PATH.match(path)
            if not match:
                raise RunnerError(HTTPStatus.NOT_FOUND, "not found")
            job_id, rest = match.group(1), match.group(2)
            if not JOB_ID.match(job_id):
                raise RunnerError(HTTPStatus.BAD_REQUEST, "job id must match ^[A-Za-z0-9_-]{1,64}$")
            self._job_route(method, job_id, rest)
        except RunnerError as exc:
            if not self.body_read:
                self._discard_body()
            self._send(exc.status, {"error": exc.message})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # never drop a request without a response
            self.close_connection = True
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def _job_route(self, method: str, job_id: str, rest: str | None) -> None:
        runner = self.server.runner
        if rest is not None and rest.startswith("inputs/") and method == "PUT":
            name = rest[len("inputs/"):]
            if name not in INPUT_LIMITS:
                raise RunnerError(HTTPStatus.BAD_REQUEST, f"input name must be one of {', '.join(INPUT_LIMITS)}")
            length = self._content_length()
            if length > INPUT_LIMITS[name]:
                self.close_connection = True
                raise RunnerError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                  f"{name} is limited to {INPUT_LIMITS[name] // MiB} MiB")
            with self.server.runner.lock:
                job = runner.jobs.get(job_id)
                if job is not None and job.running:
                    self.close_connection = True
                    raise RunnerError(HTTPStatus.CONFLICT, "job is running")
            try:
                runner.put_input(job_id, name, length, self.rfile)
            except RunnerError as exc:
                # A short body leaves the stream unusable; a 409 at commit read it all.
                self.body_read = exc.status != HTTPStatus.BAD_REQUEST
                raise
            self.body_read = True
            self._send(HTTPStatus.NO_CONTENT)
            return
        if rest == "run" and method == "POST":
            length = self._content_length()
            if length > RUN_BODY_LIMIT:
                self.close_connection = True
                raise RunnerError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "run body is limited to 64 KiB")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RunnerError(HTTPStatus.BAD_REQUEST, "request body is shorter than Content-Length")
            self.body_read = True
            try:
                request = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                raise RunnerError(HTTPStatus.BAD_REQUEST, "the body is not JSON") from None
            self._send(HTTPStatus.OK, runner.run(job_id, request))
            return
        if rest == "output" and method == "GET":
            handle, size, sha256 = runner.open_output(job_id)
            with handle:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", GLB)
                self.send_header("Content-Length", str(size))
                self.send_header("X-Sha256", sha256)
                self.end_headers()
                remaining = size
                while remaining and (chunk := handle.read(min(CHUNK, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        if rest == "cancel" and method == "POST":
            runner.cancel(job_id)
            self._send(HTTPStatus.ACCEPTED, {"ok": True})
            return
        if rest is None and method == "DELETE":
            runner.delete(job_id)
            self._send(HTTPStatus.NO_CONTENT)
            return
        raise RunnerError(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._dispatch("GET")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")


class RunnerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runner: Runner, *, quiet: bool = False):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        try:
            super().__init__(address, Handler)
        except OSError:
            runner.close()
            raise
        self.runner = runner
        self.quiet = quiet

    def server_close(self) -> None:
        super().server_close()
        self.runner.close()


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, work_root: Path | None = None,
                allow_private: bool = False, engine_command: list[str] | None = None,
                kill_grace: float = KILL_GRACE_SECONDS, quiet: bool = False) -> RunnerServer:
    """Check the host first, then clean the work root, then bind."""
    bind = check_host(host, allow_private=allow_private)
    runner = Runner(work_root or DEFAULT_WORK_ROOT, engine_command=engine_command, kill_grace=kill_grace)
    return RunnerServer((bind, port), runner, quiet=quiet)


def serve(host: str, port: int, work_root: Path | None, allow_private: bool) -> int:
    try:
        server = make_server(host, port, work_root=work_root, allow_private=allow_private)
    except (ValueError, OSError) as exc:
        print(f"aismith-engine serve: {exc}", file=sys.stderr)
        return 2

    def stop(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    bound_host, bound_port = server.server_address[:2]
    print(f"aismith-engine serve: listening on http://{bound_host}:{bound_port}, "
          f"work root {server.runner.work_root}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.runner.shutdown()
        server.server_close()
    return 0
