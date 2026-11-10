"""`aismith-engine serve`: the engine sidecar's loopback HTTP runner.

The stage subprocess is replaced by a small fake engine script, so no GPU,
model or native runtime is involved. The fake behaves by `settings.mode`.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from backend.engine import cli, serve

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GLB_BYTES = b"glTF" + (2).to_bytes(4, "little") + (128).to_bytes(4, "little") + b"\x00" * 116

FAKE_ENGINE = textwrap.dedent('''
    import hashlib, json, os, signal, subprocess, sys, time

    args = sys.argv[1:]
    env = {k: os.environ.get(k) for k in ("AISMITH_OFFLINE", "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS")}
    if args[0] == "inspect":
        if os.environ.get("FAKE_INSPECT_FAIL"):
            sys.exit(3)
        print(json.dumps({"engine": {"name": "aismith-engine"}, "full": "--full" in args, "env": env}))
        sys.exit(0)
    _, stage, _, spec_path, _, workdir = args
    spec = json.load(open(spec_path))
    mode = spec["settings"].get("mode", "ok")
    inputs = {name: {"file": os.path.basename(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
              for name, path in spec["inputs"].items()}
    base = {"schema_version": 1, "stage": stage, "engine": {"name": "fake"}, "gate": None,
            "diagnostics": {"env": env, "inputs": inputs, "settings": spec["settings"]}}

    def write(result):
        with open(os.path.join(workdir, "result.json"), "w") as handle:
            json.dump(result, handle)

    if mode in ("ok", "outside"):
        data = b"glTF" + (2).to_bytes(4, "little") + (4096).to_bytes(4, "little") + os.urandom(4084)
        out = os.path.join(workdir, "mesh.glb") if mode == "ok" else os.path.abspath(spec_path)
        if mode == "ok":
            open(out, "wb").write(data)
        else:
            data = open(out, "rb").read()
        write({**base, "status": "ok", "output": {"path": out, "content_type": "model/gltf-binary",
               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}})
        sys.exit(0)
    if mode in ("rejected", "failed"):
        write({**base, "status": mode, "output": None, "error": "fake " + mode})
        sys.exit({"failed": 1, "rejected": 2}[mode])
    if mode == "crash":
        os._exit(7)
    if mode in ("sleep", "setsid", "native"):
        # A child that ignores SIGTERM and reports its pid and process group.
        # sleep: an ordinary child. setsid: its own session, as native.run_process
        # spawns workers by default. native: spawned through native.run_process itself.
        ready = os.path.join(workdir, "child-ready")
        child_code = ("import os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      "open(sys.argv[1] + '.tmp', 'w').write(f'{os.getpid()} {os.getpgid(0)}'); "
                      "os.replace(sys.argv[1] + '.tmp', sys.argv[1]); time.sleep(600)")
        if mode == "native":
            import threading
            sys.path.insert(0, os.getcwd())  # the runner's cwd is this checkout
            from backend.app import native
            threading.Thread(target=native.run_process, args=([sys.executable, "-c", child_code, ready],),
                             daemon=True).start()
        else:
            subprocess.Popen([sys.executable, "-c", child_code, ready], start_new_session=mode == "setsid")
        while not os.path.exists(ready):
            time.sleep(0.01)
        child_pid, child_pgid = open(ready).read().split()
        with open(os.path.join(workdir, "pids.tmp"), "w") as handle:
            handle.write(f"{os.getpid()} {child_pid} {os.getpgid(0)} {child_pgid}")
        os.replace(os.path.join(workdir, "pids.tmp"), os.path.join(workdir, "pids"))
        time.sleep(600)
''')


class Client:
    def __init__(self, port: int):
        self.port = port

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
                timeout: float = 60) -> tuple[int, dict, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def json(self, method: str, path: str, payload=None, **kwargs) -> tuple[int, dict]:
        body = None if payload is None else json.dumps(payload).encode()
        status, _, data = self.request(method, path, body, **kwargs)
        return status, (json.loads(data) if data else {})

    def headers_only(self, method: str, path: str, headers: dict, timeout: float = 30) -> tuple[int, dict]:
        """Send headers and no body; a server that tried to read the body would hang."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.putrequest(method, path)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            response = connection.getresponse()
            return response.status, json.loads(response.read() or b"{}")
        finally:
            connection.close()

    def put(self, job: str, name: str, data: bytes) -> int:
        return self.request("PUT", f"/jobs/{job}/inputs/{name}", data)[0]

    def run(self, job: str, stage: str = "mesh", inputs=("front",), deadline: int = 60, **settings) -> tuple[int, dict]:
        return self.json("POST", f"/jobs/{job}/run", {"schema_version": 1, "stage": stage, "inputs": list(inputs),
                                                     "settings": settings, "deadline_seconds": deadline})


@pytest.fixture
def work_root(tmp_path) -> Path:
    return tmp_path / "work-root"


@pytest.fixture
def client(tmp_path, work_root):
    fake = tmp_path / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    server = serve.make_server("127.0.0.1", 0, work_root=work_root, engine_command=[sys.executable, str(fake)],
                               kill_grace=0.5, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(server.server_address[1])
    server.shutdown()
    server.runner.shutdown()
    server.server_close()
    thread.join(timeout=10)


def _wait_for(predicate, timeout: float = 30, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _background_run(client: Client, job: str, **kwargs) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def target():
        box["response"] = client.run(job, **kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def _pids(work_root: Path, job: str) -> list[int]:
    """[stage pid, child pid]; the stage's and child's process groups are in _groups."""
    return _ids(work_root, job)[:2]


def _groups(work_root: Path, job: str) -> list[int]:
    return _ids(work_root, job)[2:]


def _ids(work_root: Path, job: str) -> list[int]:
    pids = work_root / job / "run" / "pids"
    _wait_for(pids.is_file, message="the fake stage to start")
    return [int(p) for p in pids.read_text().split()]


def test_health(client):
    assert client.json("GET", "/health") == (200, {"ok": True, "busy": False, "schema_version": 1})


def test_inspect_runs_the_cli_offline(client):
    status, report = client.json("GET", "/inspect")
    assert status == 200 and report["full"] is False and report["env"]["AISMITH_OFFLINE"] == "1"
    status, report = client.json("GET", "/inspect?full=1")
    assert status == 200 and report["full"] is True
    assert client.json("GET", "/inspect?full=yes")[0] == 400


def test_inspect_failure_is_an_error(client, monkeypatch):
    monkeypatch.setenv("FAKE_INSPECT_FAIL", "1")
    status, body = client.json("GET", "/inspect")
    assert status == 500 and "code 3" in body["error"]


def test_upload_run_and_fetch_the_output(client, work_root):
    assert client.put("job-1", "front", PNG) == 204
    assert client.put("job-1", "back", PNG + b"x") == 204
    status, result = client.run("job-1", inputs=["front", "back"], seed=7)
    assert status == 200 and result["status"] == "ok"
    assert "path" not in result["output"] and result["output"]["url"] == "/jobs/job-1/output"
    diagnostics = result["diagnostics"]
    assert diagnostics["env"] == {"AISMITH_OFFLINE": "1", "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS": "0"}
    assert diagnostics["inputs"]["front"] == {"file": "front.png", "sha256": hashlib.sha256(PNG).hexdigest()}
    assert diagnostics["inputs"]["back"]["sha256"] == hashlib.sha256(PNG + b"x").hexdigest()
    assert diagnostics["settings"] == {"seed": 7}

    status, headers, data = client.request("GET", "/jobs/job-1/output")
    assert status == 200 and headers["Content-Type"] == "model/gltf-binary"
    assert int(headers["Content-Length"]) == len(data) == result["output"]["bytes"]
    assert headers["X-Sha256"] == hashlib.sha256(data).hexdigest() == result["output"]["sha256"]
    assert data[:4] == b"glTF"


def test_a_mesh_input_is_stored_as_glb_and_replaced_on_reupload(client, work_root):
    assert client.put("job-2", "mesh", b"old") == 204
    assert client.put("job-2", "mesh", GLB_BYTES) == 204
    assert [p.name for p in (work_root / "job-2" / "inputs").iterdir()] == ["mesh.glb"]
    status, result = client.run("job-2", stage="clean", inputs=["mesh"])
    assert status == 200 and result["diagnostics"]["inputs"]["mesh"]["sha256"] == hashlib.sha256(GLB_BYTES).hexdigest()


@pytest.mark.parametrize("mode", ["rejected", "failed"])
def test_rejected_and_failed_results_pass_through_with_200(client, mode):
    client.put("job-3", "front", PNG)
    status, result = client.run("job-3", mode=mode)
    assert status == 200 and result["status"] == mode and result["error"] == f"fake {mode}"
    assert result["output"] is None
    assert client.request("GET", "/jobs/job-3/output")[0] == 404


def test_a_crash_without_a_result_is_failed(client):
    client.put("job-4", "front", PNG)
    status, result = client.run("job-4", mode="crash")
    assert status == 200 and result["status"] == "failed" and "code 7" in result["error"]
    assert result["engine"]["name"] == "aismith-engine"
    assert result["diagnostics"]["runner"]["returncode"] == 7


def test_an_output_outside_the_work_directory_is_not_served(client):
    client.put("job-5", "front", PNG)
    status, result = client.run("job-5", mode="outside")
    assert status == 200 and result["status"] == "failed" and "outside" in result["error"]
    assert client.request("GET", "/jobs/job-5/output")[0] == 404


def test_one_job_at_a_time_and_cancel(client, work_root):
    client.put("busy-1", "front", PNG)
    client.put("busy-2", "front", PNG)
    thread, box = _background_run(client, "busy-1", mode="sleep", deadline=600)
    pids = _pids(work_root, "busy-1")
    assert client.json("GET", "/health")[1]["busy"] is True
    assert client.run("busy-2") == (409, {"error": "busy"})
    assert client.run("busy-1")[0] == 409
    assert client.put("busy-1", "back", PNG) == 409
    assert client.put("busy-2", "back", PNG) == 204  # another job's uploads are fine

    started = time.monotonic()
    assert client.json("POST", "/jobs/busy-1/cancel") == (202, {"ok": True})
    thread.join(timeout=60)
    status, result = box["response"]
    assert status == 200 and result["status"] == "failed" and result["error"] == "cancelled"
    assert time.monotonic() - started < 30
    _wait_for(lambda: not any(map(_alive, pids)), message="the process group to die")
    assert client.json("GET", "/health")[1]["busy"] is False
    assert client.run("busy-2")[1]["status"] == "ok"


def test_the_deadline_kills_the_process_group(client, work_root):
    client.put("slow", "front", PNG)
    thread, box = _background_run(client, "slow", mode="sleep", deadline=1)
    pids = _pids(work_root, "slow")
    thread.join(timeout=60)
    status, result = box["response"]
    assert status == 200 and result["status"] == "failed" and result["error"] == "deadline exceeded"
    # The grandchild ignores SIGTERM, so this also covers the SIGKILL escalation.
    _wait_for(lambda: not any(map(_alive, pids)), message="the process group to die")


@pytest.mark.parametrize("name, limit", [("front", 20), ("left", 20), ("mesh", 256)])
def test_oversized_inputs_get_413_before_the_body_is_read(client, name, limit):
    status, body = client.headers_only("PUT", f"/jobs/big/inputs/{name}",
                                       {"Content-Length": str(limit * 1024 * 1024 + 1)})
    assert status == 413 and f"{limit} MiB" in body["error"]


def test_an_oversized_run_body_gets_413(client):
    status, _ = client.headers_only("POST", "/jobs/big/run", {"Content-Length": str(64 * 1024 + 1)})
    assert status == 413


def test_content_length_is_required(client):
    status, body = client.headers_only("PUT", "/jobs/j/inputs/front", {})
    assert status == 411 and "Content-Length" in body["error"]
    status, _ = client.headers_only("PUT", "/jobs/j/inputs/front", {"Transfer-Encoding": "chunked"})
    assert status == 411


@pytest.mark.parametrize("path", ["/jobs/bad.id/run", "/jobs/" + "a" * 65 + "/run", "/jobs/a%2Fb/run", "/jobs//run"])
def test_bad_job_ids_get_400(client, path):
    status, body = client.json("POST", path, {"schema_version": 1})
    assert status == 400 and "job id" in body["error"]


@pytest.mark.parametrize("name", ["texture", "front.png", "MESH", "..", ""])
def test_bad_input_names_get_400(client, name):
    assert client.put("j", name, PNG) == 400


@pytest.mark.parametrize("payload, message", [
    (b"{not json", "not JSON"),
    (b"[]", "JSON object"),
    (json.dumps({"schema_version": 2, "stage": "mesh", "inputs": ["front"], "deadline_seconds": 5}).encode(), "schema_version"),
    (json.dumps({"schema_version": 1, "stage": "rig", "inputs": ["front"], "deadline_seconds": 5}).encode(), "stage"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["texture"], "deadline_seconds": 5}).encode(), "inputs"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": [], "deadline_seconds": 5}).encode(), "inputs"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front", "front"], "deadline_seconds": 5}).encode(), "inputs"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front"], "settings": [], "deadline_seconds": 5}).encode(), "settings"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front"], "deadline_seconds": 0}).encode(), "deadline"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front"], "deadline_seconds": True}).encode(), "deadline"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front"]}).encode(), "deadline"),
    (json.dumps({"schema_version": 1, "stage": "mesh", "inputs": ["front"], "deadline_seconds": 5, "x": 1}).encode(), "unknown"),
])
def test_malformed_run_bodies_get_400(client, payload, message):
    client.put("j", "front", PNG)
    status, _, data = client.request("POST", "/jobs/j/run", payload)
    assert status == 400 and message in json.loads(data)["error"]


def test_missing_inputs_and_outputs_get_404(client):
    client.put("j", "front", PNG)
    status, body = client.run("j", inputs=["front", "back"])
    assert status == 404 and "'back'" in body["error"]
    assert client.run("never-uploaded")[0] == 404
    assert client.request("GET", "/jobs/j/output")[0] == 404
    assert client.request("GET", "/nope")[0] == 404
    assert client.request("GET", "/jobs/j/inputs/front")[0] == 404


def test_keep_alive_survives_an_error_response(client):
    connection = http.client.HTTPConnection("127.0.0.1", client.port, timeout=30)
    try:
        connection.request("PUT", "/jobs/k/inputs/texture", body=PNG)
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
    finally:
        connection.close()


def test_delete_removes_the_work_directory(client, work_root):
    client.put("gone", "front", PNG)
    client.run("gone")
    assert (work_root / "gone" / "run" / "mesh.glb").is_file()
    assert client.request("DELETE", "/jobs/gone")[0] == 204
    assert not (work_root / "gone").exists()
    assert client.request("GET", "/jobs/gone/output")[0] == 404
    assert client.request("DELETE", "/jobs/never-existed")[0] == 204


def test_delete_cancels_a_running_job(client, work_root):
    client.put("running", "front", PNG)
    thread, box = _background_run(client, "running", mode="sleep", deadline=600)
    pids = _pids(work_root, "running")
    assert client.request("DELETE", "/jobs/running")[0] == 204
    thread.join(timeout=60)
    assert box["response"][1]["error"] == "cancelled"
    assert not (work_root / "running").exists()
    _wait_for(lambda: not any(map(_alive, pids)), message="the process group to die")


@pytest.mark.parametrize("mode", ["setsid", "native"])
@pytest.mark.parametrize("how", ["cancel", "deadline"])
def test_native_workers_outside_the_stage_are_killed(client, work_root, mode, how):
    """A worker spawned as native.run_process does (its own session by default)
    ignores SIGTERM and is out of reach of the job's group kill; it must still
    be killed and reaped on cancel and on a deadline."""
    client.put("native", "front", PNG)
    thread, box = _background_run(client, "native", mode=mode,
                                  deadline=600 if how == "cancel" else 2)
    stage_pid, child_pid = _pids(work_root, "native")
    stage_group, child_group = _groups(work_root, "native")
    if mode == "setsid":
        assert child_group == child_pid != stage_group  # the fallback case: its own session
    else:
        assert child_group == stage_group  # AISMITH_JOB_PROCESS_GROUP=1 kept it in the job's group
    if how == "cancel":
        assert client.json("POST", "/jobs/native/cancel")[0] == 202
    thread.join(timeout=60)
    assert box["response"][1]["error"] == ("cancelled" if how == "cancel" else "deadline exceeded")
    # Gone and reaped: kill(pid, 0) still succeeds on a zombie.
    _wait_for(lambda: not _alive(child_pid) and not _alive(stage_pid), message="the native worker to die")


def test_the_runner_sets_the_switch_native_honours():
    from backend.app import native

    assert serve.JOB_ENV[native.JOB_PROCESS_GROUP_ENV] == "1"


def test_a_second_runner_on_the_same_root_refuses_to_start(tmp_path):
    root = tmp_path / "root"
    first = serve.Runner(root)
    (root / "live-job" / "inputs").mkdir(parents=True)
    with pytest.raises(ValueError, match="in use by another aismith-engine serve"):
        serve.Runner(root)
    with pytest.raises(ValueError, match="in use"):
        serve.make_server("127.0.0.1", 0, work_root=root, quiet=True)
    assert (root / "live-job" / "inputs").is_dir()  # the first runner's job survived
    first.close()
    second = serve.Runner(root)  # the lock is released with the first runner
    assert not (root / "live-job").exists()
    second.close()


def test_startup_removes_stale_work_directories(tmp_path):
    root = tmp_path / "root"
    serve.Runner(root).close()
    (root / "stale-job" / "run").mkdir(parents=True)
    (root / "stale-job" / "run" / "mesh.glb").write_bytes(GLB_BYTES)
    (root / ".uploads" / "half-written").write_bytes(b"x")
    runner = serve.Runner(root)
    assert sorted(p.name for p in root.iterdir()) == sorted([serve.LOCK_FILE, serve.MARKER, ".uploads"])
    assert not any((root / ".uploads").iterdir()) and runner.jobs == {}
    runner.close()


def test_startup_refuses_to_clean_a_directory_it_does_not_own(tmp_path):
    root = tmp_path / "someone-elses"
    root.mkdir()
    (root / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not a runner work root"):
        serve.Runner(root)
    assert [p.name for p in root.iterdir()] == ["notes.txt"]  # not even the lock file is left behind


@pytest.mark.parametrize("host, allow_private, ok", [
    ("127.0.0.1", False, True),
    ("localhost", False, True),
    ("::1", False, True),
    ("10.0.0.5", False, False),
    ("0.0.0.0", False, False),
    ("::", False, False),
    ("10.0.0.5", True, True),
    ("172.17.0.2", True, True),
    ("0.0.0.0", True, True),
    ("8.8.8.8", True, False),
    ("engine.example.com", True, False),
])
def test_bind_addresses(host, allow_private, ok):
    if ok:
        serve.check_host(host, allow_private=allow_private)
    else:
        with pytest.raises(ValueError):
            serve.check_host(host, allow_private=allow_private)


def test_the_cli_refuses_a_non_loopback_bind(tmp_path, capsys):
    root = tmp_path / "root"
    assert cli.main(["serve", "--host", "10.0.0.5", "--port", "0", "--work-root", str(root)]) == 2
    assert "--allow-private" in capsys.readouterr().err
    assert not root.exists()  # refused before touching the work root
