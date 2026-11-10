from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import struct
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from ..engine import weights as engine_weights
from . import process_tree
from .clip_store import clip_store_path
from .deformation_review import deformation_review_frame_specs
from .deformation_review import resolve_review_focus_bone
from .quality_tiers import QUALITY_TIER_TARGETS, QualityTarget, resolve_quality_target
from .voxel_remesh import voxel_raw_face_band


ROOT = Path(__file__).resolve().parents[2]
Progress = Callable[[str], None]
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None

# Grace period between SIGTERM and SIGKILL when reclaiming a worker's
# process group, and how long we wait after SIGKILL for the OS to confirm
# the pid is actually gone.
_CANCEL_TERM_GRACE_SECONDS = 5.0
_CANCEL_KILL_CONFIRM_SECONDS = 5.0

# Basenames/markers that identify an AISmith-launched GPU worker in a
# process's /proc/<pid>/cmdline. Used to recognize an orphaned worker left
# behind by a previous job (see reclaim_gpu_holders below) -- covers every
# GPU job kind, not just generate.
_KNOWN_WORKER_MARKERS = (
    "trellis2_fp8_worker.py",
    "trellis_refiner_worker.py",
    "blender_worker.py",
    "blender_obj_worker.py",
    "unirig_bind_worker.py",
)


@dataclass(frozen=True)
class CuratedSwordDonor:
    path: Path
    sha256: str
    action_name: str = "Sword_Regular_A"


CURATED_SWORD_DONOR_SHA256 = "406eb0a8dc4ab366e623b79b6e3005a4951392e1bda78ae39c1099d31147733c"


# DEPRECATED for automated rigging: the curated template-donor flow predates
# the UniRig stage and is retained only for the manual template path.
def curated_sword_donor() -> CuratedSwordDonor:
    path = ROOT / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb"
    if path.is_symlink():
        raise RuntimeError(
            f"The bundled grounded Sword Attack library must not be a symlink: {path}"
        )
    if not path.is_file():
        raise RuntimeError(f"The bundled grounded Sword Attack library is missing: {path}")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != CURATED_SWORD_DONOR_SHA256:
        raise RuntimeError(
            "The bundled grounded Sword Attack library SHA-256 does not match "
            f"the pinned value: expected {CURATED_SWORD_DONOR_SHA256}, got {sha256}: {path}"
        )
    return CuratedSwordDonor(path, sha256)


# Set by `aismith-engine serve` for the stage subprocess it runs: native
# workers then stay in the job's process group (no setsid), so the runner's
# group kill on a deadline or cancel reaches them. Unset (the app, and a
# plain `aismith-engine run`), every worker leads its own group as before.
JOB_PROCESS_GROUP_ENV = "AISMITH_JOB_PROCESS_GROUP"


def worker_popen_kwargs() -> dict[str, Any]:
    """Popen kwargs for a native worker: its own session unless the job owns the group."""
    if os.name != "posix" or os.getenv(JOB_PROCESS_GROUP_ENV, "").strip() == "1":
        return {}
    return {"start_new_session": True}


def _set_active_process(process: subprocess.Popen[str] | None) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESS = process


def _active_process_pid() -> int | None:
    """The pid of the currently tracked AISmith worker, if it's still alive."""
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return None
    return process.pid


def _log(progress: Progress | None, message: str) -> None:
    print(f"[gpu-reclaim] {message}", flush=True)
    if progress:
        progress(message)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A process we don't own but that still exists.
        return True
    return True


def terminate_process_group(
    pid: int,
    *,
    progress: Progress | None = None,
    label: str = "worker",
    term_grace_seconds: float = _CANCEL_TERM_GRACE_SECONDS,
    kill_confirm_seconds: float = _CANCEL_KILL_CONFIRM_SECONDS,
    is_alive: Callable[[], bool] | None = None,
) -> bool:
    """SIGTERM a pid's whole process group, escalate to SIGKILL after a
    grace period, and confirm the pid is actually gone before returning.

    AISmith spawns every worker (the TRELLIS.2 FP8 refiner and generator,
    Blender, UniRig) with ``start_new_session=True`` so the worker is its
    own process-group leader; killing the group (not just the pid) catches
    any Blender/python children the worker itself spawned. Returns True once
    the pid is confirmed dead (or was already gone), False if it survived
    SIGKILL.

    ``is_alive`` lets a caller that still owns the ``Popen`` object pass
    ``process.poll`` (which reaps the child via ``waitpid`` and so never
    reports a zombie as alive); it defaults to a raw ``kill(pid, 0)`` probe
    for orphaned pids we're not the direct parent of, where the OS (init)
    reaps the process once it's actually dead.
    """
    alive = is_alive or (lambda: _pid_alive(pid))
    if not alive():
        return True
    if os.name != "posix":
        # AISmith's supported runtime is WSL/Linux; best effort elsewhere.
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return True
        time.sleep(term_grace_seconds)
        if alive():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(kill_confirm_seconds)
        return not alive()

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True
    if pgid != pid or pgid == os.getpgrp():
        # The worker shares a group (AISMITH_JOB_PROCESS_GROUP=1, under the
        # engine runner): a group kill would take down the stage itself, so
        # signal the worker and its descendants one by one.
        return _terminate_tree(pid, progress=progress, label=label, term_grace_seconds=term_grace_seconds,
                               kill_confirm_seconds=kill_confirm_seconds, alive=alive)

    _log(progress, f"Sending SIGTERM to {label} process group {pgid} (pid {pid})")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + term_grace_seconds
    while time.monotonic() < deadline:
        if not alive():
            _log(progress, f"{label} (pid {pid}) exited after SIGTERM")
            return True
        time.sleep(0.2)

    _log(
        progress,
        f"{label} (pid {pid}) still alive {term_grace_seconds:.0f}s after SIGTERM; escalating to SIGKILL",
    )
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + kill_confirm_seconds
    while time.monotonic() < deadline:
        if not alive():
            _log(progress, f"{label} (pid {pid}) killed (SIGKILL)")
            return True
        time.sleep(0.2)

    _log(progress, f"{label} (pid {pid}) did not exit even after SIGKILL")
    return False


def _terminate_tree(
    pid: int,
    *,
    progress: Progress | None,
    label: str,
    term_grace_seconds: float,
    kill_confirm_seconds: float,
    alive: Callable[[], bool],
) -> bool:
    """SIGTERM a pid and its descendants, then SIGKILL whatever is left."""
    tree = {pid: "", **process_tree.descendants(pid)}
    _log(progress, f"Sending SIGTERM to {label} (pid {pid}) and {len(tree) - 1} descendant(s)")
    for target in tree:
        try:
            os.kill(target, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + term_grace_seconds
    while time.monotonic() < deadline:
        if not alive() and not process_tree.alive({k: v for k, v in tree.items() if v}):
            _log(progress, f"{label} (pid {pid}) exited after SIGTERM")
            return True
        time.sleep(0.2)
    _log(progress, f"{label} (pid {pid}) still alive {term_grace_seconds:.0f}s after SIGTERM; escalating to SIGKILL")
    if alive():
        tree.update(process_tree.descendants(pid))
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    process_tree.signal_all({k: v for k, v in tree.items() if v}, signal.SIGKILL)
    deadline = time.monotonic() + kill_confirm_seconds
    while time.monotonic() < deadline:
        if not alive():
            _log(progress, f"{label} (pid {pid}) killed (SIGKILL)")
            return True
        time.sleep(0.2)
    _log(progress, f"{label} (pid {pid}) did not exit even after SIGKILL")
    return False


def cancel_active_process(progress: Progress | None = None) -> bool:
    """Stop the currently running AISmith child worker, if any.

    Terminates the worker's whole process group (SIGTERM, escalating to
    SIGKILL after a grace period) and confirms it actually exited before
    returning, logging each step through ``progress`` so a job's log records
    what happened. Returns True only when a worker was found and confirmed
    dead.
    """
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return False
    pid = process.pid
    # process.poll() reaps via waitpid(WNOHANG) so it never mistakes a
    # SIGKILLed-but-not-yet-reaped zombie for "still alive". The worker's own
    # run_process()/run_trellis() thread also calls process.wait(), and
    # Popen's internal lock makes that safe to race against from here.
    return terminate_process_group(
        pid, progress=progress, label="cancelled job's worker", is_alive=lambda: process.poll() is None
    )


def configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def durable_data_path(*parts: str) -> Path:
    return configured_path("AISMITH_DATA_DIR", ROOT / "studio-data").joinpath(*parts).resolve()


def find_blender() -> Path | None:
    configured = os.getenv("BLENDER_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [
        Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
    ]
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path and path.exists()), None)


def trellis_gguf_model_dir() -> Path:
    """Model root for the trellis2cpp GGUF fallback (vendor/trellis2cpp)."""
    return configured_path("TRELLIS_GGUF_MODEL_DIR", durable_data_path("models", "Trellis2"))


def trellis_gguf_manifest() -> list[tuple[str, str]]:
    """All model files required for the full geometry+texture pipeline.
    Texture models are optional — geometry still works without them."""
    return [
        ("ggufs/dino_f16.gguf",        "ggufs/dino_f16.gguf"),
        ("ggufs/ss_flow_f16.gguf",     "ggufs/ss_flow_f16.gguf"),
        ("ggufs/ss_dec_f16.gguf",      "ggufs/ss_dec_f16.gguf"),
        ("ggufs/slat_flow_f16.gguf",   "ggufs/slat_flow_f16.gguf"),
        ("ggufs/shape_dec_f16.gguf",   "ggufs/shape_dec_f16.gguf"),
        # Texture pipeline (optional but included in the download target)
        ("ggufs/shape_enc_f16.gguf",           "ggufs/shape_enc_f16.gguf"),
        ("ggufs/tex_dec_f16.gguf",             "ggufs/tex_dec_f16.gguf"),
        ("ggufs/tex_slat_flow_512_f16.gguf",   "ggufs/tex_slat_flow_512_f16.gguf"),
        ("ggufs/tex_slat_flow_1024_f16.gguf",  "ggufs/tex_slat_flow_1024_f16.gguf"),
    ]


REQUIRED_GEOMETRY_GGUFS = {
    "ggufs/dino_f16.gguf",
    "ggufs/ss_flow_f16.gguf",
    "ggufs/ss_dec_f16.gguf",
    "ggufs/slat_flow_f16.gguf",
    "ggufs/shape_dec_f16.gguf",
}


def trellis_gguf_missing_files() -> list[Path]:
    """Returns files missing from the required geometry-only set."""
    root = trellis_gguf_model_dir()
    return [root / local for _, local in trellis_gguf_manifest()
            if local in REQUIRED_GEOMETRY_GGUFS
            and (not (root / local).is_file() or (root / local).stat().st_size == 0)]


def trellis_texture_ready() -> bool:
    """True when all four texture GGUFs are present on disk."""
    root = trellis_gguf_model_dir()
    tex_files = [
        root / "ggufs/shape_enc_f16.gguf",
        root / "ggufs/tex_dec_f16.gguf",
        root / "ggufs/tex_slat_flow_512_f16.gguf",
        root / "ggufs/tex_slat_flow_1024_f16.gguf",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in tex_files)


def trellis_gguf_ready() -> bool:
    return not trellis_gguf_missing_files()


def blender_worker_env(*, wsl_lib: Path = Path("/usr/lib/wsl/lib")) -> dict[str, str]:
    """Environment for headless Blender subprocesses doing Cycles GPU work.

    Blender's bundled Python only sees the WSL2 CUDA driver stub
    (``libcuda.so.1``) when ``/usr/lib/wsl/lib`` is on ``LD_LIBRARY_PATH`` at
    process launch -- without it Cycles' ``get_devices()`` reports no CUDA
    device at all and every render/bake silently falls back to CPU (this is
    a WSL2 library-path gap). ``AISMITH_BLENDER_DEVICE`` (read
    by ``blender_worker.py`` to choose OptiX/CUDA/CPU) already passes
    through unchanged via the ``os.environ.copy()`` below.
    """
    env = os.environ.copy()
    if wsl_lib.is_dir():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(wsl_lib) + (os.pathsep + current if current else "")
    return env


_GPU_BUSY_EXTERNAL_THRESHOLD_MB = 12 * 1024
_TRELLIS_LOW_VRAM_FREE_THRESHOLD_MB = 20 * 1024
_GPU_PREFLIGHT_POLL_SECONDS = 15.0
# Was 30 minutes with a silent "proceed anyway" at the end -- see the 2026-08-31
# incident (bead xckqr): a preflight that never fails just hides a genuinely
# stuck GPU from whoever's waiting on the job. 10 minutes, then a hard failure
# naming the holders, so a human finds out instead of a job silently degrading.
_GPU_PREFLIGHT_DEFAULT_MAX_WAIT_SECONDS = 10 * 60


def nvidia_smi_command() -> str:
    """Locate nvidia-smi, including WSL2's copy that is not on PATH."""
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return discovered
    wsl_command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if "microsoft" in platform.release().lower() and wsl_command.exists():
        return str(wsl_command)
    return "nvidia-smi"


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """A point-in-time read of GPU memory pressure from nvidia-smi."""

    available: bool
    used_mb: int
    total_mb: int
    free_mb: int
    busy_external: bool


def gpu_memory_snapshot() -> GpuMemorySnapshot:
    """Query current GPU memory usage.

    ``busy_external`` is true once more than ``_GPU_BUSY_EXTERNAL_THRESHOLD_MB``
    is in use. AISmith's native workers are one-shot subprocesses that free
    their VRAM on exit (see trellis2_fp8_worker.py's module docstring), so at
    any moment none of our own jobs are GPU-resident: usage above the
    threshold can only be another process -- for example a separate
    ComfyUI renderer sharing the same GPU.
    """
    try:
        result = subprocess.run(
            [nvidia_smi_command(), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            used_str, total_str = (part.strip() for part in result.stdout.splitlines()[0].split(","))
            used_mb = int(float(used_str))
            total_mb = int(float(total_str))
            free_mb = max(0, total_mb - used_mb)
            return GpuMemorySnapshot(
                available=True,
                used_mb=used_mb,
                total_mb=total_mb,
                free_mb=free_mb,
                busy_external=used_mb > _GPU_BUSY_EXTERNAL_THRESHOLD_MB,
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return GpuMemorySnapshot(available=False, used_mb=0, total_mb=0, free_mb=0, busy_external=False)


@dataclass(frozen=True)
class GpuHolder:
    """One process currently holding GPU memory, per nvidia-smi."""

    pid: int
    used_mb: int
    cmdline: str
    label: str


def _proc_cmdline(pid: int) -> str:
    """Best-effort ``/proc/<pid>/cmdline`` read, space-joined.

    nvidia-smi's ``--query-compute-apps`` reports process names as
    ``[Not Found]`` under WSL2 (it can't resolve them across the WSL/Windows
    boundary), so pid -> name attribution has to go through /proc ourselves.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part for part in raw.decode("utf-8", "replace").split("\x00") if part)


def _pid_listening_on_port(port: int) -> int | None:
    """The pid bound to ``port`` on localhost, via ``ss -ltnp``, or None."""
    if port <= 0:
        return None
    try:
        result = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    needle = f":{port} "
    for line in result.stdout.splitlines():
        if needle not in line:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _comfyui_base_url() -> str:
    """AISmith's own ComfyUI instance -- never port 8188, a different app."""
    return os.environ.get("AISMITH_COMFYUI_URL", "http://127.0.0.1:8288").strip()


def _comfyui_port(base_url: str) -> int | None:
    try:
        return urlsplit(base_url).port
    except ValueError:
        return None


def gpu_holders() -> list[GpuHolder]:
    """List every process currently holding GPU memory, with pid/cmdline.

    Classification (``label``) is intentionally not done here -- it needs
    context (the active AISmith worker pid, AISmith's ComfyUI port) that this
    function shouldn't have to know about; see classify_gpu_holder().
    """
    try:
        result = subprocess.run(
            [nvidia_smi_command(), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    holders: list[GpuHolder] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used_mb = int(float(parts[1]))
        except ValueError:
            # WSL2's nvidia-smi reports used_memory as "[N/A]" for
            # --query-compute-apps (same gap as its "[Not Found]" process
            # names) even though the pid and the GPU-wide used/total from
            # --query-gpu are both accurate. Still record the holder -- a pid
            # we can't get a per-process byte count for is far more useful
            # than one we silently drop.
            used_mb = 0
        holders.append(GpuHolder(pid=pid, used_mb=used_mb, cmdline=_proc_cmdline(pid), label="unclassified"))
    return holders


def classify_gpu_holder(holder: GpuHolder, *, comfyui_pid: int | None, active_pid: int | None) -> str:
    """Classify one GPU holder so the preflight loop knows what it can do:

    - ``aismith-comfyui``: AISmith's own ComfyUI (bound to AISMITH_COMFYUI_URL's
      port) -- safe to auto-free once its queue is empty.
    - ``aismith-worker-active``: the worker for the job we're currently
      running the preflight on behalf of -- never touch this one here.
    - ``aismith-worker-orphaned``: a process matching a known AISmith worker
      script/binary that isn't the currently tracked active worker. Since
      ModelManager serializes GPU jobs one-at-a-time, any such process
      belongs to a job we're no longer tracking as running (cancelled,
      failed, or crashed) -- safe to reclaim.
    - ``external``: someone else's process (for example a separate ComfyUI
      renderer sharing the same GPU). Never touched.
    """
    if comfyui_pid is not None and holder.pid == comfyui_pid:
        return "aismith-comfyui"
    if active_pid is not None and holder.pid == active_pid:
        return "aismith-worker-active"
    if any(marker in holder.cmdline for marker in _KNOWN_WORKER_MARKERS):
        return "aismith-worker-orphaned"
    return "external"


def describe_gpu_holders(holders: list[GpuHolder]) -> str:
    """Human-readable ``pid (label, N MiB): cmdline`` summary for job logs."""
    if not holders:
        return "no holder identified by nvidia-smi (permissions or a stale reading)"

    def _mb(holder: GpuHolder) -> str:
        # WSL2's nvidia-smi doesn't report a per-process byte count for
        # --query-compute-apps ("[N/A]"), which gpu_holders() records as 0 --
        # never claim a precise number we don't actually have.
        return f"{holder.used_mb} MiB" if holder.used_mb > 0 else "unknown MiB (WSL doesn't report per-process usage)"

    return "; ".join(
        f"pid {holder.pid} ({holder.label}, {_mb(holder)})" + (f": {holder.cmdline[:140]}" if holder.cmdline else "")
        for holder in holders
    )


def _comfyui_queue_empty(base_url: str) -> bool | None:
    """True/False if reachable, None if the request itself failed."""
    try:
        import requests

        response = requests.get(f"{base_url.rstrip('/')}/queue", timeout=3)
        if not response.ok:
            return None
        data = response.json()
        return not data.get("queue_running") and not data.get("queue_pending")
    except Exception:
        return None


def _comfyui_free(base_url: str) -> bool:
    try:
        import requests

        response = requests.post(
            f"{base_url.rstrip('/')}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        return response.ok
    except Exception:
        return False


def classified_gpu_holders(progress: Progress | None = None) -> list[GpuHolder]:
    """gpu_holders(), classified with the current AISmith/ComfyUI context."""
    holders = gpu_holders()
    if not holders:
        return holders
    comfyui_pid = _pid_listening_on_port(_comfyui_port(_comfyui_base_url()) or -1)
    active_pid = _active_process_pid()
    return [
        GpuHolder(
            pid=holder.pid,
            used_mb=holder.used_mb,
            cmdline=holder.cmdline,
            label=classify_gpu_holder(holder, comfyui_pid=comfyui_pid, active_pid=active_pid),
        )
        for holder in holders
    ]


def reclaim_gpu_holders(progress: Progress | None = None) -> list[GpuHolder]:
    """Try to reclaim GPU memory AISmith itself is responsible for.

    Two cases, both logged: (1) AISmith's own ComfyUI is holding parked
    model-cache VRAM with an empty queue -- POST its /free endpoint; (2) a
    holder matches a known AISmith worker script but isn't the worker we're
    currently tracking as active -- it's an orphan from a previous job
    (typically one that was cancelled but whose subprocess outlived the
    cancel), so terminate its process group. External holders (the shared
    renderer, or anything else) are never touched.

    Returns the holders observed *before* any reclaim action, for logging.
    """
    holders = classified_gpu_holders(progress)
    for holder in holders:
        if holder.label == "aismith-comfyui":
            base_url = _comfyui_base_url()
            queue_empty = _comfyui_queue_empty(base_url)
            if queue_empty:
                _log(
                    progress,
                    f"GPU holder pid {holder.pid} is AISmith's own ComfyUI ({base_url}, "
                    f"{holder.used_mb} MiB parked, queue empty) -- freeing its model cache",
                )
                if _comfyui_free(base_url):
                    _log(progress, f"POST {base_url}/free succeeded")
                else:
                    _log(progress, f"POST {base_url}/free failed; ComfyUI's cache may still be resident")
            elif queue_empty is False:
                _log(progress, f"GPU holder pid {holder.pid} is AISmith's own ComfyUI ({base_url}), but its queue is not empty -- leaving it alone")
        elif holder.label == "aismith-worker-orphaned":
            _log(
                progress,
                f"GPU holder pid {holder.pid} ({holder.cmdline[:140]}) looks like an orphaned AISmith "
                "worker with no job still tracking it as active -- terminating its process group",
            )
            terminate_process_group(holder.pid, progress=progress, label=f"orphaned worker pid {holder.pid}")
    return holders


def resolve_trellis_low_vram(snapshot: GpuMemorySnapshot | None = None) -> bool:
    """Decide TRELLIS.2's low_vram (staged CPU/GPU offload) setting.

    ``AISMITH_TRELLIS_LOW_VRAM`` overrides explicitly: "1"/"true" forces
    staged offloading on, "0"/"false" keeps the full FP8 stack resident on
    the GPU. Left at its default ("auto"), the decision is made from free
    VRAM at launch: the 4B FP8 model stack needs roughly 20 GB resident, so
    offloading only kicks in when less is free -- typically because the
    GPU is shared with another process such as a separate ComfyUI renderer.
    """
    override = os.environ.get("AISMITH_TRELLIS_LOW_VRAM", "auto").strip().lower()
    if override in {"1", "true", "on", "yes"}:
        return True
    if override in {"0", "false", "off", "no"}:
        return False
    snapshot = snapshot or gpu_memory_snapshot()
    if not snapshot.available:
        return True
    return snapshot.free_mb < _TRELLIS_LOW_VRAM_FREE_THRESHOLD_MB


def log_device_line(progress: Progress | None, component: str) -> GpuMemorySnapshot:
    """Emit the standard ``device: ...`` line for a native GPU stage."""
    snapshot = gpu_memory_snapshot()
    if snapshot.available:
        message = f"device: cuda ({snapshot.free_mb / 1024:.1f}/{snapshot.total_mb / 1024:.1f} GB free)"
    else:
        message = "device: cpu (no CUDA GPU detected by nvidia-smi)"
    print(f"[{component}] {message}", flush=True)
    if progress:
        progress(message)
    return snapshot


def wait_for_gpu_preflight(
    progress: Progress | None = None,
    *,
    poll_seconds: float = _GPU_PREFLIGHT_POLL_SECONDS,
    max_wait_seconds: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> GpuMemorySnapshot:
    """Block while the shared GPU is held by an external (non-AISmith) process.

    Every poll (``poll_seconds`` apart) it: (1) identifies which pids/names
    hold VRAM, logging a visible countdown against ``max_wait_seconds``; (2)
    tries to reclaim what AISmith itself is responsible for -- its own idle
    ComfyUI's parked model cache, or an orphaned AISmith worker left behind by
    a previous (cancelled/failed) job -- via reclaim_gpu_holders(). If the GPU
    is still busy after ``max_wait_seconds`` (default: the
    ``AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS`` env var, or 10 minutes), it
    raises naming the holders instead of proceeding silently -- a bounded,
    loud failure beats a job that waits forever with no visibility.

    ``cancelled`` (the same ``lambda: job.status == "cancelled"`` convention
    used elsewhere, e.g. run_comfyui_image_edit) is checked every poll too: a
    job cancelled while it's still stuck in this wait -- before any worker
    subprocess even exists for cancel_active_process() to kill -- must not be
    left to spawn a brand new GPU worker once the GPU eventually frees up.
    """
    if max_wait_seconds is None:
        max_wait_seconds = float(
            os.environ.get(
                "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS",
                _GPU_PREFLIGHT_DEFAULT_MAX_WAIT_SECONDS,
            )
        )
    waited = 0.0
    snapshot = gpu_memory_snapshot()
    while snapshot.busy_external and waited < max_wait_seconds:
        if cancelled and cancelled():
            _log(progress, "GPU preflight wait stopped: job was cancelled")
            raise InterruptedError("GPU preflight wait was cancelled")
        holders = reclaim_gpu_holders(progress)
        message = f"GPU busy: {describe_gpu_holders(holders)} -- waiting {int(waited)}s of {int(max_wait_seconds)}s"
        print(message, flush=True)
        if progress:
            progress(message)
        time.sleep(poll_seconds)
        waited += poll_seconds
        snapshot = gpu_memory_snapshot()
    if cancelled and cancelled():
        _log(progress, "GPU preflight wait stopped: job was cancelled")
        raise InterruptedError("GPU preflight wait was cancelled")
    if snapshot.busy_external:
        holders = classified_gpu_holders(progress)
        detail = describe_gpu_holders(holders)
        message = f"GPU still busy after {int(max_wait_seconds)}s ({snapshot.used_mb} MiB used): {detail}"
        print(message, flush=True)
        if progress:
            progress(message)
        raise RuntimeError(f"GPU busy timeout after {int(max_wait_seconds)}s -- holders: {detail}")
    if waited > 0:
        message = "GPU is free; proceeding"
        print(message, flush=True)
        if progress:
            progress(message)
    return snapshot


def _matting_cuda_library_dirs(interpreter: Path) -> list[Path]:
    """Shared-library directories rembg's onnxruntime-gpu needs for CUDA.

    onnxruntime-gpu links against CUDA 13's libcublasLt; the pip
    nvidia-cu13/nvidia-cudnn-cu13 wheels ship those libraries into the same
    venv as torch's own CUDA 12 libs, but nothing puts them on the runtime
    linker's search path by default. ld.so only reads LD_LIBRARY_PATH once,
    at process start, so this must be set on the *subprocess* environment
    before its python starts -- mutating os.environ from inside the worker
    after onnxruntime has already been imported has no effect.
    """
    venv_lib = interpreter.parent.parent / "lib"
    found: list[Path] = []
    for suffix in ("nvidia/cu13/lib", "nvidia/cudnn/lib"):
        found.extend(sorted(venv_lib.glob(f"python3*/site-packages/{suffix}")))
    return [path for path in found if path.is_dir()]


def trellis2_worker_env(interpreter: Path) -> dict[str, str]:
    """Environment for the isolated TRELLIS.2 FP8 worker subprocess.

    Adds the CUDA 13 runtime library directories rembg's onnxruntime-gpu
    needs for matting, so it lands on CUDAExecutionProvider instead of
    silently falling back to CPU.
    """
    env = os.environ.copy()
    extra_dirs = _matting_cuda_library_dirs(interpreter)
    if extra_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        joined = os.pathsep.join(str(directory) for directory in extra_dirs)
        env["LD_LIBRARY_PATH"] = joined + (os.pathsep + current if current else "")
    return env


def _parse_worker_device_report(worker_output: str | None) -> dict[str, Any]:
    """Extract the trellis2_fp8_worker.py ``DEVICE_REPORT_JSON:`` line, if any."""
    if not worker_output:
        return {}
    marker = "DEVICE_REPORT_JSON:"
    for line in worker_output.splitlines():
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return {}
    return {}


def instant_meshes_dir() -> Path:
    """Managed install location for the official Instant Meshes bundle."""
    return configured_path("INSTANT_MESHES_DIR", durable_data_path("tools", "instant-meshes"))


def instant_meshes_exe() -> Path | None:
    configured = os.getenv("INSTANT_MESHES_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    install = instant_meshes_dir()
    names = ("Instant Meshes.exe", "InstantMeshes.exe") if sys.platform == "win32" else ("Instant Meshes", "InstantMeshes")
    candidates += [install / name for name in names]
    if install.exists():
        for name in names:
            candidates += install.rglob(name)
    return next(
        (
            path.resolve()
            for path in candidates
            if path.exists() and (sys.platform == "win32" or os.access(path, os.X_OK))
        ),
        None,
    )


def trellis_refiner_dir() -> Path:
    """Managed, isolated ComfyUI runtime for image-guided TRELLIS.2 FP8 refine."""
    return configured_path("TRELLIS_REFINER_DIR", durable_data_path("tools", "trellis2-refiner"))


def kimodo_dir() -> Path:
    """Managed, isolated NVIDIA Kimodo runtime."""
    return configured_path("KIMODO_DIR", durable_data_path("tools", "kimodo"))


def kimodo_executable() -> Path | None:
    runtime = kimodo_dir()
    configured = os.getenv("KIMODO_EXECUTABLE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend((runtime / ".venv" / "bin" / "kimodo_gen", runtime / ".venv" / "Scripts" / "kimodo_gen.exe"))
    return next((candidate.expanduser().resolve() for candidate in candidates if candidate.exists()), None)


def trellis_refiner_python() -> Path | None:
    """Select a refiner interpreter that can actually import PyTorch."""
    runtime = trellis_refiner_dir()
    configured = os.getenv("TRELLIS_REFINER_PYTHON", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend((
        runtime / ".venv-cu128" / "bin" / "python",
        runtime / ".venv" / "bin" / "python",
        runtime / ".venv-cu128" / "Scripts" / "python.exe",
        runtime / ".venv" / "Scripts" / "python.exe",
    ))
    checked: set[Path] = set()
    probe = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in checked or not candidate.exists():
            continue
        checked.add(candidate)
        try:
            result = subprocess.run(
                [str(candidate), "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


def offline_mode() -> bool:
    """AISMITH_OFFLINE=1: stages never download; weights come from the installed roots."""
    return os.getenv("AISMITH_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def engine_weight_roots() -> dict[str, Path]:
    """Where each weights-lock root lives on this machine."""
    return {
        "trellis_models": trellis_refiner_dir() / "models",
        "unirig": unirig_dir() or Path(os.getenv("UNIRIG_DIR", "").strip() or ROOT / "vendor" / "unirig").expanduser(),
        # rembg 2.0.81 keeps models in <home>/models/<name>/; its home is
        # U2NET_HOME when set, else REMBG_HOME, else ~/.rembg.
        "u2net": Path(
            os.getenv("U2NET_HOME", "").strip() or os.getenv("REMBG_HOME", "").strip() or Path.home() / ".rembg"
        ).expanduser(),
    }


def trellis_refiner_ready() -> bool:
    """Whether the isolated FP8 TRELLIS.2 refinement runtime is ready."""
    runtime = trellis_refiner_dir()
    return trellis_refiner_python() is not None and all(path.exists() for path in (
        runtime / "ComfyUI" / "folder_paths.py",
        runtime / "ComfyUI" / "custom_nodes" / "ComfyUI-Trellis2" / "nodes.py",
        runtime / "models" / "visualbruno" / "TRELLIS.2-4B-FP8" / "pipeline_fp8.json",
        runtime / "models" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m" / "model.safetensors",
        runtime / "models" / "microsoft" / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16.safetensors",
    ))


def run_process(command: list[str], *, cwd: Path = ROOT, timeout: int = 3600, progress: Progress | None = None, env: dict[str, str] | None = None) -> str:
    if progress:
        progress(f"Launching {Path(command[0]).name}")
    merged = os.environ.copy()
    if offline_mode():
        merged.update(engine_weights.offline_environment(engine_weight_roots()))
    if env:
        merged.update(env)
    # Last, so a worker environment (built from os.environ before offline mode
    # was set, say) can never switch a stage back online.
    if offline_mode():
        merged.update(engine_weights.offline_environment(engine_weight_roots()))
    # start_new_session=True (setsid) makes this worker the leader of its
    # own process group, so cancel_active_process()/terminate_process_group()
    # can kill the whole group -- including any Blender/python children the
    # worker itself spawns -- instead of just this one pid. Under the engine
    # runner (AISMITH_JOB_PROCESS_GROUP=1) it stays in the job's group instead.
    popen_kwargs = worker_popen_kwargs()
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=merged, **popen_kwargs
    )
    _set_active_process(process)
    lines: list[str] = []

    def relay() -> None:
        if process.stdout is None:
            return
        for raw in process.stdout:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            if progress:
                progress(line[:500])

    reader = threading.Thread(target=relay, name=f"aismith-{Path(command[0]).stem}-log", daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process_group(
            process.pid, label=Path(command[0]).name, is_alive=lambda: process.poll() is None
        )
        raise RuntimeError(f"{Path(command[0]).name} exceeded its {timeout // 60}-minute time limit") from exc
    finally:
        reader.join(timeout=5)
        _set_active_process(None)
    output = "\n".join(lines).strip()
    if return_code != 0:
        raise RuntimeError(f"{Path(command[0]).name} failed ({return_code}): {output[-1800:]}")
    return output


def blender_convert(input_path: Path, output_path: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for GLB/OBJ conversion but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "convert", "input": str(input_path), "output": str(output_path)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=600)


def blender_export_obj(input_path: Path, output_path: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for GLB/OBJ conversion but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "export-obj", "input": str(input_path), "output": str(output_path)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)


def _glb_has_material(source_glb: Path) -> bool:
    with source_glb.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            return False
        json_length, _ = struct.unpack("<II", source.read(8))
        document = json.loads(source.read(json_length).decode("utf-8"))
    return any(
        "material" in primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    )


def _glb_has_texture(source_glb: Path) -> bool:
    with source_glb.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            return False
        json_length, _ = struct.unpack("<II", source.read(8))
        document = json.loads(source.read(json_length).decode("utf-8"))
    images = document.get("images", [])
    textures = document.get("textures", [])
    materials = document.get("materials", [])
    if not images or not textures:
        return False
    return any(
        isinstance(material.get("pbrMetallicRoughness", {}).get("baseColorTexture"), dict)
        for material in materials
    )


def _glb_mesh_nodes_have_identity_transforms(source_glb: Path) -> bool:
    with source_glb.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            return False
        json_length, _ = struct.unpack("<II", source.read(8))
        document = json.loads(source.read(json_length).decode("utf-8"))
    transform_keys = {"matrix", "rotation", "scale", "translation"}
    return all(
        not transform_keys.intersection(node)
        for node in document.get("nodes", [])
        if "mesh" in node
    )


def _glb_mesh_vertices_are_y_up(source_glb: Path) -> bool:
    with source_glb.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            return False
        json_length, _ = struct.unpack("<II", source.read(8))
        document = json.loads(source.read(json_length).decode("utf-8"))
    accessors = document.get("accessors", [])
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None or position_index >= len(accessors):
                continue
            accessor = accessors[position_index]
            minimum = accessor.get("min")
            maximum = accessor.get("max")
            if minimum is None or maximum is None:
                continue
            extents = [maximum[index] - minimum[index] for index in range(3)]
            return extents[1] > extents[2]
    return False


def _glb_position_extents(source_glb: Path) -> list[float] | None:
    """Overall POSITION bounding-box extents (x, y, z) across every primitive, or None."""
    with source_glb.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            return None
        json_length, _ = struct.unpack("<II", source.read(8))
        document = json.loads(source.read(json_length).decode("utf-8"))
    accessors = document.get("accessors", [])
    lows: list[float] | None = None
    highs: list[float] | None = None
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None or position_index >= len(accessors):
                continue
            accessor = accessors[position_index]
            minimum = accessor.get("min")
            maximum = accessor.get("max")
            if minimum is None or maximum is None:
                continue
            lows = minimum[:3] if lows is None else [min(a, b) for a, b in zip(lows, minimum[:3])]
            highs = maximum[:3] if highs is None else [max(a, b) for a, b in zip(highs, maximum[:3])]
    if lows is None or highs is None:
        return None
    return [high - low for high, low in zip(highs, lows)]


def _glb_axis_profile_matches(source_glb: Path, output_glb: Path, tolerance: float = 0.15) -> bool:
    """The retopology output must keep the source's axis profile.

    The source GLB is already Y-up; a correctly baked output has the same
    per-axis extent ratios. A Y/Z swap (axis conversion left in a node
    transform instead of baked into the vertices) shows up as the output's Y
    extent matching the source's Z extent and vice versa. This replaces the
    old "taller than deep" heuristic, which rejected every correctly baked
    subject that lies flat (a serpent, a vehicle, a prone figure).
    """
    source_extents = _glb_position_extents(source_glb)
    output_extents = _glb_position_extents(output_glb)
    if output_extents is None:
        return False
    if source_extents is None:
        return True
    source_scale = max(source_extents) or 1.0
    output_scale = max(output_extents) or 1.0
    source_profile = [value / source_scale for value in source_extents]
    output_profile = [value / output_scale for value in output_extents]
    source_y, source_z = source_profile[1], source_profile[2]
    output_y, output_z = output_profile[1], output_profile[2]
    axes_distinct = abs(source_y - source_z) > tolerance
    swapped = (
        axes_distinct
        and abs(output_y - source_z) <= tolerance
        and abs(output_z - source_y) <= tolerance
        and (abs(output_y - source_y) > tolerance or abs(output_z - source_z) > tolerance)
    )
    if swapped:
        return False
    return all(abs(out - ref) <= tolerance for out, ref in zip(output_profile, source_profile))


# Palette fidelity (see texture_bake.py's "Palette fidelity" section). ON by
# default: TRELLIS.2 inventing a colour scheme for a stylised source is the
# common case, not the exception, and a full-strength match against a source
# that ALREADY has the right palette is a near-no-op (measured on the alien
# chain, 2026-09-01).
DEFAULT_PALETTE_TRANSFER_STRENGTH = 1.0
# A run's asset lineage is short (image -> generate -> canonicalize -> ...);
# the bound stops a corrupted `source_artifact_id` cycle from walking the
# whole artifact store. Mirrors main.POLICY_LINEAGE_DEPTH's reasoning.
PALETTE_REFERENCE_LINEAGE_DEPTH = 8


def blender_bake_retopology(
    source_glb: Path,
    remeshed_obj: Path,
    output_glb: Path,
    progress: Progress | None = None,
    *,
    palette_reference: Path | None = None,
    palette_strength: float = DEFAULT_PALETTE_TRANSFER_STRENGTH,
    atlas_minimum: int | None = None,
) -> dict[str, Any]:
    """Bake the source's appearance AND its surface normals onto the remeshed
    target.

    `atlas_minimum` raises the bake atlas's FLOOR for a dense quality tier
    (bead i63ky). It never caps: the size still follows the source's largest
    texture up to `texture_bake.BAKE_TEXTURE_SIZE_MAXIMUM` (4K), so a High or
    Ultra mesh cannot be handed a 1K atlas just because its source textures
    were small, and a 4K-textured source still bakes 4K at every tier.

    `palette_reference`, when given, is the 2D image the generator was
    conditioned on; the worker re-statistics the baked albedo onto its palette
    (see `texture_bake.transfer_palette`) before the bake-miss dilation, which
    is what stops a monochrome source from shipping an invented colour scheme.

    Returns the worker's bake provenance report (atlas size, the source
    texture size it was derived from, packed UV utilisation, bake-miss
    fraction, the `palette_transfer` sub-report, and the `normal_bake`
    sub-report: tangent space, green-channel convention, colour space and the
    target shading the tangent frame was taken against) so callers can record
    WHY a canonical artifact looks the way it does. An empty dict means the
    worker wrote no report (older worker, or a mocked boundary in tests).
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to bake source textures onto the retopologized mesh")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    report_path = output_glb.with_suffix(".bake.json")
    spec = json.dumps({
        "op": "retopo-bake",
        "source": str(source_glb),
        "target": str(remeshed_obj),
        "output": str(output_glb),
        "bake_appearance": _glb_has_material(source_glb),
        "report": str(report_path),
        "palette_reference": str(palette_reference) if palette_reference else None,
        "palette_strength": float(palette_strength),
        "atlas_minimum": int(atlas_minimum) if atlas_minimum else None,
    })
    log_device_line(progress, "blender-retopo-bake")
    try:
        run_process(
            [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
            progress=progress,
            timeout=1800,
            env=blender_worker_env(),
        )
        if not _glb_mesh_nodes_have_identity_transforms(output_glb) or not _glb_axis_profile_matches(source_glb, output_glb):
            raise RuntimeError("Retopology output must bake the source axis orientation into mesh vertices before rigging (node transform or Y/Z swap detected)")
        if report_path.is_file():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {}
    finally:
        report_path.unlink(missing_ok=True)


def run_project_views(
    input_glb: Path,
    output_glb: Path,
    view_images: dict[str, Path],
    settings: dict[str, Any],
    progress: Progress | None = None,
) -> dict[str, Any]:
    """S2 view-projection: bake the T-pose conditioning views onto the
    canonical mesh's own UV atlas, blended over its existing base-color bake
    (`blender_worker.project_views`; see `view_projection.py`'s module
    docstring for why this recovers evidence a generative texture head can
    only approximate).

    Returns the worker's `view_projection.projection_report`-shaped provenance
    (per-role coverage, projected/fallback texel fractions, occlusion-
    rejection per role, the OKLab colour delta against the prior bake, and
    each role's frame alignment) plus `{"backend": "blender-view-projection"}`.
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to project conditioning views onto the canonical texture")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    report_path = Path(str(output_glb) + ".projection.json")
    spec = json.dumps({
        "op": "project-views",
        "input": str(input_glb),
        "output": str(output_glb),
        "views": {role: str(path) for role, path in view_images.items()},
        "report": str(report_path),
        "settings": settings,
    })
    log_device_line(progress, "blender-view-projection")
    try:
        run_process(
            [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
            progress=progress,
            timeout=900,
            env=blender_worker_env(),
        )
        # Blender exits 0 even when the worker script raises; the report and
        # output are the only proof the op ran to completion.
        if not report_path.is_file() or not output_glb.is_file():
            raise RuntimeError("project-views produced no output; see the Blender log above")
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        report["backend"] = "blender-view-projection"
        return report
    finally:
        report_path.unlink(missing_ok=True)


# Rebuild-from-views drives the Pixal3D multi-view research scripts under
# scripts/pixal_mv/. They are optional and not every checkout carries them (the
# public repository leaves them out), so the stage is offered only when its
# entry script is present.
REBUILD_VIEWS_SCRIPT = ROOT / "scripts" / "pixal_mv" / "approach9.sh"


def rebuild_views_available() -> bool:
    return REBUILD_VIEWS_SCRIPT.is_file()


def run_rebuild_views(
    input_glb: Path,
    output_glb: Path,
    view_images: dict[str, Path],
    settings: dict[str, Any],
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Rebuild the mesh from its own painted renders, then lay the real
    reference pixels back over the rebuild (`scripts/pixal_mv/approach9.sh` --
    its header documents the five steps and the single agent-in-the-loop
    decision, which is which reference views to pass).

    `overlaid.glb` (the real pixels over the Pixal3D multi-view rebuild) is
    the deliverable and becomes `output_glb`; the raw rebuild stays in the
    workdir as `final.glb`. Returns the script's own IoU report
    (`report.json`) and multiview gate table alongside both workdir paths.
    """
    front = view_images.get("front")
    if front is None:
        raise ValueError("Rebuilding from views requires a front reference view")
    if not rebuild_views_available():
        raise RuntimeError(
            "Rebuilding from views needs the optional Pixal3D multi-view scripts "
            "(scripts/pixal_mv/), which this checkout does not include"
        )
    workdir = Path(f"{output_glb.with_suffix('')}-rebuild")
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "bash", str(REBUILD_VIEWS_SCRIPT),
        "--mesh", str(input_glb),
        "--front", str(front),
        "--out", str(workdir),
    ]
    for role in ("back", "left", "right"):
        path = view_images.get(role)
        if path is not None:
            command += [f"--{role}", str(path)]
    log_device_line(progress, "pixal3d-mv-rebuild")
    # ~6 min alone, but the script waits up to 90 min for a shared ComfyUI to
    # release the GPU, and one shared run took 40 min (2026-09-05).
    output = run_process(command, timeout=3 * 3600, progress=progress, env=blender_worker_env())
    # Blender and the sub-scripts exit 0 even when a step fails; the files are
    # the only proof the process ran to completion (as in run_project_views).
    overlaid = workdir / "overlaid.glb"
    if not overlaid.is_file():
        raise RuntimeError(f"rebuild-views produced no overlaid.glb; script output:\n{output[-1800:]}")
    shutil.copy2(overlaid, output_glb)
    gate_path = workdir / "overlay" / "gate" / "multiview.json"
    report_path = workdir / "report.json"
    return {
        "backend": "pixal3d-mv",
        "mv_iou": json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None,
        "gate": json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None,
        "pixal_mv_glb": str(workdir / "final.glb"),
        "workdir": str(workdir),
    }


def build_projection_fixture(
    output_glb: Path,
    atlas_png: Path,
    meta_json: Path,
    *,
    atlas_size: int = 512,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build the S2 project-views conformance fixture (task5): a synthetic
    mesh whose UV atlas is coloured by dominant world-normal axis, so a
    wrong-face or mirrored projection reads as a wrong colour rather than
    requiring a render-and-look guess.

    Returns the worker's `meta_json` contents (bounds, diagonal, atlas size,
    the axis->colour key).
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to build the projection conformance fixture")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "build-projection-fixture",
        "output": str(output_glb),
        "atlas_png": str(atlas_png),
        "meta_json": str(meta_json),
        "atlas_size": int(atlas_size),
    })
    log_device_line(progress, "blender-projection-fixture")
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=600,
        env=blender_worker_env(),
    )
    # Blender exits 0 even when the worker script raises; the outputs are
    # the only proof the op ran to completion.
    if not output_glb.is_file() or not atlas_png.is_file() or not meta_json.is_file():
        raise RuntimeError("build-projection-fixture produced no output; see the Blender log above")
    return json.loads(meta_json.read_text(encoding="utf-8"))


def render_role_views(
    input_glb: Path,
    output_dir: Path,
    roles: tuple[str, ...] = ("front", "back", "left", "right"),
    *,
    size: int = 512,
    projection: str = "orthographic",
    fov_x_degrees: float | None = None,
    pitch_degrees: float = 0.0,
    progress: Progress | None = None,
) -> dict[str, Path]:
    """Render each role's conditioning view of `input_glb` for the S2
    project-views conformance fixture (task5), with a DIFFERENT framing
    margin than `run_project_views` uses so the conformance test proves
    `fit_frame_to_subject` actually re-fits.

    `projection="perspective"` (with `fov_x_degrees`, optionally
    `pitch_degrees`) renders from a photo-shaped camera instead -- the
    source-pose chain's case. Each role's exact camera lands in
    `output_dir / "views.json"` in the shape `run_project_views`'
    `settings["cameras"]` takes.

    Returns `{role: png_path}`.
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to render the projection conformance fixture's views")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.dumps({
        "op": "render-role-views",
        "input": str(input_glb),
        "output_dir": str(output_dir),
        "roles": list(roles),
        "size": int(size),
        "projection": projection,
        "fov_x_degrees": fov_x_degrees,
        "pitch_degrees": float(pitch_degrees),
    })
    log_device_line(progress, "blender-projection-fixture-render")
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=600,
        env=blender_worker_env(),
    )
    result: dict[str, Path] = {}
    for role in roles:
        path = output_dir / f"{role}.png"
        if not path.is_file():
            raise RuntimeError(f"render-role-views produced no output for role {role!r}; see the Blender log above")
        result[role] = path
    return result


def render_guide_views(
    input_glb: Path,
    output_dir: Path,
    *,
    roles: tuple[str, ...] = ("front", "back", "left", "right"),
    size: int = 1024,
    extra_roles: tuple[str, ...] = (),
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Render CLAY + NORMAL guide images per role from exact cameras (task9,
    mesh-guided view synthesis): `blender_worker.render_guide_views` renders
    each view FROM this mesh, so it and its camera always match the geometry
    exactly, and hands each camera back so `run_project_views` can reuse it
    verbatim via `settings["cameras"]` instead of re-deriving one.

    Returns the worker's `guides.json` contents: `image_size`, `roles`
    (per-role azimuth/center/ortho_scale/camera_distance plus clay/normal
    PNG paths), `bounds_low`/`bounds_high`, `diagonal`.
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to render guide views")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.dumps({
        "op": "render-guide-views",
        "input": str(input_glb),
        "output_dir": str(output_dir),
        "roles": list(roles),
        "extra_roles": list(extra_roles),
        "size": int(size),
    })
    log_device_line(progress, "blender-guide-views")
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=900,
        env=blender_worker_env(),
    )
    guides_path = output_dir / "guides.json"
    if not guides_path.is_file():
        raise RuntimeError("render-guide-views produced no guides.json; see the Blender log above")
    return json.loads(guides_path.read_text(encoding="utf-8"))


def run_repair_weights(input_glb: Path, output_glb: Path, progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to repair skin weights")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "repair-weights", "input": str(input_glb), "output": str(output_glb)})
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
    )
    return {"backend": "blender-automatic-weights", "preserved_armature": True}


def run_rig_transfer(
    input_glb: Path,
    donor_glb: Path,
    output_glb: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to transfer a humanoid rig")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "rig-transfer",
        "input": str(input_glb),
        "donor": str(donor_glb),
        "output": str(output_glb),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
    )
    return {
        "backend": "blender-rig-transfer",
        "weighting": "donor-surface-barycentric",
        "preserved_donor_animations": True,
    }


def run_target_proxy_rig(
    input_glb: Path,
    donor_glb: Path,
    output_glb: Path,
    progress: Progress | None = None,
    *,
    action_name: str = "Sword_Attack",
    attach_sword: bool = True,
    deformation_surface: str = "visible-original",
) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to build a target-derived deformation proxy")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "target-proxy-rig",
        "input": str(input_glb),
        "donor": str(donor_glb),
        "output": str(output_glb),
        "action_name": action_name,
        "attach_sword": attach_sword,
        "deformation_surface": deformation_surface,
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
    )
    return {
        "backend": "blender-target-proxy-rig",
        "weighting": "target-voxel-proxy",
        "preserved_donor_animations": True,
        "source_action": action_name,
        "attached_prop": "right-hand-sword" if attach_sword else None,
        "deformation_surface": deformation_surface,
    }


def deformation_review_bone_names(input_glb: Path) -> list[str]:
    """Read armature bone names from a GLB without Blender (JSON chunk scan)."""
    data = input_glb.read_bytes()
    chunk_length = struct.unpack_from("<II", data, 12)[0]
    document = json.loads(data[20:20 + chunk_length])
    names = [
        str(node.get("name"))
        for node in document.get("nodes", [])
        if node.get("name") and "mesh" not in node and "skin" not in node
    ]
    return names


def run_deformation_diagnostics(
    input_glb: Path,
    output_json: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for deformation diagnostics")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "deformation-diagnostics",
        "input": str(input_glb),
        "output": str(output_json),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=900,
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    review_files = []
    try:
        log_device_line(progress, "blender-render-review")
        for label, ratio, rest_pose, focus_bone in deformation_review_frame_specs():
            if focus_bone is not None:
                try:
                    armature_bones = deformation_review_bone_names(input_glb)
                    resolve_review_focus_bone(armature_bones, focus_bone)
                except ValueError as resolution_error:
                    report.setdefault("skipped_review_frames", []).append({
                        "label": label,
                        "focus_bone": focus_bone,
                        "reason": str(resolution_error),
                    })
                    continue
            image_path = output_json.with_name(f"{output_json.stem}-{label}.png")
            image_spec = json.dumps({
                "op": "render-review",
                "input": str(input_glb),
                "output": str(image_path),
                "frame_ratio": ratio,
                "rest_pose": rest_pose,
                "focus_bone": focus_bone,
            })
            run_process(
                [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", image_spec],
                progress=progress,
                timeout=600,
                env=blender_worker_env(),
            )
            review_files.append({
                "label": label,
                "ratio": ratio,
                "rest_pose": rest_pose,
                "focus_bone": focus_bone,
                "path": str(image_path),
            })
    except Exception:
        for item in review_files:
            Path(item["path"]).unlink(missing_ok=True)
        raise
    report["_review_files"] = review_files
    return report


def run_skeleton_sheet(input_glb: Path, progress: Progress | None = None) -> dict[str, Any]:
    """Render the front/right/left/back bone contact sheet used for visual
    left/right review, and compose the four views into one `sheet.png`.

    Unlike `run_deformation_diagnostics`, there is no caller-supplied report
    path to anchor the raw views next to -- this op owns its own throwaway
    directory. All five files (the composed sheet plus the four raw views)
    come back as `_review_files` for the caller to persist as artifacts and
    then delete, the same contract `run_deformation_diagnostics` uses.
    """
    import tempfile
    from PIL import Image, ImageDraw

    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for the skeleton contact sheet")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    temporary = Path(tempfile.mkdtemp(prefix="aismith-skeleton-sheet-"))
    review_files: list[dict[str, Any]] = []
    try:
        spec = json.dumps({
            "op": "render-skeleton-sheet",
            "input": str(input_glb),
            "output_dir": str(temporary),
            "size": 768,
        })
        run_process(
            [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
            progress=progress,
            timeout=900,
            env=blender_worker_env(),
        )
        roles = ("front", "right", "left", "back")
        views: dict[str, Any] = {}
        for role in roles:
            view_path = temporary / f"{role}.png"
            if not view_path.exists():
                raise RuntimeError(f"Skeleton sheet render did not produce {role}.png")
            views[role] = Image.open(view_path)
            review_files.append({"label": role, "path": str(view_path)})

        width, height = views["front"].size
        sheet = Image.new("RGB", (width * 2, height * 2), (12, 12, 12))
        # front, right / left, back -- the quadrant order the front camera
        # naturally reads left-to-right, top-to-bottom.
        quadrants = {
            "front": (0, 0), "right": (width, 0),
            "left": (0, height), "back": (width, height),
        }
        draw = ImageDraw.Draw(sheet)
        for role, (x, y) in quadrants.items():
            sheet.paste(views[role].convert("RGB"), (x, y))
            draw.text((x + 12, y + 12), role, fill=(255, 255, 255))
        for view in views.values():
            view.close()

        sheet_path = temporary / "sheet.png"
        sheet.save(sheet_path)
        review_files.insert(0, {"label": "skeleton-sheet", "path": str(sheet_path)})
        # Hand the five PNGs back as standalone temp files so the working
        # directory can be removed below; the caller persists each path as
        # an artifact and unlinks it (`persist_skeleton_review_artifacts`).
        for item in review_files:
            handle, standalone = tempfile.mkstemp(
                prefix=f"aismith-skeleton-{item['label']}-", suffix=".png",
            )
            os.close(handle)
            shutil.move(item["path"], standalone)
            item["path"] = standalone
    except Exception:
        for item in review_files:
            Path(item["path"]).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {"_review_files": review_files}


def run_kimodo_motion(
    prompt: str,
    output_bvh: Path,
    *,
    duration: float,
    seed: int,
    progress: Progress | None = None,
) -> dict[str, Any]:
    executable = kimodo_executable()
    if executable is None:
        raise RuntimeError("NVIDIA Kimodo is not installed in the managed tools directory")
    model = os.getenv("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1").strip()
    cache = configured_path("AISMITH_MODELS_DIR", durable_data_path("models")) / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    token_path = os.getenv("HF_TOKEN_PATH", "").strip()
    if not token_path:
        incoming_hf_home = os.getenv("HF_HOME", "").strip()
        token_candidates = (
            [Path(incoming_hf_home).expanduser() / "token"] if incoming_hf_home else []
        )
        token_candidates.append(Path.home() / ".cache" / "huggingface" / "token")
        token_path = next((str(candidate) for candidate in token_candidates if candidate.is_file()), "")
    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    stem = output_bvh.with_suffix("")
    command = [
        str(executable),
        prompt,
        "--model", model,
        "--duration", str(duration),
        "--num_samples", "1",
        "--seed", str(seed),
        "--output", str(stem),
        "--bvh",
        "--bvh_standard_tpose",
    ]
    environment = {
        "HF_HOME": str(cache.resolve()),
        "TEXT_ENCODER_DEVICE": os.getenv("KIMODO_TEXT_ENCODER_DEVICE", "cpu"),
    }
    if token_path:
        environment["HF_TOKEN_PATH"] = token_path
    try:
        run_process(
            command,
            cwd=kimodo_dir() if kimodo_dir().is_dir() else executable.parent,
            timeout=3600,
            progress=progress,
            env=environment,
        )
    except RuntimeError as error:
        detail = str(error)
        normalized_detail = detail.casefold()
        encoder_access_denied = any(
            marker in normalized_detail
            for marker in (
                "not in the authorized list",
                "requires approval",
                "awaiting a review",
                "access to this model is restricted",
            )
        )
        if "meta-llama/Meta-Llama-3-8B-Instruct" in detail and (
            encoder_access_denied
        ):
            raise RuntimeError(
                "Kimodo is installed and authenticated, but its LLM2Vec text encoder "
                "requires separate Meta Llama 3 access. Request it at "
                "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct, "
                "then retry this retained stage."
            ) from error
        raise
    if not output_bvh.exists() or output_bvh.stat().st_size < 64:
        raise RuntimeError("Kimodo completed without producing a valid BVH motion")

    # Kimodo writes standard mocap Y-up BVH; every other BVH artifact in
    # this repo is Blender-native Z-up (export_soma_bvh in
    # unirig_bind_worker.py). Normalize here, at this ingest boundary, per
    # docs/PIPELINE_POLICY.md ("normalize at the ingest boundary" /
    # "blame upstream") -- retarget_bvh must see exactly one convention
    # and never special-cases a Kimodo-sourced clip.
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to normalize Kimodo motion axes")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    normalized_bvh = output_bvh.with_name(output_bvh.stem + ".normalized.bvh")
    normalize_spec = json.dumps({
        "op": "normalize-bvh",
        "input": str(output_bvh),
        "output": str(normalized_bvh),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", normalize_spec],
        progress=progress,
        timeout=300,
    )
    if not normalized_bvh.exists() or normalized_bvh.stat().st_size < 64:
        raise RuntimeError("Blender completed without producing a normalized BVH motion")
    normalized_bvh.replace(output_bvh)

    return {
        "backend": "nvidia-kimodo",
        "model": model,
        "prompt": prompt,
        "duration_seconds": duration,
        "seed": seed,
        "standard_t_pose": True,
        "axes": "blender-z-up",
        "axes_normalized_from": "y-up",
    }


def run_bvh_retarget(
    character_glb: Path,
    motion_bvh: Path,
    output_glb: Path,
    *,
    target_bones: dict[str, str],
    animation: str = "Sword_Attack",
    progress: Progress | None = None,
) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to retarget Kimodo motion")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "retarget-bvh",
        "input": str(character_glb),
        "motion": str(motion_bvh),
        "output": str(output_glb),
        "animation": animation,
        "target_bones": target_bones,
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
    )
    if not output_glb.exists() or output_glb.stat().st_size < 64:
        raise RuntimeError("Blender completed without producing a valid animated GLB")
    return {
        "backend": "blender-soma-bvh-retarget",
        "animation": animation,
        "motion_artifact_format": "bvh",
        "baked_keyframes": True,
        "target_bones": target_bones,
    }


def run_apply_clip(
    character_glb: Path,
    output_glb: Path,
    *,
    family: str,
    clip_name: str,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """N5: direct-attach a family clip-store action onto a template-conformant
    rig by exact bone-name match -- no retarget math. This is the payoff of
    N2 (fixed skeleton contract) + N3 (rigs conform to the template) + N4
    (the clip store is pre-ported onto that same template): the rig and the
    clip store share identical bone names, so the store's action can be
    assigned to the target armature verbatim.

    See backend/app/clip_store.py for the manifest/lookup contract and
    backend/workers/blender_worker.py's apply_clip op for the bone-coverage
    hard-fail (s5.exact_bone_name_match).
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to attach a clip")
    store_path = clip_store_path(family)
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "apply-clip",
        "input": str(character_glb),
        "store": str(store_path),
        "output": str(output_glb),
        "clip_name": clip_name,
        "family": family,
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=900,
    )
    if not output_glb.exists() or output_glb.stat().st_size < 64:
        raise RuntimeError("Blender completed without producing a valid animated GLB")
    return {
        "backend": "blender-direct-attach",
        "clip": clip_name,
        "family": family,
        "mechanism": "direct-attach",
        "baked_keyframes": True,
    }


def run_slither_bake(
    input_glb: Path,
    output_glb: Path,
    animation: str,
    settings: dict[str, Any],
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Bake a procedural travelling-sine 'Slither' cycle onto a chain-rigged
    creature (see backend/workers/blender_worker.py::slither_bake and
    backend/app/slither_math.py). Mirrors
    `run_bvh_retarget`'s worker-invocation shape; there is no motion source
    artifact -- the animation is generated procedurally from the rig itself."""
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to bake the Slither animation")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({
        "op": "slither-bake",
        "input": str(input_glb),
        "output": str(output_glb),
        "animation": animation,
        "settings": settings,
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
    )
    if not output_glb.exists() or output_glb.stat().st_size < 64:
        raise RuntimeError("Blender completed without producing a valid animated GLB")
    return {
        "backend": "blender-slither",
        "animation": animation,
        "settings": dict(settings),
    }


def run_trellis_gguf(input_image: Path, output_glb: Path, resolution: int, progress: Progress | None = None) -> dict[str, Any]:
    """Run the full Trellis2 geometry+texture C++ pipeline.

    Uses the pr-1 trellis2cpp DLL which implements the complete pipeline:
    sparse-structure -> shape SLAT -> mesh + PBR texture flow -> UV-atlas GLB.
    Falls back gracefully to geometry-only clay mesh when texture GGUFs are absent.
    """
    from backend.app.trellis2_wrapper import (
        Trellis2Wrapper, T2_PIPE_AUTO, T2_PIPE_512, T2_PIPE_1024,
        T2_PIPE_COARSE, T2_BACKGROUND_AUTO, T2_CAP_TEXTURE,
    )
    import random
    import io
    from PIL import Image

    gguf_dir = trellis_gguf_model_dir() / "ggufs"

    # ── Required geometry GGUFs ───────────────────────────────────────────
    dino      = gguf_dir / "dino_f16.gguf"
    ss_flow   = gguf_dir / "ss_flow_f16.gguf"
    ss_dec    = gguf_dir / "ss_dec_f16.gguf"
    slat_flow = gguf_dir / "slat_flow_f16.gguf"
    shape_dec = gguf_dir / "shape_dec_f16.gguf"

    for p in (dino, ss_flow, ss_dec):
        if not p.is_file():
            raise RuntimeError(
                f"Required geometry model is missing: {p.name}. "
                "Download models from the Generate tab first."
            )

    # ── Optional texture GGUFs ────────────────────────────────────────────
    shape_enc     = gguf_dir / "shape_enc_f16.gguf"
    tex_dec       = gguf_dir / "tex_dec_f16.gguf"
    tex_flow_512  = gguf_dir / "tex_slat_flow_512_f16.gguf"
    tex_flow_1024 = gguf_dir / "tex_slat_flow_1024_f16.gguf"

    def _opt(p: Path) -> str | None:
        return str(p) if p.is_file() and p.stat().st_size > 0 else None

    has_slat  = slat_flow.is_file() and shape_dec.is_file()
    has_tex   = all(p.is_file() and p.stat().st_size > 0
                    for p in (shape_enc, tex_dec, tex_flow_512))

    # ── Pick pipeline type ────────────────────────────────────────────────
    if resolution <= 512:
        pipe_type = T2_PIPE_COARSE
        mode_label = "coarse (marching-cubes)"
    elif resolution >= 1024 and has_slat:
        pipe_type = T2_PIPE_1024
        mode_label = "fine 1024 cascade"
    elif has_slat:
        pipe_type = T2_PIPE_512
        mode_label = "fine 512 dual-grid"
    else:
        pipe_type = T2_PIPE_COARSE
        mode_label = "coarse (marching-cubes, slat models missing)"

    if has_tex:
        mode_label += " + PBR texture"

    # ── Convert input to PNG bytes ─────────────────────────────────────────
    with Image.open(input_image) as img:
        img_rgba = img.convert("RGBA")
        png_io = io.BytesIO()
        img_rgba.save(png_io, format="PNG")
        image_bytes = png_io.getvalue()

    if progress:
        progress(f"Loading Trellis2 pipeline ({mode_label})...")

    wrapper = Trellis2Wrapper(ROOT / "vendor" / "trellis2cpp" / "build")
    pipeline = wrapper.load_pipeline(
        dino=str(dino),
        ss_flow=str(ss_flow),
        ss_dec=str(ss_dec),
        slat_flow=_opt(slat_flow),
        slat_hr_flow=None,               # reserved for future HR cascade
        shape_dec=_opt(shape_dec),
        shape_enc=_opt(shape_enc),
        tex_dec=_opt(tex_dec),
        tex_flow=_opt(tex_flow_512),
        tex_flow_hr=_opt(tex_flow_1024),
    )
    # ggml picks ggml-cuda.dll when present and falls back to ggml-cpu.dll
    # silently otherwise (see trellis2_wrapper.py's Trellis2Wrapper.__init__);
    # t2_pipeline_backend is the ground truth for which one actually loaded.
    native_backend = wrapper.backend_name(pipeline)
    is_gpu_backend = "cuda" in native_backend.lower()
    device_message = f"device: {'cuda' if is_gpu_backend else 'cpu'} (trellis2cpp backend={native_backend})"
    print(f"[trellis-gguf] {device_message}", flush=True)
    if progress:
        progress(device_message)

    try:
        _STAGE_LABELS = {
            0:  "Pre-processing image",
            1:  "Encoding with DINOv3",
            2:  "Sparse-structure flow",
            3:  "Decoding occupancy grid",
            4:  "Shape SLAT flow",
            5:  "Decoding shape geometry",
            6:  "Extracting mesh",
            7:  "Upsampling to 1024 voxels",
            8:  "1024 shape SLAT flow",
            9:  "1024 shape decode",
            10: "PBR texture flow",
        }

        def on_progress(stage: int, step: int, total: int) -> None:
            label = _STAGE_LABELS.get(stage, f"Stage {stage}")
            if total > 0:
                label = f"{label} (step {step}/{total})"
            if progress:
                progress(label)

        if progress:
            progress("Running Trellis2 image-to-3D generation...")

        seed = random.randint(0, 2 ** 32 - 1)
        mesh_data = wrapper.generate(
            pipeline,
            image_bytes,
            seed=seed,
            pipeline_type=pipe_type,
            background_mode=T2_BACKGROUND_AUTO,
            steps=0,           # use C++ defaults (12)
            guidance=-1.0,     # use C++ defaults (7.5)
            texture_steps=0,   # use C++ defaults (12)
            progress_cb=on_progress,
        )
    finally:
        wrapper.free_pipeline(pipeline)

    if progress:
        progress("Baking GLB artifact...")

    if mesh_data["has_pbr"]:
        # Full PBR path: let the C++ xatlas bake produce the final GLB
        # This avoids trimesh stripping UV data and gives the best quality.
        if progress:
            progress("Baking UV-atlas textured GLB (xatlas)...")
        glb_bytes = wrapper.bake_glb(
            verts=mesh_data["vertices"],
            tris=mesh_data["faces"],
            pbr=mesh_data["pbr"],
            texture_size=0,       # auto: 2048 @1024, 1024 @512
            component_filter=0,   # remove tiny floating islands
        )
        output_glb.write_bytes(glb_bytes)
        textured = True
    else:
        # Geometry-only path: use trimesh with a clay PBR material
        import trimesh
        from trimesh.visual.material import PBRMaterial

        vertices = [
            [mesh_data["vertices"][i],
             mesh_data["vertices"][i + 2],
             -mesh_data["vertices"][i + 1]]
            for i in range(0, len(mesh_data["vertices"]), 3)
        ]
        faces = [
            [mesh_data["faces"][i],
             mesh_data["faces"][i + 1],
             mesh_data["faces"][i + 2]]
            for i in range(0, len(mesh_data["faces"]), 3)
        ]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        trimesh.repair.fill_holes(mesh)
        vnormals = mesh.vertex_normals
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                               vertex_normals=vnormals)
        mesh.visual.material = PBRMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.7,
            baseColorFactor=[220, 220, 220, 255],
            doubleSided=True,
        )
        mesh.export(output_glb, file_type="glb")
        textured = False

    return {
        "backend": "trellis2cpp",
        "model_set": "rms80/trellis2cpp pr-1",
        "resolution": resolution,
        "mode": mode_label,
        "textured": textured,
        "device": "cuda" if is_gpu_backend else "cpu",
        "native_backend": native_backend,
    }


_MESHFIX_TINY_COMPONENT_FACE_FRACTION = 0.005
_MESHFIX_COMPONENT_BOUNDS_RANGE = (0.75, 1.25)
_MESHFIX_COMPONENT_FACE_RANGE = (0.5, 2.5)
_MESHFIX_WHOLE_MESH_BOUNDS_RANGE = (0.5, 1.25)
_MESHFIX_AGGREGATE_FACE_GROWTH_LIMIT = 1.25


def _boundary_edge_count(mesh: Any) -> int:
    import numpy as np

    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum())


def repair_meshfix(input_obj: Path, output_obj: Path, progress: Progress | None = None) -> dict[str, Any]:
    """Repair an Instant Meshes output without modifying the original OBJ.

    MeshFix operates on triangles, so this is intentionally an opt-in repair
    stage for broken/non-manifold regions rather than part of normal retopo.

    A single whole-mesh MeshFix pass is too coarse for dense, heavily
    non-manifold Instant Meshes output: pymeshfix can silently discard almost
    everything and still report success, which a whole-mesh bounds guard then
    (correctly) rejects wholesale. Repair runs per connected component
    instead, so a handful of broken components can't sink the whole job:
    each component's repaired result is accepted only if its own bounds and
    face count stayed within a bounded envelope of that component's
    original geometry; a rejected or too-tiny-to-bother-with component is
    passed through unrepaired instead of bridging or dropping it.
    """
    try:
        import numpy as np
        import pymeshfix
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "MeshFix is not installed. It is an optional, GPL-3.0 dependency of the "
            "Instant Meshes repair step: install it with pip install -e \".[meshfix]\"."
        ) from exc
    if progress:
        progress("MeshFix: repairing holes and self-intersections")
    source = trimesh.load(input_obj, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or source.faces.size == 0:
        raise RuntimeError("MeshFix could not read a triangle mesh from Instant Meshes output")
    raw_input_vertices = len(source.vertices)
    # OBJ exports commonly carry a separate vertex row per face-corner (e.g.
    # once per distinct UV/normal), so raw index-based adjacency would see
    # almost every face as its own disconnected "component". Weld coincident
    # vertices first so splitting reflects genuine mesh topology; merge_tex
    # and merge_norm are required or vertices at UV/normal seams refuse to
    # weld and the mesh re-fragments on split (matches mesh_quality.py's
    # topology-welding convention). This only consolidates duplicate index
    # rows and does not move any geometry, and downstream retopo baking
    # regenerates UVs from scratch so discarding stale UV/normal seams here
    # is safe.
    source.merge_vertices(merge_tex=True, merge_norm=True)
    before_vertices, before_faces = len(source.vertices), len(source.faces)
    before_bounds = source.bounds.copy()
    source_diagonal = float(np.linalg.norm(before_bounds[1] - before_bounds[0]))
    source_has_holes = not source.is_watertight

    components = source.split(only_watertight=False)
    if len(components) == 0:
        components = [source]
    total_components = len(components)

    # A mesh that has fragmented into a quad/triangle soup (thousands of
    # near-degenerate "components", or no single dominant body) is not a
    # per-component hole-repair problem: every piece is "tiny" relative to
    # the whole, none of them is individually repairable, and running
    # MeshFix per-component only wastes time before the eventual
    # zero-repaired raise below. Detect that up front and skip straight to
    # returning the welded (but otherwise untouched) mesh with a warning
    # instead of raising — the job still completes, just without holes
    # filled, so the operator can retry with AI Refine or a lower target
    # vertex count instead of losing the whole run.
    largest_component_faces = max((len(component.faces) for component in components), default=0)
    largest_component_fraction = (largest_component_faces / before_faces) if before_faces else 0.0
    fragmentation_component_threshold = max(2000, 0.02 * before_faces)
    is_fragmented_beyond_repair = (
        total_components > fragmentation_component_threshold or largest_component_fraction < 0.5
    )
    if is_fragmented_beyond_repair:
        if progress:
            progress(
                f"MeshFix: skipped — {total_components} components "
                f"(largest holds {largest_component_fraction * 100:.1f}% of faces) is too "
                "fragmented to repair"
            )
        repaired = source
        repaired_diagonal = float(np.linalg.norm(repaired.bounds[1] - repaired.bounds[0]))
        boundary_edges = _boundary_edge_count(repaired)
        warning = (
            f"Hole repair skipped: the remesh is fragmented into {total_components} pieces "
            f"(largest holds {largest_component_fraction * 100:.1f}% of faces) — use AI Refine, "
            f"or lower the target vertex count below the source's ({before_vertices} vertices)"
        )
        output_obj.parent.mkdir(parents=True, exist_ok=True)
        repaired.export(output_obj, file_type="obj")
        return {
            "welded_input_vertices": before_vertices,
            "raw_input_vertices": raw_input_vertices,
            "input_faces": before_faces,
            "output_vertices": len(repaired.vertices),
            "output_faces": len(repaired.faces),
            "components": total_components,
            "repaired_components": 0,
            "kept_components": total_components,
            "warning": warning,
            "skipped_reason": "fragmented",
        }

    tiny_face_threshold = before_faces * _MESHFIX_TINY_COMPONENT_FACE_FRACTION
    bounds_low, bounds_high = _MESHFIX_COMPONENT_BOUNDS_RANGE
    faces_low, faces_high = _MESHFIX_COMPONENT_FACE_RANGE

    parts: list[Any] = []
    # Bookkeeping for the aggregate face-growth guard below: each accepted
    # repair's index into `parts` plus the original (unrepaired) component,
    # so an inflator can be swapped back to its original if the assembled
    # mesh grows too much even though every individual component passed its
    # own bound.
    accepted_info: list[dict[str, Any]] = []
    repaired_count = 0
    kept_count = 0
    for component in components:
        component_faces = len(component.faces)
        # Tiny scraps aren't worth running MeshFix on: keep them as-is and
        # count them among the kept components.
        if component_faces < tiny_face_threshold:
            parts.append(component)
            kept_count += 1
            continue

        component_diagonal = float(np.linalg.norm(component.bounds[1] - component.bounds[0]))
        try:
            fixer = pymeshfix.MeshFix(np.asarray(component.vertices), np.asarray(component.faces), verbose=False)
            # Preserve disconnected pieces such as accessories and do not bridge them.
            fixer.repair(joincomp=False, remove_smallest_components=False)
            repaired_component = trimesh.Trimesh(
                vertices=np.asarray(fixer.points), faces=np.asarray(fixer.faces), process=False
            )
        except Exception:
            repaired_component = None

        accepted = False
        if (
            repaired_component is not None
            and repaired_component.faces.size > 0
            and np.isfinite(repaired_component.vertices).all()
        ):
            repaired_component_diagonal = float(
                np.linalg.norm(repaired_component.bounds[1] - repaired_component.bounds[0])
            )
            repaired_component_faces = len(repaired_component.faces)
            bounds_ok = component_diagonal == 0 or (
                bounds_low * component_diagonal <= repaired_component_diagonal <= bounds_high * component_diagonal
            )
            faces_ok = faces_low * component_faces <= repaired_component_faces <= faces_high * component_faces
            accepted = bounds_ok and faces_ok

        if accepted:
            parts.append(repaired_component)
            accepted_info.append({"index": len(parts) - 1, "original": component})
            repaired_count += 1
        else:
            parts.append(component)
            kept_count += 1

    # Aggregate face-growth guard: a per-component envelope alone still lets
    # several moderate, individually-legal hole fills add up to a mesh that's
    # much heavier than the input. Downgrade the biggest inflators back to
    # their original (unrepaired) geometry, largest growth first, until the
    # assembled mesh is back under the cap. Every "kept" component is a
    # subset of the original mesh's faces, so downgrading all accepted
    # components is guaranteed to land at or under before_faces.
    total_faces = sum(len(part.faces) for part in parts)
    if accepted_info and total_faces > before_faces * _MESHFIX_AGGREGATE_FACE_GROWTH_LIMIT:
        accepted_info.sort(key=lambda info: len(parts[info["index"]].faces) - len(info["original"].faces), reverse=True)
        for info in accepted_info:
            if total_faces <= before_faces * _MESHFIX_AGGREGATE_FACE_GROWTH_LIMIT:
                break
            index = info["index"]
            original_component = info["original"]
            total_faces += len(original_component.faces) - len(parts[index].faces)
            parts[index] = original_component
            repaired_count -= 1
            kept_count += 1

    if repaired_count == 0:
        if source_has_holes:
            raise RuntimeError(
                f"MeshFix repair was rejected for every one of the mesh's {total_components} "
                "components and the source mesh still has unfilled holes — use AI Refine, or "
                f"lower the target vertex count below the source's ({before_vertices} vertices)"
            )
        repaired = source
    else:
        repaired = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        if not isinstance(repaired, trimesh.Trimesh):
            raise RuntimeError("MeshFix produced an invalid mesh")

    repaired_diagonal = float(np.linalg.norm(repaired.bounds[1] - repaired.bounds[0]))
    boundary_edges = _boundary_edge_count(repaired)
    warning: str | None = None
    if boundary_edges > 0 or kept_count > repaired_count:
        warning = (
            f"MeshFix repaired {repaired_count}/{total_components} pieces; "
            f"{boundary_edges} boundary edges remain"
        )
    if progress:
        progress(
            f"MeshFix: repaired {repaired_count}/{total_components} components, "
            f"kept {kept_count} unchanged (bounds {source_diagonal:.4g} → {repaired_diagonal:.4g})"
        )

    # The per-component envelope above is intentionally tighter than this;
    # keep the original whole-mesh guard as a final sanity check in case an
    # outlier combination of accepted components still slipped through.
    whole_low, whole_high = _MESHFIX_WHOLE_MESH_BOUNDS_RANGE
    if source_diagonal > 0 and not (whole_low * source_diagonal <= repaired_diagonal <= whole_high * source_diagonal):
        raise RuntimeError("MeshFix repair was rejected because it changed the model bounds by more than the safe limit")

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_obj, file_type="obj")
    result: dict[str, Any] = {
        "welded_input_vertices": before_vertices,
        "raw_input_vertices": raw_input_vertices,
        "input_faces": before_faces,
        "output_vertices": len(repaired.vertices),
        "output_faces": len(repaired.faces),
        "components": total_components,
        "repaired_components": repaired_count,
        "kept_components": kept_count,
    }
    if warning:
        result["warning"] = warning
    return result


def _read_glb_sidecar_metadata(glb_path: Path) -> dict[str, Any] | None:
    """Best-effort read of the artifact JSON sidecar next to a GLB.

    Studio artifacts are stored as `<id>.glb` plus a `<id>.json` sidecar
    carrying a `metadata` dict (stage, mesh_quality, prior retopo settings,
    etc.). This is advisory-only input for warnings, so any failure to find
    or parse the sidecar is treated as "no metadata available" rather than
    an error.
    """
    sidecar = glb_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = document.get("metadata") if isinstance(document, dict) else None
    return metadata if isinstance(metadata, dict) else None


def run_instant_meshes(
    input_glb: Path,
    output_glb: Path,
    target_vertices: int,
    repair_holes: bool = False,
    progress: Progress | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retopologize a GLB via Instant Meshes' native batch mode.

    The official executable exposes this exact non-interactive workflow:
    extrinsic field (the default), boundary alignment, target vertex count,
    orientation solve, position solve, and extraction. The GUI's default
    extraction uses a quad-dominant mesh and zero smoothing passes.

    `source_metadata` lets a caller inject the source artifact's metadata
    directly (mainly for tests); when omitted it is auto-read from the
    artifact JSON sidecar next to `input_glb`, if one exists. It is used
    only to emit up-front warnings for two known-bad combinations: asking
    Instant Meshes to substantially upscale an already-retopologized mesh,
    and re-retopologizing an artifact that is already a retopo output —
    both tend to fragment the mesh rather than clean it up.
    """
    exe = instant_meshes_exe()
    if exe is None:
        raise RuntimeError("Instant Meshes is not installed. Open Retopologize and download the Instant Meshes engine first.")
    if find_blender() is None:
        raise RuntimeError("Blender 4.2+ is required for the GLB-to-OBJ conversion and texture bake")

    effective_source_metadata = (
        source_metadata if source_metadata is not None else _read_glb_sidecar_metadata(input_glb)
    )
    warnings: list[str] = []
    if effective_source_metadata:
        stage = str(effective_source_metadata.get("stage") or "").lower()
        if stage in ("retopo", "retopologize"):
            message = (
                f"Source artifact is already a retopo output (stage={stage!r}); "
                "retopologizing it again with Instant Meshes risks fragmenting it further"
            )
            warnings.append(message)
            if progress:
                progress(f"Warning: {message}")

        source_vertices: float | int | None = None
        mesh_quality = effective_source_metadata.get("mesh_quality")
        if isinstance(mesh_quality, dict) and isinstance(mesh_quality.get("vertices"), (int, float)):
            source_vertices = mesh_quality["vertices"]
        elif isinstance(effective_source_metadata.get("target_vertices"), (int, float)):
            source_vertices = effective_source_metadata["target_vertices"]

        if source_vertices and target_vertices > source_vertices * 1.5:
            message = (
                f"target_vertices ({target_vertices}) is more than 1.5x the source's vertex "
                f"count ({int(source_vertices)}) — upscaling a mesh with Instant Meshes fragments it"
            )
            warnings.append(message)
            if progress:
                progress(f"Warning: {message}")

    temp_dir = output_glb.parent / f"instant-meshes-{output_glb.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_obj = temp_dir / "source.obj"
    result_obj = temp_dir / "remeshed.obj"
    repaired_obj = temp_dir / "remeshed-repaired.obj"
    repair_stats: dict[str, Any] | None = None
    try:
        if progress:
            progress("Converting the full source GLB to OBJ")
        blender_export_obj(input_glb, source_obj, progress)
        if progress:
            progress("Instant Meshes: solving orientation and position fields")
        run_process([
            str(exe), "--output", str(result_obj), "--vertices", str(target_vertices),
            "--boundaries", "--dominant", "--smooth", "0", str(source_obj),
        ], progress=progress, timeout=3600)
        if not result_obj.exists() or result_obj.stat().st_size < 64:
            raise RuntimeError("Instant Meshes completed without writing remeshed.obj")
        mesh_for_bake = result_obj
        if repair_holes:
            repair_stats = repair_meshfix(result_obj, repaired_obj, progress)
            mesh_for_bake = repaired_obj
        if progress:
            progress("UV unwrapping and baking source textures onto the quad mesh")
        blender_bake_retopology(input_glb, mesh_for_bake, output_glb, progress)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    result: dict[str, Any] = {
        "backend": "instant-meshes",
        "target_vertices": target_vertices,
        "extrinsic": True,
        "align_to_boundaries": True,
        "texture_transfer": "blender-selected-to-active-bake",
        "meshfix_repair": repair_stats,
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# S2 canonicalization (docs/PIPELINE_POLICY.md) -- the mandatory, no-bypass
# raw-mesh -> game-ready-mesh stage. Reuses the same Instant Meshes / MeshFix
# / blender_bake_retopology chain as `run_instant_meshes` above, adds a
# bounded retry loop against the 1-component/watertight/0-non-manifold
# contract, floor-contact normalization, and a render-based appearance-
# similarity gate between the S1 source and the S2 output. See
# canonicalize_policy.py for the blocking gate that consumes this output.
# ---------------------------------------------------------------------------

DEFAULT_CANONICALIZE_VIEWS: tuple[str, ...] = ("front", "back", "left", "right")
# Similarity renders used to be 256px, then thumbnailed to 64x64 grayscale for
# the MAD/histogram score -- a pipeline that cannot see a resolution loss at
# all (2026-09-01: a visibly muddy canonical face scored 0.997). The score
# still reads 64x64 thumbnails, so it stays comparable with history, but the
# renders are now big enough for the face-priority detail measurement to have
# something to measure. 768 keeps four whole-body + four face pairs at ~15s.
DEFAULT_CANONICALIZE_RENDER_SIZE = 768
# Fraction of the subject's height, measured from the top, that the
# face-priority renders frame. A T-posed character's arms sit at ~55% height,
# so 30% is the head and nothing else.
DEFAULT_CANONICALIZE_FACE_FRACTION = 0.30
# ~30k triangles on a closed quad-dominant mesh (Instant Meshes' --vertices
# targets vertex count; a closed quad mesh triangulates to roughly 2x that
# many triangles), landing mid-band of the biped default (10-50k tris).
DEFAULT_CANONICALIZE_TARGET_VERTICES = 15_000
DEFAULT_CANONICALIZE_TRIANGLE_BAND: tuple[int, int] = (10_000, 50_000)
# Initial attempt plus at most two knob-adjustment retries, per policy.
CANONICALIZE_MAX_ATTEMPTS = 3

# Remesh backend selection (2026-08-30): Instant Meshes shattered one
# character's S1 mesh (231,576 faces, 1,617 non-manifold edges) into 800+ shards at every
# vertex target tried -- the extrinsic quad-field solver cannot recover from
# that much non-manifold, multi-component input. Blender's voxel remesh
# rebuilds the surface from a signed-distance field instead of following the
# existing (broken) topology, so it is watertight and single-component *by
# construction*, independent of how bad the input is. It is now the default;
# Instant Meshes stays reachable via settings.method="instant-meshes" for
# comparison. See docs/PIPELINE_POLICY.md S2.
DEFAULT_CANONICALIZE_METHOD = "voxel"
CANONICALIZE_METHODS: tuple[str, ...] = ("voxel", "instant-meshes")
# The Standard tier's decimation target for the voxel path, and the raw band
# that was hand-tuned around it. Both are now DERIVED from `quality_tiers` /
# `voxel_remesh` rather than written down twice: an operator who picks a
# denser tier gets the same 5x-11.7x raw oversampling scaled to their target
# (bead i63ky), and the Standard tier still resolves to exactly the 30k /
# 150k-350k numbers these names have always meant.
DEFAULT_CANONICALIZE_TARGET_FACES = QUALITY_TIER_TARGETS["standard"]
DEFAULT_VOXEL_RAW_FACE_MIN, DEFAULT_VOXEL_RAW_FACE_MAX = voxel_raw_face_band(
    DEFAULT_CANONICALIZE_TARGET_FACES
)
# Per-retry escalation of the raw band when a decimated candidate still
# fails the contract: search for an even-more-over-tessellated raw mesh
# (smaller voxels), which gives the decimator more source detail to work
# with rather than looser gates.
VOXEL_RAW_BAND_GROWTH = 1.5


def resolve_palette_reference(
    artifact_id: str, *, artifacts_dir: Path | None = None
) -> tuple[Path | None, str | None, str]:
    """Find the 2D image an artifact's lineage was ultimately conditioned on.

    Walks `metadata.source_artifact_id` back from `artifact_id` until it
    reaches an artifact of kind `image` whose file is still on disk. Returns
    `(path, id, reason)`; `path` is None when no reference exists, and
    `reason` always explains the outcome so the artifact's metadata can say
    why a palette transfer did or did not happen.

    The FIRST image found wins -- the direct lineage parent, not the run's
    root `source_artifact_id`. That distinction is the whole correctness of
    this function: on run 62107a15 the root is a gold-vignetted screenshot of
    an armoured knight (colorfulness 21.2), while the image the generator was
    actually conditioned on is the prepared black-and-white punk character
    (4.7). Keying the palette off the root would have tinted the character
    gold -- a different wrong answer than the pink one being fixed.
    """
    artifacts = artifacts_dir or durable_data_path("artifacts")
    current = str(artifact_id or "").strip()
    if not current:
        return None, None, "the artifact has no id to resolve a lineage from"
    seen: set[str] = set()
    for _ in range(PALETTE_REFERENCE_LINEAGE_DEPTH):
        if not current or current in seen:
            break
        seen.add(current)
        sidecar = artifacts / f"{current}.json"
        if not sidecar.is_file():
            return None, None, f"artifact {current} has no metadata sidecar on disk"
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, None, f"artifact {current}'s metadata sidecar is unreadable"
        if not isinstance(payload, dict):
            return None, None, f"artifact {current}'s metadata sidecar is not an object"
        if str(payload.get("kind") or "") == "image":
            path = Path(str(payload.get("path") or ""))
            if path.is_file():
                return path, current, f"conditioning image {current} from the artifact lineage"
            return None, None, f"lineage image {current} is no longer on disk"
        current = str((payload.get("metadata") or {}).get("source_artifact_id") or "")
    return None, None, "the artifact lineage contains no conditioning image"


def resolve_source_photo(
    artifact_id: str, *, artifacts_dir: Path | None = None
) -> tuple[Path | None, str | None, str]:
    """Find the ROOT upload an artifact's lineage was ultimately generated from.

    The fidelity gate must score a render against the operator's original
    photo, not the conditioning image a generator was pointed at -- a T-pose
    re-draw handed to TRELLIS can already have drifted from what the
    operator actually submitted, so scoring against it would validate the
    very drift this gate exists to catch. This walks the same
    `metadata.source_artifact_id` chain as `resolve_palette_reference`, but
    keeps going past each image artifact reached (as long as it still has a
    resolvable `source_artifact_id`) and remembers the LAST one -- the root
    upload -- rather than stopping at the first.
    """
    artifacts = artifacts_dir or durable_data_path("artifacts")
    current = str(artifact_id or "").strip()
    if not current:
        return None, None, "the artifact has no id to resolve a lineage from"
    seen: set[str] = set()
    root_path: Path | None = None
    root_id: str | None = None
    reason = "the artifact lineage contains no conditioning image"
    for _ in range(PALETTE_REFERENCE_LINEAGE_DEPTH):
        if not current:
            break
        if current in seen:
            reason = f"artifact lineage revisits {current} (cycle)"
            break
        seen.add(current)
        sidecar = artifacts / f"{current}.json"
        if not sidecar.is_file():
            reason = f"artifact {current} has no metadata sidecar on disk"
            break
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reason = f"artifact {current}'s metadata sidecar is unreadable"
            break
        if not isinstance(payload, dict):
            reason = f"artifact {current}'s metadata sidecar is not an object"
            break
        if str(payload.get("kind") or "") == "image":
            path = Path(str(payload.get("path") or ""))
            if path.is_file():
                root_path, root_id = path, current
        current = str((payload.get("metadata") or {}).get("source_artifact_id") or "")
    else:
        reason = "the artifact lineage lookup exceeded its depth limit"

    if root_path is not None:
        return root_path, root_id, f"source photo {root_id} from the root of the artifact lineage"
    return None, None, reason


def run_canonicalize(
    input_glb: Path,
    output_glb: Path,
    target_vertices: int,
    views: tuple[str, ...] = DEFAULT_CANONICALIZE_VIEWS,
    render_size: int = DEFAULT_CANONICALIZE_RENDER_SIZE,
    progress: Progress | None = None,
    method: str = DEFAULT_CANONICALIZE_METHOD,
    face_fraction: float = DEFAULT_CANONICALIZE_FACE_FRACTION,
    settings: dict[str, Any] | None = None,
    view_images: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """S2 canonicalization: remesh the raw S1 mesh to a game-ready triangle
    budget, weld/repair per component until the result is one watertight
    component with zero non-manifold edges (iterating the remesh knobs at
    most twice before giving up with a clear per-attempt finding -- never
    silently loosening the contract), bake appearance from the ORIGINAL mesh
    (never regenerate it -- the 08-30 AI-Refine face-destruction incident),
    normalize floor contact to z=0, and render a matched-viewpoint
    appearance-similarity gate between the S1 source and the S2 output.

    `view_images`, when given (role -> PNG path, at least 2 entries), also
    runs S2 view-projection (`run_project_views`) after the bake and floor
    normalize: the T-pose conditioning views TRELLIS was given are blended
    onto the canonical texture, before the similarity gate below renders --
    so that gate judges the texture the artifact actually ships with. `None`
    (the default) skips it entirely (existing callers are unaffected);
    `settings["project_views"] = False` skips it even when views are given.
    Fewer than 2 views records `metadata["view_projection"] =
    {"skipped": "insufficient_views", ...}` instead of running it.

    `method` selects the remesh backend: `"voxel"` (default) uses Blender's
    voxel remesh + decimate (`blender_worker.voxel_remesh_canonicalize`),
    which is watertight and single-component by construction; `"instant-
    meshes"` keeps the original Instant-Meshes-based path available for
    comparison (it shatters badly-conditioned input, such as one character's
    S1 mesh, into hundreds of shards).

    `settings` is the job's free-form settings dict. It carries the palette
    knobs: `skip_palette_transfer` (bool) turns the transfer off,
    `palette_transfer_strength` (0-1) blends it, and
    `palette_reference_artifact_id` overrides the lineage walk. It also carries
    the operator's density intent (bead i63ky): `quality_tier`
    (draft/standard/high/ultra) or a custom `target_triangles` /
    `target_vertices`, resolved by `quality_tiers.resolve_quality_target` into
    the decimation target, the raw voxel search band, the gate's triangle band
    and the bake atlas floor -- all four recorded in the returned report so the
    gate and the artifact agree on which density was asked for.

    Always produces an output artifact (even when the contract isn't met)
    so the operator can inspect the candidate; `canonicalize_policy` is
    what actually blocks promotion.
    """
    if method not in CANONICALIZE_METHODS:
        raise ValueError(f"Unknown canonicalize method: {method!r} (expected one of {CANONICALIZE_METHODS})")
    exe = None
    if method == "instant-meshes":
        exe = instant_meshes_exe()
        if exe is None:
            raise RuntimeError("Instant Meshes is not installed. Open Retopologize and download the Instant Meshes engine first.")
    if find_blender() is None:
        raise RuntimeError("Blender 4.2+ is required for canonicalization")

    from .mesh_quality import analyze_mesh
    import trimesh

    job_settings = settings or {}
    quality: QualityTarget = resolve_quality_target(job_settings)
    if progress:
        progress(
            f"Mesh density: {quality.tier} -> {quality.target_triangles:,} triangles "
            f"(gate band {quality.band[0]:,}-{quality.band[1]:,})"
        )
    palette_strength = float(
        job_settings.get("palette_transfer_strength", DEFAULT_PALETTE_TRANSFER_STRENGTH)
    )
    if not 0.0 <= palette_strength <= 1.0:
        raise ValueError(
            f"canonicalize settings.palette_transfer_strength must be in [0, 1], got {palette_strength}"
        )
    # `skip_palette_transfer` drives the STRENGTH to zero rather than dropping
    # the reference. The reference is still resolved and still handed to the
    # bake, because the palette DISTANCE is measured there and a run with the
    # transfer off is exactly the run whose distance the operator wants to
    # see -- switching the fix off must not also switch off the measurement
    # that says whether it was needed.
    if job_settings.get("skip_palette_transfer"):
        palette_strength = 0.0
    override = str(job_settings.get("palette_reference_artifact_id") or "").strip()
    palette_reference, palette_reference_id, palette_reason = resolve_palette_reference(
        override or input_glb.stem
    )
    if progress:
        progress(
            f"Palette reference: {palette_reason}"
            if palette_reference is None
            else f"Palette reference: {palette_reason} (strength {palette_strength:.2f})"
        )

    temp_dir = output_glb.parent / f"canonicalize-{output_glb.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_obj: Path | None = None
        if method == "instant-meshes":
            source_obj = temp_dir / "source.obj"
            if progress:
                progress("Converting the full source GLB to OBJ")
            blender_export_obj(input_glb, source_obj, progress)

        current_target = max(500, int(target_vertices))
        # The operator's density intent (bead i63ky). Everything downstream --
        # the decimation target, the raw voxel search band, the gate's band and
        # the bake atlas floor -- is derived from this one resolution, so a
        # tier can never leave the gate judging the mesh against a band that
        # belongs to a different density.
        current_target_faces = quality.target_triangles
        raw_growth = 1.0
        raw_face_min, raw_face_max = voxel_raw_face_band(current_target_faces)
        attempts: list[dict[str, Any]] = []
        mesh_for_bake: Path | None = None
        contract_met = False
        for attempt_index in range(CANONICALIZE_MAX_ATTEMPTS):
            result_obj = temp_dir / f"remeshed-{attempt_index}.obj"
            if method == "instant-meshes":
                repaired_obj = temp_dir / f"remeshed-{attempt_index}-repaired.obj"
                if progress:
                    progress(
                        f"Canonicalize attempt {attempt_index + 1}/{CANONICALIZE_MAX_ATTEMPTS}: "
                        f"Instant Meshes at {current_target} target vertices"
                    )
                run_process([
                    str(exe), "--output", str(result_obj), "--vertices", str(current_target),
                    "--boundaries", "--dominant", "--smooth", "0", str(source_obj),
                ], progress=progress, timeout=3600)
                if not result_obj.exists() or result_obj.stat().st_size < 64:
                    raise RuntimeError("Instant Meshes completed without writing remeshed.obj")
                repair_stats = repair_meshfix(result_obj, repaired_obj, progress)
                candidate_path = repaired_obj
                method_params: dict[str, Any] = {"target_vertices": current_target, "meshfix_repair": repair_stats}
            else:
                if progress:
                    progress(
                        f"Canonicalize attempt {attempt_index + 1}/{CANONICALIZE_MAX_ATTEMPTS}: "
                        f"Blender voxel remesh targeting {current_target_faces} tris "
                        f"(raw band {raw_face_min}-{raw_face_max})"
                    )
                voxel_report = blender_voxel_remesh_canonicalize(
                    input_glb, result_obj, current_target_faces, raw_face_min, raw_face_max, progress,
                )
                if not result_obj.exists() or result_obj.stat().st_size < 64:
                    raise RuntimeError("Blender voxel remesh completed without writing remeshed.obj")
                candidate_path = result_obj
                method_params = {
                    "target_faces": current_target_faces,
                    "raw_face_band": [raw_face_min, raw_face_max],
                    "voxel_remesh": voxel_report,
                }
            candidate_mesh = trimesh.load(candidate_path, force="mesh", process=False)
            if not isinstance(candidate_mesh, trimesh.Trimesh) or candidate_mesh.faces.size == 0:
                raise RuntimeError("Canonicalize remesh produced no triangle mesh")
            candidate_report = analyze_mesh(candidate_mesh)
            # analyze_mesh's "components" is measured on welded topology; also
            # split the raw (unwelded) mesh so seam-splitting duplicate
            # geometry that a welded measure alone would miss shows up too
            # (docs/PIPELINE_POLICY.md's S2 contract requires both measures).
            components_unwelded = len(candidate_mesh.split(only_watertight=False)) or 1
            attempt_record = {
                "attempt": attempt_index + 1,
                "method": method,
                **method_params,
                "components": candidate_report["components"],
                "components_unwelded": components_unwelded,
                "watertight": candidate_report["watertight"],
                "non_manifold_edges": candidate_report["non_manifold_edges"],
                "faces": candidate_report["faces"],
            }
            attempts.append(attempt_record)
            mesh_for_bake = candidate_path
            contract_met = (
                candidate_report["components"] == 1
                and components_unwelded == 1
                and candidate_report["watertight"]
                and candidate_report["non_manifold_edges"] == 0
            )
            if contract_met or attempt_index == CANONICALIZE_MAX_ATTEMPTS - 1:
                break
            if method == "instant-meshes":
                # Iterate the vertex-target knob: a lower target has fewer,
                # larger quads for MeshFix to work with and avoids the known
                # upscaling-fragments-the-mesh failure mode.
                current_target = max(500, int(current_target * 0.7))
            else:
                # Iterate the raw over-tessellation band upward: search for a
                # denser (smaller-voxel) raw mesh so the decimator has more
                # source detail to preserve manifoldness with, never loosen
                # the contract itself. Recomputed from the target (rather than
                # multiplied in place) so the absolute raw ceiling is honoured
                # on every retry, not just the first band.
                raw_growth *= VOXEL_RAW_BAND_GROWTH
                raw_face_min, raw_face_max = voxel_raw_face_band(
                    current_target_faces, growth=raw_growth
                )

        if mesh_for_bake is None:
            raise RuntimeError("Canonicalize produced no remesh candidate")

        if progress:
            progress("UV unwrapping and baking source textures onto the canonical mesh")
        texture_bake_report = blender_bake_retopology(
            input_glb, mesh_for_bake, output_glb, progress,
            palette_reference=palette_reference,
            palette_strength=palette_strength,
            atlas_minimum=quality.atlas_minimum,
        )

        floor_normalized_glb = temp_dir / "floor-normalized.glb"
        floor_report = _blender_normalize_floor(output_glb, floor_normalized_glb, progress)
        shutil.copy2(floor_normalized_glb, output_glb)
        # These two were only ever assertions ("raise if wrong"), which is why
        # pipeline_policy reported s2.axis_convention and s2.uniform_scale as
        # not-evaluable. Record the booleans first, then still raise: a
        # recorded True on a shipped artifact is what makes the check
        # evaluable, and the raise keeps a broken artifact from shipping.
        identity_transforms = _glb_mesh_nodes_have_identity_transforms(output_glb)
        axis_profile_preserved = _glb_axis_profile_matches(input_glb, output_glb)
        geometry_contract = dict(floor_report.get("geometry_contract") or {})
        geometry_contract["mesh_nodes_identity_transforms"] = identity_transforms
        geometry_contract["axis_profile_preserved"] = axis_profile_preserved
        geometry_contract["axis_convention_ok"] = bool(
            identity_transforms and axis_profile_preserved
        )
        # The Blender-side measurement checks the object scales are uniform;
        # the GLB-side one checks nothing was left in a node transform. The
        # contract wants both.
        geometry_contract["uniform_scale_ok"] = bool(
            geometry_contract.get("uniform_scale_ok", False) and identity_transforms
        )
        if not identity_transforms or not axis_profile_preserved:
            raise RuntimeError(
                "Canonicalize floor normalization broke the source axis orientation "
                "(node transform or Y/Z swap detected)"
            )

        # S2 view-projection (bead s2-view-projection): bake the T-pose
        # conditioning views onto the just-baked/floor-normalized canonical
        # texture, blended over TRELLIS's own bake, BEFORE the appearance-
        # similarity gate renders below -- so that gate judges the texture
        # the artifact will actually ship with. `view_images` is optional
        # (older callers, or a run with no recorded conditioning views, pass
        # none) and `settings["project_views"]` (default True) is the escape
        # hatch to skip it even when views are available.
        view_projection_report: dict[str, Any] | None = None
        if view_images is not None and job_settings.get("project_views", True):
            if len(view_images) >= 2:
                if progress:
                    progress(
                        f"Projecting {len(view_images)} T-pose conditioning view(s) onto the canonical texture"
                    )
                projected_glb = temp_dir / "projected.glb"
                view_projection_report = run_project_views(
                    output_glb, projected_glb, view_images, {}, progress,
                )
                shutil.copy2(projected_glb, output_glb)
            else:
                view_projection_report = {
                    "skipped": "insufficient_views",
                    "views_available": sorted(view_images),
                }

        similarity_dir = temp_dir / "similarity"
        if progress:
            progress(f"Rendering {len(views)} matched view(s) for the appearance-similarity gate")
        similarity_manifest = render_canonicalize_similarity_views(
            input_glb, output_glb, similarity_dir, views, render_size, progress,
            face_fraction=face_fraction,
        )
        similarity_report = merge_palette_measurement(
            evaluate_canonicalize_similarity(similarity_manifest), texture_bake_report,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    final_attempt = attempts[-1] if attempts else {}
    result: dict[str, Any] = {
        "backend": (
            "blender-voxel-remesh+blender-canonicalize-bake"
            if method == "voxel"
            else "instant-meshes+meshfix+blender-canonicalize-bake"
        ),
        # Provenance: which remesh backend actually produced this artifact,
        # and its knobs -- recorded per docs/PIPELINE_POLICY.md's
        # determinism principle (same inputs + settings -> same artifact).
        "remesh_method": method,
        # The operator's density intent, resolved (bead i63ky):
        # `quality_tier` ("draft"/"standard"/"high"/"ultra"/"custom"),
        # `quality_tier_requested`, `quality_target_source`,
        # `target_triangles`, `triangle_budget_band` and `bake_atlas_minimum`.
        # `triangle_budget_band` is the field `main` hands to the S2 gate and
        # `pipeline_policy._s2_canonicalization_metrics` reads back, so the
        # gate judges an Ultra mesh against Ultra's band rather than the fixed
        # legacy family band.
        **quality.as_provenance(),
        # What the mesh actually came out at, next to what was asked for --
        # the pair the verdict card shows the operator.
        "achieved_triangles": final_attempt.get("faces"),
        # The bake atlas the tier and the source's textures agreed on.
        "bake_texture_size": (texture_bake_report or {}).get("texture_size"),
        "requested_target_vertices": target_vertices,
        "final_target_vertices": final_attempt.get("target_vertices", target_vertices) if method == "instant-meshes" else None,
        "final_target_faces": final_attempt.get("target_faces") if method == "voxel" else None,
        "attempts": attempts,
        "contract_met": contract_met,
        "floor_offset_z": floor_report.get("floor_offset_z"),
        "similarity": similarity_report,
        # Duplicated at top level under the exact key name pipeline_policy.py's
        # S2 bake-gate check requires (`s1_s2_image_similarity_score`).
        "s1_s2_image_similarity_score": similarity_report.get("s1_s2_image_similarity_score"),
        # Same mirror for the detail-preservation measurement, which is what
        # actually sees a resolution loss the similarity score cannot.
        "s1_s2_detail_ratio": similarity_report.get("s1_s2_detail_ratio"),
        # ...and for palette fidelity, which neither of the other two can see:
        # both are computed on GRAYSCALE thumbnails, so a texture head that
        # invented a colour scheme scores perfectly on both.
        "s1_s2_palette_distance": similarity_report.get("s1_s2_palette_distance"),
        # Which image the palette was matched against, and whether it was.
        "palette_reference_artifact_id": palette_reference_id,
        "palette_reference_reason": palette_reason,
        # Why this artifact looks the way it does: atlas size, the source
        # texture size it was derived from, packed UV utilisation, bake-miss
        # fraction, and the tangent-space normal bake's conventions
        # (`texture_bake.normal_bake`) -- the channel that carries the
        # source's geometric detail onto the decimated canonical mesh.
        "texture_bake": texture_bake_report,
        "geometry_contract": geometry_contract,
        # Same top-level-mirror pattern, for the S2 checks that were reported
        # not-evaluable purely because nobody wrote the number down.
        "uv_unwrapped": geometry_contract.get("uv_unwrapped"),
        "axis_convention_ok": geometry_contract.get("axis_convention_ok"),
        "uniform_scale_ok": geometry_contract.get("uniform_scale_ok"),
    }
    if view_projection_report is not None:
        result["view_projection"] = view_projection_report
    if not contract_met:
        result["warnings"] = [
            f"Canonicalize could not reach 1 component / watertight / 0 non-manifold after "
            f"{len(attempts)} attempt(s); last attempt: {final_attempt}"
        ]
    return result


def blender_voxel_remesh_canonicalize(
    input_glb: Path,
    output_obj: Path,
    target_faces: int,
    raw_face_min: int,
    raw_face_max: int,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """S2 canonicalization's primary remesh backend: Blender voxel remesh +
    decimate (`blender_worker.voxel_remesh_canonicalize`). Watertight and
    single-component by construction, unlike Instant Meshes' quad-field
    solver, which shatters badly-conditioned input (see `run_canonicalize`'s
    docstring). Writes the remeshed geometry to `output_obj` (fed into the
    same `blender_bake_retopology` bake path Instant Meshes candidates use)
    and returns the worker's provenance report (voxel size, raw/decimated
    face counts, per-iteration search log).
    """
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for voxel canonicalize remeshing")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    report_path = output_obj.with_suffix(".voxel.json")
    spec = json.dumps({
        "op": "voxel-remesh-canonicalize",
        "input": str(input_glb),
        "output": str(output_obj),
        "report": str(report_path),
        "target_faces": int(target_faces),
        "raw_face_min": int(raw_face_min),
        "raw_face_max": int(raw_face_max),
    })
    log_device_line(progress, "blender-voxel-remesh")
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=1800,
        env=blender_worker_env(),
    )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def _blender_normalize_floor(input_glb: Path, output_glb: Path, progress: Progress | None = None) -> dict[str, Any]:
    """Translate a mesh so its lowest point sits at z=0 (floor contact)."""
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to normalize floor contact")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    report_path = output_glb.with_suffix(".floor.json")
    spec = json.dumps({
        "op": "normalize-floor",
        "input": str(input_glb),
        "output": str(output_glb),
        "report": str(report_path),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=600,
        env=blender_worker_env(),
    )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def render_canonicalize_similarity_views(
    source_glb: Path,
    canonical_glb: Path,
    output_dir: Path,
    views: tuple[str, ...] = DEFAULT_CANONICALIZE_VIEWS,
    size: int = DEFAULT_CANONICALIZE_RENDER_SIZE,
    progress: Progress | None = None,
    *,
    face_fraction: float = DEFAULT_CANONICALIZE_FACE_FRACTION,
) -> dict[str, Any]:
    """Render the S1 source and the S2 canonicalized mesh from N matched
    viewpoints (identical camera + light per view) for the appearance-
    preservation gate. Each view produces two pairs: the whole-body framing
    the MAD/histogram score reads, and a face-priority framing (the top
    `face_fraction` of the subject) the detail-preservation ratio reads.
    Returns the manifest of rendered PNG paths."""
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to render the canonicalize similarity gate")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    spec = json.dumps({
        "op": "canonicalize-similarity-render",
        "source": str(source_glb),
        "canonical": str(canonical_glb),
        "output_dir": str(output_dir),
        "output": str(manifest_path),
        "views": list(views),
        "size": int(size),
        "face_fraction": float(face_fraction),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=900,
        env=blender_worker_env(),
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def compute_view_similarity(image_a: Any, image_b: Any, *, histogram_bins: int = 32) -> dict[str, float]:
    """Similarity between two same-shape grayscale float arrays in [0, 1]:
    mean-absolute-pixel-difference plus a color-histogram distance, combined
    into one similarity score in [0, 1] (1.0 == identical). This is the S2
    bake gate's per-view measurement -- did baking the canonicalized mesh's
    appearance from the original preserve the subject's look. Pure numpy,
    no new dependency, and independently unit-testable on synthetic arrays.
    """
    import numpy as np

    array_a = np.asarray(image_a, dtype=np.float64)
    array_b = np.asarray(image_b, dtype=np.float64)
    if array_a.shape != array_b.shape:
        raise ValueError(
            f"compute_view_similarity requires matching image shapes, got {array_a.shape} vs {array_b.shape}"
        )
    mad = float(np.mean(np.abs(array_a - array_b)))
    histogram_a, _ = np.histogram(array_a, bins=histogram_bins, range=(0.0, 1.0))
    histogram_b, _ = np.histogram(array_b, bins=histogram_bins, range=(0.0, 1.0))
    total_a = float(histogram_a.sum())
    total_b = float(histogram_b.sum())
    normalized_a = (histogram_a / total_a) if total_a else histogram_a.astype(np.float64)
    normalized_b = (histogram_b / total_b) if total_b else histogram_b.astype(np.float64)
    # Total variation distance between the two normalized histograms, in [0, 1].
    histogram_distance = float(np.sum(np.abs(normalized_a - normalized_b)) / 2.0)
    distance = 0.5 * mad + 0.5 * histogram_distance
    similarity = max(0.0, min(1.0, 1.0 - distance))
    return {
        "mad": round(mad, 6),
        "histogram_distance": round(histogram_distance, 6),
        "distance": round(distance, 6),
        "similarity": round(similarity, 6),
    }


def render_source_view(
    input_glb: Path, output_png: Path, framing: dict[str, Any], *, samples: int = 32, progress: Progress | None = None
) -> dict[str, Any]:
    """Render `input_glb` from a `SourceCamera` (fit or explicit, per `framing`)
    for the source-fidelity gate. Returns the worker's report dict plus the
    rendered image path."""
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to render the source-fidelity view")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    report_path = output_png.with_suffix(".json")
    spec = json.dumps({
        "op": "render-source-view",
        "input": str(input_glb),
        "output": str(output_png),
        "report": str(report_path),
        "framing": framing,
        "samples": int(samples),
    })
    run_process(
        [str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec],
        progress=progress,
        timeout=600,
        env=blender_worker_env(),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["image"] = str(output_png)
    return report


def measure_source_fidelity(
    mesh_glb: Path,
    photo_path: Path,
    output_dir: Path,
    *,
    photo_mask_path: Path | None = None,
    fov_x_degrees: float | None = None,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    camera: dict[str, Any] | None = None,
    samples: int = 32,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Render `mesh_glb` in the photo's framing and score it against
    `photo_path`: the single entry point the fidelity gate and the CLI both
    call. Writes `render.png`, `side_by_side.png`, and `report.json` into
    `output_dir`."""
    from PIL import Image
    import numpy as np
    from .source_fidelity import (
        SourceCamera,
        align_render_to_photo,
        compute_source_fidelity,
        side_by_side,
        subject_bbox,
        subject_mask_from_image,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    photo = np.array(Image.open(photo_path).convert("RGBA"))
    if photo_mask_path is not None:
        mask = np.array(Image.open(photo_mask_path).convert("L")) >= 128
    else:
        mask, _ = subject_mask_from_image(photo)

    if camera is not None:
        framing = {"camera": camera}
    else:
        height, width = photo.shape[:2]
        framing = {
            "image_size": [width, height],
            "subject_bbox": list(subject_bbox(mask)),
            "fov_x_degrees": fov_x_degrees,
            "yaw_degrees": yaw_degrees,
            "pitch_degrees": pitch_degrees,
        }

    render_png = output_dir / "render.png"
    render_report = render_source_view(mesh_glb, render_png, framing, samples=samples, progress=progress)
    render = np.array(Image.open(render_png).convert("RGBA"))

    report = compute_source_fidelity(
        photo, render, photo_mask=mask, camera=SourceCamera.from_spec(render_report["camera"])
    )

    render_mask, _ = subject_mask_from_image(render)
    aligned_render, _, _ = align_render_to_photo(render, render_mask, mask)
    side_by_side_path = output_dir / "side_by_side.png"
    side_by_side(photo, aligned_render, report["face_bbox"]).save(side_by_side_path)

    report = {
        **report,
        "render": str(render_png),
        "side_by_side": str(side_by_side_path),
        "photo": str(photo_path),
        "render_report": render_report,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# A detail ratio is capped rather than left unbounded: a canonical render can
# legitimately carry MORE high-frequency energy than its source (voxel-remesh
# faceting, UV seams), and an unbounded outlier would drag an aggregate far
# away from the blur signal the gate exists to catch.
MAX_DETAIL_RATIO = 4.0
# Below this the measured region carries no detail worth a ratio.
MIN_SOURCE_DETAIL = 1e-12
# Where the "lit surface" starts, as a fraction of the render's own dynamic
# range above its background. Measuring the whole frame does not work: the
# similarity renders put the subject on a near-black ground, and this
# character wears a black bodysuit, so whole-frame high-frequency energy is
# dominated by the SILHOUETTE -- shared geometry that cancels out of the
# ratio and drowns the texture signal (measured 2026-09-01: whole-frame ratio
# 0.96 on a bake whose face is visibly muddy, 0.70 once masked).
DETAIL_SURFACE_LEVEL = 0.45
# Silhouette edges are the strongest gradient in the frame and belong to
# geometry, not texture, so the mask is eroded this many pixels before
# anything is measured.
DETAIL_MASK_EROSION = 3
# Below this many surviving pixels a view has no measurable lit surface (a
# profile view of a dark-clad subject) and is left out of the aggregate
# rather than contributing noise.
MIN_DETAIL_MASK_PIXELS = 1_024


def laplacian_variance(image: Any, mask: Any = None) -> float:
    """High-frequency energy of a 2D grayscale array in [0, 1].

    Variance of the discrete 4-neighbour Laplacian -- the standard,
    dependency-free sharpness measure. Blur removes high spatial frequencies,
    so a blurred copy of an image always has a strictly lower value; the
    absolute number is scene-dependent, which is why the gate compares a
    RATIO between two matched renders rather than an absolute threshold.

    `mask`, when given, restricts the measurement to those pixels (same shape
    as `image`); this is how the caller confines the measurement to lit
    surface instead of silhouette and background.
    """
    import numpy as np

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"laplacian_variance needs a 2D grayscale array, got shape {array.shape}")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0
    laplacian = (
        array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
        - 4.0 * array[1:-1, 1:-1]
    )
    if mask is None:
        return float(np.var(laplacian))
    interior = np.asarray(mask, dtype=bool)
    if interior.shape != array.shape:
        raise ValueError(
            f"laplacian_variance mask shape {interior.shape} does not match image {array.shape}"
        )
    selected = laplacian[interior[1:-1, 1:-1]]
    if selected.size < 2:
        return 0.0
    return float(np.var(selected))


def lit_surface_mask(
    reference: Any,
    other: Any,
    *,
    surface_level: float = DETAIL_SURFACE_LEVEL,
    erosion: int = DETAIL_MASK_EROSION,
) -> Any:
    """Pixels where BOTH renders show lit subject surface, eroded off the edge.

    The brightness cut is adaptive -- `surface_level` of the way from the
    render's own background (the median of its border) up to its bright end
    (the 99.5th percentile) -- so it survives a different subject, a different
    albedo or a change in the render lighting without a magic constant.
    """
    import numpy as np

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(other, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"lit_surface_mask requires matching shapes, got {first.shape} vs {second.shape}"
        )
    border = np.concatenate([first[0], first[-1], first[:, 0], first[:, -1]])
    background = float(np.median(border))
    peak = float(np.percentile(first, 99.5))
    threshold = background + surface_level * max(peak - background, 0.0)
    mask = (first > threshold) & (second > threshold)
    for _ in range(max(0, erosion)):
        mask = (
            mask
            & np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
        )
    return mask


def compute_view_detail(source_image: Any, canonical_image: Any) -> dict[str, float]:
    """Detail preservation between one matched S1/S2 face-render pair.

    The S2 appearance gate's MAD + histogram score is computed on 64x64
    grayscale thumbnails of a whole-body render, which is structurally blind
    to resolution loss: on 2026-09-01 the alien's visibly muddy canonical face
    still scored 0.997. This measures what that score cannot -- the ratio of
    high-frequency energy across the lit surface both renders share. 1.0 means
    the canonical bake is as sharp as the source; well below 1.0 means detail
    was lost between S1 and S2.

    `detail_pixels` is how much lit surface the ratio was measured over, which
    is what `evaluate_canonicalize_similarity` weights views by -- a profile
    view of a dark-clad subject exposes almost no surface and should not carry
    the same weight as a front-on face.
    """
    import numpy as np

    mask = lit_surface_mask(source_image, canonical_image)
    pixels = int(np.count_nonzero(mask))
    source_detail = laplacian_variance(source_image, mask)
    canonical_detail = laplacian_variance(canonical_image, mask)
    if source_detail <= MIN_SOURCE_DETAIL:
        ratio = 1.0 if canonical_detail <= MIN_SOURCE_DETAIL else MAX_DETAIL_RATIO
    else:
        ratio = min(canonical_detail / source_detail, MAX_DETAIL_RATIO)
    return {
        "source_detail": round(source_detail, 10),
        "canonical_detail": round(canonical_detail, 10),
        "detail_ratio": round(ratio, 6),
        "detail_pixels": pixels,
    }


def _load_grayscale_array(png_path: Path, downscale: int | None = 64) -> Any:
    """Load a render as a grayscale float array in [0, 1].

    `downscale=None` keeps the render's own resolution, which is what the
    detail measurement needs -- downscaling is exactly the operation that
    destroys the high-frequency signal it is trying to read.
    """
    from PIL import Image
    import numpy as np

    with Image.open(png_path) as image:
        grayscale = image.convert("L")
        if downscale is not None:
            grayscale = grayscale.resize((downscale, downscale))
        return np.asarray(grayscale, dtype=np.float64) / 255.0


def merge_palette_measurement(
    similarity_report: dict[str, Any], texture_bake_report: dict[str, Any]
) -> dict[str, Any]:
    """Fold the bake's palette measurement into the similarity report.

    The measurement is taken in the BAKE (on the albedo atlas, against the
    conditioning image) and only mirrored here, because the atlas is the only
    unlit surface in the chain. Measuring it on the similarity renders instead
    was tried and abandoned on 2026-09-01: those renders are dim, so a texture
    that reproduces its source perfectly still scored a 0.55 colorfulness
    ratio front and 0.23 back, and the metric read the RENDER's exposure
    rather than the texture's palette.

    Two keys land here. `s1_s2_palette_distance` is the shipped artifact's
    distance from its conditioning image -- the gate. `s1_palette_distance` is
    the same measurement taken BEFORE the transfer, i.e. how far the generator
    had drifted on its own; it is what keeps the gate from becoming a
    tautology once the transfer is on, and it is the number that says a mono
    source was handed an invented palette.
    """
    palette = (texture_bake_report or {}).get("palette_transfer") or {}
    after = palette.get("palette_distance_after")
    before = palette.get("palette_distance_before")
    if not after:
        return similarity_report
    similarity_report["s1_s2_palette_distance"] = after["distance"]
    similarity_report["palette_distance_detail"] = after
    similarity_report["palette_reference_artifact_id"] = palette.get("reference_artifact_id")
    similarity_report["palette_transfer_applied"] = palette.get("applied")
    if before:
        similarity_report["s1_palette_distance"] = before["distance"]
    return similarity_report


def evaluate_canonicalize_similarity(
    view_manifest: dict[str, Any], *, downscale: int = 64, histogram_bins: int = 32,
) -> dict[str, Any]:
    """Score every matched view pair in a `canonicalize-similarity-render`
    manifest and roll them up into the S2 bake gate's aggregate score,
    recorded under `s1_s2_image_similarity_score` (the exact metric name
    pipeline_policy.py's S2 bake-gate check requires).

    When the manifest also carries face-priority renders (`source_face` /
    `canonical_face` -- the top ~30% of the subject, framed tight), each view
    additionally gets a detail-preservation ratio and the report carries the
    aggregate under `s1_s2_detail_ratio`. Manifests without them (an older
    worker) simply omit the key, and `canonicalize_policy` treats a missing
    detail ratio as advisory rather than blocking.

    Palette fidelity is NOT measured here -- see `merge_palette_measurement`,
    which folds in a measurement taken on the unlit albedo atlas during the
    bake. Every number this function computes is read off a GRAYSCALE
    thumbnail, which is precisely why an invented colour scheme was invisible
    to the S2 gate.
    """
    import numpy as np

    per_view: dict[str, Any] = {}
    measured: list[dict[str, Any]] = []
    for view, paths in (view_manifest.get("views") or {}).items():
        source_array = _load_grayscale_array(Path(paths["source"]), downscale)
        canonical_array = _load_grayscale_array(Path(paths["canonical"]), downscale)
        entry = compute_view_similarity(source_array, canonical_array, histogram_bins=histogram_bins)
        if paths.get("source_face") and paths.get("canonical_face"):
            detail = compute_view_detail(
                _load_grayscale_array(Path(paths["source_face"]), None),
                _load_grayscale_array(Path(paths["canonical_face"]), None),
            )
            entry.update(detail)
            if detail["detail_pixels"] >= MIN_DETAIL_MASK_PIXELS:
                measured.append(detail)
            else:
                entry["detail_excluded"] = "insufficient lit surface"
        per_view[view] = entry
    aggregate_similarity = float(np.mean([entry["similarity"] for entry in per_view.values()])) if per_view else 0.0
    aggregate_distance = float(np.mean([entry["distance"] for entry in per_view.values()])) if per_view else 1.0
    report: dict[str, Any] = {
        "schema_version": 2,
        "views": per_view,
        "s1_s2_image_similarity_score": round(aggregate_similarity, 6),
        "aggregate_distance": round(aggregate_distance, 6),
    }
    if measured:
        # Surface-weighted, not a plain mean of per-view ratios: a profile
        # view exposing a sliver of skin would otherwise count as much as a
        # front-on face, and its ratio is the noisiest of the set.
        source_energy = sum(item["source_detail"] * item["detail_pixels"] for item in measured)
        canonical_energy = sum(item["canonical_detail"] * item["detail_pixels"] for item in measured)
        aggregate_detail = (
            min(canonical_energy / source_energy, MAX_DETAIL_RATIO)
            if source_energy > MIN_SOURCE_DETAIL
            else 1.0
        )
        report["s1_s2_detail_ratio"] = round(float(aggregate_detail), 6)
        report["s1_s2_detail_ratio_min"] = round(
            float(min(item["detail_ratio"] for item in measured)), 6
        )
        report["detail_views_measured"] = len(measured)
        report["face_fraction"] = view_manifest.get("face_fraction")
        report["render_size"] = view_manifest.get("size")
    return report


def run_trellis_ai_refine(
    input_glb: Path,
    source_image: Path,
    output_glb: Path,
    target_vertices: int,
    repair_holes: bool = False,
    gpu_first: bool = False,
    progress: Progress | None = None,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Image-guided FP8 refinement, CuMesh reconstruction, and PBR texturing."""
    if not trellis_refiner_ready():
        raise RuntimeError("TRELLIS.2 FP8 is not downloaded. Download the generation model first.")
    interpreter = trellis_refiner_python()
    if interpreter is None:
        raise RuntimeError("TRELLIS.2 FP8 does not have a Python environment with PyTorch installed.")
    worker = ROOT / "backend" / "workers" / "trellis2_fp8_worker.py"
    wait_for_gpu_preflight(progress, cancelled=cancelled)
    log_device_line(progress, "trellis2-fp8-refine")
    policy = trellis_refine_policy()
    worker_env = trellis2_worker_env(interpreter)
    last_output = ""
    def run_worker(use_gpu_first: bool, safe_reconstruction: bool = True) -> None:
        nonlocal last_output
        spec = json.dumps({
            "operation": "refine", "runtime": str(trellis_refiner_dir()),
            "input_glb": str(input_glb), "source_image": str(source_image),
            "output_glb": str(output_glb), "resolution": policy["resolution"],
            "target_faces": target_vertices, "repair_holes": repair_holes, "gpu_first": use_gpu_first,
            "safe_reconstruction": safe_reconstruction,
        })
        last_output = run_process(
            [str(interpreter), str(worker), spec], progress=progress, timeout=7200, env=worker_env
        )

    used_gpu_first = gpu_first
    used_safe_reconstruction = False
    try:
        run_worker(gpu_first)
    except RuntimeError as exc:
        quad_crashed = "3221225477" in str(exc) or "QEF" in str(exc)
        if not gpu_first and not quad_crashed:
            raise
        if quad_crashed and not gpu_first:
            if progress:
                progress("Quad reconstruction could not complete; retrying compatible reconstruction")
            used_safe_reconstruction = True
            run_worker(False, safe_reconstruction=True)
            return {
                "backend": "trellis2-fp8-refine-reconstruct-texture",
                "source_image_guided": True,
                "model": "visualbruno/TRELLIS.2-4B-FP8",
                "target_vertices": target_vertices,
                "resolution": 1024,
                "gpu_first_requested": gpu_first,
                "gpu_first_used": False,
                "reconstruction": "compatible",
                "texture_transfer": "trellis2-image-conditioned-pbr",
                "device": _parse_worker_device_report(last_output).get("device", "unknown"),
            }
        # A full-resident FP8 pipeline can overflow or crash native CUDA
        # extensions on 8-GB cards. Retry the same job with staged offloading
        # so an experimental speed option never loses the user's work.
        if gpu_first and progress:
            progress("GPU-first could not complete; retrying in standard mode")
        used_gpu_first = False
        try:
            run_worker(False)
        except RuntimeError as retry_exc:
            if not (quad_crashed or "3221225477" in str(retry_exc) or "QEF" in str(retry_exc)):
                raise
            if progress:
                progress("Quad reconstruction could not complete; retrying compatible reconstruction")
            used_safe_reconstruction = True
            run_worker(False, safe_reconstruction=True)
    return {
        "backend": "trellis2-fp8-refine-reconstruct-texture",
        "source_image_guided": True,
        "model": "visualbruno/TRELLIS.2-4B-FP8",
        "target_vertices": target_vertices,
        "resolution": policy["resolution"],
        "gpu_first_requested": gpu_first,
        "gpu_first_used": used_gpu_first,
        "reconstruction": policy["reconstruction"],
        "conv_backend": policy["conv_backend"],
        "sparse_compute_dtype": policy["sparse_compute_dtype"],
        "use_tiled_decoder": policy["use_tiled_decoder"],
        "texture_transfer": "trellis2-image-conditioned-pbr",
        "device": _parse_worker_device_report(last_output).get("device", "unknown"),
    }


def run_trellis_fp8(
    input_image: Path,
    output_glb: Path,
    resolution: int,
    progress: Progress | None = None,
    view_images: dict[str, Path] | None = None,
    *,
    cancelled: Callable[[], bool] | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Generate image-conditioned textured geometry.

    ``view_images``, when provided, must include a ``"front"`` entry and may
    additionally carry ``"back"``/``"left"``/``"right"`` views. Multi-view
    conditioning is only exercised when more than one view is supplied; a
    single-entry (or absent) ``view_images`` keeps the legacy single-image
    spec unchanged.
    """
    if not trellis_refiner_ready():
        raise RuntimeError("TRELLIS.2 FP8 is not downloaded. Download the generation model first.")
    interpreter = trellis_refiner_python()
    if interpreter is None:
        raise RuntimeError("TRELLIS.2 FP8 does not have a Python environment with PyTorch installed.")
    worker = ROOT / "backend" / "workers" / "trellis2_fp8_worker.py"
    gpu_snapshot = wait_for_gpu_preflight(progress, cancelled=cancelled)
    log_device_line(progress, "trellis2-fp8-generate")
    low_vram = resolve_trellis_low_vram(gpu_snapshot)
    actual_resolution = 512 if resolution <= 512 else 1024
    spec_payload: dict[str, Any] = {
        "operation": "generate", "runtime": str(trellis_refiner_dir()),
        "source_image": str(input_image), "output_glb": str(output_glb),
        "resolution": actual_resolution,
        "generate_texture": True,
        "seed": int(seed),
        "gpu_first": not low_vram,
    }
    views: list[str] | None = None
    if view_images and len(view_images) > 1:
        if "front" not in view_images:
            raise RuntimeError("view_images requires a front image")
        spec_payload["view_images"] = {
            label: str(path) for label, path in view_images.items()
        }
        views = list(view_images.keys())
    spec = json.dumps(spec_payload)
    output = run_process(
        [str(interpreter), str(worker), spec],
        progress=progress,
        timeout=7200,
        env=trellis2_worker_env(interpreter),
    )
    if not _glb_has_texture(output_glb):
        raise RuntimeError(
            "TRELLIS.2 generation completed, but the GLB did not contain an embedded PBR texture"
        )
    device_report = _parse_worker_device_report(output)
    result = {
        "backend": "trellis2-fp8",
        "model": "visualbruno/TRELLIS.2-4B-FP8",
        "resolution": actual_resolution,
        "seed": int(seed),
        "stage": "textured-geometry",
        "textured": True,
        "device": device_report.get("device", "cuda" if gpu_snapshot.available else "unknown"),
        "low_vram": low_vram,
    }
    if views:
        result["views"] = views
    return result


def run_paint(input_glb: Path, output_glb: Path, settings: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for texture baking but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "paint", "input": str(input_glb), "output": str(output_glb), "settings": settings})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)
    return {"backend": "blender-texture-bake", "texture_size": int(settings.get("texture_size", 1024))}

def trellis_refine_policy() -> dict[str, Any]:
    return {
        "resolution": 512,
        "conv_backend": "spconv",
        "sparse_compute_dtype": "float16",
        "use_tiled_decoder": False,
        "downsampling": 32,
        "reconstruction": "preserve-decoded",
    }
"""UniRig auto-rig functions for native.py — appended to backend/app/native.py."""

import tempfile

_UNIRIG_GENERATE_KWARGS_KEYS = (
    "repetition_penalty", "temperature", "top_k", "top_p", "num_beams", "max_new_tokens",
)


def _unirig_task_config_with_overrides(
    unirig: Path,
    task_name: str,
    generate_kwargs_overrides: dict[str, Any],
) -> tuple[str, list[Path]]:
    """Write a per-call temp task+system config pair under UniRig's own
    configs/ dirs with `generate_kwargs` overridden, without touching the
    checked-in files. Returns (task_name_to_pass_to_run_py, temp_files).
    No-op (returns the original task name, no temp files) when there is
    nothing to override -- current behaviour is unchanged.
    """
    if not generate_kwargs_overrides:
        return task_name, []
    import uuid
    import yaml

    task_path = unirig / "configs" / "task" / f"{task_name}.yaml"
    task_doc = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    system_name = task_doc["components"]["system"]
    system_path = unirig / "configs" / "system" / f"{system_name}.yaml"
    system_doc = yaml.safe_load(system_path.read_text(encoding="utf-8"))
    system_doc.setdefault("generate_kwargs", {}).update(generate_kwargs_overrides)

    temp_name = f"_agentF_sampler_{uuid.uuid4().hex[:12]}"
    temp_system_path = unirig / "configs" / "system" / f"{temp_name}.yaml"
    temp_task_path = unirig / "configs" / "task" / f"{temp_name}.yaml"
    temp_system_path.write_text(yaml.safe_dump(system_doc), encoding="utf-8")
    task_doc["components"]["system"] = temp_name
    temp_task_path.write_text(yaml.safe_dump(task_doc), encoding="utf-8")
    return temp_name, [temp_system_path, temp_task_path]


def unirig_dir() -> Path | None:
    """Locate the UniRig runtime: UNIRIG_DIR, vendored clone, or ~/UniRig."""
    candidates = []
    env_dir = os.getenv("UNIRIG_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(ROOT / "vendor" / "unirig")
    candidates.append(Path.home() / "UniRig")
    for candidate in candidates:
        if (candidate / "run.py").is_file():
            return candidate.resolve()
    return None


def unirig_python() -> Path | None:
    unirig = unirig_dir()
    if unirig is None:
        return None
    for candidate in (
        unirig / ".venv" / "bin" / "python",
        unirig / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def unirig_ready() -> bool:
    """True when the UniRig skeleton stage can run end to end."""
    unirig = unirig_dir()
    python = unirig_python()
    if unirig is None or python is None:
        return False
    return (unirig / "configs" / "task" / "quick_inference_skeleton_articulationxl_ar_256.yaml").is_file() and any(
        (python.parent.parent / "lib").glob("python3*/site-packages/bpy")
    )


def _unirig_blender_worker(spec: dict, timeout: int = 1800, progress: Progress | None = None) -> None:
    # The UniRig venv bundles bpy 4.2; run the bind worker with it directly
    # instead of requiring an external Blender installation.
    python = unirig_python()
    if python is None:
        raise RuntimeError("UniRig runtime is not installed")
    worker = ROOT / "backend" / "workers" / "unirig_bind_worker.py"
    run_process(
        [str(python), str(worker), json.dumps(spec)],
        cwd=unirig_dir(),
        progress=progress,
        timeout=timeout,
    )


def run_unirig_strip_rig(input_glb: Path, output_glb: Path) -> None:
    """Remove embedded armatures so UniRig predicts against mesh geometry only."""
    _unirig_blender_worker({
        "op": "strip-rig",
        "input": str(input_glb),
        "output": str(output_glb),
    })


def run_unirig_auto_rig(
    input_glb: Path,
    output_glb: Path,
    progress: Progress | None = None,
    unirig_settings: dict[str, Any] | None = None,
    chain_joints: int | None = None,
    profile_hint: str | None = None,
    skip_optional_bones: bool = False,
    skip_distal_refinement: bool = False,
) -> dict[str, Any]:
    """Full auto-rig: strip -> UniRig skeleton prediction (GPU) -> bone-heat bind.

    `unirig_settings` is optional and defaults to current behaviour (a
    single seed-12345 prediction with the checked-in sampler config):
      - `repetition_penalty`/`temperature`/`top_k`/`top_p`/`num_beams`/
        `max_new_tokens`, if present, override the AR decoder's
        `generate_kwargs` via a per-call temp config
        (`_unirig_task_config_with_overrides`) -- the checked-in
        ~/UniRig/configs files are never modified.
      - `candidates` (default 1) predicts that many seeds (12345, 12346,
        ... unless `seeds` explicitly lists them) and scores each with
        `skeleton_naming.score_chain_candidate` (longest unbranched main
        chain, then fewest spurs); the best-scoring candidate is bound.
    `chain_joints`, when set, is forwarded to the bind worker to densify a
    chain-profile rig's sparse predicted joints (see
    `skeleton_naming.densify_chain`); omitted (None) leaves a chain rig's
    raw predicted joint count as-is, matching current behaviour.
    `profile_hint` ("humanoid"/"chain", from the run's subject_kind) is
    forwarded to the bind worker to override automatic profile detection --
    see `skeleton_naming.semantic_joint_names` for exactly how. Omitted
    (None) is current behaviour: pure automatic detection.
    `skip_optional_bones` is the escape hatch back to a 21-bone (required
    tier only) humanoid rig: by default the bind worker now GROWS the biped
    template's 45 optional bones (fingers, toes, head_tip) from the placed
    hands/feet/head, because UniRig predicts none of them and every clip in
    the family store animates all 66 by name (see
    `skeleton_naming.build_template_conformant_skeleton`).
    `skip_distal_refinement` is the escape hatch out of the post-fit
    hand/foot anchor snap: by default the bind worker moves `hand_l`/`hand_r`
    (and, when the profile is unambiguous, `foot_l`/`foot_r`) onto the mesh's
    own wrist/ankle before building anything on top of them, because UniRig
    fits the wrist about halfway up the forearm (see
    `skeleton_naming.refine_distal_anchor`). True restores the raw predicted
    placement exactly.
    """
    unirig = unirig_dir()
    python = unirig_python()
    if unirig is None or python is None:
        raise RuntimeError(
            "UniRig runtime is not installed. Run scripts/setup_unirig.sh, "
            "or set UNIRIG_DIR to a prepared UniRig checkout with its venv."
        )

    def report(message: str) -> None:
        if progress:
            progress(message)

    settings = unirig_settings or {}
    generate_kwargs_overrides = {
        key: settings[key] for key in _UNIRIG_GENERATE_KWARGS_KEYS if key in settings
    }
    candidate_count = max(int(settings.get("candidates", 1)), 1)
    explicit_seeds = settings.get("seeds")
    if explicit_seeds:
        seeds = [int(value) for value in explicit_seeds][:candidate_count] or [12345]
    elif candidate_count == 1:
        seeds = [12345]  # unchanged default seed
    else:
        seeds = [12345 + index for index in range(candidate_count)]

    temp_root = durable_data_path("tmp")
    temp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="unirig-", dir=temp_root))
    clean_glb = work / "clean.glb"
    npz_dir = work / "npz"
    task = "quick_inference_skeleton_articulationxl_ar_256"

    report("Removing any embedded template rig from the mesh")
    run_unirig_strip_rig(input_glb, clean_glb)

    report("Extracting mesh features for UniRig")
    run_process(
        [str(python), "-m", "src.data.extract",
         "--config=configs/data/quick_inference.yaml",
         "--require_suffix=glb", "--force_override=true", "--num_runs=1", "--id=0",
         f"--time={time.time():.0f}", "--faces_target_count=50000",
         f"--input={clean_glb}", f"--output_dir={npz_dir}"],
        cwd=unirig,
        progress=progress,
        timeout=900,
    )
    stem = clean_glb.stem
    raw_candidates = [npz_dir / stem / "raw_data.npz", work / stem / "raw_data.npz"]
    found = [c for c in raw_candidates if c.is_file()] or sorted(work.rglob("raw_data.npz"))
    if not found:
        raise RuntimeError(f"UniRig extraction did not produce raw_data.npz under {work}")
    raw_npz = found[0]

    import numpy as np
    from .skeleton_naming import score_chain_candidate

    candidate_table: list[dict[str, Any]] = []
    best_index = 0
    best_key: tuple[bool, int, int] | None = None
    best_skeleton_npz: Path | None = None
    task_name, temp_config_files = _unirig_task_config_with_overrides(
        unirig, task, generate_kwargs_overrides,
    )
    try:
        for index, seed in enumerate(seeds):
            report(f"Predicting skeleton with UniRig (GPU) -- seed {seed} ({index + 1}/{len(seeds)})")
            predict_dir = work / f"predict-{index}"
            run_process(
                [str(python), "run.py", f"--task=configs/task/{task_name}.yaml",
                 f"--seed={seed}", f"--input={clean_glb}", f"--output_dir={predict_dir}", f"--npz_dir={npz_dir}"],
                cwd=unirig,
                progress=progress,
                timeout=1800,
            )
            skeleton_candidates = [predict_dir / stem / "predict_skeleton.npz", work / stem / "predict_skeleton.npz"]
            skeleton_found = (
                [c for c in skeleton_candidates if c.is_file()] or sorted(predict_dir.rglob("predict_skeleton.npz"))
            )
            if not skeleton_found:
                raise RuntimeError(f"UniRig prediction did not produce predict_skeleton.npz under {predict_dir}")
            skeleton_npz = skeleton_found[0]

            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            joints_arr = skeleton_data["joints"]
            parents_list = [None if p is None or p < 0 else int(p) for p in skeleton_data["parents"]]
            score = score_chain_candidate(joints_arr, parents_list)
            candidate_table.append({
                "seed": seed,
                "joints": int(len(joints_arr)),
                "chain_length": score.main_chain_length,
                "spur_count": score.spur_count,
                "branching": score.disqualified,
            })
            report(
                f"seed={seed} joints={len(joints_arr)} chain_length={score.main_chain_length} "
                f"spurs={score.spur_count} branching={score.disqualified}"
            )
            candidate_key = (not score.disqualified, score.main_chain_length, -score.spur_count)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_index = index
                best_skeleton_npz = skeleton_npz
    finally:
        for temp_file in temp_config_files:
            temp_file.unlink(missing_ok=True)

    if best_skeleton_npz is None:
        raise RuntimeError("UniRig produced no usable skeleton candidate")
    skeleton_npz = best_skeleton_npz
    if len(seeds) > 1:
        report(f"Selected seed {seeds[best_index]} (candidate {best_index + 1}/{len(seeds)})")

    report("Binding bone-heat weights and exporting")
    validation_path = work / "bind-validation.json"
    bind_spec = {
        "op": "unirig-bind",
        "input": str(clean_glb),
        "skeleton_npz": str(skeleton_npz),
        "raw_npz": str(raw_npz),
        "output": str(output_glb),
        "validation": str(validation_path),
    }
    if chain_joints:
        bind_spec["chain_joints"] = int(chain_joints)
    if profile_hint:
        bind_spec["profile_hint"] = profile_hint
    if skip_optional_bones:
        bind_spec["skip_optional_bones"] = True
    if skip_distal_refinement:
        bind_spec["skip_distal_refinement"] = True
    # Escape hatch out of the hanging-region weighting post-pass
    # (`unirig_bind_worker.apply_hanging_region_blend`), carried on the
    # free-form `unirig` settings dict so it needs no new API surface.
    if settings.get("skip_hanging_blend"):
        bind_spec["skip_hanging_blend"] = True
    _unirig_blender_worker(bind_spec)
    report("UniRig auto-rig complete")

    validation: dict[str, Any] = {}
    if validation_path.is_file():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    # keep the bind inputs so the review viewport can rebind after hand
    # adjustments without re-predicting
    inputs_dir = durable_data_path("rig-inputs", output_glb.stem)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clean_glb, inputs_dir / "clean.glb")
    shutil.copy2(raw_npz, inputs_dir / "raw_data.npz")
    shutil.copy2(skeleton_npz, inputs_dir / "predict_skeleton.npz")
    names_path = raw_npz.parent / "names.json"
    if names_path.is_file():
        shutil.copy2(names_path, inputs_dir / "names.json")

    shutil.rmtree(work, ignore_errors=True)
    return {
        "backend": "unirig-auto-rig",
        "runtime_dir": str(unirig),
        "bind_inputs_dir": str(inputs_dir),
        "bones": validation.get("bones"),
        "weighted_bones": validation.get("weighted_bones"),
        "empty_bones": validation.get("empty_bones"),
        "weighting": "blender-bone-heat",
        "unirig_candidates": candidate_table,
        "unirig_chosen_seed": seeds[best_index],
        "skeleton_source": validation.get("skeleton_source", "unirig"),
        "profile_source": validation.get("profile_source", "detected"),
        "synthesized_optional_bones": validation.get("synthesized_optional_bones", []),
        "optional_chain_scales": validation.get("optional_chain_scales", []),
        "optional_containment_fraction": validation.get("optional_containment_fraction"),
        "optional_weighted_bones": validation.get("optional_weighted_bones", []),
        "optional_empty_bones": validation.get("optional_empty_bones", []),
        "distal_refinement": validation.get("distal_refinement", []),
        "skip_distal_refinement": validation.get("skip_distal_refinement", False),
        "hanging_region_vertices": validation.get("hanging_region_vertices", 0),
        "hanging_region_torso_shell_vertices": validation.get(
            "hanging_region_torso_shell_vertices", 0
        ),
        "hanging_region_bridge_vertices": validation.get("hanging_region_bridge_vertices", 0),
        "hanging_region_blend": validation.get("hanging_region_blend", {}),
        "control_bones_excluded_from_skinning": validation.get(
            "control_bones_excluded_from_skinning", []
        ),
        "skip_hanging_blend": validation.get("skip_hanging_blend", False),
    }


def run_unirig_rebind(
    input_glb: Path,
    skeleton_npz: Path,
    raw_npz: Path,
    output_glb: Path,
    apply_positions: dict[str, list[float]],
    source_rig: Path | None = None,
    remove_joints: list[str] | None = None,
    skip_optional_bones: bool = False,
    skip_distal_refinement: bool = False,
    skip_hanging_blend: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Re-run weight binding with hand-adjusted joint positions applied.

    `source_rig`, when given, is the prior rigged GLB whose own armature
    the bind worker reads directly (bone positions/names/parents) instead
    of re-deriving the skeleton from `skeleton_npz`/`raw_npz` -- this keeps
    a rebind grounded in the rig a human actually reviewed rather than a
    fresh (possibly differently-named) UniRig prediction. `remove_joints`
    is a list of joint names to splice out of that source rig (see
    `skeleton_naming.remove_joints_reparent`); joints that could not be
    removed (an unknown name, the root, or a protected structural bone)
    are reported back in the result's `refused_joints` rather than
    failing the rebind.

    Any joint listed in `apply_positions` is protected from the bind
    worker's distal-anchor refinement -- a position a human placed by hand
    is never argued with by the detector (see
    `skeleton_naming.refine_distal_anchors`); `skip_distal_refinement`
    turns the refinement off for every joint.
    """
    if unirig_python() is None:
        raise RuntimeError("UniRig runtime is not installed")
    validation_path = output_glb.parent / f"{output_glb.stem}.bind-validation.json"
    spec: dict[str, Any] = {
        "op": "unirig-bind",
        "input": str(input_glb),
        "skeleton_npz": str(skeleton_npz),
        "raw_npz": str(raw_npz),
        "output": str(output_glb),
        "validation": str(validation_path),
        "apply_positions": apply_positions,
    }
    if source_rig is not None:
        spec["source_rig"] = str(source_rig)
    if remove_joints:
        spec["remove_joints"] = list(remove_joints)
    if skip_optional_bones:
        spec["skip_optional_bones"] = True
    if skip_distal_refinement:
        spec["skip_distal_refinement"] = True
    if skip_hanging_blend:
        spec["skip_hanging_blend"] = True
    _unirig_blender_worker(spec)
    validation: dict[str, Any] = {}
    if validation_path.is_file():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    validation_path.unlink(missing_ok=True)
    return {
        "backend": "unirig-rebind",
        "applied_positions": len(apply_positions),
        "bones": validation.get("bones"),
        "weighted_bones": validation.get("weighted_bones"),
        "weighting": validation.get("weighting"),
        "vertices_assigned": validation.get("vertices_assigned"),
        "removed_joints": validation.get("removed_joints", []),
        "refused_joints": validation.get("refused_joints", []),
        "synthesized_optional_bones": validation.get("synthesized_optional_bones", []),
        "optional_chain_scales": validation.get("optional_chain_scales", []),
        "optional_containment_fraction": validation.get("optional_containment_fraction"),
        "optional_weighted_bones": validation.get("optional_weighted_bones", []),
        "optional_empty_bones": validation.get("optional_empty_bones", []),
        "distal_refinement": validation.get("distal_refinement", []),
        "skip_distal_refinement": validation.get("skip_distal_refinement", False),
    }


def run_export_soma_bvh(
    output_bvh: Path,
    animation: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Export a curated library action as a SOMA-named BVH motion clip."""
    donor = curated_sword_donor()
    if unirig_python() is None:
        raise RuntimeError("UniRig runtime is not installed")
    _unirig_blender_worker({
        "op": "export-soma-bvh",
        "input": str(donor.path),
        "animation": animation or "",
        "output": str(output_bvh),
    })
    if not output_bvh.exists() or output_bvh.stat().st_size < 64:
        raise RuntimeError("BVH export did not produce a valid motion file")
    return {
        "backend": "curated-library-bvh",
        "animation": animation or "longest-clip",
        "donor": str(donor.path.relative_to(ROOT)),
    }
