"""What this engine install is: version, locks, runtimes, GPU and weights.

`identity()` is cheap and goes into every stage result. `inspect()` probes the
runtimes too; enrolment runs it with `full=True` (rehash every weight), and
the pool's per-job drift check runs it from the hash cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from . import sources, weights

ROOT = sources.ROOT
MESH_WEIGHTS = ("trellis2-4b-fp8", "trellis-image-large-ss-decoder", "dinov3-vitl16", "birefnet-general", "birefnet-general-lite")
STAGE_WEIGHTS: dict[str, tuple[str, ...]] = {"mesh": MESH_WEIGHTS, "clean": ()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 60) -> str | None:
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def identity() -> dict[str, Any]:
    """Engine version and commit, and the digests of both locks."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    commit = os.getenv("AISMITH_ENGINE_COMMIT", "").strip() or _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    return {
        "name": project["name"],
        "version": project["version"],
        "commit": commit,
        "sources_lock_sha256": _file_sha256(sources.LOCK_PATH),
        "weights_lock_sha256": _file_sha256(weights.LOCK_PATH),
    }


# WSL2 ships nvidia-smi outside PATH.
WSL_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    return str(WSL_NVIDIA_SMI) if WSL_NVIDIA_SMI.is_file() else None


def _gpus() -> list[dict[str, Any]]:
    smi = _nvidia_smi()
    if smi is None:
        return []
    output = _run([smi, "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (output or "").splitlines():
        name, compute_cap, memory, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "compute_capability": compute_cap, "memory_mb": int(float(memory)), "driver": driver})
    return gpus


def _python_torch(interpreter: Path | None) -> dict[str, Any] | None:
    if interpreter is None:
        return None
    probe = "import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda}))"
    output = _run([str(interpreter), "-c", probe], timeout=120)
    return {"python": str(interpreter), **(json.loads(output) if output else {"torch": None})}


def _comfyui_version(path: Path) -> str | None:
    """`__version__` from ComfyUI's comfyui_version.py; zip installs have no git metadata."""
    if not path.is_file():
        return None
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def runtimes() -> dict[str, Any]:
    from backend.app import native

    refiner = native.trellis_refiner_dir()
    contract_path = refiner / "native-contract.json"
    comfy_version = refiner / "ComfyUI" / "comfyui_version.py"
    blender = native.find_blender()
    unirig = native.unirig_dir()
    return {
        "trellis": {
            **(_python_torch(native.trellis_refiner_python()) or {"torch": None}),
            "native_contract": json.loads(contract_path.read_text()) if contract_path.is_file() else None,
            "comfyui_commit": _run(["git", "rev-parse", "HEAD"], cwd=refiner / "ComfyUI") if (refiner / "ComfyUI" / ".git").exists() else None,
            "comfyui_version": _comfyui_version(comfy_version),
        },
        "unirig": {
            **(_python_torch(native.unirig_python()) or {"torch": None}),
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=unirig) if unirig else None,
        },
        "blender": {
            "path": str(blender) if blender else None,
            "version": (_run([str(blender), "--version"], timeout=120) or "").splitlines()[0] if blender else None,
        },
    }


def check_stage_weights(stage: str, *, full: bool = False) -> weights.VerificationReport | None:
    """Verify the weights a stage needs; None when it needs none."""
    from backend.app import native

    ids = list(STAGE_WEIGHTS[stage])
    if not ids:
        return None
    roots = native.engine_weight_roots()
    return weights.verify(roots, ids, full=full, cache_path=weights.default_cache_path(roots))


def inspect(*, full: bool = False) -> dict[str, Any]:
    from backend.app import native

    roots = native.engine_weight_roots()
    report = weights.verify(roots, full=full, cache_path=weights.default_cache_path(roots))
    return {
        "engine": identity(),
        "gpus": _gpus(),
        "runtimes": runtimes(),
        "weights": report.as_dict(),
        "offline": native.offline_mode(),
    }
