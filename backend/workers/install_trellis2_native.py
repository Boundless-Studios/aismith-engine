"""Build TRELLIS.2 CUDA extensions for the GPU in the isolated runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


# Runs under the isolated refiner Python, so it reads the lock file directly
# instead of importing backend.engine.sources.
SOURCES_LOCK = Path(__file__).resolve().parents[2] / "vendor" / "sources.lock.json"


def _pinned(name: str) -> tuple[str, str]:
    entries = json.loads(SOURCES_LOCK.read_text(encoding="utf-8"))["sources"]
    entry = next(entry for entry in entries if entry["name"] == name)
    return entry["repository"], entry["commit"]


CUMESH_REPOSITORY, CUMESH_COMMIT = _pinned("cumesh")
CUMESH_API_REPOSITORY, CUMESH_API_COMMIT = _pinned("cumesh-api")
TRELLIS_REPOSITORY, TRELLIS_COMMIT = _pinned("trellis2")


def _run(
    command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _clone(repository: str, commit: str, destination: Path) -> None:
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--recurse-submodules",
            repository,
            str(destination),
        ]
    )
    _run(["git", "checkout", "--detach", commit], cwd=destination)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=destination)


def _cuda_build_environment() -> tuple[dict[str, str], str]:
    import nvidia
    import torch

    major, minor = torch.cuda.get_device_capability()
    architecture = f"{major}.{minor}"
    cuda_home = Path(nvidia.__file__).resolve().parent / "cu13"
    cccl_include = cuda_home / "include" / "cccl"
    environment = os.environ.copy()
    environment.update(
        {
            "BUILD_TARGET": "cuda",
            "CUDA_HOME": str(cuda_home),
            "TORCH_CUDA_ARCH_LIST": architecture,
            "CC": shutil.which("gcc-12") or "gcc",
            "CXX": shutil.which("g++-12") or "g++",
            "PATH": f"{cuda_home / 'bin'}{os.pathsep}{environment.get('PATH', '')}",
            "CPATH": f"{cccl_include}{os.pathsep}{environment.get('CPATH', '')}",
        }
    )
    return environment, architecture


def _overlay_visualbruno_remeshing(api_source: Path) -> None:
    import cumesh

    package = Path(cumesh.__file__).resolve().parent
    shutil.copy2(api_source / "cumesh" / "remeshing.py", package / "remeshing.py")


def install(marker: Path) -> None:
    environment, architecture = _cuda_build_environment()
    with tempfile.TemporaryDirectory(prefix="aismith-trellis-native-") as directory:
        root = Path(directory)
        cumesh_source = root / "cumesh"
        api_source = root / "cumesh-api"
        trellis_source = root / "trellis2"
        _clone(CUMESH_REPOSITORY, CUMESH_COMMIT, cumesh_source)
        _clone(CUMESH_API_REPOSITORY, CUMESH_API_COMMIT, api_source)
        _clone(TRELLIS_REPOSITORY, TRELLIS_COMMIT, trellis_source)

        atlas = cumesh_source / "src" / "atlas.cu"
        contents = atlas.read_text()
        anchor = "#include <c10/cuda/CUDAStream.h>\n"
        if "#include <cuda/std/functional>" not in contents:
            if anchor not in contents:
                raise RuntimeError(
                    "Pinned CuMesh no longer matches the CUDA 13 compatibility patch"
                )
            atlas.write_text(
                contents.replace(anchor, anchor + "#include <cuda/std/functional>\n")
            )

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                str(cumesh_source),
            ],
            env=environment,
        )
        _overlay_visualbruno_remeshing(api_source)
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                str(trellis_source / "o-voxel"),
            ],
            env=environment,
        )

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "cuda_architecture": architecture,
                "cumesh_binary_commit": CUMESH_COMMIT,
                "cumesh_api_commit": CUMESH_API_COMMIT,
                "o_voxel_commit": TRELLIS_COMMIT,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    arguments = parser.parse_args()
    install(arguments.marker)


if __name__ == "__main__":
    main()
