#!/usr/bin/env bash
# Build and launch the demo server (CUDA lib + Go server) in the demo container.
# The container carries a matching glibc/libstdc++/Go so both the CUDA
# libtrellis2.so and the Go binary that dlopens it share one runtime — a NixOS
# host binary won't run inside the CUDA image (different dynamic loader).
#
#   scripts/demo.sh            # fine path (needs the shape-SLAT GGUFs)
#   scripts/demo.sh -coarse    # 64^3 marching-cubes preview only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8742}"

docker build -f docker/Dockerfile.demo -t trellis2-demo docker

docker run --rm -v "$ROOT":/work -w /work -e GOCACHE=/tmp/gocache trellis2-demo bash -c '
  cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=ON >/dev/null 2>&1 &&
  cmake --build build-cuda-shared -j"$(nproc)" &&
  cd server && CGO_ENABLED=0 go build -o trellis2-server-linux .'

# Fetch prebuilt f16 GGUFs from the public LocalAI-io repos. Files already present
# are skipped, so this is a no-op once ggufs/ is populated; it lets fresh demo
# users skip the separate download_models.sh + convert_all.sh steps.
scripts/download_ggufs.sh

docker rm -f trellis2-demo-run 2>/dev/null || true
exec docker run --rm --name trellis2-demo-run --device nvidia.com/gpu=all \
    -v "$ROOT":/work -w /work/server -p "$PORT":8742 trellis2-demo \
    ./trellis2-server-linux -lib /work/build-cuda-shared/libtrellis2.so \
    -ggufs /work/ggufs -store /work/generations -unload-idle -addr :8742 "$@"
