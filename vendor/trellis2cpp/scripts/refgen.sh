#!/usr/bin/env bash
# Regenerate all PyTorch reference dumps inside the reference container.
# Usage: scripts/refgen.sh [fixture-image]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRELLIS2_PY_HOST="${TRELLIS2_PY_HOST:-/home/rich/python/TRELLIS.2}"
FIXTURE="${1:-/trellis2/assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp}"

run() {
    docker run --rm --device nvidia.com/gpu=all \
        -v "$ROOT":/work -v "$TRELLIS2_PY_HOST":/trellis2 \
        -e PYTHONPATH=/trellis2 -e TRELLIS2_PY=/trellis2 \
        -e ATTN_BACKEND=sdpa -e SPARSE_CONV_BACKEND=none \
        -e HF_HUB_OFFLINE=1 \
        -w /work trellis2-ref "$@"
}

mkdir -p "$ROOT/dumps"
run python scripts/dump_dino_reference.py --image "$FIXTURE"
run python tests/ref_ss_flow.py                 # CPU (true fp32 golden)
run python tests/ref_ss_sample.py --device cuda
run python tests/ref_ss_dec.py --device cuda
run python scripts/dump_slat_reference.py --device cuda  # TF32 disabled in ref_common
run python scripts/dump_cascade_reference.py --device cuda  # 1024 HR stage
echo "reference dumps regenerated:"
ls -la "$ROOT/dumps" "$ROOT"/tests/*.bin
