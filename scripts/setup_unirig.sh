#!/usr/bin/env bash
# Install the UniRig auto-rig runtime into vendor/unirig (Linux/WSL, NVIDIA GPU).
#
# - clones VAST-AI-Research/UniRig and applies the two AISmith-required patches
#   (sdpa attention instead of flash-attn for the skeleton stage; npz export in
#   --output_dir mode)
# - creates an isolated .venv (python 3.11) with CUDA torch, torch-cluster,
#   torch-scatter and bpy 4.2
# - installs the pinned skeleton checkpoint (vendor/weights.lock.json) into
#   experiments/, where UniRig loads it with the hub offline
#
# Sources and weights are pinned: UniRig by commit (vendor/sources.lock.json),
# the checkpoint by revision and sha256. In a container build with no GPU,
# set AISMITH_BUILD_WITHOUT_GPU=1 to skip the CUDA device check.
#
# Usage: bash scripts/setup_unirig.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${UNIRIG_DIR:-$ROOT/vendor/unirig}"

APP_PY="${AISMITH_PYTHON:-$ROOT/.venv/bin/python}"
read -r UNIRIG_REPO UNIRIG_COMMIT < <(python3 - "$ROOT/vendor/sources.lock.json" <<'PIN'
import json, sys
entry = next(s for s in json.load(open(sys.argv[1]))["sources"] if s["name"] == "unirig")
print(entry["repository"], entry["commit"])
PIN
)

if [ -f "$TARGET/run.py" ]; then
  echo "UniRig already present at $TARGET"
else
  git clone --filter=blob:none --no-checkout "$UNIRIG_REPO" "$TARGET"
  git -C "$TARGET" checkout --detach "$UNIRIG_COMMIT"
fi
HEAD_COMMIT="$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "$HEAD_COMMIT" != "$UNIRIG_COMMIT" ]; then
  echo "warning: UniRig at $TARGET is $HEAD_COMMIT; the lock pins $UNIRIG_COMMIT" >&2
fi

cd "$TARGET"

# Patch 1: skeleton stage must not require flash-attn kernels.
sed -i 's/_attn_implementation: flash_attention_2/_attn_implementation: sdpa/' \
  configs/model/unirig_ar_350m_1024_81920_float32.yaml

# Patch 2: export predict_skeleton.npz in --output_dir mode (upstream gates it
# behind single-file --output mode only).
sed -i "s/writer_config\['user_mode'\] = True/writer_config['user_mode'] = args.output is not None/" \
  run.py

if [ ! -x ".venv/bin/python" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv --python 3.11 .venv
fi

PY=".venv/bin/python"
uv pip install --python "$PY" torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$PY" torch_cluster==1.6.3 torch_scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
sed -e '/^flash_attn$/d; /^bpy==4.2$/d' requirements.txt > /tmp/unirig-reqs.txt
uv pip install --python "$PY" -r /tmp/unirig-reqs.txt
uv pip install --python "$PY" bpy==4.2

# Import stubs: the skin stage deps (flash_attn, spconv) are not needed here —
# skinning uses Blender bone-heat — but the modules are imported transitively.
SP="$($PY -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SP/spconv" "$SP/flash_attn/modules"
touch "$SP/spconv/__init__.py" "$SP/spconv/pytorch.py" \
  "$SP/flash_attn/__init__.py" "$SP/flash_attn/modules/__init__.py"
cat > "$SP/flash_attn/modules/mha.py" <<'MHA'
class MHA:
    def __init__(self, *a, **k):
        raise RuntimeError("flash_attn stub: UniRig skin stage unused (Blender bone-heat)")
MHA

AISMITH_BUILD_WITHOUT_GPU="${AISMITH_BUILD_WITHOUT_GPU:-0}" $PY - <<'CHECK'
import os
import torch
import torch_cluster, torch_scatter  # noqa: F401
import bpy  # noqa: F401
if os.environ["AISMITH_BUILD_WITHOUT_GPU"] == "1":
    print("UniRig runtime imports OK (GPU check skipped for a container build)")
else:
    assert torch.cuda.is_available(), "CUDA is required for the UniRig stage"
    print("UniRig runtime OK:", torch.cuda.get_device_name(0))
CHECK

# Pinned skeleton checkpoint, verified by sha256.
(cd "$ROOT" && "$APP_PY" -m backend.engine.weights install unirig-skeleton-ar256 --root unirig="$TARGET")
echo "UniRig installed at $TARGET"
