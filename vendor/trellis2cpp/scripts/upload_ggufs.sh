#!/usr/bin/env bash
# One-time (idempotent) publish of the prebuilt f16 GGUFs to the LocalAI-io org on
# Hugging Face. Splits by upstream provenance so each repo carries a single clean
# license:
#   TRELLIS.2-4B-GGUF                     (MIT)            8 files
#   TRELLIS-image-large-GGUF              (MIT)            ss_dec
#   dinov3-vitl16-pretrain-lvd1689m-GGUF  (DINOv3 License) dino
#
# Cards + LICENSE files live under scripts/hf/<repo>/ and are uploaded alongside the
# weights. Requires an authenticated `hf` (huggingface_hub >= 1.x) with write access
# to the org. `hf` is often not on PATH; set HF=/path/to/venv/bin/hf to override.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUFS="${GGUFS:-$ROOT/ggufs}"
CARDS="$ROOT/scripts/hf"
ORG="${GGUF_ORG:-LocalAI-io}"

# Locate the hf CLI: explicit $HF, then PATH, then known project venvs.
HF="${HF:-}"
if [ -z "$HF" ]; then
    if command -v hf >/dev/null 2>&1; then HF="$(command -v hf)"
    else
        for c in "$HOME"/.venvs/*/bin/hf "$HOME"/c/*/.venv*/bin/hf; do
            [ -x "$c" ] && { HF="$c"; break; }
        done
    fi
fi
[ -n "$HF" ] && [ -x "$HF" ] || { echo "error: hf CLI not found; set HF=/path/to/hf" >&2; exit 1; }
echo "using hf: $HF ($("$HF" --version 2>/dev/null))"
"$HF" auth whoami >/dev/null || { echo "error: not logged in (hf auth login)" >&2; exit 1; }

T2=TRELLIS.2-4B-GGUF
T1=TRELLIS-image-large-GGUF
DINO=dinov3-vitl16-pretrain-lvd1689m-GGUF

T2_FILES=(ss_flow_f16.gguf slat_flow_f16.gguf slat_flow_1024_f16.gguf shape_dec_f16.gguf \
          shape_enc_f16.gguf tex_dec_f16.gguf tex_slat_flow_512_f16.gguf tex_slat_flow_1024_f16.gguf)
T1_FILES=(ss_dec_f16.gguf)
DINO_FILES=(dino_f16.gguf)

publish() { # publish <repo> <file>...
    local repo="$1"; shift
    local id="$ORG/$repo"
    echo "== $id =="
    "$HF" repo create "$id" --repo-type model --exist-ok
    local f
    for f in "$@"; do
        [ -s "$GGUFS/$f" ] || { echo "error: missing $GGUFS/$f" >&2; exit 1; }
        echo "-- upload $f"
        "$HF" upload "$id" "$GGUFS/$f" "$f" --commit-message "Add $f"
    done
    "$HF" upload "$id" "$CARDS/$repo/README.md" README.md --commit-message "Add model card"
    "$HF" upload "$id" "$CARDS/$repo/LICENSE"   LICENSE   --commit-message "Add license"
}

publish "$T2"   "${T2_FILES[@]}"
publish "$T1"   "${T1_FILES[@]}"
publish "$DINO" "${DINO_FILES[@]}"

echo "done. repos:"
echo "  https://huggingface.co/$ORG/$T2"
echo "  https://huggingface.co/$ORG/$T1"
echo "  https://huggingface.co/$ORG/$DINO"
