#!/usr/bin/env bash
# Download the prebuilt f16 GGUFs for the demo from the public LocalAI-io repos.
#
# This is the fast path for running the demo: it replaces the two-step
# download_models.sh (safetensors) + convert_all.sh (GGUF conversion) flow with a
# direct pull of the ready-made f16 GGUFs. Developers who need the f32 validation
# variants or want to regenerate GGUFs should still use those two scripts.
#
# Files land in $GGUFS (default: repo-root ggufs/). Already-present files are
# skipped, so re-runs are cheap and downloads resume (-C -). No auth needed
# (public repos); HF_TOKEN is used if set.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUFS="${GGUFS:-$ROOT/ggufs}"
ORG="${GGUF_ORG:-LocalAI-io}"
TOKEN="${HF_TOKEN:-}"
AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

mkdir -p "$GGUFS"

fetch() { # fetch <repo> <file>
    local url="https://huggingface.co/$ORG/$1/resolve/main/$2"
    local dest="$GGUFS/$2"
    if [ -s "$dest" ]; then echo "have  $dest"; return 0; fi
    echo "fetch $ORG/$1/$2"
    curl -sSL --fail -C - "${AUTH[@]}" -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

T2=TRELLIS.2-4B-GGUF
T1=TRELLIS-image-large-GGUF
DINO=dinov3-vitl16-pretrain-lvd1689m-GGUF

fetch "$DINO" dino_f16.gguf
fetch "$T1"   ss_dec_f16.gguf
fetch "$T2"   ss_flow_f16.gguf
fetch "$T2"   slat_flow_f16.gguf
fetch "$T2"   slat_flow_1024_f16.gguf
fetch "$T2"   shape_dec_f16.gguf
fetch "$T2"   shape_enc_f16.gguf
fetch "$T2"   tex_dec_f16.gguf
fetch "$T2"   tex_slat_flow_512_f16.gguf
fetch "$T2"   tex_slat_flow_1024_f16.gguf

echo "all GGUFs present in $GGUFS:"
du -sh "$GGUFS"
