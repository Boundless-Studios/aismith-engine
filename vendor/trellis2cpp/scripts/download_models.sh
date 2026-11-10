#!/usr/bin/env bash
# Download the checkpoints needed for the stage-1 geometry pipeline ("512" path).
#
# NOTE on DINOv3: the official facebook/dinov3-vitl16-pretrain-lvd1689m repo is
# gated behind a license-acceptance click. Until access is granted on your HF
# account, this script pulls the widely-used ungated mirror
# camenduru/dinov3-vitl16-pretrain-lvd1689m (byte-identical HF-format export,
# ships Meta's LICENSE.md). Re-point DINO_REPO at the official repo once you
# have access.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

fetch() { # fetch <repo> <rfile> <dest-subdir>
    local url="https://huggingface.co/$1/resolve/main/$2"
    local dest="$MODELS/$3/$2"
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ]; then echo "have  $dest"; return 0; fi
    echo "fetch $1/$2"
    curl -sSL --fail -C - "${AUTH[@]}" -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

T2=microsoft/TRELLIS.2-4B
T1=microsoft/TRELLIS-image-large
DINO_REPO="${DINO_REPO:-camenduru/dinov3-vitl16-pretrain-lvd1689m}"

fetch $T2 pipeline.json                                          TRELLIS.2-4B
fetch $T2 ckpts/ss_flow_img_dit_1_3B_64_bf16.json                TRELLIS.2-4B
fetch $T2 ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors         TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json       TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json       TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors TRELLIS.2-4B
fetch $T2 ckpts/shape_dec_next_dc_f16c32_fp16.json               TRELLIS.2-4B
fetch $T2 ckpts/shape_dec_next_dc_f16c32_fp16.safetensors        TRELLIS.2-4B
# PBR texturing stack: the validated generation path needs the shape encoder,
# texture decoder, and both texture-SLAT flows.
fetch $T2 texturing_pipeline.json                                TRELLIS.2-4B
fetch $T2 ckpts/shape_enc_next_dc_f16c32_fp16.json               TRELLIS.2-4B
fetch $T2 ckpts/shape_enc_next_dc_f16c32_fp16.safetensors        TRELLIS.2-4B
fetch $T2 ckpts/tex_dec_next_dc_f16c32_fp16.json                 TRELLIS.2-4B
fetch $T2 ckpts/tex_dec_next_dc_f16c32_fp16.safetensors          TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json        TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors  TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json       TRELLIS.2-4B
fetch $T2 ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors TRELLIS.2-4B
fetch $T1 ckpts/ss_dec_conv3d_16l8_fp16.json                     TRELLIS-image-large
fetch $T1 ckpts/ss_dec_conv3d_16l8_fp16.safetensors              TRELLIS-image-large
fetch "$DINO_REPO" config.json               dinov3-vitl16
fetch "$DINO_REPO" preprocessor_config.json  dinov3-vitl16
fetch "$DINO_REPO" model.safetensors         dinov3-vitl16

echo "all downloads complete:"
du -sh "$MODELS"/*
