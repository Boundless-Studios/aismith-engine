---
license: mit
base_model: microsoft/TRELLIS.2-4B
tags:
  - gguf
  - trellis
  - image-to-3d
  - ggml
pipeline_tag: image-to-3d
---

# TRELLIS.2-4B — GGUF (f16)

GGUF (`f16`) conversions of [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
for the **[trellis2cpp](https://github.com/localai-org/trellis2cpp)** /
[ggml](https://github.com/ggml-org/ggml) runtime — a CUDA-free, PyTorch-free C++ port of
the TRELLIS.2 image-to-3D pipeline. Load these directly; no safetensors conversion or
Python is needed at inference.

## Files

| File | Pipeline stage | Source safetensors |
|---|---|---|
| `ss_flow_f16.gguf` | Sparse-structure flow (64³ occupancy) | `ss_flow_img_dit_1_3B_64_bf16` |
| `slat_flow_f16.gguf` | Shape-SLAT flow, 512 fine | `slat_flow_img2shape_dit_1_3B_512_bf16` |
| `slat_flow_1024_f16.gguf` | Shape-SLAT flow, 1024 cascade | `slat_flow_img2shape_dit_1_3B_1024_bf16` |
| `shape_dec_f16.gguf` | Shape decoder → dual-grid fields | `shape_dec_next_dc_f16c32_fp16` |
| `shape_enc_f16.gguf` | Shape encoder (re-encode for texture flow) | `shape_enc_next_dc_f16c32_fp16` |
| `tex_dec_f16.gguf` | PBR texture decoder | `tex_dec_next_dc_f16c32_fp16` |
| `tex_slat_flow_512_f16.gguf` | Texture-SLAT flow, 512 | `slat_flow_imgshape2tex_dit_1_3B_512_bf16` |
| `tex_slat_flow_1024_f16.gguf` | Texture-SLAT flow, 1024 | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` |

The image conditioning encoder (DINOv3) and the sparse-structure decoder are published
separately:
[dinov3-vitl16-pretrain-lvd1689m-GGUF](https://huggingface.co/LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF)
and
[TRELLIS-image-large-GGUF](https://huggingface.co/LocalAI-io/TRELLIS-image-large-GGUF).

## Conversion

Converted from the upstream `bf16`/`fp16` safetensors to GGUF `f16` (ggml ftype 1) with
the `convert_*_to_gguf.py` scripts in the trellis2cpp project. No weights were retrained
or modified — this is a format conversion only.

## License

MIT, inherited from [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B).
See [`LICENSE`](LICENSE). Copyright (c) Microsoft Corporation.
