---
license: other
license_name: dinov3-license
license_link: LICENSE
base_model: facebook/dinov3-vitl16-pretrain-lvd1689m
tags:
  - gguf
  - dinov3
  - vision-encoder
  - ggml
---

# DINOv3 ViT-L/16 (lvd1689m) — GGUF (f16)

**Built with DINOv3.**

GGUF (`f16`) conversion of the DINOv3 ViT-L/16 (`lvd1689m`) vision encoder for the
**[trellis2cpp](https://github.com/localai-org/trellis2cpp)** /
[ggml](https://github.com/ggml-org/ggml) runtime, where it produces the image conditioning
features for the TRELLIS.2 image-to-3D pipeline.

The weights derive from Meta's
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
(obtained via the ungated
[camenduru](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) mirror) and
are redistributed under Meta's **DINOv3 License**, a full copy of which is included as
[`LICENSE`](LICENSE). Your use of these weights is subject to that Agreement.

The source `model.safetensors` is **byte-identical to the official Meta release** —
verified SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`
(1,212,559,808 bytes), matching `facebook/dinov3-vitl16-pretrain-lvd1689m` exactly.

## Files

| File | Pipeline stage | Source |
|---|---|---|
| `dino_f16.gguf` | Image conditioning encoder (DINOv3 ViT-L/16) | `model.safetensors` |

## Conversion

Converted from the upstream safetensors to GGUF `f16` (ggml ftype 1) with
`convert_dino_to_gguf.py` in the trellis2cpp project. No weights were retrained or
modified — this is a format conversion only.

## License & attribution

Meta **DINOv3 License** — see [`LICENSE`](LICENSE). This model is **Built with DINOv3**.
DINOv3 is © Meta Platforms, Inc. Redistribution of these weights (or derivatives) must
provide a copy of the Agreement and prominently display "Built with DINOv3"; downstream
use is bound by the Agreement's terms (including its acceptable-use and trade-control
provisions).

## Companion models

- [TRELLIS.2-4B-GGUF](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF)
- [TRELLIS-image-large-GGUF](https://huggingface.co/LocalAI-io/TRELLIS-image-large-GGUF)
