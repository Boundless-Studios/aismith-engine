---
license: mit
base_model: microsoft/TRELLIS-image-large
tags:
  - gguf
  - trellis
  - image-to-3d
  - ggml
pipeline_tag: image-to-3d
---

# TRELLIS-image-large — GGUF (f16)

GGUF (`f16`) conversion of the sparse-structure decoder from
[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) for
the **[trellis2cpp](https://github.com/localai-org/trellis2cpp)** /
[ggml](https://github.com/ggml-org/ggml) runtime.

The TRELLIS.2 geometry pipeline reuses this TRELLIS-1 sparse-structure decoder to turn the
sparse-structure flow output into a 64³ voxel scaffold.

## Files

| File | Pipeline stage | Source safetensors |
|---|---|---|
| `ss_dec_f16.gguf` | Sparse-structure decoder → 64³ voxel scaffold | `ss_dec_conv3d_16l8_fp16` |

The rest of the TRELLIS.2 pipeline is published at
[TRELLIS.2-4B-GGUF](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF) and
[dinov3-vitl16-pretrain-lvd1689m-GGUF](https://huggingface.co/LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF).

## Conversion

Converted from the upstream `fp16` safetensors to GGUF `f16` (ggml ftype 1) with
`convert_ss_dec_to_gguf.py` in the trellis2cpp project. Format conversion only.

## License

MIT, inherited from
[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large).
See [`LICENSE`](LICENSE). Copyright (c) Microsoft Corporation.
