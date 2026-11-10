# Third-party notices

This table lists everything the engine and the studio app include, vendor,
build, install, download or run, with each component's licence. NOTICE has the
summary, including the non-commercial dependencies section. This is an
attribution guide, not legal advice, and it does not replace the licence texts
kept with each component.

"In repo" means the files are in this repository's tree. A git submodule is only
a pointer to an upstream commit: its files are fetched from upstream.
`vendor/sources.lock.json` pins every source by commit, and
`vendor/weights.lock.json` pins every model file by revision and sha256.

## Code

| Component | How the engine uses it | In repo? | Licence | Notes |
| --- | --- | --- | --- | --- |
| [AISmith-3D](https://github.com/intisarGIT/AISmith-3D) by Intisar Hasnain | The original project this repository was forked from, at `8af574d` | Yes | MIT | Copyright notice kept in `LICENSE` |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) v0.33.0 | Cloned at a pinned commit into the refiner's virtual environment. The TRELLIS.2 workers import its modules in-process; no server runs | No | GPL-3.0 | Copyleft. See NOTICE |
| [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) | Git submodule `vendor/ComfyUI-Trellis2`, copied into the refiner's ComfyUI | Submodule | MIT, Copyright (c) Microsoft Corporation | Imports ComfyUI, PyMeshLab, MeshLib and nvdiffrast when it loads. Also carries `vggt/` and `moge/` code and prebuilt wheels the engine does not use |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Model and pipeline design; the code arrives through ComfyUI-Trellis2 | No | MIT | |
| o-voxel, from [visualbruno/TRELLIS.2](https://github.com/visualbruno/TRELLIS.2) | Built from source by `backend/workers/install_trellis2_native.py` | No | MIT; bundles Eigen (MPL-2.0) | Uses nvdiffrast for GLB export |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh), with the API from [visualbruno/CuMesh](https://github.com/visualbruno/CuMesh) | Built from source by `install_trellis2_native.py` | No | MIT | |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | Rasterisation for every textured GLB export, through ComfyUI-Trellis2 and o-voxel | No | NVIDIA Source Code License (1-Way Commercial) | **Non-commercial** (research or evaluation only); redistribution under the same licence |
| [nvdiffrec](https://github.com/NVlabs/nvdiffrec) renderer | A wheel inside ComfyUI-Trellis2, imported lazily by its PBR renderer; not on the mesh stage's path as far as we know | No | NVIDIA Source Code License for nvdiffrec | **Non-commercial** |
| [MeshLib](https://github.com/MeshInspector/MeshLib) (`meshlib` on PyPI) | Imported by ComfyUI-Trellis2 when it loads; called by the TRELLIS.2 worker on the geometry-only and refine paths | No | MeshLib NON-COMMERCIAL & education LICENSE AGREEMENT | **Non-commercial, no transfer to third parties** |
| [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) | Imported by the TRELLIS.2 worker and ComfyUI-Trellis2 | No | GPL-3.0 | Copyleft, in the refiner's process |
| [pymeshfix](https://github.com/pyvista/pymeshfix) | Optional `meshfix` extra; only the Instant Meshes clean method imports it | No | GPL-3.0 (the PyPI classifier for 0.18 says AGPL-3.0) | Copyleft, in the engine's process when installed |
| [open3d](https://github.com/isl-org/Open3D) | Imported by ComfyUI-Trellis2 | No | MIT | |
| [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) | Sparse convolution for ComfyUI-Trellis2 and o-voxel | No | MIT | |
| [flash-attn](https://github.com/Dao-AILab/flash-attention) | Sparse attention backend for the TRELLIS.2 worker | No | BSD-3-Clause | |
| spconv-cu121 | Installed by the studio app's refiner installer | No | Apache-2.0 | |
| custom_rasterizer | A wheel inside ComfyUI-Trellis2; not imported by any code we found | No | Unverified; it appears to derive from Tencent's Hunyuan3D-2 and its community licence | Not exercised |
| [rembg](https://github.com/danielgatis/rembg) and onnxruntime | Background matting | No | MIT | |
| [Blender](https://www.blender.org/about/license/) 4.2 LTS or later | Runs the engine's `bpy` scripts as `blender --background --python ...` | No | GPL-2.0-or-later source, GPL-3.0 binaries | The scripts run inside Blender's process; they are MIT, which is GPL-compatible |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) | Cloned by `scripts/setup_unirig.sh` into its own environment and run as a separate program | No | MIT | Its environment installs `bpy==4.2` (GPL-3.0) |
| [trellis2cpp](https://github.com/rms80/trellis2cpp) | Vendored optional C++/ggml fallback for Generate, `vendor/trellis2cpp` | Yes | MIT, Copyright (c) 2026 rms80 | Bundles ggml (MIT), xatlas (MIT), meshoptimizer (MIT) and stb (public domain or MIT). `scripts/hf/*/LICENSE` are copies of upstream model licences; no weights |
| [Mesh2Motion](https://github.com/Mesh2Motion/mesh2motion-app) | Vendored and modified Rig + Animate workspace, `vendor/mesh2motion-app` | Yes | Code MIT, Copyright (c) 2025 Scott Petrovic; models, rigs and animations CC0 1.0, with the exceptions below | Upstream's Mixamo test rigs and Carnegie Mellon animation sample are left out |
| [Instant Meshes](https://github.com/wjakob/instant-meshes) | Downloaded by the studio app on request, run as a separate program | No | BSD-3-Clause style | |
| NVIDIA Kimodo | Optional motion generation in the studio app, installed by hand | No | Code Apache-2.0 | Its weights and text encoder are listed below |
| Frontend npm dependencies | `frontend/package-lock.json` | Lock file only | Mostly MIT, Apache-2.0 and MPL-2.0 | jszip is dual MIT or GPL-3.0, used under MIT |
| Mesh2Motion npm dependencies | `vendor/mesh2motion-app/package-lock.json` | Lock file only | Mostly MIT | sharp's libvips is LGPL-3.0, build time only |
| Python dependencies | `pyproject.toml` | No | Permissive: fastapi, uvicorn, trimesh, pygltflib, numpy, scipy, networkx, mcp, huggingface_hub, requests, pillow, filelock, tqdm (MPL-2.0 and MIT); optional transformers and lpips | |

### Mesh2Motion assets that need attribution

Every Mesh2Motion model, rig and animation is CC0 1.0 except these:

| Model | File | Author | Licence |
| --- | --- | --- | --- |
| Sophia | `static/models-variation/human-sophia.glb` | Tysan Tan | CC BY-SA 4.0 |
| Jay | `static/models-variation/human-jay.glb` | Blender Studio | CC BY |
| Sintel | `static/models-variation/human-sintel.glb` | Blender Studio | CC BY |
| Bunny | `static/models-variation/human-bunny.glb` | Blender Studio | CC BY |
| Carrot | `static/models-variation/fox-cat.glb` | David Revoy | CC BY |

Mesh2Motion records these in `src/lib/RigModelVariations.ts` and shows the
attribution when one of them is selected.

## Model weights

None of these are in the repository or its history. `aismith-engine weights
install` downloads them by `vendor/weights.lock.json`.

| Weight set | Upstream | Licence | Notes |
| --- | --- | --- | --- |
| `trellis2-4b-fp8` | [visualbruno/TRELLIS.2-4B-FP8](https://huggingface.co/visualbruno/TRELLIS.2-4B-FP8), a repack of microsoft/TRELLIS.2-4B | MIT | |
| `trellis-image-large-ss-decoder` | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | MIT | |
| `dinov3-vitl16` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License (custom, gated) | Not OSI. Each user accepts it on Hugging Face; see `LICENSES/dinov3-acceptance.md` |
| `unirig-skeleton-ar256` | [VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig) | MIT | Training-data terms not reviewed |
| `birefnet-general`, `birefnet-general-lite` | rembg's release of [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | MIT | Training-data terms not reviewed |
| Source-fidelity metrics (optional `identity` extra; not pinned) | openai/clip-vit-large-patch14, facebook/dinov2-base, VGG through lpips | CLIP: check its model card; DINOv2 Apache-2.0; LPIPS BSD-2-Clause | Loaded only when installed |
| Image preparation (a ComfyUI server you run) | FLUX.2 [klein] base 4B, its VAE, Qwen3 4B text encoder | Apache-2.0 | |
| Kimodo motion (installed by hand) | nvidia/Kimodo-SOMA-RP-v1.1; Meta Llama 3 8B Instruct | NVIDIA Open Model License; Meta Llama 3 Community License | Llama 3 needs separate access from Meta |

`briaai/RMBG-2.0` appears in a ComfyUI-Trellis2 workflow file, but the engine
does not load it and it is not in the lock. Its licence is non-commercial.
