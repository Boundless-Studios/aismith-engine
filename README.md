# AISmith engine

AISmith turns a reference image into a rigged, animated 3D character. It does
this in four steps: TRELLIS.2 generates a mesh, a gated clean pass makes it
watertight, a skeleton is fitted and bound, and the character is animated.

The repository holds two things:
- **the engine**, a pinned, offline command-line runner for each stage
  (`aismith-engine`);
- **the studio app** that drives it: a local FastAPI backend and a React UI
  with a browser-based Rig + Animate workspace built on Mesh2Motion.

[Magic Studios](https://github.com/Boundless-Studios/magic-studios) runs the
engine's stages on its hosted GPU worker pool
([outbound-gpu-worker-pool](https://github.com/Boundless-Studios/outbound-gpu-worker-pool)),
which pins one engine release at a time.

> **Before any commercial use, read [NOTICE](NOTICE).** This repository's
> code is MIT. The mesh stage's runtime, however, loads nvdiffrast and MeshLib,
> and their licences forbid commercial use. Several components are GPL, and the
> DINOv3 weights are gated under their own licence, which every user accepts
> for themselves.

## What it does

| Workflow | Result | Needs |
| --- | --- | --- |
| **Generate** | Image to textured GLB with TRELLIS.2 FP8, optionally conditioned on back and side views | NVIDIA CUDA GPU, the TRELLIS.2 runtime and its weights |
| **Clean** | A canonical mesh: voxel remesh, a bake from the original, and a blocking quality gate (S2) | Blender 4.2 or later |
| **Rig + Animate** | Fit a skeleton, bind weights, preview and choose animation clips, export GLB | A modern browser; UniRig for automatic rigging |
| **Texture Paint** | Paint a mesh in the browser, bake a base-colour texture, export GLB | Blender 4.2 or later |

Generate can also fall back to the vendored trellis2cpp C++/ggml pipeline when
its GGUF models are present and the TRELLIS.2 runtime is not.

The stages and their contracts are defined in
[docs/PIPELINE_POLICY.md](docs/PIPELINE_POLICY.md). The engine's entry point,
pins and offline rules are in
[docs/engineering/engine-runtime.md](docs/engineering/engine-runtime.md).

## Status and known gaps

- Every third-party source is pinned by commit in `vendor/sources.lock.json`,
  and every model file by revision and sha256 in `vendor/weights.lock.json`.
- **The TRELLIS.2 refiner runtime cannot yet be rebuilt from this repository
  alone.** The in-app installer does not install everything ComfyUI-Trellis2
  imports, and it targets a different torch build from the qualified runtime.
  See "Known gap" in
  [engine-runtime.md](docs/engineering/engine-runtime.md#known-gap-building-the-refiner-runtime).
- GPU qualification of a release runs outside public CI, because it needs
  weights and a GPU.

## Requirements

- Linux, WSL2 or Windows 10/11
- Python 3.11 or 3.12
- Node.js 20 or newer, with npm
- For Generate: an NVIDIA GPU, a current driver and CUDA-capable PyTorch.
  Without one, the UI and Rig + Animate still work, and the inference buttons
  report that CUDA is required.
- For Clean, Texture Paint and Blender-assisted conversion: Blender 4.2 or
  later.

No model weights are in this repository. They are downloaded only when you
choose a workflow, or with `aismith-engine weights install`.

## Install and run

```bash
git clone --recurse-submodules https://github.com/Boundless-Studios/aismith-engine.git
cd aismith-engine
```

In an existing clone, run `git submodule update --init` once.

### Windows (PowerShell)

```powershell
.\start.ps1 -Install
```

This creates `.venv`, installs the Python and frontend dependencies, detects
NVIDIA hardware, and starts the API on `127.0.0.1:8000` and the UI on
`127.0.0.1:5173`. Add `-InstallBlender` to install Blender with `winget` when
it is missing. Later launches need only `.\start.ps1`; use
`.\start.ps1 -RestartApi` to replace an API that is still running.

### Linux or WSL2

```bash
bash scripts/start-wsl.sh --install   # first run
bash scripts/start-wsl.sh             # later runs
bash scripts/stop-wsl.sh              # stops only the processes it started
```

The launcher prints the address to open. By default it binds loopback
(`127.0.0.1`); WSL2 forwards Windows' `localhost` to it, so open
`http://localhost:5173` in a Windows browser. The API has no authentication.
To reach it from another machine, prefer an SSH tunnel. Setting
`AISMITH_API_HOST` and `AISMITH_UI_HOST` to `0.0.0.0` in `.env` exposes it to
your whole network, so do that only on a network you trust.

### Configuration

Copy `.env.example` to `.env` only when you need non-default paths. Relative
paths resolve from the repository root. For example, to keep models and data
on a separate disk:

```dotenv
AISMITH_DATA_DIR=/srv/aismith/studio-data
TRELLIS_REFINER_DIR=/srv/aismith/tools/trellis2-refiner
BLENDER_EXE=/opt/blender-4.2-linux-x64/blender
```

| Variable | Purpose |
| --- | --- |
| `BLENDER_EXE` | Blender executable |
| `AISMITH_API_HOST` / `AISMITH_API_PORT` | API bind address and port |
| `AISMITH_UI_HOST` / `AISMITH_UI_PORT` | Vite UI bind address and port |
| `AISMITH_DATA_DIR` | Uploads, artifacts, jobs, runs, logs and runtime state |
| `AISMITH_CORS_ORIGINS` | Extra comma-separated UI origins |
| `TRELLIS_REFINER_DIR` | The TRELLIS.2 runtime (refiner and generator) |
| `TRELLIS_GGUF_MODEL_DIR` | GGUF models for the trellis2cpp fallback |
| `UNIRIG_DIR` | A prepared UniRig checkout, if not `vendor/unirig` |
| `PYTORCH_INDEX_URL` | Override the CUDA PyTorch package index |

### Models

In the UI, use **Install 3D model** in Generate, or **Download** for a Clean
method that needs one. Downloads go under `studio-data/`, which Git ignores.

The engine installs and verifies its pinned weights itself:

```bash
bash scripts/setup_unirig.sh                 # pinned UniRig checkout and checkpoint
python -m backend.engine.weights install     # every other weight set
python -m backend.engine.weights verify
```

The DINOv3 weights are gated. Accept the DINOv3 License on the model's Hugging
Face page with your own account, then set `HF_TOKEN` for that account. The
acceptance recorded in
[LICENSES/dinov3-acceptance.md](LICENSES/dinov3-acceptance.md) is Boundless
Games, LLC's own; it covers Boundless only and does not extend to you.

### Optional extras

```bash
pip install -e ".[meshfix]"    # pymeshfix (GPL-3.0), for the Instant Meshes clean method
pip install -e ".[identity]"   # CLIP, DINOv2 and LPIPS source-fidelity metrics (pulls torch)
```

## The engine entry point

```bash
aismith-engine run mesh  --spec spec.json --workdir DIR
aismith-engine run clean --spec spec.json --workdir DIR
aismith-engine inspect [--full]
aismith-engine weights install|verify
aismith-engine serve [--host 127.0.0.1] [--port 8190]
```

`run` is offline and writes a `result.json` that records what actually ran,
the engine version and commit, and the digests of both lock files. `serve` is
a loopback HTTP runner for a sidecar container. See
[engine-runtime.md](docs/engineering/engine-runtime.md).

## Checks

```bash
python -m pip install -e ".[test,meshfix]"
python -m pytest -q backend/tests
ruff check backend scripts tools
(cd vendor/mesh2motion-app && npm ci) && (cd frontend && npm ci && npm test && npm run build)
```

CI runs these on GitHub-hosted runners. See
[docs/engineering/releasing.md](docs/engineering/releasing.md).

## Driving the pipeline from an agent (MCP)

`backend/mcp_server.py` is a stdio MCP server. Through it, an external agent
can drive the pipeline over the HTTP API: list and step runs, submit jobs and
inspect artifacts, under the same guardrails the API enforces. See
[docs/ASTRA_OPERATOR.md](docs/ASTRA_OPERATOR.md) for the delegation contract,
the tool list and the `.mcp.json` setup.

## Architecture

`reference image → TRELLIS.2 FP8 mesh → S2 canonical clean → skeleton fit and bind → Mesh2Motion animation`

A coordinator lease serialises heavy inference. TRELLIS.2 runs in a
short-lived process in its own virtual environment. Blender stages are
headless and short-lived. Tabs pass artifact IDs and files rather than large
meshes through browser memory.

Frameworks and deep multi-step flows follow the
[layered system design and validation practice](docs/engineering/layered-system-design.md):
sketch the whole system, define every handoff, then validate layers in
dependency order.

## Asset evaluation and material sources

Two separately installable packages live under `packages/`, and neither
changes the pipeline's gates:
- [asset judge](packages/asset-judge/README.md): an engine-neutral result
  schema, independent review settlement, bounded iteration and
  adversarial-camera strategies.
- [Poly Haven adapter](packages/material-sources/README.md): searchable
  metadata, explicit map locks and verified material ingestion.

## Origin

AISmith-3D began as Intisar Hasnain's project,
[intisarGIT/AISmith-3D](https://github.com/intisarGIT/AISmith-3D), released
under the MIT License. Boundless Games, LLC forked it at `8af574d` and has
developed it since as the 3D engine behind Magic Studios.

## Licence and acknowledgements

The code in this repository is under the [MIT License](LICENSE). The
following files cover third-party components:
- [NOTICE](NOTICE) covers the components the engine fetches and runs, including
  the non-commercial and GPL ones, and the model weights;
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the full table;
- [CONTRIBUTING.md](CONTRIBUTING.md) explains how to contribute (MIT, with a
  DCO sign-off).

Thank you to the maintainers and contributors of
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2),
[ComfyUI](https://github.com/comfyanonymous/ComfyUI),
[trellis2cpp](https://github.com/rms80/trellis2cpp),
[UniRig](https://github.com/VAST-AI-Research/UniRig),
[Mesh2Motion](https://github.com/Mesh2Motion/mesh2motion-app) and
[Blender](https://www.blender.org/).
