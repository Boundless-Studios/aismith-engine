# Engine runtime: pinned sources, pinned weights, offline stages

Magic Studios runs the engine's stages on its hosted GPU worker pool, inside
a hardened runtime with no general network access. Every run must use exactly
what the pool's machine profile pinned. The goal is a runtime that can be built
from this repository alone; the refiner runtime does not meet it yet (see
[Known gap](#known-gap-building-the-refiner-runtime)).

## What is pinned

| File | Pins |
| --- | --- |
| `vendor/sources.lock.json` | Every third-party source by full commit: the vendored submodules, plus the repositories cloned at install time (ComfyUI, CuMesh, TRELLIS.2/o-voxel, UniRig) |
| `vendor/weights.lock.json` | Every model file by Hugging Face revision (or URL), size and sha256 |

Every file now has a locked sha256.

- **DINOv3** (`dinov3-vitl16`) is gated, so Hugging Face does not publish its
  hashes. They were hashed read-only from a working install on 2026-11-02. Its
  `licence_acceptance` field points at
  [LICENSES/dinov3-acceptance.md](../../LICENSES/dinov3-acceptance.md), which
  records Boundless Games, LLC's acceptance of the DINOv3 License. That
  acceptance covers Boundless only: accept the licence with your own Hugging
  Face account before installing.
- **The rembg BiRefNet ONNX files** are pinned by rembg 2.0.81 with md5 only.
  Their sha256 was recorded from the same install, after the md5 and size
  matched.

`verify` still reports any file without a locked hash as `unpinned`, together
with the observed sha256, and the pool must not attest such a runtime.

## Installing (network allowed)

```bash
git submodule update --init vendor/ComfyUI-Trellis2
bash scripts/setup_unirig.sh                        # pinned UniRig checkout + checkpoint, first
python -m backend.engine.weights install            # then every other weight set
```

Run the UniRig setup first: it clones UniRig into `vendor/unirig`, which must not exist yet, and the full weights install would create it by writing the UniRig checkpoint there. `install` fetches each file at its pinned revision, hashes it, removes and
fails on any mismatch, and caches the verified hashes in
`<trellis_models>/.weights-verified.json`. Set `HF_TOKEN` to a token for an
account that has accepted the DINOv3 License. Roots default to the app's
locations:

| Root | Default location |
| --- | --- |
| `trellis_models` | `TRELLIS_REFINER_DIR/models` |
| `unirig` | `UNIRIG_DIR` or `vendor/unirig` |
| `u2net` | `U2NET_HOME`, `REMBG_HOME` or `~/.rembg`; rembg's files go in `models/<name>/` |

Override any of them with `--root name=path`.

In a container build with no GPU, set `AISMITH_BUILD_WITHOUT_GPU=1` for
`setup_unirig.sh`. It then checks imports but not the CUDA device.

## Running offline

Set `AISMITH_OFFLINE=1`.

- Every stage subprocess started through `native.run_process` gets
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `U2NET_HOME`.
- The app's in-UI downloader refuses to run.
- UniRig loads its pinned checkpoint from `experiments/` when the hub is
  offline.
- rembg finds its ONNX files in `U2NET_HOME` and skips the download.

```bash
python -m backend.engine.weights verify          # cached hashes, fast
python -m backend.engine.weights verify --full   # rehash everything
```

## Known gap: building the refiner runtime

The TRELLIS.2 refiner runtime (`TRELLIS_REFINER_DIR`, with its own
`.venv-cu128`) is what the mesh stage runs in. The pins above fix its sources
and weights. The qualified runtime the pool uses was assembled on a
workstation, however, and the studio app's installer
(`_run_trellis_refiner_download` in `backend/app/main.py`) does not reproduce
it yet:

- It locates ComfyUI-Trellis2's prebuilt CUDA wheels but never installs them.
- It does not install MeshLib, PyMeshLab, open3d, OpenCV, flash-attn,
  nvdiffrast or FlexGEMM, all of which ComfyUI-Trellis2 or the worker imports.
  Their versions in the qualified runtime are not recorded here.
- It installs torch 2.9.1 for CUDA 13.0, while the qualified runtime uses
  torch 2.8 for CUDA 12.8.

Until a scripted, pinned build exists (a container with
`TORCH_CUDA_ARCH_LIST` that installs those packages at pinned versions), a
working runtime has to be assembled by hand. Check the result with
`aismith-engine inspect --full`. Mind the licences of nvdiffrast and MeshLib:
both are non-commercial (see [NOTICE](../../NOTICE)).

## Not part of the engine runtime

`scripts/start-wsl.sh` launches the studio app under WSL; the engine's own
entry point does not use it. The app's optional rebuild-from-views stage needs
Pixal3D research scripts that are not part of this repository, and is
disabled without them.

## The stage entry point

```bash
aismith-engine run mesh  --spec spec.json --workdir DIR   # or python -m backend.engine
aismith-engine run clean --spec spec.json --workdir DIR
aismith-engine inspect [--full]
aismith-engine weights install|verify [ids] [--root name=path]
```

A spec is one JSON object (`backend/engine/contract.py`):

```json
{"schema_version": 1, "stage": "mesh",
 "inputs": {"front": "/in/front.png", "back": "/in/back.png"},
 "settings": {"resolution": 1024, "seed": 1337}}
```

| Stage | Inputs | Settings | Output |
| --- | --- | --- | --- |
| `mesh` | `front` (required); optional `back`, `left` and `right` images | `resolution`: 512, 768 or 1024 (768 runs at 1024); `seed` | textured GLB |
| `clean` | `mesh` (GLB); optional view images | `quality_tier` or `target_triangles`; palette and `project_views` knobs | gated GLB |

`run` writes `result.json` to the workdir and prints it. The result
contains:

- `status`
- `output`: path, content type, sha256 and bytes
- `gate`: for `clean`, the same S2 decision the app's job runner makes
  (`canonicalize_policy.decide_canonicalize`)
- `effective_inputs`: what actually ran
- `measured`
- `engine`: version, commit and the digests of both locks
- `weights`: the stage's weight check
- `diagnostics` and `timings_s`

Exit codes:

| Code | Status | Meaning |
| --- | --- | --- |
| 0 | `ok` | an output was produced, even when its gate failed |
| 1 | `failed` | runtime, weights, GPU or a crash; never replayed |
| 2 | `rejected` | an invalid spec or input; terminal |

`run` always runs offline. It sets
`AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS=0`, so a busy GPU fails the job
instead of waiting. The pool hands the GPU over before each job.

## Engine runner (`serve`)

On a hardened worker the engine runs in its own sidecar container: it holds
the GPU, the read-only model stores and the engine runtimes, and **no
credentials**. The agent container holds the worker credential but no GPU.
They share only a loopback network (`network_mode: service:engine`, as
ComfyUI's sidecar does), and the agent drives the engine over HTTP:

```bash
aismith-engine serve [--host 127.0.0.1] [--port 8190] [--work-root DIR] [--allow-private]
```

| Property | Rule |
| --- | --- |
| Binding | Loopback by default. Private, link-local or unspecified addresses (including `0.0.0.0`) only with `--allow-private`; public addresses and hostnames other than `localhost` never |
| Auth | None. No secrets live in the engine container, and nothing goes out to the network |
| Concurrency | One job at a time; a second `run` gets 409 `{"error": "busy"}` |
| Offline | Every stage runs as `aismith-engine run`, with `AISMITH_OFFLINE=1` and `AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS=0` |
| Isolation | Each `run` is a subprocess in its own process group, killed as a group on deadline, cancel or shutdown. The runner sets `AISMITH_JOB_PROCESS_GROUP=1`, so native workers (`native.run_process`) stay in that group instead of calling setsid. As a fallback it records the stage's process tree while it runs (`/proc` on Linux) and SIGKILLs and reaps any descendant still alive when the stage ends, in whatever session it is |
| Work directory | `<work-root>/<job_id>/` (inputs, spec, `run/`, `engine.log`). Removed on `DELETE`. The runner holds an exclusive `flock` on `<work-root>/.lock` for its lifetime, taken before anything is cleaned, so a second runner on the same root refuses to start. At startup every other entry of the work root is removed; a non-empty directory without the runner's `.aismith-engine-serve` marker is refused rather than cleaned. Default `$TMPDIR/aismith-engine-serve` |
| Request size | `Content-Length` required (411 otherwise, chunked bodies included). Images 20 MiB, `mesh` 256 MiB, the run body 64 KiB; larger gets 413 before the body is read. Uploads stream to disk in 1 MiB chunks |
| Job id | `^[A-Za-z0-9_-]{1,64}$`, else 400 |
| Input names | `front`, `back`, `left`, `right`, `mesh`, else 400. The stage decides which it accepts (`rejected` otherwise) |

| Method and path | Response |
| --- | --- |
| `GET /health` | `{"ok": true, "busy": bool, "schema_version": 1}` |
| `GET /inspect?full=0\|1` | The `aismith-engine inspect [--full]` JSON. 500 if it fails, 504 on timeout |
| `PUT /jobs/{id}/inputs/{name}` | Raw bytes; 204. A re-upload replaces the input. 409 while that job runs |
| `POST /jobs/{id}/run` | Body `{"schema_version": 1, "stage": "mesh"\|"clean", "inputs": [names], "settings": {...}, "deadline_seconds": int}`. Blocks, then returns the result JSON with `output.path` replaced by `output.url` (`/jobs/{id}/output`). 200 for `ok`, `failed` and `rejected`; 404 for an input not uploaded; 400 for a malformed body |
| `GET /jobs/{id}/output` | The GLB (`model/gltf-binary`), with `Content-Length` and the result's sha256 in `X-Sha256`. 404 if none |
| `POST /jobs/{id}/cancel` | 202. SIGTERM to the job's process group, SIGKILL after 10 s |
| `DELETE /jobs/{id}` | 204. Cancels a running job, waits for it to stop, then removes its work directory |

The runner reports its own failures in the result schema with `status:
failed`: `"deadline exceeded"`, `"cancelled"`, an engine that exited without a
valid `result.json`, or an output outside the job's work directory. These carry
the engine identity and `diagnostics.runner` (exit code and the log's tail).
The client still applies its own deadline and checks the fetched bytes'
size and sha256.
