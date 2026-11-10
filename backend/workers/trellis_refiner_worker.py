"""One-shot, image-guided TRELLIS.2 mesh-refinement worker.

This intentionally runs in its own process.  The ComfyUI TRELLIS.2 runtime
loads several CUDA extensions and large weights; keeping it out of the API
process guarantees that VRAM is returned after every job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_wrapper(wrapper_root: Path):
    """Load the custom node as a package so its relative imports keep working."""
    init = wrapper_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "aismith_trellis2", init, submodule_search_locations=[str(wrapper_root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the managed TRELLIS.2 refiner package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    args = parser.parse_args()
    job = json.loads(args.spec)
    runtime = Path(job["runtime"]).resolve()
    comfy = runtime / "ComfyUI"
    wrapper = comfy / "custom_nodes" / "ComfyUI-Trellis2"
    if not (comfy / "folder_paths.py").exists() or not (wrapper / "nodes.py").exists():
        raise RuntimeError("AI Refine is not installed. Download the TRELLIS.2 FP8 refiner from Retopologize first.")

    # ComfyUI supplies folder_paths/comfy, while the custom node supplies the
    # TRELLIS.2 pipeline and matching CUDA extensions.
    sys.path.insert(0, str(comfy))
    import folder_paths
    from PIL import Image

    folder_paths.set_base_directory(str(runtime))
    _load_wrapper(wrapper)
    nodes = sys.modules["aismith_trellis2.nodes"]
    input_glb = Path(job["input_glb"])
    source_image = Path(job["source_image"])
    output_obj = Path(job["output_obj"])
    if not source_image.is_file():
        raise RuntimeError("The original Generate reference image is unavailable; AI Refine requires that image.")

    print("Loading FP8 TRELLIS.2 image-guided refiner", flush=True)
    pipeline = nodes.Trellis2LoadModel().process(
        "visualbruno/TRELLIS.2-4B-FP8", "sdpa", "cuda", True, False,
        "flex_gemm", "flash_attn", False,
    )[0]
    print("Loading source mesh and reference image", flush=True)
    mesh = nodes.Trellis2LoadMesh().load(str(input_glb))[0]
    with Image.open(source_image) as image:
        # The wrapper expects an IMAGE tensor in RGB(A) channel-last form.
        image_tensor = nodes.pil2tensor(image.convert("RGBA"))

    print("Refining geometry from the source image", flush=True)
    refined_mesh, _ = nodes.Trellis2MeshRefiner().process(
        pipeline, mesh, image_tensor,
        seed=0, resolution=512,
        shape_steps=12, shape_guidance_strength=6.5,
        shape_guidance_rescale=0.05, shape_rescale_t=4.0,
        texture_steps=0, texture_guidance_strength=3.0,
        texture_guidance_rescale=0.2, texture_rescale_t=3.0,
        max_num_tokens=999999, generate_texture_slat=False,
        downsampling=16, shape_guidance_interval_start=0.1,
        shape_guidance_interval_end=1.0, texture_guidance_interval_start=0.0,
        texture_guidance_interval_end=0.9, use_tiled_decoder=True,
        max_views=1, sampler="euler", verbose=False,
        dino_lock=0.0, dino_substeps=4, dino_foundation_cap=1.0,
    )
    print("Reconstructing a clean quad-dominant surface", flush=True)
    reconstructed = nodes.Trellis2ReconstructMeshWithQuad().process(
        refined_mesh, remesh_band=1.0, resolution=512,
        remove_floaters=True, remove_inner_faces=True,
    )[0]
    result = nodes.Trellis2MeshWithVoxelToTrimesh().process(reconstructed, "90 degrees")[0]
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    result.export(output_obj, file_type="obj")
    if not output_obj.is_file() or output_obj.stat().st_size < 64:
        raise RuntimeError("AI Refine did not produce a valid mesh")
    print("AI refinement complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
