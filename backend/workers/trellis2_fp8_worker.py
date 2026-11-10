"""Isolated FP8 TRELLIS.2 generation and mesh-refinement worker.

This intentionally invokes the ComfyUI wrapper's Python nodes without starting
ComfyUI's server.  The model stack is loaded in this one-shot process so CUDA
memory is returned to Windows as soon as the job exits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

_DEFAULT_MATTING_MODEL = "birefnet-general"
_FALLBACK_MATTING_MODEL = "birefnet-general-lite"


def _validate_native_contract(
    runtime: Path,
    capability: tuple[int, int] | None = None,
) -> None:
    marker = runtime / "native-contract.json"
    if not marker.is_file():
        raise RuntimeError(
            "TRELLIS.2 native extensions have no GPU architecture contract; "
            "reinstall the refiner runtime"
        )
    contract = json.loads(marker.read_text())
    if capability is None:
        import torch

        capability = torch.cuda.get_device_capability()
    required = f"{capability[0]}.{capability[1]}"
    installed = str(contract.get("cuda_architecture", ""))
    if installed != required:
        raise RuntimeError(
            f"TRELLIS.2 native extensions were built for CUDA architecture {installed}; "
            f"the current GPU requires {required}. Reinstall the refiner runtime."
        )

def _load_wrapper(wrapper_root: Path):
    init = wrapper_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "aismith_trellis2", init, submodule_search_locations=[str(wrapper_root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the managed TRELLIS.2 package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules["aismith_trellis2.nodes"]


def _load_nodes(runtime: Path):
    comfy = runtime / "ComfyUI"
    wrapper = comfy / "custom_nodes" / "ComfyUI-Trellis2"
    if (
        not (comfy / "folder_paths.py").is_file()
        or not (wrapper / "nodes.py").is_file()
    ):
        raise RuntimeError(
            "TRELLIS.2 FP8 is not installed. Download the model from Generate first."
        )
    sys.path.insert(0, str(comfy))
    import folder_paths

    # ComfyUI removed ``set_base_directory``.  This worker loads only the
    # wrapper nodes, so pointing its model registry at our managed runtime is
    # sufficient and works with both current and older ComfyUI builds.
    models_dir = str(runtime / "models")
    folder_paths.models_dir = models_dir
    for name, (paths, extensions) in folder_paths.folder_names_and_paths.items():
        folder_paths.folder_names_and_paths[name] = (
            [path.replace(str(comfy / "models"), models_dir) for path in paths],
            extensions,
        )
    folder_paths.set_output_directory(str(runtime / "output"))
    folder_paths.set_temp_directory(str(runtime / "temp"))
    folder_paths.set_input_directory(str(runtime / "input"))
    return _load_wrapper(wrapper)


def _load_pipeline(nodes, gpu_first: bool = False):
    conv_backend = "spconv"
    _patch_fp8_compute(conv_backend)
    print(
        f"Loading TRELLIS.2-4B FP8 (conv={conv_backend}, sparse_compute=float16)",
        flush=True,
    )
    # Current ComfyUI-Trellis2 unconditionally copies a helper config into
    # this directory before it checks the FP8 option.  The config itself is
    # tiny, but its parent needs to exist on a clean install.
    import folder_paths

    Path(folder_paths.models_dir, "microsoft", "TRELLIS.2-4B").mkdir(
        parents=True, exist_ok=True
    )
    return nodes.Trellis2LoadModel().process(
        modelname="visualbruno/TRELLIS.2-4B-FP8",
        backend="sdpa",
        device="cuda",
        low_vram=not gpu_first,
        keep_models_loaded=gpu_first,
        conv_backend=conv_backend,
        sparse_backend="flash_attn",
        use_reconviagen=False,
    )[0]


def _matting_model_name() -> str:
    return os.environ.get("AISMITH_MATTING_MODEL", _DEFAULT_MATTING_MODEL)


def _create_matting_session():
    from rembg import new_session

    model = _matting_model_name()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = new_session(model, providers=providers)
    except Exception as exc:
        print(
            f"Matting model '{model}' failed to load ({exc}); "
            f"falling back to '{_FALLBACK_MATTING_MODEL}'",
            flush=True,
        )
        model = _FALLBACK_MATTING_MODEL
        session = new_session(model, providers=providers)
    inner = getattr(session, "inner_session", None)
    used_providers = inner.get_providers() if inner is not None else []
    device = "cuda" if "CUDAExecutionProvider" in used_providers else "cpu"
    print(
        f"Using matting model: {model} (device: {device}, providers={used_providers})",
        flush=True,
    )
    return session


def _harden_alpha(alpha):
    from PIL import ImageFilter

    return (
        alpha.point(lambda v: 255 if v > 38 else 0)
        .filter(ImageFilter.MaxFilter(3))
        .filter(ImageFilter.GaussianBlur(0.8))
    )


def _padded_square_crop(rgba, mask):
    from PIL import Image

    bbox = mask.getbbox()
    if bbox is None:
        return rgba, bbox
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    side = max(width, height)
    pad = int(side * 0.08)
    canvas_size = side + 2 * pad
    subject = rgba.crop(bbox)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    origin = ((canvas_size - width) // 2, (canvas_size - height) // 2)
    canvas.paste(subject, origin, subject)
    return canvas, bbox


def _foreground_coverage(mask) -> float:
    import numpy as np

    array = np.asarray(mask)
    return float(np.count_nonzero(array)) / float(array.size) * 100.0


def _harden_foreground_with_stats(rgba):
    mask = _harden_alpha(rgba.getchannel("A"))
    hardened = rgba.copy()
    hardened.putalpha(mask)
    framed, bbox = _padded_square_crop(hardened, mask)
    coverage = _foreground_coverage(mask)
    return framed, coverage, bbox


def harden_foreground(rgba):
    """Harden a soft alpha mask to opaque/transparent and frame the subject
    in a padded square canvas. Pure -- no rembg dependency, safe to unit test.
    """
    framed, _coverage, _bbox = _harden_foreground_with_stats(rgba)
    return framed


def _prepare_foreground(image, remove_background):
    rgba = image.convert("RGBA")
    if rgba.getchannel("A").getextrema() != (255, 255):
        source = rgba
    else:
        print("Removing opaque image background for TRELLIS", flush=True)
        source = remove_background(image.convert("RGB")).convert("RGBA")

    framed, coverage, bbox = _harden_foreground_with_stats(source)
    print(
        f"Foreground coverage={coverage:.1f}% opaque, crop_box={bbox}",
        flush=True,
    )
    return framed


def _image_tensor(nodes, path: Path):
    from PIL import Image

    def remove_background(image):
        from rembg import remove

        return remove(image, session=_create_matting_session())

    with Image.open(path) as image:
        prepared = _prepare_foreground(image, remove_background)
        return nodes.pil2tensor(prepared)


def _scaled_square_crop(hardened, bbox, shared_side: int, canvas_size: int):
    """Crop ``hardened`` to its own ``bbox``, scale so the crop's longer side
    equals ``shared_side``, then paste centered onto a ``canvas_size`` square.
    Every view that goes through this shares the exact same output pixel
    dimensions and the exact same real-world subject scale, even though each
    view's own bbox (and therefore native crop size) differs.
    """
    from PIL import Image

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    if bbox is None or shared_side <= 0:
        return canvas
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    side = max(width, height)
    subject = hardened.crop(bbox)
    scale = shared_side / side if side else 1.0
    if abs(scale - 1.0) > 1e-9:
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        subject = subject.resize(scaled_size, Image.LANCZOS)
    else:
        scaled_size = (width, height)
    origin = (
        (canvas_size - scaled_size[0]) // 2,
        (canvas_size - scaled_size[1]) // 2,
    )
    canvas.paste(subject, origin, subject)
    return canvas


def _prepare_multiview_foregrounds(images: dict[str, object], remove_background) -> dict[str, object]:
    """Prepare every conditioning view identically for multi-view TRELLIS.2.

    Two consistency bugs this fixes relative to running ``_prepare_foreground``
    independently per view: (1) whether a view already carries usable alpha is
    a per-image accident (e.g. a "front" that came from a prepared/matted
    upstream artifact vs. a raw "back" upload) -- if ANY view lacks real alpha
    every view is matted through the same birefnet pass, and only if ALL views
    already carry real alpha does none of them get matted; (2)
    ``_padded_square_crop`` used to size/pad each view from its own bbox alone,
    so two views photographed at different distances ended up at different
    pixel scales on canvases of different sizes. Here every view is scaled to
    one shared bbox side length and centered on one shared canvas size.
    """
    labels = list(images.keys())
    rgba_by_label = {label: images[label].convert("RGBA") for label in labels}

    # (1) Matting must be all-or-nothing across views, never per-image.
    lacks_real_alpha = {
        label: rgba.getchannel("A").getextrema() == (255, 255)
        for label, rgba in rgba_by_label.items()
    }
    matte_all = any(lacks_real_alpha.values())
    if matte_all:
        print("Removing opaque image background for TRELLIS (all views)", flush=True)

    hardened_by_label: dict[str, object] = {}
    bbox_by_label: dict[str, tuple[int, int, int, int] | None] = {}
    coverage_by_label: dict[str, float] = {}
    for label in labels:
        rgba = rgba_by_label[label]
        source = (
            remove_background(rgba.convert("RGB")).convert("RGBA")
            if matte_all
            else rgba
        )
        mask = _harden_alpha(source.getchannel("A"))
        framed = source.copy()
        framed.putalpha(mask)
        hardened_by_label[label] = framed
        bbox_by_label[label] = mask.getbbox()
        coverage_by_label[label] = _foreground_coverage(mask)

    print(
        "Multi-view crops: "
        + ", ".join(
            f"{label}(coverage={coverage_by_label[label]:.1f}%, crop_box={bbox_by_label[label]})"
            for label in labels
        ),
        flush=True,
    )

    heights = [
        bbox[3] - bbox[1] for bbox in bbox_by_label.values() if bbox is not None
    ]
    if len(heights) > 1:
        max_height, min_height = max(heights), min(heights)
        if max_height > 0 and (max_height - min_height) / max_height > 0.10:
            print(
                "WARNING: multi-view bbox heights differ by "
                f"{(max_height - min_height) / max_height * 100:.1f}% across views; "
                "conditioning views may not depict the subject at a consistent framing",
                flush=True,
            )

    # (2) One shared side length (the union/max of every view's own bbox side)
    # and one shared 8% pad drive a single canvas size for every view.
    sides = [
        max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        for bbox in bbox_by_label.values()
        if bbox is not None
    ]
    shared_side = max(sides) if sides else 0
    pad = int(shared_side * 0.08)
    canvas_size = shared_side + 2 * pad

    return {
        label: _scaled_square_crop(
            hardened_by_label[label], bbox_by_label[label], shared_side, canvas_size
        )
        for label in labels
    }


def _multiview_tensors(nodes, view_paths: dict[str, Path]) -> dict[str, object]:
    from PIL import Image

    def remove_background(image):
        from rembg import remove

        return remove(image, session=_create_matting_session())

    images: dict[str, object] = {}
    try:
        for label, path in view_paths.items():
            with Image.open(path) as opened:
                images[label] = opened.convert("RGBA")
        framed = _prepare_multiview_foregrounds(images, remove_background)
    finally:
        for image in images.values():
            image.close()
    return {label: nodes.pil2tensor(frame) for label, frame in framed.items()}


def _ensure_smooth_vertex_normals(mesh) -> None:
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_vectors = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_vectors)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 0
    vertex_normals[valid] /= lengths[valid, None]
    mesh.vertex_normals = vertex_normals


def _export(trimesh, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_smooth_vertex_normals(trimesh)
    trimesh.export(output, file_type="glb")
    if not output.is_file() or output.stat().st_size < 64:
        raise RuntimeError("TRELLIS.2 did not produce a valid GLB")


def _validate_mesh(mesh) -> None:
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("TRELLIS.2 simplification produced an empty mesh")
    if not np.isfinite(vertices).all():
        raise RuntimeError("TRELLIS.2 simplification produced non-finite vertices")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("TRELLIS.2 produced non-triangular topology")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError("TRELLIS.2 produced out-of-range face indices")
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    if not nondegenerate.any():
        raise RuntimeError("TRELLIS.2 produced only degenerate faces")


def _face_components(mesh) -> list[list[int]]:
    import numpy as np

    face_count = len(mesh.faces)
    if face_count == 0:
        return []

    parents = list(range(face_count))

    def find(face: int) -> int:
        while parents[face] != face:
            parents[face] = parents[parents[face]]
            face = parents[face]
        return face

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    topology = mesh.copy()
    topology.merge_vertices(merge_tex=True, merge_norm=True)
    edges = np.asarray(topology.edges_sorted)
    edge_faces = np.asarray(topology.edges_face)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_faces = edge_faces[order]
    for index in range(1, len(sorted_edges)):
        if np.array_equal(sorted_edges[index - 1], sorted_edges[index]):
            union(int(sorted_faces[index - 1]), int(sorted_faces[index]))

    components: dict[int, list[int]] = {}
    for face in range(face_count):
        components.setdefault(find(face), []).append(face)
    return list(components.values())


def _retain_dominant_component(mesh):
    components = _face_components(mesh)
    if not components:
        raise RuntimeError("TRELLIS.2 produced no connected geometry")
    dominant_faces = max(components, key=len)
    dominant = mesh.submesh([dominant_faces], append=True, repair=False)
    dominant.update_faces(dominant.nondegenerate_faces())
    dominant.remove_unreferenced_vertices()
    return dominant


def _validate_character_mesh(mesh) -> None:
    import numpy as np

    _validate_mesh(mesh)
    topology = mesh.copy()
    topology.merge_vertices(merge_tex=True, merge_norm=True)
    vertices = np.asarray(topology.vertices)
    faces = np.asarray(topology.faces)
    areas = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    if np.any(areas < 1e-12):
        raise RuntimeError("TRELLIS.2 character contains degenerate faces")
    if len(_face_components(mesh)) != 1:
        raise RuntimeError("TRELLIS.2 character is not one connected component")
    edges = np.sort(
        np.vstack(
            (
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            )
        ),
        axis=1,
    )
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))
    # These counts are diagnostic evidence for downstream rig and deformation
    # review. Neither count alone proves that a connected character is unusable.
    print(
        "Character topology diagnostics: "
        f"boundary_edges={boundary_edges}, "
        f"nonmanifold_edges={nonmanifold_edges}",
        flush=True,
    )


def _nonmanifold_edge_count(mesh) -> int:
    import numpy as np

    faces = np.asarray(mesh.faces)
    edges = np.sort(
        np.vstack(
            (
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            )
        ),
        axis=1,
    )
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    return int(np.sum(counts > 2))


def _repair_mesh_set(mesh_set) -> None:
    mesh_set.meshing_repair_non_manifold_edges(method="Remove Faces")
    mesh_set.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    mesh_set.meshing_remove_unreferenced_vertices()
    mesh_set.meshing_close_holes(
        maxholesize=100,
        newfaceselected=False,
        selfintersection=False,
    )


def _simplify_mesh(mesh, target_faces: int):
    if len(mesh.faces) <= target_faces and _nonmanifold_edge_count(mesh) == 0:
        return mesh
    import numpy as np
    import pymeshlab
    import trimesh

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        )
    )
    _repair_mesh_set(mesh_set)
    if mesh_set.current_mesh().face_number() > target_faces:
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
        )
    _repair_mesh_set(mesh_set)
    result = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=result.vertex_matrix(),
        faces=result.face_matrix(),
        process=False,
    )


def _texture(nodes, pipeline, image, trimesh, resolution: int):
    print("Generating PBR textures for final geometry", flush=True)
    return nodes.Trellis2MeshTexturing().process(
        pipeline,
        image,
        trimesh,
        random.randint(0, 0x7FFFFFFF),
        12,
        3.0,
        0.20,
        3.0,
        resolution,
        2048,
        "OPAQUE",
        False,
        0.0,
        0.90,
        1,
        False,
        False,
        60.0,
        "euler",
        "telea",
        False,
        0.0,
        4,
        1.0,
    )[0]


def _generate(
    nodes,
    pipeline,
    image,
    output: Path,
    resolution: int,
    target_faces: int = 60_000,
    generate_texture: bool = True,
    seed: int = 1337,
) -> None:
    preset = "512" if resolution <= 512 else "1024_cascade"
    print(f"Generating {preset} character geometry", flush=True)
    mesh, _ = nodes.Trellis2MeshWithVoxelGenerator().process(
        pipeline,
        image,
        seed,
        preset,
        12,
        12,
        12,
        49152,
        1,
        32,
        generate_texture,
        False,
        "euler",
        False,
        1,
        "flood_fill",
        False,
    )
    if not generate_texture:
        print("Repairing decoded surface with CuMesh", flush=True)
        mesh = nodes.Trellis2FillHolesWithCuMesh().process(mesh, 1.0)[0]
        print("Reconstructing high-quality manifold surface", flush=True)
        mesh = nodes.Trellis2ReconstructMeshWithQuad().process(
            mesh,
            1.0,
            resolution,
            True,
            True,
        )[0]
        print("Closing reconstruction boundaries with MeshLib", flush=True)
        mesh = nodes.Trellis2FillHolesNicelyWithMeshlib().process(mesh)[0]

        final_mesh = nodes.Trellis2MeshWithVoxelToTrimesh().process(mesh, "90 degrees")[
            0
        ]
        _validate_mesh(final_mesh)
        print(f"Simplifying triangle mesh to {target_faces:,} faces", flush=True)
        final_mesh = _simplify_mesh(final_mesh, target_faces)
        final_mesh = _retain_dominant_component(final_mesh)
        _validate_character_mesh(final_mesh)
        print("Retained validated dominant character shell", flush=True)
        print("Exporting validated geometry-only GLB", flush=True)
        _export(final_mesh, output)
        return
    print("Exporting textured GLB", flush=True)
    # This is the wrapper's native O-Voxel PBR export, preserving the generated
    # texture latent rather than baking a legacy source texture in Blender.
    glb = nodes.Trellis2OvoxelExportToGLB().process(
        mesh, 512 if resolution <= 512 else 1024, 2048, 250000
    )[0]
    clean_glb = _retain_dominant_component(glb)
    _validate_character_mesh(clean_glb)
    print("Retained textured dominant character shell", flush=True)
    _export(clean_glb, output)


def _generate_multiview(
    nodes,
    pipeline,
    view_images: dict[str, object],
    output: Path,
    resolution: int,
    target_faces: int = 60_000,
    generate_texture: bool = True,
    seed: int = 1337,
) -> None:
    preset = "512" if resolution <= 512 else "1024_cascade"
    print(f"Multi-view conditioning: {', '.join(view_images.keys())}", flush=True)
    print(f"Generating {preset} character geometry", flush=True)
    mesh, _ = nodes.Trellis2MeshWithVoxelMultiViewGenerator().process(
        pipeline=pipeline,
        front_image=view_images["front"],
        seed=seed,
        pipeline_type=preset,
        sparse_structure_steps=12,
        sparse_structure_guidance_strength=6.50,
        sparse_structure_guidance_rescale=0.05,
        sparse_structure_rescale_t=4.00,
        shape_steps=12,
        shape_guidance_strength=6.50,
        shape_guidance_rescale=0.05,
        shape_rescale_t=4.00,
        texture_steps=12,
        texture_guidance_strength=3.00,
        texture_guidance_rescale=0.20,
        texture_rescale_t=3.00,
        max_num_tokens=49152,
        sparse_structure_resolution=32,
        generate_texture_slat=generate_texture,
        sparse_structure_guidance_interval_start=0.10,
        sparse_structure_guidance_interval_end=1.00,
        shape_guidance_interval_start=0.10,
        shape_guidance_interval_end=1.00,
        texture_guidance_interval_start=0.00,
        texture_guidance_interval_end=0.90,
        use_tiled_decoder=False,
        front_axis="z",
        # 1.0 is the ComfyUI-Trellis2 node's own widget default; we have no
        # measured reason yet to deviate from it (the pipeline function's own
        # keyword default is 2.0, but that's not what the node ships with).
        blend_temperature=1.0,
        sampler="euler",
        fill_holes=False,
        hole_iterations=1,
        verbose=False,
        dino_lock=0.00,
        dino_substeps=4,
        hole_fill_algorithm="flood_fill",
        dino_foundation_cap=1.00,
        keep_only_shell=False,
        back_image=view_images.get("back"),
        left_image=view_images.get("left"),
        right_image=view_images.get("right"),
    )
    if not generate_texture:
        print("Repairing decoded surface with CuMesh", flush=True)
        mesh = nodes.Trellis2FillHolesWithCuMesh().process(mesh, 1.0)[0]
        print("Reconstructing high-quality manifold surface", flush=True)
        mesh = nodes.Trellis2ReconstructMeshWithQuad().process(
            mesh,
            1.0,
            resolution,
            True,
            True,
        )[0]
        print("Closing reconstruction boundaries with MeshLib", flush=True)
        mesh = nodes.Trellis2FillHolesNicelyWithMeshlib().process(mesh)[0]

        final_mesh = nodes.Trellis2MeshWithVoxelToTrimesh().process(mesh, "90 degrees")[
            0
        ]
        _validate_mesh(final_mesh)
        print(f"Simplifying triangle mesh to {target_faces:,} faces", flush=True)
        final_mesh = _simplify_mesh(final_mesh, target_faces)
        final_mesh = _retain_dominant_component(final_mesh)
        _validate_character_mesh(final_mesh)
        print("Retained validated dominant character shell", flush=True)
        print("Exporting validated geometry-only GLB", flush=True)
        _export(final_mesh, output)
        return
    print("Exporting textured GLB", flush=True)
    # This is the wrapper's native O-Voxel PBR export, preserving the generated
    # texture latent rather than baking a legacy source texture in Blender.
    glb = nodes.Trellis2OvoxelExportToGLB().process(
        mesh, 512 if resolution <= 512 else 1024, 2048, 250000
    )[0]
    clean_glb = _retain_dominant_component(glb)
    _validate_character_mesh(clean_glb)
    print("Retained textured dominant character shell", flush=True)
    _export(clean_glb, output)


def _refine(
    nodes,
    pipeline,
    image,
    input_mesh: Path,
    output: Path,
    resolution: int,
    target_faces: int,
    repair_holes: bool,
    safe_reconstruction: bool = False,
    generate_texture: bool = True,
) -> None:
    print("Loading source mesh", flush=True)
    source = nodes.Trellis2LoadMesh().load(str(input_mesh))[0]
    print(f"Refining geometry at {resolution}", flush=True)
    refined, _ = nodes.Trellis2MeshRefiner().process(
        pipeline,
        source,
        image,
        random.randint(0, 0x7FFFFFFF),
        resolution,
        12,
        6.5,
        0.05,
        4.0,
        1,
        3.0,
        0.20,
        3.0,
        49152,
        False,
        32,
        0.10,
        1.0,
        0.0,
        0.90,
        False,
        1,
        "euler",
        False,
        0.0,
        4,
        1.0,
    )
    if safe_reconstruction:
        print("Preserving compatible decoded TRELLIS surface", flush=True)
        rebuilt = refined
    else:
        print("Reconstructing clean CuMesh surface", flush=True)
        rebuilt = nodes.Trellis2ReconstructMeshWithQuad().process(
            refined, 1.0, resolution, True, True
        )[0]
    print(
        "Decoded surface: "
        f"vertices={len(rebuilt.vertices)}, faces={len(rebuilt.faces)}, "
        f"face_width={rebuilt.faces.shape[-1]}, "
        f"face_range={int(rebuilt.faces.min())}..{int(rebuilt.faces.max())}",
        flush=True,
    )
    if repair_holes:
        print("Filling holes with Meshlib", flush=True)
        rebuilt = nodes.Trellis2FillHolesNicelyWithMeshlib().process(rebuilt)[0]
    final_mesh = nodes.Trellis2MeshWithVoxelToTrimesh().process(rebuilt, "90 degrees")[
        0
    ]
    _validate_mesh(final_mesh)
    print(f"Simplifying triangle mesh to {target_faces:,} faces", flush=True)
    final_mesh = _simplify_mesh(final_mesh, target_faces)
    _validate_mesh(final_mesh)
    if not generate_texture:
        print("Exporting validated geometry-only GLB", flush=True)
        _export(final_mesh, output)
        return
    textured = _texture(nodes, pipeline, image, final_mesh, resolution)
    _export(textured, output)


def _device_report() -> dict:
    """Ground-truth device report for this worker process, for the job log
    and for the ``metadata["device"]`` stamped on the output artifact."""
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {
                "device": "cuda",
                "free_gb": round(free_bytes / (1024**3), 2),
                "total_gb": round(total_bytes / (1024**3), 2),
            }
    except Exception:
        pass
    return {"device": "cpu", "reason": "CUDA not available in this worker's torch build"}


def _log_device_line(report: dict) -> None:
    if report["device"] == "cuda":
        print(f"device: cuda ({report['free_gb']}/{report['total_gb']} GB free)", flush=True)
    else:
        print(f"device: cpu ({report.get('reason', 'no CUDA GPU detected')})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    job = json.loads(parser.parse_args().spec)
    runtime = Path(job["runtime"]).resolve()
    output = Path(job["output_glb"]).resolve()
    device_report = _device_report()
    _log_device_line(device_report)
    if sys.platform != "win32":
        _validate_native_contract(runtime)
    nodes = _load_nodes(runtime)
    pipeline = _load_pipeline(nodes, bool(job.get("gpu_first", False)))
    view_images_spec = job.get("view_images")
    is_multiview = (
        job["operation"] == "generate"
        and view_images_spec
        and len(view_images_spec) > 1
    )
    try:
        if is_multiview:
            if not hasattr(nodes, "Trellis2MeshWithVoxelMultiViewGenerator"):
                raise RuntimeError(
                    "TRELLIS.2 multi-view conditioning requires a wrapper build that "
                    "exposes Trellis2MeshWithVoxelMultiViewGenerator; reinstall/upgrade "
                    "the refiner runtime."
                )
            view_images = _multiview_tensors(
                nodes,
                {label: Path(path) for label, path in view_images_spec.items()},
            )
            _generate_multiview(
                nodes,
                pipeline,
                view_images,
                output,
                int(job["resolution"]),
                int(job.get("target_faces", 60_000)),
                bool(job.get("generate_texture", True)),
                int(job.get("seed", 1337)),
            )
        else:
            image = _image_tensor(nodes, Path(job["source_image"]))
            if job["operation"] == "generate":
                _generate(
                    nodes,
                    pipeline,
                    image,
                    output,
                    int(job["resolution"]),
                    int(job.get("target_faces", 60_000)),
                    bool(job.get("generate_texture", True)),
                    int(job.get("seed", 1337)),
                )
            elif job["operation"] == "refine":
                _refine(
                    nodes,
                    pipeline,
                    image,
                    Path(job["input_glb"]),
                    output,
                    int(job["resolution"]),
                    int(job["target_faces"]),
                    bool(job.get("repair_holes")),
                    bool(job.get("safe_reconstruction", False)),
                    bool(job.get("generate_texture", True)),
                )
            else:
                raise ValueError(f"Unknown TRELLIS.2 operation: {job['operation']}")
    finally:
        try:
            import torch

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"DEVICE_REPORT_JSON:{json.dumps(device_report)}", flush=True)
    print("TRELLIS.2 job complete", flush=True)
    return 0


def _fp8_compute_dtype_name(module, conv_backend: str) -> str:
    sparse_types = {"SparseConv3d", "SparseInverseConv3d"}
    if conv_backend == "spconv" and module.__class__.__name__ in sparse_types:
        return "float16"
    return "bfloat16"


def _patch_fp8_compute(conv_backend: str) -> None:
    import torch
    from aismith_trellis2.trellis2.modules import utils

    def patch_fp8_forward(module) -> None:
        if hasattr(module, "_original_forward_fp8"):
            return
        module._original_forward_fp8 = module.forward

        def fp8_forward(*args, **kwargs):
            dtype = getattr(torch, _fp8_compute_dtype_name(module, conv_backend))
            with torch.no_grad():
                original_weight = module.weight.data
                module.weight.data = original_weight.to(dtype)
                has_bias = getattr(module, "bias", None) is not None
                if has_bias:
                    original_bias = module.bias.data
                    module.bias.data = original_bias.to(dtype)
            try:
                return module._original_forward_fp8(*args, **kwargs)
            finally:
                with torch.no_grad():
                    module.weight.data = original_weight
                    if has_bias:
                        module.bias.data = original_bias

        module.forward = fp8_forward

    utils.patch_fp8_forward = patch_fp8_forward


if __name__ == "__main__":
    raise SystemExit(main())
