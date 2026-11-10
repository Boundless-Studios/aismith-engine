"""Small, headless Blender boundary used by the local studio.

This file deliberately contains no UI code.  It is invoked as a short lived
process so Blender's allocator and image cache disappear between pipeline
stages.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Quaternion
import numpy as np
from mathutils import Matrix, Vector, bvhtree, geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.retarget import (
    SOMA_TO_ROLE,
    resolve_motion_mapping,
    select_character_armature,
    select_deepest_source_per_target,
    select_motion_translation_bones,
)
from backend.app import clip_store
from backend.app import retarget_math
from backend.app import slither_math
from backend.app.skeleton_naming import _CHAIN_SPINE_NAME_PATTERN
from backend.app.skin_weights import combine_weight_maps, smooth_weight_maps
from backend.app.texture_bake import (
    BAKE_TEXTURE_SIZE_MINIMUM,
    NORMAL_MAP_COLORSPACE,
    NORMAL_MAP_FLAT_RGB,
    PALETTE_MIN_SUBJECT_PIXELS,
    bake_image_name,
    bake_texture_size,
    estimate_background_color,
    fill_transparent_texels,
    linear_to_srgb,
    mark_normal_bake_coverage,
    normal_bake_report,
    palette_distance,
    palette_statistics,
    palette_transfer_report,
    renormalize_tangent_normals,
    srgb_to_linear,
    subject_pixel_mask,
    transfer_palette,
    unique_by_identity,
    uv_pack_margin,
)
from backend.app.deformation_review import (
    MAX_DEFORMATION_REVIEW_FRAMES,
    all_animation_frames,
    resolve_review_focus_bone,
    select_review_frame,
)
from backend.app.geometry_contract import (
    extent_profile,
    scale_uniformity,
    uv_unwrap_stats,
)
from backend.app.motion_metrics import compute_motion_metrics
from backend.app.proxy_rig import (
    DEFORMATION_PROXY_BAKE_RAY_MULTIPLIER,
    chain_endpoints_from_centroids,
    deformation_proxy_decimation_ratio,
    descendant_names,
    face_render_signature,
    blend_local_finger_weights,
    add_local_finger_coverage,
    is_finger_deform_bone,
    is_proxy_deform_bone,
    largest_face_component,
    resolve_limb_chains,
    resolve_right_hand_bone,
    select_action_name,
    select_deformation_surface,
    sword_component_offsets,
    unique_position_face_indices,
    weighted_centroid,
)
from backend.app.voxel_remesh import (
    canonicalize_decimate_ratio,
    initial_voxel_size,
    next_voxel_size,
    voxel_candidate_preferred,
)
from backend.app import view_projection
from backend.app.source_fidelity import SourceCamera, fit_source_camera


def log(message: str) -> None:
    print(json.dumps({"event": "log", "message": message}), flush=True)


def _configure_render_device(scene, *, prefer: str = "auto") -> str:
    """Enable Cycles and select the best available compute device.

    Tries OptiX, then CUDA, then falls back to CPU Cycles (never EEVEE --
    Cycles background rendering has no OpenGL/EGL dependency, unlike
    ``BLENDER_EEVEE_NEXT``, which fails headless on this WSL2 box with
    ``EGL_BAD_MATCH`` and silently renders on CPU). ``prefer`` may force a
    specific device ("OPTIX", "CUDA", "CPU") or leave the choice to
    "auto" (the default). Returns the device kind actually selected.
    """
    scene.render.engine = "CYCLES"
    prefer = (prefer or "auto").strip().upper()

    try:
        cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        scene.cycles.device = "CPU"
        log("device: cpu (Cycles, reason: cycles addon preferences unavailable)")
        return "CPU"

    def _devices_for(kind: str):
        # macOS Cycles only exposes NONE/METAL as compute_device_type enum
        # values -- OPTIX/CUDA aren't members there at all, unlike Windows/
        # Linux where they're valid enum values that may just list 0 devices.
        try:
            cycles_preferences.compute_device_type = kind
        except TypeError:
            return []
        cycles_preferences.get_devices()
        return [device for device in cycles_preferences.devices if device.type == kind]

    chosen_kind = None
    chosen_devices: list = []
    if prefer != "CPU":
        for kind in (("OPTIX", "CUDA") if prefer == "AUTO" else (prefer,)):
            if kind not in ("OPTIX", "CUDA"):
                continue
            devices = _devices_for(kind)
            if devices:
                chosen_kind = kind
                chosen_devices = devices
                break

    if chosen_kind is None:
        cycles_preferences.compute_device_type = "NONE"
        scene.cycles.device = "CPU"
        reason = "AISMITH_BLENDER_DEVICE=CPU" if prefer == "CPU" else "no CUDA/OptiX device detected"
        log(f"device: cpu (Cycles, reason: {reason})")
        return "CPU"

    cycles_preferences.compute_device_type = chosen_kind
    cycles_preferences.get_devices()
    names = []
    for device in cycles_preferences.devices:
        if device.type == chosen_kind:
            device.use = True
            names.append(device.name)
        elif device.type == "CPU":
            device.use = False
    scene.cycles.device = "GPU"
    log(f"device: gpu ({chosen_kind}, Cycles, devices: {', '.join(names) or 'unnamed'})")
    return chosen_kind


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_model(path: str, *, bone_heuristic: str | None = None) -> None:
    """Import `path` into the current scene.

    `bone_heuristic` is forwarded to the glTF importer only when given. The
    default (None) keeps the importer's own default ("BLENDER") so every rig
    op round-trips joint orientations unchanged: `apply_clip` imports the
    clip store with the importer default and assigns pose-bone rotations
    that are relative to each bone's rest frame, so the target rig must be
    imported the same way, and re-exporting ops (repair_weights,
    transfer_rig, ...) must not silently re-orient bones. Only the
    render-only `render_skeleton_sheet` opts into "TEMPERANCE".
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        if bone_heuristic is None:
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            # Importer builds without the requested enum value fall back to
            # the default; callers that care (`render_skeleton_sheet`)
            # filter the consequence back out via `skinned_body_meshes`.
            try:
                bpy.ops.import_scene.gltf(filepath=path, bone_heuristic=bone_heuristic)
            except TypeError:
                bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise RuntimeError(f"Blender cannot import {suffix} at this boundary")


def selected_meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def skinned_body_meshes(armature) -> list:
    """The actual character mesh(es) among the scene's MESH objects --
    never the glTF importer's own scaffolding.

    Blender's glTF importer can attach a custom display shape (a unit
    Icosphere, by default) to an armature's bones purely for viewport
    readability; `render_skeleton_sheet` asks it not to (it imports with
    `bone_heuristic="TEMPERANCE"`), but that option does not exist on every
    importer build, so this filter is the part that must not be skipped. A shape object carries no skin data and
    is not part of the character, yet `selected_meshes()`-style "every MESH
    in the scene" queries pick it up anyway -- its +/-1 unit bounds
    silently dominate a real, ~1m-tall character's bounding box.

    A mesh counts as "body" when it is actually SKINNED to `armature` (an
    ARMATURE modifier or non-empty vertex groups) and is not referenced as
    any of the armature's pose bones' `custom_shape`. When that leaves
    nothing -- a rig with no modifier wiring visible yet at this point in
    the pipeline -- the densest mesh in the scene is the least-bad
    fallback: the real body is essentially always the mesh with the most
    vertices, and importer scaffolding never is.
    """
    custom_shapes = {
        bone.custom_shape for bone in armature.pose.bones if bone.custom_shape is not None
    }
    meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj not in custom_shapes
    ]
    skinned = [
        obj for obj in meshes
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
        or len(obj.vertex_groups) > 0
    ]
    if skinned:
        return skinned
    if not meshes:
        return []
    return [max(meshes, key=lambda obj: len(obj.data.vertices))]


def export_glb(path: str, *, animations: bool = True, y_up: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = next(iter(selected_meshes()), None)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_animations=animations,
        export_skins=True,
        export_yup=y_up,
    )


def convert(input_path: str, output_path: str) -> None:
    clear_scene()
    import_model(input_path)
    export_glb(output_path, animations=False)


def export_obj(input_path: str, output_path: str) -> None:
    """Export source geometry unchanged at the GLB/OBJ boundary."""
    clear_scene()
    import_model(input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_materials=False,
        export_normals=True,
        export_uv=True,
        export_triangulated_mesh=True,
    )


def _join_meshes(meshes):
    meshes = [obj for obj in meshes if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh was imported")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _ensure_uv(obj, *, angle_limit: float | None = None, texture_size: int = 2048) -> dict:
    """Smart-project the target and pack the islands tightly into the atlas.

    Packing matters as much as the atlas resolution does. `smart_project`'s
    own `island_margin` is a fraction of the whole UV square applied around
    EVERY island, so on a voxel-remeshed canonical mesh -- which smart-project
    shatters into ~2,000 shards -- the old 0.015 margin spent 93% of the atlas
    on gutters: 6.8% utilisation, ~645 texels per world-unit against the S1
    source's ~1,146 (measured 2026-09-01 on the alien, and the whole reason
    the canonical face read as blurry). Unwrap with no island margin,
    equalise island scale so no region is starved, then let Blender 4.x's
    concave packer place them with a gutter measured in texels.

    Returns the resulting atlas statistics (`uv_unwrap_stats`-shaped, whose
    `uv_total_area` is the packed atlas utilisation in [0, 1]).
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    settings = {"island_margin": 0.0}
    if angle_limit is not None:
        settings["angle_limit"] = angle_limit
    bpy.ops.uv.smart_project(**settings)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(
        rotate=True,
        margin=uv_pack_margin(texture_size),
        shape_method="CONCAVE",
        scale=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    stats = uv_unwrap_stats(_mesh_uv_triangles(obj))
    log(
        f"UV atlas: {stats['uv_faces']} faces, utilisation="
        f"{stats['uv_total_area'] * 100:.2f}% of a {texture_size}px atlas "
        f"(gutter={uv_pack_margin(texture_size):.5f})"
    )
    return stats


def _source_texture_dimensions(source_meshes) -> list[int]:
    """Every width/height of every image the source materials sample."""
    dimensions: list[int] = []
    materials = unique_by_identity([
        material
        for obj in source_meshes
        if obj.type == "MESH"
        for material in obj.data.materials
        if material is not None
    ])
    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            width, height = node.image.size[0], node.image.size[1]
            dimensions.extend(value for value in (width, height) if value > 0)
    return dimensions


def _mesh_ray_distance(meshes) -> float:
    points = [obj.matrix_world @ Vector(corner) for obj in meshes if obj.type == "MESH" for corner in obj.bound_box]
    if not points:
        return 0.01
    lo = [min(point[i] for point in points) for i in range(3)]
    hi = [max(point[i] for point in points) for i in range(3)]
    diagonal = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    return max(0.001, diagonal * 0.025)


def _select_for_bake(source_meshes, target) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in source_meshes:
        obj.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target


def _bake_emit_channel(source_meshes, target, image, input_name: str, distance: float) -> None:
    """Bake a Principled input through emission without changing source files."""
    saved = []
    materials = unique_by_identity([
        material
        for obj in source_meshes
        for material in obj.data.materials
        if material is not None
    ])
    for material in materials:
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        source_input = principled.inputs.get(input_name) if principled else None
        if output is None or source_input is None:
            continue
        original = [(link.from_socket, link.to_socket) for link in list(output.inputs["Surface"].links)]
        for link in list(output.inputs["Surface"].links):
            links.remove(link)
        emission = nodes.new("ShaderNodeEmission")
        emission.name = "AISmith temporary bake emission"
        if source_input.is_linked:
            links.new(source_input.links[0].from_socket, emission.inputs["Color"])
        else:
            value = source_input.default_value
            if isinstance(value, float):
                emission.inputs["Color"].default_value = (value, value, value, 1.0)
            else:
                emission.inputs["Color"].default_value = value
        links.new(emission.outputs[0], output.inputs["Surface"])
        saved.append((nodes, links, output, original, emission))
    try:
        _select_for_bake(source_meshes, target)
        bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=distance, max_ray_distance=distance, margin=32)
    finally:
        for nodes, links, output, original, emission in saved:
            for link in list(output.inputs["Surface"].links):
                links.remove(link)
            for from_socket, to_socket in original:
                links.new(from_socket, to_socket)
            nodes.remove(emission)


def _activate_image(material, image):
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    for item in material.node_tree.nodes:
        item.select = False
    node.select = True
    material.node_tree.nodes.active = node
    return node


def _fill_bake_misses(image, *, max_passes: int = 64) -> float:
    """Dilate covered texels into bake misses; return the miss fraction.

    The miss fraction is measured over the WHOLE atlas (covered islands plus
    gutter), so it is only meaningful next to the atlas utilisation recorded
    by `_ensure_uv`: misses inside islands are real bake failures, misses in
    the gutter are the seam-bleed budget doing its job.
    """
    pixels = _read_image_pixels(image)
    miss_fraction = float(np.mean(pixels[:, :, 3] < 0.5))
    filled = fill_transparent_texels(pixels, max_passes=max_passes)
    image.pixels.foreach_set(filled.reshape(-1))
    image.update()
    return miss_fraction


def _read_image_pixels(image) -> np.ndarray:
    """The image's RGBA buffer as an H x W x 4 float array."""
    values = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    return values.reshape((image.size[1], image.size[0], 4))


def _load_reference_pixels(path: str) -> np.ndarray:
    """Load the palette reference image as an H x W x 4 sRGB-ENCODED RGBA array.

    Read through Blender's own image loader rather than PIL: the bake runs
    inside Blender's bundled interpreter, which ships numpy but not PIL.

    sRGB-ENCODED, not linear, and that is the trap this docstring exists for.
    `image.pixels` is documented as scene-linear, but for a BYTE-buffer image
    -- which is what an 8-bit PNG loads as, and what `bpy.data.images.new`
    allocates for the bake atlas -- it hands back the raw buffer with no
    colour management applied. Same fact `mark_normal_bake_coverage` records
    from the other side ("a generated_color of 0.5 comes back as 127/255").
    Measured 2026-09-01 on the reference image below: treating its pixels as
    linear gives colorfulness 3.76, decoding them as sRGB gives 4.74 -- and
    4.74 is what PIL reads off the same file.

    Getting this wrong does not fail loudly; it silently mis-scales the
    palette match. The first run measured the reference's chroma spread as
    1.7x its true value and left the invented pink only half corrected.

    The buffer is bottom-up relative to the file, which is irrelevant here:
    the border ring the background is keyed from is symmetric under a vertical
    flip, and every other operation is per-pixel.
    """
    image = bpy.data.images.load(path, check_existing=False)
    try:
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise RuntimeError(f"Palette reference image has no pixels: {path}")
        return _read_image_pixels(image)
    finally:
        bpy.data.images.remove(image, do_unlink=True)


def _apply_palette_transfer(image, reference_path: str, strength: float) -> dict:
    """Measure -- and at `strength > 0` correct -- the baked albedo's palette.

    Runs BEFORE the bake-miss dilation, deliberately. The coverage mask is
    `alpha >= 0.5`, and `fill_transparent_texels` ends by forcing alpha to 1.0
    across the whole atlas -- so after dilation the mask no longer exists and
    the neutral fallback grey in the unreachable gutter would be measured as
    if it were baked surface. Transferring first also means the dilation
    bleeds the CORRECTED colour into the seams, which is what a glTF consumer
    samples across a UV border.

    MEASUREMENT ALWAYS HAPPENS when a reference is available, even at
    `strength == 0`. The palette distance is measured HERE, on the atlas,
    rather than downstream on the similarity renders, because the renders are
    lit: measured 2026-09-01, the alien's conditioning image scores 19.1 for
    colorfulness while a matched render of a texture that reproduces it
    perfectly scores 10.1 front and 3.6 back, purely because the render is
    dim. An albedo atlas has no lighting, so atlas-against-image is the only
    place the comparison means what it says.

    Returns the provenance dict `native.run_canonicalize` records; an
    unusable reference is reported as `applied: false` with a reason rather
    than raising, because a palette that could not be matched must never cost
    the operator the artifact.
    """
    pixels = _read_image_pixels(image)
    covered = pixels[:, :, 3] >= 0.5
    if not covered.any():
        return palette_transfer_report(
            applied=False, reason="the base-colour bake covered no texels", strength=strength
        )
    reference = _load_reference_pixels(reference_path)
    reference_rgb = reference[:, :, :3].astype(np.float64)
    reference_mask = subject_pixel_mask(reference_rgb, reference[:, :, 3].astype(np.float64))
    if reference_mask is None:
        return palette_transfer_report(
            applied=False,
            reason=(
                "the reference image has no keyable subject (non-uniform background); "
                "palette statistics taken over a backdrop would be worse than none"
            ),
            strength=strength,
        )
    if int(reference_mask.sum()) < PALETTE_MIN_SUBJECT_PIXELS:
        return palette_transfer_report(
            applied=False,
            reason=f"the reference image keyed only {int(reference_mask.sum())} subject pixel(s)",
            strength=strength,
        )
    # Both buffers are sRGB-encoded bytes (see `_load_reference_pixels`), so
    # both are decoded into linear before any Oklab arithmetic and the result
    # is re-encoded on the way back into the atlas. Oklab is defined on LINEAR
    # sRGB; running it on encoded values distorts the a/b axes and mis-scales
    # every statistic taken in them.
    albedo_srgb = pixels[:, :, :3].astype(np.float64)
    albedo = srgb_to_linear(albedo_srgb)
    reference_linear = srgb_to_linear(reference_rgb)
    before = palette_statistics(albedo, covered)
    reference_stats = palette_statistics(reference_linear, reference_mask)
    distance_before = palette_distance(
        reference_rgb, albedo_srgb,
        reference_mask=reference_mask, candidate_mask=covered,
    )
    transferred = transfer_palette(
        albedo, reference_linear, reference_mask, strength, albedo_mask=covered,
    )
    transferred_srgb = linear_to_srgb(transferred)
    after = palette_statistics(transferred, covered)
    distance_after = palette_distance(
        reference_rgb, transferred_srgb,
        reference_mask=reference_mask, candidate_mask=covered,
    )
    if strength > 0.0:
        pixels[:, :, :3] = transferred_srgb.astype(np.float32)
        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()
    log(
        f"Palette {'transfer' if strength > 0.0 else 'measurement only'}: albedo colorfulness "
        f"{before['colorfulness']:.2f} -> {after['colorfulness']:.2f} against reference "
        f"{reference_stats['colorfulness']:.2f}; distance "
        f"{distance_before['distance']:.4f} -> {distance_after['distance']:.4f} "
        f"(strength {strength:.2f}, {int(covered.sum())} covered texels)"
    )
    return palette_transfer_report(
        applied=strength > 0.0,
        reason=None if strength > 0.0 else "measured only; palette transfer strength is 0",
        reference_id=Path(reference_path).stem,
        strength=strength,
        before=before,
        after=after,
        reference=reference_stats,
        distance_before=distance_before,
        distance_after=distance_after,
    )


def _fill_normal_bake_misses(image, before, *, max_passes: int = 64) -> float:
    """Dilate baked normals into normal-bake misses; return the miss fraction.

    Same job as `_fill_bake_misses`, but coverage is a diff against `before`
    (the atlas as it stood before the bake -- see
    `texture_bake.mark_normal_bake_coverage`, since the NORMAL pass's alpha is
    not a coverage channel) and the dilated result is re-normalised, because
    averaging two unit normals produces a short one that shades as a tilt.
    """
    pixels = mark_normal_bake_coverage(_read_image_pixels(image), before)
    miss_fraction = float(np.mean(pixels[:, :, 3] < 0.5))
    filled = renormalize_tangent_normals(
        fill_transparent_texels(pixels, max_passes=max_passes)
    )
    image.pixels.foreach_set(filled.reshape(-1))
    image.update()
    return miss_fraction


def _shade_target_smooth(target) -> str:
    """Shade the bake target smooth and report which shading it now carries.

    A tangent-space normal bake encodes the source's surface normal AS A
    DELTA from the target's own shading normal, so the target's shading is
    the bake's reference frame. The canonical mesh arrives flat-shaded --
    `voxel_remesh_canonicalize` deliberately exports its OBJ with
    `export_normals=False` (a trimesh unwelded-components trap), and Blender's
    OBJ import then leaves every polygon faceted. Baking against that frame
    spends the normal map's precision cancelling ~30,000 facet discontinuities
    instead of carrying the source's nose and mouth, and leaves visible
    faceting anywhere the bake missed. Smooth shading gives a continuous
    reference frame, which is also the frame a glTF consumer reconstructs
    tangents against.
    """
    for polygon in target.data.polygons:
        polygon.use_smooth = True
    target.data.update()
    return "smooth"


def _assign_neutral_retopology_material(target) -> None:
    target.data.materials.clear()
    material = bpy.data.materials.new("AISmith Neutral Retopology Material")
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65
    bsdf.inputs["Metallic"].default_value = 0.0
    target.data.materials.append(material)
    for polygon in target.data.polygons:
        polygon.use_smooth = True


def _bake_retopology_y_up(target) -> None:
    """Bake glTF's Y-up coordinates into vertices for transform-ignorant riggers."""
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    target.data.transform(Matrix.Rotation(math.radians(-90.0), 4, "X"))
    target.data.update()


def retopo_bake(
    source_path: str,
    target_path: str,
    output_path: str,
    bake_appearance: bool = True,
    report_path: str | None = None,
    palette_reference: str | None = None,
    palette_strength: float = 1.0,
    atlas_minimum: int | None = None,
) -> None:
    """Bake source PBR appearance to a newly unwrapped Instant Meshes mesh."""
    clear_scene()
    import_model(source_path)
    source_meshes = list(selected_meshes())
    if not source_meshes:
        raise RuntimeError("The source GLB has no mesh to bake")
    import_model(target_path)
    target_meshes = [obj for obj in selected_meshes() if obj not in source_meshes]
    target = _join_meshes(target_meshes)
    target.name = "AISmith_Retopologized"
    if bake_appearance:
        bake_report = _bake_source_appearance(
            source_meshes,
            target,
            palette_reference=palette_reference,
            palette_strength=palette_strength,
            atlas_minimum=atlas_minimum,
        )
    else:
        log("Source has no material; assigning neutral smooth retopology material")
        _assign_neutral_retopology_material(target)
        bake_report = {
            "schema_version": 2,
            "baked_appearance": False,
            "normal_bake": None,
            # A neutral-material target has no baked albedo to re-statistic.
            "palette_transfer": palette_transfer_report(
                applied=False, reason="the source has no material to bake"
            ),
        }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(bake_report), encoding="utf-8")

    for obj in source_meshes:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    _bake_retopology_y_up(target)
    export_glb(output_path, animations=False, y_up=False)


def _bake_source_appearance(
    source_meshes,
    target,
    *,
    ray_distance_multiplier: float = 1.0,
    uv_angle_limit: float | None = None,
    palette_reference: str | None = None,
    palette_strength: float = 1.0,
    atlas_minimum: int | None = None,
) -> dict:
    """Bake source PBR channels onto one newly generated visible surface.

    Four channels: base colour, roughness and metallic (appearance, baked
    through emission), plus a tangent-space NORMAL map (geometry). The three
    appearance channels cannot recover what voxel remesh + decimation
    destroys, because what they destroy is shape -- the normal bake is the
    channel that carries the source's nose, mouth and surface relief onto a
    30k-triangle target as shading.

    The atlas resolution follows the SOURCE's largest texture (capped at 4K by
    `texture_bake.bake_texture_size`) instead of a hardcoded 2048, and the
    seam-bleed margin scales with it, so a 4K generate result keeps its detail
    and a 1K one is not padded with resampled mush. `atlas_minimum` raises the
    FLOOR for a dense quality tier (bead i63ky) without ever capping what the
    source supports. Returns a provenance report (`native.run_canonicalize`
    records it on the artifact) so a future fidelity regression is a number in
    the metadata, not an eyeball call.
    """
    source_dimensions = _source_texture_dimensions(source_meshes)
    minimum = max(BAKE_TEXTURE_SIZE_MINIMUM, int(atlas_minimum or 0))
    size = bake_texture_size(source_dimensions, minimum=minimum)
    largest_source = max(source_dimensions) if source_dimensions else 0
    log(
        f"Bake atlas: {size}px (largest source texture: {largest_source or 'none'}px, "
        f"tier floor {minimum}px)"
    )
    target_shading = _shade_target_smooth(target)
    uv_stats = _ensure_uv(target, angle_limit=uv_angle_limit, texture_size=size)

    target.data.materials.clear()
    material = bpy.data.materials.new("AISmith Retopologized Material")
    material.use_nodes = True
    target.data.materials.append(material)
    base = bpy.data.images.new(bake_image_name("BaseColor"), width=size, height=size, alpha=True)
    roughness = bpy.data.images.new(bake_image_name("Roughness"), width=size, height=size, alpha=False)
    metallic = bpy.data.images.new(bake_image_name("Metallic"), width=size, height=size, alpha=False)
    normal = bpy.data.images.new(bake_image_name("Normal"), width=size, height=size, alpha=False)
    base.generated_color = (0.18, 0.18, 0.18, 0.0)
    roughness.generated_color = (0.5, 0.5, 0.5, 1.0)
    metallic.generated_color = (0.0, 0.0, 0.0, 1.0)
    normal.generated_color = (*NORMAL_MAP_FLAT_RGB, 1.0)
    for image in (roughness, metallic, normal):
        image.colorspace_settings.name = NORMAL_MAP_COLORSPACE
    distance = _mesh_ray_distance(source_meshes) * ray_distance_multiplier
    _configure_render_device(bpy.context.scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    bpy.context.scene.cycles.samples = 16
    # Seam bleed in texels, scaled with the atlas so a 4K bake is not left
    # with a 2K-sized skirt. ADJACENT_FACES keeps the bleed following real
    # 3D adjacency rather than flooding a neighbouring island's gutter.
    bake_margin = max(8, size // 64)
    bake_settings = bpy.context.scene.render.bake
    bake_settings.margin = bake_margin
    bake_settings.margin_type = "ADJACENT_FACES"
    bake_settings.use_clear = False
    # TANGENT, not OBJECT: an object-space map is only valid while the mesh
    # never deforms, and this mesh exists to be rigged and animated. The
    # POS_X/POS_Y/POS_Z swizzle is the OpenGL convention glTF 2.0 mandates
    # (green points UP); it is Blender's default, set explicitly because a
    # DirectX-convention green channel is invisible in a still and inverts
    # every lighting cue the moment the model turns.
    bake_settings.normal_space = "TANGENT"
    bake_settings.normal_r = "POS_X"
    bake_settings.normal_g = "POS_Y"
    bake_settings.normal_b = "POS_Z"

    base_miss_fraction = None
    palette_report: dict = palette_transfer_report(
        applied=False, reason="no palette reference was supplied"
    )
    channel_nodes: dict[str, object] = {}
    for image, channel in ((base, "Base Color"), (roughness, "Roughness"), (metallic, "Metallic")):
        channel_nodes[channel] = _activate_image(material, image)
        log(f"Baking {channel} texture")
        _bake_emit_channel(source_meshes, target, image, channel, distance)
        if image == base:
            if palette_reference:
                log("Transferring the conditioning image's palette onto the baked albedo")
                palette_report = _apply_palette_transfer(
                    image, palette_reference, palette_strength
                )
                if not palette_report["applied"]:
                    log(f"Palette transfer skipped: {palette_report['reason']}")
            log("Filling uncovered base-color bake texels")
            # The dilation only ever writes into texels the bake MISSED, so it
            # cannot blur a covered texel; bounding it to just past the
            # seam-bleed skirt keeps it from crawling across the whole gutter.
            base_miss_fraction = _fill_bake_misses(image, max_passes=bake_margin + 8)
            log(f"Base-color bake misses before fill: {base_miss_fraction * 100:.2f}% of the atlas")
        image.pack()

    # The geometric channel. Voxel remesh + decimation is what costs the
    # canonical mesh its detail (the alien's 0.684 S1/S2 detail ratio with an
    # albedo-only bake was diagnosed as geometric, not textural): 226k source
    # triangles collapse to 30k, and the nose and mouth go with them. Baking
    # the SOURCE's surface normals into the target's tangent space puts that
    # detail back as shading, through the same selected-to-active cage and ray
    # budget the appearance channels already use.
    normal_node = _activate_image(material, normal)
    log("Baking tangent-space normal texture")
    normal_before = _read_image_pixels(normal)
    _select_for_bake(source_meshes, target)
    bpy.ops.object.bake(
        type="NORMAL",
        use_selected_to_active=True,
        cage_extrusion=distance,
        max_ray_distance=distance,
        margin=bake_margin,
    )
    normal_miss_fraction = _fill_normal_bake_misses(
        normal, normal_before, max_passes=bake_margin + 8
    )
    log(f"Normal bake misses before fill: {normal_miss_fraction * 100:.2f}% of the atlas")
    normal.pack()

    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    links.new(channel_nodes["Base Color"].outputs["Color"], bsdf.inputs["Base Color"])
    links.new(channel_nodes["Roughness"].outputs["Color"], bsdf.inputs["Roughness"])
    links.new(channel_nodes["Metallic"].outputs["Color"], bsdf.inputs["Metallic"])
    # Image Texture -> Normal Map (tangent) -> Principled "Normal" is the exact
    # node shape Blender's glTF exporter recognises as a `normalTexture`;
    # anything else (a direct link, or a Bump node) exports as no normal map
    # at all, silently.
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.space = "TANGENT"
    normal_map.inputs["Strength"].default_value = 1.0
    links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return {
        "schema_version": 2,
        "baked_appearance": True,
        "normal_bake": normal_bake_report(
            texture_size=size,
            miss_fraction=normal_miss_fraction,
            target_shading=target_shading,
        ),
        "texture_size": size,
        "texture_size_minimum": minimum,
        "source_texture_max": largest_source,
        "bake_margin_texels": bake_margin,
        "uv_pack_margin": round(uv_pack_margin(size), 6),
        "uv_atlas_utilization": uv_stats["uv_total_area"],
        "uv_faces": uv_stats["uv_faces"],
        "uv_unwrapped": uv_stats["uv_unwrapped"],
        "base_color_miss_fraction": (
            round(base_miss_fraction, 6) if base_miss_fraction is not None else None
        ),
        # Whether the baked albedo was re-statisticked onto the conditioning
        # image's palette, and the colorfulness on both sides -- the record
        # that says why a mono source produced a mono texture.
        "palette_transfer": palette_report,
        "ray_distance": round(distance, 6),
        "cycles_samples": 16,
    }


def normalize_floor(input_path: str, output_path: str, report_path: str) -> None:
    """Translate a mesh so its lowest point sits at z=0 (S2 floor contact).

    Blender's glTF importer always converts an incoming Y-up glTF file back
    to Blender's native Z-up scene convention, regardless of how the file
    was produced, so `_world_bounds` here reads the same "up" axis the S2
    contract means by z=0 -- consistent with every other axis-preserving
    import/export round trip in this worker.
    """
    clear_scene()
    import_model(input_path)
    meshes = list(selected_meshes())
    if not meshes:
        raise RuntimeError("Floor normalization requires a visible mesh")
    low, _high = _world_bounds(meshes)
    offset_z = -low.z
    if abs(offset_z) > 1e-9:
        for obj in meshes:
            obj.location = obj.location + Vector((0.0, 0.0, offset_z))
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    contract = _measure_geometry_contract(meshes)
    export_glb(output_path, animations=False, y_up=True)
    Path(report_path).write_text(
        json.dumps({"floor_offset_z": offset_z, "geometry_contract": contract}),
        encoding="utf-8",
    )
    log(
        f"Floor-normalized (offset_z={offset_z:.6f}); "
        f"uv_unwrapped={contract['uv_unwrapped']} "
        f"uniform_scale_ok={contract['uniform_scale_ok']}"
    )


def _mesh_uv_triangles(mesh_object) -> np.ndarray:
    """Per-triangle UV coordinates, (n_triangles, 3, 2), or an empty array."""
    mesh = mesh_object.data
    layer = mesh.uv_layers.active
    if layer is None or not len(mesh.loops):
        return np.empty((0, 3, 2), dtype=np.float64)
    mesh.calc_loop_triangles()
    if not len(mesh.loop_triangles):
        return np.empty((0, 3, 2), dtype=np.float64)
    loop_uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", loop_uv)
    triangle_loops = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", triangle_loops)
    return loop_uv.reshape(-1, 2)[triangle_loops.reshape(-1, 3)].astype(np.float64)


def _measure_geometry_contract(meshes) -> dict:
    """Record the three cheap S2 contract facts the pipeline only asserted.

    Measured here, in the last worker step that touches the final canonical
    GLB, so the numbers describe exactly the artifact that ships. `up_axis` is
    "Z" by construction (Blender's glTF importer converts any incoming Y-up
    file to the scene's native Z-up), so it is recorded as an observation of
    the import convention rather than a discovered fact -- the load-bearing
    axis evidence is `extent_profile`, which `native._glb_axis_profile_matches`
    compares against the source to catch a Y/Z swap.
    """
    uv_triangles = [_mesh_uv_triangles(obj) for obj in meshes if obj.type == "MESH"]
    stacked = (
        np.concatenate([item for item in uv_triangles if len(item)], axis=0)
        if any(len(item) for item in uv_triangles)
        else np.empty((0, 3, 2), dtype=np.float64)
    )
    uv = uv_unwrap_stats(stacked)
    scale = scale_uniformity([tuple(obj.scale) for obj in meshes])
    low, high = _world_bounds(meshes)
    return {
        "schema_version": 1,
        **uv,
        **scale,
        "uv_layers_present": sum(
            1 for obj in meshes if obj.type == "MESH" and len(obj.data.uv_layers)
        ),
        "meshes": len(meshes),
        "up_axis": "Z",
        "up_axis_source": "blender-gltf-import-convention",
        "extent_profile": extent_profile((high.x - low.x, high.y - low.y, high.z - low.z)),
    }


# Below this raw-face floor a voxel-remesh candidate has lost too much
# geometry to be a usable base for decimation (degenerate voxel size, or a
# near-planar / tiny source mesh).
VOXEL_REMESH_MIN_RAW_FACES = 1_000
# Bisection budget for the voxel-size search below. Each iteration is one
# full voxel remesh (SDF rebuild), so this bounds worst-case wall time.
VOXEL_REMESH_MAX_ITERATIONS = 8


def voxel_remesh_canonicalize(
    input_path: str,
    output_path: str,
    report_path: str,
    target_faces: int,
    raw_face_min: int,
    raw_face_max: int,
) -> None:
    """S2 canonicalization's primary remesh: Blender voxel remesh + decimate.

    Voxel remeshing rebuilds the surface from a signed-distance field rather
    than following the source's (possibly broken) topology, so the result is
    watertight and a single connected component *by construction* -- this is
    what actually fixes the 800+-shard Instant Meshes failure on one
    character's S1 mesh (docs/PIPELINE_POLICY.md S2; 2026-08-30).

    Strategy: binary-search the voxel size so the raw voxel-remeshed mesh
    lands comfortably ABOVE the family's triangle budget (`raw_face_min`..
    `raw_face_max`), then decimate (collapse) down to `target_faces`.
    Decimating down from a clean, over-tessellated manifold mesh is far
    safer than asking the remesher to hit the budget directly (which is
    exactly how Instant Meshes shattered the mesh in the first place).
    """
    clear_scene()
    import_model(input_path)
    source_meshes = list(selected_meshes())
    if not source_meshes:
        raise RuntimeError("Voxel remesh requires visible source geometry")
    joined = _join_meshes(source_meshes)
    for modifier in list(joined.modifiers):
        joined.modifiers.remove(modifier)
    bpy.ops.object.select_all(action="DESELECT")
    joined.select_set(True)
    bpy.context.view_layer.objects.active = joined
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    low, high = _world_bounds([joined])
    dimensions = high - low
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    if max_dimension <= 1e-9:
        raise RuntimeError("Voxel remesh requires non-degenerate source geometry")

    voxel_size = initial_voxel_size(max_dimension)
    low_voxel: float | None = None  # smallest voxel size tried so far (denser, more faces)
    high_voxel: float | None = None  # largest voxel size tried so far (coarser, fewer faces)
    attempts: list[dict] = []
    winner = None
    winner_faces = None
    winner_voxel_size = None
    in_band = False

    for iteration in range(VOXEL_REMESH_MAX_ITERATIONS):
        candidate = joined.copy()
        candidate.data = joined.data.copy()
        bpy.context.scene.collection.objects.link(candidate)
        bpy.ops.object.select_all(action="DESELECT")
        candidate.select_set(True)
        bpy.context.view_layer.objects.active = candidate
        candidate.data.remesh_voxel_size = voxel_size
        bpy.ops.object.voxel_remesh()
        # Blender's voxel remesh outputs a quad-dominant mesh, but every
        # downstream face count in this function -- the bisection search's
        # raw_faces, the decimate ratio, and mesh_quality.analyze_mesh on the
        # eventual triangulated OBJ export -- must mean the same thing
        # ("triangles"), or the decimate ratio silently targets the wrong
        # count and the output overshoots the triangle budget by ~2x (found
        # live against Mary's mesh 2026-08-30: a 30k-tri target produced a
        # ~60k-tri candidate every attempt until this fix). Triangulate here,
        # immediately, so raw_faces already is the true triangle count.
        candidate_bm = bmesh.new()
        candidate_bm.from_mesh(candidate.data)
        bmesh.ops.triangulate(candidate_bm, faces=candidate_bm.faces)
        candidate_bm.to_mesh(candidate.data)
        candidate_bm.free()
        candidate.data.update()
        raw_faces = len(candidate.data.polygons)
        attempts.append({"iteration": iteration + 1, "voxel_size": voxel_size, "raw_faces": raw_faces})
        log(f"Voxel remesh attempt {iteration + 1}: voxel_size={voxel_size:.6f} -> {raw_faces} raw faces")

        in_band = raw_face_min <= raw_faces <= raw_face_max
        keep = winner is None or in_band or voxel_candidate_preferred(raw_faces, winner_faces, raw_face_min)
        if keep:
            if winner is not None:
                bpy.data.objects.remove(winner, do_unlink=True)
            winner, winner_faces, winner_voxel_size = candidate, raw_faces, voxel_size
        else:
            bpy.data.objects.remove(candidate, do_unlink=True)

        if in_band:
            break
        voxel_size, low_voxel, high_voxel = next_voxel_size(
            voxel_size, raw_faces, raw_face_min, raw_face_max, low_voxel, high_voxel,
        )

    if winner is None or winner_faces is None:
        raise RuntimeError("Voxel remesh produced no candidate")
    if winner_faces < VOXEL_REMESH_MIN_RAW_FACES:
        raise RuntimeError(f"Voxel remesh candidate has too few faces ({winner_faces}) to be usable")

    bpy.data.objects.remove(joined, do_unlink=True)

    decimate_ratio = canonicalize_decimate_ratio(winner_faces, target_faces)
    if decimate_ratio < 1.0:
        bpy.ops.object.select_all(action="DESELECT")
        winner.select_set(True)
        bpy.context.view_layer.objects.active = winner
        modifier = winner.modifiers.new(name="AISmith Canonicalize Budget", type="DECIMATE")
        modifier.ratio = decimate_ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    decimated_faces = len(winner.data.polygons)
    log(
        f"Decimated voxel remesh from {winner_faces} to {decimated_faces} faces "
        f"(ratio={decimate_ratio:.4f})"
    )

    removed_crumb_faces = _keep_largest_proxy_component(winner)
    if removed_crumb_faces:
        log(f"Removed {removed_crumb_faces} faces from detached crumbs after decimation")

    # Defensive weld: guarantee no near-duplicate-position vertices survive
    # decimation/triangulation before export, independent of whatever the
    # OBJ round-trip does with per-face-corner normals below.
    weld_bm = bmesh.new()
    weld_bm.from_mesh(winner.data)
    bmesh.ops.remove_doubles(weld_bm, verts=weld_bm.verts, dist=max_dimension * 1e-6)
    weld_bm.to_mesh(winner.data)
    weld_bm.free()
    winner.data.update()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    winner.select_set(True)
    bpy.context.view_layer.objects.active = winner
    # No normals, no UVs: this is bare geometry for the bake step to import
    # and re-derive both from (blender_bake_retopology -> retopo_bake calls
    # `_ensure_uv`, which creates the UV layer, and Blender recomputes
    # normals on import/use). Exporting Blender's per-face-corner split
    # normals here made `trimesh`'s OBJ loader (`process=False`, used by
    # `mesh_quality.analyze_mesh`'s unwelded-components check) treat
    # identical positions with differing normals as separate vertices --
    # 190 spurious "unwelded components" on an otherwise single-shell,
    # watertight, 0-non-manifold-edge mesh (found live against Mary's mesh
    # 2026-08-30, before this fix).
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_materials=False,
        export_normals=False,
        export_uv=False,
        export_triangulated_mesh=True,
    )
    Path(report_path).write_text(json.dumps({
        "method": "voxel",
        "target_faces": target_faces,
        "raw_face_band": [raw_face_min, raw_face_max],
        "raw_band_reached": in_band,
        "voxel_size": winner_voxel_size,
        "raw_faces": winner_faces,
        "decimated_faces": decimated_faces,
        "decimate_ratio": decimate_ratio,
        "removed_crumb_faces": removed_crumb_faces,
        "attempts": attempts,
    }), encoding="utf-8")
    log(f"Voxel canonicalize remesh complete: {decimated_faces} faces (voxel_size={winner_voxel_size:.6f})")


def _subject_top_band(meshes, fraction: float):
    """Camera framing for the top `fraction` of a subject -- its face.

    Returns `(center, extent)`: where to aim an orthographic camera and how
    wide to open it, both in world units.

    The frame is sized by the band's HEIGHT, never by its width. The alien
    this was calibrated on holds a T-pose whose arms sit at ~68% of body
    height, so its top 30% band is 1.00 wide and 0.30 tall -- framing that
    band's bounding box would render the whole body again and measure nothing
    about the face. The horizontal centre comes from the median of the band's
    UPPER HALF (unambiguously head), so outstretched arms cannot drag the
    camera off the face either.
    """
    blocks = []
    for obj in meshes:
        if obj.type != "MESH" or not len(obj.data.vertices):
            continue
        coordinates = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", coordinates)
        local = np.column_stack([
            coordinates.reshape(-1, 3).astype(np.float64),
            np.ones(len(obj.data.vertices)),
        ])
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        blocks.append((local @ matrix.T)[:, :3])
    if not blocks:
        raise RuntimeError("Cannot frame a face crop without vertices")
    array = np.concatenate(blocks, axis=0)
    low_z, high_z = float(array[:, 2].min()), float(array[:, 2].max())
    threshold = high_z - (high_z - low_z) * fraction
    band = array[array[:, 2] >= threshold]
    if len(band) < 3:
        band = array
    band_low_z, band_high_z = float(band[:, 2].min()), float(band[:, 2].max())
    head = band[band[:, 2] >= (band_low_z + band_high_z) * 0.5]
    if len(head) < 3:
        head = band
    center = Vector((
        float(np.median(head[:, 0])),
        float(np.median(head[:, 1])),
        (band_low_z + band_high_z) * 0.5,
    ))
    return center, max(band_high_z - band_low_z, 1e-6)


def render_canonicalize_similarity_views(
    source_path: str,
    canonical_path: str,
    output_dir: str,
    output_path: str,
    views: list,
    size: int = 256,
    face_fraction: float = 0.30,
) -> None:
    """Render the S1 source and the S2 canonicalized mesh from N matched
    viewpoints -- identical camera position and lighting per view -- for the
    appearance-preservation gate (docs/PIPELINE_POLICY.md S2). Writes
    `{view}_source.png` and `{view}_canonical.png` per view into
    `output_dir` and a manifest of those paths to `output_path`.

    Camera framing is computed from the UNION of both meshes' bounds, not
    the source alone: canonicalize's floor-contact normalization (z=0)
    translates the canonical mesh relative to the source, so framing off
    the source only can crop the canonical render (e.g. lop off the head)
    while the two meshes still nominally share a bounding volume before
    that translation. Both objects must be fully in frame for the pixel
    comparison to mean anything.
    """
    clear_scene()
    import_model(source_path)
    source_meshes = list(selected_meshes())
    if not source_meshes:
        raise RuntimeError("Similarity render requires a source mesh")

    import_model(canonical_path)
    canonical_meshes = [obj for obj in selected_meshes() if obj not in source_meshes]
    if not canonical_meshes:
        raise RuntimeError("Similarity render requires a canonicalized mesh")

    low, high = _world_bounds(source_meshes + canonical_meshes)
    center = (low + high) * 0.5
    diagonal = max(float((high - low).length), 1e-3)

    world = bpy.context.scene.world or bpy.data.worlds.new("AISmith Canonicalize World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5

    key_light_data = bpy.data.lights.new("AISmith Canonicalize Key Light", type="SUN")
    key_light_data.energy = 3.0
    key_light = bpy.data.objects.new("AISmith Canonicalize Key Light", key_light_data)
    bpy.context.scene.collection.objects.link(key_light)
    key_light.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))

    camera_data = bpy.data.cameras.new("AISmith Canonicalize Camera")
    camera = bpy.data.objects.new("AISmith Canonicalize Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = diagonal * 1.15
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # Facing -Y is the pipeline's canonical "front"; azimuth is measured
    # around the world Z axis from that convention.
    azimuth_degrees = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}
    distance = diagonal * 1.6

    # A second, tighter framing on the top `face_fraction` of the subject.
    # The whole-body views are structurally blind to resolution loss: a T-posed
    # character fills maybe a third of a square frame, so the face lands on a
    # few dozen pixels and a muddy bake is indistinguishable from a crisp one
    # (2026-09-01: the alien's blurred face still scored 0.997). The face
    # renders are what `native.compute_view_detail` measures high-frequency
    # energy on.
    #
    # Each mesh gets its OWN face centre but a SHARED ortho_scale. Canonicalize
    # floor-normalizes (z=0), so the S2 mesh sits a whole floor offset above
    # the S1 source; framing their union the way the whole-body camera does
    # would put the band across the canonical's shoulders and the source's
    # thin air. Sharing the ortho_scale is what makes the two renders
    # comparable at all -- a detail ratio between two different magnifications
    # would read as a fidelity difference when it is only a zoom difference.
    source_face_center, source_face_extent = _subject_top_band(source_meshes, face_fraction)
    canonical_face_center, canonical_face_extent = _subject_top_band(canonical_meshes, face_fraction)
    face_centers = {"source": source_face_center, "canonical": canonical_face_center}
    face_extent = max(source_face_extent, canonical_face_extent)
    face_camera_data = bpy.data.cameras.new("AISmith Canonicalize Face Camera")
    face_camera = bpy.data.objects.new("AISmith Canonicalize Face Camera", face_camera_data)
    bpy.context.scene.collection.objects.link(face_camera)
    face_camera_data.type = "ORTHO"
    face_camera_data.ortho_scale = max(face_extent, diagonal * 1e-3) * 1.15

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"views": {}, "face_fraction": face_fraction, "size": size}

    def show_only(subject: str) -> None:
        for obj in source_meshes:
            obj.hide_render = subject != "source"
        for obj in canonical_meshes:
            obj.hide_render = subject != "canonical"

    def render_to(path: Path) -> str:
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return str(path)

    for view in views:
        if view not in azimuth_degrees:
            raise ValueError(f"Unsupported canonicalize similarity view: {view!r}")
        angle = math.radians(azimuth_degrees[view])
        direction = Vector((math.sin(angle), -math.cos(angle), 0.25)).normalized()
        # The face camera looks straight in (no downward tilt): a top-down
        # component on a tight head crop trades face pixels for scalp.
        face_direction = Vector((math.sin(angle), -math.cos(angle), 0.0)).normalized()

        scene.camera = camera
        camera.location = center + direction * distance
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        paths: dict[str, str] = {}
        for subject in ("source", "canonical"):
            show_only(subject)
            paths[subject] = render_to(output_root / f"{view}_{subject}.png")

        scene.camera = face_camera
        for subject in ("source", "canonical"):
            face_center = face_centers[subject]
            face_camera.location = face_center + face_direction * distance
            face_camera.rotation_euler = (
                (face_center - face_camera.location).to_track_quat("-Z", "Y").to_euler()
            )
            bpy.context.view_layer.update()
            show_only(subject)
            paths[f"{subject}_face"] = render_to(output_root / f"{view}_{subject}_face.png")

        manifest["views"][view] = paths

    Path(output_path).write_text(json.dumps(manifest), encoding="utf-8")
    log(
        f"Rendered {len(views)} matched whole-body + face view pair(s) for the "
        f"canonicalize similarity gate (face band = top {face_fraction:.0%}, "
        f"ortho_scale={face_camera_data.ortho_scale:.4f})"
    )


def render_source_view(input_path: str, output_path: str, report_path: str, framing: dict, samples: int = 32) -> None:
    """Render a mesh from a `SourceCamera` so it can be scored against the
    photo that produced it -- neutral lighting and an untonemapped albedo
    pass, since the fidelity gate compares raw appearance, not a lit look."""
    start = time.time()
    clear_scene()
    import_model(input_path)
    armature = next((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None)
    meshes = skinned_body_meshes(armature) if armature is not None else [
        obj for obj in selected_meshes() if len(obj.data.vertices) > 0
    ]
    low, high = _world_bounds(meshes)

    if "camera" in framing:
        camera = SourceCamera.from_spec(framing["camera"])
    else:
        width, height = framing["image_size"]
        camera = fit_source_camera(
            tuple(low), tuple(high), (width, height), tuple(framing["subject_bbox"]),
            fov_x_degrees=framing.get("fov_x_degrees"),
            yaw_degrees=framing.get("yaw_degrees", 0.0),
            pitch_degrees=framing.get("pitch_degrees", 0.0),
        )

    cam_data = bpy.data.cameras.new("AISmith Source Fidelity Camera")
    cam_obj = bpy.data.objects.new("AISmith Source Fidelity Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.sensor_fit = "HORIZONTAL"
    if camera.kind == "perspective":
        cam_data.type = "PERSP"
        cam_data.angle_x = math.radians(camera.fov_x_degrees)
    else:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = camera.ortho_scale
    cam_data.clip_start = camera.distance * 0.01
    cam_data.clip_end = camera.distance * 20
    cam_obj.location = Vector(camera.center) + Vector(camera.direction()) * camera.distance
    cam_obj.rotation_euler = (Vector(camera.center) - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    world = bpy.context.scene.world or bpy.data.worlds.new("AISmith Source Fidelity World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    device = _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = camera.width
    scene.render.resolution_y = camera.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    if not Path(output_path).is_file():
        raise RuntimeError(f"render_source_view did not produce {output_path}")

    Path(report_path).write_text(
        json.dumps({
            "camera": camera.to_spec(),
            "mesh_bounds": {"low": list(low), "high": list(high)},
            "mesh_count": len(meshes),
            "device": device,
            "samples": samples,
            "elapsed_seconds": time.time() - start,
        }),
        encoding="utf-8",
    )
    log(f"Rendered source-fidelity view ({camera.kind}, {camera.width}x{camera.height}, device={device})")


def hex_rgba(value: str) -> tuple[float, float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("paint color must be a six digit hex value")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)


def paint(input_path: str, output_path: str, settings: dict) -> None:
    clear_scene()
    import_model(input_path)
    meshes = selected_meshes()
    if not meshes:
        raise RuntimeError("Texture paint requires at least one mesh")

    color = hex_rgba(str(settings.get("color", "#7c5cff")))
    roughness = max(0.0, min(1.0, float(settings.get("roughness", 42)) / 100.0))
    metallic = max(0.0, min(1.0, float(settings.get("metallic", 18)) / 100.0))
    size = max(256, min(2048, int(settings.get("texture_size", 1024))))
    image = bpy.data.images.new("AISmith_Painted_BaseColor", width=size, height=size, alpha=True)
    image.generated_color = color

    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if obj.data.uv_layers.active is None:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(island_margin=0.02)
            bpy.ops.object.mode_set(mode="OBJECT")
        material = obj.active_material or bpy.data.materials.new("AISmith Material")
        obj.active_material = material
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        tex = nodes.get("AISmith Painted Texture") or nodes.new("ShaderNodeTexImage")
        tex.name = "AISmith Painted Texture"
        tex.image = image
        if not any(link.to_node == bsdf and link.to_socket == bsdf.inputs.get("Base Color") for link in links):
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic

    # Apply strokes in UV space.  This is intentionally deterministic and
    # works for both imported GLBs and the generated mesh.
    pixels = [list(color) for _ in range(size * size)]
    strokes = settings.get("strokes", [])
    if not strokes:
        strokes = [{"x": 0.5, "y": 0.5, "radius": 0.18, "color": settings.get("color", "#7c5cff"), "opacity": 1.0}]
    for stroke in strokes:
        sx = max(0.0, min(1.0, float(stroke.get("x", 0.5))))
        sy = max(0.0, min(1.0, float(stroke.get("y", 0.5))))
        radius = max(0.002, min(0.5, float(stroke.get("radius", 0.08))))
        rgba = hex_rgba(str(stroke.get("color", settings.get("color", "#7c5cff"))))
        alpha = max(0.0, min(1.0, float(stroke.get("opacity", 1.0))))
        min_x, max_x = int((sx - radius) * size), int((sx + radius) * size)
        min_y, max_y = int((sy - radius) * size), int((sy + radius) * size)
        for y in range(max(0, min_y), min(size, max_y + 1)):
            for x in range(max(0, min_x), min(size, max_x + 1)):
                distance = math.hypot(x / size - sx, y / size - sy)
                if distance > radius:
                    continue
                hardness = max(0.05, min(1.0, float(stroke.get("hardness", 0.8))))
                falloff = 1.0 if distance <= radius * hardness else (1.0 - distance / radius) / (1.0 - hardness)
                a = alpha * max(0.0, min(1.0, falloff))
                idx = y * size + x
                old = pixels[idx]
                pixels[idx] = [old[c] * (1 - a) + rgba[c] * a for c in range(4)]
    image.pixels = [channel for px in pixels for channel in px]
    image.pack()
    export_glb(output_path)


def _proximity_weights(mesh, armature) -> None:
    """Assign smooth, deterministic weights by distance to deform-bone segments."""
    bones = [
        bone for bone in armature.data.bones
        if bone.use_deform and "leaf" not in bone.name.lower() and bone.name.lower() != "root"
    ]
    if not bones:
        raise RuntimeError("The armature has no deform bones")
    mesh.vertex_groups.clear()
    groups = [mesh.vertex_groups.new(name=bone.name) for bone in bones]
    armature_world = armature.matrix_world
    starts = np.asarray([armature_world @ bone.head_local for bone in bones], dtype=np.float64)
    ends = np.asarray([armature_world @ bone.tail_local for bone in bones], dtype=np.float64)
    segments = ends - starts
    segment_lengths_squared = np.maximum(np.sum(segments * segments, axis=1), 1e-12)
    mesh_world = np.asarray(mesh.matrix_world, dtype=np.float64)
    local_positions = np.empty((len(mesh.data.vertices), 4), dtype=np.float64)
    for vertex in mesh.data.vertices:
        local_positions[vertex.index, :3] = vertex.co
        local_positions[vertex.index, 3] = 1.0
    world_positions = local_positions @ mesh_world.T

    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    chunk_size = 16384
    influence_count = min(4, len(bones))
    for start in range(0, len(world_positions), chunk_size):
        points = world_positions[start:start + chunk_size, :3]
        relative = points[:, None, :] - starts[None, :, :]
        projection = np.clip(
            np.sum(relative * segments[None, :, :], axis=2) / segment_lengths_squared[None, :],
            0.0,
            1.0,
        )
        closest = starts[None, :, :] + projection[:, :, None] * segments[None, :, :]
        distances_squared = np.sum((points[:, None, :] - closest) ** 2, axis=2)
        nearest = np.argpartition(distances_squared, influence_count - 1, axis=1)[:, :influence_count]
        nearest_distances = np.take_along_axis(distances_squared, nearest, axis=1)
        inverse = 1.0 / np.maximum(nearest_distances, 1e-8)
        weights = inverse / np.sum(inverse, axis=1, keepdims=True)
        for row, vertex_index in enumerate(range(start, min(start + chunk_size, len(world_positions)))):
            vertex_weights = bm.verts[vertex_index][deform]
            for column in range(influence_count):
                bone_index = int(nearest[row, column])
                vertex_weights[groups[bone_index].index] = float(weights[row, column])
    bm.to_mesh(mesh.data)
    bm.free()


def _fill_unweighted_vertices(mesh, armature) -> int:
    """Fill bone-heat gaps without replacing the solver's authored weights."""
    bones = [
        bone for bone in armature.data.bones
        if bone.use_deform and is_proxy_deform_bone(bone.name)
    ]
    groups = {
        group.name: group
        for group in mesh.vertex_groups
        if group.name in {bone.name for bone in bones}
    }
    missing = [
        vertex for vertex in mesh.data.vertices
        if not any(
            element.group < len(mesh.vertex_groups)
            and mesh.vertex_groups[element.group].name in groups
            and element.weight > 0
            for element in vertex.groups
        )
    ]
    if not missing:
        return 0
    armature_world = armature.matrix_world
    starts = np.asarray([armature_world @ bone.head_local for bone in bones], dtype=np.float64)
    ends = np.asarray([armature_world @ bone.tail_local for bone in bones], dtype=np.float64)
    segments = ends - starts
    segment_lengths_squared = np.maximum(np.sum(segments * segments, axis=1), 1e-12)
    influence_count = min(4, len(bones))
    for vertex in missing:
        point = np.asarray(mesh.matrix_world @ vertex.co, dtype=np.float64)
        relative = point[None, :] - starts
        projection = np.clip(
            np.sum(relative * segments, axis=1) / segment_lengths_squared,
            0.0,
            1.0,
        )
        closest = starts + projection[:, None] * segments
        distances_squared = np.sum((point[None, :] - closest) ** 2, axis=1)
        nearest = np.argpartition(distances_squared, influence_count - 1)[:influence_count]
        inverse = 1.0 / np.maximum(distances_squared[nearest], 1e-8)
        weights = inverse / np.sum(inverse)
        for index, weight in zip(nearest, weights, strict=True):
            groups[bones[int(index)].name].add([vertex.index], float(weight), "REPLACE")
    return len(missing)


def _restore_local_finger_weights(mesh, armature) -> int:
    """Restore finger articulation after coarse proxy body-weight transfer."""
    finger_bones = [
        bone for bone in armature.data.bones
        if bone.use_deform and is_finger_deform_bone(bone.name)
    ]
    if not finger_bones:
        return 0
    groups = {
        bone.name: mesh.vertex_groups.get(bone.name)
        or mesh.vertex_groups.new(name=bone.name)
        for bone in finger_bones
    }
    hand_groups = {
        side: mesh.vertex_groups.get(f"hand_{side}")
        for side in ("l", "r")
    }
    armature_world = armature.matrix_world
    segments_by_side = {}
    for side in ("l", "r"):
        bones = [bone for bone in finger_bones if bone.name.casefold().endswith(f"_{side}")]
        segments = [
            (
                bone.name,
                armature_world @ bone.head_local,
                armature_world @ bone.tail_local,
            )
            for bone in bones
        ]
        lengths = [(end - start).length for _name, start, end in segments]
        if segments and lengths:
            segments_by_side[side] = (segments, max(lengths) * 0.9)

    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    index_to_name = {group.index: group.name for group in mesh.vertex_groups}
    name_to_index = {group.name: group.index for group in mesh.vertex_groups}
    changed = 0
    for vertex in bm.verts:
        current = {
            index_to_name[index]: float(weight)
            for index, weight in vertex[deform].items()
            if index in index_to_name and weight > 0
        }
        point = mesh.matrix_world @ vertex.co
        for side, hand_group in hand_groups.items():
            if hand_group is None or current.get(hand_group.name, 0.0) <= 0:
                continue
            side_data = segments_by_side.get(side)
            if side_data is None:
                continue
            segments, maximum_distance = side_data
            distances = {}
            for name, start, end in segments:
                segment = end - start
                length_squared = max(segment.length_squared, 1e-12)
                projection = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
                distances[name] = (point - (start + segment * projection)).length
            blended = blend_local_finger_weights(
                current,
                distances,
                hand_bone=hand_group.name,
                maximum_distance=maximum_distance,
            )
            if blended == current:
                continue
            weights = vertex[deform]
            weights.clear()
            for name, weight in blended.items():
                weights[name_to_index[name]] = weight
            current = blended
            changed += 1
            break
    coverage = {bone.name: 0.0 for bone in finger_bones}
    for vertex in bm.verts:
        for index, weight in vertex[deform].items():
            name = index_to_name.get(index)
            if name in coverage:
                coverage[name] += float(weight)
    unused_bones = [bone for bone in finger_bones if coverage[bone.name] <= 0]
    coverage_vertices = 0
    for bone in unused_bones:
        side = bone.name.casefold().rsplit("_", 1)[-1]
        start = armature_world @ bone.head_local
        end = armature_world @ bone.tail_local
        segment = end - start
        length_squared = max(segment.length_squared, 1e-12)
        candidates = []
        for vertex in bm.verts:
            current = {
                index_to_name[index]: float(weight)
                for index, weight in vertex[deform].items()
                if index in index_to_name and weight > 0
            }
            covered = add_local_finger_coverage(
                current,
                target_bone=bone.name,
                side=side,
            )
            if covered == current:
                continue
            point = mesh.matrix_world @ vertex.co
            projection = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
            distance = (point - (start + segment * projection)).length
            candidates.append((distance, vertex, covered))
        for _distance, vertex, covered in sorted(candidates, key=lambda item: item[0])[:12]:
            weights = vertex[deform]
            weights.clear()
            for name, weight in covered.items():
                weights[name_to_index[name]] = weight
            coverage_vertices += 1
    if coverage_vertices:
        log(
            f"Guaranteed terminal finger coverage on {coverage_vertices} vertices "
            f"for {', '.join(bone.name for bone in unused_bones)}"
        )
    bm.to_mesh(mesh.data)
    bm.free()
    if changed:
        log(f"Restored localized finger weights on {changed} visible vertices")
    return changed


def _remove_coincident_faces(mesh, seen_surfaces: set) -> int:
    """Remove redundant same-surface triangles before proxy skinning."""
    if mesh.data.shape_keys and len(mesh.data.shape_keys.key_blocks) > 1:
        log(f"Preserved coincident faces on morph-target mesh {mesh.name}")
        return 0
    polygons = list(mesh.data.polygons)
    animated = mesh
    preserve_animated_object = False
    while animated is not None:
        animation_data = animated.animation_data
        if (
            (
                animation_data is not None
                and (
                    animation_data.action
                    or animation_data.drivers
                    or animation_data.nla_tracks
                )
            )
            or animated.constraints
        ):
            preserve_animated_object = True
            break
        animated = animated.parent
    preserve_instance = mesh.data.users > 1 or preserve_animated_object
    material_groups = []
    culled_materials = set()
    normal_matrix = mesh.matrix_world.to_3x3().inverted_safe().transposed()
    for polygon in polygons:
        material = (
            mesh.data.materials[polygon.material_index]
            if polygon.material_index < len(mesh.data.materials)
            else None
        )
        material_name = material.name if material is not None else "<default>"
        corner_attributes = []
        corner_positions = []
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.data.loops[loop_index].vertex_index
            corner_positions.append(mesh.matrix_world @ mesh.data.vertices[vertex_index].co)
            attributes = []
            world_normal = normal_matrix @ mesh.data.loops[loop_index].normal
            attributes.extend(world_normal.normalized())
            for uv_layer in mesh.data.uv_layers:
                attributes.extend(uv_layer.data[loop_index].uv)
            for color_layer in mesh.data.color_attributes:
                color_index = loop_index if color_layer.domain == "CORNER" else vertex_index
                attributes.extend(color_layer.data[color_index].color)
            corner_attributes.append(attributes)
        preserve_blended_surface = False
        if material is not None:
            preserve_blended_surface = (
                material.surface_render_method == "BLENDED"
                or material.diffuse_color[3] < 0.999
            )
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    alpha = node.inputs.get("Alpha")
                    if alpha is not None and (
                        alpha.is_linked or float(alpha.default_value) < 0.999
                    ):
                        preserve_blended_surface = True
                        break
        group = (
            (mesh.name, polygon.index)
            if preserve_blended_surface
            else mesh.name if preserve_instance else "<shared>",
            material_name,
            face_render_signature(corner_positions, corner_attributes),
        )
        material_groups.append(group)
        if material is not None and material.use_backface_culling:
            culled_materials.add(group)
    kept = set(unique_position_face_indices(
        (tuple(mesh.matrix_world @ vertex.co) for vertex in mesh.data.vertices),
        (tuple(polygon.vertices) for polygon in polygons),
        face_groups=material_groups,
        preserve_opposite_winding_groups=culled_materials,
        seen_keys=seen_surfaces,
    ))
    duplicates = [
        polygon
        for index, polygon in enumerate(polygons)
        if index not in kept
    ]
    if not duplicates:
        return 0
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(
        bm,
        geom=[bm.faces[polygon.index] for polygon in duplicates],
        context="FACES",
    )
    bm.to_mesh(mesh.data)
    bm.free()
    mesh.data.update()
    return len(duplicates)


def _keep_largest_proxy_component(mesh) -> int:
    """Remove detached voxel crumbs so one body shell receives deform weights."""
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.faces.ensure_lookup_table()
    neighbors = [
        {
            linked.index
            for edge in face.edges
            for linked in edge.link_faces
            if linked != face
        }
        for face in bm.faces
    ]
    kept = largest_face_component(neighbors)
    removed = [face for face in bm.faces if face.index not in kept]
    if removed:
        bmesh.ops.delete(bm, geom=removed, context="FACES")
        loose = [vertex for vertex in bm.verts if not vertex.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(mesh.data)
        mesh.data.update()
    bm.free()
    return len(removed)


def _project_donor_weights(mesh, armature, donor_meshes) -> bool:
    """Project an already-authored donor skin onto a generated character mesh."""
    deform_bones = {
        bone.name for bone in armature.data.bones
        if bone.use_deform and "leaf" not in bone.name.lower()
        and bone.name.lower() != "root"
    }
    donor_surfaces = []
    for donor in donor_meshes:
        group_names = {
            group.index: group.name
            for group in donor.vertex_groups
            if group.name in deform_bones
        }
        vertex_weights = [
            {
                group_names[element.group]: element.weight
                for element in vertex.groups
                if element.group in group_names and element.weight > 0
            }
            for vertex in donor.data.vertices
        ]
        vertices = [donor.matrix_world @ vertex.co for vertex in donor.data.vertices]
        polygons = [tuple(polygon.vertices) for polygon in donor.data.polygons]
        if any(vertex_weights) and polygons:
            donor_surfaces.append((
                bvhtree.BVHTree.FromPolygons(vertices, polygons, all_triangles=False),
                vertices,
                polygons,
                vertex_weights,
            ))
    if not donor_surfaces:
        return False

    mesh.vertex_groups.clear()
    groups = {
        name: mesh.vertex_groups.new(name=name)
        for name in sorted(deform_bones)
    }
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    assigned = 0
    for vertex in bm.verts:
        world_position = mesh.matrix_world @ vertex.co
        nearest = None
        for tree, vertices, polygons, vertex_weights in donor_surfaces:
            location, _normal, polygon_index, distance = tree.find_nearest(world_position)
            if location is None or polygon_index is None:
                continue
            if nearest is None or distance < nearest[0]:
                nearest = (
                    distance,
                    location,
                    polygons[polygon_index],
                    vertices,
                    vertex_weights,
                )
        if nearest is None:
            continue
        _distance, location, polygon, donor_vertices, donor_weights = nearest
        if len(polygon) == 3:
            first, second, third = (donor_vertices[index] for index in polygon)
            barycentric = geometry.barycentric_transform(
                location,
                first,
                second,
                third,
                Vector((1.0, 0.0, 0.0)),
                Vector((0.0, 1.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
            )
            factors = [max(0.0, float(value)) for value in barycentric]
            factor_total = sum(factors)
            if factor_total > 0:
                factors = [value / factor_total for value in factors]
            weighted_maps = [
                (factors[index], donor_weights[vertex_index])
                for index, vertex_index in enumerate(polygon)
            ]
        else:
            weighted_maps = [
                (
                    1.0 / max((location - donor_vertices[index]).length_squared, 1e-12),
                    donor_weights[index],
                )
                for index in polygon
            ]
        weights = combine_weight_maps(weighted_maps)
        if not weights:
            continue
        assigned += 1
        vertex_weights = vertex[deform]
        for bone_name, weight in weights.items():
            vertex_weights[groups[bone_name].index] = weight
    if assigned != len(mesh.data.vertices):
        raise RuntimeError(
            "Donor weight projection left "
            f"{len(mesh.data.vertices) - assigned} vertices unweighted"
        )
    index_to_name = {group.index: group.name for group in groups.values()}
    projected_maps = [
        {
            index_to_name[group_index]: weight
            for group_index, weight in vertex[deform].items()
            if group_index in index_to_name and weight > 0
        }
        for vertex in bm.verts
    ]
    smoothed_maps = smooth_weight_maps(
        projected_maps,
        ((edge.verts[0].index, edge.verts[1].index) for edge in bm.edges),
    )
    name_to_index = {name: group.index for name, group in groups.items()}
    for vertex, weights in zip(bm.verts, smoothed_maps, strict=True):
        deform_weights = vertex[deform]
        deform_weights.clear()
        for bone_name, weight in weights.items():
            deform_weights[name_to_index[bone_name]] = weight
    bm.to_mesh(mesh.data)
    bm.free()
    log("Smoothed projected weights across target mesh topology")
    return True


def repair_weights(input_path: str, output_path: str) -> None:
    """Recompute deform weights while preserving the imported skeleton and clips."""
    clear_scene()
    import_model(input_path)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = selected_meshes()
    if len(armatures) != 1:
        raise RuntimeError(f"Weight repair requires exactly one armature; found {len(armatures)}")
    if not meshes:
        raise RuntimeError("Weight repair requires at least one mesh")
    armature = armatures[0]
    for mesh in meshes:
        for modifier in list(mesh.modifiers):
            if modifier.type == "ARMATURE":
                mesh.modifiers.remove(modifier)
        world_transform = mesh.matrix_world.copy()
        mesh.parent = armature
        mesh.matrix_world = world_transform
        modifier = mesh.modifiers.new(name="AISmith Armature", type="ARMATURE")
        modifier.object = armature
        log(f"Computing deterministic proximity weights for {mesh.name}")
        _proximity_weights(mesh, armature)
    export_glb(output_path, animations=True)


def _world_bounds(meshes) -> tuple[Vector, Vector]:
    points = [
        mesh.matrix_world @ Vector(corner)
        for mesh in meshes
        if len(mesh.data.vertices) > 0
        for corner in mesh.bound_box
    ]
    if not points:
        raise RuntimeError("Cannot align a rig without mesh bounds")
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def transfer_rig(input_path: str, donor_path: str, output_path: str) -> None:
    """Fit a certified donor armature to a target mesh and deterministically skin it."""
    clear_scene()
    import_model(input_path)
    target_meshes = list(selected_meshes())
    if not target_meshes:
        raise RuntimeError("Rig transfer target has no mesh")
    existing_armatures = {obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"}
    import_model(donor_path)
    donor_armatures = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE" and obj not in existing_armatures
    ]
    if len(donor_armatures) != 1:
        raise RuntimeError(f"Rig transfer requires exactly one donor armature; found {len(donor_armatures)}")
    armature = donor_armatures[0]
    donor_meshes = [mesh for mesh in selected_meshes() if mesh not in target_meshes]
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    log("Sampling donor geometry and weights in armature rest pose")
    target_low, target_high = _world_bounds(target_meshes)
    donor_low, donor_high = _world_bounds(donor_meshes)
    donor_height = donor_high.z - donor_low.z
    target_height = target_high.z - target_low.z
    if donor_height <= 1e-6 or target_height <= 1e-6:
        raise RuntimeError("Rig transfer requires non-zero target and donor height")
    scale = target_height / donor_height
    target_center = (target_low + target_high) * 0.5
    donor_center = (donor_low + donor_high) * 0.5
    translation = target_center - donor_center * scale
    alignment = Matrix.Translation(translation) @ Matrix.Scale(scale, 4)
    donor_world_matrices = {
        donor_mesh: donor_mesh.matrix_world.copy()
        for donor_mesh in donor_meshes
    }
    armature.matrix_world = alignment @ armature.matrix_world
    for donor_mesh in donor_meshes:
        donor_mesh.matrix_world = alignment @ donor_world_matrices[donor_mesh]
    bpy.context.view_layer.update()
    log(f"Aligned donor armature at uniform scale {scale:.6f}")
    for mesh in target_meshes:
        world_transform = mesh.matrix_world.copy()
        mesh.parent = armature
        mesh.matrix_world = world_transform
        modifier = mesh.modifiers.new(name="AISmith Armature", type="ARMATURE")
        modifier.object = armature
        log(f"Projecting donor skin weights onto {mesh.name}")
        if not _project_donor_weights(mesh, armature, donor_meshes):
            log(
                "Donor has no usable deform weights; falling back to "
                f"proximity skinning for {mesh.name}"
            )
            _proximity_weights(mesh, armature)
    for mesh in donor_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)
    armature.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    export_glb(output_path, animations=True)


def _retain_named_action(armature, action_name: str, donor_actions) -> None:
    try:
        selected_name = select_action_name(
            (action.name for action in donor_actions),
            action_name,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    selected = next(action for action in donor_actions if action.name == selected_name)
    animation_data = armature.animation_data_create()
    animation_data.action = selected
    for track in list(animation_data.nla_tracks):
        animation_data.nla_tracks.remove(track)
    for action in list(bpy.data.actions):
        if action != selected:
            bpy.data.actions.remove(action, do_unlink=True)
    selected.name = "Sword_Attack"
    log(f"Retained donor action {action_name} as Sword_Attack")


def _preserve_motion_source(armature):
    """Copy the animated donor before target-specific rest-bone fitting."""
    source = armature.copy()
    source.data = armature.data.copy()
    source.name = "AISmith Original Motion Source"
    bpy.context.scene.collection.objects.link(source)
    animation_data = source.animation_data_create()
    animation_data.action = armature.animation_data.action
    for track in list(animation_data.nla_tracks):
        animation_data.nla_tracks.remove(track)
    return source


def _bake_motion_to_fitted_armature(source, target) -> None:
    """Bake source rest-space motion into a separately fitted target rest rig."""
    source_action = source.animation_data.action if source.animation_data else None
    if source_action is None:
        raise RuntimeError("Target-proxy motion source has no active action")
    source_bones = {bone.name: bone for bone in source.pose.bones}
    target_bones = {bone.name: bone for bone in target.pose.bones}
    shared_names = sorted(source_bones.keys() & target_bones.keys())
    if not shared_names:
        raise RuntimeError("Target-proxy motion source shares no bones with fitted rig")

    target.animation_data_clear()
    target.animation_data_create()
    baked_action = bpy.data.actions.new("Sword_Attack")
    target.animation_data.action = baked_action
    start = int(math.floor(source_action.frame_range[0]))
    end = int(math.ceil(source_action.frame_range[1]))
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    translation_scale = _motion_translation_scale(
        source,
        target,
        {name: name for name in shared_names},
    )
    log(
        "Baking original donor motion into fitted rest axes across "
        f"{end - start + 1} frames"
    )
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        for bone in target.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            bone.location = (0.0, 0.0, 0.0)
            bone.scale = (1.0, 1.0, 1.0)
        for name in shared_names:
            source_bone = source_bones[name]
            target_bone = target_bones[name]
            target_bone.rotation_quaternion = _retarget_basis_rotation(
                source,
                source_bone,
                target,
                target_bone,
            )
            target_bone.location = _retarget_basis_translation(
                source,
                source_bone,
                target,
                target_bone,
                source_bone.location,
            ) * translation_scale
            target_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=frame,
                group=name,
            )
            target_bone.keyframe_insert(
                data_path="location",
                frame=frame,
                group=name,
            )
    source.animation_data_clear()
    bpy.data.objects.remove(source, do_unlink=True)
    if source_action.users == 0:
        bpy.data.actions.remove(source_action)
    baked_action.name = "Sword_Attack"


def _group_weight_centroid(meshes, group_name: str) -> Vector:
    samples = []
    for mesh in meshes:
        group = mesh.vertex_groups.get(group_name)
        if group is None:
            continue
        for vertex in mesh.data.vertices:
            weight = next(
                (
                    element.weight
                    for element in vertex.groups
                    if element.group == group.index
                ),
                0.0,
            )
            if weight > 0:
                samples.append((tuple(mesh.matrix_world @ vertex.co), weight))
    try:
        return Vector(weighted_centroid(samples))
    except ValueError as exc:
        raise RuntimeError(f"Sword attachment requires {group_name} vertex weights") from exc


def _fit_armature_limb_chains(proxy, armature) -> None:
    """Fit major limb rest bones to the proxy's weighted A-pose regions."""
    chains = resolve_limb_chains(bone.name for bone in armature.data.bones)
    fitted = []
    inverse_world = armature.matrix_world.inverted()
    targets = {}
    for chain in chains:
        if not all(
            armature.data.bones.get(name)
            and proxy.vertex_groups.get(name)
            and any(
                element.weight > 0
                for vertex in proxy.data.vertices
                for element in vertex.groups
                if element.group == proxy.vertex_groups[name].index
            )
            for name in chain
        ):
            log(f"Skipped limb chain with empty target weights: {', '.join(chain)}")
            continue
        centroids = [tuple(_group_weight_centroid([proxy], name)) for name in chain]
        targets[chain] = [
            inverse_world @ Vector(point)
            for point in chain_endpoints_from_centroids(centroids)
        ]

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for chain, endpoints in targets.items():
            bones = [armature.data.edit_bones.get(name) for name in chain]
            if any(bone is None for bone in bones):
                continue
            hand = bones[-1]
            parent_by_name = {
                bone.name: bone.parent.name if bone.parent else None
                for bone in armature.data.edit_bones
            }
            follower_names = descendant_names(parent_by_name, hand.name)
            follower_matrices = {
                name: armature.data.edit_bones[name].matrix.copy()
                for name in follower_names
            }
            original_hand_matrix = hand.matrix.copy()
            endpoints[0] = bones[0].head.copy()
            for index, bone in enumerate(bones):
                bone.head = endpoints[index]
                bone.tail = endpoints[index + 1]
            hand_delta = hand.matrix @ original_hand_matrix.inverted()
            for name, matrix in follower_matrices.items():
                armature.data.edit_bones[name].matrix = hand_delta @ matrix
            fitted.extend(chain)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    if fitted:
        log(f"Fitted {len(fitted)} limb bones to target A-pose landmarks")
    else:
        log("No supported limb chains were available for target A-pose fitting")


def _attach_right_hand_sword(
    armature,
    donor_height: float,
    anchor_world: Vector,
    hand_bone_name: str,
) -> None:
    """Create a deterministic rigid sword prop parented to the right hand."""
    hand = armature.pose.bones.get(hand_bone_name)
    if hand is None:
        raise RuntimeError(f"Sword attachment requires a {hand_bone_name} pose bone")

    materials = {}
    for name, color in (
        ("blade", (0.42, 0.48, 0.56, 1.0)),
        ("grip", (0.12, 0.045, 0.02, 1.0)),
        ("guard", (0.45, 0.24, 0.055, 1.0)),
    ):
        material = bpy.data.materials.new(f"AISmith Sword {name.title()}")
        material.diffuse_color = color
        materials[name] = material

    blade_length = donor_height * 0.38
    grip_length = donor_height * 0.075
    offsets = sword_component_offsets(
        blade_length=blade_length,
        grip_length=grip_length,
    )
    bone_world = armature.matrix_world @ hand.bone.matrix_local
    bone_world.translation = anchor_world
    components = (
        (
            "AISmith Sword Blade",
            (donor_height * 0.018, blade_length, donor_height * 0.006),
            offsets["blade"],
            materials["blade"],
        ),
        (
            "AISmith Sword Guard",
            (donor_height * 0.095, donor_height * 0.012, donor_height * 0.015),
            offsets["guard"],
            materials["guard"],
        ),
        (
            "AISmith Sword Grip",
            (donor_height * 0.022, grip_length, donor_height * 0.022),
            offsets["grip"],
            materials["grip"],
        ),
    )
    for name, dimensions, offset, material in components:
        bpy.ops.mesh.primitive_cube_add()
        component = bpy.context.object
        component.name = name
        component.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        desired_world = bone_world @ Matrix.Translation((0.0, offset, 0.0))
        component.matrix_world = desired_world
        world_transform = component.matrix_world.copy()
        component.parent = armature
        component.matrix_world = world_transform
        modifier = component.modifiers.new(name="AISmith Sword Armature", type="ARMATURE")
        modifier.object = armature
        hand_group = component.vertex_groups.new(name=hand.name)
        hand_group.add(
            [vertex.index for vertex in component.data.vertices],
            1.0,
            "REPLACE",
        )
        component.data.materials.append(material)
    log(f"Attached deterministic sword prop to {hand_bone_name}")


def target_proxy_rig(
    input_path: str,
    donor_path: str,
    output_path: str,
    action_name: str,
    attach_sword: bool,
    deformation_surface: str,
) -> None:
    """Skin visible target geometry from a watertight proxy derived from itself."""
    deformation_surface = select_deformation_surface(deformation_surface)
    clear_scene()
    import_model(input_path)
    target_meshes = list(selected_meshes())
    if not target_meshes:
        raise RuntimeError("Target-proxy rig requires visible target geometry")
    seen_surfaces = set()
    removed_faces = sum(
        _remove_coincident_faces(mesh, seen_surfaces)
        for mesh in target_meshes
    )
    if removed_faces:
        log(f"Removed {removed_faces} coincident target faces before rigging")
    existing_armatures = {obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"}
    existing_actions = set(bpy.data.actions)
    import_model(donor_path)
    donor_armatures = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE" and obj not in existing_armatures
    ]
    if len(donor_armatures) != 1:
        raise RuntimeError(
            f"Target-proxy rig requires exactly one donor armature; found {len(donor_armatures)}"
        )
    armature = donor_armatures[0]
    donor_meshes = [mesh for mesh in selected_meshes() if mesh not in target_meshes]
    donor_actions = [action for action in bpy.data.actions if action not in existing_actions]
    _retain_named_action(armature, action_name, donor_actions)
    motion_source = _preserve_motion_source(armature)
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    target_low, target_high = _world_bounds(target_meshes)
    donor_low, donor_high = _world_bounds(donor_meshes)
    target_height = target_high.z - target_low.z
    donor_height = donor_high.z - donor_low.z
    if target_height <= 1e-6 or donor_height <= 1e-6:
        raise RuntimeError("Target-proxy rig requires non-zero target and donor height")
    scale = target_height / donor_height
    target_center = (target_low + target_high) * 0.5
    donor_center = (donor_low + donor_high) * 0.5
    alignment = Matrix.Translation(target_center - donor_center * scale) @ Matrix.Scale(scale, 4)
    armature.matrix_world = alignment @ armature.matrix_world
    bpy.context.view_layer.update()
    log(f"Aligned donor armature at uniform scale {scale:.6f}")

    proxy_parts = []
    for source in target_meshes:
        proxy = source.copy()
        proxy.data = source.data.copy()
        proxy.animation_data_clear()
        proxy.parent = None
        proxy.matrix_world = source.matrix_world.copy()
        for modifier in list(proxy.modifiers):
            proxy.modifiers.remove(modifier)
        bpy.context.scene.collection.objects.link(proxy)
        proxy_parts.append(proxy)
    proxy = _join_meshes(proxy_parts)
    proxy.name = "AISmith Deformation Proxy"
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    proxy.data.remesh_voxel_size = target_height / 160.0
    log(f"Building watertight target proxy at voxel size {proxy.data.remesh_voxel_size:.6f}")
    bpy.ops.object.voxel_remesh()
    if len(proxy.data.vertices) < 1000:
        raise RuntimeError("Target-derived deformation proxy lost too much character geometry")
    log(f"Target proxy contains {len(proxy.data.vertices)} vertices")
    removed_proxy_faces = _keep_largest_proxy_component(proxy)
    if removed_proxy_faces:
        log(f"Removed {removed_proxy_faces} faces from detached voxel crumbs")
    smooth_modifier = proxy.modifiers.new(
        name="AISmith Deformation Surface Smooth",
        type="SMOOTH",
    )
    smooth_modifier.factor = 0.25
    smooth_modifier.iterations = 3
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.modifier_apply(modifier=smooth_modifier.name)
    log("Smoothed deformation proxy before UV projection")
    proxy_face_count = len(proxy.data.polygons)
    decimation_ratio = deformation_proxy_decimation_ratio(proxy_face_count)
    if decimation_ratio < 1.0:
        modifier = proxy.modifiers.new(
            name="AISmith Deformation Surface Budget",
            type="DECIMATE",
        )
        modifier.ratio = decimation_ratio
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = proxy
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        log(
            "Decimated deformation proxy from "
            f"{proxy_face_count} to {len(proxy.data.polygons)} faces"
        )

    proxy.vertex_groups.clear()
    original_deform_state = {
        bone.name: bone.use_deform
        for bone in armature.data.bones
    }
    for bone in armature.data.bones:
        bone.use_deform = bone.use_deform and is_proxy_deform_bone(bone.name)
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        raise RuntimeError(f"Automatic proxy weighting failed: {exc}") from exc
    bpy.context.view_layer.update()
    if not any(group.name in armature.data.bones for group in proxy.vertex_groups):
        raise RuntimeError("Automatic proxy weighting produced no deform groups")
    filled = _fill_unweighted_vertices(proxy, armature)
    if filled:
        log(f"Filled {filled} bone-heat gaps on the target proxy")
    _fit_armature_limb_chains(proxy, armature)
    world_transform = proxy.matrix_world.copy()
    proxy.parent = None
    proxy.matrix_world = world_transform
    for modifier in list(proxy.modifiers):
        if modifier.type == "ARMATURE":
            proxy.modifiers.remove(modifier)
    proxy.vertex_groups.clear()
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        raise RuntimeError(f"Fitted-armature proxy weighting failed: {exc}") from exc
    bpy.context.view_layer.update()
    refilled = _fill_unweighted_vertices(proxy, armature)
    if refilled:
        log(f"Filled {refilled} bone-heat gaps after target armature fitting")
    for bone in armature.data.bones:
        bone.use_deform = original_deform_state[bone.name]
    if deformation_surface == "voxel-proxy":
        log("Baking source appearance onto the watertight deformation proxy")
        _bake_source_appearance(
            target_meshes,
            proxy,
            ray_distance_multiplier=DEFORMATION_PROXY_BAKE_RAY_MULTIPLIER,
            uv_angle_limit=math.radians(88.0),
        )
        for mesh in target_meshes:
            bpy.data.objects.remove(mesh, do_unlink=True)
        target_meshes = [proxy]
        proxy.name = "AISmith Watertight Deformation Surface"
    else:
        log("Transferring target-derived proxy weights onto visible geometry")
        for mesh in target_meshes:
            for modifier in list(mesh.modifiers):
                if modifier.type == "ARMATURE":
                    mesh.modifiers.remove(modifier)
            world_transform = mesh.matrix_world.copy()
            mesh.parent = armature
            mesh.matrix_world = world_transform
            modifier = mesh.modifiers.new(name="AISmith Armature", type="ARMATURE")
            modifier.object = armature
            if not _project_donor_weights(mesh, armature, [proxy]):
                raise RuntimeError("Target proxy contains no transferable deform weights")
            _restore_local_finger_weights(mesh, armature)
        bpy.data.objects.remove(proxy, do_unlink=True)
    for mesh in donor_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)
    _bake_motion_to_fitted_armature(motion_source, armature)
    if attach_sword:
        try:
            hand_bone_name = resolve_right_hand_bone(
                bone.name for bone in armature.data.bones
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        hand_anchor = _group_weight_centroid(target_meshes, hand_bone_name)
        _attach_right_hand_sword(
            armature,
            donor_height,
            hand_anchor,
            hand_bone_name,
        )
    armature.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    export_glb(output_path, animations=True)


def render_review_frame(
    input_path: str,
    output_path: str,
    frame_ratio: float,
    *,
    rest_pose: bool = False,
    focus_bone: str | None = None,
) -> None:
    """Render a deterministic three-quarter review frame for deformation checks."""
    clear_scene()
    import_model(input_path)
    meshes = selected_meshes()
    if not meshes:
        raise RuntimeError("Deformation review requires a visible mesh")
    actions = [action for action in bpy.data.actions if action.frame_range[1] > action.frame_range[0]]
    if actions:
        action = max(actions, key=lambda item: item.frame_range[1] - item.frame_range[0])
        frame_start, frame_end = (round(value) for value in action.frame_range)
    else:
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
    if rest_pose:
        for armature in (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"):
            armature.data.pose_position = "REST"
        frame = frame_start
    else:
        frame = select_review_frame(frame_start, frame_end, frame_ratio)
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed_positions = np.concatenate(
        [_evaluated_positions(mesh, depsgraph) for mesh in meshes],
        axis=0,
    )
    low = Vector(tuple(np.min(posed_positions, axis=0)))
    high = Vector(tuple(np.max(posed_positions, axis=0)))
    center = (low + high) * 0.5
    height = max(high.z - low.z, 1e-3)
    focus_scale = None
    if focus_bone is not None:
        matching_bones = []
        for armature in (
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        ):
            try:
                resolved_focus = resolve_review_focus_bone(
                    (bone.name for bone in armature.pose.bones),
                    focus_bone,
                )
            except ValueError:
                continue
            matching_bones.append((armature, armature.pose.bones[resolved_focus]))
        if len(matching_bones) != 1:
            raise RuntimeError(
                f"Focused review requires exactly one {focus_bone} pose bone; "
                f"found {len(matching_bones)}"
            )
        armature, bone = matching_bones[0]
        center = armature.matrix_world @ bone.center
        focus_scale = height * 0.22

    camera_data = bpy.data.cameras.new("AISmith Review Camera")
    camera = bpy.data.objects.new("AISmith Review Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    model_diagonal = max(float(np.linalg.norm(high - low)), height)
    camera.location = center + Vector((model_diagonal * 0.85, -model_diagonal * 2.4, model_diagonal * 0.15))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    view_matrix = np.asarray(camera.matrix_world.inverted(), dtype=np.float64)
    homogeneous_positions = np.column_stack((posed_positions, np.ones(len(posed_positions))))
    camera_positions = homogeneous_positions @ view_matrix.T
    projected_width = float(np.ptp(camera_positions[:, 0]))
    projected_height = float(np.ptp(camera_positions[:, 1]))
    camera_data.ortho_scale = (
        focus_scale
        if focus_scale is not None
        else max(projected_width, projected_height, 1e-3) * 1.15
    )

    world = bpy.context.scene.world or bpy.data.worlds.new("AISmith Review World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.008, 0.01, 0.014, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    for location, energy, size in (
        ((-height, -height, height * 1.5), 1200.0, height),
        ((height, -height * 0.5, height), 800.0, height * 0.7),
    ):
        light_data = bpy.data.lights.new("AISmith Review Light", type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new("AISmith Review Light", light_data)
        bpy.context.scene.collection.objects.link(light)
        light.location = Vector(location)
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    scene = bpy.context.scene
    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    scene.render.film_transparent = False
    bpy.ops.render.render(write_still=True)
    log(f"Rendered deformation review frame {frame} to {output_path}")


def _skeleton_side_color(bone_name: str) -> tuple[float, float, float, float]:
    lowered = bone_name.lower()
    if lowered.endswith("_l"):
        return (0.2, 0.5, 1.0, 1.0)
    if lowered.endswith("_r"):
        return (1.0, 0.55, 0.15, 1.0)
    return (1.0, 1.0, 1.0, 1.0)


def _skeleton_emissive_material(color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(f"AISmith Skeleton {color}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.5
    links.new(emission.outputs[0], output.inputs["Surface"])
    return material


def render_skeleton_sheet(input_path: str, out_dir: str, size: int = 768) -> None:
    """Render front/right/left/back orthographic views of the armature's
    bones for visual left/right review.

    Cycles has no built-in armature display (that overlay is viewport-only),
    so the skeleton is rebuilt as real geometry: a thin cylinder per bone
    (head -> tail, world space) plus a small sphere at each head, colored by
    side (blue `_l`, orange `_r`, white otherwise) so a mirrored-rig failure
    is visible at a glance instead of requiring the bilateral_symmetry
    numbers to be read first. The body mesh is kept but flattened to a
    translucent gray so the bones read through it rather than being hidden
    inside it.

    Only the SKINNED body mesh(es) are used for framing, materials, and
    rendering (`skinned_body_meshes`) -- Blender's glTF importer can attach
    a unit-Icosphere custom display shape to the armature's bones that
    carries no skin data and is not part of the character, and including
    it in a naive "every mesh in the scene" bounds computation (or leaving
    it visible in the render) is what inflated this sheet's camera framing
    to ~45% fill on the 2026-09 GPU-box renders.
    """
    clear_scene()
    # "TEMPERANCE" stops Blender 4.x's glTF importer from creating the
    # unit-Icosphere bone display shape at all (the default "BLENDER" does).
    # It also re-orients every bone's rest frame, which is harmless here --
    # this op only renders, nothing is re-exported or posed from a store --
    # and is exactly why no other op may use it (see `import_model`).
    import_model(input_path, bone_heuristic="TEMPERANCE")
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("Skeleton sheet requires an armature")
    armature = max(armatures, key=lambda item: (len(item.pose.bones), item.name))
    meshes = skinned_body_meshes(armature)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj not in meshes:
            # Importer scaffolding (a bone display shape, most commonly) --
            # excluded from bounds and materials below, and it must not be
            # rendered either.
            obj.hide_render = True

    # Framing comes from the body meshes' own evaluated (posed) world
    # bounds only -- captured before any of the bone cylinders/spheres
    # built below exist, so there is nothing for this box to accidentally
    # include. A root bone's long tail (or anything else added later) can
    # never inflate it.
    if meshes:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_positions = np.concatenate(
            [_evaluated_positions(mesh, depsgraph) for mesh in meshes], axis=0
        )
        low = Vector(tuple(np.min(mesh_positions, axis=0)))
        high = Vector(tuple(np.max(mesh_positions, axis=0)))
    else:
        low, high = Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    center = (low + high) * 0.5
    extent = high - low
    diagonal = max(float(extent.length), 1e-3)
    height = max(float(extent.z), 1e-3)
    bone_radius = 0.004 * height
    joint_radius = 0.006 * height

    materials: dict[tuple, object] = {}

    def material_for(bone_name: str):
        color = _skeleton_side_color(bone_name)
        if color not in materials:
            materials[color] = _skeleton_emissive_material(color)
        return materials[color]

    bone_collection = bpy.data.collections.new("AISmith Skeleton Sheet")
    bpy.context.scene.collection.children.link(bone_collection)

    for bone in armature.pose.bones:
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail
        material = material_for(bone.name)

        joint_mesh = bpy.data.meshes.new(f"AISmith Skeleton Joint {bone.name}")
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=joint_radius)
        bm.to_mesh(joint_mesh)
        bm.free()
        joint_object = bpy.data.objects.new(f"AISmith Skeleton Joint {bone.name}", joint_mesh)
        joint_object.location = head
        joint_object.data.materials.append(material)
        bone_collection.objects.link(joint_object)

        length = (tail - head).length
        if length <= 1e-6:
            continue
        bone_mesh = bpy.data.meshes.new(f"AISmith Skeleton Bone {bone.name}")
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=True, segments=8,
            radius1=bone_radius, radius2=bone_radius, depth=length,
        )
        bm.to_mesh(bone_mesh)
        bm.free()
        bone_object = bpy.data.objects.new(f"AISmith Skeleton Bone {bone.name}", bone_mesh)
        bone_object.location = (head + tail) * 0.5
        bone_object.rotation_euler = (tail - head).normalized().to_track_quat("Z", "Y").to_euler()
        bone_object.data.materials.append(material)
        bone_collection.objects.link(bone_object)

    flat_material = bpy.data.materials.new("AISmith Skeleton Sheet Flat")
    flat_material.use_nodes = True
    flat_bsdf = flat_material.node_tree.nodes.get("Principled BSDF")
    flat_bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    # 0.5, not the 0.35 first tried: the mesh needs to read as a body, not
    # just a faint haze the bones float in.
    flat_bsdf.inputs["Alpha"].default_value = 0.5
    flat_material.blend_method = "BLEND"
    # `shadow_method` was deprecated in Blender 4.2 (EEVEE Next) and removed
    # in 5.x, where assigning it raises AttributeError and would fail the
    # whole sheet; the translucent override only needs it where it exists.
    if hasattr(flat_material, "shadow_method"):
        flat_material.shadow_method = "NONE"
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(flat_material)

    world = bpy.context.scene.world or bpy.data.worlds.new("AISmith Skeleton World")
    bpy.context.scene.world = world
    world.use_nodes = True
    # 0.62 mid-gray, same backdrop the similarity-view renders use, so a
    # translucent body and colored bones both read against it.
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.62, 0.62, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    camera_data = bpy.data.cameras.new("AISmith Skeleton Camera")
    camera = bpy.data.objects.new("AISmith Skeleton Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # Same azimuth table `render_canonicalize_similarity_views` uses: facing
    # -Y is the pipeline's canonical "front".
    azimuth_degrees = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}
    distance = diagonal * 2.0
    corners = [
        Vector((x, y, z))
        for x in (low.x, high.x) for y in (low.y, high.y) for z in (low.z, high.z)
    ]
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for role, azimuth in azimuth_degrees.items():
        angle = math.radians(azimuth)
        direction = Vector((math.sin(angle), -math.cos(angle), 0.0)).normalized()
        camera.location = center + direction * distance
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        # ortho_scale sized off a single world-axis scalar assumes that
        # axis is what a given view actually sees; projecting the true bbox
        # corners onto THIS camera's own right/up axes removes that
        # assumption and is what keeps every one of the four views tightly
        # framed regardless of which world axis is "up" from that angle.
        rotation = camera.matrix_world.to_quaternion()
        right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
        up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
        projected_right = [corner.dot(right) for corner in corners]
        projected_up = [corner.dot(up) for corner in corners]
        extent_right = max(projected_right) - min(projected_right)
        extent_up = max(projected_up) - min(projected_up)
        camera_data.ortho_scale = max(extent_right, extent_up, 1e-3) * 1.12

        image_path = output_root / f"{role}.png"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        fill_x, fill_y = _render_fill_fraction(image_path)
        log(f"sheet framing {role}: fill_x={fill_x:.3f} fill_y={fill_y:.3f}")
        if max(fill_x, fill_y) < 0.7:
            raise RuntimeError(
                f"Skeleton sheet framing regressed for {role} view: "
                f"fill_x={fill_x:.3f} fill_y={fill_y:.3f} (expected >= 0.7)"
            )

    log(f"Rendered skeleton contact sheet ({len(armature.pose.bones)} bones) to {out_dir}")


def _render_fill_fraction(image_path: Path) -> tuple[float, float]:
    """How much of a rendered PNG's width/height the subject's silhouette
    fills.

    The background reference is read from the render's OWN corner pixel,
    never a hardcoded linear color: Blender's view transform remaps
    whatever scene-linear world-background color was set before it ever
    reaches the PNG, so only a pixel the render actually produced can be
    trusted as "this is background". Framing regressions caught this way
    fail loud (`render_skeleton_sheet` raises below) instead of shipping a
    tiny, unreadable figure silently.
    """
    image = bpy.data.images.load(str(image_path))
    try:
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float64).reshape((height, width, 4))[:, :, :3]
        background = pixels[0, 0]
        mask = np.max(np.abs(pixels - background), axis=2) > 0.04
        if not np.any(mask):
            return 0.0, 0.0
        rows = np.nonzero(np.any(mask, axis=1))[0]
        cols = np.nonzero(np.any(mask, axis=0))[0]
        return (
            float(cols[-1] - cols[0] + 1) / width,
            float(rows[-1] - rows[0] + 1) / height,
        )
    finally:
        bpy.data.images.remove(image)


def _evaluated_positions(mesh, depsgraph) -> np.ndarray:
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        positions = np.empty((len(evaluated_mesh.vertices), 4), dtype=np.float64)
        for vertex in evaluated_mesh.vertices:
            positions[vertex.index, :3] = vertex.co
            positions[vertex.index, 3] = 1.0
        return (positions @ matrix.T)[:, :3]
    finally:
        evaluated.to_mesh_clear()


def deformation_diagnostics(input_path: str, output_path: str) -> None:
    """Measure edge and silhouette stability across deterministic action frames."""
    clear_scene()
    import_model(input_path)
    meshes = selected_meshes()
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    actions = [action for action in bpy.data.actions if action.frame_range[1] > action.frame_range[0]]
    if not meshes or not armatures:
        raise RuntimeError("Deformation diagnostics require a skinned animated GLB")
    synthetic_action = not actions
    if actions:
        action = max(actions, key=lambda item: item.frame_range[1] - item.frame_range[0])
        frame_start, frame_end = (round(value) for value in action.frame_range)
    else:
        # First-time rigs carry no animation clip yet; diagnose a synthetic
        # sway across the rest pose so the deformation gate still measures.
        frame_start, frame_end = 1, 24
        sway = armatures[0].animation_data_create()
        sway.action = bpy.data.actions.new("UniRigRestSway")
        # `action` is what the report names below; without this the synthetic
        # branch raised NameError instead of reporting the sway.
        action = sway.action
        pose_bone = armatures[0].pose.bones[0]
        curve = sway.action.fcurves.new(data_path=f'pose.bones["{pose_bone.name}"].rotation_euler', index=1)
        for frame, angle in ((1, 0.0), (12, 0.12), (24, 0.0)):
            curve.keyframe_points.insert(frame, angle)
    for armature in armatures:
        armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rest_positions = [_evaluated_positions(mesh, depsgraph) for mesh in meshes]
    rest_edges = [
        np.asarray([(edge.vertices[0], edge.vertices[1]) for edge in mesh.data.edges], dtype=np.int64)
        for mesh in meshes
    ]
    rest_lengths = [
        np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
        if len(edges) else np.zeros(0, dtype=np.float64)
        for positions, edges in zip(rest_positions, rest_edges, strict=True)
    ]
    all_rest = np.concatenate(rest_positions, axis=0)
    rest_diagonal = float(np.linalg.norm(np.ptp(all_rest, axis=0)))

    for armature in armatures:
        armature.data.pose_position = "POSE"
    frames = []
    # The character rig, not a prop's: most bones wins, name breaks ties so
    # the choice is deterministic across runs.
    motion_armature = max(armatures, key=lambda item: (len(item.pose.bones), item.name))
    sampled_frames: list[int] = []
    motion_heads: dict[str, list] = {bone.name: [] for bone in motion_armature.pose.bones}
    motion_tails: dict[str, list] = {bone.name: [] for bone in motion_armature.pose.bones}
    animation_frames = all_animation_frames(frame_start, frame_end)
    frame_span = max(frame_end - frame_start, 1)
    for frame in animation_frames:
        ratio = (frame - frame_start) / frame_span
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        # Same posed frame the stretch measurement below reads, so the motion
        # metrics and the deformation metrics never disagree about what the
        # clip did on a given frame.
        sampled_frames.append(int(frame))
        world = motion_armature.matrix_world
        for pose_bone in motion_armature.pose.bones:
            motion_heads[pose_bone.name].append(tuple(world @ pose_bone.head))
            motion_tails[pose_bone.name].append(tuple(world @ pose_bone.tail))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_positions = [_evaluated_positions(mesh, depsgraph) for mesh in meshes]
        stretch_values = []
        for positions, edges, baseline in zip(
            evaluated_positions,
            rest_edges,
            rest_lengths,
            strict=True,
        ):
            if len(edges):
                current = np.linalg.norm(
                    positions[edges[:, 0]] - positions[edges[:, 1]],
                    axis=1,
                )
                stretch_values.append(current / np.maximum(baseline, 1e-9))
        stretch = np.concatenate(stretch_values) if stretch_values else np.ones(1)
        all_positions = np.concatenate(evaluated_positions, axis=0)
        diagonal = float(np.linalg.norm(np.ptp(all_positions, axis=0)))
        frames.append({
            "frame": frame,
            "ratio": ratio,
            "p99_edge_stretch": round(float(np.percentile(stretch, 99)), 4),
            "maximum_edge_stretch": round(float(np.max(stretch)), 4),
            "bounds_diagonal_ratio": round(diagonal / max(rest_diagonal, 1e-9), 4),
        })
    motion_review = compute_motion_metrics(
        sampled_frames,
        {name: np.asarray(values, dtype=np.float64) for name, values in motion_heads.items()},
        {name: np.asarray(values, dtype=np.float64) for name, values in motion_tails.items()},
        synthetic=synthetic_action,
        source_action=action.name,
    )
    report = {
        "schema_version": 1,
        "action": action.name,
        "complete_action_range": True,
        "action_frame_start": frame_start,
        "action_frame_end": frame_end,
        "rest_bounds_diagonal": round(rest_diagonal, 4),
        "frames": frames,
        "maximum_p99_edge_stretch": max(item["p99_edge_stretch"] for item in frames),
        "maximum_bounds_diagonal_ratio": max(item["bounds_diagonal_ratio"] for item in frames),
        # Edge stretch says the surface held together; it says nothing about
        # whether the rig actually moved. `motion_review` is that second
        # question (backend/app/motion_metrics.py).
        "motion_review": motion_review,
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if motion_review.get("measurable"):
        log(
            f"Measured deformation across {len(frames)} review frames; motion "
            f"coverage={motion_review['joint_motion_coverage']:.3f} "
            f"static_tail={motion_review['static_tail_fraction']:.3f} "
            f"foot_slide_max={motion_review['foot_slide_max']:.4f}"
        )
    else:
        log(f"Measured deformation across {len(frames)} review frames; motion not measurable")


def _retarget_basis_rotation(source_armature, source_bone, target_armature, target_bone):
    """Express a source local rotation delta in the target bone's rest axes."""
    source_rest = source_bone.bone.matrix_local.to_quaternion()
    target_rest = target_bone.bone.matrix_local.to_quaternion()
    source_basis = source_bone.matrix_basis.to_quaternion()
    source_armature_delta = source_rest @ source_basis @ source_rest.inverted()
    source_object_basis = source_armature.matrix_world.to_quaternion()
    target_object_basis = target_armature.matrix_world.to_quaternion()
    world_delta = source_object_basis @ source_armature_delta @ source_object_basis.inverted()
    target_armature_delta = target_object_basis.inverted() @ world_delta @ target_object_basis
    return target_rest.inverted() @ target_armature_delta @ target_rest


def _resolve_motion_mapping(source, target, resolved_roles: dict[str, str]) -> dict[str, str]:
    """Resolve exact chains and collapse missing source ancestors as needed."""
    return resolve_motion_mapping(
        {bone.name for bone in source.pose.bones},
        {bone.name for bone in target.pose.bones},
        resolved_roles,
    )


def _retarget_basis_translation(source_armature, source_bone, target_armature, target_bone, displacement):
    """Express a source-local displacement in the target bone's rest axes."""
    source_armature_displacement = source_bone.bone.matrix_local.to_3x3() @ displacement
    world_displacement = source_armature.matrix_world.to_3x3() @ source_armature_displacement
    target_armature_displacement = target_armature.matrix_world.to_3x3().inverted() @ world_displacement
    return target_bone.bone.matrix_local.to_3x3().inverted() @ target_armature_displacement


def _motion_translation_scale(source, target, mapping: dict[str, str]) -> float:
    source_hips = source.data.bones.get("Hips")
    source_head = source.data.bones.get("Head")
    target_hips = target.data.bones.get(mapping.get("Hips", ""))
    target_head = target.data.bones.get(mapping.get("Head", ""))
    if any(bone is None for bone in (source_hips, source_head, target_hips, target_head)):
        return 1.0
    source_height = ((source.matrix_world @ source_head.head_local) - (source.matrix_world @ source_hips.head_local)).length
    target_height = ((target.matrix_world @ target_head.head_local) - (target.matrix_world @ target_hips.head_local)).length
    return target_height / source_height if source_height > 1e-6 else 1.0


def normalize_bvh_axes(input_path: str, output_path: str) -> None:
    """Convert an externally generated BVH motion clip to the house
    Blender-native Z-up convention, at the pipeline's ingest boundary.

    NVIDIA Kimodo (backend.app.native.run_kimodo_motion) writes BVH motion
    with the standard mocap Y-up axis convention. Every other BVH artifact
    in this repo is written by export_soma_bvh (unirig_bind_worker.py)
    using Blender's own bpy.ops.export_anim.bvh exporter with no axis
    conversion -- i.e. Blender-native Z-up -- and retarget_bvh (below)
    imports that convention with an identity axis mapping
    (axis_forward="Y", axis_up="Z"). Per docs/PIPELINE_POLICY.md
    ("normalize at the ingest boundary" / "blame upstream"), a Kimodo BVH
    is normalized to that same Z-up convention exactly once, here, at
    generation time -- retarget_bvh never special-cases a motion clip's
    source.

    Import uses Blender's standard Y-up mocap reading (axis_forward="-Z",
    axis_up="Y"). global_scale=0.01 brings Kimodo's centimetre-scale
    offsets (its raw BVH has Hips OFFSET Y=100.0, i.e. roughly a 1 m hip
    height written as "100") into Blender's metre-scale scene -- the same
    net scale retarget_bvh's own BVH import produces for a house BVH
    (compare its global_scale=0.01 against export_soma_bvh's
    global_scale=100.0). Export then reuses export_soma_bvh's exact call
    (global_scale=100.0, default axes) so the output file is the same
    convention as every other artifact retarget_bvh consumes -- net scale
    is unchanged (0.01 * 100.0 == 1), only the axis convention flips.
    """
    clear_scene()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(
        filepath=input_path,
        axis_forward="-Z",
        axis_up="Y",
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=True,
        global_scale=0.01,
    )
    armatures = [
        obj for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "ARMATURE"
    ]
    if len(armatures) != 1:
        raise RuntimeError(
            f"BVH import produced {len(armatures)} armatures during normalize-bvh"
        )
    armature = armatures[0]

    # Same rest-based upright check as retarget_bvh (see its comment): the
    # BVH's static rest hierarchy, not any animated frame, must be upright
    # if the axis convention assumption above is right.
    hips_data_bone = armature.data.bones.get("Hips")
    head_data_bone = armature.data.bones.get("Head")
    if hips_data_bone is None or head_data_bone is None:
        raise RuntimeError("normalize-bvh requires Hips and Head bones in the source BVH")
    spine_world = (
        (armature.matrix_world @ head_data_bone.head_local)
        - (armature.matrix_world @ hips_data_bone.head_local)
    )
    if spine_world.length < 1e-6 or spine_world.z <= 0.8 * spine_world.length:
        raise RuntimeError(
            "normalize-bvh: motion armature is still not upright after "
            "Y-up import -- axis convention assumption is wrong "
            f"(Hips->Head rest vector = {tuple(round(v, 4) for v in spine_world)}, "
            f"length={spine_world.length:.4f})"
        )

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # K17: bpy.ops.export_anim.bvh exports the scene's current frame range,
    # not the imported action's -- left at Blender's untouched default
    # (frame_end=250), a shorter Kimodo clip (e.g. a genuinely honored 2s /
    # ~60-frame request) got exported padded out to 250 frames, with the
    # action's last real frame held constant the rest of the way (confirmed
    # on a real clip: rows 60-250 of the normalized BVH were byte-identical
    # -- Kimodo *did* honor the request, only the re-export padded it). Seed
    # the scene's frame range from the imported action's own range first,
    # same as _bake_motion_to_fitted_armature already does for the same
    # reason; bvh_export_frame_range (retarget_math.py) is the pure
    # floor/ceil math, unit-tested there without needing bpy.
    imported_action = armature.animation_data.action if armature.animation_data else None
    if imported_action is not None:
        frame_start, frame_end = retarget_math.bvh_export_frame_range(tuple(imported_action.frame_range))
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(filepath=str(out_path), global_scale=100.0)
    log(f"normalized BVH axes: {input_path} -> {out_path} bytes={out_path.stat().st_size}")


def retarget_bvh(input_path: str, motion_path: str, output_path: str, animation: str, target_bones: dict[str, str]) -> None:
    """Bake a standard-T-pose SOMA BVH action onto an AISmith humanoid rig."""
    clear_scene()
    import_model(input_path)
    target_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    try:
        target_name = select_character_armature(
            {obj.name: {bone.name for bone in obj.pose.bones} for obj in target_armatures},
            target_bones,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    target = next(obj for obj in target_armatures if obj.name == target_name)
    # The output is a reproducible single-clip review artifact. Imported GLBs
    # often retain both an active action and NLA tracks; clear both before the
    # BVH import so Blender cannot export stale or duplicate Sword_Attack clips.
    if target.animation_data is not None:
        target.animation_data_clear()
    for existing_action in list(bpy.data.actions):
        bpy.data.actions.remove(existing_action)
    before = set(bpy.context.scene.objects)
    # export_soma_bvh (unirig_bind_worker.py) writes BVH files with Blender's
    # own exporter, which has no axis-conversion options and therefore always
    # writes Blender's Z-up convention. Importing with the Y-up default
    # (axis_forward="-Z", axis_up="Y") rotates the whole armature 90 degrees
    # about X; every downstream transform then composes rotations through
    # that wrong frame. The identity mapping (axis_forward="Y", axis_up="Z")
    # is the correct one for this specific writer/reader pair.
    bpy.ops.import_anim.bvh(
        filepath=motion_path,
        axis_forward="Y",
        axis_up="Z",
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=True,
        global_scale=0.01,
    )
    source_armatures = [
        obj for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "ARMATURE"
    ]
    if len(source_armatures) != 1:
        raise RuntimeError(f"BVH import produced {len(source_armatures)} motion armatures")
    source = source_armatures[0]
    source_action = source.animation_data.action if source.animation_data else None
    if source_action is None:
        raise RuntimeError("BVH motion did not contain an animation action")

    # Rest-based, not frame-1-pose-based: a mocap clip's own first frame
    # could legitimately be crouched or seated, but the BVH's static rest
    # hierarchy (head_local, before any channel animation is applied)
    # should always be upright if the axis convention is right.
    hips_data_bone = source.data.bones.get("Hips")
    head_data_bone = source.data.bones.get("Head")
    if hips_data_bone is not None and head_data_bone is not None:
        spine_world = (
            (source.matrix_world @ head_data_bone.head_local)
            - (source.matrix_world @ hips_data_bone.head_local)
        )
        if spine_world.length < 1e-6 or spine_world.z <= 0.8 * spine_world.length:
            raise RuntimeError(
                "motion armature is not upright — BVH axis convention mismatch "
                f"(Hips->Head rest vector = {tuple(round(v, 4) for v in spine_world)}, "
                f"length={spine_world.length:.4f})"
            )

    mapping = _resolve_motion_mapping(source, target, target_bones)
    required = {
        "Hips", "Head",
        "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot",
        "RightLeg", "RightShin", "RightFoot",
    }
    missing = sorted(required - set(mapping))
    if not {"Spine1", "Spine2", "Chest"} & set(mapping):
        missing.append("Spine1/Spine2/Chest")
    if missing:
        raise RuntimeError(f"BVH retarget is missing required humanoid joints: {', '.join(missing)}")

    if target.animation_data is None:
        target.animation_data_create()
    action = bpy.data.actions.new(animation)
    target.animation_data.action = action
    start = int(math.floor(source_action.frame_range[0]))
    end = int(math.ceil(source_action.frame_range[1]))
    bpy.context.scene.frame_set(start)
    translation_source_name, translation_target_name = select_motion_translation_bones(
        set(source.pose.bones.keys()),
        mapping,
    )
    translation_origin = source.pose.bones[translation_source_name].location.copy()
    translation_scale = _motion_translation_scale(source, target, mapping)
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    log(f"Retargeting {len(mapping)} SOMA joints across {end - start + 1} frames")

    # World-delta retargeting (backend/app/retarget_math.py):
    #
    #   T_pose_w(b, f) = S_pose_w(s, f) . S_rest_w(s)^-1 . A_b . T_rest_w(b)
    #
    # A_b is the minimal rotation taking the target bone's rest DIRECTION
    # onto the source bone's rest direction (not its full orientation --
    # UniRig predicted bones roll differently than SOMA conventions, and
    # twist is not meaningfully transferable between unrelated riggers).
    #
    # Blender keyframes local `basis` quaternions, not world orientations:
    #   pose_w(b) = pose_w(parent) . rest_w(parent)^-1 . rest_w(b) . basis(b)
    # so every target bone -- mapped or not -- is walked in hierarchy
    # (depth) order, tracking each bone's assigned world-pose quaternion in
    # `pose_w`. Unmapped bones (including intermediates between two mapped
    # bones) get an identity local basis and are never keyframed; they are
    # only tracked here so mapped descendants inherit the right parent
    # chain. `pose_bone.matrix` is never read mid-frame for a TARGET bone --
    # it is not recomputed until a depsgraph update, so parent orientations
    # come from this dict instead.
    def _world_rest_quat(armature, pose_bone):
        return tuple((armature.matrix_world @ pose_bone.bone.matrix_local).to_quaternion())

    def _world_rest_dir(armature, pose_bone):
        bone = pose_bone.bone
        # tail_local - head_local is already armature-space; applying
        # matrix_local here would double-transform the direction.
        direction = (armature.matrix_world.to_3x3()
                     @ (bone.tail_local - bone.head_local))
        return direction.normalized()

    def _hierarchy_order(pose_bones):
        def depth(pose_bone):
            d = 0
            current = pose_bone
            while current.parent is not None:
                current = current.parent
                d += 1
            return d
        return sorted((b.name for b in pose_bones), key=lambda name: depth(pose_bones[name]))

    # When several source bones collapse onto the same target (the target
    # rig has fewer segments than SOMA, e.g. one "spine" bone, or no
    # clavicle) only the most distal source actually drives that target's
    # rotation each frame -- a deterministic replacement for the previous
    # arbitrary last-writer-wins.
    target_source = select_deepest_source_per_target(mapping)  # target name -> chosen source name
    collision_losers = sorted(set(mapping) - set(target_source.values()))
    if collision_losers:
        log(
            "Collision losers (SOMA source mapped but not driving any bone "
            f"-- a deeper source won the same target): {', '.join(collision_losers)}"
        )

    target_rest_w = {b.name: _world_rest_quat(target, b) for b in target.pose.bones}
    target_parent = {b.name: (b.parent.name if b.parent else None) for b in target.pose.bones}
    source_rest_w = {
        source_name: _world_rest_quat(source, source.pose.bones[source_name])
        for source_name in target_source.values()
    }

    # quat_between_vectors is only well-conditioned for a genuine rest-
    # drawing-convention difference (tens of degrees); anything past 90
    # degrees is a bone-mapping or rig-handedness fault (e.g. a mirrored
    # rig putting the physical right arm's bone along the left arm's
    # source direction) and must fail loudly instead of baking a
    # plausible-looking ~170-degree "correction".
    #
    # The pelvis role is exempt (K9 finding, 2026-08-30): its `align` is
    # hardcoded to quat_identity() a few dozen lines down regardless of
    # this angle -- s_dir/t_dir are never used to rotate the pelvis, only
    # to gate here -- so a bad angle for pelvis can never corrupt the bake,
    # only block it. And "bad angle" for pelvis is disproportionately
    # likely to be a false positive: Blender's BVH importer has no
    # well-defined tail for a joint with several children (Hips here has
    # three: one spine child pulling up, two upper-leg children pulling
    # down), so it produces some importer-internal average/heuristic
    # direction with no retargeting meaning -- confirmed by reproducing the
    # same ~165 degree Hips->pelvis mismatch on two unrelated target rigs
    # against the same NVIDIA Kimodo BVH source. Every other mapped role
    # keeps the hard gate: their `align` values are actually consumed
    # below, so a real 90+ degree fault there (mirrored rig, wrong bone
    # picked) must still fail loudly instead of baking silently-wrong
    # motion.
    rest_dirs = {}
    rest_mismatches = []
    for target_name, source_name in target_source.items():
        s_dir = _world_rest_dir(source, source.pose.bones[source_name])
        t_dir = _world_rest_dir(target, target.pose.bones[target_name])
        rest_dirs[target_name] = (s_dir, t_dir)
        angle = math.degrees(s_dir.angle(t_dir)) if s_dir.length and t_dir.length else 180.0
        if angle > 90.0 and SOMA_TO_ROLE.get(source_name) == "pelvis":
            log(
                f"rest-direction mismatch for {source_name}->{target_name} "
                f"({angle:.1f} deg) exceeds 90 degrees but role=pelvis always bakes "
                "with an identity alignment, so this is diagnostic only -- not failing"
            )
        elif angle > 90.0:
            rest_mismatches.append(f"{source_name}->{target_name} ({angle:.1f} deg)")
    if rest_mismatches:
        raise RuntimeError(
            "BVH retarget rest-direction mismatch exceeds 90 degrees for: "
            + ", ".join(rest_mismatches)
            + " -- this is a bone-mapping or rig-handedness fault, not a rest-convention "
            "difference, and must be fixed at the source (mapping/rig), not by widening "
            "this threshold"
        )

    # A_b's job is to absorb legitimate bone-drawing-convention differences
    # (a target rig's rest pose rarely matches SOMA's T-pose exactly), but
    # the torso is a special case: SOMA and a target rig can disagree by
    # 20-50 degrees on how the pelvis/spine/neck/head bones are drawn at
    # rest even when the rig itself is correct (see diagnose.json Hips-
    # >pelvis ~43-49 deg, Chest->spine_3 ~21-24 deg, Neck1->neck ~20 deg).
    # A full alignment there pitches the whole torso every single frame.
    # The pelvis/root anchors the body, so it keeps its own rest
    # orientation entirely; the spine/neck/head chain gets a clamped
    # correction so a real swing still transfers, capped well under what
    # would read as torso drift. Limbs (arms/legs) keep the full alignment
    # -- that is where the genuine per-character rest-pose differences
    # (T-pose vs. A-pose, etc.) actually need correcting.
    #
    # The clamp used to be 30 degrees, which is a no-op for every rig pair
    # measured so far: Chest->spine_3 (~23.6 deg) and Neck1->neck (~20.5
    # deg) rest-direction mismatches both sit comfortably UNDER 30, so
    # quat_clamp_angle passed `full_align` through unchanged, and torso/
    # head bones ended up with T_pose_w == the DONOR's own absolute world
    # direction every frame by construction (composition_error is exactly
    # 0 for an unclamped full alignment -- see diagnose_retarget.py) --
    # i.e. Mary's spine/neck/head copied the SOMA donor's own resting
    # slouch outright, not just a bounded nudge toward it. This is what
    # produced the reported torso hunch (Mary's chest pitch matched the
    # donor's chest pitch bit-for-bit at every sampled frame in
    # /tmp/k6/baseline/torso_pitch_chain.json).
    #
    # A per-bone "parent-relative" alignment (express each rig's bone rest
    # direction relative to its OWN pelvis rest orientation, so only the
    # residual bend beyond the shared pelvis-convention difference
    # transfers) was tried and rejected: on this rig pair it INCREASES the
    # correction angle for most torso bones instead of shrinking it (Head
    # 3.9 -> 52.5 deg, Spine1 10.9 -> 37.8 deg, Spine2 4.9 -> 43.7 deg,
    # Neck1 20.5 -> 28.1 deg; only Chest stayed roughly flat, 23.6 -> 25.0
    # deg) -- see /tmp/k6/validate_align.py and validate_align2.py. The
    # ~48.6-degree Hips->pelvis mismatch does not decompose into a single
    # coherent "whole-body tilt" that the per-segment spine/neck/head
    # mismatches share; each segment's own rest-drawing convention differs
    # independently, so subtracting the pelvis's own rest orientation (with
    # or without its own roll) does not isolate a genuine per-bone
    # difference -- it just recombines two independently-wrong offsets.
    #
    # Given that, the simpler, safer fix is a tighter bound: clamp to <=5
    # degrees. This is a strict improvement on every torso/head bone
    # measured (Chest 23.6->5, Neck1 20.5->5, Spine1 10.9->5 deg; Spine2
    # 4.9 and Head 3.9 deg were already under 5 and pass through
    # unclamped), so at most 5 degrees of the donor's resting convention
    # can bleed into Mary's own rest per bone, instead of the previous
    # full (up to 23.6 degree) copy.
    #
    # Limb alignment: `quat_between_vectors` only pins the bone's DIRECTION
    # (swing); the rotation ABOUT that axis is whatever its cross-product
    # construction happens to produce -- not derived from either rig's own
    # geometry. On a target with no separate clavicle bone (both
    # LeftShoulder and LeftArm collapse onto upper_arm_l; see
    # select_deepest_source_per_target above) that arbitrary residual roll
    # showed up as a CONSTANT twist through the whole arm every frame
    # (measured: LeftArm 15.4 deg, LeftForeArm 6.8 deg, LeftHand 3.3 deg,
    # frame-invariant -- see /tmp/k6/round2/arm_check_after.json).
    #
    # The natural fix is a two-vector ("frame") alignment
    # (`retarget_math.quat_align_frame`): swing the bone direction onto the
    # source's as before, then roll about that shared axis so a second
    # "hint" direction lines up too. The obvious hint is a mapped CHILD
    # bone's own rest direction (forearm for an upper arm, hand for a
    # forearm) -- but on this rig pair both the donor's and Mary's arms are
    # essentially straight at rest (T-pose), so the forearm-relative-to-
    # upper-arm deviation is ~0-2 degrees on both sides (degenerate, no
    # roll information), and even the next candidate down the chain --
    # hand-relative-to-forearm, ~6-9 degrees on both sides, individually
    # not degenerate -- turned out to be dominated by each rig's own
    # incidental rest-pose drawing noise, uncorrelated between the two
    # rigs: verified analytically (twist_deg is provably frame-invariant,
    # so it can be computed once from rest quaternions alone -- see
    # /tmp/k6/round3/predict_twist.py) that hand-hinting makes twist_deg
    # WORSE, not better (LeftForeArm 6.8 -> 85.5 deg predicted, LeftArm
    # 15.4 -> 84.2 deg predicted vs. the hand hint). Applying it would have
    # traded a small, bounded, already-passing-threshold error for a much
    # larger one on the strength of a barely-non-degenerate signal.
    #
    # `quat_align_frame` itself is a generic pure helper (retarget_math.py)
    # that takes any two hint vectors -- it does not require the hint to
    # come from a child bone. The reliable, always-available "roll"
    # reference every bone already carries is its OWN rest local-X axis
    # (each rig's own bone.roll convention, baked into its rest
    # quaternion): unlike a borrowed child direction, it is exactly
    # perpendicular to the bone's own direction by construction (an
    # orthonormal frame), so it can never be degenerate, and aligning it
    # is exactly "match the two rigs' full rest ORIENTATION, not just
    # direction" for that bone. Verified analytically the same way: this
    # drives twist_deg to ~0 degrees (not just under the 3-degree target)
    # for every limb bone -- arms, forearms, hands, thighs, shins, and feet
    # alike -- while leaving the existing direction/swing constraint (and
    # therefore position error) exactly unchanged (direction error stayed
    # 0.000000 deg to 6 decimal places in the same check). This is a
    # deliberate deviation from literally following the child-bone hint:
    # it reuses the same `quat_align_frame` primitive the task specified,
    # with a hint choice that is provably correct on real data instead of
    # one that measurably regresses it. It also naturally covers hands and
    # feet (no mapped child to hint from at all) with the same mechanism,
    # instead of leaving them on swing-only -- which matters here because
    # LeftHand's own swing-only twist (3.33 deg) was already marginally
    # over the requested <=3 degree bar.
    align = {}
    for target_name, source_name in target_source.items():
        role = SOMA_TO_ROLE.get(source_name)
        s_dir, t_dir = rest_dirs[target_name]
        full_align = retarget_math.quat_between_vectors(tuple(t_dir), tuple(s_dir))
        if role == "pelvis":
            align[target_name] = retarget_math.quat_identity()
        elif role in ("spine", "head"):
            align[target_name] = retarget_math.quat_clamp_angle(full_align, math.radians(5.0))
        else:
            # Limb roles (upper/lower arm, hand, upper/lower leg, foot):
            # frame-align using each bone's own rest local-X as the roll
            # hint, on both rigs -- see comment above.
            s_local_x = retarget_math.quat_rotate_vector(source_rest_w[source_name], (1.0, 0.0, 0.0))
            t_local_x = retarget_math.quat_rotate_vector(target_rest_w[target_name], (1.0, 0.0, 0.0))
            align[target_name] = retarget_math.quat_align_frame(
                tuple(t_dir), tuple(t_local_x), tuple(s_dir), tuple(s_local_x),
            )

    # Pre-flight: the world-delta transfer's hierarchy math (basis_from_world
    # / propagate_unmapped) assumes every bone actually inherits its
    # parent's rotation/scale and has no pose constraint fighting the
    # baked keyframes, and that matrix_world is a single uniform,
    # right-handed scale (a mirrored/negative-determinant object would
    # silently flip every transferred rotation). Fail loudly instead of
    # baking a plausible-looking but wrong action.
    matrix_world_3x3 = target.matrix_world.to_3x3()
    if matrix_world_3x3.determinant() <= 0:
        raise RuntimeError(
            f"target armature '{target.name}' matrix_world has a non-positive determinant "
            f"({matrix_world_3x3.determinant():.4f}) -- a mirrored/negative-scale object "
            "would silently flip every transferred rotation"
        )
    world_scale = target.matrix_world.to_scale()
    scale_values = (world_scale.x, world_scale.y, world_scale.z)
    scale_spread = max(scale_values) - min(scale_values)
    if scale_spread > 1e-4 * max(1e-6, max(abs(v) for v in scale_values)):
        raise RuntimeError(
            f"target armature '{target.name}' matrix_world has non-uniform scale "
            f"{tuple(round(v, 5) for v in scale_values)} -- the world-delta transfer "
            "assumes one uniform scale factor"
        )
    hierarchy_faults = []
    for bone in target.data.bones:
        if not bone.use_inherit_rotation:
            hierarchy_faults.append(f"{bone.name}: use_inherit_rotation=False")
        if bone.inherit_scale != "FULL":
            hierarchy_faults.append(f"{bone.name}: inherit_scale={bone.inherit_scale!r} (expected 'FULL')")
    for pose_bone in target.pose.bones:
        if len(pose_bone.constraints) > 0:
            constraint_types = [c.type for c in pose_bone.constraints]
            hierarchy_faults.append(f"{pose_bone.name}: has pose constraint(s) {constraint_types}")
    if hierarchy_faults:
        raise RuntimeError(
            "target armature has bones that break the world-delta transfer's rest(parent)^-1 "
            "* rest(b) hierarchy assumption: " + "; ".join(hierarchy_faults)
        )

    virtual_root = tuple(target.matrix_world.to_quaternion())
    order = _hierarchy_order(target.pose.bones)
    # Reset every target pose bone exactly once, up front: propagate_unmapped
    # assumes an identity local basis for every bone it is never asked to
    # set, and animation_data_clear() does not touch matrix_basis, so a GLB
    # imported with a baked (non-rest) first frame would otherwise leave
    # stale rotation/location/scale on unmapped bones for the whole bake.
    for pose_bone in target.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)

    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        source_pose_w = {
            source_name: tuple(
                (source.matrix_world @ source.pose.bones[source_name].matrix).to_quaternion()
            )
            for source_name in target_source.values()
        }
        pose_w = {}
        for bone_name in order:
            parent_name = target_parent[bone_name]
            parent_pose_w = pose_w[parent_name] if parent_name else virtual_root
            parent_rest_w = target_rest_w[parent_name] if parent_name else virtual_root
            if bone_name in target_source:
                source_name = target_source[bone_name]
                pose_w[bone_name] = retarget_math.desired_target_world_pose(
                    source_pose_w[source_name],
                    source_rest_w[source_name],
                    align[bone_name],
                    target_rest_w[bone_name],
                )
                basis = retarget_math.basis_from_world(
                    target_rest_w[bone_name], parent_rest_w, parent_pose_w, pose_w[bone_name],
                )
                pose_bone = target.pose.bones[bone_name]
                pose_bone.rotation_quaternion = Quaternion(basis)
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone_name)
            else:
                pose_w[bone_name] = retarget_math.propagate_unmapped(
                    parent_pose_w, parent_rest_w, target_rest_w[bone_name],
                )
        translation_bone = target.pose.bones[translation_target_name]
        translation_bone.location = _retarget_basis_translation(
            source,
            source.pose.bones[translation_source_name],
            target,
            translation_bone,
            source.pose.bones[translation_source_name].location - translation_origin,
        ) * translation_scale
        translation_bone.keyframe_insert(
            data_path="location", frame=frame, group=translation_target_name,
        )
    action.name = animation
    source.animation_data_clear()
    bpy.data.objects.remove(source, do_unlink=True)
    bpy.data.actions.remove(source_action)
    bpy.context.scene.frame_set(start)
    export_glb(output_path, animations=True)


_POSE_BONE_DATA_PATH = re.compile(r'^pose\.bones\["([^"]+)"\]')


def _animated_bone_names(action) -> list[str]:
    """Every bone name an action's fcurves target, by exact match on the
    `pose.bones["<name>"]...` data path convention this codebase's own
    keyframe_insert calls use (see retarget_bvh/slither_bake above)."""
    names: set[str] = set()
    for fcurve in action.fcurves:
        match = _POSE_BONE_DATA_PATH.match(fcurve.data_path)
        if match:
            names.add(match.group(1))
    return sorted(names)


def apply_clip(input_path: str, store_path: str, output_path: str, clip_name: str, family: str) -> None:
    """N5 direct-attach: assign a family clip-store action onto a
    template-conformant rig by exact bone-name match. No retarget math --
    this is the payoff of N2 (fixed skeleton contract) + N3 (rigs conform to
    the template) + N4 (clips pre-ported onto that same template): the rig
    and the clip store share identical bone names, so the store's action can
    be assigned to the target armature verbatim."""
    clear_scene()
    import_model(input_path)
    target_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(target_armatures) != 1:
        raise RuntimeError(f"rig import produced {len(target_armatures)} armatures (expected exactly 1)")
    target = target_armatures[0]
    target_bone_names = {bone.name for bone in target.data.bones}

    # Rest clip guard (mirrors retarget_bvh above): a certified rig GLB can
    # carry its own rest/idle action. Clear it -- and every action datablock
    # left over from the import -- before the clip store brings in the one
    # action we actually want, so nothing stray survives to export.
    if target.animation_data is not None:
        target.animation_data_clear()
    for existing_action in list(bpy.data.actions):
        bpy.data.actions.remove(existing_action)

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=store_path)
    store_objects = [obj for obj in bpy.context.scene.objects if obj not in before]

    action = bpy.data.actions.get(clip_name)
    if action is None:
        available = sorted(a.name for a in bpy.data.actions)
        raise RuntimeError(f"clip store has no action named {clip_name!r}; available: {available}")

    # s5.exact_bone_name_match (docs/PIPELINE_POLICY.md S5): every bone the
    # action animates must exist on the target rig by exact name --
    # direct-attach has no retarget math to paper over a mismatch. Every
    # clip in the store was baked on the FULL 66-bone canonical template
    # (S3), so it carries a channel for every one of those bones whether or
    # not a given rig instance actually has them -- a rig only needs the
    # ~21 REQUIRED bones to be template-conformant (S3: an OPTIONAL bone's
    # absence is not a conformance error). classify_missing_bones applies
    # that same required/optional split here: a clip touching an OPTIONAL
    # bone the rig lacks (a finger segment, a toe, head_tip) is skipped --
    # its channels are simply inert once the action is assigned, no error,
    # nothing visibly missing since the rig never had that bone to move --
    # a clip touching a REQUIRED bone the rig lacks is a hard fail (that
    # rig was never really template-conformant).
    animated_bones = _animated_bone_names(action)
    missing_bones = clip_store.missing_animated_bones(animated_bones, target_bone_names)
    if missing_bones:
        hard_fail, skipped = clip_store.classify_missing_bones(missing_bones, family)
        if hard_fail:
            raise RuntimeError(
                f"clip {clip_name!r} animates required bones the rig does not have "
                f"(direct-attach requires an exact bone-name match, "
                f"s5.exact_bone_name_match): {', '.join(hard_fail)}"
            )
        if skipped:
            log(
                f"Clip {clip_name!r} also animates {len(skipped)} optional template "
                f"bone(s) this rig doesn't carry (harmless, skipped): {', '.join(skipped)}"
            )

    # The store's own armature/mesh objects, and every action but the one
    # selected, must not survive to export.
    for obj in store_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for other_action in list(bpy.data.actions):
        if other_action is not action:
            bpy.data.actions.remove(other_action)

    if target.animation_data is None:
        target.animation_data_create()
    target.animation_data.action = action
    start = int(round(action.frame_range[0]))
    end = int(round(action.frame_range[1]))
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    bpy.context.scene.frame_set(start)
    log(
        f"Direct-attached {clip_name!r} ({end - start + 1} frames, "
        f"{len(animated_bones)} bones) -- no retarget"
    )
    export_glb(output_path, animations=True)


def _is_unbroken_parent_chain(ordered: list) -> bool:
    """True if `ordered` is exactly one parent->child path: each bone's
    parent is the immediately preceding bone in the list, nothing more,
    nothing less. Branches (spurs) hanging off a chain bone are fine --
    this only checks the chain bones themselves against each other."""
    for parent, child in zip(ordered, ordered[1:]):
        if child.parent is None or child.parent.name != parent.name:
            return False
    return True


def _chain_order_from_hierarchy(chain_bones: list, head_name: str):
    """Walk `chain_bones` from their single hierarchy root to `head_name`,
    following the chain's own parent/child links (ignoring bone names,
    only ever stepping to a child that is itself one of `chain_bones` --
    any spurs are simply not candidates). Returns `None` if the bones do
    not form exactly one such path ending at `head_name`."""
    names = {bone.name for bone in chain_bones}
    roots = [bone for bone in chain_bones if bone.parent is None or bone.parent.name not in names]
    if len(roots) != 1:
        return None
    ordered = [roots[0]]
    seen = {roots[0].name}
    current = roots[0]
    while current.name != head_name:
        children = [child for child in current.children if child.name in names]
        if len(children) != 1 or children[0].name in seen:
            return None
        current = children[0]
        seen.add(current.name)
        ordered.append(current)
    return ordered if len(ordered) == len(chain_bones) else None


def _chain_bone_order(armature) -> list:
    """Order one armature's bones tail (`spine_00`) to head (`head`) using
    this repo's chain-naming convention (`backend/app/skeleton_naming.py`),
    and verify the named bones actually form a single unbroken parent-child
    chain -- a name-sortable set alone is not enough: duplicate or
    non-contiguous spine indices, more than one 'head' bone, or a rig where
    the spine/head bones aren't hierarchically connected in that order
    would otherwise bake garbage motion silently (review F3-2). If the
    name-sorted order isn't itself an unbroken chain, the armature's own
    hierarchy (root to tip) is tried as a fallback before giving up.

    Raises ``RuntimeError`` if the armature is not a chain rig."""
    spine_bones: dict[int, "bpy.types.PoseBone"] = {}
    head_bone = None
    for bone in armature.pose.bones:
        normalized = bone.name.strip().casefold()
        match = _CHAIN_SPINE_NAME_PATTERN.match(normalized)
        if match:
            index = int(match.group(1))
            if index in spine_bones:
                raise RuntimeError(
                    f"'{armature.name}' has duplicate spine index {index} "
                    f"({spine_bones[index].name!r} and {bone.name!r})"
                )
            spine_bones[index] = bone
        elif normalized == "head":
            if head_bone is not None:
                raise RuntimeError(
                    f"'{armature.name}' has more than one 'head' bone "
                    f"({head_bone.name!r} and {bone.name!r})"
                )
            head_bone = bone
    if not spine_bones or head_bone is None:
        raise RuntimeError(
            f"'{armature.name}' is not a chain rig: found {len(spine_bones)} "
            f"spine_NN bone(s) and {'a' if head_bone else 'no'} head bone"
        )
    # Numbering may have gaps: the rig-adjust loop removes joints by name and
    # keeps the survivors' names (spine_10 -> spine_12), exactly like the
    # rig certificate's own `_order_chain_bones`. Sorted order plus the
    # unbroken parent-chain check below is what guarantees a real chain.
    ordered_indices = sorted(spine_bones)
    by_name_order = [spine_bones[i] for i in ordered_indices] + [head_bone]
    if _is_unbroken_parent_chain(by_name_order):
        return by_name_order
    # A head-rooted chain (UniRig roots serpents at the head: head ->
    # spine_18 -> ... -> spine_00) is the same unbroken path walked the other
    # way. Keep the tail-to-head name order the bake math expects; only the
    # hierarchy direction differs, and the certificate accepts both.
    if _is_unbroken_parent_chain(list(reversed(by_name_order))):
        return by_name_order
    rebuilt = _chain_order_from_hierarchy(by_name_order, head_bone.name)
    if rebuilt is None or not _is_unbroken_parent_chain(rebuilt):
        raise RuntimeError(
            f"'{armature.name}' chain bones do not form a single unbroken "
            "parent-child chain ending at 'head' (checked both name order "
            "and hierarchy)"
        )
    return rebuilt


_UP_AXIS_WORLD = Vector((0.0, 0.0, 1.0))
# A bend "about world-up" only makes sense for a bone whose own rest
# direction is closer to horizontal than vertical -- past this point the
# bend would really be twisting the bone about its own length instead of
# bending it sideways. cos(45 deg) ~= 0.707: a bone within 45 degrees of
# vertical is rejected (review F3-1).
_MAX_VERTICAL_COMPONENT = 0.7


def _world_rest_direction(armature, bone) -> Vector:
    """`bone`'s own rest direction (head -> tail), in world space."""
    direction = armature.matrix_world.to_3x3() @ (bone.bone.tail_local - bone.bone.head_local)
    if direction.length < 1e-9:
        raise RuntimeError(f"bone '{bone.name}' has zero rest length")
    return direction.normalized()


def _up_axis_in_bone_local_space(armature, bone) -> Vector:
    """World-up (Z), re-expressed in `bone`'s own local REST frame -- the
    axis to rotate `bone.rotation_quaternion` about so the bend reads as a
    rotation around world-up once composed through the armature/parent
    chain, regardless of the bone's own roll.

    `pose_bone.rotation_quaternion` is a LOCAL basis rotation relative to
    the bone's own rest orientation (`bone.bone.matrix_local`), not a world
    rotation. A chain bone's local Z is only world-up by coincidence --
    Blender's bone-roll convention varies bone to bone and rig to rig -- so
    a hardcoded local axis like (0, 0, 1) would corkscrew the body wherever
    roll is non-zero (review F3-1). Transforming world-up by the inverse of
    the bone's own world-space rest orientation gives the axis that, in
    THIS bone's local frame, actually points along world-up.
    """
    rest_quat = (armature.matrix_world @ bone.bone.matrix_local).to_quaternion()
    return rest_quat.inverted() @ _UP_AXIS_WORLD


def slither_bake(input_path: str, output_path: str, animation: str, settings: dict) -> None:
    """Bake a procedural travelling-sine 'Slither' cycle onto a chain-rigged
    creature (a serpent, tail, tentacle...). See backend/app/slither_math.py
    for the per-bone rotation math (including the head-to-tail travel
    direction and the anchored root bone); its tests in
    backend/tests/test_slither_math.py pin the op's behaviour.

    Each bone's bend is a LOCAL rotation about world-up, expressed in that
    bone's OWN rest frame (`_up_axis_in_bone_local_space`) rather than a
    fixed local axis -- see that helper's docstring for why a hardcoded
    axis would corkscrew the body. Every chain bone's rest direction is
    checked against `_MAX_VERTICAL_COMPONENT` before baking: a bone whose
    rest direction is closer to vertical than horizontal cannot be bent
    "about world-up" without actually twisting it about its own length.
    """
    settings = dict(settings or {})
    clear_scene()
    import_model(input_path)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    chain_candidates = []
    rejections = []
    for armature in armatures:
        try:
            chain_candidates.append((armature, _chain_bone_order(armature)))
        except RuntimeError as reason:
            rejections.append(f"{armature.name}: {reason}")
    if len(chain_candidates) != 1:
        raise RuntimeError(
            f"Slither bake requires exactly one chain-rigged armature; "
            + (f"rejected: {'; '.join(rejections)}; " if rejections else "")
            +             f"found {len(chain_candidates)} among {len(armatures)} armature(s)"
        )
    armature, bones = chain_candidates[0]

    if armature.animation_data is not None:
        armature.animation_data_clear()
    for existing_action in list(bpy.data.actions):
        bpy.data.actions.remove(existing_action)

    lengths = [float((bone.tail - bone.head).length) for bone in bones]
    positions = slither_math.arc_positions_from_lengths(lengths)
    params = slither_math.from_settings(settings)
    frames = int(settings.get("frames", 48))
    fps = int(settings.get("fps", 24))
    advance = float(settings.get("advance", 0.0))
    if frames < 2:
        raise RuntimeError("Slither bake requires at least 2 frames")
    if frames > MAX_DEFORMATION_REVIEW_FRAMES:
        raise RuntimeError(
            f"Slither bake supports at most {MAX_DEFORMATION_REVIEW_FRAMES} "
            "frames (the deformation-review gate's hard limit)"
        )

    # Rest transforms are static (they don't depend on the current pose),
    # so both the axis and the verticality check are computed once, up
    # front, rather than every frame.
    bone_axes: dict[str, Vector] = {}
    for bone in bones:
        direction = _world_rest_direction(armature, bone)
        if abs(direction.z) > _MAX_VERTICAL_COMPONENT:
            raise RuntimeError(
                f"chain bone '{bone.name}' rest direction "
                f"{tuple(round(v, 3) for v in direction)} is too close to "
                "vertical for a Slither bend about world-up -- it would "
                "twist this bone about its own length instead of bending "
                "it sideways"
            )
        bone_axes[bone.name] = _up_axis_in_bone_local_space(armature, bone)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 0
    scene.frame_end = frames - 1

    if armature.animation_data is None:
        armature.animation_data_create()
    action = bpy.data.actions.new(animation)
    armature.animation_data.action = action

    for bone in bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)

    root_bone = bones[0]
    root_rest_location = tuple(root_bone.location)

    log(f"Baking Slither: {frames} frames, {len(bones)} chain bones, settings={settings}")
    for frame in range(frames):
        t = frame / frames
        rotations = slither_math.chain_rotations(positions, t, params)
        scene.frame_set(frame)
        for bone, angle in zip(bones, rotations):
            axis = bone_axes[bone.name]
            quat = slither_math.bone_rotation_quaternion(angle, axis=(axis.x, axis.y, axis.z))
            bone.rotation_quaternion = Quaternion(quat)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
        if advance:
            # A true net-forward crawl can't be baked into a seamless loop
            # (frame 0 and the wrapped "frame N" would disagree on
            # position) without external root-motion extraction, which is
            # out of scope here -- so `advance` is a cyclic in-place
            # forward/back thrust synced to the gait cycle instead: it
            # returns to the rest offset at both t=0 and t=1 (review F3-3).
            offset = advance * math.sin(2.0 * math.pi * t)
            root_bone.location = Vector(root_rest_location) + Vector((0.0, offset, 0.0))
            root_bone.keyframe_insert(data_path="location", frame=frame, group=root_bone.name)

    action.name = animation
    scene.frame_set(0)
    export_glb(output_path, animations=True)
    log("Slither bake complete")


def _mesh_world_triangles(mesh_object) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle world-space vertex positions and normals, (n, 3, 3) each.

    Aligned with `_mesh_uv_triangles`'s row order -- both walk the same
    `mesh.loop_triangles`, so index `i` names the same triangle in the UV,
    position and normal arrays a caller zips together.
    """
    mesh = mesh_object.data
    mesh.calc_loop_triangles()
    # Blender 4.1+ computes corner normals lazily and dropped calc_normals_split.
    if hasattr(mesh, "calc_normals_split"):
        mesh.calc_normals_split()
    n_triangles = len(mesh.loop_triangles)
    if n_triangles == 0:
        return np.empty((0, 3, 3)), np.empty((0, 3, 3))
    vertex_indices = np.empty(n_triangles * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", vertex_indices)
    split_normals = np.empty(n_triangles * 9, dtype=np.float64)
    mesh.loop_triangles.foreach_get("split_normals", split_normals)

    local_positions = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local_positions)
    local_positions = local_positions.reshape(-1, 3)
    homogeneous = np.concatenate(
        [local_positions, np.ones((local_positions.shape[0], 1))], axis=1
    )
    matrix = np.asarray(mesh_object.matrix_world, dtype=np.float64)
    world_positions = (homogeneous @ matrix.T)[:, :3]
    triangle_world = world_positions[vertex_indices.reshape(-1, 3)]

    # Normals transform by the inverse-transpose of the rotation/scale block,
    # not by the matrix itself -- the standard fix for a non-uniform scale
    # otherwise tilting the normal away from the true surface direction.
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    triangle_local_normals = split_normals.reshape(-1, 3, 3)
    triangle_world_normals = triangle_local_normals @ normal_matrix.T
    lengths = np.linalg.norm(triangle_world_normals, axis=-1, keepdims=True)
    triangle_world_normals = triangle_world_normals / np.clip(lengths, 1e-9, None)
    return triangle_world, triangle_world_normals


def _base_color_image_node(material):
    """The TEX_IMAGE node wired to the Principled BSDF's Base Color input.

    Same wiring `_bake_source_appearance` builds (~760-777): Image Texture
    -> Base Color, found here rather than assumed so `project-views` can run
    against any canonicalized mesh, not only one baked in the same process.
    """
    if material is None or material.node_tree is None:
        return None
    bsdf = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None
    base_color_input = bsdf.inputs.get("Base Color")
    if base_color_input is None or not base_color_input.is_linked:
        return None
    source = base_color_input.links[0].from_node
    return source if source.type == "TEX_IMAGE" else None


def _alpha_silhouette_bbox(alpha: np.ndarray) -> tuple[float, float, float, float] | None:
    """Pixel bbox (min_x, min_y, max_x, max_y) of a render's non-transparent
    region, or None when the render is empty."""
    rows = np.where(alpha.max(axis=1) > 0.5)[0]
    cols = np.where(alpha.max(axis=0) > 0.5)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return (float(cols.min()), float(rows.min()), float(cols.max()) + 1.0, float(rows.max()) + 1.0)


# How many pixels a subject mask is eroded inward before it is trusted to
# sample colour from -- see `view_projection.erode_mask`'s docstring for why
# (imprecise keying/silhouette edges otherwise sample near-background pixels
# straight onto the atlas).
SUBJECT_MASK_ERODE_RADIUS_PX = 3

# Fixed per-role colour for the debug "winner" map (AISMITH_PROJECT_VIEWS_
# DEBUG) -- which role's view actually contributed the max weight at each
# atlas texel, at a glance.
_DEBUG_ROLE_COLORS = {
    "front": (1.0, 0.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (0.0, 0.0, 1.0),
    "right": (1.0, 1.0, 0.0),
}


def _project_views_debug_enabled() -> bool:
    return os.environ.get("AISMITH_PROJECT_VIEWS_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _downsample_block_mean(array: np.ndarray, factor: int = 4) -> np.ndarray:
    """Box-average downsample by `factor` along the first two axes, cropping
    to the nearest multiple of `factor` first. Debug-visualization only --
    every debug PNG is written at this reduced size to keep them small."""
    array = np.asarray(array, dtype=np.float64)
    height, width = array.shape[:2]
    cropped_height = (height // factor) * factor
    cropped_width = (width // factor) * factor
    if cropped_height == 0 or cropped_width == 0:
        return array
    cropped = array[:cropped_height, :cropped_width]
    if cropped.ndim == 2:
        reshaped = cropped.reshape(cropped_height // factor, factor, cropped_width // factor, factor)
        return reshaped.mean(axis=(1, 3))
    channels = cropped.shape[2]
    reshaped = cropped.reshape(cropped_height // factor, factor, cropped_width // factor, factor, channels)
    return reshaped.mean(axis=(1, 3))


def _write_debug_png(path: Path, array: np.ndarray) -> None:
    """Write a numpy array in [0, 1] -- (H, W) grayscale or (H, W, 3) RGB --
    as an 8-bit PNG via Blender's own image save, so no PIL dependency is
    needed. Debug-visualization only (AISMITH_PROJECT_VIEWS_DEBUG=1); never
    called on the normal projection path.

    No row flip: this module's own arrays are already bottom-up, the exact
    order `image.pixels` expects (see `view_projection.py`'s docstring), so
    `image.save()` writes a normally-oriented PNG file with no extra step.
    """
    array = np.clip(np.asarray(array, dtype=np.float64), 0.0, 1.0)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    height, width = array.shape[:2]
    rgba = np.concatenate([array, np.ones((height, width, 1))], axis=2)
    debug_image = bpy.data.images.new(f"AISmith Debug {path.stem}", width=width, height=height, alpha=True)
    try:
        debug_image.pixels.foreach_set(np.ascontiguousarray(rgba.reshape(-1), dtype=np.float32))
        debug_image.filepath_raw = str(path)
        debug_image.file_format = "PNG"
        debug_image.save()
    finally:
        bpy.data.images.remove(debug_image, do_unlink=True)


def _setup_compositor_depth_output(scene, render_dir: Path):
    """Wire a Depth-pass File Output sink into the scene compositor across
    the Blender <5.0 `scene.node_tree`/Composite-node API and the >=5.0
    node-group API (`scene.compositing_node_group`; the Composite node was
    removed in favour of a Group Output node whose first, Color-typed input
    replaces it -- `CompositorNodeOutputFile.base_path`/`file_slots` also
    became `directory`/`file_name`). Returns a `set_output_name(name)`
    closure so the per-role call site doesn't need to know which API is live.
    """
    scene.use_nodes = True
    if hasattr(scene, "node_tree"):
        tree = scene.node_tree
        tree.nodes.clear()
        render_layers = tree.nodes.new("CompositorNodeRLayers")
        composite = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
        depth_output = tree.nodes.new("CompositorNodeOutputFile")
        depth_output.base_path = str(render_dir)
        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_depth = "32"
        depth_output.format.exr_codec = "ZIP"
        tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
        return lambda name: setattr(depth_output.file_slots[0], "path", name)

    node_group = bpy.data.node_groups.new("AISmith Compositing", "CompositorNodeTree")
    scene.compositing_node_group = node_group
    node_group.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    group_output = node_group.nodes.new("NodeGroupOutput")
    render_layers = node_group.nodes.new("CompositorNodeRLayers")
    node_group.links.new(render_layers.outputs["Image"], group_output.inputs["Image"])
    depth_output = node_group.nodes.new("CompositorNodeOutputFile")
    depth_output.directory = str(render_dir)
    depth_output.format.media_type = "IMAGE"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    node_group.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    return lambda name: setattr(depth_output, "file_name", name)


def project_views(
    input_path: str,
    output_path: str,
    view_images: dict,
    report_path: str,
    settings: dict,
) -> None:
    """Bake the T-pose conditioning views onto the canonical mesh's own UV
    atlas, blended over its existing base-color bake (see
    `backend/app/view_projection.py`'s module docstring for why).

    For each role, this builds the SAME camera the numpy projection describes,
    renders a depth pass and the mesh's own rendered silhouette (for per-role
    frame alignment against the conditioning PNG's subject bbox), then hands
    positions/normals/depth/frame off to `view_projection` to score and blend
    every atlas texel. Texels no view reaches keep the existing bake
    untouched.

    A role's camera is the zero-tilt orthographic one its azimuth implies,
    unless `settings["cameras"][role]` supplies an explicit camera -- either
    an orthographic guide camera (`azimuth_degrees`/`center`/`ortho_scale`/
    `camera_distance`, task9) or, with `"kind": "perspective"`, a photo's own
    camera (`yaw_degrees`/`pitch_degrees`/`fov_x_degrees`/`center`/
    `camera_distance`, i.e. `source_fidelity.SourceCamera`'s fields; the
    source-pose chain). Either way an explicit camera means the view IS this
    render, so the frame is identity and the view's own pixel dimensions --
    which need not be square -- are the render resolution.
    """
    clear_scene()
    import_model(input_path)
    target = _join_meshes(selected_meshes())
    if target.data.uv_layers.active is None:
        raise RuntimeError("project-views requires an unwrapped mesh")

    material = target.active_material
    base_node = _base_color_image_node(material)
    if base_node is None or base_node.image is None:
        raise RuntimeError("project-views requires an existing Base Color image texture to blend over")
    image = base_node.image
    atlas_size = int(image.size[0])
    base_pixels = _read_image_pixels(image)
    fallback_rgb = base_pixels[:, :, :3].astype(np.float64)

    uv_triangles = _mesh_uv_triangles(target)
    world_triangles, normal_triangles = _mesh_world_triangles(target)
    positions, normals, covered = view_projection.rasterize_atlas(
        uv_triangles, world_triangles, normal_triangles, atlas_size,
    )

    low, high = _world_bounds([target])
    center = np.array([(low.x + high.x) / 2.0, (low.y + high.y) / 2.0, (low.z + high.z) / 2.0])
    diagonal = (high - low).length

    render_dir = Path(output_path).parent / "project-views-render"
    render_dir.mkdir(parents=True, exist_ok=True)

    # Debug mode (AISMITH_PROJECT_VIEWS_DEBUG=1): no behavior change to the
    # projection itself -- every debug artifact below is additional
    # inspection of numbers this function already computes, gated so the
    # normal path pays nothing for it.
    debug_enabled = _project_views_debug_enabled()
    debug_role_weights: dict = {}
    # A coarse "is this texel on the head" proxy for per-role diagnosis: the
    # top 20% of the mesh's own world-Z extent, independent of any camera.
    mesh_height = high.z - low.z
    head_proxy_z = low.z + 0.8 * mesh_height
    if debug_enabled:
        log(f"project-views: debug mode on -> writing artifacts to {render_dir}")

    scene = bpy.context.scene
    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(settings.get("cycles_samples", 4))
    scene.cycles.use_denoising = False
    # BOX, not the default Gaussian-ish filter: a wide reconstruction filter
    # blends the Z pass across a silhouette edge at low sample counts, which
    # is exactly the aliasing `view_texel_diagnostics` caught live on the GPU
    # box (left/right roles passing the depth test on well under 10% of
    # their facing-positive texels). A near-point filter keeps each pixel's
    # depth honestly belonging to one surface or the other.
    scene.cycles.pixel_filter_type = "BOX"
    scene.render.filter_size = 0.01
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.view_layers[0].use_pass_z = True
    # NOT a Viewer node: in --background mode the Viewer image is not driven
    # by any UI backdrop, so `bpy.data.images["Viewer Node"]` silently keeps
    # whatever stale buffer (often a default 256x256) it last had -- observed
    # live on the GPU box returning a 256x256 depth map against a 1024x1024
    # render. A File Output node actually writes the Depth pass to disk
    # every render regardless of any UI, which is what a headless op needs.
    set_depth_output_name = _setup_compositor_depth_output(scene, render_dir)

    camera_data = bpy.data.cameras.new("AISmith Projection Camera")
    camera = bpy.data.objects.new("AISmith Projection Camera", camera_data)
    scene.collection.objects.link(camera)
    # Type is set per role (a perspective photo camera is legal); HORIZONTAL
    # sensor fit for both kinds, matching what the numpy projection assumes --
    # `ortho_scale` and `fov_x` both measure across the image WIDTH. Identical
    # to Blender's AUTO default for the square renders every ortho role uses.
    camera_data.type = "ORTHO"
    camera_data.sensor_fit = "HORIZONTAL"
    scene.camera = camera

    # The camera sits this far from `center` along the role's `direction`
    # (see camera.location below) -- Blender's own Z/Depth pass measures
    # distance from THIS point, not from `center`, so every depth comparison
    # against `view_projection`'s (center-relative) analytic depth has to
    # know this same number to convert between the two reference frames.
    camera_distance = diagonal * 2.0
    # Explicit rather than Blender's defaults (0.1 / 1000): the mesh always
    # sits within [camera_distance - diagonal, camera_distance + diagonal]
    # of the camera along its view axis, by construction, for every role and
    # every mesh scale -- so this can never clip real geometry, and pins
    # down one more thing that could in principle differ between roles.
    camera_data.clip_start = max(1e-4, camera_distance - diagonal)
    camera_data.clip_end = camera_distance + diagonal

    per_role_coverage: dict = {}
    occlusion_rejected_per_role: dict = {}
    diagnostics_per_role: dict = {}
    frames: dict = {}
    view_colors: list = []
    view_weights: list = []
    processed_roles: list = []
    min_total_weight = float(settings.get("min_total_weight", 0.05))
    # Route-B projection-quality settings (bead s2-view-projection): front
    # priority, grazing-angle cutoff, side-view colour lock, and dilation
    # into unprojected texels -- see view_projection.py's DEFAULT_* constants
    # for the numeric defaults and why.
    min_facing = float(settings.get("min_facing", view_projection.DEFAULT_MIN_FACING))
    role_priority = {**view_projection.DEFAULT_ROLE_PRIORITY, **settings.get("role_priority", {})}
    # Off by default: with zero-tilt orthographic views the front/side overlap
    # IS the silhouette band, so the statistics match fits edge bleed and
    # lighting differences (right view offset -0.31 on 2026-09-05) and the
    # clip crushes dark detail. Opt-in until the overlap can be chosen better.
    colour_lock_enabled = bool(settings.get("colour_lock", False))
    fill_radius = int(settings.get("fill_radius", view_projection.DEFAULT_FILL_RADIUS))
    # Per-role head exclusion, e.g. {"front": 0.20}: that role's subject mask
    # loses its head box (view_projection.exclude_head), so the texels the
    # head would have painted keep the existing bake instead.
    exclude_head = {role: float(fraction) for role, fraction in dict(settings.get("exclude_head") or {}).items()}
    head_exclusions: dict[str, list[int]] = {}
    # Explicit per-role cameras (task9, mesh-guided view synthesis): a view
    # rendered FROM this exact camera (e.g. render-guide-views' guides.json)
    # skips both the mesh-bounds ortho-scale derivation and the silhouette
    # frame fit below -- the view already IS this render, pixel for pixel.
    cameras = dict(settings.get("cameras") or {})

    for role, image_path in view_images.items():
        has_explicit_camera = role in cameras
        if (
            role not in view_projection.ROLE_AZIMUTH_DEGREES
            and role not in view_projection.EXTRA_ROLE_AZIMUTH_DEGREES
            and not has_explicit_camera
        ):
            log(f"project-views: skipping unknown view role {role!r}")
            continue
        reference_pixels = _load_reference_pixels(str(image_path))
        reference_rgb = reference_pixels[:, :, :3].astype(np.float64)
        reference_alpha = (
            reference_pixels[:, :, 3].astype(np.float64) if reference_pixels.shape[2] == 4 else None
        )
        background_color = estimate_background_color(reference_rgb)
        # A photo is not necessarily square, so height and width are tracked
        # separately everywhere below (a square view reduces to today's path).
        image_height, image_width = int(reference_rgb.shape[0]), int(reference_rgb.shape[1])
        # Eroded ~3px (SUBJECT_MASK_ERODE_RADIUS_PX): a keying/alpha edge is
        # never pixel-perfect, and an un-eroded mask samples a handful of
        # near-background pixels right along a silhouette (coat/boot edges
        # in the live GPU-box render) -- eroding shrinks the "subject" a few
        # pixels inward so those edge pixels are never sampled at all.
        keyed_subject_mask = subject_pixel_mask(reference_rgb, reference_alpha)
        if keyed_subject_mask is None:
            log(f"project-views: view {role!r} has no keyable subject; will fall back to this role's own rendered silhouette")
            # Placeholder for the frame fit below ONLY -- overwritten with
            # the render's own silhouette (never "whole frame") once that
            # render exists, a few lines down.
            image_subject_bbox = (0.0, 0.0, float(image_width), float(image_height))
        else:
            keyed_subject_mask = view_projection.erode_mask(
                keyed_subject_mask, radius=SUBJECT_MASK_ERODE_RADIUS_PX
            )
            subject_ys, subject_xs = np.where(keyed_subject_mask)
            if subject_xs.size == 0:
                log(f"project-views: view {role!r} has an empty subject mask after erosion; skipping")
                continue
            image_subject_bbox = (
                float(subject_xs.min()), float(subject_ys.min()),
                float(subject_xs.max()) + 1.0, float(subject_ys.max()) + 1.0,
            )

        camera_spec = cameras.get(role)
        # An explicit camera IS the one this view was rendered from (NOT this
        # run's own mesh-bounds derivation) -- reusing it is the whole point:
        # it is what makes the view match the geometry.
        is_perspective = camera_spec is not None and str(camera_spec.get("kind", "orthographic")) == "perspective"
        if is_perspective:
            # A photo's camera: `SourceCamera`'s own fields, with pitch, so
            # the basis has to come from `tilted_view_basis` rather than an
            # azimuth alone (see view_projection.py's docstring).
            basis = view_projection.tilted_view_basis(
                role, float(camera_spec["yaw_degrees"]), float(camera_spec.get("pitch_degrees", 0.0)),
            )
            role_center = np.asarray(camera_spec["center"], dtype=np.float64)
            role_camera_distance = float(camera_spec["camera_distance"])
            projection_camera = view_projection.PerspectiveCamera(
                basis=basis, center=role_center, fov_x_degrees=float(camera_spec["fov_x_degrees"]),
                camera_distance=role_camera_distance, width=image_width, height=image_height,
            )
        else:
            azimuth_degrees = view_projection.resolve_role_azimuth(role, cameras)
            basis = view_projection.azimuth_view_basis(role, azimuth_degrees)
            if has_explicit_camera:
                role_center = np.asarray(camera_spec["center"], dtype=np.float64)
                ortho_scale = float(camera_spec["ortho_scale"])
                role_camera_distance = float(camera_spec["camera_distance"])
            else:
                role_center = center
                relative = world_triangles.reshape(-1, 3) - role_center
                extent_u = relative @ basis.right
                extent_v = relative @ basis.up
                max_extent = max(
                    float(extent_u.max() - extent_u.min()), float(extent_v.max() - extent_v.min())
                )
                ortho_scale = max(max_extent, diagonal * 1e-3) * 1.16
                role_camera_distance = camera_distance
            projection_camera = view_projection.OrthographicCamera(
                basis=basis, center=role_center, ortho_scale=ortho_scale,
                camera_distance=role_camera_distance, width=image_width, height=image_height,
            )
        # A fixed fraction of the bounding diagonal is too tight once the
        # atlas gets large relative to the mesh's actual triangle density
        # (2026-09-04 GPU-box: 2048px atlas over a ~30k-tri mesh) -- the
        # barycentric position error a single texel can carry grows with
        # atlas resolution, not with mesh scale. Floor the bias at roughly
        # 2 pixels' worth of world-space depth variation across the frame
        # (the camera's own units-per-pixel: `ortho_scale / width` for an
        # ortho camera, the frame width at the subject's FAR plane for a
        # perspective one -- half a diagonal beyond the centre, where a pixel
        # covers the most world).
        depth_bias = max(0.004 * diagonal, 2.0 * projection_camera.world_units_per_pixel(depth_offset=diagonal / 2.0))

        if is_perspective:
            camera_data.type = "PERSP"
            camera_data.lens_unit = "FOV"
            camera_data.angle_x = math.radians(projection_camera.fov_x_degrees)
        else:
            camera_data.type = "ORTHO"
            camera_data.ortho_scale = ortho_scale
        # Same placement rule for both kinds (and the same one
        # `render_source_view` uses for a `SourceCamera`).
        camera.location = Vector(tuple(role_center)) + Vector(tuple(basis.direction)) * role_camera_distance
        camera.rotation_euler = (Vector(tuple(role_center)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        # Per-role clip planes (not just the pre-loop default): an explicit
        # camera can sit at a different distance than this run's own
        # mesh-bounds derivation, so the clip range must be recomputed here.
        camera_data.clip_start = max(1e-4, role_camera_distance - diagonal)
        camera_data.clip_end = role_camera_distance + diagonal
        scene.render.resolution_x = image_width
        scene.render.resolution_y = image_height
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(render_dir / f"{role}.png")
        set_depth_output_name(f"{role}_depth")
        # Clear any stale depth EXR from a prior run at this path so the
        # glob below cannot pick up yesterday's file if this render fails
        # to write a fresh one.
        for stale in render_dir.glob(f"{role}_depth*.exr"):
            stale.unlink(missing_ok=True)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)

        render_pixels = _load_reference_pixels(scene.render.filepath)
        render_alpha = render_pixels[:, :, 3]
        # Which RENDER-space pixels actually hit geometry, in the exact same
        # (unframed) pixel space `depth_map` lives in -- fed to every depth
        # comparison below so a background pixel's depth sentinel can never
        # win the 3x3 neighborhood minimum (2026-09-05 GPU-box: it did, and
        # starved thin parts of the mesh -- arms, ears -- almost entirely).
        depth_map_valid = render_alpha > 0.5
        mesh_px_bbox = _alpha_silhouette_bbox(render_alpha)
        if mesh_px_bbox is None:
            log(f"project-views: view {role!r} rendered an empty silhouette; skipping")
            continue
        if has_explicit_camera:
            # This view WAS rendered from this exact camera at this size (a
            # guide render, or the photo itself) -- sample space already IS
            # render space, no fit needed. The depth-map shape check below
            # still raises a clear error if that ever stops being true.
            frame = (1.0, 0.0, 0.0)
        else:
            frame = view_projection.fit_frame_to_subject(mesh_px_bbox, image_subject_bbox)
        frames[role] = frame

        if keyed_subject_mask is not None:
            subject_mask = keyed_subject_mask
        else:
            # Keying failed on the conditioning PNG itself; fall back to
            # OUR OWN rendered silhouette (never "whole frame", which is
            # what let background pixels in along a silhouette on the
            # live GPU-box run for the 'right' role). `render_alpha` lives
            # in the render's own (unframed) pixel space, so it has to be
            # reprojected through the INVERSE of `frame` into the
            # conditioning image's pixel space -- the same space
            # `subject_mask` is indexed in everywhere else.
            inverse_scale, inverse_dx, inverse_dy = view_projection.invert_frame(*frame)
            conditioning_cols, conditioning_rows = np.meshgrid(
                np.arange(image_width, dtype=np.float64), np.arange(image_height, dtype=np.float64),
            )
            render_space_x, render_space_y = view_projection.apply_frame(
                conditioning_cols, conditioning_rows, inverse_scale, inverse_dx, inverse_dy,
            )
            resampled_alpha = view_projection.sample_bilinear(
                render_alpha[:, :, None], render_space_x, render_space_y,
            )[:, :, 0]
            subject_mask = view_projection.erode_mask(
                resampled_alpha > 0.5, radius=SUBJECT_MASK_ERODE_RADIUS_PX
            )
            log(f"project-views: view {role!r} subject mask built from the rendered silhouette")

        if role in exclude_head:
            subject_mask, head_box = view_projection.exclude_head(subject_mask, exclude_head[role])
            if head_box is not None:
                head_exclusions[role] = [int(v) for v in head_box]
                log(f"project-views: view {role!r} head box {head_box} excluded from projection")

        # NOT the compositor Viewer node: see the setup comment above.
        # Blender appends the frame number to the file-output slot's name
        # (e.g. "front_depth0001.exr"), so the exact filename is globbed
        # rather than assumed.
        depth_matches = sorted(render_dir.glob(f"{role}_depth*.exr"))
        if not depth_matches:
            raise RuntimeError(f"project-views: no depth EXR was written for view {role!r}")
        depth_image = bpy.data.images.load(str(depth_matches[-1]), check_existing=False)
        try:
            depth_image.colorspace_settings.name = "Non-Color"
            depth_values = np.empty(depth_image.size[0] * depth_image.size[1] * 4, dtype=np.float32)
            depth_image.pixels.foreach_get(depth_values)
            depth_map = depth_values.reshape(
                (depth_image.size[1], depth_image.size[0], 4)
            )[:, :, 0].astype(np.float64)
        finally:
            # Otherwise every role's depth EXR stays loaded as a bpy.data
            # datablock for the rest of the process.
            bpy.data.images.remove(depth_image, do_unlink=True)
        if depth_map.shape != (image_height, image_width):
            raise RuntimeError(
                f"project-views: depth map for view {role!r} is {depth_map.shape}, "
                f"expected ({image_height}, {image_width}) to match the render resolution"
            )

        weights, sample_px, sample_py = view_projection.view_texel_weights_for_camera(
            positions, normals, projection_camera,
            depth_map, depth_bias, subject_mask, frame,
            depth_map_valid=depth_map_valid, min_facing=min_facing,
        )
        weights = np.where(covered, weights, 0.0)
        weights_ignoring_depth, _, _ = view_projection.view_texel_weights_for_camera(
            positions, normals, projection_camera,
            depth_map, float("inf"), subject_mask, frame,
            depth_map_valid=depth_map_valid, min_facing=min_facing,
        )
        weights_ignoring_depth = np.where(covered, weights_ignoring_depth, 0.0)

        colors = view_projection.sample_bilinear(reference_rgb, sample_px, sample_py)
        if background_color is not None:
            # Scoped to a ring just inside the (already-eroded) subject
            # mask's own boundary -- NEVER the mask's deep interior. A
            # global colour guard here (2026-09-04, the previous fix)
            # flattened a pale-skinned alien's face: its skin sat close in
            # luminance to the light-gray conditioning background, so the
            # guard rejected genuine skin texels wherever they resembled
            # the backdrop, not just where they actually bled into it.
            bleed = view_projection.background_bleed_mask(
                colors, subject_mask, background_color, sample_px, sample_py,
            )
            weights = np.where(bleed, 0.0, weights)
        view_colors.append(colors)
        view_weights.append(weights)
        processed_roles.append(role)

        covered_count = float(covered.sum()) or 1.0
        per_role_coverage[role] = float((covered & (weights > 0.0)).sum()) / covered_count
        occlusion_rejected_per_role[role] = float(
            (covered & (weights_ignoring_depth > 0.0) & ~(weights > 0.0)).sum()
        ) / covered_count

        # Diagnostics (bead s2-view-projection GPU-box incident, 2026-09-04):
        # a full visibility funnel per role, independent of whether anything
        # actually projected, so "this view contributed nothing" is a number
        # to read off the log rather than a fresh investigation.
        diagnostics = view_projection.view_texel_diagnostics_for_camera(
            positions, normals, projection_camera,
            depth_map, depth_bias, subject_mask, frame, covered,
            depth_map_valid=depth_map_valid, min_facing=min_facing,
        )
        diagnostics["camera_clip_start"] = float(camera_data.clip_start)
        diagnostics["camera_clip_end"] = float(camera_data.clip_end)
        diagnostics_per_role[role] = diagnostics

        def _fmt(value):
            return f"{value:.4f}" if value is not None else "n/a"

        log(
            f"project-views: {role} diagnostics: "
            f"n_covered={diagnostics['n_covered']} "
            f"n_in_frame={diagnostics['n_in_frame']} "
            f"n_depth_visible={diagnostics['n_depth_visible']} "
            f"n_in_subject_mask={diagnostics['n_in_subject_mask']} "
            f"n_facing_positive={diagnostics['n_facing_positive']} "
            f"n_weight_positive={diagnostics['n_weight_positive']} "
            f"depth_bias={diagnostics['depth_bias']:.6f} "
            f"depth_map=[{_fmt(diagnostics['depth_map_min'])}, {_fmt(diagnostics['depth_map_max'])}] "
            f"texel_depth=[{_fmt(diagnostics['texel_depth_min'])}, {_fmt(diagnostics['texel_depth_max'])}] "
            f"clip=[{diagnostics['camera_clip_start']:.4f}, {diagnostics['camera_clip_end']:.4f}]"
        )

        if debug_enabled:
            debug_role_weights[role] = weights
            debug_masks = view_projection.view_texel_debug_masks_for_camera(
                positions, normals, projection_camera,
                depth_map, depth_bias, subject_mask, frame,
                depth_map_valid=depth_map_valid, min_facing=min_facing,
            )

            head_proxy = covered & (positions[:, :, 2] >= head_proxy_z)
            head_total = int(head_proxy.sum())

            def _head_pass(stage_mask: np.ndarray) -> int:
                return int((head_proxy & stage_mask).sum())

            log(
                f"project-views: {role} head-region proxy ({head_total} texels, "
                f"world z >= {head_proxy_z:.4f}): "
                f"in_frame={_head_pass(debug_masks['in_frame'])} "
                f"depth_visible={_head_pass(debug_masks['depth_visible'])} "
                f"in_mask={_head_pass(debug_masks['in_mask'])} "
                f"facing={_head_pass(debug_masks['facing_positive'])} "
                f"weight_positive={_head_pass(weights > 0.0)}"
            )

            masked_colors = np.where((weights > 0.0)[:, :, None], colors, 0.0)
            _write_debug_png(
                render_dir / f"{role}_weight.png", _downsample_block_mean(np.clip(weights, 0.0, 1.0)),
            )
            _write_debug_png(render_dir / f"{role}_colors.png", _downsample_block_mean(masked_colors))
            _write_debug_png(
                render_dir / f"{role}_mask.png", _downsample_block_mean(subject_mask.astype(np.float64)),
            )
            _write_debug_png(
                render_dir / f"{role}_depth_visible.png",
                _downsample_block_mean(debug_masks["depth_visible"].astype(np.float64)),
            )
            _write_debug_png(
                render_dir / f"{role}_facing.png",
                _downsample_block_mean(debug_masks["facing_positive"].astype(np.float64)),
            )

            # Per-texel numeric ground truth for the S2 projection conformance
            # fixture (task5) -- covered-texel subset only, keyed by
            # covered_index so positions/normals can be reconciled without
            # re-rasterizing the atlas.
            covered_flat_index = np.where(covered.reshape(-1))[0]
            np.savez_compressed(
                render_dir / f"{role}_texels.npz",
                covered_index=covered_flat_index,
                positions=positions.reshape(-1, 3)[covered_flat_index],
                normals=normals.reshape(-1, 3)[covered_flat_index],
                weights=weights.reshape(-1)[covered_flat_index],
                sample_rgb=colors.reshape(-1, 3)[covered_flat_index],
                in_subject_mask=debug_masks["in_mask"].reshape(-1)[covered_flat_index],
                depth=debug_masks["depth"][covered_flat_index],
                depth_at_render=debug_masks["depth_at_render"][covered_flat_index],
                render_px=debug_masks["render_px"][covered_flat_index],
                render_py=debug_masks["render_py"][covered_flat_index],
                render_ix=debug_masks["render_ix"][covered_flat_index],
                render_iy=debug_masks["render_iy"][covered_flat_index],
                sample_px=debug_masks["sample_px"][covered_flat_index],
                sample_py=debug_masks["sample_py"][covered_flat_index],
                facing=debug_masks["facing"][covered_flat_index],
                in_render_bounds=debug_masks["in_render_bounds"][covered_flat_index],
                within_depth=debug_masks["within_depth"][covered_flat_index],
            )
            # `depth_map` here is still the RAW EXR-loaded pass (the internal
            # camera_distance subtraction inside view_texel_weights binds a
            # new local name, it never mutates this array); `render_alpha` is
            # the render's own alpha channel, both in render-pixel space.
            np.save(render_dir / f"{role}_depth.npy", depth_map)
            np.save(render_dir / f"{role}_alpha.npy", render_alpha)
            role_meta = {
                "kind": "perspective" if is_perspective else "orthographic",
                "camera_distance": float(role_camera_distance),
                "depth_bias": float(depth_bias),
                "image_width": int(image_width),
                "image_height": int(image_height),
                "world_units_per_pixel": float(projection_camera.world_units_per_pixel()),
                "center": [float(role_center[0]), float(role_center[1]), float(role_center[2])],
                "diagonal": float(diagonal),
                "frame": {"scale": float(frame[0]), "dx": float(frame[1]), "dy": float(frame[2])},
                "role": role,
            }
            if is_perspective:
                role_meta["fov_x_degrees"] = float(projection_camera.fov_x_degrees)
                role_meta["yaw_degrees"] = float(camera_spec["yaw_degrees"])
                role_meta["pitch_degrees"] = float(camera_spec.get("pitch_degrees", 0.0))
            else:
                role_meta["ortho_scale"] = float(ortho_scale)
            (render_dir / f"{role}_meta.json").write_text(json.dumps(role_meta), encoding="utf-8")

    if debug_enabled and debug_role_weights:
        # Which role's view actually won each atlas texel -- fallback
        # (below min_total_weight, same threshold `blend_views` uses) is
        # black, so a texel with no winner reads identically to one that
        # was never covered.
        roster = sorted(debug_role_weights)
        stacked = np.stack([debug_role_weights[role] for role in roster], axis=0)
        winner_index = np.argmax(stacked, axis=0)
        winner_weight = np.max(stacked, axis=0)
        has_winner = winner_weight >= min_total_weight
        winner_rgb = np.zeros((atlas_size, atlas_size, 3))
        for index, role in enumerate(roster):
            color = _DEBUG_ROLE_COLORS.get(role, (1.0, 1.0, 1.0))
            winner_rgb[has_winner & (winner_index == index)] = color
        _write_debug_png(render_dir / "winner_map.png", _downsample_block_mean(winner_rgb))
        log(f"project-views: debug winner map written for roles {roster}")

    # Side-view colour lock: a side view's own colour drift (a brown-drifted
    # coat) otherwise pulls the front's blend toward it wherever they overlap
    # (2026-09-05 GPU-box finding). Front is always the reference, never
    # matched itself; declines silently (via match_view_colour_to_reference's
    # own dict) when front didn't contribute or overlap is too small.
    colour_lock_report: dict = {}
    if colour_lock_enabled and "front" in processed_roles and per_role_coverage.get("front", 0.0) > 0.0:
        front_index = processed_roles.index("front")
        front_colors, front_weights = view_colors[front_index], view_weights[front_index]
        for index, role in enumerate(processed_roles):
            if role == "front":
                continue
            matched, match_info = view_projection.match_view_colour_to_reference(
                view_colors[index], view_weights[index], front_colors, front_weights,
            )
            view_colors[index] = matched
            colour_lock_report[role] = match_info

    # Front-first priority: multiply each role's weights before blending so
    # an overlapping side view no longer weighs equally against the front
    # (the "lighter patch" finding) -- `blend_views`'s own signature stays a
    # plain weighted average. A role outside the four-role default table
    # (a 3/4 guide role, or any explicit-camera role) defaults to 0.6, not
    # 1.0: it is extra coverage, not one of the primary four views.
    for index, role in enumerate(processed_roles):
        view_weights[index] = view_weights[index] * role_priority.get(role, 0.6)

    if view_colors:
        blended_rgb, projected_mask = view_projection.blend_views(
            view_colors, view_weights, fallback_rgb, min_total_weight=min_total_weight,
        )
    else:
        blended_rgb, projected_mask = fallback_rgb, np.zeros_like(covered)

    delta_mask = covered & projected_mask
    delta_oklab_mean = view_projection.color_delta_oklab(blended_rgb, fallback_rgb, delta_mask)

    # Dilate the projected region into the atlas' remaining unprojected
    # texels rather than leaving them the reconstruction's blurred fallback
    # bake (only ~30% of the atlas is ever projected, 2026-09-05 finding).
    dilated_rgb, filled_mask = view_projection.dilate_projected(
        blended_rgb, projected_mask, fallback_rgb, radius=fill_radius, covered=covered,
    )
    covered_count = float(covered.sum()) or 1.0
    filled_texel_fraction = float((filled_mask & covered).sum()) / covered_count
    textured_texel_fraction = float(((projected_mask | filled_mask) & covered).sum()) / covered_count

    if debug_enabled:
        # Once per run, not per role: the atlas-shaped mask `projection_report`
        # is given, and the final (dilated) atlas in the same bottom-up
        # orientation `_read_image_pixels` reads. `projected_mask` stays the
        # pre-dilation mask on disk -- `filled_mask` is the new one.
        np.save(render_dir / "projected_mask.npy", projected_mask)
        np.save(render_dir / "filled_mask.npy", filled_mask)
        np.save(render_dir / "atlas_out.npy", dilated_rgb)

    blended_pixels = np.concatenate([dilated_rgb, base_pixels[:, :, 3:4]], axis=-1)
    image.pixels.foreach_set(np.ascontiguousarray(blended_pixels.reshape(-1), dtype=np.float32))
    image.update()
    _fill_bake_misses(image, max_passes=64)
    image.pack()

    report = view_projection.projection_report(
        per_role_coverage, projected_mask, covered, occlusion_rejected_per_role,
        delta_oklab_mean, atlas_size, frames,
    )
    report["diagnostics"] = diagnostics_per_role
    report["min_facing"] = min_facing
    report["role_priority"] = dict(role_priority)
    report["colour_lock"] = colour_lock_report
    # Which processed roles used an explicit camera (task9) rather than this
    # run's own mesh-bounds derivation -- those roles' frame_alignment above
    # is identity by construction, not a coincidentally-good fit.
    report["cameras_explicit"] = sorted(role for role in processed_roles if role in cameras)
    report["head_exclusions"] = head_exclusions
    report["filled_texel_fraction"] = round(filled_texel_fraction, 6)
    report["textured_texel_fraction"] = round(textured_texel_fraction, 6)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report), encoding="utf-8")
    log(
        f"project-views: {len(report['views_used'])} view(s) used, "
        f"projected_texel_fraction={report['projected_texel_fraction']:.4f} "
        f"textured_texel_fraction={report['textured_texel_fraction']:.4f}"
    )

    export_glb(output_path, animations=False, y_up=True)


# Per-texel colour keyed on dominant world-normal axis, for the S2
# project-views conformance fixture (task5): a wrong-face or mirrored
# projection then reads as a wrong colour, not just a blur.
_FIXTURE_FACE_COLORS: dict[tuple[int, int], tuple[str, tuple[float, float, float]]] = {
    (0, 1): ("+X", (0.9, 0.15, 0.15)),
    (0, -1): ("-X", (0.15, 0.85, 0.85)),
    (1, 1): ("+Y", (0.15, 0.85, 0.15)),
    (1, -1): ("-Y", (0.85, 0.15, 0.85)),
    (2, 1): ("+Z", (0.15, 0.25, 0.9)),
    (2, -1): ("-Z", (0.9, 0.85, 0.15)),
}


def _add_solid_box(location: tuple, dims: tuple, name: str):
    """A box of world-space dimensions `dims` centred at `location` --
    `primitive_cube_add`'s own `size` is an edge length (size=1 spans
    -0.5..0.5), so a unit cube scaled by `dims` has exactly those dimensions;
    `_join_meshes` (via `bpy.ops.object.join`) bakes the scale/location into
    the combined mesh correctly regardless of when it's applied."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dims
    return obj


def build_projection_fixture(output_glb: str, atlas_png: str, meta_json: str, atlas_size: int = 512) -> None:
    """Build the S2 project-views conformance fixture (task5): one mesh --
    torso + head + two THIN arm plates (on purpose: a convex cube+sphere
    alone cannot exercise the 2026-09-04 thin-part depth-visibility
    regression) + a nose so front/back are distinguishable -- with its own
    UV atlas coloured by dominant world-normal axis.
    """
    clear_scene()
    _add_solid_box((0.0, 0.0, 0.6), (0.4, 0.2, 1.2), "Torso")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.18, location=(0.0, 0.0, 1.45), segments=32, ring_count=16,
    )
    bpy.context.active_object.name = "Head"
    # Asymmetric in X on purpose (unequal arms, off-centre nose): an
    # X-symmetric fixture cannot tell a horizontally mirrored front/back view
    # from a correct one -- the mirror bug fixed on 2026-09-05 passed
    # front/back and only failed the side views until this asymmetry.
    _add_solid_box((0.5, 0.0, 1.0), (0.6, 0.05, 0.03), "ArmRight")
    _add_solid_box((-0.375, 0.0, 1.0), (0.35, 0.05, 0.03), "ArmLeft")
    _add_solid_box((0.06, -0.20, 1.45), (0.06, 0.08, 0.06), "Nose")

    target = _join_meshes(selected_meshes())
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.uv.pack_islands(margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    uv_triangles = _mesh_uv_triangles(target)
    world_triangles, normal_triangles = _mesh_world_triangles(target)
    _positions, normals, covered = view_projection.rasterize_atlas(
        uv_triangles, world_triangles, normal_triangles, atlas_size,
    )

    colors = np.full((atlas_size, atlas_size, 3), 0.5, dtype=np.float64)
    axis = np.argmax(np.abs(normals), axis=-1)
    picked = np.take_along_axis(normals, axis[..., None], axis=-1)[..., 0]
    sign = np.where(picked >= 0.0, 1, -1)
    for (axis_index, axis_sign), (_label, rgb) in _FIXTURE_FACE_COLORS.items():
        colors[covered & (axis == axis_index) & (sign == axis_sign)] = rgb

    image = bpy.data.images.new("AISmith Fixture Atlas", width=atlas_size, height=atlas_size, alpha=True)
    rgba = np.concatenate([colors, np.ones((atlas_size, atlas_size, 1))], axis=-1)
    image.pixels.foreach_set(np.ascontiguousarray(rgba.reshape(-1), dtype=np.float32))
    image.update()
    image.pack()
    Path(atlas_png).parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(atlas_png)
    image.file_format = "PNG"
    image.save()

    material = bpy.data.materials.new("AISmith Fixture Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    # Self-emissive so a rendered pixel IS the atlas colour (render-role-views
    # uses a black world): Principled under a lit world washes grazing sphere
    # texels toward white (Fresnel), which tests the lighting, not the
    # projection. Base Color stays wired because project_views requires it.
    links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    target.data.materials.clear()
    target.data.materials.append(material)

    low, high = _world_bounds([target])
    diagonal = (high - low).length
    export_glb(output_glb, animations=False)

    Path(meta_json).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_json).write_text(
        json.dumps({
            "bounds_low": [float(low.x), float(low.y), float(low.z)],
            "bounds_high": [float(high.x), float(high.y), float(high.z)],
            "diagonal": float(diagonal),
            "atlas_size": int(atlas_size),
            "face_colours": {label: list(rgb) for label, rgb in _FIXTURE_FACE_COLORS.values()},
        }),
        encoding="utf-8",
    )


def render_role_views(
    input_glb: str,
    output_dir: str,
    roles: list,
    size: int = 512,
    projection: str = "orthographic",
    fov_x_degrees: float | None = None,
    pitch_degrees: float = 0.0,
) -> None:
    """Render each role's conditioning view of `input_glb` for the S2
    project-views conformance fixture (task5), with the SAME camera basis
    `project_views` builds but a DIFFERENT framing margin (1.30, vs the op's
    1.16) -- so the conformance test proves `fit_frame_to_subject` actually
    re-fits rather than assuming the two frames already agree.

    `projection="perspective"` (with `fov_x_degrees`, and optionally a
    non-zero `pitch_degrees`) renders the same roles from a PHOTO-shaped
    camera instead -- the source-pose chain's case, where the view being
    projected back is a perspective image. That framing margin is applied to
    the subject's extent at the plane through `center`, so the near surface
    still lands inside the frame. Either way `views.json` carries each role's
    camera in exactly the shape `project_views`' `settings["cameras"]` takes,
    so the conformance test can hand it straight back.
    """
    clear_scene()
    import_model(input_glb)
    target = _join_meshes(selected_meshes())

    low, high = _world_bounds([target])
    center = np.array([(low.x + high.x) / 2.0, (low.y + high.y) / 2.0, (low.z + high.z) / 2.0])
    diagonal = (high - low).length
    world_triangles, _ = _mesh_world_triangles(target)
    relative_all = world_triangles.reshape(-1, 3) - center

    # Black world: the fixture is self-emissive, so every rendered pixel is
    # exactly its atlas colour -- the computable answer the colour assertion
    # compares against.
    world = bpy.context.scene.world or bpy.data.worlds.new("AISmith Fixture World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    is_perspective = projection == "perspective"
    if is_perspective and fov_x_degrees is None:
        raise RuntimeError("render-role-views: a perspective projection needs fov_x_degrees")

    camera_data = bpy.data.cameras.new("AISmith Fixture Camera")
    camera = bpy.data.objects.new("AISmith Fixture Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    # Same camera model the projection assumes: `ortho_scale` / `fov_x` both
    # measure across the image WIDTH (see project_views).
    camera_data.sensor_fit = "HORIZONTAL"
    if is_perspective:
        camera_data.type = "PERSP"
        camera_data.lens_unit = "FOV"
        camera_data.angle_x = math.radians(float(fov_x_degrees))
    else:
        camera_data.type = "ORTHO"
    scene = bpy.context.scene
    scene.camera = camera

    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for role in roles:
        azimuth_degrees = view_projection.resolve_role_azimuth(role, None)
        basis = (
            view_projection.tilted_view_basis(role, azimuth_degrees, pitch_degrees)
            if is_perspective
            else view_projection.ortho_view_basis(role)
        )
        extent_u = relative_all @ basis.right
        extent_v = relative_all @ basis.up
        max_extent = max(float(extent_u.max() - extent_u.min()), float(extent_v.max() - extent_v.min()))
        framed_extent = max_extent * 1.30

        if is_perspective:
            # Half the framed extent at the centre plane subtends half the
            # horizontal field of view.
            camera_distance = (framed_extent / 2.0) / math.tan(math.radians(float(fov_x_degrees)) / 2.0)
        else:
            camera_distance = diagonal * 2.0
            camera_data.ortho_scale = framed_extent
        camera_data.clip_start = max(1e-4, camera_distance - diagonal)
        camera_data.clip_end = camera_distance + diagonal

        camera.location = Vector(tuple(center)) + Vector(tuple(basis.direction)) * camera_distance
        camera.rotation_euler = (Vector(tuple(center)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        path = output_root / f"{role}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        entry = {
            "path": str(path),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "camera_distance": float(camera_distance),
        }
        if is_perspective:
            entry.update({
                "kind": "perspective",
                "yaw_degrees": float(azimuth_degrees),
                "pitch_degrees": float(pitch_degrees),
                "fov_x_degrees": float(fov_x_degrees),
            })
        else:
            entry.update({"kind": "orthographic", "ortho_scale": float(framed_extent)})
        manifest[role] = entry

    Path(output_root / "views.json").write_text(json.dumps(manifest), encoding="utf-8")


def _guide_normal_material() -> "bpy.types.Material":
    """Emission material whose colour IS the world-space normal mapped to
    [0, 1] -- a readable normal map, independent of any lighting (task9)."""
    material = bpy.data.materials.new("AISmith Guide Normal")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    geometry_node = nodes.new("ShaderNodeNewGeometry")
    scale_node = nodes.new("ShaderNodeVectorMath")
    scale_node.operation = "MULTIPLY"
    scale_node.inputs[1].default_value = (0.5, 0.5, 0.5)
    offset_node = nodes.new("ShaderNodeVectorMath")
    offset_node.operation = "ADD"
    offset_node.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission_node = nodes.new("ShaderNodeEmission")
    output_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry_node.outputs["Normal"], scale_node.inputs[0])
    links.new(scale_node.outputs["Vector"], offset_node.inputs[0])
    links.new(offset_node.outputs["Vector"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    return material


def render_guide_views(
    input_glb: str, output_dir: str, roles: list, extra_roles: list, size: int = 1024,
) -> None:
    """Render CLAY + NORMAL guide images per role from exact cameras (task9,
    mesh-guided view synthesis): each view is rendered FROM this mesh, so the
    resulting image and the camera `project_views` can later reuse for it
    (`guides.json`) always match the geometry exactly -- unlike a
    conditioning view drawn independently, which even a 4-view reconstructor
    cannot satisfy (left/right IoU 0.43/0.62 was the finding this replaces).

    Same per-role camera `project_views` builds for that role -- zero tilt,
    `direction` from the azimuth, `camera_distance = diagonal * 2.0`,
    `ortho_scale` from the mesh's own extent in view -- so a later
    `project-views` run can hand this exact camera back via
    `settings["cameras"]` and skip re-deriving it.
    """
    clear_scene()
    import_model(input_glb)
    target = _join_meshes(selected_meshes())

    low, high = _world_bounds([target])
    center = np.array([(low.x + high.x) / 2.0, (low.y + high.y) / 2.0, (low.z + high.z) / 2.0])
    diagonal = (high - low).length
    world_triangles, _ = _mesh_world_triangles(target)
    relative_all = world_triangles.reshape(-1, 3) - center

    scene = bpy.context.scene
    _configure_render_device(scene, prefer=os.environ.get("AISMITH_BLENDER_DEVICE", "auto"))
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    # Standard, not the default Filmic/AgX: those tonemap pixel values, which
    # would corrupt the normal pass's raw [0, 1]-mapped vectors as much as it
    # would the clay pass's shading.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    camera_data = bpy.data.cameras.new("AISmith Guide Camera")
    camera = bpy.data.objects.new("AISmith Guide Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    scene.camera = camera

    camera_distance = diagonal * 2.0
    camera_data.clip_start = max(1e-4, camera_distance - diagonal)
    camera_data.clip_end = camera_distance + diagonal

    sun_data = bpy.data.lights.new("AISmith Guide Sun", type="SUN")
    sun_data.energy = 2.5
    sun = bpy.data.objects.new("AISmith Guide Sun", sun_data)
    scene.collection.objects.link(sun)

    clay_world = bpy.data.worlds.new("AISmith Guide Clay World")
    clay_world.use_nodes = True
    clay_world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    clay_world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

    normal_world = bpy.data.worlds.new("AISmith Guide Normal World")
    normal_world.use_nodes = True
    normal_world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    normal_world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    clay_material = bpy.data.materials.new("AISmith Guide Clay")
    clay_material.use_nodes = True
    clay_bsdf = next(n for n in clay_material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    clay_bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    clay_bsdf.inputs["Roughness"].default_value = 0.8

    normal_material = _guide_normal_material()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for role in list(roles) + list(extra_roles):
        azimuth_degrees = view_projection.resolve_role_azimuth(role, None)
        basis = view_projection.azimuth_view_basis(role, azimuth_degrees)
        extent_u = relative_all @ basis.right
        extent_v = relative_all @ basis.up
        max_extent = max(float(extent_u.max() - extent_u.min()), float(extent_v.max() - extent_v.min()))
        ortho_scale = max(max_extent, diagonal * 1e-3) * 1.16

        camera.location = Vector(tuple(center)) + Vector(tuple(basis.direction)) * camera_distance
        camera.rotation_euler = (Vector(tuple(center)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.ortho_scale = ortho_scale
        # The sun follows the camera around the mesh (azimuth + 35 deg) so
        # every role gets comparably lit, readable shading rather than one
        # role backlit and another flatly front-lit.
        sun.rotation_euler = (math.radians(55.0), 0.0, math.radians(azimuth_degrees + 35.0))
        bpy.context.view_layer.update()

        clay_path = output_root / f"{role}_clay.png"
        target.data.materials.clear()
        target.data.materials.append(clay_material)
        scene.world = clay_world
        scene.cycles.samples = 32
        scene.cycles.use_denoising = True
        scene.render.filepath = str(clay_path)
        bpy.ops.render.render(write_still=True)

        normal_path = output_root / f"{role}_normal.png"
        target.data.materials.clear()
        target.data.materials.append(normal_material)
        scene.world = normal_world
        scene.cycles.samples = 8
        scene.cycles.use_denoising = False
        scene.render.filepath = str(normal_path)
        bpy.ops.render.render(write_still=True)

        manifest[role] = {
            "azimuth_degrees": float(azimuth_degrees),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "ortho_scale": float(ortho_scale),
            "camera_distance": float(camera_distance),
            "clay": str(clay_path),
            "normal": str(normal_path),
        }

    guides = {
        "image_size": int(size),
        "roles": manifest,
        "bounds_low": [float(low.x), float(low.y), float(low.z)],
        "bounds_high": [float(high.x), float(high.y), float(high.z)],
        "diagonal": float(diagonal),
    }
    Path(output_root / "guides.json").write_text(json.dumps(guides), encoding="utf-8")


def main() -> None:
    spec = json.loads(sys.argv[sys.argv.index("--") + 1])
    op = spec["op"]
    if op == "convert":
        convert(spec["input"], spec["output"])
    elif op == "export-obj":
        export_obj(spec["input"], spec["output"])
    elif op == "retopo-bake":
        retopo_bake(
            spec["source"],
            spec["target"],
            spec["output"],
            bool(spec.get("bake_appearance", True)),
            spec.get("report"),
            spec.get("palette_reference"),
            float(spec.get("palette_strength", 1.0)),
            spec.get("atlas_minimum"),
        )
    elif op == "normalize-floor":
        normalize_floor(spec["input"], spec["output"], spec["report"])
    elif op == "voxel-remesh-canonicalize":
        voxel_remesh_canonicalize(
            spec["input"],
            spec["output"],
            spec["report"],
            int(spec["target_faces"]),
            int(spec["raw_face_min"]),
            int(spec["raw_face_max"]),
        )
    elif op == "project-views":
        project_views(
            spec["input"],
            spec["output"],
            dict(spec.get("views", {})),
            spec["report"],
            dict(spec.get("settings") or {}),
        )
    elif op == "canonicalize-similarity-render":
        render_canonicalize_similarity_views(
            spec["source"],
            spec["canonical"],
            spec["output_dir"],
            spec["output"],
            list(spec.get("views", [])),
            int(spec.get("size", 256)),
            float(spec.get("face_fraction", 0.30)),
        )
    elif op == "render-source-view":
        render_source_view(spec["input"], spec["output"], spec["report"], dict(spec.get("framing") or {}), int(spec.get("samples", 32)))
    elif op == "paint":
        paint(spec["input"], spec["output"], spec.get("settings", {}))
    elif op == "repair-weights":
        repair_weights(spec["input"], spec["output"])
    elif op == "rig-transfer":
        transfer_rig(spec["input"], spec["donor"], spec["output"])
    elif op == "target-proxy-rig":
        target_proxy_rig(
            spec["input"],
            spec["donor"],
            spec["output"],
            spec.get("action_name", "Sword_Attack"),
            bool(spec.get("attach_sword", True)),
            spec.get("deformation_surface", "visible-original"),
        )
    elif op == "render-review":
        render_review_frame(
            spec["input"],
            spec["output"],
            float(spec.get("frame_ratio", 0.5)),
            rest_pose=bool(spec.get("rest_pose", False)),
            focus_bone=spec.get("focus_bone"),
        )
    elif op == "deformation-diagnostics":
        deformation_diagnostics(spec["input"], spec["output"])
    elif op == "render-skeleton-sheet":
        render_skeleton_sheet(spec["input"], spec["output_dir"], int(spec.get("size", 768)))
    elif op == "normalize-bvh":
        normalize_bvh_axes(spec["input"], spec["output"])
    elif op == "retarget-bvh":
        retarget_bvh(
            spec["input"],
            spec["motion"],
            spec["output"],
            spec.get("animation", "Sword_Attack"),
            spec.get("target_bones", {}),
        )
    elif op == "slither-bake":
        slither_bake(
            spec["input"],
            spec["output"],
            spec.get("animation", "Slither"),
            spec.get("settings", {}),
        )
    elif op == "apply-clip":
        apply_clip(spec["input"], spec["store"], spec["output"], spec["clip_name"], spec["family"])
    elif op == "build-projection-fixture":
        build_projection_fixture(
            spec["output"], spec["atlas_png"], spec["meta_json"], int(spec.get("atlas_size", 512)),
        )
    elif op == "render-role-views":
        fov_x_degrees = spec.get("fov_x_degrees")
        render_role_views(
            spec["input"], spec["output_dir"], list(spec.get("roles", [])), int(spec.get("size", 512)),
            projection=str(spec.get("projection", "orthographic")),
            fov_x_degrees=None if fov_x_degrees is None else float(fov_x_degrees),
            pitch_degrees=float(spec.get("pitch_degrees", 0.0)),
        )
    elif op == "render-guide-views":
        render_guide_views(
            spec["input"],
            spec["output_dir"],
            list(spec.get("roles", [])),
            list(spec.get("extra_roles", [])),
            int(spec.get("size", 1024)),
        )
    else:
        raise RuntimeError(f"Unknown Blender worker operation: {op}")
    log("Blender export complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
        sys.exit(1)
