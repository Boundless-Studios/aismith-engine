"""Print material, image, and UV diagnostics for one or more GLBs in Blender."""

import json
import sys

import bpy
import numpy as np


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def base_color_image(material):
    nodes = list(material.node_tree.nodes) if material and material.use_nodes and material.node_tree else []
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    base = principled.inputs.get("Base Color") if principled else None
    texture = base.links[0].from_node if base and base.is_linked else None
    return texture.image if texture and texture.type == "TEX_IMAGE" else None


def sample_material_slot(mesh, material_index: int, loop_indices: list[int]) -> dict | None:
    if material_index >= len(mesh.material_slots):
        return None
    material = mesh.material_slots[material_index].material
    image = base_color_image(material)
    if not image or not image.size[0] or not image.size[1] or not loop_indices:
        return None
    values = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    rgba = values.reshape((image.size[1], image.size[0], 4))
    uv_data = mesh.data.uv_layers.active.data
    coordinates = np.asarray([uv_data[index].uv for index in loop_indices])
    columns = np.clip(
        (coordinates[:, 0] * (image.size[0] - 1)).round().astype(int),
        0,
        image.size[0] - 1,
    )
    rows = np.clip(
        (coordinates[:, 1] * (image.size[1] - 1)).round().astype(int),
        0,
        image.size[1] - 1,
    )
    samples = rgba[rows, columns]
    sample_count = len(loop_indices)
    return {
        "material_index": material_index,
        "material": material.name if material else None,
        "image": image.name,
        "sample_count": sample_count,
        "sampled_base_near_black_fraction": float(
            np.mean(np.max(samples[:, :3], axis=1) < 0.03)
        ),
        "sampled_base_transparent_fraction": float(np.mean(samples[:, 3] < 0.5)),
        "unique_base_texel_fraction": float(
            len(np.unique(rows * image.size[0] + columns)) / sample_count
        ),
    }


def weighted_metric(reports: list[dict], key: str) -> float | None:
    sample_count = sum(report["sample_count"] for report in reports)
    if sample_count == 0:
        return None
    return float(sum(
        report[key] * report["sample_count"]
        for report in reports
    ) / sample_count)


def inspect(path: str) -> dict:
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    materials = []
    for material in bpy.data.materials:
        if material.users == 0:
            continue
        nodes = list(material.node_tree.nodes) if material.use_nodes and material.node_tree else []
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        base = principled.inputs.get("Base Color") if principled else None
        materials.append({
            "name": material.name,
            "users": material.users,
            "node_types": [node.type for node in nodes],
            "has_principled": principled is not None,
            "base_linked": bool(base and base.is_linked),
            "surface_render_method": material.surface_render_method,
        })
    images = []
    for image in bpy.data.images:
        if image.users == 0 or image.size[0] == 0 or image.size[1] == 0:
            continue
        values = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        rgba = values.reshape((-1, 4))
        images.append({
            "name": image.name,
            "size": list(image.size),
            "users": image.users,
            "alpha_zero_fraction": float(np.mean(rgba[:, 3] < 0.5)),
            "near_black_fraction": float(np.mean(np.max(rgba[:, :3], axis=1) < 0.03)),
            "rgb_mean": [float(value) for value in np.mean(rgba[:, :3], axis=0)],
        })
    mesh_reports = []
    for mesh in meshes:
        material_samples = []
        if mesh.data.uv_layers.active and mesh.material_slots:
            loops_by_material = {}
            for polygon in mesh.data.polygons:
                loops_by_material.setdefault(polygon.material_index, []).extend(
                    polygon.loop_indices
                )
            material_samples = [
                report
                for material_index, loop_indices in sorted(loops_by_material.items())
                if (report := sample_material_slot(mesh, material_index, loop_indices))
            ]
        mesh_reports.append({
            "name": mesh.name,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "materials": [slot.material.name if slot.material else None for slot in mesh.material_slots],
            "uv_layers": len(mesh.data.uv_layers),
            "material_samples": material_samples,
            "sampled_base_near_black_fraction": weighted_metric(
                material_samples, "sampled_base_near_black_fraction"
            ),
            "sampled_base_transparent_fraction": weighted_metric(
                material_samples, "sampled_base_transparent_fraction"
            ),
            "unique_base_texel_fraction": weighted_metric(
                material_samples, "unique_base_texel_fraction"
            ),
        })
    return {
        "path": path,
        "meshes": mesh_reports,
        "materials": materials,
        "images": images,
    }


paths = sys.argv[sys.argv.index("--") + 1:]
print("AISMITH_DIAGNOSTIC=" + json.dumps([inspect(path) for path in paths]))
