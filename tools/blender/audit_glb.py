"""Audit a GLB inside Blender: blender -b --python audit_glb.py -- in.glb out.json."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy


def main() -> None:
    source, destination = map(Path, sys.argv[sys.argv.index("--") + 1:][:2])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in meshes:
        obj.data.calc_loop_triangles()
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    coordinates = [value for obj in meshes for vertex in obj.data.vertices for value in obj.matrix_world @ vertex.co]
    materials = {material.name for obj in meshes for material in obj.data.materials if material}
    images = {node.image.name for material in bpy.data.materials for node in material.node_tree.nodes if material.use_nodes and node.type == "TEX_IMAGE" and node.image}
    color_attributes = {attribute.name for obj in meshes for attribute in obj.data.color_attributes}
    actions = [{"name": action.name, "frame_range": list(action.frame_range)} for action in bpy.data.actions]
    bounds = None
    if coordinates:
        bounds = {"min": [min(coordinates[index::3]) for index in range(3)], "max": [max(coordinates[index::3]) for index in range(3)]}
    report = {
        "source": str(source),
        "meshes": len(meshes),
        "mesh_names": [obj.name for obj in meshes],
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "triangles": sum(len(obj.data.loop_triangles) for obj in meshes),
        "materials": len(materials),
        "textures": len(images),
        "texture_names": sorted(images),
        "color_attributes": sorted(color_attributes),
        "finite_coordinates": all(math.isfinite(value) for value in coordinates),
        "bounds": bounds,
        "armatures": len(armatures),
        "bones": sum(len(obj.data.bones) for obj in armatures),
        "skinned_meshes": sum(any(modifier.type == "ARMATURE" for modifier in obj.modifiers) for obj in meshes),
        "vertex_groups": sum(len(obj.vertex_groups) for obj in meshes),
        "actions": actions,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
