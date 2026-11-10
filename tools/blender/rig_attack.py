"""Fit the vendored Mesh2Motion human rig, bind weights, and export Sword Attack.

Usage: blender -b --python rig_attack.py -- input.glb animations.glb output.glb
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[index] for point in corners) for index in range(3))),
        Vector(tuple(max(point[index] for point in corners) for index in range(3))),
    )


def main() -> None:
    source, animations, destination = map(Path, sys.argv[sys.argv.index("--") + 1:][:3])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    target_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not target_meshes:
        raise RuntimeError("Rigging source contains no mesh")
    target_min, target_max = world_bounds(target_meshes)

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(animations))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    armature = next((obj for obj in imported if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError("Animation library contains no humanoid armature")

    bone_points = [armature.matrix_world @ point for bone in armature.data.bones for point in (bone.head_local, bone.tail_local)]
    rig_min = Vector(tuple(min(point[index] for point in bone_points) for index in range(3)))
    rig_max = Vector(tuple(max(point[index] for point in bone_points) for index in range(3)))
    scale = (target_max.z - target_min.z) / max(rig_max.z - rig_min.z, 1e-6)
    armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    fitted_points = [armature.matrix_world @ point for bone in armature.data.bones for point in (bone.head_local, bone.tail_local)]
    fitted_min = Vector(tuple(min(point[index] for point in fitted_points) for index in range(3)))
    fitted_max = Vector(tuple(max(point[index] for point in fitted_points) for index in range(3)))
    armature.location += (target_min + target_max) / 2 - (fitted_min + fitted_max) / 2
    bpy.context.view_layer.update()

    deform_bones = [bone for bone in armature.data.bones if "leaf" not in bone.name and bone.name != "root"]
    for mesh in target_meshes:
        coordinates = np.asarray([vertex.co[:] for vertex in mesh.data.vertices], dtype=np.float32)
        scores = np.empty((len(coordinates), len(deform_bones)), dtype=np.float32)
        world_to_mesh = mesh.matrix_world.inverted()
        for index, bone in enumerate(deform_bones):
            head = np.asarray((world_to_mesh @ (armature.matrix_world @ bone.head_local))[:], dtype=np.float32)
            tail = np.asarray((world_to_mesh @ (armature.matrix_world @ bone.tail_local))[:], dtype=np.float32)
            segment = tail - head
            length_squared = max(float(np.dot(segment, segment)), 1e-8)
            projection = np.clip(((coordinates - head) @ segment) / length_squared, 0.0, 1.0)
            closest = head + projection[:, None] * segment
            distance = np.linalg.norm(coordinates - closest, axis=1)
            radius = max(length_squared ** 0.5 * 0.45, 0.025)
            scores[:, index] = distance / radius
        nearest = np.argpartition(scores, 4, axis=1)[:, :4]
        selected_scores = np.take_along_axis(scores, nearest, axis=1)
        weights = np.exp(-2.0 * selected_scores ** 2)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        groups = [mesh.vertex_groups.new(name=bone.name) for bone in deform_bones]
        for vertex_index in range(len(coordinates)):
            for slot in range(4):
                groups[int(nearest[vertex_index, slot])].add(
                    [vertex_index], float(weights[vertex_index, slot]), "REPLACE"
                )
        modifier = mesh.modifiers.new("Humanoid Armature", "ARMATURE")
        modifier.object = armature
        mesh.parent = armature
        mesh.matrix_parent_inverse = armature.matrix_world.inverted()
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and obj not in target_meshes:
            bpy.data.objects.remove(obj, do_unlink=True)

    attack = bpy.data.actions.get("Sword_Attack_Armature")
    if attack is None:
        raise RuntimeError("Sword_Attack_Armature is missing from the animation library")
    attack.name = "Attack"
    armature.animation_data_create()
    armature.animation_data.action = attack
    for action in list(bpy.data.actions):
        if action != attack:
            bpy.data.actions.remove(action)
    start, end = map(int, attack.frame_range)
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    bpy.context.scene.render.fps = 30

    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for mesh in target_meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(destination),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_frame_range=True,
        export_skins=True,
        export_morph=True,
    )
    if not destination.is_file() or destination.stat().st_size < 64:
        raise RuntimeError("Rigging export did not produce a GLB")


if __name__ == "__main__":
    main()
