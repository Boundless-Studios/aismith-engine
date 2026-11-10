"""Render deterministic review views: blender -b --python render_stage.py -- in.glb output-dir."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def aim(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:]
    source, output = map(Path, arguments[:2])
    review_frame = int(arguments[2]) if len(arguments) > 2 else None
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = 512 if review_frame is not None else 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Review World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.065, 0.09, 1.0)
    background.inputs["Strength"].default_value = 0.55
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[index] for point in corners) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in corners) for index in range(3)))
    center = (minimum + maximum) / 2
    radius = max((maximum - minimum).length * 0.75, 1.0)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(maximum.z - minimum.z, maximum.x - minimum.x) * 1.18
    scene.camera = camera
    for location, energy, size in (
        (center + Vector((-radius, -radius, radius)), 900.0, radius),
        (center + Vector((radius, -radius * 0.4, radius * 0.5)), 650.0, radius),
        (center + Vector((0, radius, radius)), 750.0, radius),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        aim(light, center)
    if review_frame is not None:
        scene.frame_set(review_frame)
    views = ((0, f"frame-{review_frame:03d}"),) if review_frame is not None else ((0, "front"), (45, "three-quarter"), (90, "side"), (180, "back"))
    for rotation, name in views:
        angle = math.radians(rotation)
        camera.location = center + Vector((math.sin(angle) * radius, -math.cos(angle) * radius, radius * 0.08))
        aim(camera, center)
        scene.render.filepath = str(output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    if review_frame is None:
        for obj in meshes:
            obj.display_type = "WIRE"
        camera.location = center + Vector((radius * 0.7, -radius * 0.7, radius * 0.12))
        aim(camera, center)
        scene.render.filepath = str(output / "wireframe.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
