"""Print right-hand pose-bone transforms from an animated GLB in Blender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.proxy_rig import select_action_name


def _arguments() -> tuple[str, str | None]:
    try:
        separator = sys.argv.index("--")
        arguments = sys.argv[separator + 1:]
        return arguments[0], arguments[1] if len(arguments) > 1 else None
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: blender --background --python inspect-action-hand.py -- FILE.glb [ACTION]"
        ) from exc


def main() -> None:
    input_path, requested_action = _arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=input_path)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit(f"expected one armature, found {len(armatures)}")
    armature = armatures[0]
    actions = sorted(bpy.data.actions, key=lambda action: action.name)
    if requested_action is not None:
        try:
            selected_name = select_action_name(
                (action.name for action in actions),
                requested_action,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        animated = next(action for action in actions if action.name == selected_name)
    else:
        animated = max(
            actions,
            key=lambda action: action.frame_range[1] - action.frame_range[0],
        )
    armature.animation_data_create().action = animated
    finger_tokens = ("thumb", "index", "middle", "ring", "pinky", "hand_r")
    bones = [
        bone
        for bone in armature.pose.bones
        if any(token in bone.name.casefold() for token in finger_tokens)
        and (bone.name.casefold().endswith("_r") or "hand_r" in bone.name.casefold())
    ]
    mesh_groups = []
    finger_bones = [
        bone for bone in bones
        if bone.name != "hand_r" and "leaf" not in bone.name.casefold()
    ]
    finger_segments = [
        (
            np.asarray(
                armature.matrix_world @ bone.bone.head_local,
                dtype=np.float64,
            ),
            np.asarray(
                armature.matrix_world @ bone.bone.tail_local,
                dtype=np.float64,
            ),
        )
        for bone in finger_bones
    ]
    for mesh in (obj for obj in bpy.context.scene.objects if obj.type == "MESH"):
        relevant = {
            group.index: group.name
            for group in mesh.vertex_groups
            if any(token in group.name.casefold() for token in finger_tokens)
            and group.name.casefold().endswith("_r")
        }
        weighted_vertices = {name: 0 for name in relevant.values()}
        weight_totals = {name: 0.0 for name in relevant.values()}
        for vertex in mesh.data.vertices:
            for element in vertex.groups:
                if element.group in relevant and element.weight > 0:
                    name = relevant[element.group]
                    weighted_vertices[name] += 1
                    weight_totals[name] += element.weight
        hand_group = mesh.vertex_groups.get("hand_r")
        hand_distances = []
        if hand_group is not None and finger_segments:
            for vertex in mesh.data.vertices:
                hand_weight = next(
                    (
                        element.weight
                        for element in vertex.groups
                        if element.group == hand_group.index
                    ),
                    0.0,
                )
                if hand_weight < 0.5:
                    continue
                point = np.asarray(mesh.matrix_world @ vertex.co, dtype=np.float64)
                distances = []
                for start, end in finger_segments:
                    segment = end - start
                    factor = np.clip(
                        np.dot(point - start, segment) / max(np.dot(segment, segment), 1e-12),
                        0.0,
                        1.0,
                    )
                    distances.append(float(np.linalg.norm(point - (start + factor * segment))))
                hand_distances.append(min(distances))
        mesh_groups.append({
            "mesh": mesh.name,
            "vertex_count": len(mesh.data.vertices),
            "weighted_vertices": weighted_vertices,
            "weight_totals": {
                name: round(total, 6)
                for name, total in weight_totals.items()
            },
            "dominant_hand_to_finger_distance": (
                {
                    "samples": len(hand_distances),
                    "p25": round(float(np.percentile(hand_distances, 25)), 6),
                    "p50": round(float(np.percentile(hand_distances, 50)), 6),
                    "p75": round(float(np.percentile(hand_distances, 75)), 6),
                    "p95": round(float(np.percentile(hand_distances, 95)), 6),
                }
                if hand_distances
                else None
            ),
        })
    frames = sorted({round(animated.frame_range[0]), round(sum(animated.frame_range) / 2), round(animated.frame_range[1])})
    payload = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        payload.append({
            "frame": frame,
            "bones": [
                {
                    "name": bone.name,
                    "rotation_mode": bone.rotation_mode,
                    "rotation_quaternion": [round(value, 6) for value in bone.rotation_quaternion],
                    "rotation_euler": [round(value, 6) for value in bone.rotation_euler],
                }
                for bone in bones
            ],
        })
    print("AISMITH_HAND_DIAGNOSTIC=" + json.dumps({
        "action": animated.name,
        "frame_range": list(animated.frame_range),
        "mesh_groups": mesh_groups,
        "samples": payload,
    }))


if __name__ == "__main__":
    main()
