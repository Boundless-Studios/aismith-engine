"""Print armature and skin-weight diagnostics for a GLB in Blender."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise SystemExit("usage: blender --background --python scripts/rig_diagnostics.py -- MODEL.glb")

    model_path = Path(argv[0]).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model_path))

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    payload: dict[str, object] = {
        "path": str(model_path),
        "actions": [{"name": action.name, "frame_range": list(action.frame_range)} for action in bpy.data.actions],
        "armatures": [
            {
                "name": armature.name,
                "bones": [bone.name for bone in armature.data.bones],
            }
            for armature in armatures
        ],
        "meshes": [],
    }

    mesh_payloads: list[dict[str, object]] = []
    for obj in meshes:
        world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        dominant_groups: Counter[str] = Counter()
        for vertex in obj.data.vertices:
            if not vertex.groups:
                dominant_groups["<unweighted>"] += 1
                continue
            group = max(vertex.groups, key=lambda assignment: assignment.weight)
            dominant_groups[obj.vertex_groups[group.group].name] += 1
        mesh_payloads.append(
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "bounds": {
                    "min": [min(vertex[index] for vertex in world_vertices) for index in range(3)],
                    "max": [max(vertex[index] for vertex in world_vertices) for index in range(3)],
                },
                "dominant_groups": dominant_groups.most_common(),
                "modifiers": [{"name": modifier.name, "type": modifier.type} for modifier in obj.modifiers],
            }
        )
    payload["meshes"] = mesh_payloads
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
