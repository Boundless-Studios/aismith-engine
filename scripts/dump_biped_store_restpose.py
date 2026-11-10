"""Reimport a produced biped clip-store GLB fresh (a clean Blender session,
independent of the conversion run) and dump its rest-pose bone data
(names/parents/world head&tail/roll) plus its action roster, for
verification against assets/skeletons/biped/template.json.

The dump is pure data; the actual conformance check runs afterwards, out of
Blender, via backend.app.skeleton_template.conformance_report (see
scripts/verify_biped_clip_store.py).

Usage:
    blender -b -P scripts/dump_biped_store_restpose.py -- <store_glb> <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
store_path, out_json = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
# disable_bone_shape=True: without it, Blender's importer links a small
# icosphere widget into a special "glTF_not_exported" collection to use as
# a pose-bone custom display shape (bone_heuristic="BLENDER", the default
# and correct choice for reimporting a Blender-authored export like this
# store -- see io_scene_gltf2's armature_display()). It's a pure UI
# convenience synthesized at import time, excluded from any re-export, and
# not part of the store's own glTF data (confirmed: the store's `meshes`
# array is empty) -- but it would otherwise show up here as a MESH object
# and confuse the "armature + animations only, no mesh" check below.
result = bpy.ops.import_scene.gltf(filepath=store_path, disable_bone_shape=True)
if result != {"FINISHED"}:
    raise SystemExit(f"gltf import failed: {result}")

armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if len(armatures) != 1:
    raise SystemExit(f"expected exactly one armature in the store, found {len(armatures)}")
armature = armatures[0]

mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]

names: list[str] = []
parents: list[str | None] = []
heads: list[list[float]] = []
tails: list[list[float]] = []
for bone in armature.data.bones:
    names.append(bone.name)
    parents.append(bone.parent.name if bone.parent else None)
    head = armature.matrix_world @ bone.head_local
    tail = armature.matrix_world @ bone.tail_local
    heads.append([head.x, head.y, head.z])
    tails.append([tail.x, tail.y, tail.z])

# Roll isn't exposed on the runtime Bone -- read it via a scratch EditBone pass.
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
roll_by_name = {eb.name: eb.roll for eb in armature.data.edit_bones}
bpy.ops.object.mode_set(mode="OBJECT")
rolls = [roll_by_name[n] for n in names]

actions = sorted(a.name for a in bpy.data.actions)

dump = {
    "store_path": store_path,
    "mesh_object_count": len(mesh_objects),
    "mesh_object_names": [o.name for o in mesh_objects],
    "bone_count": len(names),
    "names": names,
    "parents": parents,
    "heads": heads,
    "tails": tails,
    "rolls": rolls,
    "action_count": len(actions),
    "actions": actions,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(dump, indent=2), encoding="utf-8")
print("DUMP_OK", out_json, "bones", len(names), "actions", len(actions), "mesh_objects", len(mesh_objects))
