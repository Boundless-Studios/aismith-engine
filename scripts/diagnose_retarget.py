"""Retarget diagnostic: the acceptance instrument for the BVH retarget pipeline.

Usage:
    blender --background --python scripts/diagnose_retarget.py -- \
        <retargeted_or_rest.glb> <motion.bvh> <outdir> [--frames 1,25%,50%,75%]

`<retargeted_or_rest.glb>` may be a rig at rest (no motion baked in yet -- to
sanity-check the BVH axis import / mapping / rest directions before
attempting a retarget) or the animated GLB produced by
`backend/workers/blender_worker.py::retarget_bvh` (to score the transfer).

Outputs (all under <outdir>):
    diagnose.json         per-mapped-bone REST direction table (source vs
                          target bone direction, world space).
    bones.json            target armature's rest bone table: name, parent,
                          head, tail, length (world space).
    composition_error.json  per-frame, per-bone world-DIRECTION angle error.
                          NOT acceptance evidence on its own (analytically
                          ~0 for a direction-copying transfer) -- it proves
                          the bake/export/reimport round-trip is faithful.
    pose_error.json       the real acceptance metrics: (a) per-frame joint
                          POSITION error, hips-relative and divided by each
                          skeleton's own hips-to-head height (fraction of
                          body height), max/mean per bone + overall +
                          frame-1; (b) twist/roll error in degrees (angle
                          between source/target bone-local X after
                          removing the swing about the bone axis), max/mean
                          per bone + overall; (c) foot contact (each
                          skeleton's lowest foot point as a fraction of its
                          own height, per frame, plus max/mean |delta|).
                          Thresholds: position <= 3% of height on frame 1,
                          twist <= 20 deg.
    frame-<n>.png         stick-figure renders at the requested --frames:
                          source red / target (posed) green / target's OWN
                          rest pose blue (drift reference), offset -0.6 /
                          +0.6 / +1.8 on X, camera framing all three bodies.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
parser = argparse.ArgumentParser()
parser.add_argument("character_glb")
parser.add_argument("motion_bvh")
parser.add_argument("outdir")
parser.add_argument("--frames", default="1,25%,50%,75%")
args = parser.parse_args(argv)

os.makedirs(args.outdir, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "backend"))

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the BVH FIRST, purely to establish the scene's fps from the clip's
# native frame rate (update_scene_fps=True below) before importing the
# character GLB. Blender's glTF importer converts each action's stored
# keyframe TIMES into frame numbers using whatever scene fps is active at
# import time; the retargeted GLB was baked and exported at the BVH's own
# (~30.0003, not Blender's 24 default) fps, so importing the GLB first would
# silently rescale every keyframe's frame number (e.g. a 1..47 clip lands on
# 0.8..37.6) and desync every "same frame number" comparison below.
before = set(bpy.context.scene.objects)
# export_soma_bvh (backend/workers/unirig_bind_worker.py) writes BVH files
# with Blender's own exporter, which has no axis-conversion options and
# therefore always writes Blender's Z-up convention. This MUST be the
# identity mapping to match -- see backend/workers/blender_worker.py::retarget_bvh.
bpy.ops.import_anim.bvh(
    filepath=args.motion_bvh,
    axis_forward="Y",
    axis_up="Z",
    frame_start=1,
    use_fps_scale=False,
    update_scene_fps=True,
    global_scale=0.01,
)
source_arm = next(o for o in set(bpy.context.scene.objects) - before if o.type == "ARMATURE")

before = set(bpy.context.scene.objects)
bpy.ops.import_scene.gltf(filepath=args.character_glb)
target_arm = next(o for o in set(bpy.context.scene.objects) - before if o.type == "ARMATURE")

bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()

hips_bone = source_arm.pose.bones.get("Hips")
head_bone = source_arm.pose.bones.get("Head")
if hips_bone is not None and head_bone is not None:
    spine_world = (source_arm.matrix_world @ head_bone.head) - (source_arm.matrix_world @ hips_bone.head)
    if spine_world.length < 1e-6 or spine_world.z <= 0.8 * spine_world.length:
        raise RuntimeError("motion armature is not upright — BVH axis convention mismatch")

from app.retarget import resolve_motion_mapping, select_deepest_source_per_target
from app.quality_policy import evaluate_rig_certificate
from app.rig_quality import analyze_rig_path
import pathlib

report = analyze_rig_path(pathlib.Path(args.character_glb))
roles = evaluate_rig_certificate(report).get("resolved_roles", {})
mapping = resolve_motion_mapping(
    {b.name for b in source_arm.pose.bones},
    {b.name for b in target_arm.pose.bones},
    roles,
)
# Several source bones can legitimately collapse onto the same target (see
# backend/app/retarget.py); only the deepest source actually drives that
# target's pose. Score pose_error against that one-to-one mapping -- the
# collision "losers" are still listed in diagnose.json's rest-direction
# table (full SOMA coverage), but comparing their motion against a target
# bone actually driven by a *different* source is not a transfer error.
driving_mapping = {
    source_name: target_name
    for target_name, source_name in select_deepest_source_per_target(mapping).items()
}


def world_rest_dir(armature, pose_bone):
    bone = pose_bone.bone
    # tail_local - head_local is already armature-space at rest.
    direction = armature.matrix_world.to_3x3() @ (bone.tail_local - bone.head_local)
    return direction.normalized()


def world_pose_dir(armature, pose_bone):
    # pose_bone.head/.tail are Blender's own, always-correct current-pose
    # positions (armature space); this avoids re-deriving orientation
    # conventions from raw matrices.
    head_ws = armature.matrix_world @ pose_bone.head
    tail_ws = armature.matrix_world @ pose_bone.tail
    return (tail_ws - head_ws).normalized()


def angle_deg(a, b):
    return math.degrees(a.angle(b)) if a.length and b.length else 180.0


# --- diagnose.json: rest-direction table ---
diagnose_table = []
for source_name, target_name in sorted(mapping.items()):
    sb = source_arm.pose.bones[source_name]
    tb = target_arm.pose.bones[target_name]
    s_rest_dir = world_rest_dir(source_arm, sb)
    t_rest_dir = world_rest_dir(target_arm, tb)
    diagnose_table.append({
        "source": source_name,
        "target": target_name,
        "rest_dir_deg": round(angle_deg(s_rest_dir, t_rest_dir), 1),
        "s_rest_dir": [round(v, 3) for v in s_rest_dir],
        "t_rest_dir": [round(v, 3) for v in t_rest_dir],
    })
with open(os.path.join(args.outdir, "diagnose.json"), "w") as f:
    json.dump(diagnose_table, f, indent=1)

# --- bones.json: target rest bone table ---
bones_table = []
for bone in target_arm.data.bones:
    head_ws = target_arm.matrix_world @ bone.head_local
    tail_ws = target_arm.matrix_world @ bone.tail_local
    bones_table.append({
        "name": bone.name,
        "parent": bone.parent.name if bone.parent else None,
        "head": [round(v, 4) for v in head_ws],
        "tail": [round(v, 4) for v in tail_ws],
        "length": round((tail_ws - head_ws).length, 4),
    })
with open(os.path.join(args.outdir, "bones.json"), "w") as f:
    json.dump(bones_table, f, indent=1)

# --- composition_error.json: per-frame, per-bone world-DIRECTION angle
# error. This is NOT acceptance evidence: for a bone using the FULL rest-
# direction alignment, T_pose_w . Y_hat == S_pose_w . Y_hat by construction
# (see backend/app/retarget_math.py's docstring), so this is analytically
# ~0 any time the bake/round-trip is faithful. It is kept because a
# *nonzero* value here is still meaningful -- it proves (or disproves) that
# the baked action, once exported to glTF and re-imported, reproduces the
# exact rotation blender_worker.py computed (catches fps/frame-mapping
# bugs, stale bone state, export/import precision loss, etc).
#
# One deliberate exception: blender_worker.py uses an IDENTITY alignment
# for the pelvis/root (not the full rest-direction alignment) to avoid
# pitching the whole body by the SOMA/target rest-drawing-convention
# mismatch every frame (see retarget_bvh's torso-alignment comment). That
# bone's composition_error is therefore expected to sit at a CONSTANT,
# frame-invariant value equal to its raw rest-direction mismatch (e.g.
# Hips->pelvis ~48.6 degrees on this rig) rather than ~0 -- that is a known,
# intentional trade-off, not a bake bug. A bake bug instead shows up as a
# value that VARIES across frames, or as nonzero on a bone that is not the
# pelvis/root.
source_action = source_arm.animation_data.action if source_arm.animation_data else None
if source_action is None:
    raise RuntimeError("BVH motion did not contain an animation action")
clip_start = int(math.floor(source_action.frame_range[0]))
clip_end = int(math.ceil(source_action.frame_range[1]))

# --- real acceptance metrics (pose_error.json) ---
#
# (a) position error: each joint's world position, made hips-relative and
#     divided by that skeleton's OWN (rest) hips-to-head height, so it is a
#     dimensionless fraction of body height comparable across two
#     differently-sized/proportioned rigs. This is the metric composition
#     error cannot be: it is sensitive to bone-length/proportion mismatches
#     that direction-only matching is blind to.
# (b) twist/roll error: angle between the source and target bone-local X
#     axis, after correcting the target's X for the "swing" difference
#     between the two bones' current directions (quat_between_vectors,
#     i.e. Vector.rotation_difference) -- what's left is the pure twist
#     disagreement about the shared (post-swing-correction) bone axis.
# (c) foot contact: each skeleton's lowest foot-bone point (head or tail of
#     LeftFoot/RightFoot), measured as displacement from that SAME
#     skeleton's own REST foot height and divided by its own height --
#     i.e. how far the foot has lifted/sunk from its own neutral stance.
#     Raw world Z is not comparable across rigs: this target's rest pelvis
#     sits at Z~0.65 with rest feet at Z~-0.9 (root drawn near hip height),
#     while the BVH source's rest hips/feet sit near Z~0.9/~0.1 (root
#     drawn at the feet) -- two unrelated origin conventions.


def world_head(armature, pose_bone):
    return armature.matrix_world @ pose_bone.head


def world_tail(armature, pose_bone):
    return armature.matrix_world @ pose_bone.tail


def rest_height(armature, hips_name, head_name):
    hips_bone = armature.data.bones.get(hips_name)
    head_bone = armature.data.bones.get(head_name)
    if hips_bone is None or head_bone is None:
        return None
    return (
        (armature.matrix_world @ head_bone.head_local)
        - (armature.matrix_world @ hips_bone.head_local)
    ).length


def local_x_world(armature, pose_bone):
    world_quat = (armature.matrix_world @ pose_bone.matrix).to_quaternion()
    return world_quat @ Vector((1.0, 0.0, 0.0))


def twist_error_deg(source_arm_, sb, target_arm_, tb):
    s_dir = world_pose_dir(source_arm_, sb)
    t_dir = world_pose_dir(target_arm_, tb)
    # Minimal rotation taking the target's swing direction onto the
    # source's -- applying it to the target's X axis removes the "swing"
    # contribution to the X-axis difference, leaving only twist about the
    # (now shared) bone axis.
    swing_correction = t_dir.rotation_difference(s_dir)
    s_x = local_x_world(source_arm_, sb)
    t_x_corrected = swing_correction @ local_x_world(target_arm_, tb)
    return angle_deg(s_x, t_x_corrected)


target_hips_name = driving_mapping.get("Hips")
target_head_name = driving_mapping.get("Head")
source_height = rest_height(source_arm, "Hips", "Head")
target_height = (
    rest_height(target_arm, target_hips_name, target_head_name)
    if target_hips_name and target_head_name
    else None
)
if not source_height or not target_height:
    raise RuntimeError(
        "cannot compute pose_error.json: missing Hips/Head mapping needed to "
        f"normalize by body height (source_height={source_height}, target_height={target_height})"
    )

foot_sources = [name for name in ("LeftFoot", "RightFoot") if name in driving_mapping]


def lowest_foot_z(armature, pose_bones_or_data, names, use_pose):
    points = []
    for name in names:
        bone = pose_bones_or_data[name]
        if use_pose:
            points.append(armature.matrix_world @ bone.head)
            points.append(armature.matrix_world @ bone.tail)
        else:
            points.append(armature.matrix_world @ bone.head_local)
            points.append(armature.matrix_world @ bone.tail_local)
    return min(p.z for p in points)


# The two rigs do not share a "ground = world Z 0" convention (this
# target's rest pelvis sits at Z~0.65 with its rest feet at Z~-0.9, i.e.
# its root is drawn near hip height, not at the feet, while the BVH
# source's rest hips/feet sit near Z~0.9/~0.1 -- feet-on-the-ground). Raw
# world Z is therefore not comparable across skeletons. Foot contact is
# instead measured as displacement from each skeleton's OWN rest foot
# height (a "lift/sink relative to its own neutral stance"), which is
# meaningful regardless of either rig's absolute origin convention.
if foot_sources:
    source_rest_foot_z = lowest_foot_z(source_arm, source_arm.data.bones, foot_sources, use_pose=False)
    target_foot_names = [driving_mapping[n] for n in foot_sources]
    target_rest_foot_z = lowest_foot_z(target_arm, target_arm.data.bones, target_foot_names, use_pose=False)
else:
    source_rest_foot_z = target_rest_foot_z = None

per_frame = []
composition_per_frame = []
position_series = {name: [] for name in sorted(driving_mapping)}
twist_series = {name: [] for name in sorted(driving_mapping)}
foot_delta_series = []
for frame in range(clip_start, clip_end + 1):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

    composition_errors = {}
    position_errors = {}
    twist_errors = {}
    source_hips_pos = world_head(source_arm, source_arm.pose.bones["Hips"])
    target_hips_pos = world_head(target_arm, target_arm.pose.bones[target_hips_name])
    for source_name, target_name in driving_mapping.items():
        sb = source_arm.pose.bones[source_name]
        tb = target_arm.pose.bones[target_name]

        composition_errors[source_name] = round(
            angle_deg(world_pose_dir(source_arm, sb), world_pose_dir(target_arm, tb)), 2
        )

        s_rel = (world_head(source_arm, sb) - source_hips_pos) / source_height
        t_rel = (world_head(target_arm, tb) - target_hips_pos) / target_height
        position_pct = (s_rel - t_rel).length
        position_errors[source_name] = round(position_pct, 4)
        position_series[source_name].append(position_pct)

        twist_deg = twist_error_deg(source_arm, sb, target_arm, tb)
        twist_errors[source_name] = round(twist_deg, 2)
        twist_series[source_name].append(twist_deg)

    if foot_sources:
        source_foot_z = lowest_foot_z(source_arm, source_arm.pose.bones, foot_sources, use_pose=True)
        target_foot_z = lowest_foot_z(target_arm, target_arm.pose.bones, target_foot_names, use_pose=True)
        source_foot_pct = (source_foot_z - source_rest_foot_z) / source_height
        target_foot_pct = (target_foot_z - target_rest_foot_z) / target_height
        foot_delta_pct = target_foot_pct - source_foot_pct
        foot_delta_series.append(foot_delta_pct)
        foot_record = {
            "source_lift_pct": round(source_foot_pct, 4),
            "target_lift_pct": round(target_foot_pct, 4),
            "delta_pct": round(foot_delta_pct, 4),
        }
    else:
        foot_record = None

    composition_per_frame.append({"frame": frame, "errors": composition_errors})
    per_frame.append({
        "frame": frame,
        "position_pct": position_errors,
        "twist_deg": twist_errors,
        "foot": foot_record,
    })

composition_summary = {
    name: {
        "max": round(max(values), 2) if values else None,
        "mean": round(sum(values) / len(values), 2) if values else None,
    }
    for name, values in (
        (name, [f["errors"][name] for f in composition_per_frame]) for name in sorted(driving_mapping)
    )
}
composition_overall_max = max((s["max"] for s in composition_summary.values() if s["max"] is not None), default=0.0)
composition_error = {
    "frames": composition_per_frame,
    "summary": composition_summary,
    "overall_max": round(composition_overall_max, 2),
}
with open(os.path.join(args.outdir, "composition_error.json"), "w") as f:
    json.dump(composition_error, f, indent=1)
print(
    f"COMPOSITION_ERROR overall_max={composition_overall_max:.4f} "
    "(expect ~0 except a constant, frame-invariant value on the pelvis/root -- "
    "see composition_error.json comment)"
)

position_summary = {
    name: {
        "max": round(max(values), 4) if values else None,
        "mean": round(sum(values) / len(values), 4) if values else None,
    }
    for name, values in position_series.items()
}
position_overall_max = max((s["max"] for s in position_summary.values() if s["max"] is not None), default=0.0)
frame1_position = per_frame[0]["position_pct"] if per_frame else {}
frame1_position_max = max(frame1_position.values(), default=0.0)

twist_summary = {
    name: {
        "max": round(max(values), 2) if values else None,
        "mean": round(sum(values) / len(values), 2) if values else None,
    }
    for name, values in twist_series.items()
}
twist_overall_max = max((s["max"] for s in twist_summary.values() if s["max"] is not None), default=0.0)

if foot_delta_series:
    foot_summary = {
        "max_abs_delta_pct": round(max(abs(v) for v in foot_delta_series), 4),
        "mean_abs_delta_pct": round(sum(abs(v) for v in foot_delta_series) / len(foot_delta_series), 4),
    }
else:
    foot_summary = {"max_abs_delta_pct": None, "mean_abs_delta_pct": None}

pose_error = {
    "frames": per_frame,
    "source_height": round(source_height, 4),
    "target_height": round(target_height, 4),
    "thresholds": {"position_pct_max_frame1": 0.03, "twist_deg_max": 20.0},
    "position_summary": position_summary,
    "position_overall_max_pct": round(position_overall_max, 4),
    "frame1_position_max_pct": round(frame1_position_max, 4),
    "twist_summary": twist_summary,
    "twist_overall_max_deg": round(twist_overall_max, 2),
    "foot_contact": foot_summary,
}
with open(os.path.join(args.outdir, "pose_error.json"), "w") as f:
    json.dump(pose_error, f, indent=1)

print(f"POSITION_ERROR overall_max_pct={position_overall_max * 100:.2f}% frame1_max_pct={frame1_position_max * 100:.2f}%")
for name in sorted(position_summary):
    s = position_summary[name]
    print(f"POSITION_ERROR {name}: max={s['max'] * 100:.2f}% mean={s['mean'] * 100:.2f}%")
print(f"TWIST_ERROR overall_max_deg={twist_overall_max:.2f}")
for name in sorted(twist_summary):
    t = twist_summary[name]
    print(f"TWIST_ERROR {name}: max={t['max']}deg mean={t['mean']}deg")
if foot_summary["max_abs_delta_pct"] is not None:
    print(f"FOOT_CONTACT max_abs_delta_pct={foot_summary['max_abs_delta_pct'] * 100:.2f}% mean_abs_delta_pct={foot_summary['mean_abs_delta_pct'] * 100:.2f}%")
else:
    print("FOOT_CONTACT no foot bones in driving_mapping")

# --- renders: side-by-side stick figures at the requested frames ---


def parse_frames(spec, start, end):
    frames = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("%"):
            fraction = float(token[:-1]) / 100.0
            frame = start + round(fraction * (end - start))
        else:
            frame = int(token)
        frames.append(max(start, min(end, frame)))
    seen = set()
    ordered = []
    for frame in frames:
        if frame not in seen:
            seen.add(frame)
            ordered.append(frame)
    return ordered


render_frames = parse_frames(args.frames, clip_start, clip_end)


STICK_COLORS = {
    "red": (1.0, 0.1, 0.1, 1.0),      # source (BVH motion)
    "green": (0.1, 1.0, 0.3, 1.0),    # target, current (posed) frame
    "blue": (0.2, 0.5, 1.0, 1.0),     # target, its OWN rest pose (drift reference)
}


def _apply_stick_material(obj, color):
    mat = bpy.data.materials.get(color) or bpy.data.materials.new(color)
    rgba = STICK_COLORS[color]
    # `diffuse_color` is what Workbench's MATERIAL color mode reads; the
    # node-tree Base Color/Emission is set too so the same material also
    # reads correctly if re-rendered under EEVEE/Cycles.
    mat.diffuse_color = rgba
    mat.use_nodes = True
    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Emission Color"].default_value = rgba
    principled.inputs["Emission Strength"].default_value = 2.0
    obj.data.materials.append(mat)


def _add_cylinder(head, tail, name):
    length = max((tail - head).length, 0.005)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=length, location=(head + tail) / 2)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (tail - head).to_track_quat("Z", "Y").to_euler()
    return obj


def draw_armature(arm, color, offset):
    """Draw the armature's CURRENT (posed, frame-dependent) pose."""
    drawn_objects = []
    for bone in arm.pose.bones:
        head = (arm.matrix_world @ bone.head) + offset
        tail = (arm.matrix_world @ bone.tail) + offset
        obj = _add_cylinder(head, tail, f"w_{arm.name}_{bone.name}_{color}")
        _apply_stick_material(obj, color)
        drawn_objects.append(obj)
    return drawn_objects


def draw_rest_armature(arm, color, offset):
    """Draw the armature's static REST pose (bone.head_local/tail_local) --
    frame-independent, used for the target-vs-its-own-rest drift figure."""
    drawn_objects = []
    for bone in arm.data.bones:
        head = (arm.matrix_world @ bone.head_local) + offset
        tail = (arm.matrix_world @ bone.tail_local) + offset
        obj = _add_cylinder(head, tail, f"w_{arm.name}_{bone.name}_{color}_rest")
        _apply_stick_material(obj, color)
        drawn_objects.append(obj)
    return drawn_objects


SOURCE_OFFSET = Vector((-0.6, 0.0, 0.0))
TARGET_OFFSET = Vector((0.6, 0.0, 0.0))
TARGET_REST_OFFSET = Vector((1.8, 0.0, 0.0))  # third figure: target's own rest pose

# Camera framed to show ALL THREE full-body figures: the bounding box of
# every bone's rest head/tail at each figure's own offset, not the old
# crotch close-up. A square render + ortho_scale = max(width, height)
# guarantees both axes fit regardless of Blender's sensor-fit convention.
rest_offsets = (
    (source_arm, SOURCE_OFFSET),
    (target_arm, TARGET_OFFSET),
    (target_arm, TARGET_REST_OFFSET),
)
rest_points = []
for arm, offset in rest_offsets:
    for bone in arm.data.bones:
        rest_points.append((arm.matrix_world @ bone.head_local) + offset)
        rest_points.append((arm.matrix_world @ bone.tail_local) + offset)

min_corner = Vector((min(p.x for p in rest_points), min(p.y for p in rest_points), min(p.z for p in rest_points)))
max_corner = Vector((max(p.x for p in rest_points), max(p.y for p in rest_points), max(p.z for p in rest_points)))
center = (min_corner + max_corner) / 2
size = max_corner - min_corner
margin = 1.3
ortho_scale = max(size.x, size.z, 0.5) * margin

cam_data = bpy.data.cameras.new("DiagnoseCam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = ortho_scale
cam = bpy.data.objects.new("DiagnoseCam", cam_data)
bpy.context.scene.collection.objects.link(cam)
# The character faces -Y (F2); stand the camera further out along -Y and
# look toward +Y so it sees the front of both figures.
cam.location = (center.x, min_corner.y - 3.0, center.z)
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
bpy.context.scene.camera = cam

sun_data = bpy.data.lights.new("DiagnoseSun", type="SUN")
sun = bpy.data.objects.new("DiagnoseSun", sun_data)
bpy.context.scene.collection.objects.link(sun)
sun.rotation_euler = (0.6, 0.0, 0.0)

# Hide the character's own skinned mesh from these renders -- this is a
# stick-figure comparison of the two skeletons, and the (un-offset) mesh
# would otherwise sit directly on top of / obscure the offset target
# skeleton cylinders.
for obj in list(bpy.context.scene.objects):
    if obj.type == "MESH":
        obj.hide_render = True

scene = bpy.context.scene
scene.render.resolution_x = 1000
scene.render.resolution_y = 1000
scene.render.engine = "BLENDER_WORKBENCH"
# Workbench ignores node-based materials by default (flat/random shading);
# tell it to actually use each cylinder's Base Color so source/target read
# as red/green.
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "MATERIAL"

# The target's own rest pose is frame-independent, so it is drawn once and
# left in the scene for every render below -- the third (blue) figure next
# to the posed (green) target makes torso drift (pelvis/spine/neck/head
# pitching away from the target's own rest orientation) visible directly.
draw_rest_armature(target_arm, "blue", TARGET_REST_OFFSET)

drawn = []
for frame in render_frames:
    for obj in drawn:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    drawn = []
    for arm, color, offset in ((source_arm, "red", SOURCE_OFFSET), (target_arm, "green", TARGET_OFFSET)):
        drawn.extend(draw_armature(arm, color, offset))
    scene.render.filepath = os.path.join(args.outdir, f"frame-{frame}.png")
    bpy.ops.render.render(write_still=True)

print("DIAGNOSE_DONE")
