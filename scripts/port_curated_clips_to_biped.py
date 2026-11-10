"""Port the 87 curated donor animation clips onto the canonical biped
template skeleton (workstream N4, PIPELINE_POLICY.md).

Renames every donor bone to its canonical name per
`assets/skeletons/biped/template.json`'s `donor_name` field. This is a
straight 1:1 rename -- the donor's 66 bones are exactly the template's 21
required + 45 optional bones (verified below, not assumed), so nothing is
dropped or added. After renaming, the parent edges are verified to match
the template's hierarchy bone-for-bone, the donor's mesh data is stripped
(the clip store is armature + animations only), and all 87 actions are
exported into a single multi-action GLB plus a provenance manifest.

Usage (offline tooling -- run with Blender's own bpy, not the worker
container):
    blender -b -P scripts/port_curated_clips_to_biped.py -- \\
        <donor_glb> <template_json> <out_glb> <out_manifest_json> [<out_divergence_json>]

If the donor's bone set or the renamed hierarchy diverges from the
template, the script writes a divergence report (to <out_divergence_json>
if given, and always to stdout) and exits non-zero instead of forcing the
conversion -- per PIPELINE_POLICY.md's S3 contract ("the skeleton IS the
family template ... fitters never emit bespoke topology").
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy

SCRIPT_VERSION = "1.0.0"
ACTION_NAME_SUFFIX = "_Armature"
EXPECTED_ACTION_COUNT = 87


def _inject_bone_only_skin(glb_path: Path) -> tuple[int, int]:
    """Patch a `skins` entry into an exported bone-only (mesh-free) GLB.

    Blender's glTF exporter only ever writes a `skins` entry when a mesh
    object actually uses the armature (skinning requires a mesh with an
    Armature modifier) -- with no mesh in this store, the export has zero
    `skins` entries. That matters on *reimport*: Blender's glTF importer
    only builds a bpy Armature datablock for nodes reachable through a
    skin's `joints` list; without one, the bone hierarchy reimports as
    plain Empty objects, not an armature (verified empirically -- the
    mesh-free export reimported with 0 ARMATURE objects before this patch
    was added).

    glTF's `skins` entry doesn't require any mesh to reference it, and
    `inverseBindMatrices` are optional (Blender's importer falls back to
    each joint node's own TRS as the bind pose when absent -- see
    `pick_bind_pose` in the importer's gltf2_blender_vnode.py), so patching
    in `{"joints": [...], "skeleton": <root>}` after export, with no
    accessor data to add, is sufficient to make the store reimport as a
    proper armature. Returns (skeleton_root_node_index, joint_count).
    """
    raw = glb_path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise RuntimeError(f"not a GLB file: {glb_path}")
    offset = 12
    json_chunk = None
    other_chunks: list[tuple[bytes, bytes]] = []
    while offset < total_length:
        chunk_len, chunk_type = struct.unpack_from("<I4s", raw, offset)
        chunk_data = raw[offset + 8: offset + 8 + chunk_len]
        if chunk_type == b"JSON":
            json_chunk = json.loads(chunk_data)
        else:
            other_chunks.append((chunk_type, chunk_data))
        offset += 8 + chunk_len
    if json_chunk is None:
        raise RuntimeError(f"GLB has no JSON chunk: {glb_path}")
    if json_chunk.get("skins"):
        raise RuntimeError("exported GLB unexpectedly already has a skins array")

    nodes = json_chunk["nodes"]
    children: set[int] = set()
    for node in nodes:
        children.update(node.get("children", []))
    root_candidates = [i for i in range(len(nodes)) if i not in children]
    if len(root_candidates) != 1:
        raise RuntimeError(f"expected exactly one parentless node, found {root_candidates}")
    skeleton_root = root_candidates[0]
    joints = [i for i in range(len(nodes)) if i != skeleton_root]

    json_chunk["skins"] = [{"name": "biped", "joints": joints, "skeleton": skeleton_root}]

    def _padded(data: bytes, pad_byte: bytes) -> bytes:
        pad = (-len(data)) % 4
        return data + pad_byte * pad

    new_json_bytes = _padded(
        json.dumps(json_chunk, separators=(",", ":")).encode("utf-8"), b" "
    )
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", version, 0)  # length patched below
    out += struct.pack("<I4s", len(new_json_bytes), b"JSON") + new_json_bytes
    for chunk_type, chunk_data in other_chunks:
        chunk_data = _padded(chunk_data, b"\x00")
        out += struct.pack("<I4s", len(chunk_data), chunk_type) + chunk_data
    struct.pack_into("<I", out, 8, len(out))
    glb_path.write_bytes(bytes(out))
    return skeleton_root, len(joints)


def fail(divergence_path: str | None, payload: dict) -> None:
    print("DIVERGENCE", json.dumps(payload, indent=2))
    if divergence_path:
        Path(divergence_path).parent.mkdir(parents=True, exist_ok=True)
        Path(divergence_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    raise SystemExit(f"conversion aborted: {payload.get('reason')}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) < 4:
        raise SystemExit(
            "usage: blender -b -P port_curated_clips_to_biped.py -- "
            "<donor_glb> <template_json> <out_glb> <out_manifest_json> [<out_divergence_json>]"
        )
    donor_path = Path(argv[0]).resolve()
    template_path = Path(argv[1]).resolve()
    out_glb = Path(argv[2]).resolve()
    out_manifest = Path(argv[3]).resolve()
    divergence_path = argv[4] if len(argv) > 4 else None

    template = json.loads(template_path.read_text(encoding="utf-8"))
    donor_to_canonical = {b["donor_name"]: b["name"] for b in template["bones"]}
    canonical_parent = {b["name"]: b["parent"] for b in template["bones"]}
    all_canonical_names = set(canonical_parent)
    expected_sha256 = template["provenance"]["donor_sha256"]

    donor_bytes = donor_path.read_bytes()
    actual_sha256 = hashlib.sha256(donor_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        fail(divergence_path, {
            "reason": "donor_sha256_mismatch",
            "expected": expected_sha256,
            "actual": actual_sha256,
            "donor_path": str(donor_path),
        })

    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(donor_path))
    if result != {"FINISHED"}:
        fail(divergence_path, {"reason": "gltf_import_failed", "result": list(result)})

    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        fail(divergence_path, {
            "reason": "unexpected_armature_count",
            "count": len(armatures),
            "names": [a.name for a in armatures],
        })
    armature = armatures[0]

    # Strip mesh data -- the clip store is armature + animations only.
    mesh_objects = [o for o in list(bpy.context.scene.objects) if o.type == "MESH"]
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    donor_bone_names = {b.name for b in armature.data.bones}
    expected_donor_names = set(donor_to_canonical)
    if donor_bone_names != expected_donor_names:
        fail(divergence_path, {
            "reason": "donor_bone_set_mismatch",
            "missing_from_donor": sorted(expected_donor_names - donor_bone_names),
            "unexpected_in_donor": sorted(donor_bone_names - expected_donor_names),
        })

    # Capture pre-rename parent edges (by donor name) so the post-rename
    # hierarchy can be checked against the template, not just assumed.
    donor_parent_by_donor_name = {
        b.name: (b.parent.name if b.parent else None) for b in armature.data.bones
    }

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    # Two-phase rename avoids any transient name collision mid-loop (renaming
    # a bone via bpy propagates to pose bones and every action's F-curve
    # data paths automatically -- Blender's standard bone-rename behavior).
    for donor_name in donor_to_canonical:
        edit_bones[donor_name].name = f"__n4tmp__{donor_name}"
    for donor_name, canonical_name in donor_to_canonical.items():
        edit_bones[f"__n4tmp__{donor_name}"].name = canonical_name
    bpy.ops.object.mode_set(mode="OBJECT")

    # Verify: renamed bone set is exactly the template's, and every parent
    # edge (translated through the same rename map) matches the template's
    # canonical parent -- hierarchy identity by construction, checked not
    # assumed.
    renamed_names = {b.name for b in armature.data.bones}
    if renamed_names != all_canonical_names:
        fail(divergence_path, {
            "reason": "renamed_bone_set_mismatch",
            "missing": sorted(all_canonical_names - renamed_names),
            "unexpected": sorted(renamed_names - all_canonical_names),
        })

    hierarchy_mismatches = []
    for donor_name, canonical_name in donor_to_canonical.items():
        donor_parent = donor_parent_by_donor_name[donor_name]
        expected_parent = (
            donor_to_canonical[donor_parent] if donor_parent is not None else None
        )
        template_parent = canonical_parent[canonical_name]
        actual_parent_bone = armature.data.bones[canonical_name].parent
        actual_parent = actual_parent_bone.name if actual_parent_bone else None
        if expected_parent != template_parent or actual_parent != template_parent:
            hierarchy_mismatches.append({
                "canonical_name": canonical_name,
                "donor_name": donor_name,
                "donor_parent_translated": expected_parent,
                "template_parent": template_parent,
                "actual_renamed_parent": actual_parent,
            })
    if hierarchy_mismatches:
        fail(divergence_path, {
            "reason": "hierarchy_mismatch",
            "mismatches": hierarchy_mismatches,
        })

    actions = list(bpy.data.actions)
    if len(actions) != EXPECTED_ACTION_COUNT:
        fail(divergence_path, {
            "reason": "unexpected_action_count",
            "count": len(actions),
            "expected": EXPECTED_ACTION_COUNT,
            "names": sorted(a.name for a in actions),
        })
    bad_suffix = [a.name for a in actions if not a.name.endswith(ACTION_NAME_SUFFIX)]
    if bad_suffix:
        fail(divergence_path, {"reason": "action_missing_suffix", "names": bad_suffix})

    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    clips = []
    for action in sorted(actions, key=lambda a: a.name):
        frame_start = int(round(action.frame_range[0]))
        frame_end = int(round(action.frame_range[1]))
        clips.append({
            "clip_name": action.name,
            "display_name": action.name[: -len(ACTION_NAME_SUFFIX)],
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
            "duration_seconds": round((frame_end - frame_start) / fps, 4),
        })

    manifest = {
        "family": "biped",
        "store_file": out_glb.name,
        "fps": fps,
        "provenance": {
            "donor_path": "vendor/mesh2motion-app/static/animations/human-base-animations.glb",
            "donor_sha256": actual_sha256,
            "template_version": template["version"],
            "conversion_script": "scripts/port_curated_clips_to_biped.py",
            "conversion_script_version": SCRIPT_VERSION,
            "blender_version": bpy.app.version_string,
            "converted_at": datetime.now(timezone.utc).isoformat(),
        },
        "lookup_contract": (
            "Stored glTF/Blender action names keep the importer-stable "
            "'<RawDonorAnimationName>_Armature' convention (identical to "
            "backend/app/curated_motion.py's curated_action_name() suffix "
            "rule) -- a consumer selects a clip by exact match on "
            "'clip_name' (e.g. 'Sword_Attack_Armature') against "
            "bpy.data.actions after importing this store, and assigns it "
            "to the armature's animation_data.action. 'display_name' is "
            "presentation-only: clip_name with the trailing '_Armature' "
            "suffix stripped (no other normalization)."
        ),
        "clip_count": len(clips),
        "clips": clips,
    }

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_skins=False,
    )
    skeleton_root_idx, joint_count = _inject_bone_only_skin(out_glb)
    if joint_count != len(renamed_names):
        fail(divergence_path, {
            "reason": "patched_skin_joint_count_mismatch",
            "joint_count": joint_count,
            "expected": len(renamed_names),
        })
    manifest["gltf_skin_patch"] = (
        "Blender's exporter omits `skins` for a mesh-free file; this store "
        "has a `skins[0]` patched in post-export (joints = every bone node, "
        f"skeleton = node {skeleton_root_idx}, the 'Armature' object node) "
        "so glTF/Blender importers recognize it as an armature rather than "
        "plain Empty nodes. No inverseBindMatrices accessor was added -- "
        "not required; importers fall back to each joint's own rest TRS."
    )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "CONVERSION_OK",
        "bones", len(renamed_names),
        "actions", len(actions),
        "glb_bytes", out_glb.stat().st_size,
        "skeleton_root_idx", skeleton_root_idx,
    )


if __name__ == "__main__":
    main()
