"""Calibrate the S5 `foot_contact_error` band over the whole biped clip store.

Throwaway calibration tool for bead erbye. Run it inside Blender against
`assets/skeletons/biped/clips/biped-clips-v1.glb` -- the store carries the
family template armature and all 87 curated actions and no mesh, so every
clip can be sampled on the SAME skeleton the pipeline attaches it to, with no
GPU, no rig and no character.

    ~/tools/blender-4.2.14-linux-x64/blender --background --factory-startup \
        --python scripts/calibrate_foot_contact.py -- \
        --repo ~/AISmith-3D-launcher-test --out /tmp/foot_contact_calibration.json

It samples every frame of every action (the same all-frames sampling
`deformation_review.all_animation_frames` uses), feeds the world-space bone
head/tail arrays to `backend/app/motion_metrics.compute_motion_metrics` -- the
exact function the pipeline runs -- and reports the distribution of
`foot_slide_max_in_contact` (the contact-filtered figure; the raw
`foot_slide_max` over-counts on stepping/airborne clips and must never be
gated) per motion family.

Motion families are a name-keyword grouping over the clip's stored action
name, first rule that matches wins. The order matters: a clip named `*_Idle_*`
is a held pose regardless of what it is holding (`Idle_Sword`, `Pistol_Idle`,
`NinjaJump_Idle`), so `idle` is tested before the weapon and airborne rules.

    idle        idle, sitting, lay, death, zombie_idle, talking, yes, consume
    acrobatic   jump, roll, slide, dash, knockback, climb, ninja
    locomotion  walk, jog, sprint, run, swim
    attack      sword, punch, melee, pistol, shoot, throw, spell, shield,
                chop, hit, block, aim, reload, scratch
    other       everything else (manipulation/interaction clips: Farm_*,
                PickUp_*, Interact, Fixing_Kneeling, Chest_Open, Dance_Simple,
                Driving, Push, Consume ...)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- family classification (pure, importable by tests) ----------------------

FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("idle", ("idle", "sitting", "lay", "death", "talking", "yes", "consume")),
    ("acrobatic", ("jump", "roll", "slide", "dash", "knockback", "climb", "ninja")),
    ("locomotion", ("walk", "jog", "sprint", "run", "swim")),
    (
        "attack",
        (
            "sword", "punch", "melee", "pistol", "shoot", "throw", "spell",
            "shield", "chop", "hit", "block", "aim", "reload", "scratch",
        ),
    ),
)
FAMILY_ORDER = ("locomotion", "attack", "idle", "acrobatic", "other")


def motion_family(clip_name: str) -> str:
    """Rough motion family for a stored clip action name (first rule wins)."""
    lowered = clip_name.lower()
    for family, keywords in FAMILY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return family
    return "other"


# --- distribution reporting -------------------------------------------------


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(rows: list[dict]) -> dict:
    """Per-family and overall distribution of contact-filtered foot slide."""
    summary: dict = {"families": {}, "overall": {}}
    for family in FAMILY_ORDER:
        members = [row for row in rows if row["family"] == family]
        if members:
            summary["families"][family] = _describe(members)
    summary["overall"] = _describe(rows)
    return summary


def _describe(members: list[dict]) -> dict:
    slide = [float(row["foot_slide_max_in_contact"]) for row in members]
    with_contact = [row for row in members if row["stance_frames_fraction"] > 0.0]
    measured = [
        float(row["foot_slide_max_in_contact"])
        for row in members
        if row["foot_slide_max_in_contact"] > 0.0
    ]
    worst = sorted(members, key=lambda row: -row["foot_slide_max_in_contact"])[:5]
    return {
        "clips": len(members),
        "clips_with_contact_frames": len(with_contact),
        "clips_with_nonzero_slide": len(measured),
        "min": round(min(slide), 6) if slide else 0.0,
        "p50": round(_quantile(slide, 0.50), 6),
        "p75": round(_quantile(slide, 0.75), 6),
        "p90": round(_quantile(slide, 0.90), 6),
        "p95": round(_quantile(slide, 0.95), 6),
        "max": round(max(slide), 6) if slide else 0.0,
        "mean_stance_frames_fraction": round(
            sum(row["stance_frames_fraction"] for row in members) / max(len(members), 1), 4
        ),
        "worst_clips": [
            {
                "clip": row["clip"],
                "foot_slide_max_in_contact": row["foot_slide_max_in_contact"],
                "foot_slide_max": row["foot_slide_max"],
                "stance_frames_fraction": row["stance_frames_fraction"],
            }
            for row in worst
        ],
    }


def render_table(summary: dict) -> str:
    header = (
        f"{'family':<12}{'clips':>6}{'w/contact':>11}{'min':>9}{'p50':>9}"
        f"{'p75':>9}{'p90':>9}{'p95':>9}{'max':>9}{'stance%':>9}"
    )
    lines = [header, "-" * len(header)]
    for family in FAMILY_ORDER + ("__overall__",):
        block = (
            summary["overall"] if family == "__overall__"
            else summary["families"].get(family)
        )
        if not block:
            continue
        label = "ALL" if family == "__overall__" else family
        lines.append(
            f"{label:<12}{block['clips']:>6}{block['clips_with_contact_frames']:>11}"
            f"{block['min']:>9.4f}{block['p50']:>9.4f}{block['p75']:>9.4f}"
            f"{block['p90']:>9.4f}{block['p95']:>9.4f}{block['max']:>9.4f}"
            f"{block['mean_stance_frames_fraction'] * 100:>8.1f}%"
        )
    return "\n".join(lines)


# --- Blender sampling -------------------------------------------------------


def sample_store(repo: Path, store: Path) -> list[dict]:
    import bpy  # noqa: PLC0415  -- only available under Blender
    import numpy as np  # noqa: PLC0415

    sys.path.insert(0, str(repo))
    from backend.app.motion_metrics import compute_motion_metrics  # noqa: PLC0415

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(store))
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature imported from {store}")
    # Same rule blender_worker.deformation_diagnostics uses: most bones wins,
    # name breaks ties, so the pick is deterministic.
    armature = max(armatures, key=lambda item: (len(item.pose.bones), item.name))
    animation_data = armature.animation_data or armature.animation_data_create()
    actions = sorted(bpy.data.actions, key=lambda item: item.name)

    rows: list[dict] = []
    for action in actions:
        start, end = (round(value) for value in action.frame_range)
        if end <= start:
            continue
        animation_data.action = action
        armature.data.pose_position = "POSE"
        frames: list[int] = []
        heads: dict[str, list] = {bone.name: [] for bone in armature.pose.bones}
        tails: dict[str, list] = {bone.name: [] for bone in armature.pose.bones}
        for frame in range(start, end + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            frames.append(int(frame))
            world = armature.matrix_world
            for pose_bone in armature.pose.bones:
                heads[pose_bone.name].append(tuple(world @ pose_bone.head))
                tails[pose_bone.name].append(tuple(world @ pose_bone.tail))
        metrics = compute_motion_metrics(
            frames,
            {name: np.asarray(values, dtype=np.float64) for name, values in heads.items()},
            {name: np.asarray(values, dtype=np.float64) for name, values in tails.items()},
            source_action=action.name,
        )
        if not metrics.get("measurable"):
            print(f"SKIP {action.name}: {metrics.get('reason')}")
            continue
        rows.append({
            "clip": action.name,
            "family": motion_family(action.name),
            "frames": metrics["sampled_frames"],
            "foot_slide_max_in_contact": metrics["foot_slide_max_in_contact"],
            "foot_slide_mean_in_contact": metrics["foot_slide_mean_in_contact"],
            "foot_slide_max": metrics["foot_slide_max"],
            "stance_frames_fraction": metrics["stance_frames_fraction"],
            "stance_switch_count": metrics["stance_switch_count"],
            "joint_motion_coverage": metrics["joint_motion_coverage"],
            "static_tail_fraction": metrics["static_tail_fraction"],
            "root_travel_path": metrics["root_travel_path"],
            "root_travel_net": metrics["root_travel_net"],
        })
        print(
            f"{action.name:<34} {rows[-1]['family']:<11} "
            f"slide_contact={rows[-1]['foot_slide_max_in_contact']:.4f} "
            f"slide_raw={rows[-1]['foot_slide_max']:.4f} "
            f"stance={rows[-1]['stance_frames_fraction']:.3f}"
        )
    return rows


def main(argv: list[str]) -> int:
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    repo = Path.cwd()
    out = Path("/tmp/foot_contact_calibration.json")
    for index, value in enumerate(args):
        if value == "--repo":
            repo = Path(args[index + 1]).expanduser()
        elif value == "--out":
            out = Path(args[index + 1]).expanduser()
    store = repo / "assets/skeletons/biped/clips/biped-clips-v1.glb"
    rows = sample_store(repo, store)
    summary = summarize(rows)
    out.write_text(json.dumps({"clips": rows, "summary": summary}, indent=2), encoding="utf-8")
    print()
    print(render_table(summary))
    print(f"\nwrote {out} ({len(rows)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
