"""Verify a produced biped clip-store GLB (workstream N4).

Runs entirely outside Blender except for two subprocess calls into it (the
rest-pose dump, and the reused /tmp/diag motion-metrics / stick-figure
tools). Checks:

  1. Reimports the store fresh and dumps its rest pose + action roster
     (scripts/dump_biped_store_restpose.py).
  2. Bone set == template's full bone set (required + optional), hierarchy
     matches, via backend.app.skeleton_template.conformance_report --
     the same S3-contract checker the fitter path uses.
  3. Exactly 87 actions, matching the manifest's clip_name list exactly.
  4. No mesh objects in the reimported store (armature + animations only).
  5. Replays 3 sample clips (Roll_Armature, Sword_Attack_Armature,
     Walk_Armature) via /tmp/diag/motion_metrics.py: asserts each clip's
     joint_motion_coverage_pct > 0 / moved_bones > 0 (motion survives the
     round trip), and renders 1 stick-figure frame per clip via
     /tmp/diag/render_stick.py.

Usage:
    python3 scripts/verify_biped_clip_store.py \\
        --repo-root <path> --blender <path-to-blender-binary> \\
        --store <store_glb> --manifest <manifest_json> --out-dir <dir>

Exits non-zero on any failed check; prints a JSON summary either way.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SAMPLE_CLIPS = ["Roll_Armature", "Sword_Attack_Armature", "Walk_Armature"]


def run_blender(blender: str, script: Path, args: list[str]) -> str:
    cmd = [blender, "-b", "-P", str(script), "--"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"blender script {script.name} failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    from backend.app.skeleton_template import conformance_report, load_template  # noqa: E402

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = repo_root / "scripts"
    diag_dir = Path("/tmp/diag")

    findings: list[dict] = []
    summary: dict = {"checks": {}}

    # 1. rest-pose + action dump (fresh reimport).
    dump_path = out_dir / "rest_pose_dump.json"
    run_blender(
        args.blender,
        scripts_dir / "dump_biped_store_restpose.py",
        [args.store, str(dump_path)],
    )
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    summary["dump"] = {
        "bone_count": dump["bone_count"],
        "action_count": dump["action_count"],
        "mesh_object_count": dump["mesh_object_count"],
    }

    # 2. bone-set / hierarchy conformance against the shipped template.
    template = load_template("biped", skeletons_dir=repo_root / "assets" / "skeletons")
    template_names = template.bone_names()
    dumped_names = set(dump["names"])
    if dumped_names != template_names:
        findings.append({
            "check": "bone_set_exact_match",
            "missing": sorted(template_names - dumped_names),
            "unexpected": sorted(dumped_names - template_names),
        })
    report = conformance_report(
        dump["names"], dump["parents"], dump["heads"], dump["tails"], dump["rolls"],
        family="biped", template=template,
    )
    summary["checks"]["conformance_report_findings"] = report
    if report:
        findings.append({"check": "conformance_report", "findings": report})

    # 3. action roster.
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_clip_names = sorted(c["clip_name"] for c in manifest["clips"])
    dumped_actions = sorted(dump["actions"])
    if dumped_actions != manifest_clip_names:
        findings.append({
            "check": "action_roster_matches_manifest",
            "only_in_store": sorted(set(dumped_actions) - set(manifest_clip_names)),
            "only_in_manifest": sorted(set(manifest_clip_names) - set(dumped_actions)),
        })
    if len(dumped_actions) != 87:
        findings.append({"check": "action_count_87", "actual": len(dumped_actions)})
    summary["checks"]["action_count"] = len(dumped_actions)

    # 4. no mesh objects.
    if dump["mesh_object_count"] != 0:
        findings.append({
            "check": "no_mesh_objects",
            "mesh_object_names": dump["mesh_object_names"],
        })

    # 5. sample-clip motion replay + stick-figure renders.
    clip_by_name = {c["clip_name"]: c for c in manifest["clips"]}
    summary["checks"]["sample_clips"] = {}
    for clip_name in SAMPLE_CLIPS:
        if clip_name not in clip_by_name:
            findings.append({"check": "sample_clip_present", "clip_name": clip_name})
            continue
        clip = clip_by_name[clip_name]
        frame_start, frame_end = clip["frame_start"], clip["frame_end"]
        sample_step = max((frame_end - frame_start) // 3, 1)

        metrics_json = out_dir / f"{clip_name}.metrics.json"
        run_blender(
            args.blender,
            diag_dir / "motion_metrics.py",
            ["glb", args.store, clip_name, str(frame_start), str(frame_end),
             str(sample_step), str(metrics_json), clip_name],
        )
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        moved = metrics["metric_1_moved_bones"]
        coverage = metrics["metric_1_joint_motion_coverage_pct"]
        summary["checks"]["sample_clips"][clip_name] = {
            "moved_bones": moved,
            "coverage_pct": coverage,
            "valid_bones": metrics["metric_1_valid_bones"],
        }
        if moved <= 0:
            findings.append({
                "check": "sample_clip_motion_survives",
                "clip_name": clip_name,
                "moved_bones": moved,
            })

        # render_stick.py's static-bone exclusion heuristic (Pass 1) needs
        # several sampled frames spread across the clip to tell a moving
        # bone from a static one. Two failure modes to avoid: a single
        # frame (frame_start == frame_end) makes every bone look static
        # (zero range across one sample); and just two far-apart samples
        # (start/end only) also mis-flags bones that move mid-clip but
        # return close to their starting pose by the end (empirically
        # true of most limbs in these attack/loop clips -- verified this
        # under-samples on both the converted store and the raw donor
        # file alike, so it's a property of 2-sample coverage, not a
        # conversion defect). Densely sample (~9 frames) instead, then
        # keep only the frame closest to the clip's midpoint as "the"
        # sanity image and discard the rest.
        stick_dir = out_dir / "stick"
        dense_step = max((frame_end - frame_start) // 8, 1)
        run_blender(
            args.blender,
            diag_dir / "render_stick.py",
            ["glb", args.store, clip_name, str(frame_start), str(frame_end),
             str(dense_step), str(stick_dir), clip_name],
        )
        produced = sorted(stick_dir.glob(f"{clip_name}-*.png"))
        if produced:
            mid_frame = (frame_start + frame_end) / 2.0
            def _frame_num(p: Path) -> int:
                return int(p.stem.rsplit("-f", 1)[1])
            keep = min(produced, key=lambda p: abs(_frame_num(p) - mid_frame))
            for extra in produced:
                if extra != keep:
                    extra.unlink()

    summary["findings"] = findings
    summary["ok"] = not findings
    (out_dir / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
