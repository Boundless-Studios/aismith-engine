"""Headless driver for the pose-preserving view synthesis profiles.

Mirrors the exact `run_comfyui_image_edit` call `_synthesize_view_into`
(backend/app/main.py) makes for the `synthesize-views` job, so a driven run
and a job run of the identity-first path produce identical artifacts.

Usage:
    python3 scripts/source_pose_views.py \\
        --photo path/to/source.jpg --out out_dir \\
        [--views back,left,right] [--seed 7] [--family source-pose] \\
        [--comfyui http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.image_preparation import (  # noqa: E402
    resolve_view_synthesis_profile,
    run_comfyui_image_edit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--views", default="back,left,right")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--family", default="source-pose")
    parser.add_argument("--comfyui", default="http://127.0.0.1:8188")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    # Same asset root `run_comfyui_image_edit` defaults to when called from
    # inside backend/app -- explicit here since this script lives elsewhere.
    pose_guide_root = REPO_ROOT / "backend" / "app"

    paths: dict[str, str] = {}
    profile_ids: dict[str, str] = {}
    for index, view in enumerate(views):
        profile = resolve_view_synthesis_profile(view, args.family)
        destination = args.out / f"{view}.png"
        start = time.monotonic()
        run_comfyui_image_edit(
            args.photo,
            destination,
            profile=profile,
            seed=args.seed + index + 1,
            base_url=args.comfyui,
            pose_guide_root=pose_guide_root,
        )
        elapsed = time.monotonic() - start
        print(f"{view}: {destination} ({elapsed:.1f}s)")
        paths[view] = str(destination)
        profile_ids[view] = profile.id

    (args.out / "views.json").write_text(json.dumps(
        {
            **paths,
            "profile_ids": profile_ids,
            "seed": args.seed,
            "family": args.family,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
