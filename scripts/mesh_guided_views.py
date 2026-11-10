"""Headless driver for the mesh-guided view synthesis profiles.

Paints a pre-rendered clay render of the mesh (one per role, from `guides.json`)
as the reference character, using the photo as the identity reference --
mirrors the exact `run_comfyui_image_edit` call shape `scripts/source_pose_views.py`
uses for the pose-preserving family, with the clay render as `source` and the
photo passed through as `identity_image`.

Usage:
    python3 scripts/mesh_guided_views.py \\
        --guides path/to/guide_dir --photo path/to/source.jpg --out out_dir \\
        [--roles front,back,left,right,front-left,front-right,back-left,back-right] \\
        [--identity-map back=path/to/back.png,left=path/to/left.png] \\
        [--family mesh-guided] [--seed 7] [--comfyui http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.image_preparation import (  # noqa: E402
    resolve_view_synthesis_profile,
    run_comfyui_image_edit,
)

DEFAULT_ROLES = "front,back,left,right,front-left,front-right,back-left,back-right"

# Role-matched identity (lever 2): the 3/4 roles have no listed photo of
# their own most of the time, so they fall back to the nearest listed
# full role, unless the 3/4 role itself is listed.
_ROLE_FALLBACK = {
    "front-left": "front",
    "front-right": "front",
    "back-left": "back",
    "back-right": "back",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guides", required=True, type=Path)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roles", default=DEFAULT_ROLES)
    parser.add_argument(
        "--identity-map",
        default="",
        help="role=path[,role=path...]; a listed role uses that image as the "
        "identity reference instead of --photo (front-left/front-right fall "
        "back to front, back-left/back-right fall back to back, unless "
        "listed themselves).",
    )
    parser.add_argument("--family", default="mesh-guided")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--comfyui", default="http://127.0.0.1:8188")
    return parser.parse_args()


def parse_identity_map(raw: str) -> dict[str, Path]:
    identity_map: dict[str, Path] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        role, _, path = entry.partition("=")
        identity_map[role.strip()] = Path(path.strip())
    return identity_map


def resolve_identity_image(
    role: str, identity_map: dict[str, Path], default_photo: Path
) -> Path:
    if role in identity_map:
        return identity_map[role]
    fallback_role = _ROLE_FALLBACK.get(role)
    if fallback_role is not None and fallback_role in identity_map:
        return identity_map[fallback_role]
    return default_photo


def main() -> None:
    args = parse_args()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    identity_map = parse_identity_map(args.identity_map)
    args.out.mkdir(parents=True, exist_ok=True)
    # Same asset root `run_comfyui_image_edit` defaults to when called from
    # inside backend/app -- explicit here since this script lives elsewhere.
    pose_guide_root = REPO_ROOT / "backend" / "app"

    guides_manifest_path = args.guides / "guides.json"
    guides = json.loads(guides_manifest_path.read_text())["roles"]

    paths: dict[str, str] = {}
    profile_ids: dict[str, str] = {}
    for index, role in enumerate(roles):
        clay_path = Path(guides[role]["clay"])
        profile = resolve_view_synthesis_profile(role, args.family)
        identity_image = resolve_identity_image(role, identity_map, args.photo)
        destination = args.out / f"{role}.png"
        start = time.monotonic()
        run_comfyui_image_edit(
            clay_path,
            destination,
            profile=profile,
            seed=args.seed + index + 1,
            base_url=args.comfyui,
            identity_image=identity_image,
            pose_guide_root=pose_guide_root,
        )
        elapsed = time.monotonic() - start
        print(f"{role}: {destination} ({elapsed:.1f}s)")
        paths[role] = str(destination)
        profile_ids[role] = profile.id

    shutil.copyfile(guides_manifest_path, args.out / "guides.json")
    (args.out / "views.json").write_text(json.dumps(
        {
            **paths,
            "family": args.family,
            "seed": args.seed,
            "profile_ids": profile_ids,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
