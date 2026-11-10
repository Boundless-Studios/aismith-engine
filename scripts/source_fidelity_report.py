"""Score a mesh render against the source photo it was generated from.

Thin CLI wrapper around `native.measure_source_fidelity` for use on the GPU
box:

    .venv/bin/python scripts/source_fidelity_report.py \\
        --mesh character.glb --photo source.png --out out/fidelity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import native  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--camera-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = json.loads(args.camera_json.read_text(encoding="utf-8")) if args.camera_json else None
    report = native.measure_source_fidelity(
        args.mesh,
        args.photo,
        args.out,
        photo_mask_path=args.mask,
        fov_x_degrees=args.fov_x,
        yaw_degrees=args.yaw,
        pitch_degrees=args.pitch,
        camera=camera,
        samples=args.samples,
    )
    print(
        f"iou={report['silhouette_iou']} ssim={report['masked_ssim']} "
        f"face={report['face_ssim']} perceptual={report['perceptual_distance']} "
        f"luma_delta={report['masked_luma_delta']}"
    )
    print(
        f"clip_i={report['clip_i']} (face {report['clip_i_face']}) "
        f"dino_cos={report['dino_cos']} (face {report['dino_cos_face']}) "
        f"lpips={report['lpips']} (face {report['lpips_face']})"
    )
    # A None above is a missing backbone, not a failed comparison -- say which.
    for warning in report.get("metric_warnings") or []:
        print(f"note: {warning}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
