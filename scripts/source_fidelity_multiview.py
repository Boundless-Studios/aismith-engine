"""Score a mesh render against a source photo from several views at once.

For a T-pose asset a single front-view fidelity number can hide drift that
only shows on the back or the sides, so this scores every view the caller
has a photo for and reports them together:

    .venv/bin/python scripts/source_fidelity_multiview.py \\
        --mesh character.glb --out out/multiview \\
        --front front.png --back back.png --left left.png --right right.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import native  # noqa: E402

# Matches view_projection.ROLE_AZIMUTH_DEGREES / SourceCamera's yaw convention.
ROLE_YAW_DEGREES = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}

# Subject-level only: the face variants (clip_i_face / dino_cos_face /
# lpips_face) would double the width for a table already at ten columns, so
# they stay in multiview.json, which carries every key.
TABLE_COLUMNS = [
    "role", "IoU", "subject SSIM", "face SSIM", "CLIP-I", "DINO", "LPIPS", "detail", "face detail", "luma Δ",
]
TABLE_KEYS = [
    "silhouette_iou",
    "masked_ssim",
    "face_ssim",
    "clip_i",
    "dino_cos",
    "lpips",
    "detail_ratio",
    "face_detail_ratio",
    "masked_luma_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--front", type=Path, default=None)
    parser.add_argument("--back", type=Path, default=None)
    parser.add_argument("--left", type=Path, default=None)
    parser.add_argument("--right", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--fov-x", type=float, default=None)
    return parser.parse_args()


def _cell(value) -> str:
    return "-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _render_table(reports: dict[str, dict]) -> str:
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |"]
    sums = {key: [] for key in TABLE_KEYS}
    for role, report in reports.items():
        row = [role] + [_cell(report.get(key)) for key in TABLE_KEYS]
        lines.append("| " + " | ".join(row) + " |")
        for key in TABLE_KEYS:
            value = report.get(key)
            if value is not None:
                sums[key].append(value)
    means = ["mean"] + [_cell(sum(values) / len(values)) if values else "-" for values in sums.values()]
    lines.append("| " + " | ".join(means) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    views = {
        role: path
        for role, path in {"front": args.front, "back": args.back, "left": args.left, "right": args.right}.items()
        if path is not None
    }
    if not views:
        raise SystemExit("source_fidelity_multiview: at least one of --front/--back/--left/--right is required")

    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    for role, photo in views.items():
        reports[role] = native.measure_source_fidelity(
            args.mesh,
            photo,
            args.out / role,
            yaw_degrees=ROLE_YAW_DEGREES[role],
            fov_x_degrees=args.fov_x or None,
            samples=args.samples,
        )

    (args.out / "multiview.json").write_text(
        json.dumps({role: report for role, report in reports.items()}, indent=2), encoding="utf-8"
    )
    print(_render_table(reports))
    # Otherwise a "-" in the CLIP-I column looks like a bug rather than a
    # backbone that isn't installed on this box.
    for warning in dict.fromkeys(w for report in reports.values() for w in (report.get("metric_warnings") or [])):
        print(f"note: {warning}")


if __name__ == "__main__":
    main()
