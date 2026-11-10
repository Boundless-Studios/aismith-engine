"""Survey `s3.bilateral_symmetry_p95` over a corpus of stored artifacts.

`pipeline_policy.SYMMETRY_MIRROR_ERROR_P95_THRESHOLD` is a PLACEHOLDER (see
the comment above it): 0.05 was picked from the per-pair advisory band, not
measured against real rigs. This is the tool that measurement is supposed to
come from. It walks a directory of artifact JSON sidecars (the same shape
`main.save_artifact` writes -- `{"id", "name", "path", "metadata": {...}}`),
picks the ones whose `metadata.stage` is `auto-rig` or `rig-adjust` and whose
`path` is a `.glb`, re-runs `rig_quality.analyze_rig_path` on each (Blender is
not required: bind-pose symmetry is pure numpy), and prints the resulting
`mirror_error_p95` distribution so a real threshold can be chosen.

Pure stdlib plus the backend package -- no bpy. The orchestrator runs this on
the GPU box, where the artifact store actually lives:

    .venv/bin/python scripts/calibrate_symmetry.py --dir studio-data/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import rig_quality  # noqa: E402


RIG_STAGES = ("auto-rig", "rig-adjust")


def find_candidates(artifacts_dir: Path) -> list[dict[str, Any]]:
    """Every artifact sidecar under `artifacts_dir` that is a rigged GLB."""
    candidates: list[dict[str, Any]] = []
    for sidecar in sorted(artifacts_dir.glob("*.json")):
        try:
            artifact = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metadata = artifact.get("metadata") or {}
        path = str(artifact.get("path") or "")
        if metadata.get("stage") not in RIG_STAGES:
            continue
        if not path.lower().endswith(".glb"):
            continue
        candidates.append(artifact)
    return candidates


def measure(artifact: dict[str, Any]) -> dict[str, Any]:
    """Re-run `analyze_rig_document`'s bilateral-symmetry measurement on one
    artifact's GLB. Returns a row for the table; never raises -- a rig this
    measurement cannot handle is reported as unmeasurable, same as it would
    be in the pipeline."""
    row = {
        "id": artifact.get("id"),
        "name": artifact.get("name"),
        "stage": (artifact.get("metadata") or {}).get("stage"),
        "mirror_error_p95": None,
        "pairs_outside_band": None,
        "worst_pair": None,
        "note": "",
    }
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file():
        row["note"] = "missing on disk"
        return row
    try:
        report = rig_quality.analyze_rig_path(path)
    except (OSError, ValueError, RuntimeError) as exc:
        row["note"] = f"analyze_rig_path failed: {exc}"
        return row
    symmetry = report.get("bilateral_symmetry") or {}
    if not symmetry.get("measurable"):
        row["note"] = "; ".join(symmetry.get("caveats", [])) or "not measurable"
        return row
    row["mirror_error_p95"] = symmetry.get("mirror_error_p95")
    row["pairs_outside_band"] = symmetry.get("pairs_outside_band_count")
    worst = (symmetry.get("worst_offenders") or [None])[0]
    if worst:
        row["worst_pair"] = f"{worst['left']}/{worst['right']} ({worst['mirror_error_normalized']})"
    return row


def render_table(rows: list[dict[str, Any]]) -> str:
    header = f"{'id':<34}{'stage':<12}{'p95':>9}{'outside':>9}  worst pair / note"
    lines = [header, "-" * len(header)]
    for row in rows:
        p95 = f"{row['mirror_error_p95']:.4f}" if row["mirror_error_p95"] is not None else "n/a"
        outside = str(row["pairs_outside_band"]) if row["pairs_outside_band"] is not None else "n/a"
        detail = row["worst_pair"] or row["note"]
        lines.append(f"{str(row['id']):<34}{str(row['stage']):<12}{p95:>9}{outside:>9}  {detail}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=str(ROOT / "studio-data" / "artifacts"),
        help="Directory of artifact JSON sidecars (default: studio-data/artifacts)",
    )
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.dir).expanduser()
    if not artifacts_dir.is_dir():
        print(f"No such artifacts directory: {artifacts_dir}", file=sys.stderr)
        return 1

    candidates = find_candidates(artifacts_dir)
    if not candidates:
        print(f"No auto-rig/rig-adjust GLB artifacts found under {artifacts_dir}")
        return 0

    rows = [measure(artifact) for artifact in candidates]
    measured = [row["mirror_error_p95"] for row in rows if row["mirror_error_p95"] is not None]
    print(render_table(rows))
    print()
    if measured:
        ordered = sorted(measured)
        print(
            f"{len(measured)}/{len(rows)} measurable -- "
            f"min={ordered[0]:.4f} max={ordered[-1]:.4f} "
            f"median={ordered[len(ordered) // 2]:.4f}"
        )
    else:
        print(f"0/{len(rows)} measurable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
