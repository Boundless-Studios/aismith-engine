#!/usr/bin/env bash
# Retarget debug loop: diagnose -> (fix) -> rebind+retarget -> evidence.
# Usage: bash scripts/retarget_debug_loop.sh <character.glb> <motion.bvh> [--frames 1,25%,50%,75%]
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHAR=${1:?character glb}; BVH=${2:?motion bvh}; shift 2 || true
OUT=$ROOT/studio-data/retarget-debug/$(date +%s)
mkdir -p "$OUT"
BLENDER=${BLENDER_EXE:-$HOME/tools/blender-4.2.14-linux-x64/blender}
echo "[1/3] diagnose"
"$BLENDER" --background --factory-startup --python "$ROOT/scripts/diagnose_retarget.py" -- "$CHAR" "$BVH" "$OUT" "$@" 2>&1 | tee "$OUT/blender.log" | grep -E "DIAGNOSE_DONE|COMPOSITION_ERROR|POSITION_ERROR|TWIST_ERROR|FOOT_CONTACT|Error|Traceback" | tail -60

echo "--- rest-direction mismatches (>30deg) ---"
"$ROOT/.venv/bin/python" - "$OUT/diagnose.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
bad = [r for r in t if r["rest_dir_deg"] > 30]
print(f"roles mapped: {len(t)}; rest-direction mismatches >30deg: {len(bad)}")
for r in bad[:12]:
    print("  MISMATCH", r["source"], "->", r["target"], "rest_dir_deg:", r["rest_dir_deg"])
PY

echo "--- composition_error (bake/round-trip fidelity, NOT acceptance evidence) ---"
"$ROOT/.venv/bin/python" - "$OUT/composition_error.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(f"overall_max: {p['overall_max']}deg (should be ~0 -- nonzero means a bake/export/reimport bug, not a pose-fidelity problem)")
PY

echo "--- pose_error (real acceptance metrics: position % of height, twist deg, foot contact) ---"
"$ROOT/.venv/bin/python" - "$OUT/pose_error.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
thresholds = p["thresholds"]
frame1_pct = p["frame1_position_max_pct"] * 100
overall_pct = p["position_overall_max_pct"] * 100
twist_max = p["twist_overall_max_deg"]
print(f"position: frame1_max={frame1_pct:.2f}% (threshold <= {thresholds['position_pct_max_frame1']*100:.0f}%)  overall_max={overall_pct:.2f}%")
print(f"twist:    overall_max={twist_max:.2f}deg (threshold <= {thresholds['twist_deg_max']:.0f}deg)")
foot = p["foot_contact"]
if foot["max_abs_delta_pct"] is not None:
    print(f"foot contact: max_abs_delta={foot['max_abs_delta_pct']*100:.2f}%  mean_abs_delta={foot['mean_abs_delta_pct']*100:.2f}%")
else:
    print("foot contact: no foot bones in driving mapping")

bad_position = []
for name, stats in sorted(p["position_summary"].items(), key=lambda kv: -(kv[1]["max"] or 0)):
    marker = " <-- >3%" if (stats["max"] or 0) > 0.03 else ""
    print(f"  position {name:16s} max={stats['max']*100:>6.2f}% mean={stats['mean']*100:>6.2f}%{marker}")
    if (stats["max"] or 0) > 0.03:
        bad_position.append(name)
print(f"bones exceeding 3% position error: {len(bad_position)} -> {bad_position}")

bad_twist = []
for name, stats in sorted(p["twist_summary"].items(), key=lambda kv: -(kv[1]["max"] or 0)):
    marker = " <-- >20deg" if (stats["max"] or 0) > 20 else ""
    print(f"  twist    {name:16s} max={stats['max']:>6.1f}deg mean={stats['mean']:>6.1f}deg{marker}")
    if (stats["max"] or 0) > 20:
        bad_twist.append(name)
print(f"bones exceeding 20deg twist: {len(bad_twist)} -> {bad_twist}")

gate_pass = frame1_pct <= thresholds['position_pct_max_frame1'] * 100 and twist_max <= thresholds['twist_deg_max']
print(f"ACCEPTANCE GATE (frame1 position <= 3%, twist <= 20deg): {'PASS' if gate_pass else 'FAIL'}")
PY

echo "[2/3] fix + rebind + retarget: apply worker changes, then POST rig-adjust + retarget-motion via API (see runbook)"
echo "[3/3] evidence: retarget job writes review frames to studio-data/artifacts; copy latest into $OUT"
echo "debug dir: $OUT"
