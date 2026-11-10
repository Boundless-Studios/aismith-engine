#!/usr/bin/env python3
"""Dump a GLB's baseColor + metallicRoughness atlas images side by side, so the
bake's colour fidelity can be eyeballed without any renderer in the way.

    python3 scripts/glb_atlas.py in.glb [out.png]
"""
import sys, numpy as np, trimesh
from PIL import Image

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "glb_atlas.png"
scene = trimesh.load(path, process=False)
mesh = scene if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())[0]
attrs = getattr(mesh.visual, "vertex_attributes", {})
if "color" in attrs:
    color = np.asarray(attrs["color"])
    print(f"vertex-colour GLB: {len(color):,} COLOR_0 values ({color.dtype}); no UV atlas")
    sys.exit(0)
mat = mesh.visual.material
bc = mat.baseColorTexture.convert("RGB")
try:
    mr = mat.metallicRoughnessTexture.convert("RGB")
except Exception:
    mr = Image.new("RGB", bc.size)
W, H = bc.size
# downscale for a quick look if huge
scale = 1024 / max(W, H)
if scale < 1:
    bc = bc.resize((int(W*scale), int(H*scale)))
    mr = mr.resize((int(W*scale), int(H*scale)))
w, h = bc.size
canvas = Image.new("RGB", (w*2 + 16, h), (30, 30, 30))
canvas.paste(bc, (0, 0)); canvas.paste(mr, (w + 16, 0))
canvas.save(out)
a = np.asarray(bc, np.float64) / 255
nz = a.reshape(-1, 3); nz = nz[nz.sum(1) > 0.05]
print(f"atlas {W}x{H}  baseColor mean(nonblack)={nz.mean(0).round(3) if len(nz) else 'n/a'}  "
      f"filled={len(nz)/(w*h)*100:.0f}%", flush=True)
print("wrote", out, flush=True)
