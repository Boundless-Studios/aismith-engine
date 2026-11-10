#!/usr/bin/env python3
"""Render a T2VCOL01 coloured mesh (verts + per-vertex base_color) with a numpy
z-buffer, a few views, for eyeballing the PBR texture NN output."""
import struct, sys, numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "dumps/tex_vcolor.bin"
out = sys.argv[2] if len(sys.argv) > 2 else "dumps/tex_vcolor.png"
b = open(path, "rb").read()
assert b[:8] == b"T2VCOL01", b[:8]
nv, nt = struct.unpack("<II", b[8:16]); o = 16
V = np.frombuffer(b, "<f4", 3 * nv, o).reshape(-1, 3).astype(np.float64); o += 12 * nv
C = np.frombuffer(b, "<f4", 3 * nv, o).reshape(-1, 3).astype(np.float64); o += 12 * nv
o += 8 * nv  # skip metal/rough
F = np.frombuffer(b, "<i4", 3 * nt, o).reshape(-1, 3)
print(f"{nv:,} verts {nt:,} tris  color mean {C.mean(0)}", flush=True)

# face normals -> per-vertex (for a little shading on top of albedo)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
N = np.zeros_like(V)
for k in range(3):
    np.add.at(N, F[:, k], fn)
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9

SS = 2; W = H = 640 * SS
c = (V.max(0) + V.min(0)) / 2; Vc = V - c; Vc *= 0.9 / np.abs(Vc).max()
l1 = np.array([0.5, 0.8, 0.6]); l1 /= np.linalg.norm(l1)
l2 = np.array([-0.6, -0.2, -0.7]); l2 /= np.linalg.norm(l2)


def R(el, az):
    a = np.radians(az); e = np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


def render(el, az):
    r = R(el, az); P = Vc @ r.T; Nr = N @ r.T
    sx = ((P[:, 0] * .5 + .5) * (W - 1)).astype(np.int32)
    sy = ((.5 - P[:, 1] * .5) * (H - 1)).astype(np.int32)
    z = P[:, 2]
    shade = (np.abs(Nr @ l1) * .7 + np.abs(Nr @ l2) * .3) * 0.6 + 0.4  # gentle, keep albedo
    col = np.clip(C * shade[:, None], 0, 1)
    img = np.ones((H, W, 3)); zb = np.full((H, W), -1e9)
    idx = np.argsort(z); sx, sy, z, col = sx[idx], sy[idx], z[idx], col[idx]
    for dx in range(SS + 1):
        for dy in range(SS + 1):
            xx = sx + dx; yy = sy + dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            img[yy[m], xx[m]] = col[m]; zb[yy[m], xx[m]] = z[m]
    im = (img ** (1 / 2.2) * 255).reshape(H // SS, SS, W // SS, SS, 3).mean((1, 3))
    return im.astype(np.uint8)


row = np.concatenate([render(15, 25), render(15, 120), render(70, 20)], 1)
Image.fromarray(row).save(out)
print("wrote", out, flush=True)
