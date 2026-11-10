#!/usr/bin/env python3
"""Render a T2MESH02/03 (per-vertex PBR) binary mesh to a PNG for eyeballing the
demo's textured output. Mirrors the viewer's orientation-independent shading."""
import struct, sys, numpy as np
from PIL import Image

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "mesh_pbr.png"
b = open(path, "rb").read()
magic = b[:8]; nv, nt = struct.unpack("<II", b[8:16]); o = 16
V = np.frombuffer(b, "<f4", 3*nv, o).reshape(-1, 3).astype(np.float64); o += 12*nv
o += 12*nv  # skip normals (recompute from faces)
pbr = None
if magic == b"T2MESH03":
    pbr = np.frombuffer(b, "<f4", 6*nv, o).reshape(-1, 6).astype(np.float64); o += 24*nv
elif magic == b"T2MESH02":
    old = np.frombuffer(b, "<f4", 5*nv, o).reshape(-1, 5).astype(np.float64); o += 20*nv
    pbr = np.concatenate([old, np.ones((nv, 1))], axis=1)
F = np.frombuffer(b, "<i4", 3*nt, o).reshape(-1, 3)
C = pbr[:, 0:3] if pbr is not None else np.full((nv, 3), 0.65)
C_lin = np.clip(C, 0, 1) ** 2.2
metal = pbr[:, 3] if pbr is not None else np.zeros(nv)
print(f"{magic} {nv:,} verts  base_color mean {C.mean(0).round(3)}", flush=True)

fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
N = np.zeros_like(V)
for k in range(3): np.add.at(N, F[:, k], fn)
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9

SS = 2; W = H = 720 * SS
c = (V.max(0) + V.min(0)) / 2; Vc = V - c; Vc *= 0.92 / np.abs(Vc).max()
l1 = np.array([0.5, 0.8, 0.6]); l1 /= np.linalg.norm(l1)
l2 = np.array([-0.6, -0.2, -0.7]); l2 /= np.linalg.norm(l2)


def R(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


rough = pbr[:, 4] if pbr is not None else np.full(nv, 0.55)
# view-space metallic-roughness PBR, matching server/web/index.html
LC = [(np.array([.32, .55, .77]), np.array([1., .96, .88]), .85),
      (np.array([-.65, .20, .45]), np.array([.70, .80, 1.]), .45),
      (np.array([.10, -.75, .35]), np.array([.85, .85, .90]), .25)]


def render(el, az):
    r = R(el, az); P = Vc @ r.T; Nr = N @ r.T
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-9
    sx = ((P[:, 0]*.5+.5)*(W-1)).astype(np.int32); sy = ((.5-P[:, 1]*.5)*(H-1)).astype(np.int32)
    z = P[:, 2]
    V = np.array([0., 0., 1.])
    nv_ = np.abs(Nr @ V)
    F0 = 0.04*(1-metal)[:, None] + C_lin*metal[:, None]
    F = F0 + (1-F0)*((1-nv_)[:, None]**5)
    shin = 6.0 + (220.0-6.0)*np.clip(1-rough, 0, 1)**1.5
    sn = (shin+2.0)/(2*np.pi)
    diffuse = np.zeros((len(C), 3)); specular = np.zeros((len(C), 3))
    for Ld, Lc, w in LC:
        l = Ld/np.linalg.norm(Ld); ndl = np.abs(Nr @ l)
        diffuse += Lc[None]*w*ndl[:, None]
        h = l+V; h = h/np.linalg.norm(h); nh = np.abs(Nr @ h)
        specular += Lc[None]*w*(nh[:, None]**shin[:, None])*sn[:, None]*ndl[:, None]
    albedo = C_lin*(1-metal)[:, None]
    hemi = np.abs(N[:, 1])*0.5+0.5
    amb = (1-hemi)[:, None]*np.array([.20, .19, .22]) + hemi[:, None]*np.array([.55, .58, .66])
    col = np.clip(albedo*(amb*0.55 + diffuse*0.75) + specular*F + F*0.25, 0, 1)
    img = np.ones((H, W, 3))
    idx = np.argsort(z); sx, sy, col = sx[idx], sy[idx], col[idx]
    for dx in range(SS+1):
        for dy in range(SS+1):
            xx = sx+dx; yy = sy+dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            img[yy[m], xx[m]] = col[m]
    return (img**(1/2.2)*255).reshape(H//SS, SS, W//SS, SS, 3).mean((1, 3)).astype(np.uint8)


row = np.concatenate([render(12, 25), render(12, 150), render(65, 20)], 1)
Image.fromarray(row).save(out)
print("wrote", out, flush=True)
