#!/usr/bin/env python3
"""Load a GLB and render it three ways from vertex colour or a baked UV atlas.
Also prints structural stats (a sanity check that the glTF is well-formed).

    python3 scripts/render_glb.py in.glb [out.png]

Runs anywhere with trimesh + numpy + Pillow (e.g. the t2tex container).
"""
import sys, numpy as np, trimesh
from PIL import Image

path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "glb_render.png"
scene = trimesh.load(path, process=False)
mesh = scene if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())[0]

V = np.asarray(mesh.vertices, np.float64)
F = np.asarray(mesh.faces, np.int64)
visual_attrs = getattr(mesh.visual, "vertex_attributes", {})
if "color" in visual_attrs:
    raw = np.asarray(visual_attrs["color"])
    scale = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0
    base_v = np.asarray(raw[:, :3], np.float64) / scale
    # glTF COLOR_0 is already linear.
    base_linear_v = np.clip(base_v, 0, 1)
    custom = getattr(mesh, "vertex_attributes", {})
    if "_METALLIC_ROUGHNESS" in custom:
        mr = np.asarray(custom["_METALLIC_ROUGHNESS"])
        mr_scale = np.iinfo(mr.dtype).max if np.issubdtype(mr.dtype, np.integer) else 1.0
        metal_v = np.asarray(mr[:, 0], np.float64) / mr_scale
        rough_v = np.asarray(mr[:, 1], np.float64) / mr_scale
    else:
        metal_v = np.zeros(len(V)); rough_v = np.full(len(V), 0.6)
    print(f"GLB: {len(V):,} verts  {len(F):,} faces  COLOR_0 {raw.dtype}  "
          f"custom metalRough={'yes' if '_METALLIC_ROUGHNESS' in custom else 'no'}", flush=True)
else:
    uv = np.asarray(mesh.visual.uv, np.float64)
    mat = mesh.visual.material
    base_img = np.asarray(mat.baseColorTexture.convert("RGB"), np.float64) / 255.0
    try:
        mr_img = np.asarray(mat.metallicRoughnessTexture.convert("RGB"), np.float64) / 255.0
    except Exception:
        mr_img = None
    TH, TW = base_img.shape[:2]
    def sample(img, u, v):
        x = np.clip((u % 1.0) * (TW - 1), 0, TW - 1).astype(np.int32)
        y = np.clip((v % 1.0) * (TH - 1), 0, TH - 1).astype(np.int32)
        return img[y, x]
    base_v = sample(base_img, uv[:, 0], uv[:, 1])
    base_linear_v = np.clip(base_v, 0, 1) ** 2.2
    if mr_img is not None:
        rough_v = sample(mr_img, uv[:, 0], uv[:, 1])[:, 1]
        metal_v = sample(mr_img, uv[:, 0], uv[:, 1])[:, 2]
    else:
        rough_v = np.full(len(V), 0.6); metal_v = np.zeros(len(V))
    print(f"GLB: {len(V):,} verts  {len(F):,} faces  uv[{uv.min():.3f},{uv.max():.3f}]  "
          f"baseColor {base_img.shape}  metalRough {None if mr_img is None else mr_img.shape}", flush=True)

N = np.zeros_like(V)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
for k in range(3): np.add.at(N, F[:, k], fn)
N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9

SS = 2; W = H = 720 * SS
c = (V.max(0) + V.min(0)) / 2; Vc = V - c; Vc *= 0.92 / np.abs(Vc).max()
LC = [(np.array([.32, .55, .77]), np.array([1., .96, .88]), .85),
      (np.array([-.65, .20, .45]), np.array([.70, .80, 1.]), .45),
      (np.array([.10, -.75, .35]), np.array([.85, .85, .90]), .25)]


def Rmat(el, az):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


def render(el, az):
    r = Rmat(el, az); P = Vc @ r.T; Nr = N @ r.T
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-9
    sx = ((P[:, 0]*.5+.5)*(W-1)).astype(np.int32); sy = ((.5-P[:, 1]*.5)*(H-1)).astype(np.int32)
    z = P[:, 2]
    view = np.array([0., 0., 1.]); nv_ = np.abs(Nr @ view)
    F0 = 0.04*(1-metal_v)[:, None] + base_linear_v*metal_v[:, None]
    Fr = F0 + (1-F0)*((1-nv_)[:, None]**5)
    shin = 6.0 + 214.0*np.clip(1-rough_v, 0, 1)**1.5
    sn = (shin+2.0)/(2*np.pi)
    diffuse = np.zeros_like(base_v); specular = np.zeros_like(base_v)
    for Ld, Lc, w in LC:
        l = Ld/np.linalg.norm(Ld); ndl = np.abs(Nr @ l)
        diffuse += Lc[None]*w*ndl[:, None]
        h = l+view; h = h/np.linalg.norm(h); nh = np.abs(Nr @ h)
        specular += Lc[None]*w*(nh[:, None]**shin[:, None])*sn[:, None]*ndl[:, None]
    albedo = base_linear_v*(1-metal_v)[:, None]
    hemi = np.abs(N[:, 1])*0.5+0.5
    amb = (1-hemi)[:, None]*np.array([.20, .19, .22]) + hemi[:, None]*np.array([.55, .58, .66])
    col = np.clip(albedo*(amb*0.55 + diffuse*0.75) + specular*Fr + Fr*0.25, 0, 1)
    img = np.ones((H, W, 3))
    idx = np.argsort(z); sxi, syi, coli = sx[idx], sy[idx], col[idx]
    for dx in range(SS+1):
        for dy in range(SS+1):
            xx = sxi+dx; yy = syi+dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            img[yy[m], xx[m]] = coli[m]
    return (img**(1/2.2)*255).reshape(H//SS, SS, W//SS, SS, 3).mean((1, 3)).astype(np.uint8)


row = np.concatenate([render(12, 25), render(12, 150), render(65, 20)], 1)
Image.fromarray(row).save(out)
print("wrote", out, flush=True)
