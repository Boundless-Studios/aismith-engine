#!/usr/bin/env python3
"""
Reference PBR texturing, end-to-end, on OUR generated mesh -- the golden target
the C++/ggml texture port validates against, and a quick eyeball of the texture
NN stages.

Runs the real Trellis2TexturingPipeline NN stages (shape encoder -> tex SLAT
flow -> tex decoder) with sparse ops monkeypatched to pure torch (ref_common),
so it needs NO custom CUDA kernels except o-voxel's CPU mesh->dual-grid.

For the eyeball we skip the CUDA-only UV bake (nvdiffrast/cumesh/flexgemm) and
instead trilinear-sample the decoded 6-channel PBR voxels at each mesh vertex
(base_color / metallic / roughness), then dump a coloured mesh to render.

  python scripts/ref_texture.py --mesh <T2MESH01.bin> --image <rgba.png> \
      --resolution 512 --out dumps/tex_vcolor.bin
"""
import argparse, json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o_voxel  # noqa: E402  (import the REAL o-voxel BEFORE ref_common so its
import o_voxel.convert          # stub guard skips it; we need mesh_to_flexible_dual_grid)
import ref_common  # noqa: E402
ref_common.setup()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import trimesh  # noqa: E402

from safetensors.torch import load_file  # noqa: E402
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # noqa: E402
from trellis2.pipelines import Trellis2TexturingPipeline  # noqa: E402
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # noqa: E402


def load_t2mesh(path):
    b = open(path, "rb").read()
    assert b[:8] == b"T2MESH01", b[:8]
    nv, nt = struct.unpack("<II", b[8:16]); o = 16
    V = np.frombuffer(b, "<f4", 3 * nv, o).reshape(-1, 3).copy(); o += 12 * nv
    o += 12 * nv  # skip normals
    F = np.frombuffer(b, "<i4", 3 * nt, o).reshape(-1, 3).copy()
    return V, F


def trilinear_sample_sparse(feats, coords_xyz, query_vox):
    """feats [M,C], coords_xyz [M,3] int voxel coords, query_vox [Q,3] float.
    Returns (sampled [Q,C], weight [Q]) trilinear over present voxels."""
    dev = feats.device
    MULT = 4096
    enc = lambda c: (c[:, 0].long() * MULT + c[:, 1].long()) * MULT + c[:, 2].long()
    ck = enc(coords_xyz)
    order = torch.argsort(ck)
    ck_s = ck[order]; f_s = feats[order]
    base = torch.floor(query_vox)
    frac = query_vox - base
    out = torch.zeros(query_vox.shape[0], feats.shape[1], device=dev)
    wsum = torch.zeros(query_vox.shape[0], device=dev)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = base + torch.tensor([dx, dy, dz], device=dev, dtype=base.dtype)
                w = (frac[:, 0] if dx else 1 - frac[:, 0]) * \
                    (frac[:, 1] if dy else 1 - frac[:, 1]) * \
                    (frac[:, 2] if dz else 1 - frac[:, 2])
                cck = enc(corner)
                pos = torch.searchsorted(ck_s, cck).clamp(max=ck_s.shape[0] - 1)
                found = (ck_s[pos] == cck)
                out += (w * found)[:, None] * f_s[pos]
                wsum += w * found
    return out / wsum.clamp(min=1e-6)[:, None], wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "tex_vcolor.bin"))
    args = ap.parse_args()
    dev = "cuda"
    CK = os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts")
    tpa = json.load(open(os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "texturing_pipeline.json")))["args"]

    def load_model(cls, stem, **extra):
        cfg = json.load(open(os.path.join(CK, stem + ".json")))["args"]
        m = cls(**{**cfg, **extra})
        sd = {k: v.float() for k, v in load_file(os.path.join(CK, stem + ".safetensors")).items()}
        m.load_state_dict(sd)
        return m.eval().to(dev)

    R = args.resolution
    print(f"loading models (res {R}) ...", flush=True)
    shape_enc = load_model(FlexiDualGridVaeEncoder, "shape_enc_next_dc_f16c32_fp16")
    tex_dec = load_model(SparseUnetVaeDecoder, "tex_dec_next_dc_f16c32_fp16")
    tex_flow = load_model(SLatFlowModel, f"slat_flow_imgshape2tex_dit_1_3B_{R}_bf16", dtype="float32")

    dino = DinoV3FeatureExtractor(os.path.join(ref_common.MODELS, "dinov3-vitl16"), image_size=R)
    dino.model = dino.model.to(dev).eval()

    sampler = FlowEulerGuidanceIntervalSampler(**tpa["tex_slat_sampler"]["args"])
    pipe = Trellis2TexturingPipeline(
        models={"shape_slat_encoder": shape_enc, "tex_slat_decoder": tex_dec,
                f"tex_slat_flow_model_{R}": tex_flow},
        tex_slat_sampler=sampler,
        tex_slat_sampler_params=tpa["tex_slat_sampler"]["params"],
        shape_slat_normalization=tpa["shape_slat_normalization"],
        tex_slat_normalization=tpa["tex_slat_normalization"],
        image_cond_model=dino, rembg_model=None, low_vram=False,
    )
    pipe._device = dev

    V, F = load_t2mesh(args.mesh)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    img = Image.open(args.image).convert("RGBA")
    print(f"mesh {len(V)} verts / {len(F)} tris; image {img.size}", flush=True)

    torch.manual_seed(args.seed)
    with torch.no_grad():
        image = pipe.preprocess_image(img)
        mesh = pipe.preprocess_mesh(mesh)
        cond = pipe.get_cond([image], R)
        print("encoding shape SLAT (o-voxel dual grid + shape encoder) ...", flush=True)
        shape_slat = pipe.encode_shape_slat(mesh, R)
        print(f"shape_slat: {shape_slat.feats.shape} coords {shape_slat.coords.shape}", flush=True)
        print("sampling tex SLAT (flow, concat_cond) ...", flush=True)
        tex_slat = pipe.sample_tex_slat(cond, tex_flow, shape_slat)
        print("decoding tex SLAT -> PBR voxels ...", flush=True)
        pbr = pipe.decode_tex_slat(tex_slat)   # SparseTensor, 6ch, already *0.5+0.5
        print(f"pbr voxels: {pbr.feats.shape} coords {pbr.coords.shape} "
              f"range [{pbr.feats.min():.3f},{pbr.feats.max():.3f}]", flush=True)

        # per-vertex trilinear sample of the PBR voxels (mesh is now normalized to [-.5,.5])
        Vt = torch.from_numpy(mesh.vertices).float().to(dev)
        qvox = (Vt + 0.5) * R
        feats = pbr.feats.float()
        coords_xyz = pbr.coords[:, 1:].to(dev)
        vals, w = trilinear_sample_sparse(feats, coords_xyz, qvox)
        vals = vals.clamp(0, 1).cpu().numpy()
        hit = (w > 1e-4).float().mean().item()
        print(f"per-vertex sample hit-rate {hit*100:.1f}%", flush=True)

    base_color = vals[:, 0:3]
    metallic = vals[:, 3:4]; roughness = vals[:, 4:5]
    Vout = mesh.vertices.astype("<f4")
    Fout = mesh.faces.astype("<i4")
    with open(args.out, "wb") as f:
        f.write(b"T2VCOL01")
        f.write(struct.pack("<II", len(Vout), len(Fout)))
        f.write(Vout.tobytes())
        f.write(base_color.astype("<f4").tobytes())
        f.write(np.concatenate([metallic, roughness], 1).astype("<f4").tobytes())
        f.write(Fout.tobytes())
    print(f"wrote {args.out}  (verts+rgb+metal/rough+tris)", flush=True)
    print(f"base_color mean {base_color.mean(0)}  metallic {metallic.mean():.3f}  rough {roughness.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
