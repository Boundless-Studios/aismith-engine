#!/usr/bin/env python3
"""Reference dumps for the PBR-texture stages, for validating the C++/ggml port.

Produces dumps/reference_texture.gguf with per-stage golden tensors so each C++
stage validates in isolation (feed identical inputs, compare outputs):

  cond            [T, Cc]     DINOv3 conditioning at texture resolution R
  enc_vert        [N, 3]      shape-encoder input: dual-vertex offsets  (QEF)
  enc_inter       [N, 3]      shape-encoder input: intersection flags   (QEF)
  enc_coords      [N, 4]      active voxels at R (batch-idx 0 + xyz)
  shape_slat      [Nl, 32]    shape encoder output (mean), the concat_cond
  shape_coords    [Nl, 4]     latent voxels at R/16
  tex_noise       [Nl, 32]    fixed sampling noise (seed)
  tex_flow_t500   [Nl, 32]    tex-flow forward at t=500 (concat_cond, f32)
  tex_slat        [Nl, 32]    full sampler output, denormalized
  pbr             [M, 6]      decoded PBR voxels (base_color,metal,rough,alpha), *0.5+0.5
  pbr_coords      [M, 4]      decoded voxels at R (should equal enc_coords set)
  shape_slat_mean/std  [32]   shape_slat_normalization (concat_cond in-norm)
  tex_slat_mean/std    [32]   tex_slat_normalization   (output de-norm)

This dump validates the standalone arbitrary-mesh texturing path, whose encoder
is fed a reproducible QEF dual grid. Integrated image-to-3D generation instead
retains the generated shape SLat and replays the shape decoder's subdivisions;
that wiring is covered by test_slat plus the sparse PBR sampling regression.

Run inside the reference container (real o-voxel), e.g.:
  docker exec t2tex bash -lc \
    'cd /work && TRELLIS2_PY=/trellis2 python scripts/dump_texture_reference.py \
       --mesh /s/mesh_shipped.bin --image dumps/fixture_rgba.png --resolution 512'
"""
import argparse, json, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o_voxel          # noqa: E402  real o-voxel BEFORE ref_common (skips its stub)
import o_voxel.convert  # noqa: E402
import ref_common       # noqa: E402
ref_common.setup()      # sdpa attention + pure-torch sparse conv

import numpy as np      # noqa: E402
import torch            # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image   # noqa: E402
import trimesh          # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder      # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel          # noqa: E402
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # noqa: E402
from trellis2.pipelines import Trellis2TexturingPipeline                  # noqa: E402
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # noqa: E402
from trellis2.modules import sparse as sp                                 # noqa: E402


def load_t2mesh(path):
    b = open(path, "rb").read()
    assert b[:8] == b"T2MESH01", b[:8]
    nv, nt = struct.unpack("<II", b[8:16]); o = 16
    V = np.frombuffer(b, "<f4", 3 * nv, o).reshape(-1, 3).copy(); o += 12 * nv
    o += 12 * nv  # skip normals
    F_ = np.frombuffer(b, "<i4", 3 * nt, o).reshape(-1, 3).copy()
    return V, F_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t", type=float, default=500.0)
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "reference_texture.gguf"))
    args = ap.parse_args()
    dev = "cuda"
    R = args.resolution
    CK = os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts")
    tpa = json.load(open(os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "texturing_pipeline.json")))["args"]

    def load_model(cls, stem, fp32_flags=False, **extra):
        cfg = json.load(open(os.path.join(CK, stem + ".json")))["args"]
        if fp32_flags:
            cfg.pop("use_fp16", None); cfg["use_fp16"] = False
        m = cls(**{**cfg, **extra})
        sd = {k: v.float() for k, v in load_file(os.path.join(CK, stem + ".safetensors")).items()}
        m.load_state_dict(sd)
        return m.eval().to(dev)

    print(f"loading models (res {R}) ...", flush=True)
    shape_enc = load_model(FlexiDualGridVaeEncoder, "shape_enc_next_dc_f16c32_fp16", fp32_flags=True)
    tex_dec = load_model(SparseUnetVaeDecoder, "tex_dec_next_dc_f16c32_fp16", fp32_flags=True)
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

    V, Ftri = load_t2mesh(args.mesh)
    mesh = trimesh.Trimesh(vertices=V, faces=Ftri, process=False)
    img = Image.open(args.image).convert("RGBA")
    print(f"mesh {len(V)} verts / {len(Ftri)} tris; image {img.size}", flush=True)

    caps = {}
    torch.manual_seed(args.seed)
    with torch.no_grad():
        image = pipe.preprocess_image(img)
        mesh = pipe.preprocess_mesh(mesh)
        cond = pipe.get_cond([image], R)
        caps["cond"] = cond["cond"][0]  # [T, Cc]

        # ── shape encoder (QEF dual grid input) ──────────────────────────────
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(), grid_size=R,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=True)
        enc_vert_feats = dual_vertices * R - voxel_indices          # offset in voxel
        vtx = sp.SparseTensor(
            feats=enc_vert_feats,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)).to(dev)
        inter = vtx.replace(intersected).to(dev)
        caps["enc_vert"] = vtx.feats
        caps["enc_inter"] = inter.feats.float()
        caps["enc_coords"] = vtx.coords.float()
        print(f"enc input: {vtx.feats.shape[0]} voxels at {R}^3", flush=True)

        shape_slat = shape_enc(vtx, inter)
        caps["shape_slat"] = shape_slat.feats
        caps["shape_coords"] = shape_slat.coords.float()
        print(f"shape_slat: {shape_slat.feats.shape} coords {shape_slat.coords.shape}", flush=True)

        # ── tex flow: forward parity point + full sampler ────────────────────
        s_std = torch.tensor(tpa["shape_slat_normalization"]["std"], device=dev)[None]
        s_mean = torch.tensor(tpa["shape_slat_normalization"]["mean"], device=dev)[None]
        shape_slat_n = shape_slat.replace((shape_slat.feats - s_mean) / s_std)
        caps["shape_slat_mean"] = s_mean[0]
        caps["shape_slat_std"] = s_std[0]

        Nl = shape_slat.coords.shape[0]
        g = torch.Generator(device=dev).manual_seed(args.seed)
        noise_feats = torch.randn(Nl, tex_flow.out_channels, generator=g, device=dev)
        caps["tex_noise"] = noise_feats
        x0 = shape_slat.replace(noise_feats.clone())

        out = tex_flow(x0, torch.tensor([args.t], device=dev), cond["cond"], concat_cond=shape_slat_n)
        caps["tex_flow_t500"] = out.feats
        print(f"tex_flow t={args.t}: mean={out.feats.mean().item():.6f} l2={out.feats.norm().item():.4f}", flush=True)

        slat = sampler.sample(
            tex_flow, x0, concat_cond=shape_slat_n,
            cond=cond["cond"], neg_cond=cond["neg_cond"],
            **tpa["tex_slat_sampler"]["params"], verbose=True).samples
        t_std = torch.tensor(tpa["tex_slat_normalization"]["std"], device=dev)[None]
        t_mean = torch.tensor(tpa["tex_slat_normalization"]["mean"], device=dev)[None]
        slat = slat.replace(slat.feats * t_std + t_mean)
        caps["tex_slat"] = slat.feats
        caps["tex_slat_mean"] = t_mean[0]
        caps["tex_slat_std"] = t_std[0]
        print(f"tex_slat: mean={slat.feats.mean().item():.5f} std={slat.feats.std().item():.5f}", flush=True)

        # ── tex decoder → PBR voxels ─────────────────────────────────────────
        pbr = tex_dec(slat) * 0.5 + 0.5
        caps["pbr"] = pbr.feats
        caps["pbr_coords"] = pbr.coords.float()
        print(f"pbr: {pbr.feats.shape} range [{pbr.feats.min():.3f},{pbr.feats.max():.3f}]", flush=True)

        # sanity: the tex decoder should reconstruct exactly the encoder's voxel set
        enc_set = set(map(tuple, vtx.coords[:, 1:].cpu().numpy().tolist()))
        pbr_set = set(map(tuple, pbr.coords[:, 1:].cpu().numpy().tolist()))
        print(f"voxel-set match: pbr=={len(pbr_set)} enc=={len(enc_set)} "
              f"symdiff={len(enc_set ^ pbr_set)} ({100.0*len(enc_set & pbr_set)/max(1,len(enc_set)):.2f}% common)",
              flush=True)

    import gguf
    writer = gguf.GGUFWriter(args.out, "reference")
    manifest = {"shapes": {}, "atol": 2e-3, "rtol": 2e-3, "resolution": R}
    for name, t in caps.items():
        a = t.detach().cpu().float().numpy()
        manifest["shapes"][name] = list(a.shape)
        writer.add_tensor(name, np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with open(os.path.join(ref_common.DUMPS, "manifest_texture.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes), {len(caps)} tensors", flush=True)


if __name__ == "__main__":
    main()
