#!/usr/bin/env python3
"""
Reference for the full stage-1 flow-Euler sampling loop, to validate the C++
trellis2_ss_flow_sample against the real FlowEulerGuidanceIntervalSampler.

Builds SparseStructureFlowModel in float32, loads the DINOv3 cond from a
.dinodata file, draws fixed noise, runs the sampler, and writes a
self-describing binary `ss_sample_ref.bin`:

  magic   : 8 bytes "SSSAMP01"
  int32   : resolution, in_channels, cond_tokens, cond_channels, steps
  float32 : guidance_strength, guidance_rescale, gi_min, gi_max, rescale_t, sigma_min
  float32 : noise [in_channels * R^3]   channel-major
  float32 : cond  [cond_tokens * cond_channels] token-major
  float32 : latent[in_channels * R^3]   channel-major  (reference z_s)

Usage:
    python ref_ss_sample.py [--device mps|cpu] [--dinodata .../MushroomBoy.dinodata]
"""

import argparse
import json
import os
import struct
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SHIV = os.environ.get("TRELLIS2_PY", "/trellis2")
sys.path.insert(0, SHIV)

import numpy as np
import torch

DEFAULT_CKPT = os.environ.get("TRELLIS2_CKPT", os.path.join(
    os.path.dirname(__file__), "..", "models", "TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16"))


def load_dinodata(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"DINOCOND", "bad magic"
        _v, _d, ndim = struct.unpack("<III", f.read(12))
        shape = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(shape)
    return arr  # [1, tokens, channels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinodata", default=os.path.join(os.path.dirname(__file__), "..", "dumps", "fixture.dinodata"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ss_sample_ref.bin"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.pipelines import samplers

    dev = torch.device(args.device)
    with open(args.ckpt + ".json") as f:
        cfg = json.load(f)["args"]

    model = SparseStructureFlowModel(**cfg)
    model.convert_to(torch.float32)
    model.eval()
    sd = {k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and missing == ["rope_phases"], (missing, unexpected)
    model.to(dev)

    R, Cin = cfg["resolution"], cfg["in_channels"]
    cond_np = load_dinodata(args.dinodata)              # [1, Lkv, Cctx]
    Lkv, Cctx = cond_np.shape[1], cond_np.shape[2]
    cond = torch.from_numpy(cond_np.copy()).float().to(dev)
    neg_cond = torch.zeros_like(cond)

    # Fixed noise (CPU generator for reproducibility, then move to device).
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, Cin, R, R, R, generator=g).to(dev)

    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    params = dict(steps=12, rescale_t=5.0, guidance_strength=7.5,
                  guidance_interval=[0.6, 1.0], guidance_rescale=0.7)
    print(f"sampling on {dev}  steps={params['steps']} gs={params['guidance_strength']} "
          f"rescale={params['guidance_rescale']} interval={params['guidance_interval']}")
    with torch.no_grad():
        z_s = sampler.sample(model, noise, cond, neg_cond, verbose=True, **params).samples

    z_np     = z_s.detach().cpu().numpy().astype(np.float32)   # [1, Cin, R,R,R]
    noise_np = noise.detach().cpu().numpy().astype(np.float32)
    print(f"z_s: min={z_np.min():.5f} max={z_np.max():.5f} mean={z_np.mean():.6f} "
          f"l2={np.linalg.norm(z_np):.5f}  (occupancy>0: {(z_np>0).mean()*100:.2f}%)")

    noise_cm = noise_np.reshape(Cin, -1).reshape(-1)
    z_cm     = z_np.reshape(Cin, -1).reshape(-1)
    cond_tm  = cond_np.reshape(-1)

    with open(args.out, "wb") as f:
        f.write(b"SSSAMP01")
        f.write(struct.pack("<5i", R, Cin, Lkv, Cctx, params["steps"]))
        f.write(struct.pack("<6f", params["guidance_strength"], params["guidance_rescale"],
                            params["guidance_interval"][0], params["guidance_interval"][1],
                            params["rescale_t"], 1e-5))
        f.write(noise_cm.astype("<f4").tobytes())
        f.write(cond_tm.astype("<f4").tobytes())
        f.write(z_cm.astype("<f4").tobytes())
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
