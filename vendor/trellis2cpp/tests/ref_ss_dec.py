#!/usr/bin/env python3
"""
Reference for the stage-1 SS decoder, to validate the C++ trellis2_ss_dec_decode
against the real SparseStructureDecoder.

Builds SparseStructureDecoder in float32 (lossless upcast of the fp16
checkpoint), decodes a sparse-structure latent z_s, and writes a
self-describing binary `ss_dec_ref.bin`:

  magic   : 8 bytes "SSDEC001"
  int32   : latent_channels, res_in, out_channels, res_out
  float32 : latent[latent_channels * res_in^3]   channel-major
  float32 : logits[out_channels * res_out^3]      channel-major

The latent comes from --latent (a .latent produced by ss_sample / ref_ss_sample)
if given; otherwise a fixed-seed standard-normal z_s is used so the C++ side can
feed the identical input.

Usage:
    python ref_ss_dec.py [--device mps|cpu] [--latent z_s.latent] [--seed N]
"""

import argparse
import json
import os
import struct
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SHIV = os.environ.get("TRELLIS2_PY", "/trellis2")
sys.path.insert(0, SHIV)

import numpy as np
import torch

DEFAULT_CKPT = os.environ.get("TRELLIS2_CKPT", os.path.join(
    os.path.dirname(__file__), "..", "models", "TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--latent", default=None, help="optional z_s .latent (channel-major float32)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ss_dec_ref.bin"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder

    dev = torch.device(args.device)
    with open(args.ckpt + ".json") as f:
        cfg = json.load(f)["args"]

    model = SparseStructureDecoder(**cfg)
    model.convert_to_fp32()   # set self.dtype=f32 AND convert torso modules
    model.eval()
    sd = {k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.to(dev)

    Cin = cfg["latent_channels"]
    Rin = 16
    Oc = cfg["out_channels"]

    if args.latent:
        z_cm = np.fromfile(args.latent, dtype="<f4")
        assert z_cm.size == Cin * Rin**3, (z_cm.size, Cin * Rin**3)
        z = torch.from_numpy(z_cm.reshape(1, Cin, Rin, Rin, Rin).copy()).float().to(dev)
        print(f"latent : loaded {args.latent}")
    else:
        g = torch.Generator().manual_seed(args.seed)
        z = torch.randn(1, Cin, Rin, Rin, Rin, generator=g).to(dev)
        print(f"latent : random seed={args.seed}")

    with torch.no_grad():
        logits = model(z)   # [1, Oc, Rout, Rout, Rout]

    Rout = logits.shape[-1]
    lg = logits.detach().cpu().numpy().astype(np.float32)
    print(f"logits : [{Oc},{Rout},{Rout},{Rout}] min={lg.min():.5f} max={lg.max():.5f} "
          f"mean={lg.mean():.6f}  occupied(>0)={(lg > 0).mean() * 100:.2f}%")

    z_out  = z.detach().cpu().numpy().astype(np.float32).reshape(Cin, -1).reshape(-1)
    lg_out = lg.reshape(Oc, -1).reshape(-1)

    with open(args.out, "wb") as f:
        f.write(b"SSDEC001")
        f.write(struct.pack("<4i", Cin, Rin, Oc, Rout))
        f.write(z_out.astype("<f4").tobytes())
        f.write(lg_out.astype("<f4").tobytes())
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
