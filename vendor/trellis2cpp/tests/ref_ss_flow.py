#!/usr/bin/env python3
"""
Generate a reference forward pass of the TRELLIS.2 SS-flow DiT in float32, so the
C++ implementation can be validated against it bit-for-bit (modulo fp rounding).

Loads SparseStructureFlowModel directly from the checkpoint (no full pipeline),
runs forward(x, t, cond) on CPU in float32 with a fixed seed, and writes a
self-describing binary `ss_flow_ref.bin`:

  magic  : 8 bytes "SSFREF01"
  int32  : resolution, in_channels, out_channels, cond_tokens, cond_channels
  float32: t
  float32: x   [in_channels  * resolution^3]   channel-major (x[c*R^3 + n])
  float32: cond[cond_tokens  * cond_channels]  token-major (the .dinodata layout)
  float32: out[out_channels * resolution^3]    channel-major  (reference output)

Usage:
    python ref_ss_flow.py --dinodata /path/MushroomBoy.dinodata [--t 500] [--out ss_flow_ref.bin]
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
        version, dtype, ndim = struct.unpack("<III", f.read(12))
        shape = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(shape)
    return arr  # [1, tokens, channels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinodata", default=os.path.join(os.path.dirname(__file__), "..", "dumps", "fixture.dinodata"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint stem (no extension)")
    ap.add_argument("--t", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ss_flow_ref.bin"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel

    with open(args.ckpt + ".json") as f:
        cfg = json.load(f)["args"]
    print(f"building SparseStructureFlowModel: {cfg['num_blocks']} blocks, d={cfg['model_channels']}")

    torch.manual_seed(args.seed)
    model = SparseStructureFlowModel(**cfg)
    model.convert_to(torch.float32)  # blocks -> f32 AND sets self.dtype=f32 (else
                                     # forward's manual_cast downcasts activations)
    model.eval()
    sd = {k: v.float() for k, v in load_file(args.ckpt + ".safetensors").items()}
    # rope_phases is a computed buffer (not stored in the checkpoint).
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert missing == ["rope_phases"], f"unexpected missing keys: {missing}"

    R = cfg["resolution"]
    Cin = cfg["in_channels"]
    Cout = cfg["out_channels"]

    cond_np = load_dinodata(args.dinodata)  # [1, Lkv, Cctx]
    Lkv, Cctx = cond_np.shape[1], cond_np.shape[2]
    assert Cctx == cfg["cond_channels"], f"cond channels {Cctx} != {cfg['cond_channels']}"
    cond = torch.from_numpy(cond_np.copy()).float()

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal((1, Cin, R, R, R)).astype(np.float32)
    x = torch.from_numpy(x_np)
    t = torch.tensor([args.t], dtype=torch.float32)

    with torch.no_grad():
        out = model(x, t, cond)  # [1, Cout, R, R, R]
    out_np = out.detach().cpu().numpy().astype(np.float32)
    print(f"forward done. x{tuple(x_np.shape)} t={args.t} cond{tuple(cond_np.shape)} -> out{tuple(out_np.shape)}")
    print(f"out: min={out_np.min():.5f} max={out_np.max():.5f} mean={out_np.mean():.6f} l2={np.linalg.norm(out_np):.5f}")

    # Flatten to the C++ layouts.
    x_cm    = x_np.reshape(Cin, -1).reshape(-1)         # channel-major [Cin * R^3]
    out_cm  = out_np.reshape(Cout, -1).reshape(-1)      # channel-major [Cout * R^3]
    cond_tm = cond_np.reshape(-1)                        # token-major  [Lkv * Cctx]

    with open(args.out, "wb") as f:
        f.write(b"SSFREF01")
        f.write(struct.pack("<5i", R, Cin, Cout, Lkv, Cctx))
        f.write(struct.pack("<f", args.t))
        f.write(x_cm.astype("<f4").tobytes())
        f.write(cond_tm.astype("<f4").tobytes())
        f.write(out_cm.astype("<f4").tobytes())
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
