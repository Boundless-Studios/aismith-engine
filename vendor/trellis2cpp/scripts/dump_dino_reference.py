#!/usr/bin/env python3
"""Dump DINOv3 ViT-L/16 reference activations for the C++ port.

Replicates DinoV3FeatureExtractor exactly (manual embeddings -> rope -> layer
loop -> affine-free layer_norm; the model's own final layernorm is NOT applied)
on the preprocessed fixture image, and writes:

  dumps/reference_dino.gguf   input pixels + per-layer taps + final cond
  dumps/manifest_dino.json    shapes + tolerances
  dumps/fixture.dinodata      the conditioning tensor for the SS-flow tests
  dumps/fixture_pre.png       preprocessed (cropped, premultiplied) image
  dumps/fixture_512.png       the exact 512x512 LANCZOS-resized uint8 image

Run inside the reference container (see scripts/refgen.sh):
  python scripts/dump_dino_reference.py --image <rgba image> [--device cuda]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402  (sets sys.path for trellis2, stubs cumesh)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGBA input image")
    ap.add_argument("--model", default=os.path.join(ref_common.MODELS, "dinov3-vitl16"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "reference_dino.gguf"))
    args = ap.parse_args()

    os.makedirs(ref_common.DUMPS, exist_ok=True)

    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel.from_pretrained(args.model)
    model.eval().float().to(args.device)

    img = Image.open(args.image).convert("RGBA")
    img.save(os.path.join(ref_common.DUMPS, "fixture_rgba.png"))
    pre = ref_common.preprocess_rgba(img)
    pre.save(os.path.join(ref_common.DUMPS, "fixture_pre.png"))

    resized = pre.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
    resized.save(os.path.join(ref_common.DUMPS, "fixture_512.png"))

    x = np.array(resized).astype(np.float32) / 255.0            # HWC
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)       # 1CHW
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = ((x - mean) / std).to(args.device)

    caps = {}
    caps["img_512_u8"] = torch.from_numpy(np.array(resized).astype(np.float32))
    caps["pixel_values"] = pixel_values

    # Detail taps inside the first and last layer via forward hooks.
    detail_layers = {0, len(model.layer) - 1}
    hooks = []

    def tap(name):
        def fn(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[name] = o.detach()
        return fn

    for i in sorted(detail_layers):
        layer = model.layer[i]
        for sub in ("norm1", "attention", "layer_scale1", "norm2", "mlp", "layer_scale2"):
            m = getattr(layer, sub, None)
            if m is not None:
                hooks.append(m.register_forward_hook(tap(f"l{i}.{sub}")))

    with torch.no_grad():
        hidden = model.embeddings(pixel_values, bool_masked_pos=None)
        caps["embd"] = hidden
        rope = model.rope_embeddings(pixel_values)
        if isinstance(rope, (tuple, list)):
            for j, r in enumerate(rope):
                caps[f"rope_{j}"] = r
        else:
            caps["rope_0"] = rope
        for i, layer_module in enumerate(model.layer):
            hidden = layer_module(hidden, position_embeddings=rope)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            caps[f"l{i}.out"] = hidden
        cond = F.layer_norm(hidden, hidden.shape[-1:])
        caps["cond"] = cond

    for h in hooks:
        h.remove()

    cond_np = cond.cpu().numpy().astype(np.float32)
    ref_common.write_dinodata(os.path.join(ref_common.DUMPS, "fixture.dinodata"), cond_np)
    print(f"cond: shape={tuple(cond_np.shape)} mean={cond_np.mean():.6f} "
          f"min={cond_np.min():.4f} max={cond_np.max():.4f} l2={np.linalg.norm(cond_np):.4f}")

    # Also emit the 1024-resolution conditioning (4101 tokens) that the HR stage
    # of the 1024 cascade consumes. Same encode path, image_size 1024.
    resized_hr = pre.resize((1024, 1024), Image.Resampling.LANCZOS)
    xhr = np.array(resized_hr).astype(np.float32) / 255.0
    xhr = torch.from_numpy(xhr).permute(2, 0, 1).unsqueeze(0)
    pv_hr = ((xhr - mean) / std).to(args.device)
    with torch.no_grad():
        h = model.embeddings(pv_hr, bool_masked_pos=None)
        rope_hr = model.rope_embeddings(pv_hr)
        for layer_module in model.layer:
            h = layer_module(h, position_embeddings=rope_hr)
            if isinstance(h, tuple):
                h = h[0]
        cond_hr = F.layer_norm(h, h.shape[-1:]).cpu().numpy().astype(np.float32)
    ref_common.write_dinodata(os.path.join(ref_common.DUMPS, "fixture_1024.dinodata"), cond_hr)
    print(f"cond_1024: shape={tuple(cond_hr.shape)} l2={np.linalg.norm(cond_hr):.4f}")

    import gguf
    writer = gguf.GGUFWriter(args.out, "reference")
    manifest = {"resolution": args.resolution, "atol": 2e-3, "rtol": 2e-3, "shapes": {}}
    for name, t in caps.items():
        a = t.detach().cpu().float().numpy()
        manifest["shapes"][name] = list(a.shape)
        writer.add_tensor(name, np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with open(os.path.join(ref_common.DUMPS, "manifest_dino.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes), "
          f"{len(caps)} tensors")


if __name__ == "__main__":
    main()
