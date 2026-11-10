# Provenance: pose guides

`humanoid-t-pose-v2.png` is a pose reference for the
`humanoid-t-pose-rig-v1` preparation profile in
`backend/app/image_preparation.py`. That profile sends its guide to the
image-edit model (`pose_guide_asset`), and job metadata records the guide's
path and sha256.

It replaces `humanoid-t-pose-v1.png`, which has no recorded provenance. The
public repository carries only v2. This repository keeps v1 active, through
`active.json`, until a controlled image-edit run accepts v2. To opt in to v2,
set `AISMITH_T_POSE_GUIDE=humanoid-t-pose-v2.png`. After that acceptance,
point `active.json` at v2 and delete v1.

It is a Blender render made for this repository by
`scripts/render_cc0_fixtures.py`. Its only input is a CC0 3D model vendored
with Mesh2Motion. No image generator, downloaded image or earlier guide was
used. The render is released under the repository's MIT licence (`LICENSE`).
The source model is CC0 1.0 Universal.

## Source

| | |
| --- | --- |
| Model | "Mannequin" |
| Path in this repository | `vendor/mesh2motion-app/static/models-variation/human-base.glb` |
| Upstream author | Quaternius, via Mesh2Motion |
| Licence | CC0 1.0 Universal |
| sha256 of the GLB | `c7c445f4309d8883667ca9f85ef6ba226c71f492c827af115c46c52bc450a019` |
| Mesh2Motion tree | `git log -1 --format=%H -- vendor/mesh2motion-app` = `3d60ae683c9ba9862b9725c9513e1832bced7724` (the GLB was last changed in `8af574dd3011ff5668e572798a644996b79b93d8`) |

Licence checks, made against the vendored files:

- `vendor/mesh2motion-app/LICENSE-CC0.MD` (sha256
  `52fec1c484df708c16b4c27b2bf0d7f0d3994105c57adc4b9e6f95bcfd9de65f`)
  dedicates "All 3d models, blend files, rigs, animations" to CC0 1.0
  Universal; `vendor/mesh2motion-app/README.md` ("Licenses") says the same.
- Not every Mesh2Motion model is CC0, so the model's own entry was checked
  too: in `vendor/mesh2motion-app/src/lib/RigModelVariations.ts` (sha256
  `1aeb33b6a62aa2edeb0663a4e41333b795117bf610df6c155589f8efed384e52`),
  variant `base` is "Mannequin", attribution "Quaternius", license "CC0".
- `THIRD_PARTY_NOTICES.md` records the Mesh2Motion models as CC0 1.0.

## What the script changed

- Deleted the GLB's unparented 42-vertex "Icosphere" helper mesh.
- Kept the model's rest pose, which is already a T-pose with the arms
  horizontal at shoulder height and the hands flat, palms down.
- Swung each straight leg 3.5 degrees outward at the hip and turned each
  foot back by the same angle so the soles stay flat.
- Raised each thumb into the horizontal plane of its hand (at rest the
  thumbs angle down and forward, which reads as a hooked hand from the
  front).
- Replaced the model's palette texture with one flat light-grey material
  (sRGB 240, 240, 240) and added one level of Catmull-Clark subdivision
  after the armature deform.

## Render

- Command, from the repository root:
  `blender --background --factory-startup --python scripts/render_cc0_fixtures.py`
  (this also rewrites the view-consistency fixtures; add
  `-- --only humanoid-t-pose-v2` for just this image).
- Blender 5.1.2 (build hash ec6e62d40fa9) on an Apple M4 Mac running macOS
  26.0.1; two consecutive runs produced byte-identical files.
- Engine: Workbench, Studio lighting ("Default" studio light,
  camera-relative), no specular highlights, shadows, cavity or outline,
  32-sample anti-aliasing. Standard view transform, sRGB display, no
  dithering. 8-bit RGB PNG, no metadata stamps.
- Camera: front view, 85 mm lens on a 36 mm sensor, eye height 1.44 m (the
  arms' height, so they stay horizontal in the frame) with a vertical lens
  shift to centre the figure, which also shows the tops of the feet. The
  frame side is 1.08 times the figure's largest extent.
- Output: 1254 x 1254 px, plain white background (255, 255, 255).
- sha256 of `humanoid-t-pose-v2.png`:
  `d078a2fff238137b1b1371b9e4174b0f14bcde45f2239d09b58e40ab3d8b0405`

The guide is an articulated artist's mannequin with visible joint segments,
not the smooth anatomical figure v1 showed. It has not been tried with the
image-edit model.
