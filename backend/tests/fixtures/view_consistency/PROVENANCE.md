# Provenance: view-consistency fixtures

The seven PNGs in this directory are Blender renders made for this
repository by `scripts/render_cc0_fixtures.py`. Their only inputs are two CC0
3D models vendored with Mesh2Motion and primitive geometry and flat colours
added by that script. No image generator, downloaded image or earlier
fixture was used.

The renders are released under the repository's MIT licence (`LICENSE`).
The source models are CC0 1.0 Universal.

## Regenerate

From the repository root:

```
blender --background --factory-startup --python scripts/render_cc0_fixtures.py
```

This rewrites these seven PNGs and the pose guide in
`backend/app/assets/pose-guides/`. Add `-- --only NAME ...` to render a
subset, or `-- --out-dir DIR` to write elsewhere. The committed files were
rendered with Blender 5.1.2 (build hash ec6e62d40fa9) on an Apple M4 Mac
running macOS 26.0.1, and two consecutive runs produced byte-identical
files. Another Blender build or GPU may round anti-aliased edge pixels
differently, so compare the hashes below before replacing a fixture.

## Render settings (all seven)

- Engine: Workbench, Studio lighting ("Default" studio light,
  camera-relative), no specular highlights, shadows, cavity or outline,
  32-sample anti-aliasing.
- Colour: Standard view transform, sRGB display, no look, exposure 0,
  gamma 1, no dithering. 8-bit RGB PNG, compression 15, no metadata stamps.
- Background: plain light grey, sRGB (232, 232, 232).
- Camera: orthographic, horizontal, centred on the posed figure's bounding
  box. The frame side is 1.10 times the figure's largest extent on any
  axis, so all views of one character share a scale. 256 x 256 px.
- Views: the figure faces -Y. Front looks from -Y, back from +Y, and left
  from +X (the figure's own left side, so it faces image-left).

## Sources

The Mesh2Motion app is vendored as plain files in `vendor/mesh2motion-app`.
`git log -1 --format=%H -- vendor/mesh2motion-app` at the time of rendering
is `3d60ae683c9ba9862b9725c9513e1832bced7724`; both GLBs below were last
changed in `8af574dd3011ff5668e572798a644996b79b93d8`.

Licence checks, made against the vendored files:

- `vendor/mesh2motion-app/LICENSE-CC0.MD` (sha256
  `52fec1c484df708c16b4c27b2bf0d7f0d3994105c57adc4b9e6f95bcfd9de65f`)
  dedicates "All 3d models, blend files, rigs, animations" to CC0 1.0
  Universal. `vendor/mesh2motion-app/README.md` ("Licenses") says the same.
- That blanket statement is not true of every model, so each model's own
  entry was also checked in `vendor/mesh2motion-app/src/lib/RigModelVariations.ts`
  (sha256 `1aeb33b6a62aa2edeb0663a4e41333b795117bf610df6c155589f8efed384e52`):
  `female` is "Female Mannequin", attribution "Quaternius", license "CC0";
  `zombie` is "Zombie", attribution "Kenney.nl", license "CC0". The models
  that are not CC0 (Sophia, CC-SA 4.0; Jay, Sintel and Bunny, CC-BY; Carrot,
  CC-BY) were not used.
- `THIRD_PARTY_NOTICES.md` records the Mesh2Motion models as CC0 1.0.

| Source | Path in this repository | Upstream author | Licence | sha256 of the GLB |
| --- | --- | --- | --- | --- |
| Female Mannequin | `vendor/mesh2motion-app/static/models-variation/human-female.glb` | Quaternius, via Mesh2Motion | CC0 1.0 | `2b1c47e5eeebffd5097eb8a52add4ba6556dab85e50fc1c5240d744099bebae1` |
| Zombie | `vendor/mesh2motion-app/static/models-variation/human-zombie.glb` | Kenney.nl, via Mesh2Motion | CC0 1.0 | `ad2102eda5bcb460ccd02b6dad3356075657276e9dd1c4b89c6b83a84163fc0d` |

The Zombie's colours come from its own embedded texture (`zombieA`, stored
inside the GLB). Each GLB also contains an unparented 42-vertex "Icosphere"
helper mesh that the script deletes before rendering.

## Images

Both models are in a T-pose at rest. "Jumpsuit" below means the script
replaces the Female Mannequin's two materials with flat colours assigned by
each face's rest-pose position: purple jumpsuit, skin-coloured head, neck
and hands, a dark belt at the waist and dark boots to mid-calf. "Robe" means
the jumpsuit plus a floor-length robe the script builds: an elliptical bell
mesh (64 segments, 40 rings) from the waist to the floor, 0.9 times as deep
as it is wide, with a gold-brown sash and hem band.

| File | Role in `test_view_consistency.py` | Source | What the script changed | View | sha256 of the PNG |
| --- | --- | --- | --- | --- | --- |
| `front-legs.png` | Front reference with two separated legs | Female Mannequin | Jumpsuit; each leg swung 4 degrees outward at the hip, feet turned back flat | front | `21092f82964c49af505abb7584ce7270a6c618855421cc3920a09d023eccc2ec` |
| `left-robe-drift.png` | Drifted left view: a robe where the front shows legs | Female Mannequin | Robe; rest pose | left | `a08bc089a6e9b40c1316612e68aede91cd9113df4cb1ffb2ca3bc6a162b99855` |
| `back-robe-drift.png` | Drifted back view: a robe where the front shows legs | Female Mannequin | Robe; rest pose | back | `82c727a54b8fad8b530623e548f722b7bc077a586a04a7c88a180d0ef585e393` |
| `front-robe-source.png` | Front of the robed figure (no leg separation to contradict) | Female Mannequin | Robe; rest pose | front | `04d3e4030b35f9fae1d8d99b8de3500fd670f00dd75ed82f83cba7e5bfa61988` |
| `front-zombie.png` | Known-good set, front | Zombie | None (rest pose, own texture) | front | `e82293644e41138cef37ea1799fe9dd67998331777d01fc6594c3296c89b7141` |
| `left-zombie-good.png` | Known-good set, left | Zombie | None (rest pose, own texture) | left | `ce5a8ede6ab4646f2f4eeed157a865a2c1e0112ca3d41f554b1ed7d034270d90` |
| `back-zombie-good.png` | Known-good set, back | Zombie | None (rest pose, own texture) | back | `efe74e010ef4dd29851cf1d9cb6636a37cab5614dcb04f9b6332ad0db5d15fae` |

## What the check measures on them

`backend.app.view_consistency.compare_view_to_front`, with the module's
constants unchanged:

| Front -> view | View | Front gap | View gap | Width ratio | Profile corr | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `front-legs` -> `left-robe-drift` | left | 1.000 | 0.000 | 2.462 | -0.763 | blocked (leg separation mismatch) |
| `front-legs` -> `back-robe-drift` | back | 1.000 | 0.000 | 2.762 | -0.763 | blocked (leg separation mismatch, plus a decorrelation warning) |
| `front-zombie` -> `left-zombie-good` | left | 0.720 | 0.000 | 0.548 | 0.615 | clean |
| `front-zombie` -> `back-zombie-good` | back | 0.720 | 0.720 | 1.000 | 1.000 | clean |
| `front-robe-source` -> `left-robe-drift` | left | 0.000 | 0.000 | 0.891 | 0.994 | clean |

The back view of the Zombie measures exactly like its front because an
orthographic back view of a left-right symmetric figure has the mirrored
silhouette, and every signal here is mirror-invariant.
