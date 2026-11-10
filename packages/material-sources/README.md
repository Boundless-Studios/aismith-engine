# Material sources: Poly Haven

Shared, engine-independent adapter. A game project may already fetch materials
with its own scripts; this package is a reusable successor to such scripts, not
an automatic migration or rebake of existing assets.

```sh
python -m pip install -e 'packages/material-sources[test]'
python -m pytest packages/material-sources/tests
```

```python
import json
from pathlib import Path
from material_sources.polyhaven import search, lock_material, ingest

print(search("brick floor"))  # Powered by Poly Haven
lock = lock_material("brick_floor_003", resolution="2k")
Path("material-lock.json").write_text(json.dumps(lock, indent=2))
bundle = ingest(lock, Path("material-cache"))
print(bundle / "manifest.json")
```

Search metadata includes name, tags/categories and source credit. The lock records
source URL, authors, asset CC0 license, API terms URL, physical size (API millimetres
converted to metres), known/unmeasured scale, inventory/metadata digests, available
maps and exact selected resolution/format/URLs/MD5/byte counts. Required Diffuse,
OpenGL normal and Roughness must all exist at that resolution in PNG. Optional
AO/Displacement/Metal are included only when available. No silent resampling,
missing-required-map fallback or inferred metre scale.

Ingestion checks provider MD5/size and records local SHA-256 for every file. A
content-addressed directory is published only after all files pass. Re-ingestion
verifies existing bytes. Unknown hosts, redirects, corrupt/oversize downloads and
changed locks fail. Source snapshots can be retained by callers alongside their
hashes. This is deterministic ingestion of **locked bytes**, not a promise that
future live API responses or re-rendered materials are identical.

The live metadata endpoints were exercised for `brick_floor_003`; map downloads
were tested using controlled fixtures. Blender/Unity consumers below are a design,
not implemented or visually qualified in this slice.

## Next consumer contract: Blender -> Unity

Input: immutable `manifest.json`, its SHA-256, complete map SHA-256 verification,
explicit physical scale or a separately recorded override, UV convention, selected
shader/render-pipeline profile/version, engine version and export options.
Missing scale blocks claims of physical fidelity; do not assume one metre.

1. **Blender material worker.** Verify every file before editing the scene. Use
   Principled BSDF with base color tagged sRGB and Rough/AO/normal/displacement as
   Non-Color; route `nor_gl` through a Normal Map node. Set UV scale explicitly
   from scan width/height in metres (non-square scans preserve aspect). Keep AO
   separate rather than baking it into the source albedo. Keep displacement off
   unless a declared geometry/shader budget enables it. Preserve vertex-color-only
   assets unless an authored UV material route was explicitly selected.
2. **Export worker.** Preserve source maps; emit an engine conversion manifest with
   map/UV/scale hashes, channel semantics, exporter and deterministic conversion
   versions. For glTF, pack occlusion/roughness/metallic per its channel convention.
   Do not pass roughness unchanged to a Unity smoothness slot. Missing metallic is
   an explicitly recorded dielectric default; missing normal is not silently flat.
3. **Unity importer.** Use a profile for the actual render pipeline, not a universal
   shader assumption. For URP Lit, generate its documented metallic/smoothness and
   occlusion inputs (smoothness = 1 - roughness), record packing and texture importer
   settings and verify normal orientation against a known asymmetric fixture.
   Preserve asset GUIDs and use a content-derived destination; an unchanged lock
   must not create duplicate materials. Never change project color space.
4. **Acceptance gate.** Compare source bytes, exported channels and actual GPU
   readback, then canonical normal/roughness/scale swatches and game-camera captures
   under the project's real color space. Export/import success is only transfer
   fidelity. The consuming project's surface, motion, engine, atmosphere and human
   art acceptance remain independent gates. Check the project's colour space and
   each of its material routes (for example authored PBR and generated COLOR_0)
   before promoting any material.

Each consumer writes a new immutable provenance artifact with the material lock as
parent; failed conversions retain diagnostics without replacing live resources.
Unit tests for channel conversion come before Blender roundtrip, then Unity GPU
readback, then visual review. The first unvalidated boundary is Blender material
construction/physical-UV setup. This design intentionally does not claim it passed.

Sources checked 2026-09-10: [official API schema](https://github.com/Poly-Haven/Public-API/blob/master/swagger.yml),
[asset license](https://polyhaven.com/license),
[current API terms](https://raw.githubusercontent.com/Poly-Haven/Public-API/master/ToS.md).
The API client identifies AISmith and exposes “Powered by Poly Haven” credit.
Asset licensing and live API terms are recorded separately.
