# Skeleton family registry

Per `docs/PIPELINE_POLICY.md` ("Skeleton family registry and the process for adding a
family"): each subdirectory here is a versioned family asset — template (bone set,
names, hierarchy, rest orientations), the ROM (range-of-motion) calibration clip
designation, and a README describing the family and its authoring provenance. S3
(skeleton fit) treats the template as law: fitters place the template's joints, they
never emit bespoke topology, extra bones, or renamed bones.

Planned families (initial registry from PIPELINE_POLICY.md):

| Family | Status | Notes |
|---|---|---|
| `biped` | **populated** (workstream N2) | standard humanoid; derived from the curated donor armature in `vendor/mesh2motion-app/static/animations/human-base-animations.glb` |
| `serpent` | not yet built | chain rig: `spine_00..N` + head, slither library |
| `quadruped-canine` | not yet built | dog-like: distinct from ungulate in spine flexibility, neck length, digitigrade legs |
| `quadruped-ungulate` | not yet built | deer-like: unguligrade legs |
| `bird` | not yet built | wings + tail + digitigrade legs |

## Loading a template

Use `backend.app.skeleton_template.load_template(family)` — do not parse
`template.json` ad hoc; the loader validates hierarchy and tier invariants
(exactly one root bone, no duplicate names, no dangling parent references) on load.

## Definition of done for a new family (from PIPELINE_POLICY.md)

1. Define the template + ROM clip + placement bands (design doc in the family dir).
2. Port or author a minimal clip set against the template.
3. Pick a golden source image; run S0->S6; every contract green **twice from
   scratch** (determinism check: identical artifact hashes).
4. Only then is the family available in the UI's subject picker.
