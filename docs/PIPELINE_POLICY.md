# Asset Pipeline Policy — contract-first, blame-upstream, deterministic

Operator directive (2026-08-30): a set of standard skeletons with a process for creating
them; a policy definition of "good and healthy" at every stage; any later-stage violation is
RCA'd to its owning upstream stage and remediated THERE; a deterministic loop that ends in a
well-behaved final product. This document is that policy. Agents working this repo follow it;
code that enforces it lives in `backend/app/pipeline_policy.py` (to be built — workstream N).

## Principles

1. **Contracts, not repairs.** Every stage declares measurable invariants on its output. A
   stage may only consume inputs that pass the previous stage's contract. "Repair" steps that
   compensate downstream for upstream violations are forbidden as permanent fixtures — they
   may exist only as temporary guards, each with a filed remediation for the owning stage.
2. **Blame upstream.** When stage k's output violates its contract, the defect is assigned to
   the earliest stage whose invariant would have prevented it (blame table below). The fix
   lands in that stage; downstream is re-run, never hand-adjusted to hide the symptom.
3. **Determinism.** Same inputs + same seed + same stage versions ⇒ same artifact. Every
   artifact records provenance (inputs, settings, seed, stage impl version). Re-running the
   loop after a remediation is mechanical: invalidate from the remediated stage, re-run
   downstream, re-evaluate contracts.
4. **Immutability.** Artifacts are never edited in place; a remediation produces new
   artifacts. The run's lineage is the audit trail for every RCA.
5. **Human judgment is a stage, not a patch.** Operator decisions (identity/likeness, final
   accept) are explicit gates with recorded outcomes; they never modify artifacts.

## Stages and contracts

**S0 Source & classification** — input image(s).
- Full subject visible; single subject; resolution floor.
- `subject_family` classified from the registry (below) with recorded confidence; ambiguity
  is an operator question at S0, not a downstream surprise.

**S1 Generation (image → raw mesh)** — generators are allowed to be dirty.
- Contract is only: recognizable single subject, textures present, provenance recorded
  (source views, seed, model). No topology requirements — that is S2's job, by design.

**S2 Canonicalization (mandatory, no bypass)** — raw mesh → game-ready mesh.
- One connected component (welded measure AND unwelded measure), watertight, 0 non-manifold
  edges, triangle budget within the family's band (default 10–50k), UV-unwrapped, uniform
  scale/axis convention (Z-up, facing −Y), floor-contact at z=0.
- **Appearance preservation is part of the contract**: textures are BAKED from S1 output;
  gate = image-similarity between S1 and S2 renders from N fixed viewpoints above a
  threshold. (Learned 08-30: re-generating the surface instead of baking destroyed the
  character's face — an S2 violation that must never ship.)
- Violations here are remediated in the canonicalizer. Never downstream.

*Canonical meshes are thin closed shells, not solids (measured 2026-09-01).* Mary's
canonical mesh is watertight, winding-consistent and genus 1, and encloses 0.0040 against a
0.2238 bounding box — **1.8% of its own bounding volume**, a wall roughly 0.005 units thick.
This is fine for everything the pipeline does today: rigging, skinning and animation only
ever need the surface, and S2's contract (one component, watertight, 0 non-manifold edges,
UV, scale, axis, floor contact) says nothing about interior volume. It matters for any
future consumer that assumes solid material — physics/collision volumes, boolean operations,
mass or buoyancy estimates, 3D printing — each of which would silently get a hollow answer.
`rig_quality.joint_containment.enclosed_volume_ratio` is recorded on every rig so the fact
stays visible rather than being rediscovered.

The concrete consequence today is in S3: `joints inside the mesh` cannot be answered with a
ray-parity ("odd crossings = inside solid material") test, which reports **0 of 21 joints
inside** on a correctly fitted rig, because every joint sits in the cavity rather than in
material. Joint containment therefore uses a shell-aware method instead — a joint counts as
inside when the mesh blocks both halves of all five probe axes — which is correct for a
solid and a shell alike.

**S3 Skeleton fit** — canonical mesh → rigged skeleton.
- The skeleton IS the family template: exact bone set, names, hierarchy, orientations/rolls.
  Fitters (UniRig or geometric) only PLACE the template's joints; they never emit bespoke
  topology, extra bones, or renamed bones.
- Placement invariants per family: joints inside the mesh, left/right symmetry bands,
  medial-line distance bounds, proportion sanity vs the template.
- "Cannot fit" ⇒ RCA to S2 (bad mesh) or S0 (misclassified family) — not a looser fit.

**S4 Skinning** — weights on the canonical mesh.
- Primary solver (bone-heat / learned) must succeed — envelope-style fallbacks are guards
  only. Max 4 influences; weights sum to 1; coincident vertices share identical weights
  (assertion, not a fixup); per-joint coverage within family bands.
- Gate: the family's **ROM (range-of-motion) calibration clip** plays with p99 edge stretch
  and self-intersection under family thresholds. ROM failure blames S4 first, S2 second
  (per the tearing-vs-weights blame rule below).

**S5 Animation** — clips applied to the rig.
- Clips are authored once against the family template and apply DIRECTLY (no per-asset
  retarget). Contract: exact bone-name match, root-motion policy honored, foot contact within
  band, p99 stretch under threshold on the actual clip.
- Generated motion (Kimodo) enters here as "a new clip for the template," passing the same
  gate as library clips.

**S6 Export / accept** — operator gate. Accept binds to the exact artifact.

## Skeleton family registry and the process for adding a family

Families are versioned assets in-repo: `assets/skeletons/<family>/` containing the template
(bone set, names, hierarchy, orientations), placement heuristics/bands, the ROM clip, the
clip library (ported once, offline), and a golden source image. Initial registry:
- `biped` (standard humanoid; exists de facto — formalize from the UniRig semantic set)
- `serpent` (chain; exists de facto — spine_00..N + head, slither library)
- `quadruped-canine` and `quadruped-ungulate` (dog-like vs deer-like differ in spine
  flexibility, neck length, digitigrade vs unguligrade legs — hence two templates)
- `bird` (wings + tail + digitigrade legs)

**Process for a new family (definition of done):**
1. Define the template + ROM clip + placement bands (design doc in the family dir).
2. Port or author a minimal clip set against the template.
3. Pick a golden source image; run S0→S6; every contract green **twice from scratch**
   (determinism check: identical artifact hashes).
4. Only then is the family available in the UI's subject picker.

## Blame table (symptom → owning stage)

| Symptom | Owning stage | Rationale / learned case |
|---|---|---|
| Seams split / tearing in pose | S2 (unwelded dupes) — S4 if S2 contract was green | 08-30 Mary hips |
| Wrong/mirrored limb naming | S3 (template conformance) | 08-29 mirrored rig |
| Hunch / inherited donor posture | S5 only if S3 green; else S3 orientations | 08-30 torso clamp |
| Limb roll/twist | S3 orientations (template rolls are law) | 08-30 elbow |
| Face/texture degraded | S2 bake gate | 08-30 AI-Refine regression |
| Stretch metric red but visual clean | The METRIC (fix measurement, file it) | seam-triangle pollution |
| Solver fallback engaged | S2 (mesh made the primary solver fail) | envelope weights saga |

## The deterministic loop

`run_pipeline(source, family)`: for each stage in order — execute, evaluate contract,
record verdict on the artifact. On violation: consult the blame table, open a remediation
against the owning stage (bead/issue with the violated invariant and evidence), STOP the run.
After remediation lands, re-run from the owning stage with preserved seeds; loop until S6
green. No manual artifact surgery inside the loop; operator gates are recorded verdicts.
The loop is the product: "well-behaved final product" = an S6-green run, reproducible from
its recorded provenance.
