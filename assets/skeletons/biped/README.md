# biped skeleton family (v1)

Standard humanoid template. Load via `backend.app.skeleton_template.load_template("biped")`;
check a candidate rig against it via `backend.app.skeleton_template.conformance_report(...)`.

## Provenance

The bone set, hierarchy, rest orientations, and roll conventions are derived directly from
the curated donor armature in
`vendor/mesh2motion-app/static/animations/human-base-animations.glb`
(sha256 `406eb0a8dc4ab366e623b79b6e3005a4951392e1bda78ae39c1099d31147733c`, pinned as
`CURATED_SWORD_DONOR_SHA256` in `backend/app/native.py`) — the skeleton all 87 mesh2motion
library clips are authored on, so clips port to a conformant rig directly (no per-asset
retarget; see bead N4).

The donor's rest pose is already in the family's house frame (Z-up, -Y-facing, left = +X —
the same convention `skeleton_naming.py` derives from foot-toe direction), so no coordinate
conversion was applied when building the template.

## Bone set

21 required (core) bones + 45 optional bones (fingers, toes, and the donor's zero-influence
"leaf" tip markers — helper bones, per PIPELINE_POLICY.md's tiering). A rig missing any
required bone is non-conformant; missing optional bones is fine. An unrecognized bone name
present on the rig is always flagged, in either tier.

House names follow `skeleton_naming.py`'s semantic convention (`upper_arm_l/r`,
`lower_arm_l/r`, not `forearm` — matching what `semantic_joint_names` actually emits, so a
fitted-and-named rig's bone names compare directly against this template with no adapter
layer). `clavicle_l/r` are required (the donor has them, and a shoulder-conformant biped
needs them for arm placement). Fingers and toes are optional, per PIPELINE_POLICY.md's
explicit tiering ("optional helpers (fingers, toes, twist/helper bones)"), even though the
donor authors them.

Required (21):

| canonical name | parent | donor name |
|---|---|---|
| `root` | `None` | `root` |
| `pelvis` | `root` | `pelvis` |
| `spine_1` | `pelvis` | `spine_01` |
| `spine_2` | `spine_1` | `spine_02` |
| `spine_3` | `spine_2` | `spine_03` |
| `neck` | `spine_3` | `neck_01` |
| `head` | `neck` | `head` |
| `clavicle_l` | `spine_3` | `clavicle_l` |
| `upper_arm_l` | `clavicle_l` | `upperarm_l` |
| `lower_arm_l` | `upper_arm_l` | `lowerarm_l` |
| `hand_l` | `lower_arm_l` | `hand_l` |
| `clavicle_r` | `spine_3` | `clavicle_r` |
| `upper_arm_r` | `clavicle_r` | `upperarm_r` |
| `lower_arm_r` | `upper_arm_r` | `lowerarm_r` |
| `hand_r` | `lower_arm_r` | `hand_r` |
| `thigh_l` | `pelvis` | `thigh_l` |
| `calf_l` | `thigh_l` | `calf_l` |
| `foot_l` | `calf_l` | `foot_l` |
| `thigh_r` | `pelvis` | `thigh_r` |
| `calf_r` | `thigh_r` | `calf_r` |
| `foot_r` | `calf_r` | `foot_r` |

Optional (45):

| canonical name | parent | donor name |
|---|---|---|
| `head_tip` | `head` | `head_leaf` |
| `index_1_l` | `hand_l` | `index_01_l` |
| `index_2_l` | `index_1_l` | `index_02_l` |
| `index_3_l` | `index_2_l` | `index_03_l` |
| `index_tip_l` | `index_3_l` | `index_04_leaf_l` |
| `middle_1_l` | `hand_l` | `middle_01_l` |
| `middle_2_l` | `middle_1_l` | `middle_02_l` |
| `middle_3_l` | `middle_2_l` | `middle_03_l` |
| `middle_tip_l` | `middle_3_l` | `middle_04_leaf_l` |
| `pinky_1_l` | `hand_l` | `pinky_01_l` |
| `pinky_2_l` | `pinky_1_l` | `pinky_02_l` |
| `pinky_3_l` | `pinky_2_l` | `pinky_03_l` |
| `pinky_tip_l` | `pinky_3_l` | `pinky_04_leaf_l` |
| `ring_1_l` | `hand_l` | `ring_01_l` |
| `ring_2_l` | `ring_1_l` | `ring_02_l` |
| `ring_3_l` | `ring_2_l` | `ring_03_l` |
| `ring_tip_l` | `ring_3_l` | `ring_04_leaf_l` |
| `thumb_1_l` | `hand_l` | `thumb_01_l` |
| `thumb_2_l` | `thumb_1_l` | `thumb_02_l` |
| `thumb_3_l` | `thumb_2_l` | `thumb_03_l` |
| `thumb_tip_l` | `thumb_3_l` | `thumb_04_leaf_l` |
| `index_1_r` | `hand_r` | `index_01_r` |
| `index_2_r` | `index_1_r` | `index_02_r` |
| `index_3_r` | `index_2_r` | `index_03_r` |
| `index_tip_r` | `index_3_r` | `index_04_leaf_r` |
| `middle_1_r` | `hand_r` | `middle_01_r` |
| `middle_2_r` | `middle_1_r` | `middle_02_r` |
| `middle_3_r` | `middle_2_r` | `middle_03_r` |
| `middle_tip_r` | `middle_3_r` | `middle_04_leaf_r` |
| `pinky_1_r` | `hand_r` | `pinky_01_r` |
| `pinky_2_r` | `pinky_1_r` | `pinky_02_r` |
| `pinky_3_r` | `pinky_2_r` | `pinky_03_r` |
| `pinky_tip_r` | `pinky_3_r` | `pinky_04_leaf_r` |
| `ring_1_r` | `hand_r` | `ring_01_r` |
| `ring_2_r` | `ring_1_r` | `ring_02_r` |
| `ring_3_r` | `ring_2_r` | `ring_03_r` |
| `ring_tip_r` | `ring_3_r` | `ring_04_leaf_r` |
| `thumb_1_r` | `hand_r` | `thumb_01_r` |
| `thumb_2_r` | `thumb_1_r` | `thumb_02_r` |
| `thumb_3_r` | `thumb_2_r` | `thumb_03_r` |
| `thumb_tip_r` | `thumb_3_r` | `thumb_04_leaf_r` |
| `toes_l` | `foot_l` | `ball_l` |
| `toes_tip_l` | `toes_l` | `ball_leaf_l` |
| `toes_r` | `foot_r` | `ball_r` |
| `toes_tip_r` | `toes_r` | `ball_leaf_r` |

## ROM (range-of-motion) calibration clip

`Roll_Armature`, frames 0-44, joint-motion coverage 98.48% (65 of 66 bones move their
head->tail direction more than 5deg from frame 0 across the clip — metric_1 from
`/tmp/diag/motion_metrics.py`, reused rather than reinvented).

Scored all 87 donor actions; the top coverage score was an ~20-way tie (a lot of the
donor's combat/locomotion clips move essentially every articulated bone). Tiebreak: highest
sum of per-bone max angular delta across the clip (widest overall ROM among the tied
clips), then preferred the non-root-motion variant (`Roll_Armature` over `Roll_RM_Armature`,
which tied on both prior metrics) so the joint-articulation coverage this clip calibrates
isn't diluted by a translating root bone.

## Definition of done for this family

Per PIPELINE_POLICY.md, this bead (N2) covers only step 1 ("Define the template + ROM clip +
placement bands"). Steps 2-4 (port/author a clip set against the template, run S0->S6 twice
from scratch for a golden image, enable in the UI subject picker) are later beads.

## Surprises for later beads (N3 skeleton fit, N4 clip porting)

- The donor's `root` bone is **not** a vertical/anatomical segment: its rest pose runs
  head=(0,0,0) to tail=(0,0.918,0) — flat along the forward (-Y) axis, not up the spine. It
  reads as a scale/reference bone (its length is ~53% of total skeleton height) rather than a
  skeletal segment. Any placement heuristic for N3 that assumes "root points up" needs to
  special-case this bone.
- The donor has real clavicles and a full 5-finger, 3-segment (+ leaf tip) hand on both
  sides — no proxy/collapsed hand rig to work around.
- Every donor bone, including the finger and toe "leaf" tips, has nonzero rest length; none
  needed to be dropped as degenerate when building the template.
- Left/right bones are exact mirror images in rest pose (x negated, y/z identical, roll
  negated) — confirmed bone-by-bone, not just spot-checked, when generating this template.
