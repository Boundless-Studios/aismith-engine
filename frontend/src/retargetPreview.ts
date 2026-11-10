import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js'

// Same donor library ClipBrowser.tsx previews, reused here for the instant
// client-side clip-preview-on-character feature (operator: "with
// mesh2motion, changing clips doesn't require regeneration, it immediately
// applies using the skeleton — can we do the same?"). Kept as its own
// constant (rather than importing ClipBrowser's) so this module has no
// dependency on a UI component.
export const DONOR_GLB_URL = '/animations/human-base-animations.glb'

// The donor's own bone names (Mesh2Motion convention) as they exist in the
// donor GLB's skeleton -- these are the same names used as KEYS in
// backend/workers/unirig_bind_worker.py's DONOR_TO_SOMA map, i.e. the
// donor's pre-Blender-import skeleton (this loads the donor GLB directly in
// the browser, so there is no Blender-import renaming or SOMA/BVH
// intermediate step to go through here, unlike the server-side bake).
//
// Target names verified against a REAL UniRig-produced rig
// (studio-data/artifacts/7cc47c022e024901a3d3bfced8fa7fa7.glb on the box,
// via /tmp/k5/glbinfo.py): pelvis, spine_1, spine_2, spine_3, neck, head,
// upper_arm_l/r, lower_arm_l/r, hand_l/r, thigh_l/r, calf_l/r, foot_l/r (plus
// many unnamed/finger "extra" bones this map intentionally leaves
// unmapped). The rig has no clavicle/shoulder bone and no toe/ball bone, so
// the donor's clavicle_l/r and ball_l/r are intentionally omitted -- an
// unmapped bone is simply left in its bind pose by SkeletonUtils, which is
// fine for a quick preview.
export const DONOR_TO_TARGET_BONE_NAMES: Record<string, string> = {
  pelvis: 'pelvis',
  spine_01: 'spine_1',
  spine_02: 'spine_2',
  spine_03: 'spine_3',
  neck_01: 'neck',
  head: 'head',
  upperarm_l: 'upper_arm_l',
  lowerarm_l: 'lower_arm_l',
  hand_l: 'hand_l',
  upperarm_r: 'upper_arm_r',
  lowerarm_r: 'lower_arm_r',
  hand_r: 'hand_r',
  thigh_l: 'thigh_l',
  calf_l: 'calf_l',
  foot_l: 'foot_l',
  thigh_r: 'thigh_r',
  calf_r: 'calf_r',
  foot_r: 'foot_r',
}

// SkeletonUtils' own `options.hip` (default `'hip'`) names the SOURCE's
// (donor's) hip bone -- not the target's, despite `options.names` itself
// being keyed by target bone name (see buildRetargetNameMap below). Both
// this donor (Mesh2Motion convention) and every UniRig target rig happen to
// name their hip-equivalent bone "pelvis", so one constant covers both
// directions here; passing it keeps the pelvis's own translation track
// (every other bone only carries rotation).
export const HIP_BONE_NAME = 'pelvis'

// three.js's SkeletonUtils.retargetClip wants its `options.names` map keyed
// by the TARGET bone name, valued with the corresponding SOURCE (donor)
// bone name -- the opposite direction from DONOR_TO_TARGET_BONE_NAMES above
// (which is written donor-first because that's the natural, verifiable
// direction to read against the two skeletons' own bone lists). Pure and
// unit-tested so the inversion itself never silently drops or collides an
// entry.
export function buildRetargetNameMap(donorToTarget: Record<string, string>): Record<string, string> {
  const targetToDonor: Record<string, string> = {}
  for (const [donorName, targetName] of Object.entries(donorToTarget)) targetToDonor[targetName] = donorName
  return targetToDonor
}

type DonorLibrary = {
  scene: THREE.Group
  clipsByName: Map<string, THREE.AnimationClip>
}

let donorLibraryPromise: Promise<DonorLibrary> | null = null

// Loads the curated donor GLB exactly once per page session and caches the
// promise -- ClipBrowser.tsx's own preview canvas and the main-viewport
// instant-apply preview (StudioViewport.tsx's RetargetPreview) both call
// this, so the 5.6MB donor only ever gets fetched+parsed a single time no
// matter how many consumers need it.
export function loadDonorLibrary(): Promise<DonorLibrary> {
  if (!donorLibraryPromise) {
    donorLibraryPromise = new GLTFLoader().loadAsync(DONOR_GLB_URL).then((gltf) => ({
      scene: gltf.scene,
      clipsByName: new Map(gltf.animations.map((clip) => [clip.name, clip])),
    }))
  }
  return donorLibraryPromise
}

// SkeletonUtils.retarget(Clip) reads `object.skeleton` directly off
// whatever is passed as source/target, so it needs the actual SkinnedMesh
// node, not a Scene/Group that merely contains one.
export function findSkinnedMesh(root: THREE.Object3D): THREE.SkinnedMesh | null {
  let found: THREE.SkinnedMesh | null = null
  root.traverse((node) => { if (!found && node instanceof THREE.SkinnedMesh) found = node })
  return found
}

// Retargets one donor AnimationClip onto `targetSkinnedMesh` using this
// module's verified bone-name map. Clones the donor's own scene per call
// (SkeletonUtils.clone, not a plain Object3D clone, so SkinnedMesh->bone
// bindings survive) so this never fights over shared bone transforms with
// ClipBrowser's own concurrently-playing preview mixer, which mutates the
// SAME cached donor scene's bones every frame.
export function retargetDonorClip(
  targetSkinnedMesh: THREE.SkinnedMesh,
  donorScene: THREE.Group,
  donorClip: THREE.AnimationClip,
): THREE.AnimationClip {
  const donorClone = SkeletonUtils.clone(donorScene) as THREE.Group
  const donorSkinnedMesh = findSkinnedMesh(donorClone)
  if (!donorSkinnedMesh) throw new Error('Donor library clone has no SkinnedMesh to retarget from')
  return SkeletonUtils.retargetClip(targetSkinnedMesh, donorSkinnedMesh, donorClip, {
    names: buildRetargetNameMap(DONOR_TO_TARGET_BONE_NAMES),
    hip: HIP_BONE_NAME,
  })
}
