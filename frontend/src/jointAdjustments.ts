// Pure state helpers for the Skeleton adjust workspace (RigAdjustViewport).
// Kept free of THREE.js so the undo/adjustment/removal bookkeeping — the part
// that was actually buggy (Z-undo silently did nothing for a plain drag) —
// can be unit-tested without a WebGL context.

export type JointPosition = [number, number, number]
export type JointAdjustments = Record<string, JointPosition>

const EPSILON = 1e-5

function distance(a: JointPosition, b: JointPosition): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

// Called whenever a joint's position settles (end of a drag, or a reset).
// Returns the updated adjustments record: present only when the joint now
// differs from its predicted position by more than numerical noise.
export function recordJointAdjustment(
  adjustments: JointAdjustments,
  name: string,
  current: JointPosition,
  original: JointPosition,
): JointAdjustments {
  const next = { ...adjustments }
  if (distance(current, original) > EPSILON) next[name] = current
  else delete next[name]
  return next
}

export function adjustedJointCount(adjustments: JointAdjustments): number {
  return Object.keys(adjustments).length
}

export function clearJointAdjustments(): JointAdjustments {
  return {}
}

// --- Undo history ---------------------------------------------------------
//
// One entry per completed gesture (a finished drag, or a reset) capturing
// the position immediately BEFORE that gesture, so Z always undoes the last
// thing that actually happened — including a plain drag, which previously
// never pushed anything onto the stack at all.
//
// An entry holds a *group* of moves because one gesture can legitimately
// change more than one joint's local position: a single-joint drag pins its
// children in world space, which means rewriting each child's local
// position. Undo has to put the whole group back, or the second Z would
// leave the children behind.
export type JointMove = { name: string; position: JointPosition }
export type JointUndoEntry = { moves: JointMove[] }

export function pushJointUndo(stack: JointUndoEntry[], name: string, position: JointPosition): JointUndoEntry[] {
  return pushJointUndoGroup(stack, [{ name, position }])
}

// One undo step covering every joint a single gesture moved.
export function pushJointUndoGroup(stack: JointUndoEntry[], moves: JointMove[]): JointUndoEntry[] {
  if (!moves.length) return stack
  return [...stack, { moves: moves.map((move) => ({ name: move.name, position: [...move.position] as JointPosition })) }]
}

export function popJointUndo(stack: JointUndoEntry[]): { entry: JointUndoEntry | undefined; stack: JointUndoEntry[] } {
  if (!stack.length) return { entry: undefined, stack }
  return { entry: stack[stack.length - 1], stack: stack.slice(0, -1) }
}

// --- Joint removal ----------------------------------------------------------

// child bone name -> parent bone name (or null for a root bone)
export type BoneParentMap = Record<string, string | null>

// A bone and every descendant of it, root-first (removing a joint always
// removes its whole subtree — a dangling child with no parent is invalid).
export function collectSubtree(boneName: string, parentOf: BoneParentMap): string[] {
  const childrenOf = new Map<string, string[]>()
  for (const [child, parent] of Object.entries(parentOf)) {
    if (parent === null) continue
    if (!childrenOf.has(parent)) childrenOf.set(parent, [])
    childrenOf.get(parent)!.push(child)
  }
  const out: string[] = []
  const stack = [boneName]
  while (stack.length) {
    const name = stack.pop()
    if (name === undefined) break
    out.push(name)
    for (const child of childrenOf.get(name) || []) stack.push(child)
  }
  return out
}

// Joints the rig cannot function without — removing them (or an ancestor of
// them) is refused rather than silently deleting the skeleton's backbone.
const PROTECTED_JOINT_NAMES = new Set(['root', 'hips', 'pelvis', 'spine'])

export function isProtectedJoint(name: string): boolean {
  return PROTECTED_JOINT_NAMES.has(name.toLowerCase())
}

const FINGER_JOINT_PATTERN = /hand_extra_|finger|thumb|index|middle|ring|pinky/i
export function isFingerJoint(name: string): boolean {
  return FINGER_JOINT_PATTERN.test(name)
}

const SPUR_JOINT_PATTERN = /^spur_/i
export function isSpurJoint(name: string): boolean {
  return SPUR_JOINT_PATTERN.test(name)
}

export function fingerJointNames(allJointNames: string[]): string[] {
  return allJointNames.filter(isFingerJoint)
}

export function spurJointNames(allJointNames: string[]): string[] {
  return allJointNames.filter(isSpurJoint)
}

// Expands a requested removal (one or more joints, e.g. from the "remove all
// finger joints" shortcut) into its full descendant set, dropping — and
// separately reporting — any protected joint reached along the way instead
// of silently removing it.
export function resolveRemovableJoints(
  requestedNames: string[],
  parentOf: BoneParentMap,
): { removable: string[]; refused: string[] } {
  const removable = new Set<string>()
  const refused: string[] = []
  for (const name of requestedNames) {
    if (isProtectedJoint(name)) {
      refused.push(name)
      continue
    }
    for (const descendant of collectSubtree(name, parentOf)) {
      if (isProtectedJoint(descendant)) continue
      removable.add(descendant)
    }
  }
  return { removable: [...removable], refused }
}

// --- Single-joint removal ---------------------------------------------------
//
// Splicing a single joint out of the hierarchy (as opposed to removing its
// whole subtree): every child of the removed joint re-parents to the removed
// joint's own parent, closing the gap.

// Returns a new map: every child of `name` now points at `parentOf[name]`,
// and `name` itself is dropped. Does not mutate the input map.
export function reparentOnRemove(parentOf: BoneParentMap, name: string): BoneParentMap {
  const grandparent = parentOf[name] ?? null
  const next: BoneParentMap = {}
  for (const [child, parent] of Object.entries(parentOf)) {
    if (child === name) continue
    next[child] = parent === name ? grandparent : parent
  }
  return next
}

// Whether a single joint may be spliced out on its own: refuses protected
// backbone joints and root joints (a root has no parent to re-parent onto).
export function resolveSingleRemoval(
  name: string,
  parentOf: BoneParentMap,
): { removable: boolean; reason?: string } {
  if (isProtectedJoint(name)) return { removable: false, reason: 'protected joint' }
  if (parentOf[name] === null || parentOf[name] === undefined) return { removable: false, reason: 'root joint' }
  return { removable: true }
}

// --- Drag mode (move with children / single joint) --------------------------
//
// A 66-bone rig puts a whole finger chain under each hand, so dragging a hand
// on its own is almost never what the operator means. `subtree` translates the
// dragged joint's whole subtree rigidly; `single` moves only the dragged joint
// and pins its children where they are.
export type JointDragMode = 'single' | 'subtree'

// Every descendant of a joint, excluding the joint itself.
export function descendantJointNames(name: string, parentOf: BoneParentMap): string[] {
  return collectSubtree(name, parentOf).filter((entry) => entry !== name)
}

// World-space offset a drag has travelled so far.
export function jointDelta(from: JointPosition, to: JointPosition): JointPosition {
  return [to[0] - from[0], to[1] - from[1], to[2] - from[2]]
}

// Rigidly translate a set of world-space joint positions — the subtree carried
// along by a move-with-children drag. Every joint keeps its offset from every
// other, which is exactly what "the hand's fingers come with it" means.
export function translateJointPositions(positions: JointAdjustments, delta: JointPosition): JointAdjustments {
  const next: JointAdjustments = {}
  for (const [name, position] of Object.entries(positions)) {
    next[name] = [position[0] + delta[0], position[1] + delta[1], position[2] + delta[2]]
  }
  return next
}

// The mode one gesture runs in. The panel toggle sets the default; holding
// Shift while starting the drag flips it for that gesture only. A leaf joint
// has nothing to carry, so it is always a single-joint move.
export function resolveJointDragMode(
  { moveWithChildren, shiftKey, hasDescendants }: { moveWithChildren: boolean; shiftKey: boolean; hasDescendants: boolean },
): JointDragMode {
  if (!hasDescendants) return 'single'
  const base: JointDragMode = moveWithChildren ? 'subtree' : 'single'
  if (!shiftKey) return base
  return base === 'subtree' ? 'single' : 'subtree'
}

// --- Drag mode preference ---------------------------------------------------
//
// Remembered for the session (not forever) — a rig-by-rig habit, not a
// permanent setting. Defaults to ON: the operator's expectation is that
// children come along.
export const MOVE_WITH_CHILDREN_STORAGE_KEY = 'aismith.rigAdjust.moveWithChildren'
export type PreferenceStorage = Pick<Storage, 'getItem' | 'setItem'>

export function readMoveWithChildrenPreference(storage: PreferenceStorage | null | undefined): boolean {
  try {
    const raw = storage?.getItem(MOVE_WITH_CHILDREN_STORAGE_KEY)
    if (raw === null || raw === undefined) return true
    return raw !== 'false'
  } catch {
    return true
  }
}

export function writeMoveWithChildrenPreference(storage: PreferenceStorage | null | undefined, value: boolean): void {
  try {
    storage?.setItem(MOVE_WITH_CHILDREN_STORAGE_KEY, value ? 'true' : 'false')
  } catch {
    // A blocked/absent sessionStorage must never break joint editing.
  }
}
