import { describe, expect, it } from 'vitest'
import {
  MOVE_WITH_CHILDREN_STORAGE_KEY,
  adjustedJointCount, clearJointAdjustments, collectSubtree, descendantJointNames, fingerJointNames, isFingerJoint,
  isProtectedJoint, isSpurJoint, jointDelta, popJointUndo, pushJointUndo, pushJointUndoGroup,
  readMoveWithChildrenPreference, recordJointAdjustment, reparentOnRemove, resolveJointDragMode,
  resolveRemovableJoints, resolveSingleRemoval, spurJointNames, translateJointPositions,
  writeMoveWithChildrenPreference,
} from './jointAdjustments'

describe('recordJointAdjustment', () => {
  it('records a joint once it meaningfully differs from its predicted position', () => {
    const next = recordJointAdjustment({}, 'hand_l', [1, 0, 0], [0, 0, 0])
    expect(next).toEqual({ hand_l: [1, 0, 0] })
  })

  it('drops the record once the joint is moved back within noise of the original', () => {
    const moved = recordJointAdjustment({}, 'hand_l', [1, 0, 0], [0, 0, 0])
    const reset = recordJointAdjustment(moved, 'hand_l', [0, 0, 0.0000001], [0, 0, 0])
    expect(reset).toEqual({})
  })

  it('does not mutate the input record', () => {
    const original = { hand_l: [1, 0, 0] as [number, number, number] }
    recordJointAdjustment(original, 'foot_l', [1, 0, 0], [0, 0, 0])
    expect(original).toEqual({ hand_l: [1, 0, 0] })
  })
})

describe('adjustedJointCount / clearJointAdjustments', () => {
  it('counts distinct adjusted joints and clears to empty', () => {
    expect(adjustedJointCount({ a: [0, 0, 0], b: [1, 1, 1] })).toBe(2)
    expect(clearJointAdjustments()).toEqual({})
  })
})

describe('joint undo history', () => {
  it('undoes the most recent gesture regardless of whether it was a drag or a reset', () => {
    let stack = pushJointUndo([], 'hand_l', [0, 0, 0])
    stack = pushJointUndo(stack, 'foot_l', [0, 0, 0])
    const first = popJointUndo(stack)
    expect(first.entry).toEqual({ moves: [{ name: 'foot_l', position: [0, 0, 0] }] })
    const second = popJointUndo(first.stack)
    expect(second.entry).toEqual({ moves: [{ name: 'hand_l', position: [0, 0, 0] }] })
    expect(second.stack).toEqual([])
  })

  it('pops nothing from an empty stack', () => {
    const result = popJointUndo([])
    expect(result.entry).toBeUndefined()
    expect(result.stack).toEqual([])
  })

  it('undoes every joint one gesture moved as a single step', () => {
    const stack = pushJointUndoGroup([], [
      { name: 'hand_l', position: [0, 0, 0] },
      { name: 'thumb_l', position: [1, 0, 0] },
      { name: 'index_l', position: [2, 0, 0] },
    ])
    expect(stack).toHaveLength(1)
    const { entry, stack: rest } = popJointUndo(stack)
    expect(entry?.moves.map((move) => move.name)).toEqual(['hand_l', 'thumb_l', 'index_l'])
    expect(rest).toEqual([])
  })

  it('ignores an empty group rather than pushing a no-op undo step', () => {
    expect(pushJointUndoGroup([], [])).toEqual([])
  })

  it('copies the positions it stores so a later mutation cannot rewrite history', () => {
    const position: [number, number, number] = [1, 2, 3]
    const stack = pushJointUndoGroup([], [{ name: 'hand_l', position }])
    position[0] = 99
    expect(popJointUndo(stack).entry?.moves[0].position).toEqual([1, 2, 3])
  })
})

describe('collectSubtree', () => {
  const parentOf = {
    root: null,
    spine: 'root',
    shoulder_l: 'spine',
    hand_l: 'shoulder_l',
    thumb_l: 'hand_l',
    index_l: 'hand_l',
    shoulder_r: 'spine',
  }

  it('returns the joint and every descendant, not its siblings', () => {
    const subtree = collectSubtree('hand_l', parentOf)
    expect(new Set(subtree)).toEqual(new Set(['hand_l', 'thumb_l', 'index_l']))
  })

  it('returns a single-element list for a leaf joint', () => {
    expect(collectSubtree('thumb_l', parentOf)).toEqual(['thumb_l'])
  })

  it('returns the whole rig for the root', () => {
    const subtree = collectSubtree('root', parentOf)
    expect(new Set(subtree)).toEqual(new Set(Object.keys(parentOf)))
  })
})

describe('isProtectedJoint', () => {
  it('protects the rig backbone regardless of case', () => {
    expect(isProtectedJoint('root')).toBe(true)
    expect(isProtectedJoint('Hips')).toBe(true)
    expect(isProtectedJoint('PELVIS')).toBe(true)
    expect(isProtectedJoint('spine')).toBe(true)
    expect(isProtectedJoint('hand_l')).toBe(false)
  })
})

describe('finger and spur joint detection', () => {
  it('matches humanoid finger bones by common naming conventions', () => {
    expect(isFingerJoint('hand_extra_thumb_l')).toBe(true)
    expect(isFingerJoint('finger_index_01_l')).toBe(true)
    expect(isFingerJoint('Thumb_L')).toBe(true)
    expect(isFingerJoint('pinky_03_r')).toBe(true)
    expect(isFingerJoint('hand_l')).toBe(false)
  })

  it('matches chain-rig spur bones', () => {
    expect(isSpurJoint('spur_003')).toBe(true)
    expect(isSpurJoint('chain_003')).toBe(false)
  })

  it('filters a joint list down to fingers or spurs', () => {
    const names = ['root', 'hand_l', 'thumb_l', 'index_l', 'hand_r']
    expect(fingerJointNames(names)).toEqual(['thumb_l', 'index_l'])
    expect(spurJointNames(['spine', 'spur_01', 'spur_02'])).toEqual(['spur_01', 'spur_02'])
  })
})

describe('resolveRemovableJoints', () => {
  const parentOf = {
    root: null,
    hips: 'root',
    spine: 'hips',
    hand_l: 'spine',
    thumb_l: 'hand_l',
    index_l: 'hand_l',
  }

  it('expands a requested removal to its full subtree', () => {
    const result = resolveRemovableJoints(['hand_l'], parentOf)
    expect(new Set(result.removable)).toEqual(new Set(['hand_l', 'thumb_l', 'index_l']))
    expect(result.refused).toEqual([])
  })

  it('refuses to remove a protected backbone joint', () => {
    const result = resolveRemovableJoints(['hips'], parentOf)
    expect(result.removable).toEqual([])
    expect(result.refused).toEqual(['hips'])
  })

  it('drops only the protected joint when it appears inside another (non-protected) subtree', () => {
    // A contrived tree where removing a non-protected joint would also
    // remove a protected one further down its subtree — only that
    // descendant is dropped, the rest of the subtree still goes.
    const withNestedProtected = { root: null, shoulder_l: 'root', hips: 'shoulder_l', hand_l: 'shoulder_l' }
    const result = resolveRemovableJoints(['shoulder_l'], withNestedProtected)
    expect(result.removable).not.toContain('hips')
    expect(result.removable.sort()).toEqual(['hand_l', 'shoulder_l'])
    expect(result.refused).toEqual([])
  })

  it('merges duplicate joints across a batch removal without double counting', () => {
    const result = resolveRemovableJoints(['thumb_l', 'index_l'], parentOf)
    expect(result.removable.sort()).toEqual(['index_l', 'thumb_l'])
  })
})

describe('reparentOnRemove', () => {
  it('splices a middle-of-chain joint out, re-pointing its child at its own parent', () => {
    const parentOf = { a: null, b: 'a', c: 'b' }
    const next = reparentOnRemove(parentOf, 'b')
    expect(next).toEqual({ a: null, c: 'a' })
  })

  it('re-parents every child when the removed joint has more than one', () => {
    const parentOf = { root: null, spine: 'root', hand_l: 'spine', hand_r: 'spine' }
    const next = reparentOnRemove(parentOf, 'spine')
    expect(next).toEqual({ root: null, hand_l: 'root', hand_r: 'root' })
  })

  it('does not mutate the input map', () => {
    const parentOf = { a: null, b: 'a', c: 'b' }
    reparentOnRemove(parentOf, 'b')
    expect(parentOf).toEqual({ a: null, b: 'a', c: 'b' })
  })
})

describe('resolveSingleRemoval', () => {
  const parentOf = { root: null, hips: 'root', spine: 'hips', hand_l: 'spine' }

  it('allows removing a non-protected, non-root joint', () => {
    expect(resolveSingleRemoval('hand_l', parentOf)).toEqual({ removable: true })
  })

  it('refuses a root joint', () => {
    const result = resolveSingleRemoval('root', parentOf)
    expect(result.removable).toBe(false)
    expect(result.reason).toBeTruthy()
  })

  it('refuses a protected backbone joint', () => {
    const result = resolveSingleRemoval('hips', parentOf)
    expect(result.removable).toBe(false)
    expect(result.reason).toBeTruthy()
  })
})

describe('descendantJointNames', () => {
  const parentOf = {
    root: null,
    spine: 'root',
    hand_l: 'spine',
    thumb_01_l: 'hand_l',
    thumb_02_l: 'thumb_01_l',
    index_01_l: 'hand_l',
    hand_r: 'spine',
  }

  it('returns the whole finger chain under a hand, excluding the hand itself', () => {
    expect(new Set(descendantJointNames('hand_l', parentOf)))
      .toEqual(new Set(['thumb_01_l', 'thumb_02_l', 'index_01_l']))
  })

  it('returns nothing for a leaf joint, which is what makes it a single-joint drag', () => {
    expect(descendantJointNames('thumb_02_l', parentOf)).toEqual([])
  })

  it('does not reach across to a sibling limb', () => {
    expect(descendantJointNames('hand_l', parentOf)).not.toContain('hand_r')
  })
})

describe('translateJointPositions / jointDelta', () => {
  const subtree = {
    thumb_01_l: [1, 2, 3] as [number, number, number],
    index_01_l: [1, 2.5, 3] as [number, number, number],
  }

  it('carries every joint by the same offset, so the subtree keeps its shape', () => {
    const delta = jointDelta([0, 0, 0], [0.5, -0.25, 2])
    const moved = translateJointPositions(subtree, delta)
    expect(moved).toEqual({ thumb_01_l: [1.5, 1.75, 5], index_01_l: [1.5, 2.25, 5] })
    // Relative offsets inside the subtree are untouched — a rigid move.
    expect(moved.index_01_l[1] - moved.thumb_01_l[1]).toBeCloseTo(subtree.index_01_l[1] - subtree.thumb_01_l[1])
  })

  it('is a no-op for a zero delta and never mutates the input', () => {
    const moved = translateJointPositions(subtree, jointDelta([1, 1, 1], [1, 1, 1]))
    expect(moved).toEqual(subtree)
    expect(moved).not.toBe(subtree)
    expect(subtree.thumb_01_l).toEqual([1, 2, 3])
  })

  it('translates an empty subtree to an empty record', () => {
    expect(translateJointPositions({}, [1, 1, 1])).toEqual({})
  })
})

describe('resolveJointDragMode', () => {
  it('defaults to carrying the children when the toggle is on', () => {
    expect(resolveJointDragMode({ moveWithChildren: true, shiftKey: false, hasDescendants: true })).toBe('subtree')
  })

  it('moves the joint alone when the toggle is off', () => {
    expect(resolveJointDragMode({ moveWithChildren: false, shiftKey: false, hasDescendants: true })).toBe('single')
  })

  it('lets Shift flip the toggle for one gesture, in both directions', () => {
    expect(resolveJointDragMode({ moveWithChildren: true, shiftKey: true, hasDescendants: true })).toBe('single')
    expect(resolveJointDragMode({ moveWithChildren: false, shiftKey: true, hasDescendants: true })).toBe('subtree')
  })

  it('is always a single-joint move for a joint with nothing under it', () => {
    expect(resolveJointDragMode({ moveWithChildren: true, shiftKey: false, hasDescendants: false })).toBe('single')
    expect(resolveJointDragMode({ moveWithChildren: true, shiftKey: true, hasDescendants: false })).toBe('single')
  })
})

describe('move-with-children session preference', () => {
  const fakeStorage = (initial: Record<string, string> = {}) => {
    const values = { ...initial }
    return {
      values,
      getItem: (key: string) => (key in values ? values[key] : null),
      setItem: (key: string, value: string) => { values[key] = value },
    }
  }

  it('defaults to on when nothing has been stored yet', () => {
    expect(readMoveWithChildrenPreference(fakeStorage())).toBe(true)
    expect(readMoveWithChildrenPreference(null)).toBe(true)
  })

  it('round-trips the operator\'s choice through the session store', () => {
    const storage = fakeStorage()
    writeMoveWithChildrenPreference(storage, false)
    expect(storage.values[MOVE_WITH_CHILDREN_STORAGE_KEY]).toBe('false')
    expect(readMoveWithChildrenPreference(storage)).toBe(false)
    writeMoveWithChildrenPreference(storage, true)
    expect(readMoveWithChildrenPreference(storage)).toBe(true)
  })

  it('falls back to the default rather than throwing when storage is blocked', () => {
    const blocked = {
      getItem: () => { throw new Error('blocked') },
      setItem: () => { throw new Error('blocked') },
    }
    expect(readMoveWithChildrenPreference(blocked)).toBe(true)
    expect(() => writeMoveWithChildrenPreference(blocked, false)).not.toThrow()
  })
})
