import { Bone, Color, Group } from 'three'
import { describe, expect, it } from 'vitest'

import { CustomSkeletonHelper } from './CustomSkeletonHelper.js'

describe('CustomSkeletonHelper', () => {
  it('keeps mirrored-side joint markers finite in mirror mode', () => {
    const root = new Group()
    const left = new Bone()
    left.name = 'hand_l'
    left.position.x = 1
    const right = new Bone()
    right.name = 'hand_r'
    right.position.x = -1
    root.add(left, right)
    root.updateWorldMatrix(true, true)

    const helper = new CustomSkeletonHelper(root)
    helper.setHideRightSideJoints(true)
    helper.updateMatrixWorld(true)

    const markerPositions = helper.children[0].geometry.getAttribute('position')
    expect(Array.from(markerPositions.array).every(Number.isFinite)).toBe(true)
  })

  it('renders helper tips in a neutral color distinct from editable joints', () => {
    const root = new Bone()
    root.name = 'hand_r'
    const helperTip = new Bone()
    helperTip.name = 'finger_04_leaf_r'
    root.add(helperTip)
    root.updateWorldMatrix(true, true)

    const helper = new CustomSkeletonHelper(root)
    const editableColors = helper.children[0].geometry.getAttribute('color')
    const helperColors = helper.children[1].geometry.getAttribute('color')
    const editableColor = new Color().fromBufferAttribute(editableColors, 0)
    const helperColor = new Color().fromBufferAttribute(helperColors, 0)

    expect(helperColor.getHex()).toBe(0x6b7280)
    expect(helperColor.getHex()).not.toBe(editableColor.getHex())
    expect(helper.children[1].visible).toBe(false)
  })
})
