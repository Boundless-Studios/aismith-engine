import { describe, expect, it } from 'vitest'

import { classifyBoneDisplay } from '../../../vendor/mesh2motion-app/src/lib/BoneDisplayPolicy'

describe('classifyBoneDisplay', () => {
  it('separates orientation helpers from anatomical joints', () => {
    expect(classifyBoneDisplay('index_04_leaf_l')).toBe('helper')
    expect(classifyBoneDisplay('weapon_end')).toBe('helper')
    expect(classifyBoneDisplay('Ear_Tip_L')).toBe('helper')
  })

  it('distinguishes axial, clavicle, and limb joints', () => {
    expect(classifyBoneDisplay('spine_02')).toBe('axial')
    expect(classifyBoneDisplay('clavicle_l')).toBe('clavicle')
    expect(classifyBoneDisplay('lowerarm_r')).toBe('limb')
  })
})
