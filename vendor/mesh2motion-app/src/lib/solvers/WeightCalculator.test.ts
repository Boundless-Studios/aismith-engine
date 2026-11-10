import { describe, expect, it } from 'vitest'
import { Bone, BufferGeometry, Float32BufferAttribute } from 'three'

import { SkeletonType } from '../enums/SkeletonType.js'
import { WeightCalculator } from './WeightCalculator.js'

function boneSegment (name: string, start: [number, number, number], end: [number, number, number]): Bone {
  const bone = new Bone()
  bone.name = name
  bone.position.set(...start)
  const child = new Bone()
  child.name = `${name}_end`
  child.position.set(end[0] - start[0], end[1] - start[1], end[2] - start[2])
  bone.add(child)
  return bone
}

function geometryAt (position: [number, number, number]): BufferGeometry {
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(position, 3))
  return geometry
}

describe('WeightCalculator', () => {
  it('assigns a vertex by distance to the bone segment rather than its midpoint', () => {
    const upperArm = boneSegment('upper_arm_l', [0, 0, 0], [4, 0, 0])
    const competitor = boneSegment('spine', [3.5, 1, 0], [3.5, 1.4, 0])
    const pelvis = boneSegment('pelvis', [0, -10, 0], [0, -9, 0])
    const calculator = new WeightCalculator(
      [upperArm, competitor, pelvis],
      geometryAt([3.5, 0.1, 0]),
      SkeletonType.Human
    )
    const indices: number[] = []
    const weights: number[] = []

    calculator.initialize_caches()
    calculator.calculate_median_bone_weights(indices, weights)

    expect(indices[0]).toBe(0)
    expect(weights.slice(0, 4)).toEqual([1, 0, 0, 0])
  })
})
