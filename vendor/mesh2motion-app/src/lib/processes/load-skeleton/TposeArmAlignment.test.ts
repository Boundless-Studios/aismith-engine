import { Bone, BufferGeometry, Float32BufferAttribute, Group, Object3D, Vector3 } from 'three'
import { describe, expect, it } from 'vitest'

import { align_human_tpose_arms, estimate_tpose_arm_height, model_vertex_positions } from './TposeArmAlignment.js'

function tposePoints (armY: number): Vector3[] {
  const points = [
    new Vector3(-0.1, 0, 0), new Vector3(0.1, 0, 0),
    new Vector3(-0.12, 1, 0), new Vector3(0.12, 1, 0)
  ]
  for (const side of [-1, 1]) {
    for (let index = 0; index < 12; index++) {
      points.push(new Vector3(side * (0.3 + index * 0.015), armY + (index % 3 - 1) * 0.01, 0))
    }
  }
  return points
}

function humanArmature (): { armature: Group, left: Bone, right: Bone } {
  const armature = new Group()
  const spine = new Bone()
  spine.name = 'spine_03'
  spine.position.y = 0.75
  armature.add(spine)
  const left = new Bone()
  left.name = 'upperarm_l'
  left.position.set(0.1, 0.1, 0)
  const right = new Bone()
  right.name = 'upperarm_r'
  right.position.set(-0.1, 0.1, 0)
  spine.add(left, right)
  armature.updateWorldMatrix(true, true)
  return { armature, left, right }
}

describe('T-pose arm alignment', () => {
  it('reads a structural mesh created by another Three.js module instance', () => {
    const root = new Group()
    const foreignMesh = new Object3D() as Object3D & { geometry: BufferGeometry }
    foreignMesh.type = 'Mesh'
    foreignMesh.geometry = new BufferGeometry()
    foreignMesh.geometry.setAttribute('position', new Float32BufferAttribute([1, 2, 3, -1, 0, 1], 3))
    root.add(foreignMesh)

    expect(model_vertex_positions(root).map((point) => point.toArray())).toEqual([[1, 2, 3], [-1, 0, 1]])
  })

  it('estimates the bilateral horizontal arm height', () => {
    expect(estimate_tpose_arm_height(tposePoints(0.77))).toBeCloseTo(0.77, 2)
  })

  it('ignores sparse generated-mesh outliers around a horizontal arm band', () => {
    const points = tposePoints(0.77)
    points.push(new Vector3(-0.48, 0.52, 0), new Vector3(0.48, 0.98, 0))

    expect(estimate_tpose_arm_height(points)).toBeCloseTo(0.77, 2)
  })

  it('moves both upper-arm chains to the detected mesh landmark', () => {
    const { armature, left, right } = humanArmature()

    expect(align_human_tpose_arms(armature, tposePoints(0.77))).toBe(true)
    expect(left.getWorldPosition(new Vector3()).y).toBeCloseTo(0.77, 2)
    expect(right.getWorldPosition(new Vector3()).y).toBeCloseTo(0.77, 2)
  })

  it('does not alter an armature without confident bilateral landmarks', () => {
    const { armature, left } = humanArmature()
    const originalY = left.getWorldPosition(new Vector3()).y

    expect(align_human_tpose_arms(armature, [new Vector3(-0.5, 0.7, 0)])).toBe(false)
    expect(left.getWorldPosition(new Vector3()).y).toBe(originalY)
  })
})
