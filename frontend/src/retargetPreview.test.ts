import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { buildRetargetNameMap, DONOR_TO_TARGET_BONE_NAMES, findSkinnedMesh, HIP_BONE_NAME } from './retargetPreview'

describe('buildRetargetNameMap', () => {
  it('inverts a donor->target map into the target->donor shape SkeletonUtils.retargetClip wants', () => {
    expect(buildRetargetNameMap({ pelvis: 'pelvis', upperarm_l: 'upper_arm_l' })).toEqual({
      pelvis: 'pelvis',
      upper_arm_l: 'upperarm_l',
    })
  })

  it('produces a bijective map for the real donor->target table (no two donor bones collide on the same target bone)', () => {
    const inverted = buildRetargetNameMap(DONOR_TO_TARGET_BONE_NAMES)
    expect(Object.keys(inverted)).toHaveLength(Object.keys(DONOR_TO_TARGET_BONE_NAMES).length)
  })

  it('maps every real UniRig-rig bone name verified via /tmp/k5/glbinfo.py to its donor equivalent', () => {
    const inverted = buildRetargetNameMap(DONOR_TO_TARGET_BONE_NAMES)
    expect(inverted).toMatchObject({
      pelvis: 'pelvis',
      spine_1: 'spine_01',
      spine_2: 'spine_02',
      spine_3: 'spine_03',
      neck: 'neck_01',
      head: 'head',
      upper_arm_l: 'upperarm_l',
      lower_arm_l: 'lowerarm_l',
      hand_l: 'hand_l',
      thigh_l: 'thigh_l',
      calf_l: 'calf_l',
      foot_l: 'foot_l',
    })
  })

  it('intentionally omits the donor clavicle and toe bones -- the verified UniRig rig has neither', () => {
    const targetNames = Object.values(DONOR_TO_TARGET_BONE_NAMES)
    expect(targetNames).not.toContain('clavicle_l')
    expect(targetNames).not.toContain('ball_l')
  })
})

describe('HIP_BONE_NAME', () => {
  it('matches the donor and target pelvis-equivalent bone name used for the retarget hip track', () => {
    expect(HIP_BONE_NAME).toBe('pelvis')
    expect(DONOR_TO_TARGET_BONE_NAMES.pelvis).toBe('pelvis')
  })
})

describe('findSkinnedMesh', () => {
  function makeSkinnedMesh(name: string): THREE.SkinnedMesh {
    const bone = new THREE.Bone()
    const skeleton = new THREE.Skeleton([bone])
    const mesh = new THREE.SkinnedMesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial())
    mesh.name = name
    mesh.bind(skeleton)
    return mesh
  }

  it('finds a SkinnedMesh nested inside a Group', () => {
    const root = new THREE.Group()
    const inner = new THREE.Group()
    const mesh = makeSkinnedMesh('character')
    inner.add(mesh)
    root.add(inner)

    expect(findSkinnedMesh(root)).toBe(mesh)
  })

  it('returns null when the object graph has no SkinnedMesh', () => {
    const root = new THREE.Group()
    root.add(new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial()))

    expect(findSkinnedMesh(root)).toBeNull()
  })
})
