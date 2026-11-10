import { Bone, Box3, BoxGeometry, Mesh, MeshBasicMaterial, Object3D, Vector3 } from 'three'
import { describe, expect, it } from 'vitest'
import { SkeletonType } from '../../enums/SkeletonType'
import { StepLoadSkeleton } from './StepLoadSkeleton'

describe('StepLoadSkeleton native template contract', () => {
  it('uses the programmatic template type when the hidden dropdown has no selection', () => {
    const step = Object.create(StepLoadSkeleton.prototype) as StepLoadSkeleton
    Object.assign(step, {
      manual_set_skeleton_type: SkeletonType.Human,
      skeleton_file_path: () => SkeletonType.Error,
    })

    expect(step.skeleton_type()).toBe(SkeletonType.Human)
  })

  it('aligns the baked template to the clean model before editing', () => {
    const model = new Mesh(new BoxGeometry(4, 8, 2), new MeshBasicMaterial())
    model.position.set(5, 4, -3)
    model.updateWorldMatrix(true, true)

    const armature = new Object3D()
    const root = new Bone()
    root.position.set(1, 10, 2)
    const head = new Bone()
    head.position.set(0, 20, 0)
    root.add(head)
    armature.add(root)

    const step = Object.create(StepLoadSkeleton.prototype) as StepLoadSkeleton
    Object.assign(step, {
      loaded_armature: armature,
      _load_model_step: { model_meshes: () => model },
      skeleton_type: () => SkeletonType.None,
    })

    const aligned = step.armature()
    const positions: Vector3[] = []
    aligned.traverse((object) => {
      if (object.type === 'Bone') positions.push(object.getWorldPosition(new Vector3()))
    })
    const bounds = new Box3().setFromPoints(positions)
    const center = bounds.getCenter(new Vector3())

    expect(bounds.min.y).toBeCloseTo(0)
    expect(bounds.max.y).toBeCloseTo(8)
    expect(center.x).toBeCloseTo(5)
    expect(center.z).toBeCloseTo(-3)
  })
})
