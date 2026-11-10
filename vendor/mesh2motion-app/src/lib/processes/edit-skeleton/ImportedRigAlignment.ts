import { Box3, Object3D, Vector3 } from 'three'
import { scale_to_match_height } from '../load-skeleton/SkeletonScaleContract'

export function imported_rig_alignment (model: { minY: number; maxY: number; centerX: number; centerZ: number }, skeleton: { minY: number; maxY: number; centerX: number; centerZ: number }) {
  const scale = scale_to_match_height(model.maxY - model.minY, skeleton.maxY - skeleton.minY)
  return {
    scale,
    offsetX: model.centerX - skeleton.centerX * scale,
    offsetY: model.minY - skeleton.minY * scale,
    offsetZ: model.centerZ - skeleton.centerZ * scale,
  }
}

function bone_positions (armature: Object3D): Vector3[] {
  armature.updateWorldMatrix(true, true)
  const positions: Vector3[] = []
  armature.traverse((object) => {
    if (object.type === 'Bone') positions.push(object.getWorldPosition(new Vector3()))
  })
  return positions
}

export function align_imported_armature (armature: Object3D, model: Object3D): number {
  const model_bounds = new Box3().setFromObject(model)
  const initial_positions = bone_positions(armature)
  const initial_bounds = new Box3().setFromPoints(initial_positions)
  const initial_center = initial_bounds.getCenter(new Vector3())
  const model_center = model_bounds.getCenter(new Vector3())
  const alignment = imported_rig_alignment(
    { minY: model_bounds.min.y, maxY: model_bounds.max.y, centerX: model_center.x, centerZ: model_center.z },
    { minY: initial_bounds.min.y, maxY: initial_bounds.max.y, centerX: initial_center.x, centerZ: initial_center.z },
  )
  armature.traverse((object) => {
    if (object !== armature && object.type === 'Bone') object.position.multiplyScalar(alignment.scale)
  })
  armature.position.add(new Vector3(alignment.offsetX, alignment.offsetY, alignment.offsetZ))
  armature.updateWorldMatrix(true, true)
  return alignment.scale
}
