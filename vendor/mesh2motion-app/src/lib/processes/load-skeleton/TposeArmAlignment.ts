import { Box3, type Mesh, type Object3D, Vector3 } from 'three'

const MIN_SIDE_SAMPLES = 6
const LATERAL_FRACTION = 0.25
const MIN_HEIGHT_FRACTION = 0.45
const MAX_ARM_BAND_SPAN = 0.18
const MAX_SIDE_HEIGHT_DIFFERENCE = 0.05

function median (values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

function central_span (values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const lower = sorted[Math.floor((sorted.length - 1) * 0.1)]
  const upper = sorted[Math.ceil((sorted.length - 1) * 0.9)]
  return upper - lower
}

export function model_vertex_positions (model: Object3D): Vector3[] {
  const positions: Vector3[] = []
  model.updateWorldMatrix(true, true)
  model.traverse((object) => {
    if (object.type !== 'Mesh' && object.type !== 'SkinnedMesh') return
    const mesh = object as Mesh
    const attribute = mesh.geometry.attributes.position
    if (attribute === undefined) return
    for (let index = 0; index < attribute.count; index++) {
      positions.push(mesh.localToWorld(new Vector3().fromBufferAttribute(attribute, index)))
    }
  })
  return positions
}

export function estimate_tpose_arm_height (positions: Vector3[]): number | null {
  if (positions.length === 0) return null
  const bounds = new Box3().setFromPoints(positions)
  const size = bounds.getSize(new Vector3())
  if (size.x <= 0 || size.y <= 0) return null

  const centerX = (bounds.min.x + bounds.max.x) / 2
  const lateralDistance = size.x * LATERAL_FRACTION
  const minimumY = bounds.min.y + size.y * MIN_HEIGHT_FRACTION
  const left = positions.filter((point) => point.x <= centerX - lateralDistance && point.y >= minimumY).map((point) => point.y)
  const right = positions.filter((point) => point.x >= centerX + lateralDistance && point.y >= minimumY).map((point) => point.y)
  if (left.length < MIN_SIDE_SAMPLES || right.length < MIN_SIDE_SAMPLES) return null

  const leftHeight = median(left)
  const rightHeight = median(right)
  const leftSpan = central_span(left)
  const rightSpan = central_span(right)
  if (leftSpan > size.y * MAX_ARM_BAND_SPAN || rightSpan > size.y * MAX_ARM_BAND_SPAN) return null
  if (Math.abs(leftHeight - rightHeight) > size.y * MAX_SIDE_HEIGHT_DIFFERENCE) return null
  return (leftHeight + rightHeight) / 2
}

export function align_human_tpose_arms (armature: Object3D, positions: Vector3[]): boolean {
  const armHeight = estimate_tpose_arm_height(positions)
  if (armHeight === null) {
    console.info('Native human T-pose alignment skipped: model arm band was not detected')
    return false
  }

  const upperArms: Object3D[] = []
  armature.traverse((object) => {
    const name = object.name.toLowerCase()
    if (name === 'upperarm_l' || name === 'upperarm_r') upperArms.push(object)
  })
  if (upperArms.length !== 2 || upperArms.some((bone) => bone.parent === null)) {
    console.info(`Native human T-pose alignment skipped: found ${upperArms.length} upper-arm bones`)
    return false
  }

  armature.updateWorldMatrix(true, true)
  for (const bone of upperArms) {
    const target = bone.getWorldPosition(new Vector3())
    target.y = armHeight
    bone.position.copy(bone.parent!.worldToLocal(target))
  }
  armature.updateWorldMatrix(true, true)
  return true
}
