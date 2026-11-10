import { Vector3 } from 'three'

type Coordinates = { x: number, y: number, z: number }

function largest_dimension (size: Coordinates): number {
  return Math.max(size.x, size.y, size.z, 0.5)
}

export function target_model_camera_target (center: Coordinates, size: Coordinates): Vector3 {
  const extent = largest_dimension(size)
  return new Vector3(center.x, center.y - extent * 0.1, center.z)
}

export function target_model_camera_position (center: Coordinates, size: Coordinates): Vector3 {
  const extent = largest_dimension(size)
  return new Vector3(
    center.x + extent * 0.15,
    center.y - extent * 0.05,
    center.z + extent * 2.2
  )
}
