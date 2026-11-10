import { type Object3D, Mesh, SphereGeometry, MeshBasicMaterial, type Scene, Group, Vector3 } from 'three'

const ORIGIN_MARKER_GROUP_NAME: string = 'origin_marker_group'

export function add_origin_markers (root: Scene): void {
  remove_origin_markers(root)

  // Create a group to hold all markers
  let marker_group = root.getObjectByName(ORIGIN_MARKER_GROUP_NAME) as Group
  if (!marker_group) {
    marker_group = new Group()
    marker_group.name = ORIGIN_MARKER_GROUP_NAME
    root.add(marker_group)
  }

  // with the root scene, there is a group called "Imported Model" that contains
  // the meshes we want to put markers on
  const imported_meshes: Scene = root.children.find(x => x.name === 'Imported Model') as Scene

  for (const mesh of imported_meshes.children) {
    // Get mesh world position
    const world_pos = new Vector3()
    mesh.getWorldPosition(world_pos)

    const marker = new Mesh(
      new SphereGeometry(0.02, 8, 8),
      new MeshBasicMaterial({ color: 0xff4c00 }) // orange
    )
    marker.name = 'origin_marker_' + mesh.name
    marker.position.copy(world_pos)
    marker_group.add(marker)
  }
}

export function remove_origin_markers(root: Object3D): void {
  const marker_group = root.getObjectByName(ORIGIN_MARKER_GROUP_NAME)
  if (marker_group !== undefined && marker_group.parent) {
    marker_group.parent.remove(marker_group)
  }
}
