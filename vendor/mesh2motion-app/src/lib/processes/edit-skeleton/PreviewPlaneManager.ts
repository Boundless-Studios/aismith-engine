import { Group, Mesh, PlaneGeometry, MeshBasicMaterial, type Scene, DoubleSide, GridHelper } from 'three'

/**
 * PreviewPlaneManager - Singleton class for managing a preview 3D plane in the edit skeleton step
 * The plane can be adjusted in height for later use in the skinning process
 */
export class PreviewPlaneManager {
  private static instance: PreviewPlaneManager
  private readonly preview_plane_group_name: string = 'preview_plane_group'
  
  // State tracking
  private scene_ref: Scene | null = null
  private plane_group: Group | null = null
  private plane_mesh: Mesh | null = null
  private grid_helper: GridHelper | null = null
  private current_height: number = 0.0
  private current_size: number = 2.0
  private is_visible: boolean = false

  private constructor () { }

  /**
   * Get the singleton instance of PreviewPlaneManager
   */
  public static getInstance (): PreviewPlaneManager {
    if (PreviewPlaneManager.instance === undefined) {
      PreviewPlaneManager.instance = new PreviewPlaneManager()
    }
    return PreviewPlaneManager.instance
  }

  /**
   * Initialize the manager with a scene reference
   * @param scene The main scene
   */
  public initialize (scene: Scene): void {
    this.scene_ref = scene
  }

  /**
   * Add a preview 3D plane to the scene at the origin
   * @param height The height position of the plane (Y coordinate)
   * @param size The size of the plane (width and depth)
   * @returns The created plane mesh
   */
  public add_plane (height: number = 0.0, size: number = 5.0): Mesh {
    if (this.scene_ref === null) {
      throw new Error('PreviewPlaneManager not initialized with scene reference')
    }

    // Remove existing plane if it exists
    this.remove_plane()

    // Update state
    this.current_height = height
    this.current_size = size
    this.is_visible = true

    // Create new preview plane group
    this.plane_group = new Group()
    this.plane_group.name = this.preview_plane_group_name

    // Create solid plane geometry and material
    const geometry = new PlaneGeometry(size, size)
    const material = new MeshBasicMaterial({
      color: 0x00ff00, // Green color for visibility
      transparent: true,
      opacity: 0.7,
      side: DoubleSide, // Make it visible from both sides
      wireframe: false
    })

    // Create the solid plane mesh
    this.plane_mesh = new Mesh(geometry, material)
    this.plane_mesh.name = 'preview_plane'
    
    // Position the plane at the specified height
    this.plane_mesh.position.set(0, height, 0)
    
    // Rotate the plane to be horizontal (lying flat)
    this.plane_mesh.rotation.x = -Math.PI / 2

    // Create grid helper for rectangular grid lines
    const grid_divisions = 15 // Number of divisions
    this.grid_helper = new GridHelper(size, grid_divisions, 0x888888, 0x888888) // Gray grid
    this.grid_helper.name = 'preview_plane_grid'
    
    // Position the grid at the specified height (GridHelper is already horizontal)
    this.grid_helper.position.set(0, height, 0)

    // Add both the plane and grid to the group and group to scene
    this.plane_group.add(this.plane_mesh)
    this.plane_group.add(this.grid_helper)

    // main scene to add to
    this.scene_ref.add(this.plane_group) 

    return this.plane_mesh
  }

  /**
   * Update the height of the existing preview plane
   * @param height The new height position for the plane
   */
  public update_height (height: number): void {
    if (this.plane_mesh !== null && this.grid_helper !== null && this.is_visible) {
      this.plane_mesh.position.y = height
      this.grid_helper.position.y = height // Grid and plane at same height since no z-fighting
      this.current_height = height
    }
  }

  /**
   * Update the size of the existing preview plane
   * @param size The new size for the plane
   */
  public update_size (size: number): void {
    if (this.plane_mesh !== null && this.grid_helper !== null && this.is_visible) {
      // Update the solid plane geometry
      const new_geometry = new PlaneGeometry(size, size)
      this.plane_mesh.geometry.dispose() // Clean up old geometry
      this.plane_mesh.geometry = new_geometry
      
      // For GridHelper, we need to recreate it with new size
      const current_height = this.grid_helper.position.y
      this.plane_group?.remove(this.grid_helper)
      this.grid_helper.dispose() // Clean up old grid
      
      const grid_divisions = 10
      this.grid_helper = new GridHelper(size, grid_divisions, 0x888888, 0x888888)
      this.grid_helper.name = 'preview_plane_grid'
      this.grid_helper.position.set(0, current_height, 0)
      this.plane_group?.add(this.grid_helper)
      
      this.current_size = size
    }
  }

  /**
   * Get the current height of the preview plane
   * @returns The current height of the plane, or null if no plane exists
   */
  public get_height (): number | null {
    return this.is_visible ? this.current_height : null
  }

  /**
   * Get the current size of the preview plane
   * @returns The current size of the plane, or null if no plane exists
   */
  public get_size (): number | null {
    return this.is_visible ? this.current_size : null
  }

  /**
   * Check if preview plane exists and is visible
   * @returns True if preview plane exists and is visible, false otherwise
   */
  public is_plane_visible (): boolean {
    return this.is_visible
  }

  /**
   * Set the visibility of the preview plane
   * @param visible Whether the plane should be visible
   */
  public set_visibility (visible: boolean): void {
    if (visible && !this.is_visible) {
      // Add the plane if it should be visible but isn't
      this.add_plane(this.current_height, this.current_size)
    } else if (!visible && this.is_visible) {
      // Remove the plane if it should be hidden but is visible
      this.remove_plane()
    }
  }

  /**
   * Remove the preview plane from the scene
   */
  public remove_plane (): void {
    if (this.plane_group !== null && this.scene_ref !== null) {
      // Clean up solid plane geometry and material
      if (this.plane_mesh !== null) {
        this.plane_mesh.geometry.dispose()
        if (this.plane_mesh.material instanceof MeshBasicMaterial) {
          this.plane_mesh.material.dispose()
        }
      }
      
      // Clean up grid helper
      if (this.grid_helper !== null) {
        this.grid_helper.dispose()
      }
      
      this.scene_ref.remove(this.plane_group)
      this.plane_group = null
      this.plane_mesh = null
      this.grid_helper = null
      this.is_visible = false
    }
  }

  /**
   * Clean up all resources and reset state
   */
  public cleanup (): void {
    this.remove_plane()
    this.scene_ref = null
    this.current_height = 0.0
    this.current_size = 2.0
    this.is_visible = false
  }
}
