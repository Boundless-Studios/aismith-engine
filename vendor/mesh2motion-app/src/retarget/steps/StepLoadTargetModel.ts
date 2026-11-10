import { type Bone, Box3, type Scene, type SkinnedMesh, Vector3 } from 'three'
import { target_model_camera_position, target_model_camera_target } from './TargetModelCamera.ts'
import { type Mesh2MotionEngine } from '../../Mesh2MotionEngine.ts'
import { ModalDialog } from '../../lib/ModalDialog.ts'
import { RetargetUtils } from '../RetargetUtils.ts'
import { TargetBoneTreeDialog } from '../TargetBoneTreeDialog.ts'

/**
 * Handles loading the target model (user-uploaded model) for retargeting
 */
export class StepLoadTargetModel extends EventTarget {
  private readonly mesh2motion_engine: Mesh2MotionEngine
  private readonly target_bone_tree_dialog: TargetBoneTreeDialog
  private file_input: HTMLInputElement | null = null
  private load_model_button: HTMLLabelElement | null = null
  private load_status: HTMLElement | null = null

  private retargetable_meshes: Scene | null = null

  constructor (mesh2motion_engine: Mesh2MotionEngine) {
    super()
    this.mesh2motion_engine = mesh2motion_engine
    this.target_bone_tree_dialog = new TargetBoneTreeDialog()
  }

  public begin (): void {
    this.add_event_listeners()
    this.target_bone_tree_dialog.begin()
  }

  public get_retargetable_meshes (): Scene | null {
    return this.retargetable_meshes
  }

  public get_first_skinned_mesh_bones (): Map<string, Bone> {
    const bones: Map<string, Bone> = new Map<string, Bone>()

    if (this.retargetable_meshes === null) {
      return bones
    }

    let first_skinned_mesh: SkinnedMesh | null = null
    this.retargetable_meshes.traverse((child) => {
      if (first_skinned_mesh === null && child.type === 'SkinnedMesh') {
        first_skinned_mesh = child as SkinnedMesh
      }
    })

    if (first_skinned_mesh === null) {
      return bones
    }

    first_skinned_mesh.skeleton.bones.forEach((bone) => {
      bones.set(bone.uuid, bone)
    })

    return bones
  }

  private add_event_listeners (): void {
    // Get DOM elements
    this.file_input = document.getElementById('upload-file') as HTMLInputElement
    this.load_model_button = document.getElementById('load-model-button') as HTMLLabelElement
    this.load_status = document.getElementById('target-model-load-status')

    if (this.file_input === null) {
      console.error('Could not find file input element')
      return
    }

    // Add event listener for file selection
    this.file_input.addEventListener('change', (event) => {
      console.log('File input changed', event)
      this.handle_file_select(event)
    })
  }

  private handle_file_select (event: Event): void {
    const target = event.target as HTMLInputElement
    if (target.files === null || target.files.length === 0) return
    const file = target.files[0]
    const file_name = file.name.toLowerCase()
    const file_extension = file_name.endsWith('.glb') ? 'glb' : file_name.endsWith('.fbx') ? 'fbx' : file_name.endsWith('.zip') ? 'zip' : null
    if (file_extension === null) {
      new ModalDialog('Unsupported file type. Please select a GLB, FBX, or ZIP file.', 'Error').show()
      return
    }
    const file_url = URL.createObjectURL(file)
    this.load_model(file_url, file_extension, () => URL.revokeObjectURL(file_url))
  }

  public load_model_url (source: string, extension: 'glb'): void {
    this.load_model(source, extension)
  }

  private load_model (source: string, extension: string, cleanup: () => void = () => undefined): void {
    if (this.load_status !== null) {
      this.load_status.textContent = 'Downloading and validating the selected character…'
      this.load_status.dataset.state = 'loading'
    }
    this.mesh2motion_engine.load_model_step.set_preserve_skinned_mesh(true)
    this.mesh2motion_engine.load_model_step.addEventListener('modelLoadedForRetargeting', () => {
      cleanup()
      const retargetable_meshes: Scene = this.mesh2motion_engine.load_model_step.get_final_retargetable_model_data()
      if (!RetargetUtils.validate_skinned_mesh_has_bones(retargetable_meshes)) return

      RetargetUtils.reset_skinned_mesh_to_rest_pose(retargetable_meshes)
      this.mesh2motion_engine.get_scene().add(retargetable_meshes)
      const largest_dimension = this.calculate_max_mesh_dimension(retargetable_meshes)
      if (largest_dimension > 20) {
        const target_height = 1.5
        const scale_factor = target_height / largest_dimension
        new ModalDialog('Large Rig Warning',
          `The model you imported is large (${largest_dimension.toFixed(1)} meters). Mesh2Motion expects 1 unit = 1 meter. Your model will be scaled down. Mixamo rigs will work as long as they only have one skeleton in the model file.`).show()
        retargetable_meshes.scale.set(scale_factor, scale_factor, scale_factor)
      }

      this.frame_loaded_model(retargetable_meshes)
      this.add_skeleton_helper(retargetable_meshes)
      this.retargetable_meshes = retargetable_meshes
      this.target_bone_tree_dialog.set_target_bones(this.get_first_skinned_mesh_bones())
      this.target_bone_tree_dialog.set_target_skinned_mesh_count(this.get_target_skinned_mesh_count())
      if (this.load_status !== null) {
        this.load_status.textContent = `${this.get_first_skinned_mesh_bones().size} target bones loaded`
        this.load_status.dataset.state = 'ready'
      }
      this.dispatchEvent(new CustomEvent('target-model-loaded'))
    }, { once: true })

    try {
      this.mesh2motion_engine.load_model_step.load_model_file(source, extension)
    } catch (error) {
      cleanup()
      console.error('Error loading model:', error)
      if (this.load_status !== null) {
        this.load_status.textContent = 'The selected character could not be loaded. Choose another rig or retry.'
        this.load_status.dataset.state = 'failed'
      }
      new ModalDialog('Error loading model file.', 'Error').show()
    }
  }

  private frame_loaded_model (retargetable_meshes: Scene): void {
    retargetable_meshes.updateMatrixWorld(true)
    const bounding_box = new Box3().setFromObject(retargetable_meshes)
    const center = new Vector3()
    const size = new Vector3()
    bounding_box.getCenter(center)
    bounding_box.getSize(size)
    this.mesh2motion_engine.set_camera_target(target_model_camera_target(center, size))
    this.mesh2motion_engine.set_camera_position(target_model_camera_position(center, size))
  }

  private fix_fbx_mixamo_bones (): void {
    // TODO: The FBX files for Mixamo do something weird with the bone hieararchy in three.js.
    // The root bone is is the "hips" bone, and the first child of the hips bone is also named "hips".
    // This is messing up stuff like the skeleton helper and probably the animations by assigning keyframes in the wrong way.
    // The root hips bone is not in the "bones list" with the skeleton data, so I am not sure why that is there. I might have to look into the FBX loader
    // class to see what is going on.
  }

  private add_skeleton_helper (retargetable_meshes: Scene): void {
    retargetable_meshes.traverse((child) => {
      if (child.type === 'SkinnedMesh') {
        const skinned_mesh = child as SkinnedMesh
        this.mesh2motion_engine.regenerate_skeleton_helper(skinned_mesh.skeleton, 'Retarget Skeleton Helper')
      }
    })
  }

  private get_target_skinned_mesh_count (): number {
    let skinned_mesh_count = 0

    this.retargetable_meshes?.traverse((child) => {
      if (child.type === 'SkinnedMesh') {
        skinned_mesh_count++
      }
    })

    return skinned_mesh_count
  }

  // gets max dimension of the model for scaling
  // returns a unitless number representing the largest dimension
  private calculate_max_mesh_dimension (retargetable_meshes: Scene): number {
    const bounding_box = new Box3().setFromObject(retargetable_meshes)
    const size = new Vector3()
    bounding_box.getSize(size)
    return Math.max(size.x, size.y, size.z)
  }
}
