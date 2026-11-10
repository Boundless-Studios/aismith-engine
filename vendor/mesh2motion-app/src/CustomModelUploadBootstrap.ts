import { Mesh2MotionEngine } from './Mesh2MotionEngine'
import { SkeletonType } from './lib/enums/SkeletonType.ts'
import {
  acceptsJointRecipeMessage,
  isJointRecipeCaptureMessage,
  jointRecipeContextFromLocation,
  type JointRecipePayload
} from './lib/processes/edit-skeleton/JointRecipeBridge.ts'

export class CustomModelUploadBootstrap {
  private readonly mesh2motion_engine: Mesh2MotionEngine

  constructor () {
    this.mesh2motion_engine = new Mesh2MotionEngine()
    this.setup_joint_recipe_bridge()
    this.load_aismith_source()
  }

  private setup_joint_recipe_bridge (): void {
    const context = jointRecipeContextFromLocation(window.location.search)
    if (context === null) return
    let save_timer: number | undefined
    const post_current_recipe = (): void => {
      window.parent.postMessage({
        source: 'aismith-mesh2motion',
        type: 'joint-recipe-changed',
        ...context,
        recipe: this.mesh2motion_engine.edit_skeleton_step.create_joint_recipe(context.cleanMeshArtifactId)
      }, window.location.origin)
    }
    this.mesh2motion_engine.edit_skeleton_step.addEventListener('jointRecipeChanged', () => {
      window.clearTimeout(save_timer)
      save_timer = window.setTimeout(post_current_recipe, 300)
    })
    window.addEventListener('message', (event: MessageEvent) => {
      if (event.origin === window.location.origin && isJointRecipeCaptureMessage(event.data, context)) {
        post_current_recipe()
        return
      }
      if (event.origin !== window.location.origin ||
        event.data?.source !== 'aismith-studio' ||
        event.data?.type !== 'joint-recipe-restore' ||
        !acceptsJointRecipeMessage(event.data, context)) return
      const recipe = event.data.recipe as JointRecipePayload
      if (recipe.clean_mesh_artifact_id !== context.cleanMeshArtifactId) return
      this.mesh2motion_engine.edit_skeleton_step.apply_joint_recipe(recipe)
    })
  }

  /**
   * AISmith passes the current artifact as a same-origin URL. Keeping this in
   * the bootstrap means the upstream editor keeps owning its interactive UI,
   * while our studio continues to own artifacts and pipeline handoff.
   */
  private load_aismith_source (): void {
    const source = new URLSearchParams(window.location.search).get('source')
    if (source === null || source.length === 0) return

    const extension = new URLSearchParams(window.location.search).get('extension')?.toLowerCase()
    if (extension !== 'glb') {
      console.error('AISmith Rig + Animate requires a GLB artifact')
      return
    }

    const mode = new URLSearchParams(window.location.search).get('mode')
    if (mode === 'edit-clean-mesh') {
      this.load_clean_mesh(source)
      return
    }
    if (mode === 'edit-existing-rig') {
      console.error('Rebinding an existing skinned GLB is blocked; open the immutable clean mesh instead')
      return
    }
    this.mesh2motion_engine.load_model_step.load_model_file(source, extension)
  }

  private load_clean_mesh (source: string): void {
    this.mesh2motion_engine.load_skeleton_step.addEventListener('skeletonLoaded', () => {
      window.parent.postMessage({
        source: 'aismith-mesh2motion',
        type: 'joint-recipe-ready',
        ...jointRecipeContextFromLocation(window.location.search)
      }, window.location.origin)
    }, { once: true })
    this.mesh2motion_engine.load_model_step.addEventListener('modelLoaded', () => {
      void this.mesh2motion_engine.load_skeleton_step.load_scaled_template(SkeletonType.Human)
    }, { once: true })
    this.mesh2motion_engine.load_model_step.load_model_file(source, 'glb')
  }

}

// instantiate the class to setup event listeners
const app = new CustomModelUploadBootstrap()
