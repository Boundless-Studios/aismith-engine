import { type Scene, type AnimationClip } from 'three'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter'
import { AnimationRetargetService } from '../AnimationRetargetService'

export class StepExportRetargetedAnimations extends EventTarget {
  public animation_clips_to_export: AnimationClip[] = []

  public set_animation_clips_to_export (all_animations_clips: AnimationClip[], animation_checkboxes: number[]): void {
    this.animation_clips_to_export = []
    animation_checkboxes.forEach((indx) => {
      const original_clip: AnimationClip = all_animations_clips[indx]
      const cloned_clip: AnimationClip = original_clip.clone()
      this.animation_clips_to_export.push(cloned_clip)
    })
  }

  public export (filename = 'exported_model'): void {
    if (this.animation_clips_to_export.length === 0) {
      console.log('ERROR: No animation clips added to export')
      return
    }

    // Retarget all animation clips before export
    let retargeted_clips: AnimationClip[] = []
    retargeted_clips = this.animation_clips_to_export.map((clip) =>
      AnimationRetargetService.getInstance().retarget_animation_clip(clip)
    )
    console.log('Retargeted animation clips:', retargeted_clips)

    const target_rig_scene: Scene = AnimationRetargetService.getInstance().get_target_armature()
    this.export_glb(target_rig_scene, retargeted_clips, filename)
      .then(() => {
        console.log('Exported GLB successfully')
      })
      .catch((error) => { console.log('Error exporting GLB:', error) })
  }

  public async export_glb (exported_scene: Scene, animations_to_export: AnimationClip[], file_name: string): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const gltf_exporter = new GLTFExporter()

      const export_options = {
        binary: true,
        onlyVisible: false,
        embedImages: true,
        animations: animations_to_export
      }

      gltf_exporter.parse(
        exported_scene,
        (result: ArrayBuffer) => {
          // Handle the result of the export
          if (result !== null) {
            this.save_array_buffer(result, `${file_name}.glb`)
            resolve() // Resolve the promise when the export is complete
          } else {
            console.log('ERROR: result is not an instance of ArrayBuffer')
            reject(new Error('Export result is not an ArrayBuffer'))
          }
        },
        (error: any) => {
          console.log('An error happened during parsing', error)
          reject(error) // Reject the promise if an error occurs
        },
        export_options
      )
    })
  }

  private save_file (blob: Blob, filename: string): void {
    const export_button_hidden_link: HTMLAnchorElement | null = document.querySelector('#download-hidden-link')
    if (export_button_hidden_link != null) {
      export_button_hidden_link.href = URL.createObjectURL(blob)
      export_button_hidden_link.download = filename
      export_button_hidden_link.click()
    } else {
      console.log('ERROR: dom_export_button_hidden_link is null')
    }
  }

  // used for GLB to turn content into a byte array for saving
  private save_array_buffer (buffer: ArrayBuffer, filename: string): void {
    if (window.parent === window) {
      this.save_file(new Blob([buffer], { type: 'application/octet-stream' }), filename)
      return
    }
    const launch_context = new URLSearchParams(window.location.search)
    window.parent.postMessage({
      source: 'aismith-mesh2motion',
      type: 'export-glb',
      run: launch_context.get('run'),
      input: launch_context.get('input'),
      mode: launch_context.get('mode'),
      filename,
      animations: this.animation_clips_to_export.map((clip) => clip.name),
      skeleton: 'Retargeted',
      exportContents: 'Skinned textured model and selected animations',
      boneNaming: 'Preserve target rig',
      buffer
    }, window.location.origin, [buffer])
  }
}
