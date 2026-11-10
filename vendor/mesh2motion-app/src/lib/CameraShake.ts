import { PerspectiveCamera, Vector3 } from 'three'

/**
 * Applies a gentle screen-shake to a PerspectiveCamera.
 * Call `start()` to begin, then `update(delta)` every frame.
 * The shake decays over the configured duration and restores the
 * original camera position when finished.
 */
export class CameraShake {
  private readonly camera: PerspectiveCamera
  private is_shaking: boolean = false
  private elapsed: number = 0
  private duration: number = 0
  private intensity: number = 0
  private original_position: Vector3 = new Vector3()

  constructor (camera: PerspectiveCamera) {
    this.camera = camera
  }

  /**
   * Begin a shake effect.
   * @param intensity  Maximum offset in world units (default 0.01)
   * @param duration   How long the shake lasts in seconds (default 0.5)
   */
  public start (intensity: number = 0.02, duration: number = 0.7): void {
    this.original_position.copy(this.camera.position)
    this.intensity = intensity
    this.duration = duration
    this.elapsed = 0
    this.is_shaking = true
  }

  /** Call once per frame with the frame delta in seconds. */
  public update (delta: number): void {
    if (!this.is_shaking) return

    this.elapsed += delta
    const progress = Math.min(this.elapsed / this.duration, 1)

    if (progress >= 1) {
      // restore original position and stop
      this.camera.position.copy(this.original_position)
      this.is_shaking = false
      return
    }

    // decay multiplier so shake eases out
    const decay = 1 - progress
    const strength = this.intensity * decay

    // use sine waves at different frequencies per axis for smooth, organic motion
    const t = this.elapsed * 2 * Math.PI
    this.camera.position.set(
      this.original_position.x + Math.sin(t * 3.1) * strength,
      this.original_position.y + Math.sin(t * 4.7) * strength,
      this.original_position.z + Math.sin(t * 2.3) * strength
    )
  }

  public get active (): boolean {
    return this.is_shaking
  }
}
