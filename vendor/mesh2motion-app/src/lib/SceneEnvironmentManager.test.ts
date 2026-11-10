import { BoxGeometry, Mesh, MeshBasicMaterial, PerspectiveCamera, Scene, Vector3 } from 'three'
import { describe, expect, it } from 'vitest'
import { SceneEnvironmentManager } from './SceneEnvironmentManager'

describe('SceneEnvironmentManager model framing', () => {
  it('frames the loaded model around its center at a bounds-derived distance', () => {
    const camera = new PerspectiveCamera(15, 2, 0.1, 1000)
    camera.position.set(0, 1.7, 15)
    const manager = new SceneEnvironmentManager(
      new Scene(),
      {} as never,
      camera,
      {} as never,
      {} as never,
      {} as never,
    )
    const model = new Mesh(new BoxGeometry(2, 0.6, 1), new MeshBasicMaterial())
    model.position.set(4, 1, -2)
    model.updateWorldMatrix(true, true)
    const frameObject = (manager as unknown as {
      frame_object?: (object: Mesh) => void
    }).frame_object

    expect(frameObject).toBeTypeOf('function')
    frameObject?.call(manager, model)

    expect(camera.position.x).toBeCloseTo(4)
    expect(camera.position.y).toBeCloseTo(1)
    expect(camera.position.z).toBeGreaterThan(-1.5)
    expect(camera.position.z).toBeLessThan(15)
    expect(camera.getWorldDirection(new Vector3()).y).toBeCloseTo(0)
  })
})
