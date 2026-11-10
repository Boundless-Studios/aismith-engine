import { describe, expect, it } from 'vitest'
import { target_model_camera_position, target_model_camera_target } from '../../../vendor/mesh2motion-app/src/retarget/steps/TargetModelCamera'

describe('target model camera framing', () => {
  it('keeps a full-height humanoid close enough for animation review', () => {
    const position = target_model_camera_position(
      { x: 0, y: 1, z: 0 },
      { x: 1, y: 2, z: 0.5 },
    )

    expect(position).toMatchObject({ x: 0.3, y: 0.9, z: 4.4 })
    expect(target_model_camera_target(
      { x: 0, y: 1, z: 0 },
      { x: 1, y: 2, z: 0.5 },
    )).toMatchObject({ x: 0, y: 0.8, z: 0 })
  })
})
