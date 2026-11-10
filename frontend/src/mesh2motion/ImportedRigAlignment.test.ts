import { describe, expect, it } from 'vitest'
import { imported_rig_alignment } from '../../../vendor/mesh2motion-app/src/lib/processes/edit-skeleton/ImportedRigAlignment'

describe('align_imported_armature', () => {
  it('matches skeleton height and feet to the normalized model', () => {
    expect(imported_rig_alignment(
      { minY: 0, maxY: 4, centerX: 0, centerZ: 0 },
      { minY: 10, maxY: 20, centerX: 2, centerZ: -3 },
    )).toEqual({ scale: 0.4, offsetX: -0.8, offsetY: -4, offsetZ: 1.2000000000000002 })
  })
})
