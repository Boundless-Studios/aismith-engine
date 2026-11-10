import { describe, expect, it } from 'vitest'
import { resolveRetargetLaunch } from './RetargetLaunchContract'

describe('resolveRetargetLaunch', () => {
  it('accepts a same-origin GLB rig and defaults to the animation library', () => {
    expect(resolveRetargetLaunch('?source=%2Fapi%2Fassets%2Frig%2Fdownload&extension=glb')).toEqual({
      source: '/api/assets/rig/download',
      extension: 'glb',
      autoOpenLibrary: true,
    })
    expect(resolveRetargetLaunch('?source=http%3A%2F%2F127.0.0.1%3A8100%2Fapi%2Fassets%2Frig%2Fdownload&extension=glb')).toMatchObject({
      source: 'http://127.0.0.1:8100/api/assets/rig/download',
    })
    expect(resolveRetargetLaunch(
      '?source=http%3A%2F%2F192.0.2.10%3A8100%2Fapi%2Fassets%2Frig%2Fdownload&extension=glb',
      '192.0.2.10',
    )).toMatchObject({ source: 'http://192.0.2.10:8100/api/assets/rig/download' })
  })

  it('rejects incomplete and unsupported launches', () => {
    expect(resolveRetargetLaunch('?extension=glb')).toBeNull()
    expect(resolveRetargetLaunch('?source=%2Fmodel.fbx&extension=fbx')).toBeNull()
    expect(resolveRetargetLaunch('?source=https%3A%2F%2Fevil.example%2Fmodel.glb&extension=glb')).toBeNull()
  })
})
