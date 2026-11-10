import { describe, expect, it } from 'vitest'
import { requiresHumanRetargeting } from './retargetContract'

describe('animation retarget contract', () => {
  it('converts human motion even when fitted target bones keep Mesh2Motion names', () => {
    expect(requiresHumanRetargeting('human')).toBe(true)
  })
})
