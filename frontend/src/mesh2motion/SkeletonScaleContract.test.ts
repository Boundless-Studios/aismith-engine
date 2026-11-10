import { describe, expect, it } from 'vitest'
import { scale_to_match_height, vertical_span } from '../../../vendor/mesh2motion-app/src/lib/processes/load-skeleton/SkeletonScaleContract'

describe('scale_to_match_height', () => {
  it('normalizes a skeleton to the loaded character height', () => {
    expect(scale_to_match_height(1.8, 3.6)).toBe(0.5)
  })

  it('clamps unsafe or invalid scale ratios', () => {
    expect(scale_to_match_height(1, 0)).toBe(1)
    expect(scale_to_match_height(1, 100)).toBe(0.1)
    expect(scale_to_match_height(100, 1)).toBe(2)
  })

  it('measures a bone-only armature from joint positions', () => {
    expect(vertical_span([0.2, -1.8, 1.8, 0.9])).toBe(3.6)
    expect(vertical_span([])).toBe(0)
  })
})
