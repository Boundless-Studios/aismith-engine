import { describe, expect, it } from 'vitest'
import { batchClipIds, pendingClipIds, poseTimeForClip, THUMBNAIL_BATCH_SIZE, THUMBNAIL_POSE_FRACTION } from './clipGrid'

describe('batchClipIds', () => {
  it('splits into batchSize-sized chunks, preserving order', () => {
    const ids = ['a', 'b', 'c', 'd', 'e']
    expect(batchClipIds(ids, 2)).toEqual([['a', 'b'], ['c', 'd'], ['e']])
  })

  it('uses THUMBNAIL_BATCH_SIZE by default', () => {
    const ids = Array.from({ length: 20 }, (_, index) => `clip-${index}`)
    const batches = batchClipIds(ids)
    expect(batches.every((batch) => batch.length <= THUMBNAIL_BATCH_SIZE)).toBe(true)
    expect(batches.flat()).toEqual(ids)
  })

  it('returns one exact-sized batch when the list divides evenly', () => {
    expect(batchClipIds(['a', 'b', 'c', 'd'], 2)).toEqual([['a', 'b'], ['c', 'd']])
  })

  it('returns an empty array for an empty input', () => {
    expect(batchClipIds([])).toEqual([])
  })

  it('falls back to one batch when batchSize is non-positive', () => {
    expect(batchClipIds(['a', 'b'], 0)).toEqual([['a', 'b']])
  })
})

describe('pendingClipIds', () => {
  it('excludes ids already present in the cache', () => {
    const cache = new Map([['a', 'data:a'], ['c', 'data:c']])
    expect(pendingClipIds(['a', 'b', 'c', 'd'], cache)).toEqual(['b', 'd'])
  })

  it('preserves the order of clipIds', () => {
    const cache = new Map<string, string>()
    expect(pendingClipIds(['z', 'a', 'm'], cache)).toEqual(['z', 'a', 'm'])
  })

  it('returns an empty array once everything is cached', () => {
    const cache = new Map([['a', 'data:a'], ['b', 'data:b']])
    expect(pendingClipIds(['a', 'b'], cache)).toEqual([])
  })

  it('treats an empty-string cache entry as still cached (a resolved miss, not pending)', () => {
    const cache = new Map([['broken', '']])
    expect(pendingClipIds(['broken'], cache)).toEqual([])
  })
})

describe('poseTimeForClip', () => {
  it('applies THUMBNAIL_POSE_FRACTION by default', () => {
    expect(poseTimeForClip(2)).toBeCloseTo(2 * THUMBNAIL_POSE_FRACTION)
  })

  it('applies a custom fraction', () => {
    expect(poseTimeForClip(10, 0.25)).toBe(2.5)
  })

  it('returns 0 for a zero duration', () => {
    expect(poseTimeForClip(0)).toBe(0)
  })

  it('returns 0 for a negative duration', () => {
    expect(poseTimeForClip(-3)).toBe(0)
  })

  it('returns 0 for a non-finite duration instead of NaN', () => {
    expect(poseTimeForClip(NaN)).toBe(0)
    expect(poseTimeForClip(Infinity)).toBe(0)
  })
})
