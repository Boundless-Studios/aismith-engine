import { describe, expect, it } from 'vitest'

import { generateSourceFields, latestCompletedAttempt, rehydrateViewSlots, sourceViewChipLabels, VIEW_LABEL_ORDER } from './workspace'

describe('generateSourceFields', () => {
  it('keeps the legacy single-image shape when no optional views are set', () => {
    expect(generateSourceFields('front-id', {})).toEqual({ source_artifact_id: 'front-id' })
    expect(generateSourceFields('front-id', { back: undefined, left: undefined, right: undefined })).toEqual({
      source_artifact_id: 'front-id',
    })
  })

  it('switches to source_views once a back view is set, and omits source_artifact_id', () => {
    const payload = generateSourceFields('front-id', { back: 'back-id' })
    expect(payload).toEqual({ source_views: { front: 'front-id', back: 'back-id' } })
    expect(payload).not.toHaveProperty('source_artifact_id')
  })

  it('carries every set optional view, limited to back/left/right', () => {
    expect(generateSourceFields('front-id', { back: 'back-id', left: 'left-id', right: 'right-id' })).toEqual({
      source_views: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
    })
  })

  it('omits an optional view that was cleared', () => {
    expect(generateSourceFields('front-id', { back: 'back-id', left: undefined })).toEqual({
      source_views: { front: 'front-id', back: 'back-id' },
    })
  })
})

describe('generateSourceFields after a rehydrated views state (no silent single-view fallback)', () => {
  // The guard the operator asked for: after a journey-step round trip (or a
  // full reload) wipes the in-memory `views` state and rehydrateViewSlots()
  // backfills it from the run's synthesize-views artifacts, launching
  // Generate must submit the exact same 4-view source_views payload it
  // would have immediately after synthesis -- never a silent fallback to
  // the single-image `source_artifact_id` shape just because `views` was
  // momentarily empty.
  const frontId = 'front-id'
  const synthesizedEntries: Array<readonly ['back' | 'left' | 'right', string]> = [
    ['back', 'synth-back-id'],
    ['left', 'synth-left-id'],
    ['right', 'synth-right-id'],
  ]

  it('produces the identical payload before the reset and after rehydration restores it', () => {
    const rightAfterSynthesis = generateSourceFields(frontId, {
      back: synthesizedEntries[0][1],
      left: synthesizedEntries[1][1],
      right: synthesizedEntries[2][1],
    })

    // Simulate the step-switch bug: the in-memory state is wiped...
    const wipedByNavigation: Partial<Record<'back' | 'left' | 'right', string>> = {}
    // ...then rehydrateViewSlots backfills it from the run's persisted
    // synthesize-views artifacts before Generate is launched again.
    const rehydrated = rehydrateViewSlots(wipedByNavigation, synthesizedEntries, [])

    expect(generateSourceFields(frontId, rehydrated)).toEqual(rightAfterSynthesis)
    expect(generateSourceFields(frontId, rehydrated)).toHaveProperty('source_views')
    expect(generateSourceFields(frontId, rehydrated)).not.toHaveProperty('source_artifact_id')
  })

  it('never falls back to a single-view payload once any view was ever synthesized', () => {
    const rehydrated = rehydrateViewSlots({}, synthesizedEntries, [])
    const payload = generateSourceFields(frontId, rehydrated) as { source_views?: Record<string, string> }

    expect(payload.source_views).toBeDefined()
    expect(Object.keys(payload.source_views || {})).toEqual(['front', 'back', 'left', 'right'])
  })
})

describe('sourceViewChipLabels', () => {
  it('returns the labels in the fixed front/back/left/right order regardless of input order', () => {
    expect(sourceViewChipLabels(['right', 'front', 'back'])).toEqual(['front', 'back', 'right'])
  })

  it('ignores unknown labels and non-array metadata', () => {
    expect(sourceViewChipLabels(['front', 'top-down'])).toEqual(['front'])
    expect(sourceViewChipLabels(undefined)).toEqual([])
    expect(sourceViewChipLabels(null)).toEqual([])
    expect(sourceViewChipLabels('front')).toEqual([])
  })

  it('is limited to the four labels in VIEW_LABEL_ORDER', () => {
    expect(VIEW_LABEL_ORDER).toEqual(['front', 'back', 'left', 'right'])
  })
})

describe('latestCompletedAttempt', () => {
  const attempts = [
    { stage: 'generate', status: 'completed', output_artifact_id: 'gen-1' },
    { stage: 'retopo', status: 'completed', output_artifact_id: 'clean-1' },
    { stage: 'generate', status: 'failed', output_artifact_id: null },
    { stage: 'generate', status: 'completed', output_artifact_id: 'gen-2' },
  ]

  it('returns the most recent completed attempt for the stage with an output artifact', () => {
    expect(latestCompletedAttempt(attempts, 'generate')?.output_artifact_id).toBe('gen-2')
  })

  it('returns undefined when the stage never completed', () => {
    expect(latestCompletedAttempt(attempts, 'paint-bake')).toBeUndefined()
    expect(latestCompletedAttempt(undefined, 'generate')).toBeUndefined()
  })
})
