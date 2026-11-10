import { describe, expect, it } from 'vitest'

import { artifactForHistoryUse, bilateralSymmetrySummary, canBakeMotionChild, canPreviewArtifact, currentArtifactStage, deformationReviewArtifacts, formatSetting, hasActiveMotionBake, isAutomatedQualityAction, isRetainedCandidate, isRigPlaybackStage, isRunPipelineLocked, qualityActionsForDisplay, shouldRestoreRefineImage, skeletonReviewArtifacts, stageForQualityAction, studioTabForSourceKind, studioTabForStage, waitForRunTerminal } from './history'


describe('studioTabForStage', () => {
  it.each([
    ['prepare-image', 'generate'],
    ['generate', 'retopologize'],
    ['retopologize', 'retopologize'],
    ['retopo', 'retopologize'],
    ['paint-bake', 'paint'],
    ['project-views', 'retopologize'],
    ['rig', 'rig'],
  ])('carries a %s artifact into the %s workspace', (stage, expected) => {
    expect(studioTabForStage(stage)).toBe(expected)
  })
})

describe('deformationReviewArtifacts', () => {
  it('returns only durable, well-formed review image references', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: { deformation_review: { review_artifacts: [
        { artifact_id: 'rest', label: 'rest', frame_ratio: 0, rest_pose: true },
        { artifact_id: 'hand', label: 'weapon-hand-50', frame_ratio: 0.5, rest_pose: false, focus_bone: 'right-hand' },
        { artifact_id: 4, label: 'bad', frame_ratio: 0.5, rest_pose: false },
      ] } },
    }

    expect(deformationReviewArtifacts(artifact)).toEqual([
      { artifact_id: 'rest', label: 'rest', frame_ratio: 0, rest_pose: true },
      { artifact_id: 'hand', label: 'weapon-hand-50', frame_ratio: 0.5, rest_pose: false, focus_bone: 'right-hand' },
    ])
  })
})

describe('skeletonReviewArtifacts', () => {
  it('returns only well-formed skeleton review image references', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: { skeleton_review: { review_artifacts: [
        { artifact_id: 'sheet', label: 'skeleton-sheet' },
        { artifact_id: 'front', label: 'front' },
        { artifact_id: 4, label: 'bad' },
      ] } },
    }

    expect(skeletonReviewArtifacts(artifact)).toEqual([
      { artifact_id: 'sheet', label: 'skeleton-sheet' },
      { artifact_id: 'front', label: 'front' },
    ])
  })

  it('returns nothing without a skeleton_review block', () => {
    const artifact = { id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0, metadata: {} }

    expect(skeletonReviewArtifacts(artifact)).toEqual([])
  })
})

describe('bilateralSymmetrySummary', () => {
  it('reads the p95, band, outside-band count, and worst offender', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: { rig_quality: {
        symmetry_mirror_error_p95: 0.08,
        symmetry_pairs_outside_band_count: 1,
        bilateral_symmetry: {
          band: 0.03,
          worst_offenders: [{ left: 'hand_l', right: 'hand_r', mirror_error_normalized: 0.08 }],
        },
      } },
    }

    expect(bilateralSymmetrySummary(artifact)).toEqual({
      p95: 0.08, band: 0.03, pairsOutsideBand: 1, worstPair: 'hand_l/hand_r',
    })
  })

  it('returns null when a rig predates the measurement', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: { rig_quality: { joints: 21 } },
    }

    expect(bilateralSymmetrySummary(artifact)).toBeNull()
  })
})

describe('formatSetting', () => {
  it('preserves structured generation settings', () => {
    expect(formatSetting([{ x: 0.25, color: '#fff' }])).toBe('[{"x":0.25,"color":"#fff"}]')
  })
})

describe('artifactForHistoryUse', () => {
  it('returns the persisted artifact including its rig certificate metadata', () => {
    const artifact = {
      id: 'rigged-mary',
      name: 'mary.glb',
      kind: 'glb',
      size_bytes: 10,
      created_at: 1,
      metadata: { quality_decisions: { 'rig-certificate': { status: 'ready' as const, policy_version: 1, target_stage: 'rig-certificate', findings: [], recommended_actions: [] } } },
    }

    expect(artifactForHistoryUse({ [artifact.id]: artifact }, artifact.id)).toBe(artifact)
  })
})

describe('waitForRunTerminal', () => {
  it('keeps polling while any durable attempt is queued or running', async () => {
    const states = [
      { attempts: [{ status: 'queued' }] },
      { attempts: [{ status: 'running' }] },
      { attempts: [{ status: 'completed' }] },
    ]
    let waits = 0

    const result = await waitForRunTerminal(
      async () => states.shift()!,
      async () => { waits += 1 },
    )

    expect(result.attempts[0]?.status).toBe('completed')
    expect(waits).toBe(2)
  })
})

describe('studioTabForSourceKind', () => {
  it.each([
    ['image', 'generate'],
    ['glb', 'rig'],
    ['gltf', 'retopologize'],
    ['fbx', 'retopologize'],
    ['obj', 'retopologize'],
  ])('restores a %s source into %s', (kind, expected) => {
    expect(studioTabForSourceKind(kind)).toBe(expected)
  })
})

describe('stageForQualityAction', () => {
  it.each([
    ['refine-reconstruct', 'retopologize'],
    ['retopologize', 'retopo'],
    ['repair-topology', 'retopo'],
    ['inspect-weight-paint', 'rig-edit'],
    ['fit-bind-pose', 'rig-edit'],
    ['attach-animation', 'rig-edit'],
  ])('routes %s into %s', (action, expected) => {
    expect(stageForQualityAction(action)).toBe(expected)
  })
})

describe('shouldRestoreRefineImage', () => {
  it('retains an image source when a generated artifact is sent back through Refine', () => {
    expect(shouldRestoreRefineImage('retopologize', 'image', false)).toBe(true)
  })

  it.each([
    ['retopo', 'image', false],
    ['retopologize', 'glb', false],
    ['retopologize', 'image', true],
  ])('does not attach a reference for stage=%s source=%s isSource=%s', (stage, sourceKind, isSource) => {
    expect(shouldRestoreRefineImage(stage, sourceKind, isSource)).toBe(false)
  })
})

describe('quality action execution', () => {
  it('keeps automated recovery visible on compact retained candidates', () => {
    expect(qualityActionsForDisplay(['inspect-weight-paint'], true)).toEqual([])
    expect(qualityActionsForDisplay(['inspect-weight-paint'], false)).toEqual(['inspect-weight-paint'])
  })

  it.each(['retry-generation', 'refine-reconstruct', 'repair-topology', 'retopologize', 'repair-weights'])('marks %s as backend executable', (action) => {
    expect(isAutomatedQualityAction(action)).toBe(true)
  })

  it.each(['inspect-boundaries', 'manual-blender-cleanup', 'inspect-weight-paint'])('keeps %s in the human workflow', (action) => {
    expect(isAutomatedQualityAction(action)).toBe(false)
  })
})

describe('isRigPlaybackStage', () => {
  it.each(['rig', 'auto-rig', 'rig-adjust', 'repair-weights', 'rig-transfer', 'target-proxy-rig', 'retarget-motion', 'apply-clip'])('plays a certified %s artifact in the animation viewer', (stage) => {
    expect(isRigPlaybackStage(stage)).toBe(true)
  })

  it.each(['rig-edit', 'retopo', 'retopologize'])('keeps an unfinished %s artifact in its editing workflow', (stage) => {
    expect(isRigPlaybackStage(stage)).toBe(false)
  })
})

describe('isRetainedCandidate', () => {
  it('identifies a completed output that quality policy declined to promote', () => {
    expect(isRetainedCandidate({ status: 'completed', output_artifact_id: 'candidate', promoted: false }, 'current')).toBe(true)
  })

  it('does not label the current artifact or an ordinary historical output as retained', () => {
    expect(isRetainedCandidate({ status: 'completed', output_artifact_id: 'current', promoted: false }, 'current')).toBe(false)
    expect(isRetainedCandidate({ status: 'completed', output_artifact_id: 'old', promoted: true }, 'current')).toBe(false)
  })

  it('keeps BVH motion children out of model candidate promotion', () => {
    expect(isRetainedCandidate({ status: 'completed', output_artifact_id: 'motion', promoted: false }, 'character', 'bvh')).toBe(false)
  })

  it('keeps completed preparation images recoverable for later approval', () => {
    expect(isRetainedCandidate({ status: 'completed', output_artifact_id: 'prepared', promoted: false }, 'source', 'image')).toBe(true)
  })
})

describe('canPreviewArtifact', () => {
  it('sends playable GLBs to the viewport and keeps BVH text out', () => {
    expect(canPreviewArtifact('glb')).toBe(true)
    expect(canPreviewArtifact('image')).toBe(true)
    expect(canPreviewArtifact('bvh')).toBe(false)
  })
})

describe('canBakeMotionChild', () => {
  it('offers retargeting only for a BVH child and certified current GLB', () => {
    const readyGates = { rig: { status: 'ready' }, 'animation-review': { status: 'ready' }, 'deformation-review': { status: 'ready' } }
    const certified = { kind: 'glb', metadata: { quality_decisions: { ...readyGates, 'rig-certificate': { status: 'ready', policy_version: 2 } } } }
    const stale = { kind: 'glb', metadata: { quality_decisions: { ...readyGates, 'rig-certificate': { status: 'ready', policy_version: 1 } } } }
    const topologyBlocked = { kind: 'glb', metadata: { quality_decisions: { ...readyGates, rig: { status: 'blocked' }, 'rig-certificate': { status: 'ready', policy_version: 2 } } } }
    expect(canBakeMotionChild({ kind: 'bvh' }, certified, true)).toBe(true)
    expect(canBakeMotionChild({ kind: 'bvh' }, certified, false)).toBe(false)
    expect(canBakeMotionChild({ kind: 'glb' }, certified, true)).toBe(false)
    expect(canBakeMotionChild({ kind: 'bvh' }, { kind: 'glb', metadata: {} }, true)).toBe(false)
    expect(canBakeMotionChild({ kind: 'bvh' }, stale, true)).toBe(false)
    expect(canBakeMotionChild({ kind: 'bvh' }, topologyBlocked, true)).toBe(false)
  })
})

describe('hasActiveMotionBake', () => {
  it.each(['queued', 'running'])('restores the bake lock for a durable %s attempt', (status) => {
    expect(hasActiveMotionBake([{ stage: 'retarget-motion', status }])).toBe(true)
  })

  it('ignores completed motion bakes and unrelated active stages', () => {
    expect(hasActiveMotionBake([
      { stage: 'retarget-motion', status: 'completed' },
      { stage: 'retopologize', status: 'running' },
    ])).toBe(false)
  })
})

describe('isRunPipelineLocked', () => {
  it('locks every run mutation for either a local action or any durable active stage', () => {
    expect(isRunPipelineLocked([], 'repair-topology')).toBe(true)
    expect(isRunPipelineLocked([{ stage: 'retarget-motion', status: 'running' }], '')).toBe(true)
    expect(isRunPipelineLocked([{ stage: 'paint-bake', status: 'queued' }], '')).toBe(true)
    expect(isRunPipelineLocked([{ stage: 'retarget-motion', status: 'completed' }], '')).toBe(false)
  })
})

describe('currentArtifactStage', () => {
  it('reports the producing stage instead of a later promotion audit event', () => {
    expect(currentArtifactStage({
      current_artifact_id: 'refined',
      attempts: [
        { stage: 'retopologize', output_artifact_id: 'refined' },
        { stage: 'retopo', output_artifact_id: 'rejected' },
        { stage: 'promote-artifact', output_artifact_id: 'refined' },
      ],
    })).toBe('retopologize')
  })

  it('keeps an animated rig in playback after diagnostics inspect the same GLB', () => {
    expect(currentArtifactStage({
      current_artifact_id: 'animated-rig',
      attempts: [
        { stage: 'target-proxy-rig', output_artifact_id: 'animated-rig' },
        { stage: 'rig-diagnostics', output_artifact_id: 'animated-rig' },
        { stage: 'deformation-review', output_artifact_id: 'animated-rig' },
      ],
    })).toBe('target-proxy-rig')
  })
})
