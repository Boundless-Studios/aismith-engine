import { describe, expect, it } from 'vitest'

import { ACTIVE_JOB_STORAGE_KEY, ACTIVE_RUN_STORAGE_KEY, activeJobResyncDecision, activeJobResyncId, applyClipJobPayload, autoRigCaption, canAcceptMesh2MotionContext, canAcceptMesh2MotionExport, canGenerateFromPreparedImage, canGenerateMotion, canonicalizeCandidatePinnable, canRestoreHistoryArtifact, cleanMethodForStage, cleanRecommendation, cleanTargetExceedsSource, exportGlbNameHint, generateSourceViewArtifactIds, hasActiveStudioJob, isActiveJobStatus, isArrayBufferPayload, isMotionPipelineJob, isReCleaningCleanOutput, isTemplateConformantRig, jobPollIntervalMs, mergeSynthesizedViews, mesh2MotionExportInputArtifactId, motionJobPayload, pendingPreparedCandidateId, pollJobUntilTerminal, preparationJobPayload, readRememberedJob, rehydrateViewSlots, restoreHistoryArtifacts, retargetMotionJobPayload, retryNetworkPost, rigWorkspaceSurface, runAssetOptions, shouldActivateRecordedArtifact, shouldApplyCompletedArtifact, shouldApplyHydratedRun, shouldMarkTerminalHydrationComplete, shouldPreviewRecordedArtifact, shouldPromoteCompletedOutput, serializeRememberedJob, shouldRefreshCapabilities, shouldRestoreRun, shouldResyncActiveJob, shouldStartTerminalHydration, showsPrepareCard, autoRigJobPayload, synthesizeViewsJobPayload, synthesizedViewArtifactIds, viewConsistencyNotice,
  createTposeSetJobPayload, isPreparedFrontStage, latestPreparedFrontAttempt, latestViewSourceAttempt, tposeSetMembership, tposeSetViewArtifactIds, viewRehydrationSources,
} from './workspace'


describe('motionJobPayload', () => {
  it('creates a reproducible Sword Attack request attached to the current run', () => {
    expect(motionJobPayload('rigged-mary', 'mary-run', 23)).toEqual({
      kind: 'generate-motion',
      source_artifact_id: 'rigged-mary',
      run_id: 'mary-run',
      seed: 23,
      settings: {
        prompt: 'A stylized female sword fighter performs one decisive horizontal sword slash, plants her feet, follows through, and recovers to a combat-ready stance',
        duration_seconds: 2,
      },
    })
  })
})

describe('active journey restoration', () => {
  it('restores only a durable run that still exists', () => {
    const runs = [{ id: 'mary' }, { id: 'tess' }]

    expect(ACTIVE_RUN_STORAGE_KEY).toBe('aismith-active-run-id')
    expect(shouldRestoreRun('mary', runs)).toBe('mary')
    expect(shouldRestoreRun('deleted', runs)).toBeNull()
    expect(shouldRestoreRun(null, runs)).toBeNull()
  })
})

describe('preparationJobPayload', () => {
  it('requests the pinned local T-pose workflow without promoting the result', () => {
    expect(preparationJobPayload('mary-source', 'mary-run', 42)).toEqual({
      kind: 'prepare-image',
      source_artifact_id: 'mary-source',
      run_id: 'mary-run',
      seed: 42,
      settings: {
        adapter: 'comfyui',
        profile: 'humanoid-t-pose-rig-v1',
      },
    })
  })

  it('supports the upload pass-through when ComfyUI is unavailable', () => {
    expect(preparationJobPayload('mary-source', null, 7, 'upload').settings.adapter).toBe('upload')
  })

  it('requests the pinned T-pose workflow explicitly', () => {
    expect(
      preparationJobPayload('mary-source', 'mary-run', 42, 'comfyui', 'humanoid-t-pose-rig-v1').settings.profile,
    ).toBe('humanoid-t-pose-rig-v1')
  })

  it('retains the A-pose workflow as an explicit compatibility profile', () => {
    expect(
      preparationJobPayload('mary-source', null, 7, 'comfyui', 'humanoid-a-pose-rig-v3').settings.profile,
    ).toBe('humanoid-a-pose-rig-v3')
  })
})

describe('synthesizeViewsJobPayload', () => {
  it('requests all three views by default without promoting the result', () => {
    expect(synthesizeViewsJobPayload('mary-front', 'mary-run', 42)).toEqual({
      kind: 'synthesize-views',
      source_artifact_id: 'mary-front',
      run_id: 'mary-run',
      seed: 42,
      settings: { views: ['back', 'left', 'right'] },
    })
  })

  it('supports requesting a narrower subset of views', () => {
    expect(synthesizeViewsJobPayload('mary-front', null, 7, ['back']).settings.views).toEqual(['back'])
  })
})

describe('synthesizedViewArtifactIds', () => {
  it('reads the primary artifact\'s own synthesized_view label', () => {
    expect(synthesizedViewArtifactIds('back-id', { synthesized_view: 'back' })).toEqual([
      ['back', 'back-id'],
    ])
  })

  it('combines the primary label with sibling_views, ordered front/back/left/right', () => {
    expect(synthesizedViewArtifactIds('back-id', {
      synthesized_view: 'back',
      sibling_views: { right: 'right-id', left: 'left-id' },
    })).toEqual([
      ['back', 'back-id'],
      ['left', 'left-id'],
      ['right', 'right-id'],
    ])
  })

  it('ignores malformed or missing metadata', () => {
    expect(synthesizedViewArtifactIds('back-id', undefined)).toEqual([])
    expect(synthesizedViewArtifactIds('back-id', { synthesized_view: 'top' })).toEqual([])
    expect(synthesizedViewArtifactIds('back-id', { sibling_views: 'not-an-object' })).toEqual([])
  })
})

describe('mergeSynthesizedViews', () => {
  it('fills empty slots with synthesized artifacts', () => {
    expect(mergeSynthesizedViews({}, [['back', 'back-id'], ['left', 'left-id']])).toEqual({
      back: 'back-id',
      left: 'left-id',
    })
  })

  it('never replaces a slot the operator already filled', () => {
    expect(mergeSynthesizedViews(
      { back: 'operator-uploaded-back' },
      [['back', 'synthesized-back'], ['left', 'synthesized-left']],
    )).toEqual({
      back: 'operator-uploaded-back',
      left: 'synthesized-left',
    })
  })
})

describe('generateSourceViewArtifactIds', () => {
  it('reads back/left/right ids from a completed generate output\'s source_views metadata', () => {
    expect(generateSourceViewArtifactIds({
      source_views: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
    })).toEqual([
      ['back', 'back-id'],
      ['left', 'left-id'],
      ['right', 'right-id'],
    ])
  })

  it('omits front and any label not present', () => {
    expect(generateSourceViewArtifactIds({ source_views: { front: 'front-id', back: 'back-id' } })).toEqual([
      ['back', 'back-id'],
    ])
  })

  it('ignores malformed or missing metadata', () => {
    expect(generateSourceViewArtifactIds(undefined)).toEqual([])
    expect(generateSourceViewArtifactIds({})).toEqual([])
    expect(generateSourceViewArtifactIds({ source_views: 'not-an-object' })).toEqual([])
  })
})

describe('rehydrateViewSlots', () => {
  it('backfills empty slots from the synthesized set when no generate attempt exists yet', () => {
    // The K18/t0z17 case (bead reproduction, "zix" run): synthesize-views
    // completed but Generate has never run, so there is no source_views
    // metadata yet -- the synthesized set is the only thing to rehydrate from.
    expect(rehydrateViewSlots(
      {},
      [['back', 'synth-back'], ['left', 'synth-left'], ['right', 'synth-right']],
      [],
    )).toEqual({ back: 'synth-back', left: 'synth-left', right: 'synth-right' })
  })

  it('prefers the generate attempt\'s own source_views over an older or newer synthesis set', () => {
    // A completed generate attempt's source_views is the exact set it was
    // actually submitted with -- it wins over whatever synthesize-views
    // produced, whether that ran before or after (e.g. a re-synthesis the
    // operator never actually generated with).
    expect(rehydrateViewSlots(
      {},
      [['back', 'synth-back'], ['left', 'synth-left']],
      [['back', 'generated-back'], ['right', 'generated-right']],
    )).toEqual({ back: 'generated-back', left: 'synth-left', right: 'generated-right' })
  })

  it('never overwrites a slot the operator already has filled in this session', () => {
    expect(rehydrateViewSlots(
      { back: 'operator-current-back' },
      [['back', 'synth-back'], ['left', 'synth-left']],
      [['back', 'generated-back']],
    )).toEqual({ back: 'operator-current-back', left: 'synth-left' })
  })

  it('is a no-op on an empty run with nothing to rehydrate', () => {
    expect(rehydrateViewSlots({}, [], [])).toEqual({})
    expect(rehydrateViewSlots({ back: 'kept' }, [], [])).toEqual({ back: 'kept' })
  })
})

// Bead 4vquq: the composite step and the set it registers.
describe('canonical T-pose set', () => {
  const setBlock = {
    set_id: 'set-1',
    role: 'front',
    front_artifact_id: 'front-id',
    reference_artifact_id: 'reference-id',
    members: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
    failed_members: {},
    complete: true,
  }

  it('builds the composite job payload with the three fanned views', () => {
    expect(createTposeSetJobPayload('reference-id', 'punk-run', 11)).toEqual({
      kind: 'create-tpose-set',
      source_artifact_id: 'reference-id',
      run_id: 'punk-run',
      seed: 11,
      settings: { views: ['back', 'left', 'right'] },
    })
  })

  it('reads a member block into the studio shape', () => {
    const membership = tposeSetMembership({ tpose_set: setBlock })

    expect(membership).toEqual({
      setId: 'set-1',
      frontArtifactId: 'front-id',
      referenceArtifactId: 'reference-id',
      members: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
      failedRoles: [],
      complete: true,
    })
  })

  it('names the roles a partially-filled set is missing', () => {
    const membership = tposeSetMembership({
      tpose_set: {
        ...setBlock,
        members: { front: 'front-id', back: 'back-id', right: 'right-id' },
        failed_members: { left: 'ComfyUI rejected the left prompt' },
        complete: false,
      },
    })

    expect(membership?.failedRoles).toEqual(['left'])
    expect(membership?.complete).toBe(false)
    expect(membership?.members.left).toBeUndefined()
  })

  it('treats an artifact with no set block, or a malformed one, as not a member', () => {
    expect(tposeSetMembership(undefined)).toBeNull()
    expect(tposeSetMembership({})).toBeNull()
    expect(tposeSetMembership({ tpose_set: 'not-an-object' })).toBeNull()
    expect(tposeSetMembership({ tpose_set: { members: {} } })).toBeNull()
  })

  it('accepts either form of the step as an approvable T-pose front', () => {
    expect(isPreparedFrontStage('prepare-image')).toBe(true)
    expect(isPreparedFrontStage('create-tpose-set')).toBe(true)
    expect(isPreparedFrontStage('synthesize-views')).toBe(false)
    expect(isPreparedFrontStage(undefined)).toBe(false)
  })

  it('takes the newest front from whichever form of the step produced it', () => {
    const attempts = [
      { stage: 'prepare-image', status: 'completed', output_artifact_id: 'old-candidate' },
      { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
    ]
    expect(latestPreparedFrontAttempt(attempts)?.output_artifact_id).toBe('set-front')
    // A failed composite must never shadow the last good front.
    expect(latestPreparedFrontAttempt([
      ...attempts,
      { stage: 'create-tpose-set', status: 'failed', output_artifact_id: null },
    ])?.output_artifact_id).toBe('set-front')
  })

  it('rehydrates the view grid from the run’s newest view source, in run order', () => {
    // A run that rebuilt its canonical set must not be refilled from the
    // older loose synthesis the set replaced.
    expect(latestViewSourceAttempt([
      { stage: 'synthesize-views', status: 'completed', output_artifact_id: 'loose-set' },
      { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
    ])?.output_artifact_id).toBe('set-front')
    expect(latestViewSourceAttempt([
      { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
      { stage: 'synthesize-views', status: 'completed', output_artifact_id: 'regenerated-tile' },
    ])?.output_artifact_id).toBe('regenerated-tile')
    expect(latestViewSourceAttempt([])).toBeUndefined()
  })

  it('keeps the set front across a reload after a single-tile regenerate', () => {
    // regenerateView -> autoGenerateViews submits a `synthesize-views`
    // attempt on the same run; it must not shadow the set's front.
    const attempts = [
      { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
      { stage: 'synthesize-views', status: 'completed', output_artifact_id: 'regenerated-left' },
    ]
    const sources = viewRehydrationSources(attempts)
    expect(sources.setFront?.output_artifact_id).toBe('set-front')
    expect(sources.looseSynth?.output_artifact_id).toBe('regenerated-left')
    // A set that replaced an older loose synthesis must not be refilled from it.
    expect(viewRehydrationSources([
      { stage: 'synthesize-views', status: 'completed', output_artifact_id: 'loose-set' },
      { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
    ])).toEqual({
      setFront: { stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' },
      looseSynth: undefined,
    })
    // A run with no set keeps the loose flow; a failed set never counts.
    expect(viewRehydrationSources([
      { stage: 'create-tpose-set', status: 'failed', output_artifact_id: null },
      { stage: 'synthesize-views', status: 'completed', output_artifact_id: 'loose-set' },
    ]).setFront).toBeUndefined()
    expect(viewRehydrationSources([])).toEqual({ setFront: undefined, looseSynth: undefined })
  })

  it('restores the sibling tiles from the set membership, regenerated tile included', () => {
    // replace_tpose_set_member restamps `tpose_set.members` but not the
    // front's original `sibling_views`, so members are the truth.
    expect(tposeSetViewArtifactIds('set-front', {
      tpose_set: { set_id: 'set-1', front_artifact_id: 'set-front', members: { front: 'set-front', back: 'back-1', left: 'left-2', right: 'right-1' } },
      sibling_views: { back: 'back-1', left: 'left-1', right: 'right-1' },
    })).toEqual([['back', 'back-1'], ['left', 'left-2'], ['right', 'right-1']])
    // A front stamped before the membership block falls back to sibling_views.
    expect(tposeSetViewArtifactIds('set-front', {
      sibling_views: { back: 'back-1', right: 'right-1' },
    })).toEqual([['back', 'back-1'], ['right', 'right-1']])
    expect(tposeSetViewArtifactIds('set-front', undefined)).toEqual([])
  })

  it('names the set front explicitly when regenerating one tile of a set', () => {
    // Left to `select_view_synthesis_source`, an UNAPPROVED set front loses
    // to the raw upload -- the exact reference the set exists to retire.
    expect(synthesizeViewsJobPayload('reference-id', 'punk-run', 5, ['left'], 'set-front')).toEqual({
      kind: 'synthesize-views',
      source_artifact_id: 'reference-id',
      run_id: 'punk-run',
      seed: 5,
      settings: { views: ['left'], source_artifact_id: 'set-front' },
    })
  })

  it('surfaces a composite front as the candidate awaiting approval', () => {
    expect(pendingPreparedCandidateId({
      current_artifact_id: 'reference-id',
      attempts: [{ stage: 'create-tpose-set', status: 'completed', output_artifact_id: 'set-front' }],
    })).toBe('set-front')
  })

  it('lets a composite front be generated from once it is the source', () => {
    expect(canGenerateFromPreparedImage({ stage: 'create-tpose-set' }, false)).toBe(true)
    expect(canGenerateFromPreparedImage({ metadata: { stage: 'create-tpose-set' } }, false)).toBe(true)
  })
})

describe('pendingPreparedCandidateId', () => {
  it('surfaces the latest prepare-image candidate when it has not been promoted to current', () => {
    // The zix "backv4 final gate" bug: several agents/tabs each ran their
    // own prepare-image probe against a shared run tonight, and only the
    // newest one is the operator's actual approval target.
    const run = {
      current_artifact_id: 'front-upload',
      attempts: [
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'candidate-a' },
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'candidate-b' },
      ],
    }
    expect(pendingPreparedCandidateId(run)).toBe('candidate-b')
  })

  it('returns null once the latest candidate is already the run current artifact', () => {
    const run = {
      current_artifact_id: 'candidate-b',
      attempts: [
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'candidate-a' },
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'candidate-b' },
        { stage: 'promote-artifact', status: 'completed', output_artifact_id: 'candidate-b' },
      ],
    }
    expect(pendingPreparedCandidateId(run)).toBeNull()
  })

  it('returns null for a run that has never prepared an image', () => {
    expect(pendingPreparedCandidateId({ current_artifact_id: 'front-upload', attempts: [] })).toBeNull()
    expect(pendingPreparedCandidateId(null)).toBeNull()
  })
})

describe('canGenerateFromPreparedImage', () => {
  it('requires an explicitly approved preparation artifact and no active preparation', () => {
    expect(canGenerateFromPreparedImage({ stage: 'prepare-image' }, false)).toBe(true)
    expect(canGenerateFromPreparedImage({ metadata: { stage: 'prepare-image' } }, false)).toBe(true)
    expect(canGenerateFromPreparedImage({ stage: 'prepare-image' }, true)).toBe(false)
    expect(canGenerateFromPreparedImage({ stage: 'upload' }, false)).toBe(false)
    expect(canGenerateFromPreparedImage(null, false)).toBe(false)
  })
})

describe('restoreHistoryArtifacts', () => {
  it('restores an approved preparation output as the Generate source', () => {
    const original = { id: 'mary-source', stage: 'upload' }
    const prepared = { id: 'mary-prepared', stage: 'prepare-image' }

    expect(restoreHistoryArtifacts('prepare-image', false, original, prepared)).toEqual({
      source: prepared,
      output: null,
    })
  })
})

describe('shouldRefreshCapabilities', () => {
  it('polls only from a visible studio tab', () => {
    expect(shouldRefreshCapabilities('visible')).toBe(true)
    expect(shouldRefreshCapabilities('hidden')).toBe(false)
  })
})

describe('autoRigJobPayload', () => {
  it('posts an auto-rig job for the source mesh', () => {
    expect(autoRigJobPayload('unrigged-mary', 'mary-run')).toEqual({
      kind: 'auto-rig',
      source_artifact_id: 'unrigged-mary',
      run_id: 'mary-run',
      settings: {},
    })
  })
})

describe('canAcceptMesh2MotionExport', () => {
  it('rejects exports while the automated launch is busy or queued', () => {
    expect(canAcceptMesh2MotionExport(true, null)).toBe(false)
    expect(canAcceptMesh2MotionExport(false, { kind: 'target-proxy-rig', status: 'queued' })).toBe(false)
    expect(canAcceptMesh2MotionExport(false, { kind: 'target-proxy-rig', status: 'running' })).toBe(false)
    expect(canAcceptMesh2MotionExport(false, { kind: 'generate', status: 'running' })).toBe(false)
  })

  it('accepts an export when the studio is idle', () => {
    expect(canAcceptMesh2MotionExport(false, null)).toBe(true)
    expect(canAcceptMesh2MotionExport(false, { kind: 'target-proxy-rig', status: 'completed' })).toBe(true)
  })

  it('keeps a completed job locked until terminal hydration settles', () => {
    const completed = { kind: 'target-proxy-rig', status: 'completed' }
    expect(canAcceptMesh2MotionExport(false, completed, true)).toBe(false)
    expect(canAcceptMesh2MotionExport(false, completed, false)).toBe(true)
  })
})

describe('canAcceptMesh2MotionContext', () => {
  const expected = { runId: 'mary', artifactId: 'rigged-mary', mode: 'animate' as const }

  it('accepts only the iframe bound to the active character and artifact', () => {
    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'rigged-mary', mode: 'animate' }, expected)).toBe(true)
    expect(canAcceptMesh2MotionContext({ run: 'other', input: 'rigged-mary', mode: 'animate' }, expected)).toBe(false)
    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'old-rig', mode: 'animate' }, expected)).toBe(false)
    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'rigged-mary', mode: 'rig' }, expected)).toBe(false)
  })

  it('accepts a rig export only from the immutable clean-mesh editor', () => {
    const expected = { runId: 'mary', artifactId: 'clean-mary', mode: 'rig' as const }

    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'clean-mary', mode: 'edit-clean-mesh' }, expected)).toBe(true)
    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'old-rig', mode: 'edit-existing-rig' }, expected)).toBe(false)
    expect(canAcceptMesh2MotionContext({ run: 'mary', input: 'clean-mary', mode: 'animate' }, expected)).toBe(false)
  })

  it('records a rig against its clean mesh and animation against its current rig', () => {
    expect(mesh2MotionExportInputArtifactId('old-rig', 'clean-mary', 'rig')).toBe('clean-mary')
    expect(mesh2MotionExportInputArtifactId('rigged-mary', 'clean-mary', 'animate')).toBe('rigged-mary')
  })
})

describe('isArrayBufferPayload', () => {
  it('accepts branded binary payloads without relying on the sender realm', () => {
    expect(isArrayBufferPayload(new ArrayBuffer(8))).toBe(true)
    expect(isArrayBufferPayload({ byteLength: 8 })).toBe(false)
    expect(isArrayBufferPayload(new Uint8Array(8))).toBe(false)
  })
})

describe('retargetMotionJobPayload', () => {
  it('creates an idempotent bake request for the generated motion child', () => {
    expect(retargetMotionJobPayload('rigged-mary', 'attack-bvh', 'mary-run')).toEqual({
      kind: 'retarget-motion',
      submission_id: 'auto-retarget-attack-bvh',
      source_artifact_id: 'rigged-mary',
      run_id: 'mary-run',
      settings: { motion_artifact_id: 'attack-bvh', animation: 'Sword_Attack' },
    })
  })

  it('names the baked output action after whichever curated clip was actually attached', () => {
    expect(retargetMotionJobPayload('rigged-mary', 'walk-bvh', 'mary-run', 'Walk').settings).toEqual({
      motion_artifact_id: 'walk-bvh', animation: 'Walk',
    })
  })
})

describe('applyClipJobPayload', () => {
  it('direct-attaches the exact family clip_name, no BVH round trip', () => {
    expect(applyClipJobPayload('rigged-mary', 'mary-run', 'Sword_Attack_Armature')).toEqual({
      kind: 'apply-clip',
      submission_id: 'auto-apply-clip-Sword_Attack_Armature',
      source_artifact_id: 'rigged-mary',
      run_id: 'mary-run',
      settings: { clip: 'Sword_Attack_Armature', family: 'biped' },
    })
  })

  it('defaults family to biped but accepts an explicit override', () => {
    expect(applyClipJobPayload('rigged-mary', 'mary-run', 'Roll_Armature', 'quadruped').settings).toEqual({
      clip: 'Roll_Armature', family: 'quadruped',
    })
  })
})

describe('isTemplateConformantRig', () => {
  const certified = { status: 'ready', policy_version: 2 }

  it('is true only for a certified rig with a clean (empty) conformance list', () => {
    expect(isTemplateConformantRig({
      quality_decisions: { 'rig-certificate': certified },
      rig_quality: { template_conformance: [] },
    })).toBe(true)
  })

  it('is false when the certificate is not ready', () => {
    expect(isTemplateConformantRig({
      quality_decisions: { 'rig-certificate': { status: 'blocked', policy_version: 2 } },
      rig_quality: { template_conformance: [] },
    })).toBe(false)
  })

  it('is false when there is no conformance data at all (pre-N3 or non-humanoid), even certified', () => {
    expect(isTemplateConformantRig({
      quality_decisions: { 'rig-certificate': certified },
    })).toBe(false)
  })

  it('is false when the conformance list carries findings', () => {
    expect(isTemplateConformantRig({
      quality_decisions: { 'rig-certificate': certified },
      rig_quality: { template_conformance: [{ code: 'reparented_bone' }] },
    })).toBe(false)
  })

  it('is false for missing/undefined metadata', () => {
    expect(isTemplateConformantRig(undefined)).toBe(false)
    expect(isTemplateConformantRig(null)).toBe(false)
    expect(isTemplateConformantRig({})).toBe(false)
  })
})

describe('exportGlbNameHint', () => {
  it('prefers the current run name over the raw artifact filename', () => {
    expect(exportGlbNameHint('Mary', 'unrigged-mary.glb')).toBe('Mary')
  })

  it('falls back to the artifact name with its extension stripped', () => {
    expect(exportGlbNameHint(null, 'unrigged-mary.glb')).toBe('unrigged-mary')
    expect(exportGlbNameHint(undefined, 'unrigged-mary.glb')).toBe('unrigged-mary')
    expect(exportGlbNameHint('', 'unrigged-mary.glb')).toBe('unrigged-mary')
    expect(exportGlbNameHint('   ', 'unrigged-mary.glb')).toBe('unrigged-mary')
  })

  it('never hands the backend an empty hint', () => {
    expect(exportGlbNameHint('   ', '.glb')).toBe('model')
  })
})

describe('rigWorkspaceSurface', () => {
  it('keeps the applied skeleton inspector separate from animation curation', () => {
    expect(rigWorkspaceSurface('rig', true)).toBe('skeleton')
    expect(rigWorkspaceSurface('animate', true)).toBe('animations')
    expect(rigWorkspaceSurface(undefined, true)).toBe('animations')
    expect(rigWorkspaceSurface(undefined, false)).toBe('authoring')
  })
})

describe('isMotionPipelineJob', () => {
  it('groups generation, retargeting, and direct-attach into one visible workflow', () => {
    expect(isMotionPipelineJob('generate-motion')).toBe(true)
    expect(isMotionPipelineJob('retarget-motion')).toBe(true)
    expect(isMotionPipelineJob('apply-clip')).toBe(true)
    expect(isMotionPipelineJob('rig-transfer')).toBe(false)
  })
})

describe('hasActiveStudioJob', () => {
  it('keeps motion controls locked for every queued or running studio job', () => {
    expect(hasActiveStudioJob({ status: 'queued' })).toBe(true)
    expect(hasActiveStudioJob({ status: 'running' })).toBe(true)
    expect(hasActiveStudioJob({ status: 'completed' })).toBe(false)
    expect(hasActiveStudioJob(null)).toBe(false)
  })
})

describe('shouldApplyCompletedArtifact', () => {
  it('rejects a response from a job or workspace context that is no longer active', () => {
    expect(shouldApplyCompletedArtifact('job-1', 4, 'job-2', 4)).toBe(false)
    expect(shouldApplyCompletedArtifact('job-1', 4, 'job-1', 5)).toBe(false)
    expect(shouldApplyCompletedArtifact('job-1', 4, 'job-1', 4)).toBe(true)
  })
})

describe('terminal hydration lifecycle', () => {
  it('keeps failed hydration retryable and marks only an active success as hydrated', () => {
    expect(shouldMarkTerminalHydrationComplete(false, true)).toBe(false)
    expect(shouldMarkTerminalHydrationComplete(true, false)).toBe(false)
    expect(shouldMarkTerminalHydrationComplete(true, true)).toBe(true)
  })

  it('requires an explicit retry after a failed attempt before starting again', () => {
    expect(shouldStartTerminalHydration('job-1', null, null, null)).toBe(true)
    expect(shouldStartTerminalHydration('job-1', null, 'job-1', null)).toBe(false)
    expect(shouldStartTerminalHydration('job-1', null, 'job-1', 'job-1')).toBe(true)
    expect(shouldStartTerminalHydration('job-1', 'job-1', null, 'job-1')).toBe(false)
  })
})

describe('shouldApplyHydratedRun', () => {
  it('rejects stale or mismatched run snapshots before they replace the active run', () => {
    const expected = { id: 'run-1', source_artifact_id: 'source-1', current_artifact_id: 'source-1', updated_at: 10 }
    expect(shouldApplyHydratedRun({ ...expected, current_artifact_id: 'rigged-1', updated_at: 11 }, expected, 'rigged-1', true, 'run-1')).toBe(true)
    expect(shouldApplyHydratedRun({ ...expected, current_artifact_id: 'rigged-1', updated_at: 11 }, expected, 'rigged-1', true, 'job-1')).toBe(false)
    expect(shouldApplyHydratedRun({ ...expected, id: 'run-2', current_artifact_id: 'rigged-1', updated_at: 11 }, expected, 'rigged-1', true, expected.id)).toBe(false)
    expect(shouldApplyHydratedRun({ ...expected, source_artifact_id: 'source-2', current_artifact_id: 'rigged-1', updated_at: 11 }, expected, 'rigged-1', true, expected.id)).toBe(false)
    expect(shouldApplyHydratedRun({ ...expected, current_artifact_id: 'rigged-1', updated_at: 9 }, expected, 'rigged-1', true, expected.id)).toBe(false)
  })

  it('accepts retained output hydration only when the current artifact is preserved', () => {
    const expected = { id: 'run-1', source_artifact_id: 'source-1', current_artifact_id: 'source-1', updated_at: 10 }
    expect(shouldApplyHydratedRun({ ...expected, updated_at: 11 }, expected, 'retained-1', false, expected.id)).toBe(true)
    expect(shouldApplyHydratedRun({ ...expected, current_artifact_id: 'retained-1', updated_at: 11 }, expected, 'retained-1', false, expected.id)).toBe(false)
  })
})

describe('canGenerateMotion', () => {
  it('requires both Kimodo and a ready persisted rig certificate', () => {
    const decisions = { rig: { status: 'ready' }, 'rig-certificate': { status: 'ready', policy_version: 2 }, 'animation-review': { status: 'ready' }, 'deformation-review': { status: 'ready' } }
    const ready = { metadata: { quality_decisions: decisions } }
    const stale = { metadata: { quality_decisions: { ...decisions, 'rig-certificate': { status: 'ready', policy_version: 1 } } } }
    const blocked = { metadata: { quality_decisions: { ...decisions, 'deformation-review': { status: 'blocked' } } } }
    expect(canGenerateMotion(ready, true, true)).toBe(true)
    expect(canGenerateMotion(ready, false, true)).toBe(false)
    expect(canGenerateMotion(ready, true, false)).toBe(false)
    expect(canGenerateMotion(blocked, true, true)).toBe(false)
    expect(canGenerateMotion(stale, true, true)).toBe(false)
    expect(canGenerateMotion({}, true, true)).toBe(false)
  })
})

describe('canRestoreHistoryArtifact', () => {
  it('keeps retained rig outputs inspectable in history but out of the active workflow', () => {
    expect(canRestoreHistoryArtifact('target-proxy-rig', 'rejected', 'current')).toBe(false)
    expect(canRestoreHistoryArtifact('rig', 'current', 'current')).toBe(true)
    expect(canRestoreHistoryArtifact('rig', 'older-good', 'current', true)).toBe(true)
    expect(canRestoreHistoryArtifact('generate', 'candidate', 'current')).toBe(true)
  })

  it('applies the same rig-candidate gate to UniRig auto-rig and rig-adjust outputs', () => {
    expect(canRestoreHistoryArtifact('auto-rig', 'rejected', 'current')).toBe(false)
    expect(canRestoreHistoryArtifact('auto-rig', 'current', 'current')).toBe(true)
    expect(canRestoreHistoryArtifact('rig-adjust', 'rejected', 'current')).toBe(false)
    expect(canRestoreHistoryArtifact('rig-adjust', 'older-good', 'current', true)).toBe(true)
  })
})

describe('shouldPromoteCompletedOutput', () => {
  it('keeps a quality-rejected retarget candidate out of the active viewport', () => {
    expect(shouldPromoteCompletedOutput({ kind: 'retarget-motion', promoted_output: false })).toBe(false)
    expect(shouldPromoteCompletedOutput({ kind: 'target-proxy-rig', promoted_output: false })).toBe(false)
    expect(shouldPromoteCompletedOutput({ kind: 'repair-weights', promoted_output: false })).toBe(false)
  })

  it('keeps ordinary completed outputs compatible when no promotion field is present', () => {
    expect(shouldPromoteCompletedOutput({ kind: 'retarget-motion', promoted_output: true })).toBe(true)
    expect(shouldPromoteCompletedOutput({ kind: 'generate' })).toBe(true)
  })
})

describe('shouldActivateRecordedArtifact', () => {
  it('activates only an artifact the run actually promoted', () => {
    expect(shouldActivateRecordedArtifact('rig-ready', 'rig-ready')).toBe(true)
    expect(shouldActivateRecordedArtifact('previous-mesh', 'rig-rejected')).toBe(false)
  })
})

describe('shouldPreviewRecordedArtifact', () => {
  it('previews the exact saved candidate without requiring backend promotion', () => {
    expect(shouldPreviewRecordedArtifact('saved-walk', 'saved-walk')).toBe(true)
    expect(shouldPreviewRecordedArtifact('saved-walk', 'different-artifact')).toBe(false)
  })
})

describe('isActiveJobStatus', () => {
  it('keeps controls locked only while work is queued or running', () => {
    expect(isActiveJobStatus('queued')).toBe(true)
    expect(isActiveJobStatus('running')).toBe(true)
    expect(isActiveJobStatus('completed')).toBe(false)
    expect(isActiveJobStatus('failed')).toBe(false)
  })
})

describe('retryNetworkPost', () => {
  it('retries one lost network response without changing the prepared request', async () => {
    const payload = { submission_id: 'stable-id' }
    let calls = 0
    const result = await retryNetworkPost(async () => {
      calls += 1
      if (calls === 1) throw new TypeError('connection reset')
      return payload
    })

    expect(calls).toBe(2)
    expect(result).toBe(payload)
  })

  it('does not retry an application error', async () => {
    let calls = 0
    await expect(retryNetworkPost(async () => {
      calls += 1
      throw new Error('422')
    })).rejects.toThrow('422')
    expect(calls).toBe(1)
  })

  it('bounds a hanging retarget request when terminal hydration aborts', async () => {
    const hydrationController = new AbortController()
    let settled = false
    const retarget = retryNetworkPost(
      () => new Promise<never>(() => undefined),
      hydrationController.signal,
    ).then(() => { settled = true }, () => { settled = true })

    hydrationController.abort()
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(settled).toBe(true)
    await retarget
  })
})

describe('subject kind (H)', () => {
  it('shows the T-pose card only for biped or auto subjects', () => {
    expect(showsPrepareCard('biped')).toBe(true)
    expect(showsPrepareCard('auto')).toBe(true)
    expect(showsPrepareCard('object')).toBe(false)
    expect(showsPrepareCard('creature')).toBe(false)
  })

  it('captions the auto-rig action by rig topology', () => {
    expect(autoRigCaption('creature')).toBe('UniRig · chain rig')
    expect(autoRigCaption('biped')).toBe('UniRig · biped rig')
    expect(autoRigCaption('auto')).toBe('UniRig · biped rig')
    expect(autoRigCaption('object')).toBe('UniRig · biped rig')
  })
})

describe('runAssetOptions (I)', () => {
  const run = {
    source_artifact_id: 'source-image',
    source_artifact_kind: 'image',
    attempts: [
      { stage: 'prepare-image', status: 'completed', output_artifact_id: 'prepared-1', completed_at: 0.5 },
      { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1', completed_at: 1 },
      { stage: 'retopo', status: 'completed', output_artifact_id: 'clean-1', completed_at: 2 },
      { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1', completed_at: 3 },
      { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'anim-1', completed_at: 4 },
      { stage: 'generate', status: 'failed', output_artifact_id: null, completed_at: 5 },
    ],
  }

  it('offers images for generate, newest first, including a compatible source', () => {
    expect(runAssetOptions(run, 'generate')).toEqual([
      { artifactId: 'prepared-1', stage: 'prepare-image', completedAt: 0.5, isSource: false },
      { artifactId: 'source-image', stage: 'source', completedAt: -Infinity, isSource: true },
    ])
  })

  it('offers meshes for clean, skeleton-pre-rig, and paint', () => {
    expect(runAssetOptions(run, 'clean')).toEqual([
      { artifactId: 'mesh-1', stage: 'generate', completedAt: 1, isSource: false },
    ])
    expect(runAssetOptions(run, 'skeleton-pre-rig').map((option) => option.artifactId)).toEqual(['clean-1', 'mesh-1'])
    expect(runAssetOptions(run, 'paint').map((option) => option.artifactId)).toEqual(['clean-1', 'mesh-1'])
  })

  it('offers a canonicalize output to the mesh-consuming pickers (r029o)', () => {
    const canonical = {
      ...run,
      attempts: [...run.attempts, { stage: 'canonicalize', status: 'completed', output_artifact_id: 'canon-1', completed_at: 6 }],
    }
    // Clean's own picker still lists what Clean consumes (Generate output),
    // never its own canonical result.
    expect(runAssetOptions(canonical, 'clean').map((option) => option.artifactId)).toEqual(['mesh-1'])
    expect(runAssetOptions(canonical, 'skeleton-pre-rig').map((option) => option.artifactId)).toEqual(['canon-1', 'clean-1', 'mesh-1'])
    expect(runAssetOptions(canonical, 'paint').map((option) => option.artifactId)).toEqual(['canon-1', 'clean-1', 'mesh-1'])
  })

  it('offers rig candidates for skeleton and animate (which rig is being animated — the Animations list is the separate picker for which clip)', () => {
    expect(runAssetOptions(run, 'skeleton').map((option) => option.artifactId)).toEqual(['rig-1'])
    expect(runAssetOptions(run, 'animate').map((option) => option.artifactId)).toEqual(['rig-1'])
  })

  it('includes a GLB source only for a rig-only run', () => {
    const rigOnly = { source_artifact_id: 'source-glb', source_artifact_kind: 'glb', attempts: [] }
    expect(runAssetOptions(rigOnly, 'clean')).toEqual([
      { artifactId: 'source-glb', stage: 'source', completedAt: -Infinity, isSource: true },
    ])
    expect(runAssetOptions(rigOnly, 'skeleton')).toEqual([])
  })

  it('returns nothing for a null run', () => {
    expect(runAssetOptions(null, 'generate')).toEqual([])
  })
})

describe('clean hints (I3)', () => {
  it('flags a target vertex count above the source', () => {
    expect(cleanTargetExceedsSource(30000, 25000)).toBe(true)
    expect(cleanTargetExceedsSource(20000, 25000)).toBe(false)
    expect(cleanTargetExceedsSource(30000, null)).toBe(false)
  })

  it('flags re-cleaning an artifact that is itself a Clean output', () => {
    expect(isReCleaningCleanOutput('retopo')).toBe(true)
    expect(isReCleaningCleanOutput('retopologize')).toBe(true)
    expect(isReCleaningCleanOutput('paint-bake')).toBe(true)
    expect(isReCleaningCleanOutput('canonicalize')).toBe(true)
    expect(isReCleaningCleanOutput('project-views')).toBe(true)
    expect(isReCleaningCleanOutput('generate')).toBe(false)
    expect(isReCleaningCleanOutput(undefined)).toBe(false)
  })
})

describe('cleanMethodForStage (r029o)', () => {
  it('opens the S2 canonical path for anything that is not a legacy repair output', () => {
    expect(cleanMethodForStage('generate')).toBe('canonicalize')
    expect(cleanMethodForStage('canonicalize')).toBe('canonicalize')
    expect(cleanMethodForStage('source')).toBe('canonicalize')
    expect(cleanMethodForStage(undefined)).toBe('canonicalize')
  })

  it('reopens a legacy repair output on the method that produced it', () => {
    expect(cleanMethodForStage('retopologize')).toBe('ai-refine')
    expect(cleanMethodForStage('retopo')).toBe('instant-meshes')
    expect(cleanMethodForStage('rebuild-views')).toBe('rebuild-views')
  })
})

describe('canonicalizeCandidatePinnable (r029o)', () => {
  const blocked = { status: 'blocked' as const, findings: [], recommended_actions: [] }

  it('offers the pin for a retained candidate the gate refused to promote', () => {
    expect(canonicalizeCandidatePinnable(blocked, false)).toBe(true)
  })

  it('does not offer a pin for a candidate the run already promoted', () => {
    expect(canonicalizeCandidatePinnable({ status: 'ready', findings: [], recommended_actions: [] }, true)).toBe(false)
  })

  it('offers nothing when the stage never ran', () => {
    expect(canonicalizeCandidatePinnable(null, false)).toBe(false)
  })
})

describe('cleanRecommendation', () => {
  it('recommends skipping Clean for a healthy single-component TRELLIS.2 mesh', () => {
    const artifact = { metadata: { backend: 'trellis2', mesh_quality: { components: 1, faces: 120000, non_manifold_edges: 20, boundary_edges: 0 } } }
    expect(cleanRecommendation(artifact)).toMatchObject({ choice: 'skip' })
  })

  it('recommends AI Refine for damaged or fragmented geometry', () => {
    const nonManifold = { metadata: { backend: 'trellis2', mesh_quality: { components: 1, faces: 100000, non_manifold_edges: 900, boundary_edges: 0 } } }
    const openBoundary = { metadata: { backend: 'trellis2', mesh_quality: { components: 1, faces: 100000, non_manifold_edges: 10, boundary_edges: 5 } } }
    const fragmented = { metadata: { backend: 'trellis2', mesh_quality: { components: 3, faces: 100000, non_manifold_edges: 10, boundary_edges: 0 } } }
    expect(cleanRecommendation(nonManifold)).toMatchObject({ choice: 'ai-refine' })
    expect(cleanRecommendation(openBoundary)).toMatchObject({ choice: 'ai-refine' })
    expect(cleanRecommendation(fragmented)).toMatchObject({ choice: 'ai-refine' })
  })

  it('recommends Instant Meshes for an imported or dense scan', () => {
    const imported = { metadata: { backend: 'photogrammetry-scan', mesh_quality: { components: 1, faces: 400000, non_manifold_edges: 0, boundary_edges: 0 } } }
    expect(cleanRecommendation(imported)).toMatchObject({ choice: 'instant-meshes' })
  })

  it('returns nothing without mesh quality metadata', () => {
    expect(cleanRecommendation(null)).toBeNull()
    expect(cleanRecommendation({ metadata: {} })).toBeNull()
  })
})

describe('pollJobUntilTerminal', () => {
  it('keeps polling the accepted job through a transient network failure', async () => {
    let reads = 0
    const terminal = await pollJobUntilTerminal(
      { id: 'job-1', status: 'running' },
      async () => {
        reads += 1
        if (reads === 1) throw new TypeError('connection reset')
        return { id: 'job-1', status: 'completed' }
      },
      async () => undefined,
    )

    expect(reads).toBe(2)
    expect(terminal.status).toBe('completed')
  })

  it('stops polling when the owning view is gone', async () => {
    let active = true
    let reads = 0
    const last = await pollJobUntilTerminal(
      { id: 'job-1', status: 'running' },
      async () => { reads += 1; throw new TypeError('offline') },
      async () => { active = false },
      () => active,
    )

    expect(reads).toBe(0)
    expect(last.status).toBe('running')
  })
})

describe('active job resync after the tab comes back', () => {
  const running = { id: 'job-1', status: 'running', run_id: 'mary' }
  const finished = { id: 'job-1', status: 'completed', run_id: 'mary' }

  it('remembers the running job under its own storage key and reads it back', () => {
    expect(ACTIVE_JOB_STORAGE_KEY).toBe('aismith-active-job-id')
    const saved = serializeRememberedJob('job-1', 'mary')

    expect(readRememberedJob(saved)).toEqual({ jobId: 'job-1', runId: 'mary' })
    expect(readRememberedJob(serializeRememberedJob('job-1', null))).toEqual({ jobId: 'job-1', runId: null })
  })

  it('treats missing, malformed, or id-less storage as nothing remembered', () => {
    expect(readRememberedJob(null)).toBeNull()
    expect(readRememberedJob('not json at all')).toBeNull()
    expect(readRememberedJob('{"runId":"mary"}')).toBeNull()
    expect(readRememberedJob('{"jobId":"","runId":"mary"}')).toBeNull()
  })

  it('asks about the in-memory job first and falls back to the persisted one', () => {
    expect(activeJobResyncId({ id: 'live' }, { jobId: 'stored', runId: 'mary' })).toBe('live')
    expect(activeJobResyncId(null, { jobId: 'stored', runId: 'mary' })).toBe('stored')
    expect(activeJobResyncId(null, null)).toBeNull()
  })

  it('restores a job that never stopped running, so the progress panel comes back', () => {
    // The reported bug: the tab was away, the job kept running server-side,
    // and the remounted UI held no job at all.
    expect(activeJobResyncDecision(running, null, 'mary')).toBe('restore')
    expect(activeJobResyncDecision(running, { id: 'job-1', status: 'running' }, 'mary')).toBe('restore')
  })

  it('reconciles a job that finished while the tab was away', () => {
    expect(activeJobResyncDecision(finished, null, 'mary')).toBe('reconcile')
    expect(activeJobResyncDecision(finished, { id: 'job-1', status: 'running' }, 'mary')).toBe('reconcile')
    expect(activeJobResyncDecision({ id: 'job-1', status: 'failed', run_id: 'mary' }, null, 'mary')).toBe('reconcile')
  })

  it('never clobbers newer local state or another journey', () => {
    expect(activeJobResyncDecision(null, { id: 'job-2', status: 'running' }, 'mary')).toBe('ignore')
    // This tab already moved on to a different job.
    expect(activeJobResyncDecision(finished, { id: 'job-2', status: 'running' }, 'mary')).toBe('ignore')
    // The job belongs to a character journey that is no longer open.
    expect(activeJobResyncDecision(finished, null, 'tess')).toBe('ignore')
    // Its terminal state is already on screen — nothing left to reconcile.
    expect(activeJobResyncDecision(finished, { id: 'job-1', status: 'completed' }, 'mary')).toBe('ignore')
  })

  it('adopts a run-less job whichever journey is open', () => {
    expect(activeJobResyncDecision({ id: 'job-1', status: 'running' }, null, 'mary')).toBe('restore')
  })

  it('resyncs on focus and mount unconditionally, on visibilitychange only when visible', () => {
    // Remote/automated Chrome sessions report visibilityState 'hidden' even
    // for the focused window, so focus must not be gated on it.
    expect(shouldResyncActiveJob('focus', 'hidden')).toBe(true)
    expect(shouldResyncActiveJob('mount', 'hidden')).toBe(true)
    expect(shouldResyncActiveJob('visibilitychange', 'visible')).toBe(true)
    expect(shouldResyncActiveJob('visibilitychange', 'hidden')).toBe(false)
  })

  it('keeps polling while hidden, just slower — it never stops', () => {
    expect(jobPollIntervalMs('visible')).toBe(500)
    expect(jobPollIntervalMs('hidden')).toBe(2000)
    expect(jobPollIntervalMs('hidden')).toBeGreaterThan(0)
  })
})

// Bead 8c9gf: the view grid reads the backend's per-view verdict off the
// artifact in the slot, so what the tile says is always about the image the
// operator is actually looking at.
describe('viewConsistencyNotice', () => {
  const result = (status: string, codes: string[] = []) => ({
    view_consistency: {
      status,
      findings: codes.map((code) => ({ code, severity: status === 'blocked' ? 'error' : 'warning', message: 'x' })),
    },
  })

  it('says nothing about a clean view', () => {
    expect(viewConsistencyNotice('left', result('clean'))).toBeNull()
  })

  it('says nothing when the artifact carries no check at all', () => {
    expect(viewConsistencyNotice('left', {})).toBeNull()
    expect(viewConsistencyNotice('left', undefined)).toBeNull()
    expect(viewConsistencyNotice('left', { view_consistency: 'not-an-object' })).toBeNull()
  })

  it('names the view and the remedy for a leg-separation mismatch', () => {
    const notice = viewConsistencyNotice('left', result('blocked', ['view_consistency.leg_separation_mismatch']))

    expect(notice).toEqual({
      tone: 'error',
      headline: 'LEFT contradicts FRONT',
      detail: 'legs not visible — regenerate this view',
    })
  })

  it('softens the wording for a decorrelation warning', () => {
    const notice = viewConsistencyNotice('back', result('warning', ['view_consistency.lower_silhouette_decorrelated']))

    expect(notice?.tone).toBe('warning')
    expect(notice?.headline).toBe('BACK differs from FRONT')
    // Advisory, not an accusation: it never tells the operator to redo the view.
    expect(notice?.detail).not.toContain('regenerate')
  })

  it('marks a refusal to score as neutral, never as a defect', () => {
    const notice = viewConsistencyNotice('right', {
      view_consistency: {
        status: 'not_evaluable',
        findings: [{ code: 'view_consistency.not_evaluable', severity: 'not-evaluable', message: 'x' }],
      },
    })

    expect(notice).toEqual({
      tone: 'unknown',
      headline: "RIGHT: couldn't verify",
      detail: 'This view could not be measured against the front reference.',
    })
  })

  it('an error outranks a warning when a view trips both', () => {
    const notice = viewConsistencyNotice('left', result('blocked', [
      'view_consistency.lower_silhouette_decorrelated',
      'view_consistency.leg_separation_mismatch',
    ]))

    expect(notice?.tone).toBe('error')
  })

  it('still flags an unrecognised finding code rather than staying silent', () => {
    const notice = viewConsistencyNotice('left', result('blocked', ['view_consistency.something_new']))

    expect(notice?.tone).toBe('error')
    expect(notice?.headline).toBe('LEFT differs from FRONT')
  })

  it('clears once the slot holds a regenerated, clean view', () => {
    const dirty = viewConsistencyNotice('left', result('blocked', ['view_consistency.leg_separation_mismatch']))
    const regenerated = viewConsistencyNotice('left', result('clean'))

    expect(dirty).not.toBeNull()
    expect(regenerated).toBeNull()
  })
})
