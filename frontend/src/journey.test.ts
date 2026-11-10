import { describe, expect, it } from 'vitest'
import { characterJourney, journeyDestination, journeyWorkspace, nextJourneyStep } from './journey'

describe('characterJourney', () => {
  it('starts with image preparation and explains the first human checkpoint', () => {
    const journey = characterJourney(null, 'generate')

    expect(journey.map((step) => step.label)).toEqual([
      'Prepare', 'Generate', 'Clean', 'Skeleton', 'Animate',
    ])
    expect(journey[0]).toMatchObject({ state: 'active', owner: 'shared' })
    expect(journey[0].humanAction).toContain('full body')
  })

  it('marks completed durable stages and stops at skeleton approval after rigging', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
        { stage: 'retopo', status: 'completed' },
        { stage: 'target-proxy-rig', status: 'completed' },
      ],
    }, 'rig')

    expect(journey.slice(0, 3).map((step) => step.state)).toEqual([
      'complete', 'complete', 'complete',
    ])
    expect(journey[3]).toMatchObject({ state: 'checkpoint', owner: 'shared' })
    expect(journey[4].state).toBe('upcoming')
  })

  it('does not claim Animate is complete until the animated run is accepted', () => {
    // Animate absorbed the former Review step's accept/reject checkpoint —
    // no separate Review tab (operator's ask).
    const base = {
      checkpoints: { skeleton: { note: 'Approved.', completed_at: 40, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'animated-1' },
        { stage: 'deformation-review', status: 'completed' },
      ],
    }

    expect(characterJourney({ ...base, review_status: 'unreviewed', current_artifact_id: 'animated-1' }, 'rig')[4].state).toBe('checkpoint')
    expect(characterJourney({ ...base, review_status: 'accepted', current_artifact_id: 'animated-1', reviewed_artifact_id: 'animated-1' }, 'rig')[4].state).toBe('complete')
  })

  it('counts a UniRig rig-adjust attempt as the Skeleton stage and unlocks Animate once approved', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-adjusted-1',
      checkpoints: { skeleton: { note: 'Adjusted joints approved.', completed_at: 42, artifact_id: 'rig-adjusted-1' } },
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
        { stage: 'retopo', status: 'completed' },
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'rig-adjust', status: 'completed', output_artifact_id: 'rig-adjusted-1' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('complete')
    expect(journey[4].state).toBe('active')
  })

  it('advances from skeleton approval after a durable handoff', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-1',
      checkpoints: { skeleton: { note: 'Landmarks approved.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
        { stage: 'retopo', status: 'completed' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('complete')
    expect(journey[4].state).toBe('active')
  })

  it('unlocks Animate for an approved rig the run pointer never advanced onto', () => {
    // A fresh auto-rig whose certificate lands `needs_review` is retained
    // rather than promoted, so current_artifact_id still names the mesh it
    // was built from. The approval binds to the rig itself, and the strip
    // must read the approval against that same rig -- the completeness of
    // Skeleton is an identity question, not a pointer question.
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'mesh-1',
      checkpoints: { skeleton: { note: 'Skeleton reviewed.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1' },
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('complete')
    expect(journey[4].state).toBe('active')
  })

  it('keeps Animate unlocked once approval pins the approved rig as current', () => {
    // The state the backend leaves behind after hand_back pins the rig:
    // pointer and checkpoint name the same artifact.
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-1',
      checkpoints: { skeleton: { note: 'Skeleton reviewed.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1' },
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'promote-artifact', status: 'completed', output_artifact_id: 'rig-1' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('complete')
    expect(journey[4].state).toBe('active')
  })

  it('revokes skeleton approval when a replacement rig is produced', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-2',
      checkpoints: { skeleton: { note: 'Approved old rig.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-2' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('checkpoint')
  })

  it('does not treat acceptance as complete when the reviewed artifact was never actually produced by this run', () => {
    // Acceptance alone is not enough — reviewed_artifact_id must name one of
    // Animate's own completed outputs, guarding against a stale or foreign id.
    const journey = characterJourney({
      review_status: 'accepted',
      reviewed_artifact_id: 'animated-1',
      current_artifact_id: 'animated-2',
      attempts: [
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'animated-2' },
        { stage: 'deformation-review', status: 'completed' },
      ],
    }, 'rig')

    expect(journey[4].state).toBe('checkpoint')
  })

  it('does not reopen a rejected retained animation instead of the restored current rig', () => {
    const workspace = journeyWorkspace('animate', 'rig', {
      source_artifact_id: 'source-1',
      current_artifact_id: 'rig-1',
      checkpoints: { skeleton: { artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'rejected-animation' },
        { stage: 'promote-artifact', status: 'completed', output_artifact_id: 'rig-1' },
      ],
    })

    expect(workspace.artifactId).toBe('rig-1')
    expect(workspace.stage).toBe('target-proxy-rig')
  })

  it('surfaces a failed stage instead of advancing the lifecycle', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'generate', status: 'failed' },
      ],
    }, 'generate')

    expect(journey[1].state).toBe('failed')
    expect(journey[2].state).toBe('upcoming')
    expect(journey[2].blockedReason).toBe('Generate needs your attention first.')
  })

  it('explains why Clean is unreachable instead of just disabling it, when Generate has not completed yet', () => {
    // The operator's second report: a run that has synthesized views but
    // never actually run Generate (the "zix" run) shows Clean as an
    // unclickable "upcoming" step with no explanation. This is a legitimate
    // gate -- Clean has nothing to clean yet -- so the strip must say why
    // instead of dead-clicking.
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'synthesize-views', status: 'completed' },
      ],
    }, 'generate')

    expect(journey[0].state).toBe('complete')
    expect(journey[1].state).toBe('active')
    expect(journey[2].state).toBe('upcoming')
    expect(journey[2].blockedReason).toBe('Complete Generate first.')
  })

  it('explains an upcoming step blocked by another stage actively running', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [{ stage: 'generate', status: 'running' }],
    }, 'generate')

    expect(journey[0].state).toBe('upcoming')
    expect(journey[0].blockedReason).toBe('Generate is still running.')
  })

  it('uses a successful retry instead of an older failed attempt', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'retopo', status: 'failed' },
        { stage: 'retopo', status: 'completed' },
      ],
    }, 'retopologize')

    expect(journey[2].state).toBe('complete')
  })

  it('marks Clean skipped, not complete, once a downstream durable stage succeeds without it', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'retopo', status: 'failed' },
        { stage: 'target-proxy-rig', status: 'completed' },
      ],
    }, 'rig')

    expect(journey[2].state).toBe('skipped')
    expect(journey[3].state).toBe('checkpoint')
  })

  it('selects the stage that is actually running even when optional prerequisites are absent', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [{ stage: 'generate', status: 'running' }],
    }, 'generate')

    expect(journey[0].state).toBe('upcoming')
    expect(journey[1].state).toBe('active')
  })

  it('routes clean work to its actual Clean or paint workspace', () => {
    expect(journeyDestination('clean', 'retopologize', [])).toBe('retopologize')
    expect(journeyDestination('clean', 'retopologize', [{ stage: 'retopo' }])).toBe('retopologize')
    expect(journeyDestination('clean', 'retopologize', [{ stage: 'retopologize' }])).toBe('retopologize')
    expect(journeyDestination('clean', 'paint', [{ stage: 'paint-bake' }])).toBe('paint')
  })

  it('opens each stage with its latest compatible artifact', () => {
    const run = {
      source_artifact_id: 'source-image',
      checkpoints: { skeleton: { artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'prepared-1' },
        { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1' },
        { stage: 'retopo', status: 'completed', output_artifact_id: 'retopo-1' },
        { stage: 'target-proxy-rig', status: 'completed', input_artifact_id: 'retopo-1', output_artifact_id: 'rig-1' },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'animated-1' },
        { stage: 'deformation-review', status: 'completed', output_artifact_id: 'deformation-report-json' },
      ],
    }

    expect(journeyWorkspace('prepare', 'rig', run)).toMatchObject({ destination: 'generate', artifactId: 'prepared-1', stage: 'prepare-image' })
    expect(journeyWorkspace('clean', 'rig', run)).toMatchObject({ destination: 'retopologize', artifactId: 'retopo-1', stage: 'retopo' })
    expect(journeyWorkspace('skeleton', 'rig', run)).toMatchObject({ destination: 'rig', artifactId: 'rig-1', stage: 'target-proxy-rig', mode: 'rig' })
    expect(journeyWorkspace('rig', 'rig', run)).toMatchObject({ destination: 'rig', artifactId: 'rig-1', stage: 'target-proxy-rig', mode: 'rig' })
    expect(journeyWorkspace('animate', 'rig', run)).toMatchObject({ destination: 'rig', artifactId: 'animated-1', stage: 'retarget-motion', mode: 'animate' })
  })

  it('treats a chain rig\'s generate-motion output as the playable animate clip only when told to via animatedOutputStages', () => {
    // A chain (serpent) rig bakes motion straight onto the GLB with no
    // separate retarget step, so its generate-motion output is itself a
    // playable clip — but only once the caller (App.tsx, from the run's
    // subject_kind) says so; the default keeps excluding it (a humanoid
    // rig's generate-motion output is only an intermediate BVH).
    const run = {
      source_artifact_id: 'source-glb',
      checkpoints: { skeleton: { artifact_id: 'chain-rig' } },
      attempts: [
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'chain-rig' },
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1' },
      ],
    }

    expect(journeyWorkspace('animate', 'rig', run)).toMatchObject({
      destination: 'rig', artifactId: 'chain-rig', stage: 'auto-rig', mode: 'animate',
    })
    expect(journeyWorkspace('animate', 'rig', run, { animatedOutputStages: ['retarget-motion', 'generate-motion'] })).toMatchObject({
      destination: 'rig', artifactId: 'slither-1', stage: 'generate-motion', mode: 'animate',
    })
  })

  it('pinning a generate-motion clip on animate routes to that clip even when animatedOutputStages excludes it', () => {
    // animatedOutputStages only narrows the DEFAULT (nothing-pinned)
    // resolution — whatever the Animations list actually shows must always
    // be selectable by an explicit pin, regardless of subject_kind/profile.
    const run = {
      source_artifact_id: 'source-glb',
      checkpoints: { skeleton: { artifact_id: 'chain-rig' } },
      attempts: [
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'chain-rig' },
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1' },
      ],
    }

    // No animatedOutputStages passed at all (e.g. a run whose subject_kind
    // wasn't detected as 'creature') — the default resolution would land on
    // the bare rig, but an explicit pin on the clip still wins outright.
    expect(journeyWorkspace('animate', 'rig', run, { pinnedArtifactId: 'slither-1' })).toMatchObject({
      destination: 'rig', artifactId: 'slither-1', stage: 'generate-motion', mode: 'animate',
    })
  })

  it('uses the latest playable rig for animation when no separate motion bake exists', () => {
    const run = {
      source_artifact_id: 'source-image',
      checkpoints: { skeleton: { artifact_id: 'playable-rig' } },
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'playable-rig' },
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'motion-only-bvh' },
      ],
    }

    expect(journeyWorkspace('animate', 'generate', run)).toMatchObject({
      destination: 'rig', artifactId: 'playable-rig', stage: 'target-proxy-rig', mode: 'animate',
    })
  })

  it('approves and then animates the exact retained rig the Skeleton card displays', () => {
    // The Skeleton card promises approval of "this exact artifact". With the
    // rig retained (certificate needs_review) the run still points at the
    // mesh, so the card's artifact and the run's pointer differ -- and both
    // the approval target and the Animate gate must follow the artifact.
    const retained = {
      source_artifact_id: 'source-image',
      current_artifact_id: 'mesh-1',
      attempts: [
        { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1' },
        { stage: 'auto-rig', status: 'completed', input_artifact_id: 'mesh-1', output_artifact_id: 'rig-1' },
      ],
    }

    // before approval: the card shows rig-1 (the id the approval posts), and
    // Animate is still gated back to the rig
    expect(journeyWorkspace('skeleton', 'rig', retained)).toMatchObject({
      destination: 'rig', artifactId: 'rig-1', stage: 'auto-rig', mode: 'rig',
    })
    expect(journeyWorkspace('animate', 'rig', retained)).toMatchObject({ blocked: true, artifactId: 'rig-1' })

    // after approval, whether or not the pointer moved with it
    const approved = { ...retained, checkpoints: { skeleton: { artifact_id: 'rig-1' } } }
    expect(journeyWorkspace('animate', 'rig', approved).blocked).toBeUndefined()
    const pinned = { ...approved, current_artifact_id: 'rig-1' }
    expect(journeyWorkspace('animate', 'rig', pinned)).toMatchObject({ artifactId: 'rig-1', mode: 'animate' })
    expect(journeyWorkspace('animate', 'rig', pinned).blocked).toBeUndefined()
  })

  it('returns the newest retained rig candidate to Skeleton until it is approved', () => {
    const run = {
      source_artifact_id: 'source-image',
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'promoted-rig' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-candidate' },
      ],
    }

    expect(journeyWorkspace('animate', 'rig', run)).toEqual({
      destination: 'rig', artifactId: 'rig-candidate', stage: 'target-proxy-rig',
      isSource: false, mode: 'rig', blocked: true,
    })
  })
})

describe('nextJourneyStep', () => {
  it('offers the step after Generate even before the user has ever visited it, and Skeleton is already reachable', () => {
    // Clean has no attempts yet and its tab isn't open, so it reads "ready" (not the
    // dead-looking "upcoming") — with the tool tabs gone, the strip button and the
    // Next affordance are the only way to reach a stage before its tab is visited.
    // Clean is optional (H): Skeleton is ready the moment Generate completes,
    // whether or not Clean ever ran — Next still offers Clean first, but the
    // Skeleton step in the strip is clickable too.
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
      ],
    }, 'generate')

    expect(journey[2].state).toBe('ready')
    expect(journey[3].state).toBe('ready')
    expect(nextJourneyStep(journey)).toMatchObject({ id: 'clean', label: 'Clean' })
  })

  it('points back at a regressed step instead of the one after it', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-2',
      checkpoints: { skeleton: { note: 'Approved old rig.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
        { stage: 'retopo', status: 'completed' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-2' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('checkpoint')
    expect(nextJourneyStep(journey)).toMatchObject({ id: 'skeleton', label: 'Skeleton' })
  })

  it('offers nothing once the final step is complete', () => {
    const journey = characterJourney({
      review_status: 'accepted',
      reviewed_artifact_id: 'animated-1',
      current_artifact_id: 'animated-1',
      checkpoints: { skeleton: { note: 'Approved.', completed_at: 40, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'animated-1' },
      ],
    }, 'rig')

    expect(journey[4].state).toBe('complete')
    expect(nextJourneyStep(journey)).toBeNull()
  })

  it('never offers the last step as "Next" — reaching it is the destination, not a waypoint toward something else', () => {
    // Skeleton just got approved and Animate is now reachable (checkpoint/
    // active), but Animate is also the LAST step in the whole journey — the
    // strip already shows it as clickable on its own; the "Next: Animate"
    // button must not appear (previously it did, since Animate used to have
    // Review after it).
    const journey = characterJourney({
      review_status: 'unreviewed',
      current_artifact_id: 'rig-1',
      checkpoints: { skeleton: { note: 'Approved.', completed_at: 42, artifact_id: 'rig-1' } },
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
        { stage: 'retopo', status: 'completed' },
        { stage: 'target-proxy-rig', status: 'completed', output_artifact_id: 'rig-1' },
      ],
    }, 'rig')

    expect(journey[3].state).toBe('complete')
    expect(['ready', 'checkpoint', 'active']).toContain(journey[4].state)
    expect(nextJourneyStep(journey)).toBeNull()
  })

  it('marks a rig-only run\'s Skeleton step ready before the Rig tab is ever opened', () => {
    const journey = characterJourney({
      review_status: 'unreviewed',
      source_artifact_kind: 'glb',
      attempts: [],
    }, 'generate')

    expect(journey[0].state).toBe('complete')
    expect(journey[1].state).toBe('complete')
    expect(journey[2].state).toBe('complete')
    expect(journey[3].state).toBe('ready')
    expect(nextJourneyStep(journey)).toMatchObject({ id: 'skeleton', label: 'Skeleton' })
  })

  it('never offers a step it cannot actually reach yet', () => {
    // Nothing has run at all: Generate itself is the only reachable step, so
    // there is nothing "after" it to point Next at.
    const journey = characterJourney(null, 'generate')

    expect(journey[1].state).toBe('upcoming')
    expect(nextJourneyStep(journey)).toBeNull()
  })
})

describe('journeyWorkspace for a GLB-sourced (rig-only) run', () => {
  it('resolves the Animate step to the retarget-motion output when no separate rig attempt exists', () => {
    const run = {
      source_artifact_id: 'source-glb',
      current_artifact_id: 'e8d893f0',
      attempts: [
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'e8d893f0' },
      ],
    }

    expect(journeyWorkspace('animate', 'rig', run)).toMatchObject({
      destination: 'rig', artifactId: 'e8d893f0', stage: 'retarget-motion', mode: 'animate',
    })
  })
})

describe('a direct-generate run whose Prepare step never got its own attempt', () => {
  it('paints Prepare complete via the shortcut and, crucially, still counts it satisfied for readiness', () => {
    // The operator's actual blocking case: image-sourced, no prepare-image attempt
    // at all (skipped straight to Generate), so Prepare is only "complete" via
    // index < furthestCompletedIndex — readiness must honor that same shortcut
    // or Clean never leaves "upcoming".
    const journey = characterJourney({
      review_status: 'unreviewed',
      attempts: [
        { stage: 'generate', status: 'completed' },
      ],
    }, 'generate')

    expect(journey[0].state).toBe('complete')
    expect(journey[1].state).toBe('complete')
    expect(journey[2].state).toBe('ready')
    expect(nextJourneyStep(journey)).toMatchObject({ id: 'clean', label: 'Clean' })
  })
})

describe('journeyWorkspace for the Clean step before Clean has ever run', () => {
  it('opens Clean with the Generate output as source, not the run\'s original upload', () => {
    const run = {
      source_artifact_id: 'uploaded-photo',
      attempts: [
        { stage: 'prepare-image', status: 'completed', output_artifact_id: 'prepared-photo' },
        { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1' },
      ],
    }

    expect(journeyWorkspace('clean', 'generate', run)).toMatchObject({
      destination: 'retopologize', artifactId: 'mesh-1', stage: 'generate', isSource: false,
    })
  })
})

it('skeleton with no rig candidate opens on the cleaned mesh, else the generated mesh, never the source image', () => {
  const run = {
    source_artifact_id: 'img',
    source_artifact_kind: 'image',
    current_artifact_id: 'gen',
    review_status: 'unreviewed' as const,
    checkpoints: {},
    attempts: [
      { stage: 'generate', status: 'completed', output_artifact_id: 'gen', created_at: 1 },
    ],
  }
  expect(journeyWorkspace('skeleton', 'generate', run)).toMatchObject({ destination: 'rig', artifactId: 'gen', isSource: false, mode: 'rig' })
  const cleaned = { ...run, attempts: [...run.attempts, { stage: 'retopo', status: 'completed', output_artifact_id: 'clean', created_at: 2 }] }
  expect(journeyWorkspace('skeleton', 'generate', cleaned)).toMatchObject({ destination: 'rig', artifactId: 'clean', isSource: false, mode: 'rig' })
})

describe('apply-clip animations are selectable and exportable', () => {
  // N5 direct-attach outputs are animated GLBs exactly like a retarget-motion
  // bake. journey.ts never learned the stage, so clicking one in the Animate
  // tab's clip list set the pin and then resolved straight back to the rig —
  // which is why the "Selected clip" panel (the only Export/Download
  // affordance for an animation) never appeared.
  const run = {
    source_artifact_id: 'source-image',
    current_artifact_id: 'rig-1',
    checkpoints: { skeleton: { artifact_id: 'rig-1', note: 'Skeleton approved.', completed_at: 2 } },
    review_status: 'unreviewed' as const,
    attempts: [
      { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1', completed_at: 1 },
      { stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1', completed_at: 2 },
      { stage: 'apply-clip', status: 'completed', output_artifact_id: 'clip-1', completed_at: 3 },
      { stage: 'apply-clip', status: 'completed', output_artifact_id: 'clip-2', completed_at: 4 },
    ],
  }

  it('honours a pin on a direct-attach clip instead of falling back to the rig', () => {
    expect(journeyWorkspace('animate', 'rig', run, { pinnedArtifactId: 'clip-1' })).toEqual({
      destination: 'rig', artifactId: 'clip-1', stage: 'apply-clip', isSource: false, mode: 'animate',
    })
  })

  it('opens Animate on the newest direct-attach clip when nothing is pinned', () => {
    const unpinned = { ...run, current_artifact_id: null }

    expect(journeyWorkspace('animate', 'rig', unpinned)).toMatchObject({
      artifactId: 'clip-2', stage: 'apply-clip', mode: 'animate',
    })
  })

  it('counts a direct-attach clip as an animated output in the journey strip', () => {
    const steps = characterJourney(run, 'rig')

    expect(steps[4].state).toBe('checkpoint')
    expect(characterJourney({ ...run, review_status: 'accepted', reviewed_artifact_id: 'clip-2' }, 'rig')[4].state).toBe('complete')
  })
})
