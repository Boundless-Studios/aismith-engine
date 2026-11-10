import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  curatedActionLabel,
  curatedClipId,
  curatedDonorActionName,
  curatedMotionClipsFromResponse,
  fetchCuratedMotionClips,
  generateMotionFallbackLabel,
  journeyAnimations,
} from './animations'

describe('journeyAnimations', () => {
  it('returns nothing for a run that has not produced any animated clip yet', () => {
    expect(journeyAnimations(null)).toEqual([])
    expect(journeyAnimations({ attempts: [{ stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1' }] })).toEqual([])
  })

  it('lists only completed retarget-motion / generate-motion attempts with an output, newest first', () => {
    const run = {
      review_status: 'unreviewed',
      attempts: [
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', settings: { animation: 'Sword_Attack' }, completed_at: 100 },
        { stage: 'retarget-motion', status: 'failed', output_artifact_id: 'clip-failed' },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: null, settings: { animation: 'Sword_Attack' }, completed_at: 200 },
        { stage: 'auto-rig', status: 'completed', output_artifact_id: 'rig-1', completed_at: 150 },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-2', settings: { animation: 'Sword_Attack' }, completed_at: 300 },
      ],
    }

    expect(journeyAnimations(run).map((animation) => animation.artifactId)).toEqual(['clip-2', 'clip-1'])
  })

  it('prettifies the settings.animation name into a human label', () => {
    const run = {
      attempts: [{ stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', settings: { animation: 'Sword_Attack' }, completed_at: 1 }],
    }

    expect(journeyAnimations(run)[0].label).toBe('Sword Attack')
  })

  it('lists completed apply-clip attempts (N5 direct-attach) alongside retarget-motion/generate-motion', () => {
    const run = {
      attempts: [
        { stage: 'apply-clip', status: 'completed', output_artifact_id: 'direct-1', settings: { clip: 'Sword_Attack_Armature', family: 'biped' }, completed_at: 1 },
        { stage: 'apply-clip', status: 'failed', output_artifact_id: 'direct-failed' },
      ],
    }

    expect(journeyAnimations(run).map((animation) => animation.artifactId)).toEqual(['direct-1'])
  })

  it('prettifies settings.clip into a human label for apply-clip (no settings.animation to fall back to)', () => {
    const run = {
      attempts: [{ stage: 'apply-clip', status: 'completed', output_artifact_id: 'direct-1', settings: { clip: 'Sword_Attack_Armature', family: 'biped' }, completed_at: 1 }],
    }

    expect(journeyAnimations(run)[0].label).toBe('Sword Attack Armature')
  })

  it('falls back to "Slither" for a chain-rig generate-motion clip with no settings.animation', () => {
    const run = {
      attempts: [{ stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1', completed_at: 1 }],
    }

    expect(journeyAnimations(run, true)[0]).toMatchObject({ artifactId: 'slither-1', label: 'Slither', stage: 'generate-motion' })
  })

  it('label order is settings.animation, then the per-stage fallback — never the artifact\'s own file name', () => {
    // journeyAnimations doesn't even accept an artifact-details map any
    // more (it used to, and briefly fell through to a generic export name
    // like "Generate Motion Result" before the settings.animation check
    // ever ran wrong) — label is derived purely from the attempt itself.
    const run = {
      attempts: [
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1', completed_at: 1 },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', completed_at: 2 },
      ],
    }

    const animations = journeyAnimations(run, true)
    expect(animations.find((animation) => animation.artifactId === 'slither-1')?.label).toBe('Slither')
    expect(animations.find((animation) => animation.artifactId === 'clip-1')?.label).toBe('Retarget Motion')
  })

  // 7zdn5b: a non-chain rig's generate-motion attempt is always NVIDIA
  // Kimodo's raw (unretargeted) BVH -- not a standalone playable clip -- so
  // it must never show up as a row here, unlike a chain rig's own
  // generate-motion attempt (its procedural Slither bake IS the playable
  // GLB, no separate retarget step).
  it('drops a non-chain-rig generate-motion attempt entirely -- its raw BVH is not playable', () => {
    const run = {
      attempts: [
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'kimodo-bvh-1', settings: { prompt: 'A sword slash' }, completed_at: 1 },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', settings: { animation: 'Sword_Attack' }, completed_at: 2 },
      ],
    }

    expect(journeyAnimations(run, false).map((animation) => animation.artifactId)).toEqual(['clip-1'])
    expect(journeyAnimations(run).map((animation) => animation.artifactId)).toEqual(['clip-1'])
  })

  it('keeps a chain-rig generate-motion attempt (its own output is already the playable clip)', () => {
    const run = {
      attempts: [{ stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1', completed_at: 1 }],
    }

    expect(journeyAnimations(run, true).map((animation) => animation.artifactId)).toEqual(['slither-1'])
  })

  it('marks reviewState accepted or rejected only for the artifact the run actually reviewed', () => {
    const run = {
      review_status: 'accepted',
      reviewed_artifact_id: 'clip-1',
      attempts: [
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', completed_at: 2 },
        { stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-2', completed_at: 1 },
      ],
    }

    const animations = journeyAnimations(run)
    expect(animations.find((animation) => animation.artifactId === 'clip-1')?.reviewState).toBe('accepted')
    expect(animations.find((animation) => animation.artifactId === 'clip-2')?.reviewState).toBe('unreviewed')
  })

  it('surfaces a "retained" signal via promoted:false without overriding an explicit review state', () => {
    const run = {
      review_status: 'unreviewed',
      attempts: [{ stage: 'retarget-motion', status: 'completed', output_artifact_id: 'clip-1', promoted: false, completed_at: 1 }],
    }

    expect(journeyAnimations(run)[0]).toMatchObject({ promoted: false, reviewState: 'unreviewed' })
  })

  it('carries the input rig artifact id through as a raw id, for the row\'s "on <rig> · <clip id>" caption', () => {
    const run = {
      attempts: [
        { stage: 'retarget-motion', status: 'completed', input_artifact_id: 'rig-abc123', output_artifact_id: 'clip-1', completed_at: 1 },
        { stage: 'generate-motion', status: 'completed', output_artifact_id: 'slither-1', completed_at: 2 },
      ],
    }

    const animations = journeyAnimations(run, true)
    expect(animations.find((animation) => animation.artifactId === 'clip-1')?.inputArtifactId).toBe('rig-abc123')
    // No input_artifact_id on the attempt (shouldn't happen in practice —
    // the backend always stamps one — but never crash the caption on it).
    expect(animations.find((animation) => animation.artifactId === 'slither-1')?.inputArtifactId).toBe('')
  })
})

describe('generateMotionFallbackLabel', () => {
  it('labels a chain-rig bake "Slither" regardless of any prompt setting', () => {
    expect(generateMotionFallbackLabel(true, undefined)).toBe('Slither')
    expect(generateMotionFallbackLabel(true, 'A sword slash')).toBe('Slither')
  })

  it('labels a non-chain (Kimodo) clip with its prompt when one was recorded', () => {
    expect(generateMotionFallbackLabel(false, 'A stylized female sword fighter performs one decisive slash'))
      .toBe('A stylized female sword fighter performs one decisive slash')
  })

  it('trims whitespace and falls back to "Generated Motion" for a non-chain clip with no usable prompt -- never "Slither"', () => {
    expect(generateMotionFallbackLabel(false, undefined)).toBe('Generated Motion')
    expect(generateMotionFallbackLabel(false, '')).toBe('Generated Motion')
    expect(generateMotionFallbackLabel(false, '   ')).toBe('Generated Motion')
    expect(generateMotionFallbackLabel(false, '  A sword slash  ')).toBe('A sword slash')
  })
})

describe('curatedClipId', () => {
  it('strips the Blender import suffix and kebab-cases the remainder', () => {
    expect(curatedClipId('Sword_Attack_Armature')).toBe('sword-attack')
  })

  it('matches this file\'s own hardcoded fallback id for the same clip', () => {
    expect(curatedClipId('Sword_Attack_Armature')).toBe('sword-attack')
  })
})

describe('curatedActionLabel', () => {
  it('strips the Blender import suffix but keeps underscores, for the baked output action name', () => {
    expect(curatedActionLabel('Walk_Armature')).toBe('Walk')
    expect(curatedActionLabel('Sword_Attack_Armature')).toBe('Sword_Attack')
  })
})

describe('curatedDonorActionName', () => {
  it('strips the Blender import suffix to find the clip inside the loaded donor GLB', () => {
    expect(curatedDonorActionName('Dance_Simple_Armature')).toBe('Dance_Simple')
    expect(curatedDonorActionName('Sword_Attack_Armature')).toBe('Sword_Attack')
  })

  it('leaves a name with no suffix untouched, defensively', () => {
    expect(curatedDonorActionName('Walk')).toBe('Walk')
  })
})

describe('curatedMotionClipsFromResponse', () => {
  it('maps the backend clip list into the picker shape, deriving each id', () => {
    const payload = {
      donor: 'vendor/mesh2motion-app/static/animations/human-base-animations.glb',
      clips: [
        { animation: 'Sword_Attack_Armature', label: 'Sword Attack', frames: 47, duration_seconds: 1.9 },
        { animation: 'Walk_Armature', label: 'Walk', frames: 41, duration_seconds: 1.6 },
      ],
    }

    expect(curatedMotionClipsFromResponse(payload)).toEqual([
      { id: 'sword-attack', label: 'Sword Attack', animation: 'Sword_Attack_Armature' },
      { id: 'walk', label: 'Walk', animation: 'Walk_Armature' },
    ])
  })
})

describe('fetchCuratedMotionClips', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('fetches and maps the curated clip list from the given API base', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        donor: 'vendor/mesh2motion-app/static/animations/human-base-animations.glb',
        clips: [{ animation: 'Sword_Attack_Armature', label: 'Sword Attack', frames: 47, duration_seconds: 1.9 }],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const clips = await fetchCuratedMotionClips('http://localhost:8100/api')

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:8100/api/animations/curated')
    expect(clips).toEqual([{ id: 'sword-attack', label: 'Sword Attack', animation: 'Sword_Attack_Armature' }])
  })

  it('throws on a non-OK response, so the caller can keep its fallback list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))

    await expect(fetchCuratedMotionClips('http://localhost:8100/api')).rejects.toThrow('500')
  })
})
