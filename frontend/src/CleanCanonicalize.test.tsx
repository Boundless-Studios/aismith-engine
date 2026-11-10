import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

// App.tsx imports StudioViewport, which registers the <model-viewer> custom
// element as an import side effect and needs a browser DOM this suite's Node
// environment does not provide (same stub as RigAnimateWorkspace.test.tsx).
vi.mock('@google/model-viewer', () => ({}))

import { CanonicalizeVerdictCard } from './App'
import { canonicalizeJobPayload, canonicalizeVerdict, type CanonicalizeVerdict } from './workspace'

// The exact shape backend/app/canonicalize_policy.py writes into
// metadata.quality_decisions["canonicalize"] (see _finding/_decision there).
const blockedVerdict: CanonicalizeVerdict = {
  policy_version: 1,
  status: 'blocked',
  findings: [
    {
      code: 'canonicalize-components',
      severity: 'error',
      message: 'Canonical mesh must be a single connected component.',
      metric: 'components',
      actual: 4,
      expected: '== 1',
    },
    {
      code: 'canonicalize-similarity',
      severity: 'warning',
      message: 'Baked appearance drifted from the source views.',
      metric: 'similarity',
      actual: 0.81,
      expected: '>= 0.88',
    },
  ],
  recommended_actions: ['retry-canonicalize', 'manual-blender-cleanup'],
}

describe('canonicalize job submission (r029o)', () => {
  it('submits method=voxel and no vertex target by default', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7')).toEqual({
      kind: 'canonicalize',
      source_artifact_id: 'mesh-1',
      run_id: 'run-7',
      settings: { method: 'voxel' },
    })
  })

  it('carries an explicit target_vertices when the caller pins one, rounded to an integer', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7', 24999.6).settings)
      .toEqual({ method: 'voxel', target_vertices: 25000 })
  })

  it('omits target_vertices for a non-finite target rather than sending NaN', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7', Number.NaN).settings).toEqual({ method: 'voxel' })
    expect(canonicalizeJobPayload('mesh-1', 'run-7', null).settings).toEqual({ method: 'voxel' })
  })

  it('drops run_id when the mesh has no journey yet, so the backend opens one', () => {
    expect(canonicalizeJobPayload('mesh-1', null).run_id).toBeUndefined()
  })

  it('can request the Instant Meshes comparison backend explicitly', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7', null, 'instant-meshes').settings)
      .toEqual({ method: 'instant-meshes' })
  })
})

describe('canonicalizeVerdict', () => {
  it('reads the gate decision off the output artifact metadata', () => {
    const verdict = canonicalizeVerdict({ quality_decisions: { canonicalize: blockedVerdict } })
    expect(verdict).toMatchObject({ status: 'blocked', policy_version: 1 })
    expect(verdict?.findings).toHaveLength(2)
  })

  it('is null for a mesh that never went through the stage', () => {
    expect(canonicalizeVerdict(undefined)).toBeNull()
    expect(canonicalizeVerdict({})).toBeNull()
    expect(canonicalizeVerdict({ quality_decisions: { rig: { status: 'ready' } } })).toBeNull()
  })

  it('still reports the status when a partial decision carries no findings array', () => {
    const verdict = canonicalizeVerdict({ quality_decisions: { canonicalize: { status: 'ready' } } })
    expect(verdict).toEqual({ status: 'ready', findings: [], recommended_actions: [], policy_version: undefined })
  })
})

describe('CanonicalizeVerdictCard', () => {
  it('renders every finding of a blocked verdict, with its metric and expectation', () => {
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard verdict={blockedVerdict} artifactName="alien-canonical.glb" promoted={false} pinned={false} onPin={vi.fn()} />,
    )

    expect(markup).toContain('quality-card blocked')
    expect(markup).toContain('blocked')
    expect(markup).toContain('Canonical mesh must be a single connected component.')
    expect(markup).toContain('components')
    expect(markup).toContain('== 1')
    expect(markup).toContain('Baked appearance drifted from the source views.')
    expect(markup).toContain('&gt;= 0.88')
    expect(markup).toContain('retry-canonicalize')
  })

  it('offers the pin for a retained (non-promoted) candidate', () => {
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard verdict={blockedVerdict} artifactName="alien-canonical.glb" promoted={false} pinned={false} onPin={vi.fn()} />,
    )

    expect(markup).toContain('Pin canonical candidate')
  })

  it('reports promotion and still offers a way back to the canonical mesh once the gate passed', () => {
    // Clean's picker only lists what Clean consumes, so a later pin drops the
    // canonical artifact out of it — the card must stay the route back.
    const ready: CanonicalizeVerdict = { policy_version: 1, status: 'ready', findings: [], recommended_actions: [] }
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard verdict={ready} artifactName="alien-canonical.glb" promoted pinned={false} onPin={vi.fn()} />,
    )

    expect(markup).toContain('quality-card ready')
    expect(markup).toContain('No findings')
    expect(markup).toContain('Promoted')
    expect(markup).toContain('alien-canonical.glb')
    expect(markup).toContain('Work on the canonical mesh')
    expect(markup).not.toContain('Pin canonical candidate')
  })

  it('shows the pin as already taken when the candidate is the pinned artifact', () => {
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard verdict={blockedVerdict} artifactName="alien-canonical.glb" promoted={false} pinned onPin={vi.fn()} />,
    )

    expect(markup).toContain('Working on this mesh')
    expect(markup).toContain('disabled')
  })
})
