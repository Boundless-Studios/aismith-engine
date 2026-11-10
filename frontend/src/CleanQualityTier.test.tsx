import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

// App.tsx imports StudioViewport, which registers the <model-viewer> custom
// element as an import side effect and needs a browser DOM this suite's Node
// environment does not provide (same stub as CleanCanonicalize.test.tsx).
vi.mock('@google/model-viewer', () => ({}))

import { CanonicalizeVerdictCard } from './App'
import {
  CANONICALIZE_DEFAULT_DENSITY,
  CUSTOM_TRIANGLE_TARGET_MAXIMUM,
  CUSTOM_TRIANGLE_TARGET_MINIMUM,
  QUALITY_TIER_OPTIONS,
  canonicalizeDensityReport,
  canonicalizeJobPayload,
  customTriangleTarget,
  readCanonicalizeDensity,
  resolvedTriangleTarget,
  serializeCanonicalizeDensity,
  type CanonicalizeVerdict,
} from './workspace'

// --- Mesh density: quality tiers (i63ky) ---------------------------------
//
// The operator's "more control over the number of generated vertices" knob.
// What matters is that the SELECTION survives into the payload and that the
// panel never sends a number the API would reject: a rejected canonicalize is
// a twenty-minute round trip the operator only finds out about at the end.

describe('quality tier payload (i63ky)', () => {
  it('carries the selected tier on the submitted settings', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7', null, 'voxel', { tier: 'ultra', customTriangles: null }).settings)
      .toEqual({ method: 'voxel', quality_tier: 'ultra' })
  })

  it('sends no tier at all when the caller selects none, so the backend default stands', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7').settings).toEqual({ method: 'voxel' })
  })

  it('adds a custom triangle target alongside the tier it overrides', () => {
    expect(canonicalizeJobPayload('mesh-1', 'run-7', null, 'voxel', { tier: 'draft', customTriangles: 96_000 }).settings)
      .toEqual({ method: 'voxel', quality_tier: 'draft', target_triangles: 96_000 })
  })

  it('drops an out-of-range or half-typed custom target rather than submitting a 400', () => {
    for (const custom of [10, 900_000, Number.NaN, null]) {
      expect(canonicalizeJobPayload('mesh-1', 'run-7', null, 'voxel', { tier: 'high', customTriangles: custom as number | null }).settings)
        .toEqual({ method: 'voxel', quality_tier: 'high' })
    }
  })
})

describe('quality tier resolution helpers', () => {
  it('advertises the four tiers with the backend\'s triangle targets', () => {
    expect(QUALITY_TIER_OPTIONS.map((option) => [option.id, option.targetTriangles])).toEqual([
      ['draft', 10_000], ['standard', 30_000], ['high', 80_000], ['ultra', 150_000],
    ])
  })

  it('gives every tier a plain-language tradeoff caption', () => {
    for (const option of QUALITY_TIER_OPTIONS) expect(option.caption.length).toBeGreaterThan(20)
  })

  it('resolves a custom target over the selected tier, and falls back when it is unusable', () => {
    expect(resolvedTriangleTarget({ tier: 'draft', customTriangles: 96_000 })).toBe(96_000)
    expect(resolvedTriangleTarget({ tier: 'ultra', customTriangles: null })).toBe(150_000)
    expect(resolvedTriangleTarget({ tier: 'ultra', customTriangles: 10 })).toBe(150_000)
  })

  it('accepts the backend\'s exact custom range and nothing outside it', () => {
    expect(customTriangleTarget(CUSTOM_TRIANGLE_TARGET_MINIMUM)).toBe(CUSTOM_TRIANGLE_TARGET_MINIMUM)
    expect(customTriangleTarget(CUSTOM_TRIANGLE_TARGET_MAXIMUM)).toBe(CUSTOM_TRIANGLE_TARGET_MAXIMUM)
    expect(customTriangleTarget(CUSTOM_TRIANGLE_TARGET_MINIMUM - 1)).toBeNull()
    expect(customTriangleTarget(CUSTOM_TRIANGLE_TARGET_MAXIMUM + 1)).toBeNull()
    expect(customTriangleTarget('48000')).toBe(48_000)
    expect(customTriangleTarget('')).toBeNull()
  })
})

describe('per-session density memory', () => {
  it('round-trips a selection through session storage', () => {
    const density = { tier: 'high' as const, customTriangles: 96_000 }
    expect(readCanonicalizeDensity(serializeCanonicalizeDensity(density))).toEqual(density)
  })

  it('falls back to the default for missing, corrupt or unknown stored values', () => {
    expect(readCanonicalizeDensity(null)).toEqual(CANONICALIZE_DEFAULT_DENSITY)
    expect(readCanonicalizeDensity('not json')).toEqual(CANONICALIZE_DEFAULT_DENSITY)
    expect(readCanonicalizeDensity('{"tier":"cinematic"}')).toEqual(CANONICALIZE_DEFAULT_DENSITY)
  })

  it('drops a stored custom target that is no longer in range', () => {
    expect(readCanonicalizeDensity('{"tier":"ultra","custom_triangles":900000}'))
      .toEqual({ tier: 'ultra', customTriangles: null })
  })
})

describe('canonicalizeDensityReport', () => {
  it('reads what was asked for and what came out off the artifact metadata', () => {
    expect(canonicalizeDensityReport({
      quality_tier: 'ultra',
      quality_tier_requested: 'ultra',
      target_triangles: 150_000,
      achieved_triangles: 148_412,
      triangle_budget_band: [75_000, 225_000],
      bake_texture_size: 4096,
    })).toEqual({
      tier: 'ultra',
      requestedTier: 'ultra',
      targetTriangles: 150_000,
      achievedTriangles: 148_412,
      band: [75_000, 225_000],
      atlasSize: 4096,
    })
  })

  it('is null for an artifact canonicalized before tiers existed', () => {
    expect(canonicalizeDensityReport({ attempts: [{ faces: 29_992 }] })).toBeNull()
    expect(canonicalizeDensityReport(null)).toBeNull()
  })
})

describe('CanonicalizeVerdictCard density line', () => {
  const ready: CanonicalizeVerdict = { policy_version: 1, status: 'ready', findings: [], recommended_actions: [] }

  it('shows the tier, the achieved triangles and the band the gate used', () => {
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard
        verdict={ready} artifactName="alien-ultra.glb" promoted pinned={false} onPin={vi.fn()}
        density={{ tier: 'ultra', requestedTier: 'ultra', targetTriangles: 150_000, achievedTriangles: 148_412, band: [75_000, 225_000], atlasSize: 4096 }}
      />,
    )

    expect(markup).toContain('canonicalize-density')
    expect(markup).toContain('ultra')
    expect(markup).toContain('148,412')
    expect(markup).toContain('4096px')
  })

  it('says nothing about density for a pre-tier artifact rather than inventing a tier', () => {
    const markup = renderToStaticMarkup(
      <CanonicalizeVerdictCard verdict={ready} artifactName="alien-canonical.glb" promoted pinned={false} onPin={vi.fn()} />,
    )

    expect(markup).not.toContain('canonicalize-density')
  })
})
