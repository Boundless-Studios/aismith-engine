import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { QualityPanels, SkeletonReviewGallery } from './HistoryView'
import type { Artifact, RunManifest } from './history'

const RUN = {
  id: 'run-1',
  name: 'Mary',
  status: 'completed',
  source_artifact_id: 'source-1',
  source_artifact_kind: 'image',
  review_status: 'unreviewed',
  review_note: '',
  attempts: [],
  created_at: 0,
  updated_at: 0,
} as unknown as RunManifest

describe('SkeletonReviewGallery', () => {
  it('renders nothing without skeleton review artifacts', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0, metadata: {},
    } as Artifact

    expect(renderToStaticMarkup(<SkeletonReviewGallery artifact={artifact} />)).toBe('')
  })

  it('shows the composed sheet first, ahead of the four raw views', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: { skeleton_review: { review_artifacts: [
        { artifact_id: 'front-1', label: 'front' },
        { artifact_id: 'sheet-1', label: 'skeleton-sheet' },
        { artifact_id: 'back-1', label: 'back' },
      ] } },
    } as unknown as Artifact

    const markup = renderToStaticMarkup(<SkeletonReviewGallery artifact={artifact} />)

    expect(markup).toContain('SKELETON REVIEW')
    expect(markup.indexOf('sheet-1')).toBeLessThan(markup.indexOf('front-1'))
    expect(markup.indexOf('sheet-1')).toBeLessThan(markup.indexOf('back-1'))
  })
})

describe('QualityPanels bilateral symmetry advisory', () => {
  it('renders the p95, band, outside-band count, and worst offender', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: {
        quality_decisions: {
          'rig-certificate': { status: 'ready', policy_version: 1, target_stage: 'rig-certificate', findings: [], recommended_actions: [] },
        },
        rig_quality: {
          symmetry_mirror_error_p95: 0.08,
          symmetry_pairs_outside_band_count: 1,
          bilateral_symmetry: {
            band: 0.03,
            worst_offenders: [{ left: 'hand_l', right: 'hand_r', mirror_error_normalized: 0.08 }],
          },
        },
      },
    } as unknown as Artifact

    const markup = renderToStaticMarkup(
      <QualityPanels artifact={artifact} run={RUN} onUseArtifact={vi.fn()} />,
    )

    expect(markup).toContain('Bilateral symmetry p95: 0.0800')
    expect(markup).toContain('band 0.03')
    expect(markup).toContain('1 pairs outside band')
    expect(markup).toContain('hand_l/hand_r')
  })

  it('omits the advisory line on a rig that predates the measurement', () => {
    const artifact = {
      id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
      metadata: {
        quality_decisions: {
          'rig-certificate': { status: 'ready', policy_version: 1, target_stage: 'rig-certificate', findings: [], recommended_actions: [] },
        },
        rig_quality: { joints: 21 },
      },
    } as unknown as Artifact

    const markup = renderToStaticMarkup(
      <QualityPanels artifact={artifact} run={RUN} onUseArtifact={vi.fn()} />,
    )

    expect(markup).not.toContain('Bilateral symmetry')
  })
})

describe('QualityPanels decisions without a target stage', () => {
  // The backend's source_fidelity / project_views decisions carry no
  // `target_stage`; the panel used to call `.replace` on it and unmount the
  // whole History view.
  const artifact = {
    id: 'mary', name: 'mary.glb', kind: 'glb', size_bytes: 1, created_at: 0,
    metadata: {
      quality_decisions: {
        'rig-certificate': { status: 'ready', policy_version: 1, target_stage: 'rig-certificate', findings: [], recommended_actions: [] },
        source_fidelity: { status: 'green', policy_version: 1, findings: [], recommended_actions: [], summary: { silhouette_iou: 0.9 } },
        project_views: {
          status: 'needs_review', policy_version: 1,
          findings: [{ code: 'projection.low_coverage', severity: 'warning', message: 'Projected texels cover little of the atlas', metric: 'projected_texel_fraction', actual: 0.2, expected: '>= 0.5' }],
          recommended_actions: [],
        },
      },
    },
  } as unknown as Artifact

  it('labels them by their decision key instead of throwing', () => {
    const markup = renderToStaticMarkup(
      <QualityPanels artifact={artifact} run={RUN} onUseArtifact={vi.fn()} />,
    )

    expect(markup).toContain('rig certificate')
    expect(markup).toContain('source fidelity')
    expect(markup).toContain('project views')
    expect(markup).toContain('needs review')
    expect(markup).toContain('Projected texels cover little of the atlas')
  })

  it('renders the compact summary line too', () => {
    const markup = renderToStaticMarkup(
      <QualityPanels artifact={artifact} run={RUN} onUseArtifact={vi.fn()} compact />,
    )

    expect(markup).toContain('source fidelity: green')
    expect(markup).toContain('project views: needs review')
  })
})
