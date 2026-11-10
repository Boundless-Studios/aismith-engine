import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

// App.tsx imports StudioViewport, which registers the <model-viewer> custom
// element as an import side effect and needs a browser DOM this suite's Node
// environment does not provide (same stub as CleanCanonicalize.test.tsx).
vi.mock('@google/model-viewer', () => ({}))

import { API } from './api'
import { ProjectionVerdictCard, SourceFidelityCard } from './App'
import { projectionReport, projectionVerdict, type ProjectionVerdict, sourceFidelityReport, sourceFidelityVerdict, type SourceFidelityVerdict } from './workspace'

// The exact shape backend/app/projection_policy.py's evaluate_projection
// writes into metadata.quality_decisions["project_views"].
const needsReviewVerdict: ProjectionVerdict = {
  policy_version: 'projection-v1',
  status: 'needs_review',
  findings: [
    {
      code: 'project_views.low_projected_coverage',
      severity: 'warning',
      message: 'Less than half the atlas received a projected texel; most of the texture still comes from the existing bake.',
      metric: 'projected_texel_fraction',
      actual: 0.31,
      expected: '>= 0.5',
    },
    {
      code: 'project_views.low_role_coverage',
      severity: 'warning',
      message: 'view left contributed little',
      metric: 'per_role_coverage.left',
      actual: 0.05,
      expected: '>= 0.15',
    },
  ],
  recommended_actions: ['inspect-projection-coverage'],
}

const blockedVerdict: ProjectionVerdict = {
  policy_version: 'projection-v1',
  status: 'blocked',
  findings: [
    {
      code: 'project_views.insufficient_views',
      severity: 'error',
      message: 'Only 1 conditioning view(s) contributed to the projection; at least 2 are required.',
      metric: 'views_used',
      actual: 1,
      expected: '>= 2',
    },
  ],
  recommended_actions: ['resolve-conditioning-views'],
}

const goodReport = {
  schema_version: 1,
  views_used: ['back', 'front', 'left'],
  per_role_coverage: { front: 0.5, back: 0.4, left: 0.3 },
  projected_texel_fraction: 0.72,
  fallback_texel_fraction: 0.28,
  occlusion_rejected_fraction: { front: 0.05, back: 0.02, left: 0.01 },
  mean_color_delta_vs_bake: 0.031,
  atlas_size: 2048,
  frame_alignment: {},
  caveats: [],
}

describe('projectionVerdict', () => {
  it('reads the gate decision off the output artifact metadata', () => {
    const verdict = projectionVerdict({ quality_decisions: { project_views: needsReviewVerdict } })
    expect(verdict).toMatchObject({ status: 'needs_review', policy_version: 'projection-v1' })
    expect(verdict?.findings).toHaveLength(2)
  })

  it('is null when the artifact never went through the pass', () => {
    expect(projectionVerdict(undefined)).toBeNull()
    expect(projectionVerdict({})).toBeNull()
    expect(projectionVerdict({ quality_decisions: { canonicalize: { status: 'ready' } } })).toBeNull()
  })

  it('still reports the status when a partial decision carries no findings array', () => {
    const verdict = projectionVerdict({ quality_decisions: { project_views: { status: 'green' } } })
    expect(verdict).toEqual({ status: 'green', findings: [], recommended_actions: [], policy_version: undefined })
  })
})

describe('projectionReport', () => {
  it('reads a project-views job output, whose report is flattened onto metadata', () => {
    const report = projectionReport(goodReport)
    expect(report).toMatchObject({
      viewsUsed: ['back', 'front', 'left'],
      projectedTexelFraction: 0.72,
      fallbackTexelFraction: 0.28,
      meanColorDeltaVsBake: 0.031,
      atlasSize: 2048,
    })
    expect(report?.perRoleCoverage).toEqual({ front: 0.5, back: 0.4, left: 0.3 })
  })

  it('reads a canonicalize job output, whose report is nested under view_projection', () => {
    const report = projectionReport({ view_projection: goodReport, projected_texel_fraction: 'not-this-one' })
    expect(report?.viewsUsed).toEqual(['back', 'front', 'left'])
    expect(report?.projectedTexelFraction).toBe(0.72)
  })

  it('is null when canonicalize skipped the pass', () => {
    expect(
      projectionReport({ view_projection: { skipped: 'insufficient_views', views_available: ['front'] } }),
    ).toBeNull()
  })

  it('is null for metadata that never carried a projection report', () => {
    expect(projectionReport(undefined)).toBeNull()
    expect(projectionReport({})).toBeNull()
  })
})

describe('ProjectionVerdictCard', () => {
  it('renders every finding of a needs_review verdict, with coverage stats', () => {
    const markup = renderToStaticMarkup(
      <ProjectionVerdictCard verdict={needsReviewVerdict} report={projectionReport(goodReport)} />,
    )

    expect(markup).toContain('quality-card needs_review')
    expect(markup).toContain('needs review')
    expect(markup).toContain('Less than half the atlas received a projected texel')
    expect(markup).toContain('view left contributed little')
    expect(markup).toContain('per_role_coverage.left')
    expect(markup).toContain('inspect-projection-coverage')
    expect(markup).toContain('72%')
    expect(markup).toContain('back, front, left')
  })

  it('renders a green verdict with no findings', () => {
    const green: ProjectionVerdict = { policy_version: 'projection-v1', status: 'green', findings: [], recommended_actions: [] }
    const markup = renderToStaticMarkup(<ProjectionVerdictCard verdict={green} />)

    expect(markup).toContain('quality-card green')
    expect(markup).toContain('Every conditioning view contributed meaningfully')
  })

  it('renders a blocked verdict', () => {
    const markup = renderToStaticMarkup(<ProjectionVerdictCard verdict={blockedVerdict} />)

    expect(markup).toContain('quality-card blocked')
    expect(markup).toContain('Only 1 conditioning view(s) contributed')
    expect(markup).toContain('resolve-conditioning-views')
  })

  it('surfaces the report caveats when present', () => {
    const report = projectionReport({ ...goodReport, caveats: ["View(s) contributed nothing: right."] })
    const green: ProjectionVerdict = { policy_version: 'projection-v1', status: 'green', findings: [], recommended_actions: [] }
    const markup = renderToStaticMarkup(<ProjectionVerdictCard verdict={green} report={report} />)

    expect(markup).toContain('View(s) contributed nothing: right.')
  })
})

// The exact shape backend/app/source_fidelity.py writes into
// metadata.quality_decisions["source_fidelity"] and metadata.source_fidelity.
const needsReviewFidelityDecision = {
  policy_version: 'source-fidelity-v1',
  status: 'needs_review',
  findings: [
    {
      code: 'source_fidelity.low_silhouette_iou',
      severity: 'warning',
      message: 'The render silhouette diverges from the source photo silhouette.',
      metric: 'silhouette_iou',
      actual: 0.61,
      expected: '>= 0.75',
    },
  ],
  recommended_actions: ['re-align-camera'],
  summary: {
    silhouette_iou: 0.61,
    masked_ssim: 0.72,
    face_ssim: null,
    perceptual_distance: 0.18,
    masked_luma_delta: 0.05,
    detail_ratio: 0.55,
    face_detail_ratio: 0.42,
  },
}

const fullFidelityReport = {
  side_by_side_artifact_id: 'artifact-side-by-side-1',
  photo_artifact_id: 'artifact-photo-1',
  alignment: { bbox_width_ratio: 0.94, scale: 1.02 },
  camera: { kind: 'perspective', fov_x_degrees: 42.5, yaw_degrees: 3.1, pitch_degrees: -1.2 },
}

describe('sourceFidelityVerdict', () => {
  it('reads the gate decision off the output artifact metadata', () => {
    const verdict = sourceFidelityVerdict({ quality_decisions: { source_fidelity: needsReviewFidelityDecision } })
    expect(verdict).toMatchObject({ status: 'needs_review', policy_version: 'source-fidelity-v1' })
    expect(verdict?.findings).toHaveLength(1)
    expect(verdict?.summary).toEqual({
      silhouetteIou: 0.61,
      maskedSsim: 0.72,
      faceSsim: null,
      perceptualDistance: 0.18,
      maskedLumaDelta: 0.05,
      detailRatio: 0.55,
      faceDetailRatio: 0.42,
    })
  })

  it('is null when the artifact never went through the pass', () => {
    expect(sourceFidelityVerdict(undefined)).toBeNull()
    expect(sourceFidelityVerdict({})).toBeNull()
    expect(sourceFidelityVerdict({ quality_decisions: { canonicalize: { status: 'ready' } } })).toBeNull()
  })

  it('still reports the status and null summary numbers when a partial decision carries no summary', () => {
    const verdict = sourceFidelityVerdict({ quality_decisions: { source_fidelity: { status: 'not_evaluable' } } })
    expect(verdict).toEqual({
      status: 'not_evaluable',
      findings: [],
      recommended_actions: [],
      policy_version: undefined,
      summary: { silhouetteIou: null, maskedSsim: null, faceSsim: null, perceptualDistance: null, maskedLumaDelta: null, detailRatio: null, faceDetailRatio: null },
    })
  })
})

describe('sourceFidelityReport', () => {
  it('reads the full report, including the side-by-side artifact and camera', () => {
    const report = sourceFidelityReport({ source_fidelity: fullFidelityReport })
    expect(report).toEqual({
      skipped: null,
      sideBySideArtifactId: 'artifact-side-by-side-1',
      photoArtifactId: 'artifact-photo-1',
      bboxWidthRatio: 0.94,
      cameraKind: 'perspective',
    })
  })

  it('reads a skipped report, leaving the rest null', () => {
    const report = sourceFidelityReport({ source_fidelity: { skipped: 'no_source_photo' } })
    expect(report).toEqual({
      skipped: 'no_source_photo',
      sideBySideArtifactId: null,
      photoArtifactId: null,
      bboxWidthRatio: null,
      cameraKind: null,
    })
  })

  it('is null for metadata that never carried a source-fidelity report', () => {
    expect(sourceFidelityReport(undefined)).toBeNull()
    expect(sourceFidelityReport({})).toBeNull()
  })
})

describe('SourceFidelityCard', () => {
  it('renders the status, the numbers, camera and the side-by-side image', () => {
    const verdict = sourceFidelityVerdict({ quality_decisions: { source_fidelity: needsReviewFidelityDecision } })!
    const report = sourceFidelityReport({ source_fidelity: fullFidelityReport })
    const markup = renderToStaticMarkup(<SourceFidelityCard verdict={verdict} report={report} />)

    expect(markup).toContain('quality-card needs_review')
    expect(markup).toContain('needs review')
    expect(markup).toContain('0.610')
    expect(markup).toContain('0.720')
    expect(markup).toContain('0.180')
    expect(markup).toContain('0.55')
    expect(markup).toContain('0.42')
    expect(markup).toContain('perspective')
    expect(markup).toContain('The render silhouette diverges from the source photo silhouette.')
    expect(markup).toContain('re-align-camera')
    // Same API base as every other asset URL (api.ts), not a hard-coded /api.
    expect(markup).toContain(`src="${API}/assets/artifact-side-by-side-1/download"`)
  })

  it('renders a green verdict with no findings and no report', () => {
    const green: SourceFidelityVerdict = {
      status: 'green',
      findings: [],
      recommended_actions: [],
      summary: { silhouetteIou: 0.9, maskedSsim: 0.91, faceSsim: 0.88, perceptualDistance: 0.05, maskedLumaDelta: 0.01, detailRatio: 0.95, faceDetailRatio: 0.92 },
    }
    const markup = renderToStaticMarkup(<SourceFidelityCard verdict={green} />)

    expect(markup).toContain('quality-card green')
    expect(markup).toContain('No findings')
  })

  it('shows the skipped reason instead of the side-by-side image', () => {
    const green: SourceFidelityVerdict = {
      status: 'not_evaluable',
      findings: [],
      recommended_actions: [],
      summary: { silhouetteIou: null, maskedSsim: null, faceSsim: null, perceptualDistance: null, maskedLumaDelta: null, detailRatio: null, faceDetailRatio: null },
    }
    const report = sourceFidelityReport({ source_fidelity: { skipped: 'no_source_photo' } })
    const markup = renderToStaticMarkup(<SourceFidelityCard verdict={green} report={report} />)

    expect(markup).toContain('no_source_photo')
    expect(markup).not.toContain('source-fidelity-side-by-side')
  })
})
