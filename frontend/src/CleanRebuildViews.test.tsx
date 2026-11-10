import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

// App.tsx imports StudioViewport, which registers the <model-viewer> custom
// element as an import side effect and needs a browser DOM this suite's Node
// environment does not provide (same stub as CleanCanonicalize.test.tsx).
vi.mock('@google/model-viewer', () => ({}))

import { CleanControls, RebuildViewsCard } from './App'
import { cleanMethodForStage, rebuildViewsJobPayload, rebuildViewsReport } from './workspace'

const cleanProps = {
  targetVertices: 25_000,
  setTargetVertices: vi.fn(),
  repairHoles: false,
  setRepairHoles: vi.fn(),
  gpuFirst: false,
  setGpuFirst: vi.fn(),
  suggestedVertices: null,
  source: null,
  refineImage: null,
  onAttachImage: vi.fn(),
  onUpload: vi.fn(),
}

// The exact metadata backend/app/main.py's `rebuild-views` branch merges onto
// the output artifact: `mv_iou` is approach9's report.json and `gate` is
// scripts/source_fidelity_multiview.py's multiview.json.
const metadata = {
  stage: 'rebuild-views',
  reference_roles: ['front', 'back'],
  mv_iou: { mean_iou: 0.94, front: { iou: 0.95 }, back: { iou: 0.93 } },
  gate: {
    front: { silhouette_iou: 0.91, masked_ssim: 0.544, face_ssim: 0.32 },
    back: { silhouette_iou: 0.9, masked_ssim: 0.672, face_ssim: 0.56 },
  },
}

describe('rebuild-views job submission', () => {
  it('posts the stage with no settings of its own', () => {
    expect(rebuildViewsJobPayload('mesh-1', 'run-7', 42)).toEqual({
      kind: 'rebuild-views',
      source_artifact_id: 'mesh-1',
      run_id: 'run-7',
      seed: 42,
      settings: {},
    })
  })

  it('drops run_id when the mesh has no journey yet, so the backend opens one', () => {
    expect(rebuildViewsJobPayload('mesh-1', null, 0).run_id).toBeUndefined()
  })

  it('is the method a restored rebuild-views output reopens on', () => {
    expect(cleanMethodForStage('rebuild-views')).toBe('rebuild-views')
  })
})

describe('rebuildViewsReport', () => {
  it('reads the mean IoU and the front/back gate scores off the artifact metadata', () => {
    expect(rebuildViewsReport(metadata)).toEqual({
      meanIou: 0.94,
      front: { ssim: 0.544, face: 0.32 },
      back: { ssim: 0.672, face: 0.56 },
    })
  })

  it('reports what it has for a job whose gate never ran', () => {
    expect(rebuildViewsReport({ mv_iou: { mean_iou: 0.9 }, gate: null }))
      .toEqual({ meanIou: 0.9, front: null, back: null })
  })

  it('never invents numbers for a missing or half-written report', () => {
    expect(rebuildViewsReport(undefined)).toEqual({ meanIou: null, front: null, back: null })
    expect(rebuildViewsReport({})).toEqual({ meanIou: null, front: null, back: null })
    expect(rebuildViewsReport({ mv_iou: {}, gate: { front: {} } }))
      .toEqual({ meanIou: null, front: { ssim: null, face: null }, back: null })
  })
})

describe('Clean method selector', () => {
  it('offers Rebuild from views alongside the three existing methods', () => {
    const markup = renderToStaticMarkup(
      <CleanControls {...cleanProps} method="canonicalize" setMethod={vi.fn()} />,
    )

    expect(markup).toContain('Canonical (S2)')
    expect(markup).toContain('AI Refine')
    expect(markup).toContain('Instant Meshes')
    // short label in the selector, full name on the tooltip
    expect(markup).toContain('title="Rebuild from views">From views<')
  })

  it('hides Rebuild from views when the backend lacks its optional scripts', () => {
    const markup = renderToStaticMarkup(
      <CleanControls {...cleanProps} method="canonicalize" setMethod={vi.fn()} rebuildViewsAvailable={false} />,
    )

    expect(markup).toContain('Canonical (S2)')
    expect(markup).not.toContain('From views')
  })

  it('explains what the selected method does and which references it uses', () => {
    const markup = renderToStaticMarkup(
      <CleanControls {...cleanProps} method="rebuild-views" setMethod={vi.fn()} />,
    )

    expect(markup).toContain('Pixal3D multi-view')
    expect(markup).toContain('Front + back')
    expect(markup).toContain('Left / right')
  })

  it('shows the result card once a rebuild has completed', () => {
    const markup = renderToStaticMarkup(
      <CleanControls
        {...cleanProps} method="rebuild-views" setMethod={vi.fn()}
        rebuildResult={rebuildViewsReport(metadata)}
      />,
    )

    expect(markup).toContain('rebuild-views-result')
    expect(markup).toContain('mean IoU 0.94')
  })
})

describe('RebuildViewsCard', () => {
  it('states how well the result matches its views and the real references', () => {
    const markup = renderToStaticMarkup(<RebuildViewsCard report={rebuildViewsReport(metadata)} />)

    expect(markup).toContain('rebuild-views-result')
    expect(markup).toContain('mean IoU 0.94')
    expect(markup).toContain('front SSIM 0.54 / face 0.32')
    expect(markup).toContain('back 0.67 / 0.56')
  })

  it('shows the mean IoU alone when the multiview gate produced nothing', () => {
    const markup = renderToStaticMarkup(
      <RebuildViewsCard report={{ meanIou: 0.9, front: null, back: null }} />,
    )

    expect(markup).toContain('mean IoU 0.90')
    expect(markup).not.toContain('front SSIM')
  })
})
