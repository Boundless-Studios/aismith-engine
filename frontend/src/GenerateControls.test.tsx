import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { GenerateControls } from './GenerateControls'
import type { TposeSetMembership } from './workspace'

const baseProps = {
  resolution: 1024 as const,
  setResolution: vi.fn(),
  seed: 0,
  setSeed: vi.fn(),
  preparedCandidate: null,
  preparing: false,
  onChooseOutput: vi.fn(),
  onUpload: vi.fn(),
  onPrepare: vi.fn(),
  onApprove: vi.fn(),
  onUploadView: vi.fn(),
  onRemoveView: vi.fn(),
}

// Bead 4vquq split this panel in two. A biped/auto run leads with the
// canonical T-pose set; object and creature runs keep the original
// source-image grid with its optional views. Every legacy-layout assertion
// below therefore names an object subject explicitly.
const legacyProps = { ...baseProps, subjectKind: 'object' as const }

describe('GenerateControls source image grid (object/creature runs)', () => {
  it('renders the front card plus three empty dashed view slots when no views are set', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={null} views={{}} sourceViewLabels={[]} />,
    )

    expect(markup).toContain('source-view-grid')
    expect(markup).toContain('upload-card')
    expect(markup).toContain('Add back view')
    expect(markup).toContain('Add left view')
    expect(markup).toContain('Add right view')
    expect(markup).not.toContain('view-slot-filled')
  })

  it('renders a thumbnail and remove control for a filled optional view', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...legacyProps}
        source={{ id: 'front-id', name: 'front.png', kind: 'image' }}
        views={{ back: { id: 'back-id', name: 'back.png', kind: 'image' } }}
        sourceViewLabels={[]}
      />,
    )

    expect(markup).toContain('view-slot-filled')
    expect(markup).toContain('view-slot-thumb')
    expect(markup).toContain('Remove back view')
    expect(markup).toContain('Add left view')
    expect(markup).toContain('Add right view')
  })

  it('shows the helper hint below the grid', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={null} views={{}} sourceViewLabels={[]} />,
    )

    expect(markup).toContain('Add side or back views of the same subject for a fuller reconstruction')
  })

  it('shows the Subject control and defaults to Auto', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls {...baseProps} source={null} views={{}} sourceViewLabels={[]} />,
    )
    expect(markup).toContain('subject-kind-segmented')
    expect(markup).toContain('Biped character')
  })

  it('shows read-only view chips only when the run recorded more than one view', () => {
    const withChips = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={null} views={{}} sourceViewLabels={['front', 'back']} />,
    )
    expect(withChips).toContain('Views: front · back')

    const withoutChips = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={null} views={{}} sourceViewLabels={['front']} />,
    )
    expect(withoutChips).not.toContain('Views:')
  })

  it('disables Auto-generate views without a front image, enables it with one', () => {
    const noSource = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={null} views={{}} sourceViewLabels={[]} />,
    )
    expect(noSource).toContain('Auto-generate views')
    // Matched on the button ELEMENT, not on "…text then a later disabled":
    // renderToStaticMarkup emits attributes before children, so the loose
    // form only ever passed by accident, via a different button downstream.
    expect(noSource).toMatch(/<button[^>]*auto-generate-views[^>]*disabled/)

    const withSource = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={{ id: 'front-id', name: 'front.png', kind: 'image' }} views={{}} sourceViewLabels={[]} />,
    )
    expect(withSource).not.toMatch(/<button[^>]*auto-generate-views[^>]*disabled/)
  })

  it('shows a synthesizing label and stays disabled while a synthesis job is running', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...legacyProps}
        source={{ id: 'front-id', name: 'front.png', kind: 'image' }}
        views={{}}
        sourceViewLabels={[]}
        autoGeneratingViews
      />,
    )
    expect(markup).toContain('Synthesizing views…')
  })

  it('tags a synthesized view slot but not an operator-uploaded one', () => {
    const synthesized = renderToStaticMarkup(
      <GenerateControls
        {...legacyProps}
        source={{ id: 'front-id', name: 'front.png', kind: 'image' }}
        views={{ back: { id: 'back-id', name: 'back.png', kind: 'image', metadata: { synthesized_view: 'back' } } }}
        sourceViewLabels={[]}
      />,
    )
    expect(synthesized).toContain('view-slot-synth-tag')

    const uploaded = renderToStaticMarkup(
      <GenerateControls
        {...legacyProps}
        source={{ id: 'front-id', name: 'front.png', kind: 'image' }}
        views={{ back: { id: 'back-id', name: 'back.png', kind: 'image' } }}
        sourceViewLabels={[]}
      />,
    )
    expect(uploaded).not.toContain('view-slot-synth-tag')
  })
})

// Bead 4vquq: the one operator step, and the quad it produces.
describe('GenerateControls canonical T-pose set', () => {
  const front = {
    id: 'front-id',
    name: 'T-pose front.png',
    kind: 'image',
    metadata: {
      tpose_set: {
        set_id: 'set-1',
        role: 'front',
        front_artifact_id: 'front-id',
        reference_artifact_id: 'reference-id',
        members: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
        failed_members: {},
        complete: true,
      },
    },
  }
  const membership: TposeSetMembership = {
    setId: 'set-1',
    frontArtifactId: 'front-id',
    referenceArtifactId: 'reference-id',
    members: { front: 'front-id', back: 'back-id', left: 'left-id', right: 'right-id' },
    failedRoles: [],
    complete: true,
  }

  it('demotes the uploaded image to a reference chip on a biped run', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls {...baseProps} subjectKind="biped" source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }} views={{}} sourceViewLabels={[]} />,
    )

    expect(markup).toContain('Reference — used for palette/identity')
    expect(markup).toContain('never what a biped generate is conditioned on')
    // The old two-step framing is gone from the biped panel entirely.
    expect(markup).not.toContain('Prepare for rigging')
    expect(markup).not.toContain('Auto-generate views')
  })

  it('leads with one Create T-pose set button, disabled until a reference exists', () => {
    const noSource = renderToStaticMarkup(
      <GenerateControls {...baseProps} subjectKind="biped" source={null} views={{}} sourceViewLabels={[]} />,
    )
    expect(noSource).toContain('Create T-pose set')
    expect(noSource).toMatch(/<button[^>]*create-tpose-set[^>]*disabled/)

    const withSource = renderToStaticMarkup(
      <GenerateControls {...baseProps} subjectKind="biped" source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }} views={{}} sourceViewLabels={[]} />,
    )
    expect(withSource).toContain('Create T-pose set')
    expect(withSource).not.toMatch(/<button[^>]*create-tpose-set[^>]*disabled/)
  })

  it('shows the job’s own staged progress line while the composite runs', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...baseProps}
        subjectKind="biped"
        source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }}
        views={{}}
        sourceViewLabels={[]}
        creatingTposeSet
        tposeSetProgress="Generating the back view… 2/4"
      />,
    )

    expect(markup).toContain('Generating the back view… 2/4')
  })

  it('renders the quad with the front tile marked canonical', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...baseProps}
        subjectKind="biped"
        source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }}
        views={{
          back: { id: 'back-id', name: 'back.png', kind: 'image', metadata: { synthesized_view: 'back' } },
          left: { id: 'left-id', name: 'left.png', kind: 'image', metadata: { synthesized_view: 'left' } },
          right: { id: 'right-id', name: 'right.png', kind: 'image', metadata: { synthesized_view: 'right' } },
        }}
        sourceViewLabels={[]}
        tposeSetFront={front}
        tposeSet={membership}
      />,
    )

    expect(markup).toContain('view-slot-front')
    expect(markup).toContain('view-slot-canonical-tag')
    expect(markup).toContain('canonical')
    expect(markup).toContain('Canonical T-pose front')
    expect(markup).toContain('Rebuild the set')
    // A set member is replaced, never emptied.
    expect(markup).not.toContain('Remove back view')
    expect(markup).toContain('Replace…')
  })

  it('names the missing role when a view failed during the composite', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...baseProps}
        subjectKind="biped"
        source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }}
        views={{ back: { id: 'back-id', name: 'back.png', kind: 'image' } }}
        sourceViewLabels={[]}
        tposeSetFront={front}
        tposeSet={{ ...membership, complete: false, failedRoles: ['left'] }}
      />,
    )

    expect(markup).toContain('missing its left view')
  })

  it('offers the approval as approving the SET, not one image', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls
        {...baseProps}
        subjectKind="biped"
        source={{ id: 'reference-id', name: 'punk.png', kind: 'image' }}
        views={{}}
        sourceViewLabels={[]}
        preparedCandidate={front}
        tposeSetFront={front}
        tposeSet={membership}
      />,
    )

    expect(markup).toContain('Approve this T-pose set')
    expect(markup).toContain('Approve the set')
    expect(markup).toContain('conditioned on exactly these four images')
  })

  it('keeps the object layout untouched', () => {
    const markup = renderToStaticMarkup(
      <GenerateControls {...legacyProps} source={{ id: 'reference-id', name: 'crate.png', kind: 'image' }} views={{}} sourceViewLabels={[]} />,
    )

    expect(markup).toContain('Source image')
    expect(markup).toContain('Auto-generate views')
    expect(markup).not.toContain('Canonical T-pose set')
    expect(markup).not.toContain('view-slot-canonical-tag')
  })
})

// Bead 8c9gf: the warning card on the offending tile.
describe('GenerateControls view-consistency notices', () => {
  const viewWith = (status: string, code?: string) => ({
    id: 'left-id',
    name: 'left.png',
    kind: 'image',
    metadata: {
      synthesized_view: 'left',
      view_consistency: {
        status,
        findings: code ? [{ code, severity: status === 'blocked' ? 'error' : 'warning', message: 'x' }] : [],
      },
    },
  })

  const render = (artifact: ReturnType<typeof viewWith>) => renderToStaticMarkup(
    <GenerateControls
      {...legacyProps}
      source={{ id: 'front-id', name: 'front.png', kind: 'image' }}
      views={{ left: artifact }}
      sourceViewLabels={[]}
    />,
  )

  it('names the offending view and offers the regenerate remedy', () => {
    const markup = render(viewWith('blocked', 'view_consistency.leg_separation_mismatch'))

    expect(markup).toContain('LEFT contradicts FRONT')
    expect(markup).toContain('legs not visible')
    expect(markup).toContain('view-slot-notice-error')
    expect(markup).toContain('Regenerate this view')
  })

  it('uses the softer tone for a decorrelation warning, with no regenerate demand', () => {
    const markup = render(viewWith('warning', 'view_consistency.lower_silhouette_decorrelated'))

    expect(markup).toContain('view-slot-notice-warning')
    expect(markup).toContain('LEFT differs from FRONT')
    expect(markup).not.toContain('Regenerate this view')
  })

  it('shows a neutral mini-tag when the check declined to score', () => {
    const markup = render(viewWith('not_evaluable', 'view_consistency.not_evaluable'))

    expect(markup).toContain('view-slot-notice-unknown')
    expect(markup).toContain('couldn&#x27;t verify')
    // No alarm styling: a refusal to score is not a defect in the view.
    expect(markup).not.toContain('view-slot-error')
    expect(markup).not.toContain('view-slot-notice-error')
  })

  it('shows no notice at all once the view is clean', () => {
    const markup = render(viewWith('clean'))

    expect(markup).toContain('view-slot-filled')
    expect(markup).not.toContain('view-slot-notice')
  })
})
