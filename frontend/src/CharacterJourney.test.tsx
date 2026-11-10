import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import CharacterJourney from './CharacterJourney'
import type { RunManifest } from './history'

describe('CharacterJourney', () => {
  it('keeps verbose stage guidance in an on-demand disclosure', () => {
    const markup = renderToStaticMarkup(
      <CharacterJourney
        run={null}
        activeTab="generate"
        onNavigate={vi.fn()}
      />,
    )

    expect(markup).toContain('<details class="journey-detail')
    expect(markup).toContain('<summary')
    expect(markup).not.toContain('class="journey-heading"')
  })

  it('offers an enabled Next affordance for a reachable step not yet on its own tab', () => {
    // Clean has no attempts yet and its tab isn't open, so characterJourney marks
    // it "ready" (not the dead-looking "upcoming") — with the tool tabs gone, the
    // strip button and Next are the only way to reach a stage before its tab is
    // ever visited, so both must render enabled.
    const run = {
      review_status: 'unreviewed',
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'generate', status: 'completed' },
      ],
    } as unknown as RunManifest

    const markup = renderToStaticMarkup(
      <CharacterJourney run={run} activeTab="generate" onNavigate={vi.fn()} />,
    )

    expect(markup).toContain('next-step-button')
    expect(markup).toContain('Next: Clean')
    expect(markup).not.toMatch(/class="primary-action next-step-button" disabled=""/)
    expect(markup).toContain('class="journey-step ready ')
  })

  it('offers an enabled Next affordance for a step the journey already flags as needing attention', () => {
    const run = {
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
    } as unknown as RunManifest

    const markup = renderToStaticMarkup(
      <CharacterJourney run={run} activeTab="rig" onNavigate={vi.fn()} />,
    )

    expect(markup).toContain('Next: Skeleton')
    expect(markup).not.toMatch(/class="primary-action next-step-button" disabled=""/)
  })

  it('defaults the disclosure to the actionable Next step, not the bare last (unreachable) one', () => {
    // The operator's actual run: image-sourced, only a completed `generate`
    // attempt (no `prepare-image` attempt at all). Nothing is active/checkpoint/
    // failed, so the old "steps[length - 1]" fallback landed on Review — still
    // "upcoming" and unreachable on this run.
    const run = {
      review_status: 'unreviewed',
      attempts: [{ stage: 'generate', status: 'completed' }],
    } as unknown as RunManifest

    const markup = renderToStaticMarkup(
      <CharacterJourney run={run} activeTab="generate" onNavigate={vi.fn()} />,
    )

    expect(markup).toContain('<summary><strong>Clean</strong>')
    expect(markup).not.toContain('<summary><strong>Review</strong>')
  })

  it('renders no Next affordance once the final stage is complete', () => {
    const run = {
      review_status: 'accepted',
      reviewed_artifact_id: 'animated-1',
      current_artifact_id: 'animated-1',
      attempts: [{ stage: 'deformation-review', status: 'completed' }],
    } as unknown as RunManifest

    const markup = renderToStaticMarkup(
      <CharacterJourney run={run} activeTab="rig" onNavigate={vi.fn()} />,
    )

    expect(markup).not.toContain('next-step-button')
  })

  it('explains a disabled step instead of a silent dead-click, when the gate is legitimate', () => {
    // The operator's second report: on the "zix" run (synthesized views but
    // Generate never run), Clean shows as unclickable with no explanation.
    // The disabled strip button must carry a title tooltip naming why.
    const run = {
      review_status: 'unreviewed',
      attempts: [
        { stage: 'prepare-image', status: 'completed' },
        { stage: 'synthesize-views', status: 'completed' },
      ],
    } as unknown as RunManifest

    const markup = renderToStaticMarkup(
      <CharacterJourney run={run} activeTab="generate" onNavigate={vi.fn()} />,
    )

    expect(markup).toContain('class="journey-step upcoming ')
    expect(markup).toContain('title="Complete Generate first."')
  })
})
