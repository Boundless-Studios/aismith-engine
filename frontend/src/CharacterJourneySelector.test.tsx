import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import CharacterJourneySelector from './CharacterJourneySelector'
import type { RunManifest } from './history'

function run(id: string, name: string): RunManifest {
  return {
    id,
    name,
    status: 'completed',
    source_artifact_id: `${id}-source`,
    source_artifact_kind: 'image',
    review_status: 'unreviewed',
    review_note: '',
    attempts: [],
    created_at: 1,
    updated_at: 2,
  }
}

describe('CharacterJourneySelector', () => {
  it('renders durable journeys, the active journey, and a new-character action', () => {
    const markup = renderToStaticMarkup(<CharacterJourneySelector
      runs={[run('mary', 'Mary'), run('other', 'Other hero')]}
      activeRunId="mary"
      loading={false}
      onSelect={vi.fn()}
      onNew={vi.fn()}
    />)

    expect(markup).toContain('Character journey')
    expect(markup).toContain('<option value="mary" selected="">Mary</option>')
    expect(markup).toContain('<option value="other">Other hero</option>')
    expect(markup).toContain('New character')
  })

  it('hides an archived run but keeps it if it is the one currently open', () => {
    const hidden = renderToStaticMarkup(<CharacterJourneySelector
      runs={[run('mary', 'Mary'), run('other', 'Other hero')]}
      activeRunId={null}
      loading={false}
      onSelect={vi.fn()}
      onNew={vi.fn()}
      archivedRunIds={new Set(['other'])}
    />)
    expect(hidden).not.toContain('Other hero')
    expect(hidden).toContain('Mary')

    const keptWhenOpen = renderToStaticMarkup(<CharacterJourneySelector
      runs={[run('mary', 'Mary'), run('other', 'Other hero')]}
      activeRunId="other"
      loading={false}
      onSelect={vi.fn()}
      onNew={vi.fn()}
      archivedRunIds={new Set(['other'])}
    />)
    expect(keptWhenOpen).toContain('Other hero')
  })
})
