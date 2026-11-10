import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { WorkingOn } from './WorkingOn'

describe('WorkingOn', () => {
  const run = {
    source_artifact_id: 'source-image',
    source_artifact_kind: 'image',
    attempts: [
      { stage: 'generate', status: 'completed', output_artifact_id: 'mesh-1', completed_at: 1 },
      { stage: 'retopo', status: 'completed', output_artifact_id: 'clean-1', completed_at: 2 },
    ],
  }

  it('shows the current artifact and its size when only one candidate exists', () => {
    // A GLB-sourced (rig-only) run with no other rig attempts yet: the
    // current rig candidate is the only compatible option, so no picker.
    const markup = renderToStaticMarkup(
      <WorkingOn
        stage="skeleton"
        stageLabel="Skeleton"
        run={{ source_artifact_id: 'source-glb', source_artifact_kind: 'glb', attempts: [] }}
        current={{ id: 'mesh-1', name: 'mesh-1.glb', kind: 'glb', size_bytes: 2 * 1024 * 1024 }}
        detailsById={{}}
        pinnedArtifactId={null}
        onPin={vi.fn()}
        onUpload={vi.fn()}
      />,
    )

    expect(markup).toContain('mesh-1.glb')
    expect(markup).toContain('2.0 MB')
    expect(markup).not.toContain('working-on-picker')
  })

  it('offers a picker once the run has more than one compatible candidate', () => {
    const markup = renderToStaticMarkup(
      <WorkingOn
        stage="skeleton-pre-rig"
        stageLabel="Skeleton"
        run={run}
        current={{ id: 'clean-1', name: 'clean-1.glb', kind: 'glb', size_bytes: 1_000_000 }}
        detailsById={{ 'clean-1': { vertices: 12000 } }}
        pinnedArtifactId={null}
        onPin={vi.fn()}
        onUpload={vi.fn()}
      />,
    )

    expect(markup).toContain('working-on-picker')
    expect(markup).toContain('12,000 verts')
  })

  it('binds the select to the current working-on artifact even when it is not one of the stage\'s own input options', () => {
    // Reproduces the reported defect: Clean's own options are Generate
    // outputs, but `current` can already be a Clean (retopo) output once
    // cleaning has run once — the select must still show that as selected,
    // as the first option, not silently fall back to its own first entry.
    const markup = renderToStaticMarkup(
      <WorkingOn
        stage="clean"
        stageLabel="Clean"
        run={run}
        current={{ id: 'clean-1', name: 'Retopo result.glb', kind: 'glb', size_bytes: 36_800_000 }}
        detailsById={{ 'mesh-1': { name: 'Generate result.glb' } }}
        pinnedArtifactId={null}
        onPin={vi.fn()}
        onUpload={vi.fn()}
      />,
    )

    expect(markup).toContain('Retopo result.glb')
    const selectMatch = markup.match(/<select[^>]*>(.*?)<\/select>/)
    expect(selectMatch).not.toBeNull()
    const selectMarkup = selectMatch![1]
    const optionValues = [...selectMarkup.matchAll(/<option value="([^"]+)"/g)].map((match) => match[1])
    expect(optionValues[0]).toBe('clean-1')
    const selectedOption = selectMarkup.match(/<option value="([^"]+)"[^>]*selected/)
    expect(selectedOption?.[1]).toBe('clean-1')
  })

  it('shows the elapsed time when provided', () => {
    const markup = renderToStaticMarkup(
      <WorkingOn
        stage="generate"
        stageLabel="Generate"
        run={null}
        current={null}
        detailsById={{}}
        pinnedArtifactId={null}
        onPin={vi.fn()}
        onUpload={vi.fn()}
        elapsedMs={65_000}
      />,
    )

    expect(markup).toContain('01:05')
    expect(markup).toContain('No artifact selected')
  })
})
