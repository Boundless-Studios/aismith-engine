import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import SkeletonWorkspace from './SkeletonWorkspace'

describe('SkeletonWorkspace', () => {
  it('reviews and approves the already selected rig without a load-skeleton step', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-1', name: 'mary-rig.glb' }}
      cleanMeshArtifactId="mesh-1"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
    />)

    expect(markup).toContain('Skeleton')
    expect(markup).toContain('rig-1')
    expect(markup).toContain('Approve skeleton')
    expect(markup).not.toMatch(/load skeleton/i)
  })

  it('offers approval of the exact retained rig it displays when the certificate only needs review', () => {
    // The P1 case: a fresh 66-bone auto-rig lands `needs_review` (advisory
    // finding), so it is retained and never becomes the run's current
    // artifact. The card still names that exact rig and still offers the
    // approve action -- the approval posts this id, not the run pointer.
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'retained-rig-1', name: 'lyra-rig.glb', metadata: { quality_decisions: { 'rig-certificate': { status: 'needs_review', profile: 'humanoid', findings: [{ message: 'Optional finger joints are unused.' }] } } } }}
      cleanMeshArtifactId="mesh-1"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
    />)

    expect(markup).toContain('retained-rig-1')
    expect(markup).toContain('Animation stays locked until this exact artifact is approved.')
    expect(markup).toContain('Approve skeleton')
    expect(markup).not.toContain('disabled')
  })

  it('shows approval only when the checkpoint targets the visible artifact', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-2', name: 'mary-rig.glb' }}
      cleanMeshArtifactId="mesh-2"
      runId="run-1"
      checkpoint={{ artifact_id: 'rig-2', note: 'Reviewed', completed_at: 42 }}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
    />)

    expect(markup).toContain('This exact rig is unlocked for animation.')
    expect(markup).toContain('Skeleton approved')
  })

  it('surfaces a blocked rig certificate instead of allowing approval', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-3', name: 'mary-rig.glb', metadata: { quality_decisions: { 'rig-certificate': { status: 'blocked', findings: [{ message: 'Thumb joints have no weights.' }] } } } }}
      cleanMeshArtifactId="mesh-3"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
    />)
    expect(markup).toContain('Thumb joints have no weights.')
    expect(markup).toContain('Approve skeleton')
    expect(markup).not.toContain('Rig repair required')
    expect(markup).toContain('Blocked by 1 certificate finding — adjust joints or re-run auto-rig.')
    expect(markup).toContain('disabled')
  })

  it('offers a way forward — re-run auto-rig and adjust joints — even when the certificate blocks approval', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-4', name: 'serpent-rig.glb', metadata: { quality_decisions: { 'rig-certificate': { status: 'blocked', profile: 'humanoid', findings: [{ message: 'A humanoid skin with at least 15 joints is required.' }] } } } }}
      cleanMeshArtifactId="mesh-4"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
      onGenerateAutoRig={vi.fn(async () => {})}
    />)

    expect(markup).toContain('Adjust joints')
    expect(markup).toContain('Re-run auto-rig')
  })

  it('shows the chain profile chip, the chain-specific inspection copy, and the profile-aware findings', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-5', name: 'serpent-rig.glb', metadata: { quality_decisions: { 'rig-certificate': { status: 'blocked', profile: 'chain', findings: [{ message: 'a single joint controls too much of the mesh' }] } } } }}
      cleanMeshArtifactId="mesh-5"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
      onGenerateAutoRig={vi.fn(async () => {})}
    />)

    expect(markup).toContain('Chain')
    expect(markup).toContain('Check that the spine chain follows the body from tail to head.')
    expect(markup).toContain('a single joint controls too much of the mesh')
    expect(markup).not.toContain('Check shoulders, elbows')
  })

  it('shows re-run progress instead of a blank state while auto-rig is re-running', () => {
    const markup = renderToStaticMarkup(<SkeletonWorkspace
      artifact={{ id: 'rig-6', name: 'serpent-rig.glb', metadata: { quality_decisions: { 'rig-certificate': { status: 'blocked', profile: 'chain', findings: [{ message: 'a single joint controls too much of the mesh' }] } } } }}
      cleanMeshArtifactId="mesh-6"
      runId="run-1"
      checkpoint={null}
      onApprove={vi.fn()} onSaveAdjustments={vi.fn(async () => {})}
      onGenerateAutoRig={vi.fn(async () => {})}
      job={{ id: 'job-1', kind: 'auto-rig', status: 'running', progress: 55, logs: ['Binding bone-heat weights'] }}
    />)

    expect(markup).toContain('55%')
    expect(markup).toContain('Binding bone-heat weights')
    expect(markup).not.toContain('Re-run auto-rig')
  })
})
