import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

// StudioViewport registers the <model-viewer> custom element as an import
// side effect; that registration needs a browser DOM (HTMLElement,
// customElements) this suite's Node environment does not provide.
// renderToStaticMarkup below never mounts StudioViewport's effects, so
// stubbing the registration itself is enough to render its static markup.
vi.mock('@google/model-viewer', () => ({}))

import { RigAnimateWorkspace } from './App'

const noop = () => {}
const asyncNoop = async () => {}

describe('RigAnimateWorkspace — unrigged (authoring) surface', () => {
  const baseProps = {
    runId: 'run-1',
    source: { id: 'clean-mary', name: 'clean-mary.glb', kind: 'glb', size_bytes: 4096 },
    onApprove: vi.fn(asyncNoop),
    onReview: vi.fn(asyncNoop),
    onUpload: vi.fn(noop),
    onGenerateMotion: vi.fn(noop),
    onGenerateAutoRig: vi.fn(asyncNoop),
    onSaveAdjustments: vi.fn(asyncNoop),
    onAttachCuratedMotion: vi.fn(asyncNoop),
    kimodoReady: false,
    onCancel: vi.fn(noop),
    motionReady: false,
    blenderAvailable: true,
    busy: false,
    job: null,
    terminalHydrating: false,
    terminalHydrationError: '',
    onRetryTerminalHydration: vi.fn(noop),
  }

  it('shows the source mesh viewport and inspector instead of a bare toolbar', () => {
    const markup = renderToStaticMarkup(<RigAnimateWorkspace
      {...baseProps}
      mode="rig"
      unirigReady={true}
      modelDetails={{ vertices: 12345, triangles: 6789, materials: 1, meshes: 1, format: 'GLB', size_bytes: 1024 }}
    />)

    expect(markup).toContain('rig-authoring-viewport')
    expect(markup).toContain('viewport')
    expect(markup).toContain('clean-mary.glb')
    expect(markup).toContain('12,345')
    expect(markup).toContain('6,789')
    expect(markup).toContain('Auto-rig character')
    expect(markup).not.toContain('mesh2motion-fallback')
    expect(markup).not.toMatch(/manual fallback/i)
  })

  it('keeps the mesh visible and shows job progress instead of a blank overlay while auto-rig runs', () => {
    const markup = renderToStaticMarkup(<RigAnimateWorkspace
      {...baseProps}
      mode="rig"
      unirigReady={true}
      job={{ id: 'job-1', kind: 'auto-rig', submission_id: 's1', status: 'running', progress: 42, message: '', logs: ['Binding bone-heat weights and exporting'], warnings: [] }}
    />)

    expect(markup).toContain('rig-authoring-viewport')
    expect(markup).toContain('42%')
    expect(markup).toContain('Binding bone-heat weights and exporting')
    expect(markup).not.toContain('Automated rig in progress')
    expect(markup).not.toContain('Auto-rig character')
  })

  it('shows a checking-worker hint until capabilities have loaded', () => {
    const markup = renderToStaticMarkup(<RigAnimateWorkspace {...baseProps} mode="rig" unirigReady={null} />)

    expect(markup).toContain('Checking local worker')
    expect(markup).not.toContain('Auto-rig character')
  })
})

// N5: the Animate panel's "Attach motion" button routes to apply-clip (no
// BVH export, no retarget math) for a certified, template-conformant rig,
// and stays on the legacy retarget-motion path otherwise. App.tsx's
// RigAnimateWorkspace computes that decision once (isTemplateConformantRig)
// from the pinned rig's own metadata and threads it into AnimatePanel only
// as a label -- onAttachCuratedMotion itself (mocked here) does the actual
// routing, so this only has to prove the label the operator/tester sees
// tracks the rig's conformance the same way the routing logic does.
describe('RigAnimateWorkspace — rigged (animate) surface, N5 direct-attach routing', () => {
  const certifiedRigger = { status: 'ready', policy_version: 2, profile: 'humanoid' }
  const riggedProps = {
    runId: 'run-1',
    onApprove: vi.fn(asyncNoop),
    onReview: vi.fn(asyncNoop),
    onUpload: vi.fn(noop),
    onGenerateMotion: vi.fn(noop),
    onGenerateAutoRig: vi.fn(asyncNoop),
    onSaveAdjustments: vi.fn(asyncNoop),
    onAttachCuratedMotion: vi.fn(asyncNoop),
    kimodoReady: false,
    onCancel: vi.fn(noop),
    motionReady: false,
    blenderAvailable: true,
    busy: false,
    job: null,
    terminalHydrating: false,
    terminalHydrationError: '',
    onRetryTerminalHydration: vi.fn(noop),
    unirigReady: true,
    mode: 'animate' as const,
  }

  it('labels the attach button "Attach (direct)" for a certified, template-conformant rig', () => {
    const markup = renderToStaticMarkup(<RigAnimateWorkspace
      {...riggedProps}
      source={{
        id: 'conformant-rig', name: 'mary.glb', kind: 'glb', size_bytes: 4096, stage: 'auto-rig',
        metadata: {
          quality_decisions: { 'rig-certificate': certifiedRigger },
          rig_quality: { template_conformance: [] },
        },
      }}
    />)

    // "Attach motion" is also the static "NEW ANIMATION" section heading
    // (unaffected by conformance), so the assertion targets the submit
    // button's own closing text specifically, not the whole markup.
    expect(markup).toContain('Attach (direct)</button>')
    expect(markup).not.toContain('Attach motion</button>')
  })

  it('keeps the legacy "Attach motion" label for a rig with no template-conformance data', () => {
    const markup = renderToStaticMarkup(<RigAnimateWorkspace
      {...riggedProps}
      source={{
        id: 'legacy-rig', name: 'mary.glb', kind: 'glb', size_bytes: 4096, stage: 'auto-rig',
        metadata: {
          quality_decisions: { 'rig-certificate': certifiedRigger },
        },
      }}
    />)

    expect(markup).toContain('Attach motion')
    expect(markup).not.toContain('Attach (direct)')
  })
})
