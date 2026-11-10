import { useEffect, useId, useRef, useState } from 'react'
import { CheckCircle2, ShieldCheck } from 'lucide-react'
import { API } from './api'
import JobProgress from './JobProgress'
import RigSkeletonViewport from './RigSkeletonViewport'
import RigAdjustViewport, { type JointPositionCollector } from './RigAdjustViewport'
import { hasActiveStudioJob } from './workspace'
import { WorkingOn, type WorkingOnAssetDetails } from './WorkingOn'

type SkeletonWorkspaceRun = { source_artifact_id: string; source_artifact_kind: string; attempts: Array<{ stage: string; status: string; output_artifact_id?: string | null; completed_at?: number | null; created_at?: number }> } | null

type Checkpoint = { artifact_id: string; note: string; completed_at: number }
type RecipeEnvelope = { recipe: { clean_mesh_artifact_id: string }; artifact_id: string }
type RigCertificate = { status: string; profile?: string; findings: Array<{ message: string }> }
type SkeletonJob = { id: string; kind: string; status: string; progress: number; error?: string | null; logs?: string[] } | null

function profileLabel(profile?: string): string {
  if (profile === 'chain') return 'Chain'
  if (profile === 'humanoid') return 'Humanoid'
  return profile || 'Unknown'
}

export type RigAdjustmentSave = { positions: Record<string, [number, number, number]>; removeJoints: string[] }

export default function SkeletonWorkspace({
  artifact, cleanMeshArtifactId, runId, checkpoint, onApprove, onSaveAdjustments, onGenerateAutoRig, job, busy, onCancel,
  run = null, pinnedArtifactId = null, onPinArtifact = () => {}, detailsById = {}, onUpload = () => {},
}: {
  artifact: { id: string; name: string; metadata?: { quality_decisions?: Record<string, RigCertificate> } }
  cleanMeshArtifactId: string | null
  runId: string
  checkpoint?: Checkpoint | null
  onApprove: (artifactId: string, note: string) => Promise<void>
  onSaveAdjustments: (adjustment: RigAdjustmentSave) => Promise<void>
  onGenerateAutoRig?: (targetArtifactId: string) => Promise<void>
  job?: SkeletonJob
  busy?: boolean
  onCancel?: () => void
  // I1: the Working-on card — this surface always has a rig candidate
  // already selected, so its picker offers other rig candidates on the run.
  run?: SkeletonWorkspaceRun
  pinnedArtifactId?: string | null
  onPinArtifact?: (artifactId: string) => void
  detailsById?: Record<string, WorkingOnAssetDetails>
  onUpload?: () => void
}) {
  const approved = checkpoint?.artifact_id === artifact.id
  const certificate = artifact.metadata?.quality_decisions?.['rig-certificate']
  const blocked = certificate?.status === 'blocked'
  const hasAppliedRig = Boolean(certificate)
  const rerunActive = hasActiveStudioJob(job) && (job?.kind === 'auto-rig' || job?.kind === 'rig-adjust')
  const [adjusting, setAdjusting] = useState(false)
  const [savingAdjust, setSavingAdjust] = useState(false)
  // The adjust viewport portals its controls into this element so they sit in
  // the right-hand control bar with the rest of the workspace's actions
  // instead of floating over the top-left of the canvas. It is state, not a
  // ref, so the viewport re-renders once the host element exists.
  const [adjustControlsHost, setAdjustControlsHost] = useState<HTMLDivElement | null>(null)
  const collectRef = useRef<JointPositionCollector | null>(null)
  const [note, setNote] = useState('Skeleton alignment and rest pose reviewed.')
  const [saving, setSaving] = useState(false)
  const [draftStatus, setDraftStatus] = useState('Loading saved alignment…')
  const [error, setError] = useState('')
  const frame = useRef<HTMLIFrameElement>(null)
  const editorId = useId()
  const session = `${runId}:${artifact.id}:${editorId}`
  const cleanMeshId = cleanMeshArtifactId ?? ''
  const context = { run: runId, cleanMeshArtifactId: cleanMeshId, session }
  // Switching the journey selector swaps `artifact` under this workspace
  // without unmounting it — drop back out of the adjust sub-view so it
  // doesn't keep showing RigAdjustViewport (and its now-stale removed/
  // adjusted bookkeeping) wired to the artifact it was opened for.
  useEffect(() => { setAdjusting(false) }, [artifact.id])
  useEffect(() => {
    const receiveRecipe = async (event: MessageEvent) => {
      if (event.origin !== window.location.origin || event.source !== frame.current?.contentWindow ||
        event.data?.source !== 'aismith-mesh2motion' ||
        event.data?.run !== context.run || event.data?.cleanMeshArtifactId !== context.cleanMeshArtifactId ||
        event.data?.session !== context.session) return
      if (event.data?.type === 'joint-recipe-ready') { await restoreRecipe(); return }
      if (event.data?.type !== 'joint-recipe-changed') return
      setDraftStatus('Saving alignment…')
      try {
        const response = await fetch(`${API}/runs/${runId}/joint-recipe`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(event.data.recipe),
        })
        if (!response.ok) throw new Error(`Draft save failed (${response.status})`)
        const saved = await response.json() as RecipeEnvelope
        setDraftStatus('Alignment saved')
        frame.current?.contentWindow?.postMessage({ source: 'aismith-studio', type: 'joint-recipe-saved', ...context, artifactId: saved.artifact_id }, window.location.origin)
      } catch (reason) {
        setDraftStatus(reason instanceof Error ? reason.message : 'Could not save alignment')
      }
    }
    window.addEventListener('message', receiveRecipe)
    return () => window.removeEventListener('message', receiveRecipe)
  }, [artifact.id, runId, session])
  const restoreRecipe = async () => {
    try {
      const response = await fetch(`${API}/runs/${runId}/joint-recipe`)
      if (response.status === 404) {
        setDraftStatus('Saving baseline alignment…')
        frame.current?.contentWindow?.postMessage({ source: 'aismith-studio', type: 'joint-recipe-capture', ...context }, window.location.origin)
        return
      }
      if (!response.ok) throw new Error(`Draft load failed (${response.status})`)
      const saved = await response.json() as RecipeEnvelope
      if (saved.recipe.clean_mesh_artifact_id !== cleanMeshId) throw new Error('Saved alignment targets another mesh')
      frame.current?.contentWindow?.postMessage({ source: 'aismith-studio', type: 'joint-recipe-restore', ...context, recipe: saved.recipe }, window.location.origin)
      setDraftStatus('Alignment restored')
    } catch (reason) {
      setDraftStatus(reason instanceof Error ? reason.message : 'Could not restore alignment')
    }
  }
  const saveAdjustments = async () => {
    const collected = collectRef.current?.()
    if (!collected || (!Object.keys(collected.positions).length && !collected.removeJoints.length)) {
      setError('No joint adjustments to save — drag a joint, or remove one, first.')
      return
    }
    setSavingAdjust(true); setError('')
    try { await onSaveAdjustments(collected); setAdjusting(false) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save adjustments') }
    finally { setSavingAdjust(false) }
  }
  const approve = async () => {
    setSaving(true); setError('')
    try { await onApprove(artifact.id, note.trim()) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not approve skeleton') }
    finally { setSaving(false) }
  }
  return <section className="skeleton-workspace">
    <div className="skeleton-editor">{hasAppliedRig && adjusting
      ? <RigAdjustViewport artifactId={artifact.id} collectRef={collectRef} controlsHost={adjustControlsHost} />
      : hasAppliedRig
      ? <RigSkeletonViewport artifactId={artifact.id} />
      : <><div className="mesh2motion-label">Manual fallback · template rigging in Mesh2Motion · edit joints, inspect weights, then bind once <span>{draftStatus}</span></div>{cleanMeshId ? <iframe ref={frame} className="mesh2motion-frame" title="Skeleton editor" src={`/mesh2motion.html?source=${encodeURIComponent(`${API}/assets/${cleanMeshId}/download`)}&extension=glb&mode=edit-clean-mesh&run=${encodeURIComponent(runId)}&input=${encodeURIComponent(cleanMeshId)}&session=${encodeURIComponent(session)}`} /> : <div className="mesh2motion-overlay" role="alert">This rig has no immutable clean-mesh provenance. Rebuild it from the Clean stage; rebinding a skinned GLB is blocked.</div>}</>}</div>
    <aside className="skeleton-inspector">
      <WorkingOn stage="skeleton" stageLabel="Skeleton" run={run} current={artifact} detailsById={detailsById} pinnedArtifactId={pinnedArtifactId} onPin={onPinArtifact} onUpload={onUpload} />
      <span className="eyebrow">CURRENT RIG ARTIFACT</span>
      <h1>Skeleton</h1>
      <code>{artifact.id}</code>
      {certificate?.profile && <span className={`skeleton-profile-chip skeleton-profile-${certificate.profile}`}>{profileLabel(certificate.profile)}</span>}
      <p>{certificate?.profile === 'chain' ? 'Check that the spine chain follows the body from tail to head.' : 'Inspect the textured mesh and applied skeleton together. Check shoulders, elbows, wrists, hips, knees, ankles, neck, and hands.'}</p>
      <div className={`skeleton-validation ${approved ? 'approved' : ''}`}>
        {approved ? <CheckCircle2 size={18} /> : <ShieldCheck size={18} />}
        <div><strong>{approved ? 'Approved' : 'Awaiting review'}</strong><small>{approved ? 'This exact rig is unlocked for animation.' : 'Animation stays locked until this exact artifact is approved.'}</small></div>
      </div>
      {rerunActive && <JobProgress label={job?.kind === 'rig-adjust' ? 'Rebinding with your adjustments…' : 'Re-running auto-rig…'} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={onCancel} />}
      {hasAppliedRig && !approved && !adjusting && !rerunActive && (
        <div className="skeleton-repair-actions">
          <button className="secondary-button" onClick={() => setAdjusting(true)}>Adjust joints</button>
          {onGenerateAutoRig && cleanMeshArtifactId && (
            <button className="secondary-button" onClick={() => onGenerateAutoRig(cleanMeshArtifactId)} disabled={Boolean(busy)}>Re-run auto-rig</button>
          )}
        </div>
      )}
      {job?.kind === 'auto-rig' && job.status === 'failed' && <span className="motion-error">{job.error || 'The UniRig auto-rig failed'}</span>}
      {adjusting && (
        <div className="skeleton-adjust-panel">
          <span className="eyebrow">ADJUST JOINTS</span>
          <div className="skeleton-adjust-controls" ref={setAdjustControlsHost} />
          <div className="skeleton-adjust-actions">
            <button className="primary-action" onClick={saveAdjustments} disabled={savingAdjust}>
              {savingAdjust ? 'Rebinding…' : 'Save & rebind'}
            </button>
            <button className="secondary-button" onClick={() => setAdjusting(false)} disabled={savingAdjust}>Cancel</button>
          </div>
        </div>
      )}
      {blocked && <div className="skeleton-blockers"><strong>Fix before approval</strong>{certificate.findings.map((finding) => <span key={finding.message}>{finding.message}</span>)}</div>}
      <label>Review note<textarea value={note} onChange={(event) => setNote(event.target.value)} disabled={approved} /></label>
      <button className="primary-action" onClick={approve} disabled={blocked || approved || saving || adjusting || savingAdjust || !note.trim()}>{saving ? 'Saving…' : approved ? 'Skeleton approved' : 'Approve skeleton'}</button>
      {blocked && <small className="motion-error">Blocked by {certificate.findings.length} certificate finding{certificate.findings.length === 1 ? '' : 's'} — adjust joints or re-run auto-rig.</small>}
      {error && <span className="motion-error">{error}</span>}
    </aside>
  </section>
}
