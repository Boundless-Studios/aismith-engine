import { useEffect, useState } from 'react'
import { Download, LoaderCircle, ShieldCheck, Sparkles, X } from 'lucide-react'
import JobProgress from './JobProgress'
import { WorkingOn, type WorkingOnAssetDetails } from './WorkingOn'
import { AnimationList } from './AnimationList'
import { ClipBrowser } from './ClipBrowser'
import { CURATED_MOTION_CLIPS, fetchCuratedMotionClips, journeyAnimations, type CuratedMotionClip } from './animations'
import { API } from './api'

type AnimatePanelAttempt = { stage: string; status: string; output_artifact_id?: string | null; settings?: Record<string, unknown>; promoted?: boolean; completed_at?: number | null; created_at?: number }
type AnimatePanelRun = { source_artifact_id: string; source_artifact_kind: string; review_status?: string; reviewed_artifact_id?: string | null; attempts: AnimatePanelAttempt[] } | null
type AnimatePanelSource = { id: string; name: string; kind: string; stage?: string }
type AnimatePanelJob = { kind: string; status: string; progress: number; message: string; logs: string[]; warnings: string[]; error?: string | null; promoted_output?: boolean } | null
type AnimatePanelExportState = { status: 'idle' | 'exporting' | 'saved' | 'downloaded' | 'error'; path?: string; error?: string }

// The Animate tab's whole right-hand aside — one surface, no Playback/
// Animation-library toggle (BOU: "I don't think we need a separate Review
// tab; I should be able to accept/save/export an animation inline in
// Animate"). Top to bottom: which rig is being animated, the list of every
// clip already embedded on it, accept/reject/export/download for whichever
// one is currently selected, and the controls to attach or bake a new one.
export function AnimatePanel({
  run, source, detailsById, pinnedArtifactId, onPinArtifact, onUpload,
  job, busy, isChainRig, isConformantRig = false,
  onGenerateMotion, onAttachCuratedMotion, kimodoReady,
  onReview, reviewStatus, reviewedArtifactId,
  onExport, exportState,
  onCancel, terminalHydrationError, onRetryTerminalHydration,
  onPreviewClip = () => {},
}: {
  run: AnimatePanelRun
  source: AnimatePanelSource
  detailsById: Record<string, WorkingOnAssetDetails>
  pinnedArtifactId: string | null
  onPinArtifact: (artifactId: string) => void
  onUpload: () => void
  job: AnimatePanelJob
  busy: boolean
  isChainRig: boolean
  // N5: true when the pinned rig is certified AND template-conformant
  // (App.tsx's isTemplateConformantRig) -- decides only the "Attach motion"
  // button's label here (App.tsx's onAttachCuratedMotion itself already
  // routes to apply-clip vs. the legacy retarget path based on the same
  // check). Defaults to false so every other caller/test is unaffected.
  isConformantRig?: boolean
  onGenerateMotion: () => void
  onAttachCuratedMotion: (clipAnimation: string) => Promise<void>
  kimodoReady: boolean
  onReview: (artifactId: string, status: 'accepted' | 'rejected', note: string) => Promise<void>
  reviewStatus?: 'unreviewed' | 'accepted' | 'rejected'
  reviewedArtifactId?: string | null
  onExport: (artifactId: string) => void
  exportState: AnimatePanelExportState
  onCancel: () => void
  terminalHydrationError: string
  onRetryTerminalHydration: () => void
  // Fires with the picker's current selection (or null) so the main
  // viewport can instantly preview it on the loaded rig — operator: "with
  // mesh2motion, changing clips doesn't require regeneration, it
  // immediately applies using the skeleton". Only meaningful for the
  // "Attach motion" section (a biped rig with no clip already selected);
  // suppressed for a chain rig (no curated clips apply) or while an
  // already-embedded clip is the current selection (that clip is already
  // playing for real, so a live-retargeted stand-in would fight it).
  onPreviewClip?: (clip: CuratedMotionClip | null) => void
}) {
  // CURATED_MOTION_CLIPS is the fallback shown while GET /api/animations/curated
  // is loading, or if it fails — the picker must never render an empty select.
  const [curatedClips, setCuratedClips] = useState<CuratedMotionClip[]>(CURATED_MOTION_CLIPS)
  const [selectedClipId, setSelectedClipId] = useState(CURATED_MOTION_CLIPS[0].id)
  const [reviewing, setReviewing] = useState(false)
  const [reviewError, setReviewError] = useState('')
  const animations = journeyAnimations(run, isChainRig)
  const isSelectedClip = source.stage === 'retarget-motion' || source.stage === 'generate-motion' || source.stage === 'apply-clip'
  const motionJobActive = Boolean(job && ['queued', 'running'].includes(job.status))
  const reviewed = isSelectedClip && reviewedArtifactId === source.id ? reviewStatus ?? 'unreviewed' : 'unreviewed'

  useEffect(() => {
    let active = true
    fetchCuratedMotionClips(API)
      .then((clips) => {
        if (!active || clips.length === 0) return
        setCuratedClips(clips)
        setSelectedClipId((current) => (clips.some((clip) => clip.id === current) ? current : clips[0].id))
      })
      .catch(() => { /* keep the hardcoded fallback list above */ })
    return () => { active = false }
  }, [])

  // No cleanup tied to these deps -- RetargetPreview's own load+retarget is
  // expensive (network fetch, GLTFLoader parse, SkeletonUtils compute), so
  // clearing-then-resetting the preview on every dependency wobble would
  // tear it down and restart from frame 0 on every recompute, not just a
  // real selection change (this is exactly what happened before: it looked
  // "frozen" because it kept restarting before ever animating past its
  // first couple of frames). Full clearing still happens on unmount, below
  // -- but the moment this effect genuinely re-runs *into* isChainRig or
  // isSelectedClip (an already-embedded clip just got pinned from the
  // Animations list, with no tab leave+return in between), it must also
  // explicitly null the preview here, not just skip setting a new one.
  // Leaving the stale curated-clip quick-preview in place is exactly what
  // produced the reported black/frozen viewport: StudioViewport keeps
  // rendering the abandoned RetargetPreview instance instead of ever
  // mounting <model-viewer> for the real embedded clip, indefinitely, since
  // nothing else in this component ever unmounts to trigger the cleanup.
  useEffect(() => {
    if (isChainRig || isSelectedClip) { onPreviewClip(null); return }
    onPreviewClip(curatedClips.find((clip) => clip.id === selectedClipId) ?? null)
  }, [onPreviewClip, isChainRig, isSelectedClip, curatedClips, selectedClipId])

  useEffect(() => () => onPreviewClip(null), [onPreviewClip])

  const submitReview = async (status: 'accepted' | 'rejected') => {
    const prompted = window.prompt(status === 'accepted' ? 'What did you verify across the full deformation suite?' : 'What failed visually?')
    if (prompted === null) return
    const note = prompted.trim()
    if (!note) { setReviewError('Describe what you inspected or what failed before saving this decision.'); return }
    setReviewing(true); setReviewError('')
    try { await onReview(source.id, status, note) } catch (error) { setReviewError(error instanceof Error ? error.message : 'Could not save the review.') } finally { setReviewing(false) }
  }

  return <aside className="rig-authoring-inspector">
    <WorkingOn stage="animate" stageLabel="Animate" run={run} current={source} detailsById={detailsById} pinnedArtifactId={pinnedArtifactId} onPin={onPinArtifact} onUpload={onUpload} />
    <AnimationList animations={animations} selectedArtifactId={source.id} onSelect={onPinArtifact} detailsById={detailsById} />
    {isSelectedClip && <div className="animate-clip-actions">
      <div className="rig-curated-copy"><span className="eyebrow">SELECTED CLIP</span><strong>Accept, export, or download</strong></div>
      <div className="rig-review-actions">
        <button className="primary-action" disabled={reviewing || reviewed === 'accepted'} onClick={() => submitReview('accepted')}>
          <ShieldCheck size={16} />{reviewed === 'accepted' ? 'Accepted' : reviewing ? 'Saving…' : 'Accept animation'}
        </button>
        <button className="secondary-button" disabled={reviewing} onClick={() => submitReview('rejected')}><X size={16} /> Reject</button>
      </div>
      {reviewed === 'rejected' && <small className="motion-error">This candidate was rejected. Attach or bake a replacement below.</small>}
      {reviewError && <small className="motion-error">{reviewError}</small>}
      <div className="export-glb-group">
        <div className="export-glb-row">
          <button className="secondary-button" onClick={() => onExport(source.id)} disabled={exportState.status === 'exporting'}>
            {exportState.status === 'exporting' ? <LoaderCircle className="spin" size={14} /> : <Download size={14} />}
            {exportState.status === 'exporting' ? 'Saving…' : 'Export GLB'}
          </button>
          <a className="secondary-button" href={`${API}/assets/${source.id}/download`}><Download size={14} /> Download</a>
        </div>
        {exportState.status === 'saved' && exportState.path && <small className="export-status export-status-ok">Saved to <code>{exportState.path}</code></small>}
        {exportState.status === 'downloaded' && <small className="export-status export-status-ok">No output folder configured — downloaded via your browser instead.</small>}
        {exportState.status === 'error' && exportState.error && <small className="export-status export-status-error">{exportState.error}</small>}
      </div>
    </div>}
    <div className="rig-curated-copy"><span className="eyebrow">NEW ANIMATION</span><strong>{isChainRig ? 'Procedural motion' : 'Attach motion'}</strong></div>
    {motionJobActive ? (
      <JobProgress label={job?.kind === 'generate-motion' ? (isChainRig ? 'Baking slither…' : 'Generating motion…') : 'Attaching motion…'} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={onCancel} />
    ) : isChainRig ? (
      <>
        <small>Bakes an S-curve travel cycle along the spine chain with Blender.</small>
        <button className="primary-action" onClick={onGenerateMotion} disabled={busy || motionJobActive}>
          <Sparkles size={16} /> Bake slither cycle
        </button>
      </>
    ) : (
      <>
        {/* The grid + filter IS the picker (operator: "I liked mesh2motion's
            unified grid of clips ... that makes it easier to pick an
            appropriate animation") — no separate select or name list.
            Renders even before GET /api/animations/curated resolves (against
            the single-item CURATED_MOTION_CLIPS fallback above) since
            ClipBrowser shows its own "Loading previews…" status rather than
            needing an empty slot. */}
        <ClipBrowser clips={curatedClips} selectedClipId={selectedClipId} onSelect={setSelectedClipId} />
        <button className="primary-action" onClick={() => onAttachCuratedMotion((curatedClips.find((clip) => clip.id === selectedClipId) || curatedClips[0]).animation)} disabled={busy || motionJobActive}>
          {busy && (job?.kind === 'retarget-motion' || job?.kind === 'apply-clip') ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}
          {/* N5: a template-conformant rig attaches the exact clip action by
              bone name (no retarget math) -- label it distinctly so the
              operator can see which path a click will take. */}
          {isConformantRig ? 'Attach (direct)' : 'Attach motion'}
        </button>
        {kimodoReady && <button className="secondary-button" disabled title="Not wired yet">
          <Sparkles size={16} /> Generate motion with Kimodo
        </button>}
      </>
    )}
    {terminalHydrationError && <small className="motion-error">{terminalHydrationError}</small>}
    {terminalHydrationError && <button className="secondary-button retry-hydration-button" onClick={onRetryTerminalHydration}>Retry loading output</button>}
  </aside>
}
