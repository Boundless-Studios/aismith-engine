import type { ReactNode } from 'react'
import { AlertTriangle, Box, Gauge, HelpCircle, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Upload, Wand2, X } from 'lucide-react'
import { API } from './api'
import { isImageArtifactKind } from './history'
import { showsPrepareCard, viewConsistencyNotice, type OptionalViewLabel, type SubjectKind, type TposeSetMembership } from './workspace'
import { WorkingOn, type WorkingOnAssetDetails } from './WorkingOn'

// Kept deliberately narrower than App.tsx's own `Artifact` type (which is a
// superset) so this module never has to import from App.tsx — that import
// would pull in StudioViewport's `@google/model-viewer` custom-element
// registration, which throws outside a browser DOM (including in vitest's
// default node environment). `metadata` is optional so a synthesized view's
// `synthesized_view` tag can render without widening the type further.
export type ControlArtifact = { id: string; name: string; kind: string; metadata?: Record<string, unknown> }

export function SectionLabel({ children }: { children: ReactNode }) {
  return <div className="section-label">{children}</div>
}

export function UploadCard({ source, onUpload, label = 'Upload model', hint = 'GLB, GLTF, FBX or OBJ · max 20 MB' }: { source: ControlArtifact | null; onUpload: () => void; label?: string; hint?: string }) {
  const image = source && isImageArtifactKind(source.kind)
  return <button className={`upload-card ${source ? 'has-file' : ''}`} onClick={onUpload}>
    {source
      ? <><div className="upload-preview">{image ? <img src={`${API}/assets/${source.id}/download`} alt="Source reference" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <Box size={22} />}</div><div><strong>{source.name}</strong><span>Replace source</span></div></>
      : <><div className="upload-icon"><Upload size={19} /></div><div><strong>{label}</strong><span>{hint}</span></div></>}
  </button>
}

const VIEW_SLOTS: { label: OptionalViewLabel; title: string }[] = [
  { label: 'back', title: 'Back' },
  { label: 'left', title: 'Left' },
  { label: 'right', title: 'Right' },
]

function ViewSlot({ label, title, artifact, onUpload, onRemove, onRegenerate, inSet = false }: { label: OptionalViewLabel; title: string; artifact: ControlArtifact | null; onUpload: () => void; onRemove: () => void; onRegenerate: () => void; inSet?: boolean }) {
  if (artifact) {
    const synthesized = typeof artifact.metadata?.synthesized_view === 'string'
    // Bead 8c9gf: the backend measured this view against the front reference
    // it was drawn from and stored the verdict on the artifact itself, so the
    // notice survives a reload and a rehydrate without any local bookkeeping.
    const notice = viewConsistencyNotice(label, artifact.metadata)
    return <div className={`view-slot view-slot-filled${notice ? ` view-slot-${notice.tone}` : ''}`}>
      <span className="view-slot-label">{title}{synthesized && <span className="view-slot-synth-tag" title="Auto-generated with ComfyUI">synth</span>}</span>
      <div className="view-slot-thumb"><img src={`${API}/assets/${artifact.id}/download`} alt={`${title} view`} /></div>
      {notice && <div className={`view-slot-notice view-slot-notice-${notice.tone}`} role={notice.tone === 'error' ? 'alert' : undefined}>
        <strong>{notice.tone === 'unknown' ? <HelpCircle size={10} /> : <AlertTriangle size={10} />} {notice.headline}</strong>
        <span>{notice.detail}</span>
        {notice.tone === 'error' && <button type="button" className="view-slot-regenerate" onClick={onRegenerate}>Regenerate this view</button>}
      </div>}
      {inSet
        // A set member is REPLACED, never emptied: the backend conditions the
        // next generate on the set, so an emptied slot would silently come
        // back from the run's stored membership at submit time (bead 4vquq).
        ? <div className="view-slot-set-actions">
          <button type="button" className="view-slot-regenerate" onClick={onRegenerate}>Regenerate</button>
          <button type="button" className="view-slot-regenerate" onClick={onUpload}>Replace…</button>
        </div>
        : <button type="button" className="view-slot-remove" onClick={onRemove} aria-label={`Remove ${title.toLowerCase()} view`}><X size={12} /></button>}
    </div>
  }
  return <button type="button" className="view-slot view-slot-empty" onClick={inSet ? onRegenerate : onUpload}>
    <span className="view-slot-label">{title}</span>
    {inSet ? <Wand2 size={16} /> : <Upload size={16} />}
    <span className="view-slot-hint">{inSet ? `Generate ${title.toLowerCase()} view` : `Add ${title.toLowerCase()} view`}</span>
  </button>
}

// The set's front tile. It is not an upload slot: the front is what the other
// three views were drawn from and gated against, so changing it means
// building a new set rather than swapping one image (bead 4vquq).
function TposeFrontTile({ artifact, onCreate, busy }: { artifact: ControlArtifact | null; onCreate: () => void; busy: boolean }) {
  if (!artifact) {
    return <button type="button" className="view-slot view-slot-empty view-slot-front" onClick={onCreate} disabled={busy}>
      <span className="view-slot-label">Front<span className="view-slot-canonical-tag">canonical</span></span>
      <Sparkles size={16} />
      <span className="view-slot-hint">Create the T-pose set</span>
    </button>
  }
  return <div className="view-slot view-slot-filled view-slot-front">
    <span className="view-slot-label">Front<span className="view-slot-canonical-tag" title="The gated T-pose front every other view was drawn from, and the only image a biped generate is conditioned on.">canonical</span></span>
    <div className="view-slot-thumb"><img src={`${API}/assets/${artifact.id}/download`} alt="Canonical T-pose front" /></div>
    <button type="button" className="view-slot-regenerate" onClick={onCreate} disabled={busy}>Rebuild the set</button>
  </div>
}

const SUBJECT_OPTIONS: { value: SubjectKind; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'object', label: 'Object' },
  { value: 'creature', label: 'Creature' },
  { value: 'biped', label: 'Biped character' },
]

type WorkingOnRunProp = { source_artifact_id: string; source_artifact_kind: string; attempts: Array<{ stage: string; status: string; output_artifact_id?: string | null; completed_at?: number | null; created_at?: number }> } | null

export function GenerateControls({
  resolution, setResolution, seed, setSeed, source, views, sourceViewLabels, onUploadView, onRemoveView,
  onAutoGenerateViews = () => {}, onRegenerateView = () => {}, autoGeneratingViews = false,
  // `onPrepare` stays in the prop contract (the single-step `prepare-image`
  // path is still live for power users and other tooling, and App.tsx's
  // "Review uploaded image without ComfyUI" button still drives it) but this
  // panel no longer offers it: the composite leads (bead 4vquq).
  preparedCandidate, preparing, outputDirectory, onChooseOutput, onUpload, onApprove,
  subjectKind = 'auto', onSubjectKindChange = () => {},
  run = null, pinnedArtifactId = null, onPinArtifact = () => {}, detailsById = {},
  onCreateTposeSet = () => {}, creatingTposeSet = false, tposeSetProgress = '',
  tposeSetFront = null, tposeSet = null,
}: {
  resolution: 512 | 768 | 1024
  setResolution: (v: 512 | 768 | 1024) => void
  seed: number
  setSeed: (v: number) => void
  source: ControlArtifact | null
  // Optional back/left/right conditioning views (E2). Front stays `source`.
  views: Partial<Record<OptionalViewLabel, ControlArtifact>>
  // Read-only "Views: front · back" chips for the run's latest completed
  // generate attempt output, independent of `views` (the editable inputs).
  sourceViewLabels: string[]
  onUploadView: (label: OptionalViewLabel) => void
  onRemoveView: (label: OptionalViewLabel) => void
  // Auto-generate views (E3): synthesizes the empty back/left/right slots
  // from the front image via ComfyUI. Never replaces a slot already filled.
  onAutoGenerateViews?: () => void
  // Bead 8c9gf: clears one slot and re-synthesizes just that view, the
  // remedy the consistency notice on an offending tile names.
  onRegenerateView?: (label: OptionalViewLabel) => void
  autoGeneratingViews?: boolean
  preparedCandidate: ControlArtifact | null
  preparing: boolean
  outputDirectory?: string
  onChooseOutput: () => void
  onUpload: () => void
  onPrepare: () => void
  onApprove: () => void
  // H: subject-agnostic generation — the Prepare (T-pose) card only makes
  // sense for a biped or an as-yet-undetermined ("auto") subject.
  subjectKind?: SubjectKind
  onSubjectKindChange?: (kind: SubjectKind) => void
  // I1: the Working-on card above the source grid — the run's compatible
  // image candidates (its own upload, or an approved T-pose candidate).
  run?: WorkingOnRunProp
  pinnedArtifactId?: string | null
  onPinArtifact?: (artifactId: string) => void
  detailsById?: Record<string, WorkingOnAssetDetails>
  // Bead 4vquq: the one composite step. Prepare and Auto-generate-views used
  // to be two buttons the operator had to sequence correctly; this is the
  // single one that draws the gated T-pose front and fans the three views
  // off it, reporting its own staged progress while it runs.
  onCreateTposeSet?: () => void
  creatingTposeSet?: boolean
  tposeSetProgress?: string
  // The set's front artifact, and the membership block it carries. Non-null
  // once this run owns a canonical set, whether it was just built or was
  // rehydrated from the run's own attempts.
  tposeSetFront?: ControlArtifact | null
  tposeSet?: TposeSetMembership | null
}) {
  return <>
    <SectionLabel>Subject</SectionLabel>
    <div className="segmented subject-kind-segmented">
      {SUBJECT_OPTIONS.map((option) => <button key={option.value} className={subjectKind === option.value ? 'selected' : ''} onClick={() => onSubjectKindChange(option.value)}>{option.label}</button>)}
    </div>
    <WorkingOn
      stage="generate"
      stageLabel="Generate"
      run={run}
      current={source}
      detailsById={detailsById}
      pinnedArtifactId={pinnedArtifactId}
      onPin={onPinArtifact}
      onUpload={onUpload}
      hideUploadButton
    />
    {showsPrepareCard(subjectKind)
      // Bead 4vquq -- biped/auto: the source image is demoted to a REFERENCE
      // chip and the canonical T-pose set is the thing the operator builds,
      // reviews, and generates from. Object and creature runs keep the
      // original source-plus-optional-views layout below, unchanged.
      ? <>
        <SectionLabel>1 · Reference image</SectionLabel>
        <UploadCard source={source} onUpload={onUpload} label="Upload reference image" />
        <small className="view-grid-hint reference-chip">Reference — used for palette/identity. It is never what a biped generate is conditioned on; the canonical T-pose set below is.</small>
        <SectionLabel>2 · Canonical T-pose set</SectionLabel>
        <div className="source-view-grid tpose-set-grid">
          <TposeFrontTile artifact={tposeSetFront} onCreate={onCreateTposeSet} busy={creatingTposeSet || preparing} />
          {VIEW_SLOTS.map((slot) => <ViewSlot key={slot.label} label={slot.label} title={slot.title} artifact={views[slot.label] || null} inSet={Boolean(tposeSet)} onUpload={() => onUploadView(slot.label)} onRemove={() => onRemoveView(slot.label)} onRegenerate={() => onRegenerateView(slot.label)} />)}
        </div>
        {tposeSet && !tposeSet.complete && <small className="view-grid-hint">This set is missing its {tposeSet.failedRoles.join(' and ')} view — regenerate that tile or upload your own.</small>}
        <button type="button" className="secondary-button auto-generate-views create-tpose-set" onClick={onCreateTposeSet} disabled={!source || creatingTposeSet || preparing}>
          {creatingTposeSet ? <LoaderCircle className="spin" size={14} /> : <Sparkles size={14} />}
          {creatingTposeSet ? (tposeSetProgress || 'Creating T-pose set…') : tposeSetFront ? 'Rebuild T-pose set' : 'Create T-pose set'}
        </button>
        <small className="view-grid-hint">One step: the T-pose front is drawn and checked, then the back, left, and right views are drawn from that front — never from the reference.</small>
        {sourceViewLabels.length > 1 && <small className="status-chip view-chip-row">Views: {sourceViewLabels.join(' · ')}</small>}
        {preparedCandidate && <div className="preparation-card">
          <div><strong>Approve this T-pose set</strong><span>Flux.2 Klein · gated front · back, left, and right drawn from it</span></div>
          <div className="preparation-actions"><button className="approve-preparation" onClick={onApprove}><ShieldCheck size={14} /> Approve the set</button><button onClick={onCreateTposeSet} disabled={creatingTposeSet}><RotateCcw size={14} /> Rebuild</button></div>
          <small>Approving binds the run to this quad: Generate is conditioned on exactly these four images. Inspect every tile first; every result stays in History.</small>
        </div>}
      </>
      : <>
        <SectionLabel>1 · Source image</SectionLabel>
        <div className="source-view-grid">
          <UploadCard source={source} onUpload={onUpload} label="Upload reference image" />
          {VIEW_SLOTS.map((slot) => <ViewSlot key={slot.label} label={slot.label} title={slot.title} artifact={views[slot.label] || null} onUpload={() => onUploadView(slot.label)} onRemove={() => onRemoveView(slot.label)} onRegenerate={() => onRegenerateView(slot.label)} />)}
        </div>
        <small className="view-grid-hint">Add side or back views of the same subject for a fuller reconstruction (2–4 images, ~2× generation time).</small>
        <button type="button" className="secondary-button auto-generate-views" onClick={() => onAutoGenerateViews()} disabled={!source || autoGeneratingViews}>
          {autoGeneratingViews ? <LoaderCircle className="spin" size={14} /> : <Wand2 size={14} />}
          {autoGeneratingViews ? 'Synthesizing views…' : 'Auto-generate views'}
        </button>
        <small className="view-grid-hint">Fills empty back/left/right slots from the front image; replace any result you don't like.</small>
        {sourceViewLabels.length > 1 && <small className="status-chip view-chip-row">Views: {sourceViewLabels.join(' · ')}</small>}
      </>}
    <SectionLabel>3 · Generate 3D</SectionLabel>
    <SectionLabel>Output folder</SectionLabel>
    <div className="output-location"><span title={outputDirectory}>{outputDirectory || 'AISmith 3D artifact cache'}</span><button onClick={onChooseOutput}>Choose…</button></div>
    <div className="option-row"><label>Detail level</label><div className="segmented"><button className={resolution === 512 ? 'selected' : ''} onClick={() => setResolution(512)}>Fast</button><button className={resolution === 1024 || resolution === 768 ? 'selected' : ''} onClick={() => setResolution(1024)}>High detail</button></div></div>
    <div className="option-row"><label>Seed</label><input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(Math.max(0, Math.min(2147483647, Number(event.target.value) || 0)))} /></div>
    <div className="info-line"><Gauge size={14} /> Same seed, same result.</div>
  </>
}
