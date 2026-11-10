import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertTriangle, ArrowLeft, Box, Brush, Download, FolderClock, FolderOpen, Gauge, LoaderCircle, MemoryStick, Paintbrush, Play, RefreshCw, RotateCcw, ScanLine, Settings2, ShieldCheck, Sparkles, Upload, UserRound, Wrench, X } from 'lucide-react'
import StudioViewport from './StudioViewport'
import JobProgress from './JobProgress'
import SkeletonWorkspace, { type RigAdjustmentSave } from './SkeletonWorkspace'
import HistoryView from './HistoryView'
import CharacterJourney from './CharacterJourney'
import CharacterJourneySelector from './CharacterJourneySelector'
import { GenerateControls, SectionLabel, UploadCard } from './GenerateControls'
import { journeyWorkspace } from './journey'
import { formatSetting, isImageArtifactKind, isRigPlaybackStage, shouldRestoreRefineImage, studioTabForSourceKind, studioTabForStage, type RunManifest, type StudioTab } from './history'
import { API, createSubmissionId } from './api'
import { ACTIVE_JOB_STORAGE_KEY, ACTIVE_RUN_STORAGE_KEY, activeJobResyncDecision, activeJobResyncId, applyClipJobPayload, autoRigCaption, canAcceptMesh2MotionContext, canAcceptMesh2MotionExport, canGenerateFromPreparedImage, canGenerateMotion, canLaunchAutoRig, autoRigJobPayload, canonicalizeCandidatePinnable, canonicalizeDensityReport, type CanonicalizeDensity, type CanonicalizeDensityReport, canonicalizeJobPayload, canonicalizeVerdict, type CanonicalizeVerdict, projectionVerdict, type ProjectionVerdict, projectionReport, type ProjectionReport, rebuildViewsJobPayload, rebuildViewsReport, type RebuildViewsReport, sourceFidelityVerdict, type SourceFidelityVerdict, sourceFidelityReport, type SourceFidelityReport, CANONICALIZE_DEFAULT_DENSITY, CANONICALIZE_DENSITY_STORAGE_KEY, CUSTOM_TRIANGLE_TARGET_MAXIMUM, CUSTOM_TRIANGLE_TARGET_MINIMUM, customTriangleTarget, QUALITY_TIER_OPTIONS, readCanonicalizeDensity, resolvedTriangleTarget, serializeCanonicalizeDensity, canRestoreHistoryArtifact, type CleanMethod, cleanMethodForStage, cleanRecommendation, cleanTargetExceedsSource, exportGlbNameHint, generateSourceFields, generateSourceViewArtifactIds, hasActiveStudioJob, isArrayBufferPayload, isReCleaningCleanOutput, isTemplateConformantRig, jobPollIntervalMs, latestCompletedAttempt, mergeSynthesizedViews, mesh2MotionExportInputArtifactId, motionJobPayload, type OptionalViewLabel, pendingPreparedCandidateId, pollJobUntilTerminal, preparationJobPayload, readRememberedJob, rehydrateViewSlots, restoreHistoryArtifacts, retargetMotionJobPayload, retryNetworkPost, rigWorkspaceSurface, runAssetOptions, shouldApplyCompletedArtifact, shouldApplyHydratedRun, shouldMarkTerminalHydrationComplete, shouldPreviewRecordedArtifact, shouldPromoteCompletedOutput, serializeRememberedJob, shouldRefreshCapabilities, shouldRestoreRun, shouldResyncActiveJob, shouldStartTerminalHydration, showsPrepareCard, sourceViewChipLabels, synthesizeViewsJobPayload, synthesizedViewArtifactIds, createTposeSetJobPayload, isPreparedFrontStage, latestPreparedFrontAttempt, tposeSetMembership, tposeSetViewArtifactIds, viewRehydrationSources, type SubjectKind } from './workspace'
import { WorkingOn, type WorkingOnAssetDetails } from './WorkingOn'
import { AnimatePanel } from './AnimatePanel'
import { CURATED_MOTION_CLIPS, curatedActionLabel } from './animations'
import { ARCHIVED_RUNS_STORAGE_KEY, readArchivedRunIds } from './runOrganization'

type Tab = StudioTab
type Artifact = { id: string; name: string; kind: string; size_bytes: number; stage?: string; metadata?: Record<string, unknown> }
type Capabilities = { gpu: { available: boolean; name: string; vram_total_mb: number; vram_used_mb: number; cuda: boolean; torch_cuda_available?: boolean; native_inference_ready?: boolean; model_variant: string; recommended_resolution: number }; ram: { total_mb: number; used_mb: number }; blender: { available: boolean }; models: Record<string, boolean>; diagnostics?: string[]; memory: { loaded_model: string | null }; output_directory?: string }
type Job = { id: string; kind: string; submission_id: string; status: string; progress: number; message: string; logs: string[]; warnings: string[]; output_artifact_id?: string | null; error?: string | null; run_id?: string; attempt_id?: string; promoted_output?: boolean }
type DownloadState = { id: string; model: string; status: string; progress: number; message: string; logs: string[]; error?: string | null; file_name?: string | null; file_index?: number; file_count?: number; file_bytes?: number; file_total_bytes?: number }
type PaintStroke = { x: number; y: number; radius: number; color: string; opacity: number; hardness: number }
type ModelDetails = { vertices: number; triangles: number; materials: number; meshes: number; format: string; size_bytes: number }
type ExportState = { status: 'idle' | 'exporting' | 'saved' | 'downloaded' | 'error'; path?: string; error?: string }
// Attaches the HTTP status onto a thrown Error so a caller (export's 409
// no-output-folder fallback) can branch on it without parsing the message
// text -- ApiError narrows the plain `Error` type json() actually throws.
type ApiError = Error & { status?: number }

const tabs: { id: Tab; label: string; icon: typeof Sparkles; kicker: string }[] = [
  { id: 'generate', label: 'Generate', icon: Sparkles, kicker: 'Image to 3D' },
  { id: 'retopologize', label: 'Clean', icon: ScanLine, kicker: 'Repair topology and materials' },
  { id: 'rig', label: 'Rig + Animate', icon: UserRound, kicker: 'Mesh2Motion workspace' },
  { id: 'paint', label: 'Texture Paint', icon: Paintbrush, kicker: 'Material studio' },
]

const TERMINAL_HYDRATION_TIMEOUT_MS = 15_000

async function json<T>(url: string, init?: RequestInit): Promise<T> { const res = await fetch(`${API}${url}`, init); if (!res.ok) { const body = await res.json().catch(() => ({})); const detail = body.detail; const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item) => item.msg || 'Invalid request').join('; ') : `Request failed (${res.status})`; const apiError: ApiError = new Error(message); apiError.status = res.status; throw apiError }; return res.json() }
// Browser-download fallback for Export: triggers Chrome's own downloads
// flow (save-dialog/Downloads folder) for whatever the caller pointed it
// at -- used both as the explicit "Download" affordance and as Export's
// automatic fallback when no output folder is configured server-side.
function triggerBrowserDownload(url: string) {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.rel = 'noopener'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}
async function upload(file: File): Promise<Artifact> { const form = new FormData(); form.append('file', file); return json<Artifact>('/assets/upload', { method: 'POST', body: form }) }
async function postJsonWithNetworkRetry<T>(url: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }
  return retryNetworkPost(() => json<T>(url, init), signal)
}

function formatMb(mb: number) { return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB` }
// One shared unit for a used/total pair (header meters) instead of formatMb's
// per-value unit switch, which can print mismatched units across the slash.
function formatGbPair(usedMb: number, totalMb: number) { return `${(usedMb / 1024).toFixed(1)} / ${(totalMb / 1024).toFixed(1)} GB` }
function formatDuration(milliseconds: number) { const seconds = Math.max(0, Math.floor(milliseconds / 1000)); return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}` }
function formatCount(count?: number) { return count === undefined ? '—' : count.toLocaleString() }

export default function App() {
  const [view, setView] = useState<'studio' | 'history' | 'mesh2motion'>('studio')
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false)
  const [systemPopoverOpen, setSystemPopoverOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('generate')
  const [rigWorkspaceMode, setRigWorkspaceMode] = useState<'rig' | 'animate' | undefined>(undefined)
  const [caps, setCaps] = useState<Capabilities | null>(null)
  const [source, setSource] = useState<Artifact | null>(null)
  const [output, setOutput] = useState<Artifact | null>(null)
  const [preparedCandidate, setPreparedCandidate] = useState<Artifact | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  // Default to 1024 (Fine mode); 512 maps to Coarse mode.
  const [resolution, setResolution] = useState<512 | 768 | 1024>(1024)
  const [seed, setSeed] = useState(0)
  // S2 (docs/PIPELINE_POLICY.md) makes the canonical clean the mandatory path
  // before rigging, so the Clean panel opens on it; AI Refine and Instant
  // Meshes stay reachable as the pre-S2 fallbacks.
  const [cleanMethod, setCleanMethod] = useState<CleanMethod>('canonicalize')
  // The canonicalize gate's verdict for the most recent completed canonicalize
  // job, kept in App state rather than read off `output` because a blocked
  // candidate is never promoted into `output` — the operator still has to see
  // its findings and decide whether to pin it.
  const [canonicalResult, setCanonicalResult] = useState<{ jobId: string; artifactId: string; artifactName: string; verdict: CanonicalizeVerdict; promoted: boolean; density: CanonicalizeDensityReport | null; projection: { verdict: ProjectionVerdict; report: ProjectionReport | null } | null; sourceFidelity: { verdict: SourceFidelityVerdict; report: SourceFidelityReport | null } | null } | null>(null)
  const [rebuildResult, setRebuildResult] = useState<{ jobId: string; report: RebuildViewsReport } | null>(null)
  // i63ky: the operator's mesh-density choice for the canonical clean, kept
  // for the session (not forever — a stale Ultra silently applying weeks later
  // costs twenty minutes of GPU nobody asked to spend).
  const [canonicalDensity, setCanonicalDensity] = useState<CanonicalizeDensity>(() => (
    typeof window === 'undefined'
      ? CANONICALIZE_DEFAULT_DENSITY
      : readCanonicalizeDensity(window.sessionStorage.getItem(CANONICALIZE_DENSITY_STORAGE_KEY))
  ))
  useEffect(() => {
    if (typeof window === 'undefined') return
    try { window.sessionStorage.setItem(CANONICALIZE_DENSITY_STORAGE_KEY, serializeCanonicalizeDensity(canonicalDensity)) } catch { /* a blocked sessionStorage must never break Clean */ }
  }, [canonicalDensity])
  const [refineTargetVertices, setRefineTargetVertices] = useState(25000)
  const [retopoTargetVertices, setRetopoTargetVertices] = useState(25000)
  const [cleanRepairHoles, setCleanRepairHoles] = useState(false)
  const [gpuFirst, setGpuFirst] = useState(false)
  const [refineImage, setRefineImage] = useState<Artifact | null>(null)
  // Optional back/left/right conditioning views for `generate` (E2). The front
  // view stays the existing `source` state; these are per-generate inputs and
  // are reset only when the run or the front image identity actually changes
  // (startNewJourney, a fresh upload, or History's restore) -- journey-step
  // navigation within the same run and front image leaves this alone, and
  // rehydrateViews() (below) backfills it from the run's own artifacts on a
  // fresh mount, a run switch, or a full page reload.
  const [views, setViews] = useState<Partial<Record<OptionalViewLabel, Artifact>>>({})
  const [sourceViewLabels, setSourceViewLabels] = useState<string[]>([])
  const [generateSourceFidelity, setGenerateSourceFidelity] = useState<{ verdict: SourceFidelityVerdict; report: SourceFidelityReport | null } | null>(null)
  // True while rehydrateViews() is fetching the run's synthesize-views/
  // generate artifacts to backfill `views` -- the Generate submit button
  // stays disabled meanwhile so a click can never race the fetch and submit
  // a silent single-view (`source_artifact_id`-only) job right after a step
  // switch, before the real back/left/right ids have loaded back in.
  const [viewsRehydrating, setViewsRehydrating] = useState(false)
  // The front image id `views` was last rehydrated/cleared against, so a
  // subsequent open of the Generate step can tell "same front, keep what's
  // there" from "front changed, these back/left/right ids no longer apply".
  const viewsSourceIdRef = useRef<string | null>(null)
  // Auto-generate views (E3): synthesizes the empty back/left/right slots
  // from the front image via ComfyUI; a self-contained submit+poll+hydrate
  // flow rather than routing through the shared terminal-hydration effect,
  // since it fills `views` slots rather than replacing source/output.
  const [synthesizingViews, setSynthesizingViews] = useState(false)
  // Bead 4vquq: the run's canonical T-pose FRONT, read back from the run's
  // own attempts. `preparedCandidate` and `source` carry the set's membership
  // while this tab is the one that built or approved it, but after a reload
  // (or a run switch) `source` is a metadata-less journey stub -- and without
  // this the front tile rendered EMPTY beside three rehydrated view tiles on
  // a run whose set was already approved. Server truth, same as
  // rehydrateViews/rehydratePreparedCandidate.
  const [tposeSetFrontArtifact, setTposeSetFrontArtifact] = useState<Artifact | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  // True only while `error` holds bead 8c9gf's pre-generate refusal, which is
  // the one 4xx an operator is offered a deliberate override for.
  const [viewConsistencyBlocked, setViewConsistencyBlocked] = useState(false)
  const [paintColor, setPaintColor] = useState('#7c5cff')
  const [roughness, setRoughness] = useState(42)
  const [metallic, setMetallic] = useState(18)
  const [paintStrokes, setPaintStrokes] = useState<PaintStroke[]>([])
  const [brushSize, setBrushSize] = useState(9)
  const [brushOpacity, setBrushOpacity] = useState(90)
  const [brushHardness, setBrushHardness] = useState(80)
  const [downloads, setDownloads] = useState<DownloadState[]>([])
  const [modelDetails, setModelDetails] = useState<ModelDetails | null>(null)
  const [exportState, setExportState] = useState<ExportState>({ status: 'idle' })
  const [generationStartedAt, setGenerationStartedAt] = useState<number | null>(null)
  const [generationElapsed, setGenerationElapsed] = useState(0)
  const [generationTime, setGenerationTime] = useState<number | null>(null)
  const [generationPendingPreview, setGenerationPendingPreview] = useState(false)
  const [refineStartedAt, setRefineStartedAt] = useState<number | null>(null)
  const [refineElapsed, setRefineElapsed] = useState(0)
  const [refineTime, setRefineTime] = useState<number | null>(null)
  const [refinePendingPreview, setRefinePendingPreview] = useState(false)
  const [retopoStartedAt, setRetopoStartedAt] = useState<number | null>(null)
  const [retopoElapsed, setRetopoElapsed] = useState(0)
  const [retopoTime, setRetopoTime] = useState<number | null>(null)
  const [retopoPendingPreview, setRetopoPendingPreview] = useState(false)
  const [activeRun, setActiveRun] = useState<RunManifest | null>(null)
  const [runs, setRuns] = useState<RunManifest[]>([])
  const [runsLoading, setRunsLoading] = useState(true)
  const [terminalHydrationJobId, setTerminalHydrationJobId] = useState<string | null>(null)
  const [terminalHydrationFailedJobId, setTerminalHydrationFailedJobId] = useState<string | null>(null)
  const [terminalHydrationRetryJobId, setTerminalHydrationRetryJobId] = useState<string | null>(null)
  // I2: the artifact a stage's picker explicitly selected — wins over the
  // journey's auto-resolved artifact for the stages where more than one
  // compatible candidate can exist. Cleared on a journey change or once a
  // new job output lands for the run (a stale pin must not silently persist).
  const [pinnedArtifactId, setPinnedArtifactId] = useState<string | null>(null)
  // H: run subject — read from the active run once one exists; remembered
  // here for the run-creating job when there isn't one yet.
  const [subjectKind, setSubjectKind] = useState<SubjectKind>('auto')
  // I1: a small cache of { name, vertices, size_bytes } for artifacts shown
  // only in a Working-on picker (fetched lazily, once per artifact id).
  const [assetDetailsById, setAssetDetailsById] = useState<Record<string, WorkingOnAssetDetails>>({})
  const fetchedAssetDetailIdsRef = useRef<Set<string>>(new Set())
  // §3-C7: mirrors HistoryView's localStorage archive set so the journey
  // selector hides the same runs — re-read whenever Studio becomes visible
  // again (the one place a run actually gets archived is History).
  const [archivedRunIds, setArchivedRunIds] = useState<Set<string>>(() => (
    typeof window === 'undefined' ? new Set() : readArchivedRunIds(window.localStorage.getItem(ARCHIVED_RUNS_STORAGE_KEY))
  ))
  useEffect(() => {
    if (view !== 'studio') return
    setArchivedRunIds(readArchivedRunIds(window.localStorage.getItem(ARCHIVED_RUNS_STORAGE_KEY)))
  }, [view])
  const activeRunIdRef = useRef<string | null>(null)
  const activeJobIdRef = useRef<string | null>(null)
  const mesh2MotionExportInFlightRef = useRef(false)
  const automatedRigLaunchRef = useRef(false)
  const terminalHydrationJobIdRef = useRef<string | null>(null)
  const terminalHydratedJobIdRef = useRef<string | null>(null)
  const contextVersionRef = useRef(0)
  const fileRef = useRef<HTMLInputElement>(null)
  const pendingUploadTargetRef = useRef<'input' | 'model' | 'view-back' | 'view-left' | 'view-right'>('input')
  const toolsMenuRef = useRef<HTMLDivElement>(null)
  const systemPopoverRef = useRef<HTMLDivElement>(null)
  const refineImageRef = useRef<HTMLInputElement>(null)
  const refineDefaultedAsset = useRef<string | null>(null)
  const retopoDefaultedAsset = useRef<string | null>(null)
  const displayAsset = output || source
  const activeSource = tab === 'generate' ? source : output || source
  const activeViewCount = 1 + Object.values(views).filter(Boolean).length
  const preparationActive = busy || hasActiveStudioJob(job)
  const preparedForGeneration = canGenerateFromPreparedImage(source, preparationActive)
    || Boolean(source && source.kind !== 'glb' && !isPreparedFrontStage(source.stage) && !source.stage)
    || Boolean(source && source.stage === 'generate')
  // Bead 4vquq: the canonical T-pose set is presented as one thing. The front
  // artifact carries the whole set's membership block, so the studio reads it
  // off whichever image is currently standing in as the front -- the
  // candidate awaiting approval, or the approved front already in `source`.
  const tposeSetFront = [preparedCandidate, source, tposeSetFrontArtifact]
    .find((candidate) => candidate?.metadata && tposeSetMembership(candidate.metadata)) || null
  const tposeSet = tposeSetMembership(tposeSetFront?.metadata)
  // The staged line the composite job is currently on ("Generating the back
  // view… 2/4"), shown on the button that started it.
  const tposeSetProgress = job?.kind === 'create-tpose-set' && hasActiveStudioJob(job) ? job.message : ''
  const creatingTposeSet = job?.kind === 'create-tpose-set' && hasActiveStudioJob(job)
  useEffect(() => { activeRunIdRef.current = activeRun?.id || null }, [activeRun?.id])
  useEffect(() => { activeJobIdRef.current = job?.id || null }, [job?.id])
  // Read the run's own subject_kind once one exists (older runs, or a
  // backend build that hasn't landed the field yet, default to "auto").
  useEffect(() => { setSubjectKind(activeRun?.subject_kind || 'auto') }, [activeRun?.id, activeRun?.subject_kind])
  // Setting the pin alone never moves the viewport — journeyWorkspace's
  // resolution only gets re-consulted when something calls
  // openJourneyWorkspace (a journey-strip click, restoring a run, etc). A
  // picker's onPin (WorkingOn's <select>, or an Animations-list row) needs
  // the pin to apply immediately, so this re-runs the same resolution for
  // whichever step is currently open, passing the freshly-picked id
  // explicitly — the just-set pinnedArtifactId state isn't visible yet in
  // this same synchronous call (React state updates are async).
  const pinArtifact = (artifactId: string) => {
    setPinnedArtifactId(artifactId)
    if (!activeRun) return
    // Must mirror rigWorkspaceSurface's own rig/animate split exactly
    // (workspace.ts): that function treats anything other than the literal
    // 'rig' mode -- including the undefined default a fresh page load or a
    // "Preview / continue" from History starts with -- as the *animate*
    // surface. This used to test the opposite way (only the literal
    // 'animate' counted, so undefined fell through to 'skeleton'), which
    // meant clicking a row in the already-visible Animations list, while
    // still in the undefined default mode, silently kicked the operator out
    // to the Skeleton review screen instead of switching to the clip they
    // just clicked -- the viewport-goes-wrong-after-clicking-a-clip bug.
    const stepId = tab === 'generate' ? 'generate'
      : tab === 'retopologize' ? 'clean'
      : tab === 'paint' ? 'paint'
      : tab === 'rig' ? (rigWorkspaceMode === 'rig' ? 'skeleton' : 'animate')
      : tab
    openJourneyWorkspace(stepId, activeRun, artifactId)
  }
  const fetchAssetDetails = (artifactId: string) => {
    if (!artifactId || fetchedAssetDetailIdsRef.current.has(artifactId)) return
    fetchedAssetDetailIdsRef.current.add(artifactId)
    json<Artifact>(`/assets/${artifactId}`).then((artifact) => {
      setAssetDetailsById((current) => ({ ...current, [artifactId]: { name: artifact.name, size_bytes: artifact.size_bytes } }))
      if (artifact.kind === 'glb') {
        json<ModelDetails>(`/assets/${artifactId}/details`)
          .then((details) => setAssetDetailsById((current) => ({ ...current, [artifactId]: { ...current[artifactId], vertices: details.vertices } })))
          .catch(() => undefined)
      }
    }).catch(() => { fetchedAssetDetailIdsRef.current.delete(artifactId) })
  }
  const changeSubjectKind = async (kind: SubjectKind) => {
    setSubjectKind(kind)
    if (!activeRun) return
    try {
      const updated = await json<RunManifest>(`/runs/${activeRun.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_kind: kind }),
      })
      setActiveRun(updated)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not update subject') }
  }
  const retryTerminalHydration = () => {
    if (!job || terminalHydrationFailedJobId !== job.id) return
    setError('')
    setTerminalHydrationFailedJobId(null)
    setTerminalHydrationRetryJobId(job.id)
  }
  const handBackSkeleton = async (artifactId: string, note: string) => {
    if (!activeRun) throw new Error('Start or restore a run before completing a checkpoint.')
    if (activeSource?.id !== artifactId) throw new Error('The visible rig changed. Review the current rig before approval.')
    const runId = activeRun.id
    const contextVersion = contextVersionRef.current
    const updated = await postJsonWithNetworkRetry<RunManifest>(`/runs/${runId}/checkpoints/skeleton`, { note, artifact_id: artifactId })
    if (activeRunIdRef.current === runId && contextVersionRef.current === contextVersion) setActiveRun(updated)
  }
  const reviewAnimatedArtifact = async (artifactId: string, reviewStatus: 'accepted' | 'rejected', note: string) => {
    if (!activeRun) throw new Error('Start or restore a run before reviewing its animation.')
    if (activeSource?.id !== artifactId) throw new Error('The visible animation changed. Review the current candidate again.')
    const runId = activeRun.id
    const contextVersion = contextVersionRef.current
    const updated = await json<RunManifest>(`/runs/${runId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: reviewStatus, review_note: note, artifact_id: artifactId }),
    })
    if (activeRunIdRef.current === runId && contextVersionRef.current === contextVersion) setActiveRun(updated)
  }
  // Rehydrates the Generate step's back/left/right view slots from the run's
  // own backend artifacts on a fresh mount, a run switch, or a full page
  // reload -- the K18 auto-synthesized (or previously operator-uploaded)
  // views are persisted server-side (`synthesized_view`/`sibling_views` on a
  // synthesize-views output, `source_views` on a generate output); only the
  // editable `views` state that feeds the next generate payload starts
  // empty. See workspace.ts's rehydrateViewSlots for the priority order
  // (operator-filled `current` wins, then the latest generate attempt's own
  // source_views, then the latest synthesize-views set). A no-op when the
  // run has neither a completed generate nor synthesize-views attempt yet.
  const rehydrateViews = async (run: RunManifest) => {
    const generateAttempt = latestCompletedAttempt(run.attempts, 'generate')
    // The canonical set's front (bead 4vquq) is resolved from the newest
    // `create-tpose-set` attempt on its own, NOT from "the newest view-source
    // attempt": regenerating one tile of a set submits a `synthesize-views`
    // attempt on the same run, and letting that shadow the set attempt lost
    // the whole set (front tile back to "Create the T-pose set", empty
    // siblings) on the next reload. A loose synthesis only counts when it is
    // newer than the set, so a rebuilt set is never refilled from the older
    // synthesis it replaced.
    const { setFront: setFrontAttempt, looseSynth: synthAttempt } = viewRehydrationSources(run.attempts)
    // Cleared BEFORE the early return: switching to a run that never built a
    // set must not leave the previous run's canonical front on screen.
    if (!setFrontAttempt) setTposeSetFrontArtifact(null)
    if (!generateAttempt?.output_artifact_id && !synthAttempt?.output_artifact_id && !setFrontAttempt?.output_artifact_id) return
    const runId = run.id
    setViewsRehydrating(true)
    try {
      let generatedEntries: Array<readonly [OptionalViewLabel, Artifact]> = []
      let synthesizedEntries: Array<readonly [OptionalViewLabel, Artifact]> = []
      let setEntries: Array<readonly [OptionalViewLabel, Artifact]> = []
      if (generateAttempt?.output_artifact_id) {
        const output = await json<Artifact>(`/assets/${generateAttempt.output_artifact_id}`)
        const ids = generateSourceViewArtifactIds(output.metadata)
        generatedEntries = await Promise.all(ids.map(async ([label, id]) => [label, await json<Artifact>(`/assets/${id}`)] as const))
      }
      if (synthAttempt?.output_artifact_id) {
        const primary = await json<Artifact>(`/assets/${synthAttempt.output_artifact_id}`)
        const ids = synthesizedViewArtifactIds(primary.id, primary.metadata)
        synthesizedEntries = await Promise.all(ids.map(async ([label, id]) => [label, id === primary.id ? primary : await json<Artifact>(`/assets/${id}`)] as const))
      }
      if (setFrontAttempt?.output_artifact_id) {
        // The `create-tpose-set` output IS the set's front; it carries the
        // membership block the canonical front tile renders from, and that
        // block (not the front's original sibling_views) names the CURRENT
        // back/left/right members, regenerated tiles included.
        const front = await json<Artifact>(`/assets/${setFrontAttempt.output_artifact_id}`)
        const ids = tposeSetViewArtifactIds(front.id, front.metadata)
        setEntries = await Promise.all(ids.map(async ([label, id]) => [label, await json<Artifact>(`/assets/${id}`)] as const))
        if (activeRunIdRef.current === runId) setTposeSetFrontArtifact({ ...front, stage: setFrontAttempt.stage })
      }
      if (activeRunIdRef.current !== runId) return
      // Set members are what the next generate is conditioned on, so they
      // outrank a loose synthesis of the same slot; the synthesis only fills
      // slots the set has no member for.
      setViews((current) => rehydrateViewSlots(current, [...synthesizedEntries, ...setEntries], generatedEntries))
    } catch {
      // Best-effort background backfill -- leave whatever views already
      // exist rather than surfacing a network error for a rehydrate.
    } finally {
      if (activeRunIdRef.current === runId) setViewsRehydrating(false)
    }
  }
  // The Approve affordance on the Prepare/Generate step reads local
  // `preparedCandidate` state, which openJourneyWorkspace/History-restore
  // reset to null on every navigation (see below) -- it was only ever
  // re-populated by THIS tab's own prepareImage() job completing (line
  // ~512). That leaves a real approved-candidate-awaiting-review invisible
  // whenever it was produced by another tab or hit the API directly (a
  // shared run several probes touched), because nothing ever asked the
  // server. Mirrors rehydrateViews() above: read the run's own attempts and
  // backfill from server truth rather than trusting this tab's memory.
  const rehydratePreparedCandidate = async (run: RunManifest) => {
    const candidateId = pendingPreparedCandidateId(run)
    if (!candidateId) return
    // Carry the stage the candidate was actually produced by (bead 4vquq):
    // `prepare-image` or the `create-tpose-set` composite. Both are approvable
    // fronts, but stamping the wrong one on the artifact would misreport the
    // run's own history in every panel that reads `stage`.
    const producingStage = latestPreparedFrontAttempt(run.attempts)?.stage || 'prepare-image'
    const runId = run.id
    try {
      const artifact = await json<Artifact>(`/assets/${candidateId}`)
      if (activeRunIdRef.current !== runId) return
      setPreparedCandidate({ ...artifact, stage: producingStage })
    } catch {
      // Best-effort background backfill -- leave the Approve affordance
      // hidden rather than surfacing a network error for a rehydrate.
    }
  }
  const openJourneyWorkspace = async (
    stepId: string,
    selectedRun: RunManifest | null = activeRun,
    // Defaults to the current pin state; pinArtifact() passes the
    // freshly-picked id explicitly instead, since it calls this in the same
    // tick it calls setPinnedArtifactId (whose new value isn't visible yet).
    pinnedOverride: string | null = pinnedArtifactId,
  ) => {
    // A chain (serpent) rig's generate-motion output IS the playable clip
    // (no separate retarget step); a humanoid rig's generate-motion output
    // is only an intermediate BVH. subject_kind is the run-level signal for
    // this, available before any specific artifact has been fetched.
    const animatedOutputStages = selectedRun?.subject_kind === 'creature' ? ['retarget-motion', 'generate-motion', 'apply-clip'] : undefined
    const selection = journeyWorkspace(stepId, tab, selectedRun, { pinnedArtifactId: pinnedOverride, animatedOutputStages })
    if (selection.destination === 'history') {
      setView('history')
      return
    }
    setView('studio')
    setTab(selection.destination)
    if (selection.destination === 'rig') setRigWorkspaceMode(selection.mode)
    if (!selection.artifactId || !selectedRun) return

    const contextVersion = contextVersionRef.current + 1
    contextVersionRef.current = contextVersion
    setError('')
    try {
      const artifact = await json<Artifact>(`/assets/${selection.artifactId}`)
      if (contextVersionRef.current !== contextVersion || activeRunIdRef.current !== selectedRun.id) return
      const sourceKind = selectedRun.source_artifact_kind
      const restoredSource = selection.isSource
        ? artifact
        : { id: selectedRun.source_artifact_id, name: 'Journey source', kind: sourceKind, size_bytes: 0 }
      const stage = selection.stage || 'source'
      const restoredArtifact = { ...artifact, stage: selection.isSource ? undefined : stage }
      const restored = restoreHistoryArtifacts(stage, Boolean(selection.isSource), restoredSource, restoredArtifact)
      if (selection.destination === 'retopologize') setCleanMethod(cleanMethodForStage(stage))
      resetPendingPreview()
      setPreparedCandidate(null)
      setRefineImage(shouldRestoreRefineImage(stage, sourceKind, Boolean(selection.isSource)) ? restoredSource : null)
      setPaintStrokes([])
      // `views` only matters on the Generate step -- leave it untouched when
      // navigating elsewhere (a step switch away and back preserves whatever
      // the operator filled in). Reaching Generate itself: clear+rehydrate
      // only when the front image identity actually changed (a different
      // run, or a different candidate picked as the front); otherwise the
      // existing state already reflects this run's front image correctly.
      if (selection.destination === 'generate') {
        if (viewsSourceIdRef.current !== restored.source.id) setViews({})
        viewsSourceIdRef.current = restored.source.id
        void rehydrateViews(selectedRun)
        void rehydratePreparedCandidate(selectedRun)
      }
      setSource(restored.source)
      setOutput(restored.output)
    } catch (reason) {
      if (contextVersionRef.current === contextVersion) setError(reason instanceof Error ? reason.message : 'Could not open journey workspace')
    }
  }

  const selectJourney = async (runId: string) => {
    const contextVersion = contextVersionRef.current + 1
    contextVersionRef.current = contextVersion
    activeRunIdRef.current = runId
    setRunsLoading(true)
    setError('')
    setPinnedArtifactId(null)
    // A different run's back/left/right ids never apply to this one's front
    // image -- force openJourneyWorkspace's clear+rehydrate below rather
    // than risk it reading a stale "same front, keep it" comparison.
    viewsSourceIdRef.current = null
    try {
      // GET /runs (the list/selector index) is summarized server-side — it
      // blanks attempt.settings/logs/warnings/error to keep that payload
      // small (see backend/app/run_store.py summarize_run). The `runs`
      // state array below is built from that list, so it — and any
      // previously-cached entry in it — must never be used as the active
      // run: the Animations list needs the real settings.animation on each
      // attempt (and other stage-detail views need the rest). Always fetch
      // the full run detail here instead.
      const selectedRun = await json<RunManifest>(`/runs/${runId}`)
      if (contextVersionRef.current !== contextVersion || activeRunIdRef.current !== runId) return
      window.localStorage.setItem(ACTIVE_RUN_STORAGE_KEY, runId)
      setActiveRun(selectedRun)
      setRuns((current) => current.some((run) => run.id === runId)
        ? current.map((run) => run.id === runId ? selectedRun : run)
        : [selectedRun, ...current])
      const stepId = tab === 'rig' ? 'rig' : ['retopologize', 'paint'].includes(tab) ? 'clean' : 'generate'
      await openJourneyWorkspace(stepId, selectedRun)
    } catch (reason) {
      if (contextVersionRef.current === contextVersion) setError(reason instanceof Error ? reason.message : 'Could not load character journey')
    } finally {
      if (activeRunIdRef.current === runId) setRunsLoading(false)
    }
  }

  const startNewJourney = () => {
    contextVersionRef.current += 1
    activeRunIdRef.current = null
    window.localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY)
    window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY)
    setActiveRun(null)
    setSource(null)
    setOutput(null)
    setPreparedCandidate(null)
    setRefineImage(null)
    setPaintStrokes([])
    setViews({})
    viewsSourceIdRef.current = null
    setPinnedArtifactId(null)
    setSubjectKind('auto')
    resetPendingPreview()
    setView('studio')
    setTab('generate')
  }

  const refreshCaps = () => json<Capabilities>('/system/capabilities').then(setCaps).catch(() => setError('Start the Python worker on port 8000 to connect the local studio.'))
  // Tab-visibility recovery. Switching away from a long job (auto-rig) used to
  // leave the studio looking permanently idle: the job id lives only in React
  // state, so anything that remounted the app -- a reload, Chrome's Memory
  // Saver discarding the backgrounded tab, a dev-server reconnect -- dropped
  // it while the job kept running server-side, and its completion never
  // landed. The run manifest goes stale the same way (an Animate tab kept
  // saying "No animated clips yet" while the server had three). Re-read both
  // on mount, on window focus, and whenever the tab becomes visible again --
  // the same rehydrate-on-focus pattern the view-grid / preparedCandidate
  // rehydration already established.
  const resyncActiveJob = async (trigger: 'mount' | 'focus' | 'visibilitychange') => {
    if (!shouldResyncActiveJob(trigger, document.visibilityState)) return
    const runId = activeRunIdRef.current
    if (runId) {
      const freshRun = await json<RunManifest>(`/runs/${runId}`).catch(() => null)
      if (freshRun && activeRunIdRef.current === runId) setActiveRun(freshRun)
    }
    const remembered = readRememberedJob(window.localStorage.getItem(ACTIVE_JOB_STORAGE_KEY))
    const currentJobId = activeJobIdRef.current
    const jobId = activeJobResyncId(currentJobId ? { id: currentJobId } : null, remembered)
    if (!jobId) return
    const fresh = await json<Job>(`/jobs/${jobId}`).catch(() => null)
    const runIdForDecision = activeRunIdRef.current
    setJob((current) => activeJobResyncDecision(fresh, current, runIdForDecision) === 'ignore' ? current : fresh)
  }
  useEffect(() => {
    const onFocus = () => { void resyncActiveJob('focus') }
    const onVisibility = () => { void resyncActiveJob('visibilitychange') }
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])
  useEffect(() => {
    // Always fetch once on mount regardless of visibility — the tab is by
    // definition visible to whoever just navigated to it, and gating the very
    // first request on document.visibilityState risks never firing it at all
    // in a background/prerendered load.
    refreshCaps()
    const refreshWhenVisible = () => {
      if (shouldRefreshCapabilities(document.visibilityState)) refreshCaps()
    }
    const timer = window.setInterval(refreshWhenVisible, 10_000)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [])
  // Header popovers: close on Escape, on an outside click, and whenever the
  // route (studio/history view or active tab) changes under them.
  useEffect(() => { setToolsMenuOpen(false); setSystemPopoverOpen(false) }, [view, tab])
  useEffect(() => {
    if (!toolsMenuOpen && !systemPopoverOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setToolsMenuOpen(false)
      setSystemPopoverOpen(false)
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node
      if (toolsMenuOpen && toolsMenuRef.current && !toolsMenuRef.current.contains(target)) setToolsMenuOpen(false)
      if (systemPopoverOpen && systemPopoverRef.current && !systemPopoverRef.current.contains(target)) setSystemPopoverOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('pointerdown', onPointerDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('pointerdown', onPointerDown)
    }
  }, [toolsMenuOpen, systemPopoverOpen])
  useEffect(() => {
    let cancelled = false
    json<RunManifest[]>('/runs').then(async (durableRuns) => {
      if (cancelled) return
      setRuns(durableRuns)
      const restoredRunId = shouldRestoreRun(window.localStorage.getItem(ACTIVE_RUN_STORAGE_KEY), durableRuns)
      const restoredRun = durableRuns.find((run) => run.id === restoredRunId)
      if (restoredRunId && restoredRun) await selectJourney(restoredRunId)
      if (!cancelled) await resyncActiveJob('mount')
    }).catch((reason) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load character journeys')
    }).finally(() => {
      if (!cancelled) setRunsLoading(false)
    })
    return () => { cancelled = true }
  }, [])
  useEffect(() => { const active = downloads.filter((item) => ['queued', 'running'].includes(item.status)); if (!active.length) return; const timer = window.setInterval(() => { Promise.all(active.map((item) => json<DownloadState>(`/downloads/${item.id}`))).then((fresh) => { setDownloads((current) => current.map((item) => fresh.find((next) => next.id === item.id) || item)); if (fresh.some((item) => item.status === 'completed')) refreshCaps() }).catch(() => undefined) }, 700); return () => window.clearInterval(timer) }, [downloads])
  // Poll the active job until it is terminal. Keyed on the job *id* and
  // whether it is still active -- not on the whole `job` object, which the
  // poll itself replaces on every tick and which used to tear the interval
  // down and rebuild it on every response. The loop reschedules itself with
  // a visibility-aware cadence (jobPollIntervalMs) so a hidden tab keeps
  // polling at a throttle-friendly rate instead of orphaning a running job's
  // UI state, and polls immediately the moment the tab comes back.
  const activeJobId = job?.id
  const activeJobRunning = hasActiveStudioJob(job)
  useEffect(() => {
    if (!activeJobId || !activeJobRunning) return
    let cancelled = false
    let timer = 0
    const readOnce = () => json<Job>(`/jobs/${activeJobId}`)
      .then((fresh) => { if (!cancelled) setJob((current) => current?.id === fresh.id ? fresh : current) })
      .catch(() => undefined)
    const schedule = () => {
      if (cancelled) return
      timer = window.setTimeout(() => { void readOnce().then(schedule) }, jobPollIntervalMs(document.visibilityState))
    }
    const pollNow = () => { window.clearTimeout(timer); void readOnce().then(schedule) }
    schedule()
    document.addEventListener('visibilitychange', pollNow)
    window.addEventListener('focus', pollNow)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', pollNow)
      window.removeEventListener('focus', pollNow)
    }
  }, [activeJobId, activeJobRunning])
  // Persist the in-flight job so a remount can find it again (resyncActiveJob
  // below). Only ever while it is actually running: once a job is terminal in
  // this tab there is nothing left to recover, and a stale entry would let a
  // later focus event re-adopt an already-finished job.
  useEffect(() => {
    if (!job) return
    if (hasActiveStudioJob(job)) {
      window.localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, serializeRememberedJob(job.id, job.run_id || activeRunIdRef.current))
      return
    }
    const remembered = readRememberedJob(window.localStorage.getItem(ACTIVE_JOB_STORAGE_KEY))
    if (remembered?.jobId === job.id) window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY)
  }, [job?.id, job?.status, job?.run_id])
  useEffect(() => {
    if (job?.run_id && activeRun && activeRun.id !== job.run_id) return
    if (job?.status !== 'completed' || !job.output_artifact_id) return
    const completedJobId = job.id
    if (terminalHydrationJobIdRef.current || terminalHydratedJobIdRef.current === completedJobId) return
    if (!shouldStartTerminalHydration(completedJobId, terminalHydratedJobIdRef.current, terminalHydrationFailedJobId, terminalHydrationRetryJobId)) return
    if (terminalHydrationRetryJobId === completedJobId) setTerminalHydrationRetryJobId(null)
    terminalHydrationJobIdRef.current = completedJobId
    setTerminalHydrationJobId(completedJobId)
    const artifactId = job.output_artifact_id
    const completedContextVersion = contextVersionRef.current
    const expectedRun = activeRun && job.run_id && activeRun.id === job.run_id ? activeRun : null
    let hydrationSucceeded = false
    const remainsActive = () => shouldApplyCompletedArtifact(
      completedJobId,
      completedContextVersion,
      activeJobIdRef.current,
      contextVersionRef.current,
    )
    const hydrationController = new AbortController()
    const hydrationTimeout = window.setTimeout(() => hydrationController.abort(), TERMINAL_HYDRATION_TIMEOUT_MS)
    const artifactRequest = json<Artifact>(`/assets/${artifactId}`, { signal: hydrationController.signal })
    const runRequest = job.run_id
      ? json<RunManifest>(`/runs/${job.run_id}`, { signal: hydrationController.signal }).catch((reason) => {
        if (hydrationController.signal.aborted) throw reason
        return null
      })
      : Promise.resolve<RunManifest | null>(null)
    Promise.all([artifactRequest, runRequest]).then(async ([artifact, run]) => {
      const stillActive = remainsActive()
      if (!stillActive) return
      // A new job output just landed for this run — an older pin no longer
      // reflects the operator's intent and must not silently keep winning.
      setPinnedArtifactId(null)
      if (run && expectedRun && shouldApplyHydratedRun(run, expectedRun, artifact.id, shouldPromoteCompletedOutput(job), activeRunIdRef.current)) setActiveRun(run)
      else if (run && !expectedRun && activeRunIdRef.current === null && run.id === job.run_id) setActiveRun(run)
      if (artifact.kind === 'image' && isPreparedFrontStage(job.kind)) {
        setPreparedCandidate({ ...artifact, stage: job.kind })
        setOutput({ ...artifact, stage: job.kind })
        if (job.kind === 'create-tpose-set') {
          // The composite's output IS the set's front, and its metadata names
          // the three views it fanned off that front. They REPLACE whatever
          // was in the slots: a freshly built set is authoritative, unlike
          // Auto-generate-views which only ever filled the empty ones.
          const entries = synthesizedViewArtifactIds(artifact.id, artifact.metadata)
          const fetched = await Promise.all(entries.map(async ([label, id]) => (
            [label, await json<Artifact>(`/assets/${id}`)] as const
          )))
          if (!remainsActive()) return
          setViews(Object.fromEntries(fetched) as Partial<Record<OptionalViewLabel, Artifact>>)
          setTposeSetFrontArtifact({ ...artifact, stage: job.kind })
          viewsSourceIdRef.current = artifact.id
        }
      } else if (artifact.kind === 'glb' && ['auto-rig', 'rig-adjust', 'retarget-motion', 'generate-motion', 'apply-clip'].includes(job.kind)) {
        // motion/rig stages always surface their output in the workspace;
        // retention still governs History promotion
        setOutput({ ...artifact, stage: job.kind })
      } else if (artifact.kind === 'glb' && shouldPromoteCompletedOutput(job)) {
        setOutput({ ...artifact, stage: job.kind })
      } else if (artifact.kind === 'bvh' && job.kind === 'generate-motion' && job.run_id && activeSource?.kind === 'glb') {
        const retarget = await postJsonWithNetworkRetry<Job>(
          '/jobs',
          retargetMotionJobPayload(activeSource.id, artifact.id, job.run_id),
          hydrationController.signal,
        )
        if (!remainsActive()) return
        setJob(retarget)
      }
      hydrationSucceeded = true
    }).catch((reason) => {
      if (!remainsActive()) return
      setTerminalHydrationFailedJobId(completedJobId)
      setError(hydrationController.signal.aborted
        ? 'Completed job hydration timed out; the workspace is unlocked for retry.'
        : reason instanceof Error ? reason.message : 'Could not finish loading the completed job')
    })
      .finally(() => {
        window.clearTimeout(hydrationTimeout)
        if (terminalHydrationJobIdRef.current !== completedJobId) return
        hydrationController.abort()
        terminalHydrationJobIdRef.current = null
        if (shouldMarkTerminalHydrationComplete(hydrationSucceeded, remainsActive())) {
          terminalHydratedJobIdRef.current = completedJobId
          setTerminalHydrationFailedJobId((current) => current === completedJobId ? null : current)
        }
        setTerminalHydrationJobId((current) => current === completedJobId ? null : current)
      })
  }, [job?.id, job?.kind, job?.status, job?.output_artifact_id, job?.run_id, activeRun?.id, activeSource?.id, activeSource?.kind, terminalHydrationRetryJobId])
  useEffect(() => { if (job?.kind === 'generate' && (['failed', 'cancelled'].includes(job?.status || '') || (job.status === 'completed' && job.promoted_output === false))) { setGenerationPendingPreview(false); setGenerationStartedAt(null) } }, [job?.kind, job?.status, job?.promoted_output])
  useEffect(() => { if (!generationStartedAt || !generationPendingPreview) return; const timer = window.setInterval(() => setGenerationElapsed(Date.now() - generationStartedAt), 250); return () => window.clearInterval(timer) }, [generationStartedAt, generationPendingPreview])
  useEffect(() => { if (job?.kind === 'retopologize' && (['failed', 'cancelled'].includes(job?.status || '') || (job.status === 'completed' && job.promoted_output === false))) { setRefinePendingPreview(false); setRefineStartedAt(null) } }, [job?.kind, job?.status, job?.promoted_output])
  useEffect(() => { if (!refineStartedAt || !refinePendingPreview) return; const timer = window.setInterval(() => setRefineElapsed(Date.now() - refineStartedAt), 250); return () => window.clearInterval(timer) }, [refineStartedAt, refinePendingPreview])
  useEffect(() => { if (job?.kind === 'retopo' && (['failed', 'cancelled'].includes(job?.status || '') || (job.status === 'completed' && job.promoted_output === false))) { setRetopoPendingPreview(false); setRetopoStartedAt(null) } }, [job?.kind, job?.status, job?.promoted_output])
  useEffect(() => { if (!retopoStartedAt || !retopoPendingPreview) return; const timer = window.setInterval(() => setRetopoElapsed(Date.now() - retopoStartedAt), 250); return () => window.clearInterval(timer) }, [retopoStartedAt, retopoPendingPreview])
  // r029o: read the canonicalize gate's verdict off the completed job's output
  // artifact. This runs for a blocked candidate too — main.py sets
  // job.promoted_output=false there, so the shared terminal-hydration effect
  // never surfaces it as `output`, and the findings would otherwise be
  // invisible to the operator.
  useEffect(() => {
    if (job?.kind !== 'canonicalize' || job.status !== 'completed' || !job.output_artifact_id) return
    if (canonicalResult?.jobId === job.id) return
    const jobId = job.id
    const artifactId = job.output_artifact_id
    const promoted = job.promoted_output !== false
    let cancelled = false
    json<Artifact>(`/assets/${artifactId}`).then((artifact) => {
      if (cancelled) return
      const verdict = canonicalizeVerdict(artifact.metadata)
      if (!verdict) return
      const density = canonicalizeDensityReport(artifact.metadata)
      // S2 view-projection (bead s2-view-projection): when this canonicalize
      // run was given resolvable T-pose conditioning views, native.py bakes
      // them onto the same artifact and stamps its own verdict alongside
      // canonicalize's -- surfaced here rather than through a separate job
      // watcher since it is the same completed job's output.
      const projectVerdict = projectionVerdict(artifact.metadata)
      const projection = projectVerdict
        ? { verdict: projectVerdict, report: projectionReport(artifact.metadata) }
        : null
      const fidelityVerdict = sourceFidelityVerdict(artifact.metadata)
      const sourceFidelity = fidelityVerdict
        ? { verdict: fidelityVerdict, report: sourceFidelityReport(artifact.metadata) }
        : null
      setCanonicalResult({ jobId, artifactId, artifactName: artifact.name, verdict, promoted, density, projection, sourceFidelity })
    }).catch(() => undefined)
    return () => { cancelled = true }
  }, [job?.id, job?.kind, job?.status, job?.output_artifact_id, job?.promoted_output, canonicalResult?.jobId])
  // The rebuild-views result card's numbers: the deliverable is promoted like
  // any other Clean output, so this only carries what to show, not a gate.
  useEffect(() => {
    if (job?.kind !== 'rebuild-views' || job.status !== 'completed' || !job.output_artifact_id) return
    if (rebuildResult?.jobId === job.id) return
    const jobId = job.id
    let cancelled = false
    json<Artifact>(`/assets/${job.output_artifact_id}`).then((artifact) => {
      if (!cancelled) setRebuildResult({ jobId, report: rebuildViewsReport(artifact.metadata) })
    }).catch(() => undefined)
    return () => { cancelled = true }
  }, [job?.id, job?.kind, job?.status, job?.output_artifact_id, rebuildResult?.jobId])
  // A different journey's canonicalize verdict must never linger on screen.
  useEffect(() => { setCanonicalResult(null); setRebuildResult(null) }, [activeRun?.id])
  useEffect(() => { if (!displayAsset || displayAsset.kind !== 'glb') { setModelDetails(null); return }; json<ModelDetails>(`/assets/${displayAsset.id}/details`).then(setModelDetails).catch(() => setModelDetails(null)) }, [displayAsset?.id, displayAsset?.kind])
  useEffect(() => { setExportState({ status: 'idle' }) }, [displayAsset?.id])
  // Read-only "Views: front · back" chips (E2): the label list a completed
  // `generate` attempt's output artifact carries, independent of whichever
  // stage the journey is currently showing.
  useEffect(() => {
    const attempt = latestCompletedAttempt(activeRun?.attempts, 'generate')
    if (!attempt?.output_artifact_id) { setSourceViewLabels([]); setGenerateSourceFidelity(null); return }
    let cancelled = false
    json<Artifact>(`/assets/${attempt.output_artifact_id}`)
      .then((artifact) => {
        if (cancelled) return
        setSourceViewLabels(sourceViewChipLabels(artifact.metadata?.views))
        const verdict = sourceFidelityVerdict(artifact.metadata)
        setGenerateSourceFidelity(verdict ? { verdict, report: sourceFidelityReport(artifact.metadata) } : null)
      })
      .catch(() => { if (!cancelled) { setSourceViewLabels([]); setGenerateSourceFidelity(null) } })
    return () => { cancelled = true }
  }, [activeRun?.id, activeRun?.attempts])
  useEffect(() => { if (tab !== 'retopologize' || cleanMethod !== 'ai-refine' || !activeSource || !modelDetails || displayAsset?.id !== activeSource.id || refineDefaultedAsset.current === activeSource.id) return; setRefineTargetVertices(modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))); refineDefaultedAsset.current = activeSource.id }, [tab, cleanMethod, activeSource?.id, displayAsset?.id, modelDetails?.vertices])
  useEffect(() => { if (tab !== 'retopologize' || cleanMethod !== 'instant-meshes' || !activeSource || !modelDetails || displayAsset?.id !== activeSource.id || retopoDefaultedAsset.current === activeSource.id) return; setRetopoTargetVertices(modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))); retopoDefaultedAsset.current = activeSource.id }, [tab, cleanMethod, activeSource?.id, displayAsset?.id, modelDetails?.vertices])
 
  // I1: lazily populate the Working-on picker's names/vertex counts for the
  // stage panel currently open, once per artifact id (fetchAssetDetails
  // itself de-dupes via fetchedAssetDetailIdsRef).
  useEffect(() => {
    // The 'rig' tab covers three different Working-on cards (pre-rig mesh,
    // rig candidate, animate) depending on state that isn't known statically
    // here — fetch all of them; the extra lookups are cheap and
    // fetchAssetDetails already de-dupes by artifact id.
    const stages = tab === 'generate' ? ['generate' as const]
      : tab === 'paint' ? ['paint' as const]
      : tab === 'retopologize' ? ['clean' as const]
      : tab === 'rig' ? ['skeleton-pre-rig' as const, 'skeleton' as const, 'animate' as const]
      : []
    for (const stage of stages) for (const option of runAssetOptions(activeRun, stage)) fetchAssetDetails(option.artifactId)
  }, [activeRun?.id, activeRun?.attempts, tab])

  const gpuPercent = useMemo(() => caps?.gpu.vram_total_mb ? Math.min(100, (caps.gpu.vram_used_mb / caps.gpu.vram_total_mb) * 100) : 0, [caps])
  const ramPercent = useMemo(() => caps?.ram?.total_mb ? Math.min(100, (caps.ram.used_mb / caps.ram.total_mb) * 100) : 0, [caps])
  const activeTab = tabs.find((item) => item.id === tab)!
  const requiredDownload = useMemo(() => {
    if (!caps || tab === 'paint' || tab === 'rig') return null
    if (tab === 'generate') {
      return { name: 'trellis-refiner', label: 'Install 3D model', ready: Boolean(caps.models.trellis_refiner_ready || caps.models.trellis_gguf_ready), detail: 'Download the model to generate 3D assets.' }
    }
    if (tab === 'retopologize') {
      // The canonical path runs entirely in Blender (native.run_canonicalize's
      // voxel remesh + bake), so there is no model to download for it.
      if (cleanMethod === 'canonicalize') return null
      return cleanMethod === 'ai-refine'
        ? { name: 'trellis2', label: 'Refine model', ready: Boolean(caps.models.trellis_refiner_ready), detail: 'Download the model to refine an asset.' }
        : { name: 'instant-meshes', label: 'Retopology engine', ready: Boolean(caps.models.instant_meshes_native), detail: 'Download the engine to retopologize an asset.' }
    }
    return null
  }, [caps, tab, cleanMethod])
  function resetPendingPreview() { setJob(null); setGenerationPendingPreview(false); setGenerationStartedAt(null); setRefinePendingPreview(false); setRefineStartedAt(null); setRetopoPendingPreview(false); setRetopoStartedAt(null) }
  const chooseFile = async (file: File | undefined, target: 'input' | 'model' | 'refine-image' | 'view-back' | 'view-left' | 'view-right' = 'input') => { if (!file) return; setError(''); try { const artifact = await upload(file); if (target === 'view-back' || target === 'view-left' || target === 'view-right') { const label = target.slice('view-'.length) as OptionalViewLabel; setViews((current) => ({ ...current, [label]: artifact })); await registerTposeSetMember(label, artifact.id, 'Operator upload replaced this set member') } else if (target === 'refine-image') setRefineImage(artifact); else { contextVersionRef.current += 1; setPreparedCandidate(null); setTposeSetFrontArtifact(null); if (target === 'input' && tab === 'generate') { setSource(artifact); setOutput(null); setActiveRun(null); resetPendingPreview(); setViews({}); viewsSourceIdRef.current = artifact.id } else { setSource(artifact); setOutput(artifact); setActiveRun(null); resetPendingPreview() } } } catch (e) { setError(e instanceof Error ? e.message : 'Upload failed') } }
  const openViewPicker = (label: OptionalViewLabel) => {
    pendingUploadTargetRef.current = `view-${label}`
    if (fileRef.current) fileRef.current.accept = 'image/png,image/jpeg,image/webp'
    fileRef.current?.click()
  }
  const removeView = (label: OptionalViewLabel) => setViews((current) => { const next = { ...current }; delete next[label]; return next })
  // "Replace source" and the viewport's "Import GLB" share one hidden file input;
  // this tracks which of the two triggered the picker so onChange knows which
  // chooseFile() behavior to apply.
  // Set the input's accept filter imperatively at click-time (a React prop
  // update wouldn't land in the DOM until the next render, after click() has
  // already opened the OS file dialog with the stale filter).
  const openSourcePicker = () => {
    pendingUploadTargetRef.current = 'input'
    if (fileRef.current) fileRef.current.accept = tab === 'generate' ? 'image/png,image/jpeg,image/webp' : tab === 'rig' ? '.glb,model/gltf-binary' : '.glb,.gltf,.fbx,.obj'
    fileRef.current?.click()
  }
  const openModelImportPicker = () => {
    pendingUploadTargetRef.current = 'model'
    if (fileRef.current) fileRef.current.accept = '.glb,model/gltf-binary'
    fileRef.current?.click()
  }
  const prepareImage = async (adapter: 'comfyui' | 'upload' = 'comfyui') => {
    if (!source || !isImageArtifactKind(source.kind)) return
    setError(''); setPreparedCandidate(null); setOutput(null); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...preparationJobPayload(source.id, activeRun?.id || null, seed, adapter),
        submission_id: createSubmissionId(),
        run_name: `${source.name.replace(/\.[^.]+$/, '')} preparation`,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not prepare image') } finally { setBusy(false) }
  }
  // Bead 4vquq: the one operator step. Submits the composite job and hands
  // it to the shared job-progress machinery -- the staged messages
  // ("Generating the back view… 2/4") are the job's own, and the terminal
  // hydration effect above lands the front as the candidate awaiting
  // approval and refills the view grid from the set it registered.
  const createTposeSet = async () => {
    if (!source || !isImageArtifactKind(source.kind)) return
    setError(''); setPreparedCandidate(null); setOutput(null); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...createTposeSetJobPayload(source.id, activeRun?.id || null, seed),
        submission_id: createSubmissionId(),
        run_name: `${source.name.replace(/\.[^.]+$/, '')} T-pose set`,
        subject_kind: activeRun ? undefined : subjectKind,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not create the T-pose set') } finally { setBusy(false) }
  }
  const approvePreparedImage = async () => {
    if (!activeRun || !preparedCandidate) return
    setError(''); setBusy(true)
    const runId = activeRun.id
    const contextVersion = contextVersionRef.current
    try {
      const updated = await json<RunManifest>(
        `/runs/${runId}/artifacts/${preparedCandidate.id}/promote`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reason: 'Approved pose, framing, separated limbs, and character identity',
          }),
        },
      )
      if (contextVersionRef.current !== contextVersion || activeRunIdRef.current !== runId) return
      setActiveRun(updated)
      setSource(preparedCandidate)
      setOutput(null)
      setPreparedCandidate(null)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not approve prepared image') } finally { setBusy(false) }
  }
  // Auto-generate views (E3): submits a `synthesize-views` job for the
  // current front image, polls it to completion, then fills only the empty
  // back/left/right slots -- a slot the operator already filled (upload or
  // an earlier synth) is left alone.
  const autoGenerateViews = async (labels: OptionalViewLabel[] = ['back', 'left', 'right'], synthesisSeed = seed) => {
    if (!source || !isImageArtifactKind(source.kind)) return
    setError(''); setSynthesizingViews(true)
    const contextVersion = contextVersionRef.current
    // On a run that owns a canonical T-pose set, every view is drawn from the
    // set's own front -- never from the raw reference the set retired.
    const referenceArtifactId = tposeSetFront?.id
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...synthesizeViewsJobPayload(source.id, activeRun?.id || null, synthesisSeed, labels, referenceArtifactId),
        submission_id: createSubmissionId(),
        run_name: `${source.name.replace(/\.[^.]+$/, '')} view synthesis`,
      })
      if (contextVersionRef.current !== contextVersion) return
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
      const finished = await pollJobUntilTerminal(
        created,
        (id) => json<Job>(`/jobs/${id}`),
        () => new Promise((resolve) => window.setTimeout(resolve, 700)),
        () => contextVersionRef.current === contextVersion,
      )
      if (contextVersionRef.current !== contextVersion) return
      if (finished.status !== 'completed' || !finished.output_artifact_id) {
        setError(finished.error || 'Could not synthesize views')
        return
      }
      const primary = await json<Artifact>(`/assets/${finished.output_artifact_id}`)
      if (contextVersionRef.current !== contextVersion) return
      const entries = synthesizedViewArtifactIds(primary.id, primary.metadata)
      const fetched = await Promise.all(entries.map(async ([label, id]) => (
        [label, id === primary.id ? primary : await json<Artifact>(`/assets/${id}`)] as const
      )))
      if (contextVersionRef.current !== contextVersion) return
      setViews((current) => mergeSynthesizedViews(current, fetched))
      // Keep the SET, not just the browser's slot, in step with what the
      // operator just regenerated -- the next generate is conditioned on the
      // set, so a swap that only landed locally would be silently ignored.
      for (const [label, artifact] of fetched) await registerTposeSetMember(label, artifact.id, 'Regenerated tile')
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not synthesize views') } finally { setSynthesizingViews(false) }
  }
  // Records an operator-chosen image (a regenerated tile, or an upload) as
  // the run's T-pose set member for that role. A no-op when the run has no
  // set: object/creature runs and pre-4vquq runs keep the loose-slot flow.
  const registerTposeSetMember = async (label: OptionalViewLabel, artifactId: string, reason: string) => {
    if (!activeRun || !tposeSet) return
    const runId = activeRun.id
    try {
      const updatedFront = await json<Artifact>(`/runs/${runId}/tpose-set/members/${label}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artifact_id: artifactId, reason }),
      })
      if (activeRunIdRef.current !== runId) return
      // The membership block the studio reads lives on the front artifact.
      setPreparedCandidate((current) => current && current.id === updatedFront.id ? { ...updatedFront, stage: current.stage } : current)
      setSource((current) => current && current.id === updatedFront.id ? { ...updatedFront, stage: current.stage } : current)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not update the T-pose set') }
  }
  // Bead 8c9gf: the remedy a view-consistency notice names. Empties the one
  // offending slot so the synthesis (which only ever fills EMPTY slots) can
  // refill it, and advances the seed first -- a single-view job always uses
  // seed+1, so re-running the same seed would redraw the identical image and
  // the warning would never clear. The new seed lands in the visible Seed
  // field, so the run stays reproducible from what the operator can see.
  const regenerateView = (label: OptionalViewLabel) => {
    const nextSeed = (seed + 1) % 2147483648
    setSeed(nextSeed)
    removeView(label)
    void autoGenerateViews([label], nextSeed)
  }
  // S2 canonical clean (r029o): voxel remesh + bake-from-original behind the
  // blocking canonicalize gate. Submitted through its own payload builder
  // rather than runJob's generic settings blob, because the contract stage
  // takes only `method` (+ an optional target_vertices) and must not inherit
  // the legacy Clean panel's repair_holes / gpu_first / refine-image fields.
  const runCanonicalClean = async () => {
    if (!activeSource) return
    setError(''); setBusy(true); setCanonicalResult(null)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...canonicalizeJobPayload(activeSource.id, activeRun?.id || null, null, 'voxel', canonicalDensity),
        submission_id: createSubmissionId(),
        run_name: source ? `${source.name.replace(/\.[^.]+$/, '')} experiment` : undefined,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start canonical clean') } finally { setBusy(false) }
  }
  // Rebuild from views: approach9's one-command process (paint the mesh's own
  // renders, rebuild with Pixal3D multi-view, lay the real pixels on top).
  // Its own payload builder for the same reason canonicalize has one -- the
  // stage takes no legacy Clean panel settings at all.
  const runRebuildViews = async () => {
    if (!activeSource) return
    setError(''); setBusy(true); setRebuildResult(null)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...rebuildViewsJobPayload(activeSource.id, activeRun?.id || null, seed),
        submission_id: createSubmissionId(),
        run_name: source ? `${source.name.replace(/\.[^.]+$/, '')} experiment` : undefined,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start the rebuild from views') } finally { setBusy(false) }
  }
  // `skipViewConsistency` is the escape hatch for bead 8c9gf's pre-generate
  // gate: the backend refuses a multi-view generate whose conditioning views
  // contradict the front image, and this is how the operator overrides that
  // after looking at the tiles.
  const runJob = async ({ skipViewConsistency = false }: { skipViewConsistency?: boolean } = {}) => {
    if (tab === 'retopologize' && cleanMethod === 'canonicalize') return runCanonicalClean()
    if (tab === 'retopologize' && cleanMethod === 'rebuild-views') return runRebuildViews()
    setError(''); setViewConsistencyBlocked(false); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      const usesInstantMeshes = tab === 'retopologize' && cleanMethod === 'instant-meshes'
      const kind = tab === 'paint' ? 'paint-bake' : tab === 'retopologize' ? (usesInstantMeshes ? 'retopo' : 'retopologize') : tab
      const solverTargetVertices = usesInstantMeshes ? Math.max(500, Math.round(retopoTargetVertices * 0.66)) : refineTargetVertices
      const settings = { color: paintColor, roughness, metallic, strokes: paintStrokes, repair_holes: cleanRepairHoles, gpu_first: gpuFirst, refine_image_artifact_id: refineImage?.id, resolution, seed, target_vertices: solverTargetVertices, ...(skipViewConsistency ? { skip_view_consistency: true } : {}) }
      const submissionId = createSubmissionId()
      // Multi-view conditioning (E2): only `generate` accepts side/back views. Every
      // other job kind keeps today's plain source_artifact_id shape unconditionally.
      const generateFields = kind === 'generate'
        ? generateSourceFields(activeSource?.id, { back: views.back?.id, left: views.left?.id, right: views.right?.id })
        : { source_artifact_id: activeSource?.id }
      const created = await postJsonWithNetworkRetry<Job>('/jobs', { kind, submission_id: submissionId, ...generateFields, resolution, seed, target_vertices: solverTargetVertices, settings, run_id: activeRun?.id, run_name: source ? `${source.name.replace(/\.[^.]+$/, '')} experiment` : undefined })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
      if (kind === 'generate') { const started = Date.now(); setGenerationStartedAt(started); setGenerationElapsed(0); setGenerationTime(null); setGenerationPendingPreview(true) }
      if (kind === 'retopologize') { const started = Date.now(); setRefineStartedAt(started); setRefineElapsed(0); setRefineTime(null); setRefinePendingPreview(true) }
      if (kind === 'retopo') { const started = Date.now(); setRetopoStartedAt(started); setRetopoElapsed(0); setRetopoTime(null); setRetopoPendingPreview(true) }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start job')
      // A 422 naming a contradicting view is the one refusal the operator can
      // deliberately override; every other failure gets the plain error box.
      const status = (e as ApiError | undefined)?.status
      setViewConsistencyBlocked(status === 422 && e instanceof Error && e.message.includes('contradicts the front image'))
    } finally { setBusy(false) }
  }
  const generateAttackMotion = async () => {
    if (!activeSource || activeSource.kind !== 'glb') return
    setError(''); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...motionJobPayload(activeSource.id, activeRun?.id || null, seed),
        submission_id: createSubmissionId(),
        run_name: `${activeSource.name.replace(/\.[^.]+$/, '')} attack`,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not generate attack motion') } finally { setBusy(false) }
  }
  const generateAutoRig = async (targetArtifactId: string) => {
    if (!targetArtifactId) return
    if (!canLaunchAutoRig(caps?.models.unirig_ready ?? null)) {
      setError('UniRig runtime is not installed. Run scripts/setup_unirig.sh on the backend host.')
      return
    }
    if (mesh2MotionExportInFlightRef.current) {
      setError('Finish saving the Mesh2Motion export before starting the automated rig.')
      return
    }
    automatedRigLaunchRef.current = true
    setError(''); setBusy(true)
    const contextVersion = contextVersionRef.current
    const runNameBase = (activeRun?.name || activeSource?.name || 'model').replace(/\.[^.]+$/, '')
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        ...autoRigJobPayload(targetArtifactId, activeRun?.id || null),
        submission_id: createSubmissionId(),
        run_name: `${runNameBase} unirig`,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start the UniRig auto-rig') } finally { automatedRigLaunchRef.current = false; setBusy(false) }
  }
  const saveRigAdjustments = async (adjustment: RigAdjustmentSave) => {
    if (!activeSource || activeSource.kind !== 'glb') return
    setError(''); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      const created = await postJsonWithNetworkRetry<Job>('/jobs', {
        kind: 'rig-adjust' as const,
        source_artifact_id: activeSource.id,
        run_id: activeRun?.id || null,
        submission_id: createSubmissionId(),
        settings: { positions: adjustment.positions, remove_joints: adjustment.removeJoints },
        run_name: `${activeSource.name.replace(/\.[^.]+$/, '')} adjusted`,
      })
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not save joint adjustments') } finally { setBusy(false) }
  }
  const attachCuratedMotion = async (clipAnimation: string = CURATED_MOTION_CLIPS[0].animation) => {
    if (!activeSource || activeSource.kind !== 'glb') return
    setError(''); setBusy(true)
    const contextVersion = contextVersionRef.current
    try {
      // N5: a template-conformant certified rig skips the BVH export +
      // retarget-motion round trip entirely -- the curated clip's exact
      // action is direct-attached by bone name (clipAnimation is already
      // the family clip store's clip_name, e.g. "Sword_Attack_Armature",
      // same string CuratedMotionClip.animation carries for the legacy
      // path). Any other rig keeps the existing retarget path unchanged.
      const created = isTemplateConformantRig(activeSource.metadata)
        ? await postJsonWithNetworkRetry<Job>('/jobs', {
          ...applyClipJobPayload(activeSource.id, activeRun?.id || null, clipAnimation),
          submission_id: createSubmissionId(),
          run_name: `${activeSource.name.replace(/\.[^.]+$/, '')} motion`,
        })
        : await (async () => {
          const bvh = await postJsonWithNetworkRetry<Artifact>('/animations/export-curated-bvh', {
            animation: clipAnimation,
          })
          return postJsonWithNetworkRetry<Job>('/jobs', {
            // The retarget job's settings.animation only names the action
            // Blender bakes onto the retargeted output (see
            // curatedActionLabel) — it must not stay hardcoded to
            // "Sword_Attack" once the picker can attach any curated clip.
            ...retargetMotionJobPayload(activeSource.id, bvh.id, activeRun?.id || null, curatedActionLabel(clipAnimation)),
            submission_id: createSubmissionId(),
            run_name: `${activeSource.name.replace(/\.[^.]+$/, '')} motion`,
          })
        })()
      if (contextVersionRef.current !== contextVersion) return
      setJob(created)
      if (created.run_id) {
        const createdRun = await json<RunManifest>(`/runs/${created.run_id}`).catch(() => null)
        if (createdRun && contextVersionRef.current === contextVersion) setActiveRun(createdRun)
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not attach the curated motion') } finally { setBusy(false) }
  }
  const clearMemory = async () => { try { await json('/system/clear-memory', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'Memory cleanup failed') } }
  const clearRam = async () => { try { await json('/system/clear-ram', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'RAM cleanup failed') } }
  const cancelTask = async () => { if (!job || !['queued', 'running'].includes(job.status)) return; try { setJob(await json<Job>(`/jobs/${job.id}/cancel`, { method: 'POST' })) } catch (e) { setError(e instanceof Error ? e.message : 'Could not cancel task') } }
  const openLocation = async () => { if (displayAsset) await json(`/assets/${displayAsset.id}/open-location`, { method: 'POST' }).catch((e) => setError(e.message)) }
  // Exports whichever artifact is named — the canvas footer's own display
  // asset, or (from the Animate panel) whichever clip is currently selected
  // there, which on the rig tab is always the same artifact as displayAsset.
  const exportArtifact = async (artifactId: string) => {
    const asset = artifactId === displayAsset?.id ? displayAsset : null
    if (!asset) return
    setExportState({ status: 'exporting' })
    try {
      const saved = await json<{ path: string; size_bytes: number }>(`/assets/${asset.id}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name_hint: exportGlbNameHint(activeRun?.name, asset.name) }),
      })
      setExportState({ status: 'saved', path: saved.path })
    } catch (e) {
      // No output folder configured server-side (409) is not a hard failure
      // any more -- fall straight back to a normal browser download so
      // Chrome's own save-dialog/downloads flow picks the destination,
      // same as the always-present "Download" link next to this button.
      if (e instanceof Error && (e as ApiError).status === 409) {
        triggerBrowserDownload(`${API}/assets/${asset.id}/download`)
        setExportState({ status: 'downloaded' })
        return
      }
      setExportState({ status: 'error', error: e instanceof Error ? e.message : 'Could not export GLB' })
    }
  }
  const chooseOutputLocation = async () => { try { await json('/system/output-location/select', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'Could not choose output location') } }
  const startDownload = async (requirement: { name: string; ready: boolean }) => { if (requirement.ready) return; setError(''); try { const state = await json<DownloadState>(`/models/${requirement.name}/download`, { method: 'POST' }); setDownloads((current) => current.some((item) => item.id === state.id) ? current : [...current, state]) } catch (e) { setError(e instanceof Error ? e.message : 'Could not start model download') } }
  useEffect(() => {
    const receiveMesh2MotionExport = async (event: MessageEvent<{ source?: string; type?: string; filename?: string; animations?: string[]; skeleton?: string; exportContents?: string; boneNaming?: string; buffer?: ArrayBuffer; run?: string; input?: string; mode?: string }>) => {
      if (event.origin !== window.location.origin || event.data?.source !== 'aismith-mesh2motion' || event.data.type !== 'export-glb' || !isArrayBufferPayload(event.data.buffer)) return
      const exportingRun = activeRun
      const exportingSource = activeSource
      const exportingMode = rigWorkspaceMode || 'rig'
      const cleanMeshArtifactId = typeof exportingSource?.metadata?.source_artifact_id === 'string'
        ? exportingSource.metadata.source_artifact_id
        : undefined
      const exportingInputArtifactId = exportingSource
        ? mesh2MotionExportInputArtifactId(exportingSource.id, cleanMeshArtifactId, exportingMode)
        : ''
      if (!exportingRun || !exportingSource || !canAcceptMesh2MotionContext(event.data, { runId: exportingRun.id, artifactId: exportingInputArtifactId, mode: exportingMode })) {
        setError('Ignored an export from a stale character or workspace. Reopen the active pipeline stage and export again.')
        return
      }
      if (mesh2MotionExportInFlightRef.current) {
        setError('A Mesh2Motion export is already being saved. Wait for it to finish before exporting again.')
        return
      }
      const terminalHydrating = terminalHydrationJobIdRef.current !== null
      if (!canAcceptMesh2MotionExport(busy, job, terminalHydrating) || automatedRigLaunchRef.current) {
        setError(terminalHydrating
          ? 'The automated rig is finishing. Wait for its output to load before exporting from Mesh2Motion.'
          : 'A studio job is in progress. Cancel it or wait for it to finish before exporting from Mesh2Motion.')
        return
      }
      mesh2MotionExportInFlightRef.current = true
      try {
        setError('')
        const contextVersion = contextVersionRef.current
        const artifact = await upload(new File([event.data.buffer], event.data.filename || 'rigged-animated-model.glb', { type: 'model/gltf-binary' }))
        if (exportingSource) {
          const stage = exportingMode === 'animate' ? 'retarget-motion' : 'rig'
          const updated = await postJsonWithNetworkRetry<RunManifest>('/runs/record-artifact', { submission_id: createSubmissionId(), stage, input_artifact_id: exportingInputArtifactId, output_artifact_id: artifact.id, run_id: exportingRun?.id, run_name: `${exportingSource.name.replace(/\.[^.]+$/, '')} ${exportingMode}`, settings: { skeleton: event.data.skeleton || 'Unknown', animations: event.data.animations || [], export_contents: event.data.exportContents || 'Unknown', bone_naming: event.data.boneNaming || 'Unknown' }, logs: [`Embedded ${(event.data.animations || []).join(', ') || 'selected animations'}`] })
          const certifiedArtifact = await json<Artifact>(`/assets/${artifact.id}`)
          if (contextVersionRef.current === contextVersion && activeRunIdRef.current === (exportingRun?.id || null)) {
            if (shouldPreviewRecordedArtifact(artifact.id, certifiedArtifact.id)) setOutput({ ...certifiedArtifact, stage })
            setActiveRun(updated)
          }
        }
      } catch (e) { setError(e instanceof Error ? e.message : 'Could not save the Mesh2Motion export') } finally { mesh2MotionExportInFlightRef.current = false }
    }
    window.addEventListener('message', receiveMesh2MotionExport)
    return () => window.removeEventListener('message', receiveMesh2MotionExport)
  }, [activeRun, activeSource, busy, job, rigWorkspaceMode])

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Box size={19} /></div><div><strong>AISMITH 3D</strong><span>LOCAL 3D STUDIO</span></div></div>
      <div className="top-tabs">
        <button className={`top-tab ${view === 'studio' ? 'active' : ''}`} onClick={() => setView('studio')}><Sparkles size={15} />Studio</button>
        <button className={`top-tab ${view === 'history' ? 'active' : ''}`} onClick={() => setView('history')}><FolderClock size={15} />History</button>
      </div>
      <div className="top-actions">
        <div className="header-memory" aria-label="Local resource usage">
          <div className="header-memory-item"><span>RAM</span><strong>{caps ? formatGbPair(caps.ram.used_mb, caps.ram.total_mb) : '–'}</strong><div className="memory-track"><i className={ramPercent > 80 ? 'danger' : ramPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, ramPercent)}%` }} /></div></div>
          <div className="header-memory-item"><span>VRAM</span><strong>{caps ? formatGbPair(caps.gpu.vram_used_mb, caps.gpu.vram_total_mb) : '–'}</strong><div className="memory-track"><i className={gpuPercent > 80 ? 'danger' : gpuPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, gpuPercent)}%` }} /></div></div>
        </div>
        <div className="connection"><span className={`status-dot ${caps ? 'on' : ''}`} /> {caps ? 'Local worker ready' : 'Connecting'}</div>
        <div className="header-menu" ref={toolsMenuRef}>
          <button className="icon-button" title="Tools" aria-haspopup="menu" aria-expanded={toolsMenuOpen} onClick={() => { setSystemPopoverOpen(false); setToolsMenuOpen((open) => !open) }}><Paintbrush size={16} /></button>
          {toolsMenuOpen && <div className="header-dropdown" role="menu">
            <button role="menuitem" onClick={() => { setView('studio'); setTab('paint'); setToolsMenuOpen(false) }}><Paintbrush size={14} /> Texture Paint</button>
            <button role="menuitem" disabled={!(displayAsset && displayAsset.kind === 'glb')} onClick={() => { setView('mesh2motion'); setToolsMenuOpen(false) }}><Wrench size={14} /> Mesh2Motion (manual)</button>
          </div>}
        </div>
        <div className="header-menu" ref={systemPopoverRef}>
          <button className="icon-button" title="System" aria-haspopup="menu" aria-expanded={systemPopoverOpen} onClick={() => { setToolsMenuOpen(false); setSystemPopoverOpen((open) => !open) }}><Gauge size={16} /></button>
          {systemPopoverOpen && <div className="header-dropdown system-popover" role="menu">
            <div className="memory"><MemoryStick size={14} /><span>RAM</span><strong>{caps ? `${formatMb(caps.ram.used_mb)} / ${formatMb(caps.ram.total_mb)}` : '—'}</strong><div className="memory-track"><i className={ramPercent > 80 ? 'danger' : ramPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, ramPercent)}%` }} /></div></div>
            <button className="quiet-button" onClick={clearRam}><RefreshCw size={13} /> Clear RAM</button>
            <div className="memory"><MemoryStick size={14} /><span>VRAM</span><strong>{caps ? `${formatMb(caps.gpu.vram_used_mb)} / ${formatMb(caps.gpu.vram_total_mb)}` : '—'}</strong><div className="memory-track"><i className={gpuPercent > 80 ? 'danger' : gpuPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, gpuPercent)}%` }} /></div></div>
            <button className="quiet-button" onClick={clearMemory}><RefreshCw size={13} /> Clear VRAM</button>
          </div>}
        </div>
      </div>
    </header>
    <div className="pipeline-topbar">
      <CharacterJourneySelector runs={runs} activeRunId={activeRun?.id || null} loading={runsLoading} onSelect={selectJourney} onNew={startNewJourney} archivedRunIds={archivedRunIds} />
      {view === 'studio' && <CharacterJourney run={activeRun} activeTab={tab} onNavigate={openJourneyWorkspace} />}
    </div>
    {view === 'history' ? <HistoryView initialRunId={activeRun?.id} blenderAvailable={Boolean(caps?.blender.available)} onUseArtifact={(artifact, stage, run, isSource = false) => { const wasPromoted = run.attempts.some((attempt) => attempt.output_artifact_id === artifact.id && attempt.promoted); if (!isSource && !canRestoreHistoryArtifact(stage, artifact.id, run.current_artifact_id, wasPromoted)) return; contextVersionRef.current += 1; const sourceKind = run.source_artifact_kind; const targetTab = isSource ? studioTabForSourceKind(sourceKind) : studioTabForStage(stage); const restoredSource = isSource ? artifact : { id: run.source_artifact_id, name: 'Restored source', kind: sourceKind, size_bytes: 0 }; const restoredArtifact = { ...artifact, stage: isSource ? undefined : stage }; const restored = restoreHistoryArtifacts(stage, isSource, restoredSource, restoredArtifact); if (targetTab === 'retopologize') setCleanMethod(cleanMethodForStage(stage)); setActiveRun(run); resetPendingPreview(); setPreparedCandidate(null); setRefineImage(shouldRestoreRefineImage(stage, sourceKind, isSource) ? restoredSource : null); setPaintStrokes([]); setViews({}); viewsSourceIdRef.current = restored.source.id; if (targetTab === 'generate') { void rehydrateViews(run); void rehydratePreparedCandidate(run) }; setSource(restored.source); setOutput(restored.output); setTab(targetTab); setView('studio') }} /> : view === 'mesh2motion' ? <Mesh2MotionManualView source={displayAsset} runId={activeRun?.id || null} onBack={() => setView('studio')} /> : <>
    <input ref={fileRef} type="file" hidden accept="image/png,image/jpeg,image/webp" onChange={(e) => chooseFile(e.target.files?.[0], pendingUploadTargetRef.current)} />
    <main className={tab === 'rig' ? 'rig-workspace' : 'workspace'}>
      {tab === 'rig' ? <RigAnimateWorkspace mode={rigWorkspaceMode} runId={activeRun?.id || null} run={activeRun} source={activeSource} modelDetails={modelDetails} checkpoint={activeRun?.checkpoints?.skeleton} reviewStatus={activeRun?.review_status} reviewedArtifactId={activeRun?.reviewed_artifact_id} onApprove={handBackSkeleton} onReview={reviewAnimatedArtifact} onUpload={openSourcePicker} onGenerateMotion={generateAttackMotion} onGenerateAutoRig={generateAutoRig} unirigReady={caps?.models.unirig_ready ?? null} onAttachCuratedMotion={attachCuratedMotion} kimodoReady={Boolean(caps?.models.kimodo_ready)} onSaveAdjustments={saveRigAdjustments} onCancel={cancelTask} motionReady={Boolean(activeSource && canGenerateMotion(activeSource, Boolean(caps?.models.kimodo_ready), Boolean(caps?.blender.available)))} blenderAvailable={caps?.blender.available ?? null} busy={busy} job={job} terminalHydrating={terminalHydrationJobId !== null} terminalHydrationError={terminalHydrationFailedJobId === job?.id ? error : ''} onRetryTerminalHydration={retryTerminalHydration} pinnedArtifactId={pinnedArtifactId} onPinArtifact={pinArtifact} detailsById={assetDetailsById} onExport={exportArtifact} exportState={exportState} /> : <>
      <aside className="control-panel">
        <div className="panel-heading"><h1>{activeTab.label}</h1></div>
        <div className="panel-scroll">
          {tab === 'generate' && <><GenerateControls resolution={resolution} setResolution={setResolution} seed={seed} setSeed={setSeed} source={source} views={views} sourceViewLabels={sourceViewLabels} onUploadView={openViewPicker} onRemoveView={removeView} onAutoGenerateViews={autoGenerateViews} onRegenerateView={regenerateView} autoGeneratingViews={synthesizingViews} preparedCandidate={preparedCandidate} preparing={preparationActive} outputDirectory={caps?.output_directory} onChooseOutput={chooseOutputLocation} onUpload={openSourcePicker} onPrepare={prepareImage} onApprove={approvePreparedImage} onCreateTposeSet={createTposeSet} creatingTposeSet={creatingTposeSet} tposeSetProgress={tposeSetProgress} tposeSetFront={tposeSetFront} tposeSet={tposeSet} subjectKind={subjectKind} onSubjectKindChange={changeSubjectKind} run={activeRun} pinnedArtifactId={pinnedArtifactId} onPinArtifact={pinArtifact} detailsById={assetDetailsById} />{generateSourceFidelity && <SourceFidelityCard verdict={generateSourceFidelity.verdict} report={generateSourceFidelity.report} />}{preparationActive ? <button className="secondary-button cancel-task" onClick={cancelTask}>Cancel image preparation</button> : source && !preparedCandidate && <button className="secondary-button" onClick={() => prepareImage('upload')}>Review uploaded image without ComfyUI</button>}</>}
        {tab === 'retopologize' && <CleanControls
          method={cleanMethod} setMethod={setCleanMethod}
          targetVertices={cleanMethod === 'ai-refine' ? refineTargetVertices : retopoTargetVertices}
          setTargetVertices={cleanMethod === 'ai-refine' ? setRefineTargetVertices : setRetopoTargetVertices}
          repairHoles={cleanRepairHoles} setRepairHoles={setCleanRepairHoles}
          gpuFirst={gpuFirst} setGpuFirst={setGpuFirst}
          suggestedVertices={modelDetails ? (modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))) : null}
          source={activeSource} refineImage={refineImage} onAttachImage={() => refineImageRef.current?.click()} onUpload={openSourcePicker}
          canonicalResult={canonicalResult}
          rebuildResult={rebuildResult?.report ?? null}
          density={canonicalDensity} setDensity={setCanonicalDensity}
          rebuildViewsAvailable={Boolean(caps?.models.rebuild_views_ready)}
          run={activeRun} pinnedArtifactId={pinnedArtifactId} onPinArtifact={pinArtifact} detailsById={assetDetailsById}
        />}
        {tab === 'paint' && <PaintControls color={paintColor} setColor={setPaintColor} roughness={roughness} setRoughness={setRoughness} metallic={metallic} setMetallic={setMetallic} brushSize={brushSize} setBrushSize={setBrushSize} brushOpacity={brushOpacity} setBrushOpacity={setBrushOpacity} brushHardness={brushHardness} setBrushHardness={setBrushHardness} strokes={paintStrokes} setStrokes={setPaintStrokes} source={activeSource} onUpload={openSourcePicker} run={activeRun} pinnedArtifactId={pinnedArtifactId} onPinArtifact={pinArtifact} detailsById={assetDetailsById} />}
          <input ref={refineImageRef} type="file" hidden accept="image/png,image/jpeg,image/webp" onChange={(e) => chooseFile(e.target.files?.[0], 'refine-image')} />
        </div>
        <div className="panel-footer">
        {error && <div className="error-box"><X size={14} /><span>{error}</span>{terminalHydrationFailedJobId === job?.id && <button className="secondary-button retry-hydration-button" onClick={retryTerminalHydration}>Retry loading output</button>}{viewConsistencyBlocked && <button className="secondary-button retry-hydration-button" onClick={() => runJob({ skipViewConsistency: true })}>Generate anyway</button>}</div>}
        {!error && tab !== 'generate' && caps?.diagnostics?.[0] && <div className="info-line"><Settings2 size={14} /> {caps.diagnostics[0]}</div>}
        <button className="primary-action" onClick={() => runJob()} disabled={busy || !activeSource || (tab === 'generate' && (!preparedForGeneration || Boolean(preparedCandidate) || viewsRehydrating))}>{busy ? <LoaderCircle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{busy ? 'Starting…' : tab === 'generate' && viewsRehydrating ? 'Restoring views…' : tab === 'generate' ? (preparedCandidate ? 'Review prepared image first' : 'Generate 3D model') : tab === 'retopologize' ? (cleanMethod === 'canonicalize' ? 'Clean (canonical)' : cleanMethod === 'rebuild-views' ? 'Rebuild from views' : 'Clean mesh') : tab === 'paint' ? 'Bake texture' : `Run ${activeTab.label}`}</button>
        {tab === 'generate' && !preparedCandidate && !busy && <small className="button-caption">{activeViewCount > 1 ? `TRELLIS.2 · ${activeViewCount} views` : 'TRELLIS.2 · T-pose prepare is optional'}</small>}
        {tab === 'retopologize' && !busy && <small className="button-caption">{cleanMethod === 'canonicalize' ? `S2 contract · ${customTriangleTarget(canonicalDensity.customTriangles) === null ? canonicalDensity.tier : 'custom'} · ~${resolvedTriangleTarget(canonicalDensity).toLocaleString()} tris` : cleanMethod === 'rebuild-views' ? 'Pixal3D multi-view · paints the mesh\'s own renders, rebuilds, lays the real pixels on top' : cleanMethod === 'ai-refine' ? 'TRELLIS.2 image-guided reconstruction (fallback)' : 'Instant Meshes quad remesh (legacy)'}</small>}
        {job && hasActiveStudioJob(job) && <JobProgress label={job.message || 'Working…'} progress={job.progress} logLine={job.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={cancelTask} />}
        {job && job.status === 'failed' && <div className="job-status failed"><div className="job-line"><Activity size={14} /><span>{job.error || job.message || 'Job failed'}</span></div></div>}
        {job && job.warnings?.[0] && <small className="job-warning">{job.warnings[0]}</small>}
        </div>
      </aside>
      <section className="canvas-area"><StudioViewport artifactId={displayAsset?.id} artifactKind={displayAsset?.kind} artifactName={displayAsset?.name} onImportModel={openModelImportPicker} onModelLoaded={() => { if (generationPendingPreview && generationStartedAt) { const elapsed = Date.now() - generationStartedAt; setGenerationElapsed(elapsed); setGenerationTime(elapsed); setGenerationPendingPreview(false); setGenerationStartedAt(null) }; if (refinePendingPreview && refineStartedAt) { const elapsed = Date.now() - refineStartedAt; setRefineElapsed(elapsed); setRefineTime(elapsed); setRefinePendingPreview(false); setRefineStartedAt(null) }; if (retopoPendingPreview && retopoStartedAt) { const elapsed = Date.now() - retopoStartedAt; setRetopoElapsed(elapsed); setRetopoTime(elapsed); setRetopoPendingPreview(false); setRetopoStartedAt(null) } }} />{generationPendingPreview && <div className="generation-blocker"><JobProgress label={`Generating 3D model · ${formatDuration(generationElapsed)}`} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={cancelTask} /></div>}{refinePendingPreview && <div className="generation-blocker"><JobProgress label={`Cleaning mesh · ${formatDuration(refineElapsed)}`} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={cancelTask} /></div>}{retopoPendingPreview && <div className="generation-blocker"><JobProgress label={`Cleaning mesh · ${formatDuration(retopoElapsed)}`} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={cancelTask} /></div>}<div className="canvas-footer"><div><span className="eyebrow">CURRENT OUTPUT</span><strong>{displayAsset ? displayAsset.name : 'No asset loaded'}</strong></div><div className="footer-actions">{generationTime !== null && <span className="generation-time">Generation time: {formatDuration(generationTime)}</span>}{refineTime !== null && <span className="generation-time">Clean time: {formatDuration(refineTime)}</span>}{retopoTime !== null && <span className="generation-time">Clean time: {formatDuration(retopoTime)}</span>}{displayAsset && <><button className="secondary-button" onClick={openLocation}><FolderOpen size={14} /> Open location</button>{displayAsset.kind === 'glb' && <ExportGlbControl asset={displayAsset} state={exportState} onExport={() => exportArtifact(displayAsset.id)} />}</>}</div></div></section>
      <AssetInspector asset={displayAsset} details={modelDetails} requirement={requiredDownload} downloads={downloads} onDownload={() => requiredDownload && startDownload(requiredDownload)} />
      </>}
    </main>
    </>}
  </div>
}

function DownloadDock({ downloads }: { downloads: DownloadState[] }) { const visible = downloads.filter((item) => ['queued', 'running', 'completed', 'failed'].includes(item.status)); if (!visible.length) return null; return <div className="download-dock"><div className="download-dock-title"><Download size={14} /><strong>Local downloads</strong><span>{visible.filter((item) => ['queued', 'running'].includes(item.status)).length || ' '}</span></div>{visible.slice(-3).map((item) => { const hasFile = Boolean(item.file_name); const fileProgress = item.file_total_bytes ? `${formatMb((item.file_bytes || 0) / 1024 / 1024)} / ${formatMb(item.file_total_bytes / 1024 / 1024)}` : ''; const label = item.model === 'trellis2' || item.model === 'trellis-refiner' ? 'TRELLIS.2 FP8 models' : item.model === 'instant-meshes' ? 'Instant Meshes engine' : 'Local model'; return <div className="download-item" key={item.id}><div className="download-item-line"><span>{label}</span><strong>{item.status === 'completed' ? 'Ready' : item.status === 'failed' ? 'Failed' : `${item.progress}%`}</strong></div><div className="download-track"><i className={item.status === 'failed' ? 'failed' : item.status === 'completed' ? 'complete' : ''} style={{ width: `${Math.max(4, item.progress)}%` }} /></div><small>{item.error || (hasFile ? `(${item.file_index || 0}/${item.file_count || 10}) ${item.file_name} · ${fileProgress}` : item.message)}</small></div> })}</div> }

function DownloadProgress({ download }: { download: DownloadState }) { const bytes = download.file_total_bytes ? `${formatMb((download.file_bytes || 0) / 1024 / 1024)} / ${formatMb(download.file_total_bytes / 1024 / 1024)}` : ''; return <div className="right-download-progress"><div className="download-item-line"><span>Downloading</span><strong>{download.progress}%</strong></div><div className="download-track"><i style={{ width: `${Math.max(4, download.progress)}%` }} /></div><small>{download.file_name ? `${download.file_name} · ${bytes}` : download.message}</small></div> }
function ExportGlbControl({ asset, state, onExport }: { asset: Artifact; state: ExportState; onExport: () => void }) {
  return <div className="export-glb-group">
    <div className="export-glb-row">
      <button className="secondary-button" onClick={onExport} disabled={state.status === 'exporting'}>
        {state.status === 'exporting' ? <LoaderCircle className="spin" size={14} /> : <Download size={14} />}
        {state.status === 'exporting' ? 'Saving…' : 'Export GLB'}
      </button>
      <a className="secondary-button" href={`${API}/assets/${asset.id}/download`}><Download size={14} /> Download</a>
    </div>
    {state.status === 'saved' && state.path && <small className="export-status export-status-ok">Saved to <code>{state.path}</code></small>}
    {state.status === 'downloaded' && <small className="export-status export-status-ok">No output folder configured — downloaded via your browser instead.</small>}
    {state.status === 'error' && state.error && <small className="export-status export-status-error">{state.error}</small>}
  </div>
}
function Mesh2MotionManualView({ source, runId, onBack }: { source: Artifact | null; runId: string | null; onBack: () => void }) {
  if (!source || source.kind !== 'glb') {
    return <section className="rig-empty"><div><Wrench size={34} /><h1>Mesh2Motion (manual)</h1><p>Load a GLB in the studio, then reopen this tool from the header menu.</p><button className="primary-action" onClick={onBack}><ArrowLeft size={17} /> Back to journey</button></div></section>
  }
  const sourceUrl = `${API}/assets/${source.id}/download`
  const workspaceUrl = `/mesh2motion.html?source=${encodeURIComponent(sourceUrl)}&extension=glb&mode=rig&run=${encodeURIComponent(runId || '')}&input=${encodeURIComponent(source.id)}`
  return <main className="rig-workspace mesh2motion-manual-view">
    <div className="mesh2motion-manual-toolbar">
      <button className="secondary-button" onClick={onBack}><ArrowLeft size={14} /> Back to journey</button>
      <span className="eyebrow">MESH2MOTION · MANUAL</span>
    </div>
    <iframe key={source.id} className="mesh2motion-frame" title="Mesh2Motion manual workspace" src={workspaceUrl} />
  </main>
}
function AssetInspector({ asset, details, requirement, downloads, onDownload }: { asset: Artifact | null; details: ModelDetails | null; requirement: { name: string; label: string; ready: boolean; detail: string } | null; downloads: DownloadState[]; onDownload: () => void }) {
  const value = (number?: number) => number === undefined ? '—' : number.toLocaleString()
  const activeDownload = requirement ? downloads.find((item) => item.model === requirement.name && ['queued', 'running'].includes(item.status)) : undefined
  return <aside className="inspector">
    <div className="inspector-body inspector-main">
      <div className="asset-icon"><Box size={23} /></div>
      <div className="inspector-title-row">
        <h2>{asset ? asset.name.replace(/\.glb$/i, '') : 'Untitled asset'}</h2>
        {requirement && <span className={`status-chip ${requirement.ready ? 'ok' : 'warn'}`}>{requirement.ready ? 'Ready' : 'Setup required'}</span>}
      </div>
      {(!asset || !details) && <p className="muted">{!asset ? 'Generate or import a GLB to inspect it here.' : 'Reading model details…'}</p>}
      <div className="spec-grid"><div><span>Vertices</span><strong>{value(details?.vertices)}</strong></div><div><span>Triangles</span><strong>{value(details?.triangles)}</strong></div><div><span>Materials</span><strong>{value(details?.materials)}</strong></div><div><span>Meshes</span><strong>{value(details?.meshes)}</strong></div><div><span>Format</span><strong>{details?.format || '—'}</strong></div><div><span>File size</span><strong>{details ? formatMb(details.size_bytes / 1024 / 1024) : '—'}</strong></div></div>
      {requirement && !requirement.ready && <div className="model-requirement-inline">
        <p>{requirement.detail}</p>
        <button className="download-model-button" onClick={onDownload} disabled={Boolean(activeDownload)}><Download size={14} /> {activeDownload ? 'Downloading…' : requirement.name === 'instant-meshes' ? 'Download Instant Meshes' : 'Download model'}</button>
      </div>}
    </div>
    {activeDownload && <div className="inspector-download-area"><DownloadProgress download={activeDownload} /></div>}
  </aside>
}
// The canonicalize gate's verdict, rendered with the same quality-card
// vocabulary History uses for every other stage's quality_decisions — a
// blocked candidate must show each finding, not just its status.
export function CanonicalizeVerdictCard({ verdict, artifactName, promoted, pinned, onPin, density = null }: {
  verdict: CanonicalizeVerdict
  artifactName: string
  promoted: boolean
  pinned: boolean
  onPin: () => void
  density?: CanonicalizeDensityReport | null
}) {
  // Clean's own picker lists what Clean *consumes* (Generate outputs), so once
  // the operator pins something else the canonical mesh disappears from it.
  // The card is therefore the durable way back to the canonical artifact —
  // the pin stays offered whether the gate promoted it or merely retained it.
  const retained = canonicalizeCandidatePinnable(verdict, promoted)
  return <section className="quality-panels compact" aria-label="Canonicalize gate">
    <article className={`quality-card ${verdict.status}`} data-testid="canonicalize-verdict">
      <div className="quality-title">{verdict.status === 'ready' ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}<strong>canonicalize</strong><span>{verdict.status.replace('_', ' ')}</span></div>
      {/* i63ky: what density was asked for and what actually came out, so the
          operator can see whether Ultra bought them the triangles they paid
          for without opening the artifact's metadata. */}
      {density && <div className="retopo-options" data-testid="canonicalize-density">
        <div><span>Quality</span><strong>{density.tier}{density.targetTriangles !== null ? ` · ${density.targetTriangles.toLocaleString()} target` : ''}</strong></div>
        {density.achievedTriangles !== null && <div><span>Achieved</span><strong>{density.achievedTriangles.toLocaleString()} tris</strong></div>}
        {density.band && <div><span>Gate band</span><strong>{density.band[0].toLocaleString()}–{density.band[1].toLocaleString()}</strong></div>}
        {density.atlasSize !== null && <div><span>Texture atlas</span><strong>{density.atlasSize}px</strong></div>}
      </div>}
      {verdict.findings.length > 0
        ? <div className="quality-findings">{verdict.findings.map((finding) => <div className={finding.severity} key={finding.code}><p>{finding.message}</p><small>{finding.metric}: <b>{formatSetting(finding.actual)}</b> · expected {finding.expected}</small></div>)}</div>
        : <div className="quality-findings"><div className="ok"><p>No findings — this candidate meets the S2 contract.</p></div></div>}
      {verdict.recommended_actions.length > 0 && <small className="job-warning">Recommended: {verdict.recommended_actions.join(' · ')}</small>}
      <div className="quality-actions">
        {promoted && <span>Promoted — {artifactName} is now this journey's working mesh.</span>}
        <button type="button" disabled={pinned} onClick={onPin}>{pinned ? 'Working on this mesh' : retained ? 'Pin canonical candidate' : 'Work on the canonical mesh'}</button>
      </div>
    </article>
  </section>
}
// The project-views pass's verdict: how much of the canonical texture the
// T-pose conditioning views actually reached, blended over TRELLIS's own
// bake. Unlike canonicalize, a poor result is never gated out of the journey
// -- the fallback bake underneath is always a complete texture -- so this
// card is informational rather than an approve/retain choice.
export function ProjectionVerdictCard({ verdict, report }: {
  verdict: ProjectionVerdict
  report?: ProjectionReport | null
}) {
  return <section className="quality-panels compact" aria-label="View projection">
    <article className={`quality-card ${verdict.status}`} data-testid="projection-verdict">
      <div className="quality-title">{verdict.status === 'green' ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}<strong>view projection</strong><span>{verdict.status.replace('_', ' ')}</span></div>
      {report && <div className="retopo-options" data-testid="projection-coverage">
        <div><span>Views used</span><strong>{report.viewsUsed.length > 0 ? report.viewsUsed.join(', ') : 'none'}</strong></div>
        {report.projectedTexelFraction !== null && <div><span>Projected texels</span><strong>{Math.round(report.projectedTexelFraction * 100)}%</strong></div>}
        {report.meanColorDeltaVsBake !== null && <div><span>Colour delta vs bake</span><strong>{report.meanColorDeltaVsBake.toFixed(3)}</strong></div>}
      </div>}
      {verdict.findings.length > 0
        ? <div className="quality-findings">{verdict.findings.map((finding) => <div className={finding.severity} key={finding.code}><p>{finding.message}</p><small>{finding.metric}: <b>{formatSetting(finding.actual)}</b> · expected {finding.expected}</small></div>)}</div>
        : <div className="quality-findings"><div className="ok"><p>Every conditioning view contributed meaningfully to the canonical texture.</p></div></div>}
      {verdict.recommended_actions.length > 0 && <small className="job-warning">Recommended: {verdict.recommended_actions.join(' · ')}</small>}
      {report && report.caveats.length > 0 && <small className="job-warning">{report.caveats.join(' ')}</small>}
    </article>
  </section>
}
// The source-fidelity gate's verdict: how closely the generated mesh matches
// the source photo, with a side-by-side comparison image when the pass ran.
export function SourceFidelityCard({ verdict, report }: {
  verdict: SourceFidelityVerdict
  report?: SourceFidelityReport | null
}) {
  const { summary } = verdict
  return <section className="quality-panels compact" aria-label="Source fidelity">
    <article className={`quality-card ${verdict.status}`} data-testid="source-fidelity-verdict">
      <div className="quality-title">{verdict.status === 'green' ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}<strong>source fidelity</strong><span>{verdict.status.replace('_', ' ')}</span></div>
      <div className="retopo-options" data-testid="source-fidelity-summary">
        <div><span>Silhouette IoU</span><strong>{summary.silhouetteIou !== null ? summary.silhouetteIou.toFixed(3) : '—'}</strong></div>
        <div><span>Face SSIM</span><strong>{summary.faceSsim !== null ? summary.faceSsim.toFixed(3) : '—'}</strong></div>
        <div><span>Subject SSIM</span><strong>{summary.maskedSsim !== null ? summary.maskedSsim.toFixed(3) : '—'}</strong></div>
        <div><span>Perceptual distance</span><strong>{summary.perceptualDistance !== null ? summary.perceptualDistance.toFixed(3) : '—'}</strong></div>
        <div><span>Detail ratio</span><strong>{summary.detailRatio !== null ? summary.detailRatio.toFixed(2) : '—'}</strong></div>
        <div><span>Face detail</span><strong>{summary.faceDetailRatio !== null ? summary.faceDetailRatio.toFixed(2) : '—'}</strong></div>
        <div><span>Camera</span><strong>{report?.cameraKind ?? '—'}</strong></div>
      </div>
      {report?.sideBySideArtifactId && <img data-testid="source-fidelity-side-by-side" src={`${API}/assets/${report.sideBySideArtifactId}/download`} alt="Source photo, render, and difference" style={{ width: '100%', maxWidth: 720 }} />}
      {report?.skipped && <small className="job-warning">{report.skipped}</small>}
      {verdict.findings.length > 0
        ? <div className="quality-findings">{verdict.findings.map((finding) => <div className={finding.severity} key={finding.code}><p>{finding.message}</p><small>{finding.metric}: <b>{formatSetting(finding.actual)}</b> · expected {finding.expected}</small></div>)}</div>
        : <div className="quality-findings"><div className="ok"><p>No findings — within the source-fidelity thresholds.</p></div></div>}
      {verdict.recommended_actions.length > 0 && <small className="job-warning">Recommended: {verdict.recommended_actions.join(' · ')}</small>}
    </article>
  </section>
}
// The rebuild-views result: approach9's own IoU against the views the mesh
// was painted from, plus the multiview gate's per-role SSIM against the REAL
// reference photos. Informational -- the overlay is always a complete mesh.
export function RebuildViewsCard({ report }: { report: RebuildViewsReport }) {
  const value = (score: number | null) => score !== null ? score.toFixed(2) : '—'
  return <section className="quality-panels compact" aria-label="Rebuild from views">
    <article className="quality-card green" data-testid="rebuild-views-result">
      <div className="quality-title"><ShieldCheck size={16} /><strong>rebuild from views</strong></div>
      <div className="quality-findings"><div className="ok"><p>
        Matches its views: mean IoU {value(report.meanIou)}
        {report.front && ` · front SSIM ${value(report.front.ssim)} / face ${value(report.front.face)}`}
        {report.back && ` · back ${value(report.back.ssim)} / ${value(report.back.face)}`}
      </p></div></div>
    </article>
  </section>
}
export function CleanControls({
  method, setMethod, targetVertices, setTargetVertices, repairHoles, setRepairHoles, gpuFirst, setGpuFirst,
  suggestedVertices, source, refineImage, onAttachImage, onUpload, canonicalResult = null, rebuildResult = null,
  run = null, pinnedArtifactId = null, onPinArtifact = () => {}, detailsById = {},
  density = CANONICALIZE_DEFAULT_DENSITY, setDensity = () => {},
  rebuildViewsAvailable = true,
}: {
  method: CleanMethod; setMethod: (value: CleanMethod) => void; targetVertices: number; setTargetVertices: (v: number) => void; repairHoles: boolean; setRepairHoles: (value: boolean) => void; gpuFirst: boolean; setGpuFirst: (value: boolean) => void; suggestedVertices: number | null; source: Artifact | null; refineImage: Artifact | null; onAttachImage: () => void; onUpload: () => void
  canonicalResult?: { artifactId: string; artifactName: string; verdict: CanonicalizeVerdict; promoted: boolean; density?: CanonicalizeDensityReport | null; projection?: { verdict: ProjectionVerdict; report: ProjectionReport | null } | null; sourceFidelity?: { verdict: SourceFidelityVerdict; report: SourceFidelityReport | null } | null } | null
  rebuildResult?: RebuildViewsReport | null
  run?: RunManifest | null
  pinnedArtifactId?: string | null
  onPinArtifact?: (artifactId: string) => void
  detailsById?: Record<string, { name?: string; vertices?: number; size_bytes?: number }>
  density?: CanonicalizeDensity
  setDensity?: (value: CanonicalizeDensity) => void
  rebuildViewsAvailable?: boolean
}) {
  const sourceVertices = source ? detailsById[source.id]?.vertices ?? null : null
  const recommendation = cleanRecommendation(source)
  const recommendedMethod = recommendation && recommendation.choice !== 'skip' ? recommendation.choice : null
  const latestGenerateOutput = runAssetOptions(run, 'clean')[0]
  const targetExceedsSource = cleanTargetExceedsSource(targetVertices, sourceVertices)
  const reCleaningCleanOutput = isReCleaningCleanOutput(source?.stage)
  const canonical = method === 'canonicalize'
  // i63ky: mirrors quality_tiers.resolve_quality_target / triangle_band so the
  // panel can show the target and the gate band it is about to ask for. The
  // backend re-derives both and stays the authority.
  const customTarget = customTriangleTarget(density.customTriangles)
  const selectedTier = QUALITY_TIER_OPTIONS.find((option) => option.id === density.tier) ?? QUALITY_TIER_OPTIONS[1]
  const targetTriangles = resolvedTriangleTarget(density)
  const gateBand: [number, number] = [Math.round(targetTriangles * 0.5), Math.round(targetTriangles * 1.5)]
  return <>
    <WorkingOn stage="clean" stageLabel="Clean" run={run} current={source} detailsById={detailsById} pinnedArtifactId={pinnedArtifactId} onPin={onPinArtifact} onUpload={onUpload} />
    <div className="clean-recommendation clean-canonical-banner">
      <span><strong>Canonical clean is the recommended path before rigging.</strong> Policy S2 is mandatory: rigging expects a canonical mesh — one connected component, watertight, no non-manifold edges, a triangle count inside the family band, and an appearance match to the source.</span>
      <span className="clean-recommendation-note">Voxel remesh, then a bake from the original mesh, then a blocking gate. A candidate that fails the gate is kept but never auto-promoted; its findings are shown below so you can decide.</span>
    </div>
    <SectionLabel>Method</SectionLabel>
    <div className="segmented">
      <button className={canonical ? 'selected' : ''} onClick={() => setMethod('canonicalize')}>Canonical (S2)</button>
      <button className={method === 'ai-refine' ? 'selected' : ''} onClick={() => setMethod('ai-refine')}>AI Refine</button>
      <button className={method === 'instant-meshes' ? 'selected' : ''} onClick={() => setMethod('instant-meshes')}>Instant Meshes</button>
      {/* Two words so it wraps like its neighbours (the 4-way selector clips a single long word); the panel and primary button carry the full name. */}
      {/* Rebuild from views needs the optional Pixal3D scripts; offer it only where the backend has them (an already-selected method stays visible). */}
      {(rebuildViewsAvailable || method === 'rebuild-views') && <button className={method === 'rebuild-views' ? 'selected' : ''} onClick={() => setMethod('rebuild-views')} title="Rebuild from views">From views</button>}
    </div>
    {rebuildResult && <RebuildViewsCard report={rebuildResult} />}
    {canonicalResult && <CanonicalizeVerdictCard
      verdict={canonicalResult.verdict}
      artifactName={canonicalResult.artifactName}
      promoted={canonicalResult.promoted}
      pinned={pinnedArtifactId === canonicalResult.artifactId}
      onPin={() => onPinArtifact(canonicalResult.artifactId)}
      density={canonicalResult.density ?? null}
    />}
    {canonicalResult?.projection && <ProjectionVerdictCard
      verdict={canonicalResult.projection.verdict}
      report={canonicalResult.projection.report}
    />}
    {canonicalResult?.sourceFidelity && <SourceFidelityCard
      verdict={canonicalResult.sourceFidelity.verdict}
      report={canonicalResult.sourceFidelity.report}
    />}
    {canonical ? <>
      {/* i63ky: the operator's density choice. The gate follows the selection
          (its band is derived from the same target), so picking Ultra is a
          request for more triangles, not a way to fail the contract. */}
      <SectionLabel>Mesh density</SectionLabel>
      <div className="segmented" data-testid="quality-tier-picker">
        {QUALITY_TIER_OPTIONS.map((option) => <button
          key={option.id}
          className={density.tier === option.id && customTarget === null ? 'selected' : ''}
          onClick={() => setDensity({ tier: option.id, customTriangles: null })}
        >{option.label}</button>)}
      </div>
      <div className="retopo-target-note">{customTarget === null ? selectedTier.caption : 'Custom target — the tier captions above are the nearest comparison for file size, rig time and viewer smoothness.'}</div>
      <div className="custom-target-row">
        <label htmlFor="canonicalize-custom-triangles">Custom triangle target <strong>{customTarget === null ? 'off' : customTarget.toLocaleString()}</strong></label>
        <input
          id="canonicalize-custom-triangles"
          type="number"
          min={CUSTOM_TRIANGLE_TARGET_MINIMUM}
          max={CUSTOM_TRIANGLE_TARGET_MAXIMUM}
          step={1000}
          placeholder={`${selectedTier.targetTriangles}`}
          value={density.customTriangles === null ? '' : density.customTriangles}
          onChange={(event) => setDensity({ tier: density.tier, customTriangles: event.target.value === '' ? null : Number(event.target.value) })}
        />
      </div>
      {density.customTriangles !== null && customTarget === null && <div className="clean-hint">Custom targets must be between {CUSTOM_TRIANGLE_TARGET_MINIMUM.toLocaleString()} and {CUSTOM_TRIANGLE_TARGET_MAXIMUM.toLocaleString()} triangles — the {selectedTier.label} tier will be used until this is in range.</div>}
      <SectionLabel>Contract</SectionLabel>
      <div className="retopo-options"><div><span>Remesh</span><strong>Blender voxel</strong></div><div><span>Texture</span><strong>Baked from original</strong></div><div><span>Gate</span><strong>Blocking (S2)</strong></div><div><span>Triangle budget</span><strong>{gateBand[0].toLocaleString()}–{gateBand[1].toLocaleString()}</strong></div></div>
      <div className="retopo-target-note">Aiming for {targetTriangles.toLocaleString()} triangles; the gate accepts anything inside {gateBand[0].toLocaleString()}–{gateBand[1].toLocaleString()}.</div>
      {reCleaningCleanOutput && <div className="clean-hint">
        <span>This mesh is already a Clean output — canonicalize the Generate result instead, so the gate compares against the original geometry.</span>
        {latestGenerateOutput && <button type="button" className="secondary-button" onClick={() => onPinArtifact(latestGenerateOutput.artifactId)}>Pin latest Generate output</button>}
      </div>}
      <div className="retopo-summary"><span>Solver stages</span><strong>Voxel remesh → bake from original → S2 gate</strong></div>
    </> : method === 'rebuild-views' ? <>
      <div className="clean-hint">
        <span><strong>Rebuild from views recovers identity, not topology.</strong> It renders this mesh from Pixal3D's camera rig, paints those renders as the character, rebuilds with Pixal3D multi-view, then lays the real reference pixels over the result. Canonicalize afterwards before rigging.</span>
      </div>
      <SectionLabel>References</SectionLabel>
      <div className="retopo-options"><div><span>Used</span><strong>Front + back</strong></div><div><span>Skipped</span><strong>Left / right</strong></div></div>
      <div className="retopo-target-note">The T-pose set's left and right are three-quarter views: used as identity references they repaint the face from the side, so only true same-angle views are passed.</div>
      <div className="retopo-summary"><span>Solver stages</span><strong>Clay guides → paint → Pixal3D multi-view → real-pixel overlay</strong></div>
    </> : <>
      <div className="clean-hint">
        {method === 'ai-refine'
          ? <span><strong>AI Refine is a fallback, not the default.</strong> It re-reconstructs the mesh with TRELLIS.2 and can regress the character's appearance. Use it only when the canonical clean is blocked on geometry the voxel remesh cannot repair, then canonicalize the result.</span>
          : <span><strong>Instant Meshes is legacy (pre-S2).</strong> It produces no contract gate and fragments generated meshes. Kept for dense imported scans; canonicalize afterwards before rigging.</span>}
      </div>
      {recommendation && <div className="clean-recommendation">
        <span>{recommendation.reason}</span>
        {recommendedMethod && <button type="button" className="secondary-button" onClick={() => setMethod(recommendedMethod)}>Use {recommendedMethod === 'ai-refine' ? 'AI Refine' : 'Instant Meshes'}</button>}
      </div>}
      {method === 'ai-refine' && <><SectionLabel>Reference image</SectionLabel><UploadCard source={refineImage} onUpload={onAttachImage} label="Attach reference image" hint="PNG, JPG or WebP · required for imported models" /></>}
      <SectionLabel>Topology</SectionLabel>
      <div className="retopo-options"><div><span>Output</span><strong>Quad-dominant</strong></div><div><span>Boundaries</span><strong>Preserve</strong></div></div>
      <div className="range-row"><label>Target vertex count <strong>{targetVertices.toLocaleString()}</strong></label><input type="range" min="500" max={Math.max(1000000, targetVertices)} step="500" value={targetVertices} onChange={(e) => setTargetVertices(Number(e.target.value))} /></div>
      {suggestedVertices !== null && <div className="retopo-target-note">Suggested for this model: {suggestedVertices.toLocaleString()}. Feel free to enter your desired value.</div>}
      {targetExceedsSource && <div className="clean-hint">Higher than the source's {sourceVertices?.toLocaleString()} vertices — Instant Meshes fragments when upscaling; try ≤ source.</div>}
      {reCleaningCleanOutput && <div className="clean-hint">
        <span>This mesh is already a Clean output — re-cleaning compounds fragmentation; pick the Generate result.</span>
        {latestGenerateOutput && <button type="button" className="secondary-button" onClick={() => onPinArtifact(latestGenerateOutput.artifactId)}>Pin latest Generate output</button>}
      </div>}
      <label className={`retopo-repair-toggle ${repairHoles ? 'selected' : ''}`}><input type="checkbox" checked={repairHoles} onChange={(event) => setRepairHoles(event.target.checked)} /><span><strong>Repair holes</strong><small>Patch holes and self-intersections after cleaning; patches may add triangles.</small></span></label>
      {method === 'ai-refine' && <label className={`retopo-repair-toggle ${gpuFirst ? 'selected' : ''}`}><input type="checkbox" checked={gpuFirst} onChange={(event) => setGpuFirst(event.target.checked)} /><span><strong>GPU-first (experimental)</strong><small>Faster if it fits; may run out of memory.</small></span></label>}
      <div className="retopo-summary"><span>Solver stages</span><strong>{method === 'ai-refine' ? 'Image refine → quad reconstruction → extract' : 'Orientation → Position → Extract'}</strong></div>
    </>}
  </>
}
export function RigAnimateWorkspace({ mode, runId, run = null, source, modelDetails, checkpoint, reviewStatus, reviewedArtifactId, onApprove, onReview, onUpload, onGenerateMotion, onSaveAdjustments, onAttachCuratedMotion, kimodoReady, onGenerateAutoRig, unirigReady, onCancel, motionReady, blenderAvailable, busy, job, terminalHydrating, terminalHydrationError, onRetryTerminalHydration, pinnedArtifactId = null, onPinArtifact = () => {}, detailsById = {}, onExport = () => {}, exportState = { status: 'idle' } }: { mode?: 'rig' | 'animate'; runId: string | null; run?: RunManifest | null; source: Artifact | null; modelDetails?: ModelDetails | null; checkpoint?: { artifact_id: string; note: string; completed_at: number }; reviewStatus?: 'unreviewed' | 'accepted' | 'rejected'; reviewedArtifactId?: string | null; onApprove: (artifactId: string, note: string) => Promise<void>; onReview: (artifactId: string, status: 'accepted' | 'rejected', note: string) => Promise<void>; onUpload: () => void; onGenerateMotion: () => void; onGenerateAutoRig: (targetArtifactId: string) => Promise<void>; onSaveAdjustments: (adjustment: RigAdjustmentSave) => Promise<void>; onAttachCuratedMotion: (clipAnimation: string) => Promise<void>; unirigReady: boolean | null; kimodoReady: boolean; onCancel: () => void; motionReady: boolean; blenderAvailable: boolean | null; busy: boolean; job: Job | null; terminalHydrating: boolean; terminalHydrationError: string; onRetryTerminalHydration: () => void; pinnedArtifactId?: string | null; onPinArtifact?: (artifactId: string) => void; detailsById?: Record<string, WorkingOnAssetDetails>; onExport?: (artifactId: string) => void; exportState?: ExportState }) {
  // The Animate panel's clip picker drives this (AnimatePanel's
  // onPreviewClip) so the main viewport can instantly retarget-and-play a
  // curated clip on the loaded rig with no server job — declared above the
  // early return below so the hook always runs regardless of `rigged`.
  const [previewClip, setPreviewClip] = useState<{ animation: string; label: string } | null>(null)
  if (!source || source.kind !== 'glb') return <section className="rig-empty"><div><UserRound size={34} /><h1>Rig + Animate</h1><p>Choose a GLB from the studio or upload one to begin the Mesh2Motion workflow.</p><button className="primary-action" onClick={onUpload}><Upload size={17} /> Upload GLB</button></div></section>
  const rigged = isRigPlaybackStage(source.stage)
  const surface = rigWorkspaceSurface(mode, rigged)
  if (surface === 'skeleton' && runId) {
    const cleanMeshArtifactId = typeof source.metadata?.source_artifact_id === 'string'
      ? source.metadata.source_artifact_id
      : null
    return <SkeletonWorkspace artifact={source} cleanMeshArtifactId={cleanMeshArtifactId} runId={runId} checkpoint={checkpoint} onApprove={onApprove} onSaveAdjustments={onSaveAdjustments} onGenerateAutoRig={onGenerateAutoRig} job={job} busy={busy} onCancel={onCancel} run={run} pinnedArtifactId={pinnedArtifactId} onPinArtifact={onPinArtifact} detailsById={detailsById} onUpload={onUpload} />
  }
  // Single Animate surface for a rigged source — no separate Playback vs.
  // Animation-library toggle, and no separate Review tab (BOU: accept/save/
  // export lives inline in Animate). Chain-rig detection prefers the run's
  // own subject_kind (stable regardless of which artifact — rig or clip —
  // is currently selected as `source`); the certificate profile on
  // source.metadata is kept as a fallback for older runs without subject_kind.
  if (rigged) {
    const certificateProfile = (source.metadata?.quality_decisions as Record<string, { profile?: string }> | undefined)?.['rig-certificate']?.profile
    const isChainRig = run?.subject_kind === 'creature' || certificateProfile === 'chain'
    // N5: the pinned rig's own conformance decides which "Attach motion"
    // path the picker submits through (AnimatePanel's button label follows
    // this too) -- a chain rig is never template-conformant to the biped
    // template, so the two are mutually exclusive in practice.
    const isConformantRig = isTemplateConformantRig(source.metadata)
    return <section className="rig-manual-workspace">
      <div className="rig-authoring-body">
        <div className="rig-authoring-viewport"><StudioViewport artifactId={source.id} artifactKind={source.kind} artifactName={source.name} showAnimationControls={source.stage === 'retarget-motion' || source.stage === 'generate-motion' || source.stage === 'apply-clip'} previewClip={previewClip} /></div>
        <AnimatePanel
          run={run} source={source} detailsById={detailsById}
          pinnedArtifactId={pinnedArtifactId} onPinArtifact={onPinArtifact} onUpload={onUpload}
          job={job} busy={busy} isChainRig={isChainRig} isConformantRig={isConformantRig}
          onGenerateMotion={onGenerateMotion} onAttachCuratedMotion={onAttachCuratedMotion} kimodoReady={kimodoReady}
          onReview={onReview} reviewStatus={reviewStatus} reviewedArtifactId={reviewedArtifactId}
          onExport={onExport} exportState={exportState}
          onCancel={onCancel} terminalHydrationError={terminalHydrationError} onRetryTerminalHydration={onRetryTerminalHydration}
          onPreviewClip={setPreviewClip}
        />
      </div>
    </section>
  }
  const jobActive = hasActiveStudioJob(job)
  const autoRigBlocked = busy || jobActive || terminalHydrating
  const capsLoaded = unirigReady !== null
  const autoRigJobRunning = jobActive && (job?.kind === 'auto-rig' || job?.kind === 'rig-adjust')
  return <section className="rig-manual-workspace">
    <div className="rig-curated-toolbar">
      <div className="rig-curated-copy"><span className="eyebrow">SKELETON</span><strong>Predict a skeleton, then adjust and approve it</strong></div>
      {terminalHydrationError && <small className="motion-error">{terminalHydrationError}</small>}
      {terminalHydrationError && <button className="secondary-button retry-hydration-button" onClick={onRetryTerminalHydration}>Retry loading output</button>}
    </div>
    <div className="rig-authoring-body">
      <div className="rig-authoring-viewport"><StudioViewport artifactId={source.id} artifactKind={source.kind} artifactName={source.name} /></div>
      <aside className="rig-authoring-inspector">
        <WorkingOn stage="skeleton-pre-rig" stageLabel="Skeleton" run={run} current={source} detailsById={detailsById} pinnedArtifactId={pinnedArtifactId} onPin={onPinArtifact} onUpload={onUpload} />
        <span className="eyebrow">SOURCE MESH</span>
        <h1>{source.name}</h1>
        <div className="spec-grid">
          <div><span>Vertices</span><strong>{formatCount(modelDetails?.vertices)}</strong></div>
          <div><span>Triangles</span><strong>{formatCount(modelDetails?.triangles)}</strong></div>
        </div>
        <p>Predict a skeleton with UniRig, then adjust and approve it.</p>
        {!capsLoaded ? (
          <small className="motion-hint">Checking local worker…</small>
        ) : autoRigJobRunning ? (
          <JobProgress label={job?.kind === 'rig-adjust' ? 'Rebinding…' : 'Auto-rigging…'} progress={job?.progress ?? 0} logLine={job?.logs?.filter(Boolean).slice(-1)[0]?.slice(0, 120)} onCancel={onCancel} />
        ) : unirigReady ? (
          <button className="primary-action" onClick={() => onGenerateAutoRig(source.id)} disabled={autoRigBlocked}>
            <Sparkles size={16} /> Auto-rig character
          </button>
        ) : (
          <small className="motion-error">UniRig runtime is not installed. Run scripts/setup_unirig.sh on the backend host.</small>
        )}
        {capsLoaded && blenderAvailable === false && <small className="motion-error">Blender 4.2+ is unavailable. Install or configure Blender — UniRig binding requires it.</small>}
        {job?.kind === 'auto-rig' && job.status === 'failed' && <small className="motion-error">{job.error || 'The UniRig auto-rig failed'}</small>}
      </aside>
    </div>
  </section>
}
function PaintControls({ color, setColor, roughness, setRoughness, metallic, setMetallic, brushSize, setBrushSize, brushOpacity, setBrushOpacity, brushHardness, setBrushHardness, strokes, setStrokes, source, onUpload, run = null, pinnedArtifactId = null, onPinArtifact = () => {}, detailsById = {} }: { color: string; setColor: (v: string) => void; roughness: number; setRoughness: (v: number) => void; metallic: number; setMetallic: (v: number) => void; brushSize: number; setBrushSize: (v: number) => void; brushOpacity: number; setBrushOpacity: (v: number) => void; brushHardness: number; setBrushHardness: (v: number) => void; strokes: PaintStroke[]; setStrokes: (v: PaintStroke[]) => void; source: Artifact | null; onUpload: () => void; run?: RunManifest | null; pinnedArtifactId?: string | null; onPinArtifact?: (artifactId: string) => void; detailsById?: Record<string, WorkingOnAssetDetails> }) { return <><WorkingOn stage="paint" stageLabel="Paint" run={run} current={source} detailsById={detailsById} pinnedArtifactId={pinnedArtifactId} onPin={onPinArtifact} onUpload={onUpload} /><SectionLabel>Material</SectionLabel><div className="color-row"><input type="color" value={color} onChange={(e) => setColor(e.target.value)} /><div><strong>Base color</strong><span>{color.toUpperCase()}</span></div></div><div className="range-row"><label>Roughness <strong>{roughness}%</strong></label><input type="range" min="0" max="100" value={roughness} onChange={(e) => setRoughness(Number(e.target.value))} /></div><div className="range-row"><label>Metallic <strong>{metallic}%</strong></label><input type="range" min="0" max="100" value={metallic} onChange={(e) => setMetallic(Number(e.target.value))} /></div><SectionLabel>Brush mask</SectionLabel><PaintCanvas color={color} size={brushSize} opacity={brushOpacity} hardness={brushHardness} strokes={strokes} setStrokes={setStrokes} /><div className="range-row"><label>Size <strong>{brushSize}%</strong></label><input type="range" min="2" max="30" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} /></div><div className="range-row"><label>Opacity <strong>{brushOpacity}%</strong></label><input type="range" min="1" max="100" value={brushOpacity} onChange={(e) => setBrushOpacity(Number(e.target.value))} /></div><div className="range-row"><label>Hardness <strong>{brushHardness}%</strong></label><input type="range" min="1" max="100" value={brushHardness} onChange={(e) => setBrushHardness(Number(e.target.value))} /></div><div className="brush-row"><button className="brush selected"><Brush size={15} /></button><button className="brush" onClick={() => setStrokes([])}><RotateCcw size={15} /></button><span>{strokes.length ? `${strokes.length} stroke${strokes.length === 1 ? '' : 's'}` : 'Basic mask'}</span></div><div className="info-line"><Paintbrush size={14} /> Paint strokes are baked into a packed GLB texture</div></> }

function PaintCanvas({ color, size, opacity, hardness, strokes, setStrokes }: { color: string; size: number; opacity: number; hardness: number; strokes: PaintStroke[]; setStrokes: (v: PaintStroke[]) => void }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const painting = useRef(false)
  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => { const canvas = ref.current; if (!canvas) return; const rect = canvas.getBoundingClientRect(); setStrokes([...strokes, { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / rect.height)), radius: size / 100, color, opacity: opacity / 100, hardness: hardness / 100 }]) }
  useEffect(() => { const canvas = ref.current; if (!canvas) return; const ctx = canvas.getContext('2d')!; ctx.fillStyle = '#24232b'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.strokeStyle = '#3b3945'; ctx.lineWidth = 1; for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(i * canvas.width / 8, 0); ctx.lineTo(i * canvas.width / 8, canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * canvas.height / 8); ctx.lineTo(canvas.width, i * canvas.height / 8); ctx.stroke() } strokes.forEach((stroke) => { ctx.globalAlpha = stroke.opacity; ctx.fillStyle = stroke.color; ctx.beginPath(); ctx.arc(stroke.x * canvas.width, (1 - stroke.y) * canvas.height, stroke.radius * canvas.width, 0, Math.PI * 2); ctx.fill() }); ctx.globalAlpha = 1 }, [strokes])
  return <canvas ref={ref} width={280} height={150} className="paint-canvas" onPointerDown={(event) => { painting.current = true; event.currentTarget.setPointerCapture(event.pointerId); draw(event) }} onPointerMove={(event) => { if (painting.current) draw(event) }} onPointerUp={() => { painting.current = false }} onPointerCancel={() => { painting.current = false }} />
}
