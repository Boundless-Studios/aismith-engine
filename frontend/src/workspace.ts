import { isImageArtifactKind, type QualityDecision, type QualityFinding } from './history'

export const SWORD_ATTACK_PROMPT = 'A stylized female sword fighter performs one decisive horizontal sword slash, plants her feet, follows through, and recovers to a combat-ready stance'

export const ACTIVE_RUN_STORAGE_KEY = 'aismith-active-run-id'

export function shouldRestoreRun(saved: string | null, runs: Array<{ id: string }>): string | null {
  return saved && runs.some((run) => run.id === saved) ? saved : null
}

export function shouldRefreshCapabilities(visibilityState: DocumentVisibilityState): boolean {
  return visibilityState === 'visible'
}

export function motionJobPayload(sourceArtifactId: string, runId: string | null, seed: number) {
  return {
    kind: 'generate-motion' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    seed,
    settings: {
      prompt: SWORD_ATTACK_PROMPT,
      duration_seconds: 2,
    },
  }
}

export type PreparationProfile = 'humanoid-a-pose-rig-v3' | 'humanoid-t-pose-rig-v1'

export function preparationJobPayload(
  sourceArtifactId: string,
  runId: string | null,
  seed: number,
  adapter: 'comfyui' | 'upload' = 'comfyui',
  profile: PreparationProfile = 'humanoid-t-pose-rig-v1',
) {
  return {
    kind: 'prepare-image' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    seed,
    settings: {
      adapter,
      profile,
    },
  }
}

// Multi-view conditioning (E2): the front view is always the existing `source`
// artifact; back/left/right are optional side references uploaded alongside it.
export const VIEW_LABEL_ORDER = ['front', 'back', 'left', 'right'] as const
export type ViewLabel = typeof VIEW_LABEL_ORDER[number]
export type OptionalViewLabel = Exclude<ViewLabel, 'front'>

// Builds the source fields of a `generate` job payload. With no optional views
// set this is exactly today's `source_artifact_id` shape; once any optional view
// is present, the front id moves into `source_views` and `source_artifact_id` is
// omitted entirely (the server infers `front` from `source_views`).
export function generateSourceFields(
  frontArtifactId: string | undefined,
  views: Partial<Record<OptionalViewLabel, string | undefined>>,
): { source_artifact_id?: string } | { source_views: Partial<Record<ViewLabel, string>> } {
  const optionalLabels = (['back', 'left', 'right'] as const).filter((label) => views[label])
  if (!optionalLabels.length) return { source_artifact_id: frontArtifactId }
  const source_views: Partial<Record<ViewLabel, string>> = { front: frontArtifactId }
  for (const label of optionalLabels) source_views[label] = views[label]
  return { source_views }
}

// Reads the `views` label list the backend stamps onto a completed generate
// job's output artifact metadata, in the fixed front/back/left/right order.
export function sourceViewChipLabels(views: unknown): ViewLabel[] {
  if (!Array.isArray(views)) return []
  return VIEW_LABEL_ORDER.filter((label) => views.includes(label))
}

// --- Auto-generate views (E3) -------------------------------------------

// Given one approved front image, ask the backend to synthesize genuine
// back/left/right views with the same image-conditioned ComfyUI path the
// T-pose candidate uses (never a mirrored copy of the front image -- the
// project log recorded that a mirrored front makes results WORSE).
export function synthesizeViewsJobPayload(
  sourceArtifactId: string,
  runId: string | null,
  seed: number,
  views: OptionalViewLabel[] = ['back', 'left', 'right'],
  // Bead 4vquq: a per-tile regenerate on a run that owns a canonical T-pose
  // set must redraw from THAT SET'S FRONT, and say so explicitly. Left to
  // `select_view_synthesis_source`, an unapproved set front loses to the raw
  // upload -- which is exactly the reference the set exists to retire.
  referenceArtifactId?: string,
) {
  return {
    kind: 'synthesize-views' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    seed,
    settings: referenceArtifactId
      ? { views, source_artifact_id: referenceArtifactId }
      : { views },
  }
}

// Reads a completed synthesize-views job's primary output artifact metadata
// (`synthesized_view` on the artifact itself, `sibling_views` for the other
// two) into a flat label -> artifact id list, in front/back/left/right order.
export function synthesizedViewArtifactIds(
  primaryArtifactId: string,
  metadata: { synthesized_view?: unknown; sibling_views?: unknown } | undefined,
): Array<[OptionalViewLabel, string]> {
  const entries: Array<[OptionalViewLabel, string]> = []
  const primaryLabel = metadata?.synthesized_view
  if (typeof primaryLabel === 'string' && (VIEW_LABEL_ORDER as readonly string[]).includes(primaryLabel)) {
    entries.push([primaryLabel as OptionalViewLabel, primaryArtifactId])
  }
  const siblings = metadata?.sibling_views
  if (siblings && typeof siblings === 'object') {
    for (const [label, id] of Object.entries(siblings as Record<string, unknown>)) {
      if (typeof id === 'string' && (VIEW_LABEL_ORDER as readonly string[]).includes(label)) {
        entries.push([label as OptionalViewLabel, id])
      }
    }
  }
  const order = VIEW_LABEL_ORDER as readonly string[]
  return entries.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]))
}

// --- Canonical T-pose set (bead 4vquq) ----------------------------------

// One operator step replaces Prepare + Auto-generate-views: the backend draws
// the T-pose FRONT, gates it, then fans back/left/right off that front, and
// registers all four as one set. From then on the raw upload is a REFERENCE
// (palette and identity) and never conditions a biped generate.
export function createTposeSetJobPayload(
  sourceArtifactId: string,
  runId: string | null,
  seed: number,
  views: OptionalViewLabel[] = ['back', 'left', 'right'],
) {
  return {
    kind: 'create-tpose-set' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    seed,
    settings: { views },
  }
}

// Stages whose completed output is an approvable T-pose front. Mirrors
// backend/app/run_store.py's PREPARED_FRONT_STAGES: `prepare-image` is the
// original single-step form, `create-tpose-set` the composite.
export const PREPARED_FRONT_STAGES = ['prepare-image', 'create-tpose-set'] as const
export type PreparedFrontStage = typeof PREPARED_FRONT_STAGES[number]

export function isPreparedFrontStage(stage: string | undefined): boolean {
  return Boolean(stage && (PREPARED_FRONT_STAGES as readonly string[]).includes(stage))
}

export type TposeSetMembership = {
  setId: string
  frontArtifactId: string
  referenceArtifactId: string | null
  members: Partial<Record<ViewLabel, string>>
  failedRoles: OptionalViewLabel[]
  complete: boolean
}

// Reads the `tpose_set` block the backend stamps on every member of a set.
// Returns null for an artifact that is not part of one, so the studio can ask
// any artifact "are you a set member?" without a separate lookup.
export function tposeSetMembership(
  metadata: { tpose_set?: unknown } | undefined,
): TposeSetMembership | null {
  const block = metadata?.tpose_set
  if (!block || typeof block !== 'object') return null
  const { set_id, front_artifact_id, reference_artifact_id, members, failed_members, complete } =
    block as Record<string, unknown>
  if (typeof set_id !== 'string' || !set_id) return null
  const resolved: Partial<Record<ViewLabel, string>> = {}
  if (members && typeof members === 'object') {
    for (const label of VIEW_LABEL_ORDER) {
      const id = (members as Record<string, unknown>)[label]
      if (typeof id === 'string' && id) resolved[label] = id
    }
  }
  const failedRoles = failed_members && typeof failed_members === 'object'
    ? (['back', 'left', 'right'] as const).filter((label) => label in (failed_members as Record<string, unknown>))
    : []
  return {
    setId: set_id,
    frontArtifactId: typeof front_artifact_id === 'string' ? front_artifact_id : '',
    referenceArtifactId: typeof reference_artifact_id === 'string' ? reference_artifact_id : null,
    members: resolved,
    failedRoles: [...failedRoles],
    complete: complete === true,
  }
}

// --- View consistency (bead 8c9gf) --------------------------------------

// The backend's `view_consistency` codes (backend/app/view_consistency.py).
export const LEG_SEPARATION_MISMATCH = 'view_consistency.leg_separation_mismatch'
export const LOWER_SILHOUETTE_DECORRELATED = 'view_consistency.lower_silhouette_decorrelated'

// `error` means the view contradicts the front badly enough that the backend
// will refuse the generate; `warning` is advisory; `unknown` is the check
// declining to score, which must never look like a defect in the view.
export type ViewConsistencyTone = 'error' | 'warning' | 'unknown'
export type ViewConsistencyNotice = { tone: ViewConsistencyTone; headline: string; detail: string }

// Maps one view artifact's stored `view_consistency` result onto the compact
// notice its tile shows. Returns null for a clean view -- and, because this
// reads the CURRENT artifact in the slot rather than any remembered verdict,
// regenerating a view clears its notice by construction: the replacement
// artifact carries its own (clean) result.
//
// Deliberately reads the per-view artifact metadata rather than the job-level
// `view_consistency_summary`: the summary belongs to one synthesize-views job,
// while a slot can hold an operator upload, a view from an older job, or a
// single regenerated view. The artifact is the only thing that always travels
// with the tile, including through the rehydrate path.
export function viewConsistencyNotice(
  label: string,
  metadata: { view_consistency?: unknown } | undefined,
): ViewConsistencyNotice | null {
  const result = metadata?.view_consistency
  if (!result || typeof result !== 'object') return null
  const { status, findings } = result as { status?: unknown; findings?: unknown }
  if (status === 'clean') return null
  const view = label.toUpperCase()
  if (status === 'not_evaluable') {
    return {
      tone: 'unknown',
      headline: `${view}: couldn't verify`,
      detail: 'This view could not be measured against the front reference.',
    }
  }
  const codes = Array.isArray(findings)
    ? findings.map((item) => (item && typeof item === 'object' ? (item as { code?: unknown }).code : null))
    : []
  if (codes.includes(LEG_SEPARATION_MISMATCH)) {
    return {
      tone: 'error',
      headline: `${view} contradicts FRONT`,
      detail: 'legs not visible — regenerate this view',
    }
  }
  if (codes.includes(LOWER_SILHOUETTE_DECORRELATED)) {
    return {
      tone: 'warning',
      headline: `${view} differs from FRONT`,
      detail: 'lower body shape drifted — worth a look before generating',
    }
  }
  if (status === 'warning' || status === 'blocked') {
    return {
      tone: status === 'blocked' ? 'error' : 'warning',
      headline: `${view} differs from FRONT`,
      detail: 'this view disagrees with the front reference',
    }
  }
  return null
}

// Reads a completed `generate` job's output artifact metadata for the exact
// artifact ids it was submitted with (`metadata.source_views`, stamped by
// the backend regardless of whether each view came from Auto-generate views
// or an operator upload -- see backend/app/main.py's `resolved_source_views`).
// This is the authoritative record of "what was actually in the slots" the
// last time Generate ran, so rehydration prefers it over an older or newer
// synthesize-views set that was never actually generated with.
export function generateSourceViewArtifactIds(
  metadata: { source_views?: unknown } | undefined,
): Array<[OptionalViewLabel, string]> {
  const sourceViews = metadata?.source_views
  if (!sourceViews || typeof sourceViews !== 'object') return []
  const entries: Array<[OptionalViewLabel, string]> = []
  for (const label of ['back', 'left', 'right'] as const) {
    const id = (sourceViews as Record<string, unknown>)[label]
    if (typeof id === 'string' && id) entries.push([label, id])
  }
  return entries
}

// Fills only the empty view slots with synthesized artifacts -- a slot the
// operator already filled (upload or an earlier synth) is never replaced.
export function mergeSynthesizedViews<T>(
  current: Partial<Record<OptionalViewLabel, T>>,
  synthesized: ReadonlyArray<readonly [OptionalViewLabel, T]>,
): Partial<Record<OptionalViewLabel, T>> {
  const next = { ...current }
  for (const [label, artifact] of synthesized) {
    if (!next[label]) next[label] = artifact
  }
  return next
}

// Rehydrates the Generate step's back/left/right view slots after a fresh
// mount, a journey-step re-entry, or a full page reload -- the operator's
// filled-in grid is a persisted backend artifact (the run's `generate` and
// `synthesize-views` attempts), never actually lost, but the editable
// `views` state that feeds the next generate payload starts empty every
// time the app doesn't already hold it. Priority, highest first:
//   1. `current` -- whatever the operator has already filled in THIS
//      session (upload or a synth that hasn't been navigated away from)
//      always wins; rehydration only fills gaps.
//   2. `generatedEntries` -- the latest completed generate attempt's own
//      `source_views` (operator-uploaded or synthesized, whichever was
//      actually submitted last).
//   3. `synthesizedEntries` -- the latest completed synthesize-views
//      attempt's set, for a run that has synthesized views but has never
//      actually run Generate yet.
export function rehydrateViewSlots<T>(
  current: Partial<Record<OptionalViewLabel, T>>,
  synthesizedEntries: ReadonlyArray<readonly [OptionalViewLabel, T]>,
  generatedEntries: ReadonlyArray<readonly [OptionalViewLabel, T]>,
): Partial<Record<OptionalViewLabel, T>> {
  const backfilled: Partial<Record<OptionalViewLabel, T>> = {}
  for (const [label, value] of synthesizedEntries) backfilled[label] = value
  for (const [label, value] of generatedEntries) backfilled[label] = value
  const entries = (Object.keys(backfilled) as OptionalViewLabel[]).map((label) => [label, backfilled[label] as T] as const)
  return mergeSynthesizedViews(current, entries)
}

export function latestCompletedAttempt<T extends { stage: string; status: string; output_artifact_id?: string | null }>(
  attempts: T[] | undefined,
  stage: string,
): T | undefined {
  return attempts?.filter((attempt) => attempt.stage === stage && attempt.status === 'completed' && attempt.output_artifact_id).slice(-1)[0]
}

// The Prepare step's Approve affordance must appear for ANY candidate that
// hasn't become the run's current artifact yet -- not only one this browser
// tab itself just finished preparing. Other tabs, other agents, and direct
// API probes all land completed prepare-image attempts on the shared run
// manifest; the newest one is "awaiting approval" whenever it hasn't been
// promoted into run.current_artifact_id. Returns null once the latest
// candidate IS the current artifact (nothing pending) or when the run has
// never run prepare-image at all.
// The newest completed attempt that produced an approvable T-pose front, from
// EITHER form of the step (bead 4vquq): the original `prepare-image`, or the
// `create-tpose-set` composite whose output is the set's front.
export function latestPreparedFrontAttempt<T extends { stage: string; status: string; output_artifact_id?: string | null }>(
  attempts: T[] | undefined,
): T | undefined {
  return attempts?.filter((attempt) => (
    isPreparedFrontStage(attempt.stage) && attempt.status === 'completed' && attempt.output_artifact_id
  )).slice(-1)[0]
}

// The attempt whose output artifact holds the run's newest back/left/right
// set: either a loose `synthesize-views` job or a `create-tpose-set` front
// (whose metadata carries the same `sibling_views` shape). Ordered by the
// run's own attempt order, so a run that rebuilt its canonical set is never
// refilled from the older loose synthesis it replaced.
export const VIEW_SOURCE_STAGES = ['synthesize-views', 'create-tpose-set'] as const

export function latestViewSourceAttempt<T extends { stage: string; status: string; output_artifact_id?: string | null }>(
  attempts: T[] | undefined,
): T | undefined {
  return attempts?.filter((attempt) => (
    (VIEW_SOURCE_STAGES as readonly string[]).includes(attempt.stage)
    && attempt.status === 'completed'
    && attempt.output_artifact_id
  )).slice(-1)[0]
}

// What a reload has to read back to rebuild the Generate step's grid, resolved
// INDEPENDENTLY rather than from the single newest view-source attempt:
//   - `setFront`: the newest completed `create-tpose-set` attempt, whose
//     output is the set's front and carries the membership block. It must
//     survive any later `synthesize-views` attempt on the same run, because
//     regenerating ONE tile of a set submits exactly such an attempt
//     (regenerateView -> autoGenerateViews) -- resolving the front from
//     `latestViewSourceAttempt` alone lost the whole set on reload.
//   - `looseSynth`: the newest completed `synthesize-views` attempt, but only
//     when it is newer than the set (a run that rebuilt its canonical set
//     must not be refilled from the older loose synthesis it replaced).
export function viewRehydrationSources<T extends { stage: string; status: string; output_artifact_id?: string | null }>(
  attempts: T[] | undefined,
): { setFront: T | undefined; looseSynth: T | undefined } {
  const setFront = latestCompletedAttempt(attempts, 'create-tpose-set')
  const newest = latestViewSourceAttempt(attempts)
  return { setFront, looseSynth: newest?.stage === 'synthesize-views' ? newest : undefined }
}

// The back/left/right artifact ids a set's front resolves to. The
// `tpose_set.members` block is the truth once a tile has been replaced
// (replace_tpose_set_member restamps `tpose_set` but not the front's original
// `sibling_views`), so members win; `sibling_views` is the fallback for a
// front stamped before the membership block existed.
export function tposeSetViewArtifactIds(
  frontArtifactId: string,
  metadata: { tpose_set?: unknown; synthesized_view?: unknown; sibling_views?: unknown } | undefined,
): Array<[OptionalViewLabel, string]> {
  const membership = tposeSetMembership(metadata)
  if (membership) {
    const entries: Array<[OptionalViewLabel, string]> = []
    for (const label of ['back', 'left', 'right'] as const) {
      const id = membership.members[label]
      if (id && id !== frontArtifactId) entries.push([label, id])
    }
    if (entries.length) return entries
  }
  return synthesizedViewArtifactIds(frontArtifactId, metadata)
}

export function pendingPreparedCandidateId(
  run: { current_artifact_id?: string | null; attempts?: Array<{ stage: string; status: string; output_artifact_id?: string | null }> } | null | undefined,
): string | null {
  if (!run) return null
  const latest = latestPreparedFrontAttempt(run.attempts)
  if (!latest?.output_artifact_id) return null
  return latest.output_artifact_id === run.current_artifact_id ? null : latest.output_artifact_id
}

export function canGenerateFromPreparedImage(
  source: { stage?: string; metadata?: Record<string, unknown> } | null,
  preparationActive: boolean,
): boolean {
  if (!source || preparationActive) return false
  return isPreparedFrontStage(source.stage) || isPreparedFrontStage(source.metadata?.stage as string | undefined)
}

export function restoreHistoryArtifacts<T>(
  stage: string,
  isSource: boolean,
  originalSource: T,
  restoredArtifact: T,
): { source: T; output: T | null } {
  if (!isSource && isPreparedFrontStage(stage)) {
    return { source: restoredArtifact, output: null }
  }
  return {
    source: originalSource,
    output: isSource ? null : restoredArtifact,
  }
}

export function autoRigJobPayload(sourceArtifactId: string, runId: string | null) {
  return {
    kind: 'auto-rig' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    settings: {},
  }
}
export function canLaunchAutoRig(unirigReady: boolean | null): boolean {
  return unirigReady === true
}

// The desired filename base for a saved GLB export — the run's name when one
// exists, otherwise the artifact's own name with its file extension stripped.
// The backend still owns sanitization, the `-<artifact id>` suffix, and
// collision handling; this only decides which human-readable name to try.
export function exportGlbNameHint(runName: string | null | undefined, artifactName: string): string {
  const base = (runName && runName.trim()) || artifactName
  return base.replace(/\.[^./]+$/, '').trim() || 'model'
}

// `settings.animation` names the action Blender bakes onto the retargeted
// output (backend/workers/blender_worker.py's retarget-bvh op: only used as
// the display name of the new `bpy.data.actions.new(animation)` datablock —
// it does not need to match the BVH's own source action name, since the
// motion itself is read from the BVH file regardless of what this is
// called). Defaults to the historical curated Sword Attack label for the
// one caller (App.tsx's terminal-hydration auto-retarget of a freshly
// generated-motion BVH) that has no clip identity of its own to name it
// with; the curated-clip picker passes its selected clip's own label.
export function retargetMotionJobPayload(
  sourceArtifactId: string,
  motionArtifactId: string,
  runId: string | null,
  actionLabel: string = 'Sword_Attack',
) {
  return {
    kind: 'retarget-motion' as const,
    submission_id: `auto-retarget-${motionArtifactId}`,
    source_artifact_id: sourceArtifactId,
    run_id: runId,
    settings: { motion_artifact_id: motionArtifactId, animation: actionLabel },
  }
}

export function isMotionPipelineJob(kind?: string): boolean {
  return kind === 'generate-motion' || kind === 'retarget-motion' || kind === 'apply-clip'
}

// N5's client-side mirror of backend/app/main.py's `_is_template_conformant_rig`:
// a rig is eligible for direct clip application only when its rig-certificate
// is READY *and* the family-template conformance check (N3) came back
// completely clean. `rig_quality.template_conformance` being `[]` (an empty
// array) means "checked, zero findings"; `undefined`/anything else means no
// conformance data at all (pre-N3, or a non-humanoid rig) and must not be
// treated as conformant even though it also adds no findings.
export function isTemplateConformantRig(metadata?: Record<string, unknown> | null): boolean {
  const decisions = metadata?.quality_decisions as Record<string, { status?: string; policy_version?: number }> | undefined
  const certificate = decisions?.['rig-certificate']
  const certified = certificate?.status === 'ready' && (certificate.policy_version ?? 0) >= 2
  if (!certified) return false
  const rigQuality = metadata?.rig_quality as { template_conformance?: unknown } | undefined
  const conformance = rigQuality?.template_conformance
  return Array.isArray(conformance) && conformance.length === 0
}

// The direct-attach counterpart to retargetMotionJobPayload above: no BVH
// export, no retarget math -- `clip` is the family clip store's exact
// clip_name (assets/skeletons/<family>/clips/clips-manifest.json), the same
// string CuratedMotionClip.animation already carries (both are derived from
// the same donor with the same "_Armature" suffix convention -- see
// animations.ts), so the existing curated-clip picker needs no changes to
// feed this.
export function applyClipJobPayload(
  sourceArtifactId: string,
  runId: string | null,
  clipName: string,
  family: string = 'biped',
) {
  return {
    kind: 'apply-clip' as const,
    submission_id: `auto-apply-clip-${clipName}`,
    source_artifact_id: sourceArtifactId,
    run_id: runId,
    settings: { clip: clipName, family },
  }
}

export function rigWorkspaceSurface(mode: 'rig' | 'animate' | undefined, rigged: boolean) {
  if (!rigged) return 'authoring' as const
  return mode === 'rig' ? 'skeleton' as const : 'animations' as const
}

export function hasActiveStudioJob(job?: { status?: string } | null): boolean {
  return Boolean(job && ['queued', 'running'].includes(job.status || ''))
}

export function canAcceptMesh2MotionExport(
  busy: boolean,
  job?: { kind?: string; status?: string } | null,
  terminalHydrationActive = false,
): boolean {
  return !busy && !terminalHydrationActive && !hasActiveStudioJob(job)
}

export function canAcceptMesh2MotionContext(
  event: { run?: string; input?: string; mode?: string },
  expected: { runId: string; artifactId: string; mode: 'rig' | 'animate' },
): boolean {
  const expectedEventMode = expected.mode === 'rig' ? 'edit-clean-mesh' : 'animate'
  return event.run === expected.runId && event.input === expected.artifactId && event.mode === expectedEventMode
}

export function mesh2MotionExportInputArtifactId(
  activeArtifactId: string,
  cleanMeshArtifactId: string | undefined,
  mode: 'rig' | 'animate',
): string {
  return mode === 'rig' && cleanMeshArtifactId ? cleanMeshArtifactId : activeArtifactId
}

export function isArrayBufferPayload(value: unknown): value is ArrayBuffer {
  return Object.prototype.toString.call(value) === '[object ArrayBuffer]'
}

export function shouldApplyCompletedArtifact(
  completedJobId: string,
  completedContextVersion: number,
  activeJobId: string | null,
  currentContextVersion: number,
) {
  return completedJobId === activeJobId && completedContextVersion === currentContextVersion
}

export function shouldMarkTerminalHydrationComplete(
  hydrationSucceeded: boolean,
  workspaceStillActive: boolean,
): boolean {
  return hydrationSucceeded && workspaceStillActive
}

export function shouldStartTerminalHydration(
  jobId: string,
  hydratedJobId: string | null,
  failedJobId: string | null,
  retryRequestedJobId: string | null,
): boolean {
  return hydratedJobId !== jobId && (failedJobId !== jobId || retryRequestedJobId === jobId)
}

export function shouldApplyHydratedRun(
  run: { id: string; source_artifact_id: string; current_artifact_id?: string | null; updated_at: number },
  expected: { id: string; source_artifact_id: string; current_artifact_id?: string | null; updated_at: number },
  outputArtifactId: string,
  promoteOutput: boolean,
  activeRunId: string | null,
): boolean {
  return activeRunId === expected.id
    && run.id === expected.id
    && run.source_artifact_id === expected.source_artifact_id
    && run.updated_at >= expected.updated_at
    && (promoteOutput
      ? run.current_artifact_id === outputArtifactId
      : run.current_artifact_id === expected.current_artifact_id)
}

export function canGenerateMotion(
  artifact: { metadata?: { quality_decisions?: Record<string, { status?: string; policy_version?: number }> } },
  kimodoReady: boolean,
  blenderReady: boolean,
): boolean {
  const decisions = artifact.metadata?.quality_decisions
  const required = ['rig', 'rig-certificate', 'animation-review', 'deformation-review']
  return kimodoReady
    && blenderReady
    && required.every((name) => decisions?.[name]?.status === 'ready')
    && (decisions?.['rig-certificate']?.policy_version ?? 0) >= 2
}

export function canRestoreHistoryArtifact(stage: string, artifactId: string, currentArtifactId?: string | null, promoted = false): boolean {
  const rigStages = new Set(['rig', 'auto-rig', 'rig-adjust', 'repair-weights', 'rig-transfer', 'target-proxy-rig', 'retarget-motion'])
  return !rigStages.has(stage) || promoted || artifactId === currentArtifactId
}

export function shouldPromoteCompletedOutput(job: { kind: string; promoted_output?: boolean }): boolean {
  return job.promoted_output !== false
}

export function shouldActivateRecordedArtifact(
  currentArtifactId: string | null | undefined,
  recordedArtifactId: string,
): boolean {
  return currentArtifactId === recordedArtifactId
}

export function shouldPreviewRecordedArtifact(
  recordedArtifactId: string,
  fetchedArtifactId: string,
): boolean {
  return recordedArtifactId === fetchedArtifactId
}

export function isActiveJobStatus(status: string): boolean {
  return status === 'queued' || status === 'running'
}

function abortOnSignal<T>(request: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(signal.reason || new Error('Request aborted'))
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason || new Error('Request aborted'))
    const cleanup = () => signal.removeEventListener('abort', onAbort)
    signal.addEventListener('abort', onAbort, { once: true })
    request.then(
      (value) => { cleanup(); resolve(value) },
      (error) => { cleanup(); reject(error) },
    )
  })
}

export async function retryNetworkPost<T>(post: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  const attempt = () => {
    if (!signal) return post()
    if (signal.aborted) return Promise.reject(signal.reason || new Error('Request aborted'))
    return abortOnSignal(post(), signal)
  }
  try {
    return await attempt()
  } catch (error) {
    if (!(error instanceof TypeError) || signal?.aborted) throw error
    return attempt()
  }
}

// --- Subject kind (H) ---------------------------------------------------

export type SubjectKind = 'auto' | 'object' | 'creature' | 'biped'

// Only biped/auto subjects benefit from a T-pose reference; a pure object
// or a non-humanoid creature never needs one.
export function showsPrepareCard(subjectKind: SubjectKind): boolean {
  return subjectKind === 'biped' || subjectKind === 'auto'
}

export function autoRigCaption(subjectKind: SubjectKind): string {
  return subjectKind === 'creature' ? 'UniRig · chain rig' : 'UniRig · biped rig'
}

// --- Stage-level artifact control (I) -----------------------------------

// The canonical stage ids a "Working on" card can show a picker for. Distinct
// from the journey step ids: Skeleton has two pickers depending on whether a
// rig candidate exists yet (a bare mesh pre-rig, or the rig itself).
export type WorkingOnStage = 'generate' | 'clean' | 'skeleton-pre-rig' | 'skeleton' | 'animate' | 'paint'

export type RunAssetOption = {
  artifactId: string
  stage: string
  completedAt: number
  isSource: boolean
}

type RunAssetAttempt = { stage: string; status: string; output_artifact_id?: string | null; completed_at?: number | null; created_at?: number }
type RunAssetSource = { source_artifact_id: string; source_artifact_kind: string; attempts: RunAssetAttempt[] }

const PREPARE_OUTPUT_STAGES = [...PREPARED_FRONT_STAGES]
const GENERATE_OUTPUT_STAGES = ['generate']
// `canonicalize` is the S2 contract clean (docs/PIPELINE_POLICY.md); the
// other three are the pre-S2 repair passes. All four produce a mesh the
// downstream mesh-consuming pickers (skeleton-pre-rig, paint) can take.
const CLEAN_OUTPUT_STAGES = ['canonicalize', 'retopologize', 'retopo', 'repair-topology', 'paint-bake', 'project-views', 'rebuild-views']
const RIG_OUTPUT_STAGES = ['rig', 'auto-rig', 'rig-adjust', 'rig-transfer', 'target-proxy-rig', 'repair-weights']

const WORKING_ON_STAGES: Record<WorkingOnStage, string[]> = {
  // Generate consumes images: the run's approved T-pose (prepare-image)
  // candidates, not Generate's own mesh output.
  generate: PREPARE_OUTPUT_STAGES,
  clean: GENERATE_OUTPUT_STAGES,
  'skeleton-pre-rig': [...GENERATE_OUTPUT_STAGES, ...CLEAN_OUTPUT_STAGES],
  skeleton: RIG_OUTPUT_STAGES,
  // The "which rig is being animated" picker (I) — the Animations list
  // (animations.ts / AnimationList.tsx) is the separate picker for "which
  // embedded clip is showing," keyed off retarget-motion/generate-motion
  // outputs directly rather than through this WorkingOnStage map.
  animate: RIG_OUTPUT_STAGES,
  paint: [...GENERATE_OUTPUT_STAGES, ...CLEAN_OUTPUT_STAGES],
}

// The run's own uploaded source only belongs on the picker when its native
// kind already matches what that stage consumes: an image for Generate, or
// an already-a-mesh (GLB) upload for the mesh-consuming stages — a rig-only
// run that skipped the image pipeline entirely.
function sourceCompatibleWithStage(stage: WorkingOnStage, sourceKind: string): boolean {
  if (stage === 'generate') return isImageArtifactKind(sourceKind)
  if (stage === 'clean' || stage === 'skeleton-pre-rig' || stage === 'paint') return sourceKind === 'glb'
  return false
}

// The run's compatible artifacts for a given stage's "Working on" picker,
// newest first, with the source artifact included where it is itself a
// compatible kind. Images for generate; meshes for clean/skeleton-pre-rig/
// paint; rig candidates for skeleton/animate; animated outputs for review.
export function runAssetOptions(
  run: RunAssetSource | null,
  stage: WorkingOnStage,
): RunAssetOption[] {
  if (!run) return []
  const wanted = WORKING_ON_STAGES[stage] || []
  const seen = new Set<string>()
  const options: RunAssetOption[] = []
  for (const attempt of run.attempts) {
    if (attempt.status !== 'completed' || !attempt.output_artifact_id) continue
    if (!wanted.includes(attempt.stage)) continue
    if (seen.has(attempt.output_artifact_id)) continue
    seen.add(attempt.output_artifact_id)
    options.push({
      artifactId: attempt.output_artifact_id,
      stage: attempt.stage,
      completedAt: attempt.completed_at ?? attempt.created_at ?? 0,
      isSource: false,
    })
  }
  if (sourceCompatibleWithStage(stage, run.source_artifact_kind) && !seen.has(run.source_artifact_id)) {
    options.push({ artifactId: run.source_artifact_id, stage: 'source', completedAt: -Infinity, isSource: true })
  }
  return options.sort((a, b) => b.completedAt - a.completedAt)
}

// --- Clean hints and recommendation (I3) --------------------------------

export function cleanTargetExceedsSource(targetVertices: number, sourceVertices: number | null): boolean {
  return sourceVertices !== null && targetVertices > sourceVertices
}

const CLEAN_OUTPUT_STAGE_SET = new Set(CLEAN_OUTPUT_STAGES)
export function isReCleaningCleanOutput(stage?: string): boolean {
  return Boolean(stage && CLEAN_OUTPUT_STAGE_SET.has(stage))
}

export type MeshQuality = {
  components?: number
  largest_component_face_ratio?: number
  non_manifold_edges?: number
  boundary_edges?: number
  watertight?: boolean
  faces?: number
  vertices?: number
}

// --- Canonicalize: the S2 contract clean (r029o) -------------------------
//
// `canonicalize` is the mandatory raw-mesh -> game-ready-mesh contract stage
// (docs/PIPELINE_POLICY.md, backend/app/canonicalize_policy.py): voxel remesh
// + bake-from-original, then a blocking quality gate on one connected
// component, watertight, zero non-manifold edges, a triangle count inside the
// family band, and appearance preservation against the S1 source. The backend
// defaults `settings.method` to native.DEFAULT_CANONICALIZE_METHOD ("voxel");
// the UI sends it explicitly so the submitted payload is self-describing.
export const CANONICALIZE_DEFAULT_METHOD = 'voxel'

export type CanonicalizeMethod = 'voxel' | 'instant-meshes'

export type CanonicalizeJobPayload = {
  kind: 'canonicalize'
  source_artifact_id: string
  run_id: string | undefined
  settings: {
    method: CanonicalizeMethod
    target_vertices?: number
    quality_tier?: QualityTier
    target_triangles?: number
  }
}

// --- Mesh density: quality tiers (i63ky) ---------------------------------
//
// The operator picks how many triangles the canonical mesh should come out
// at. The selection travels as `settings.quality_tier`; a custom number
// travels as `settings.target_triangles` and overrides the tier
// (backend/app/quality_tiers.py owns the same precedence). Everything the
// backend derives from it — the decimation target, the raw voxel search band,
// the S2 gate's triangle band and the bake atlas floor — follows from this one
// choice, so the gate never blocks a mesh for being the density it was asked
// to be.
export type QualityTier = 'draft' | 'standard' | 'high' | 'ultra'

export const CANONICALIZE_DEFAULT_QUALITY_TIER: QualityTier = 'standard'

// Mirrors `quality_tiers.QUALITY_TIER_TARGETS`. The captions are the whole
// point of the control: an operator choosing a density is trading file size,
// clean/rig wall time and viewer smoothness for surface detail, and the panel
// says so in those words rather than in triangle counts alone.
export const QUALITY_TIER_OPTIONS: {
  id: QualityTier; label: string; targetTriangles: number; caption: string
}[] = [
  {
    id: 'draft', label: 'Draft', targetTriangles: 10_000,
    caption: 'Smallest file, fastest clean and rig. Soft surface detail — good for blocking out a pose.',
  },
  {
    id: 'standard', label: 'Standard', targetTriangles: 30_000,
    caption: 'The default. Game-engine budget: sharp enough to read close up, quick to rig, smooth in any viewer.',
  },
  {
    id: 'high', label: 'High', targetTriangles: 80_000,
    caption: 'Crisper silhouette and folds. Roughly 3x the file size, a noticeably longer clean and rig.',
  },
  {
    id: 'ultra', label: 'Ultra', targetTriangles: 150_000,
    caption: 'Most geometry the pipeline will keep. Large files, the slowest clean and rig, heavier in the viewer.',
  },
]

// Same bounds `quality_tiers.CUSTOM_TARGET_TRIANGLES_{MINIMUM,MAXIMUM}` enforce,
// so a value the panel accepts is a value the API accepts.
export const CUSTOM_TRIANGLE_TARGET_MINIMUM = 1_000
export const CUSTOM_TRIANGLE_TARGET_MAXIMUM = 500_000

export type CanonicalizeDensity = { tier: QualityTier; customTriangles: number | null }

export const CANONICALIZE_DEFAULT_DENSITY: CanonicalizeDensity = {
  tier: CANONICALIZE_DEFAULT_QUALITY_TIER,
  customTriangles: null,
}

export function isQualityTier(value: unknown): value is QualityTier {
  return QUALITY_TIER_OPTIONS.some((option) => option.id === value)
}

// A custom target only overrides the tier when it is a usable number inside
// the backend's range; a half-typed or out-of-range box falls back to the
// selected tier rather than submitting a job the API would reject.
export function customTriangleTarget(value: unknown): number | null {
  const number = typeof value === 'string' ? Number(value.trim()) : value
  if (typeof number !== 'number' || !Number.isFinite(number)) return null
  const rounded = Math.round(number)
  if (rounded < CUSTOM_TRIANGLE_TARGET_MINIMUM || rounded > CUSTOM_TRIANGLE_TARGET_MAXIMUM) return null
  return rounded
}

// Resolved client-side purely so the panel can show what it is about to ask
// for; the backend re-resolves the same rule and is the authority.
export function resolvedTriangleTarget(density: CanonicalizeDensity): number {
  const custom = customTriangleTarget(density.customTriangles)
  if (custom !== null) return custom
  const option = QUALITY_TIER_OPTIONS.find((item) => item.id === density.tier)
  return option ? option.targetTriangles : 30_000
}

// The density selection is remembered for the session (sessionStorage), not
// forever: it is a per-sitting intent ("I'm doing hero meshes today"), and a
// stale Ultra silently applying weeks later is the kind of surprise that costs
// twenty minutes of GPU.
export const CANONICALIZE_DENSITY_STORAGE_KEY = 'aismith.canonicalize.density'

export function serializeCanonicalizeDensity(density: CanonicalizeDensity): string {
  return JSON.stringify({ tier: density.tier, custom_triangles: density.customTriangles })
}

export function readCanonicalizeDensity(raw: string | null): CanonicalizeDensity {
  if (!raw) return CANONICALIZE_DEFAULT_DENSITY
  try {
    const parsed = JSON.parse(raw) as { tier?: unknown; custom_triangles?: unknown }
    return {
      tier: isQualityTier(parsed?.tier) ? parsed.tier : CANONICALIZE_DEFAULT_QUALITY_TIER,
      customTriangles: customTriangleTarget(parsed?.custom_triangles),
    }
  } catch {
    return CANONICALIZE_DEFAULT_DENSITY
  }
}

// `targetVertices` is optional on purpose: omitting it lets the backend derive
// the target from the selected tier rather than pinning the operator's legacy
// Clean slider value onto the contract stage.
export function canonicalizeJobPayload(
  sourceArtifactId: string,
  runId: string | null,
  targetVertices?: number | null,
  method: CanonicalizeMethod = CANONICALIZE_DEFAULT_METHOD,
  density?: CanonicalizeDensity | null,
): CanonicalizeJobPayload {
  const settings: CanonicalizeJobPayload['settings'] = { method }
  if (typeof targetVertices === 'number' && Number.isFinite(targetVertices)) {
    settings.target_vertices = Math.round(targetVertices)
  }
  if (density) {
    settings.quality_tier = density.tier
    const custom = customTriangleTarget(density.customTriangles)
    if (custom !== null) settings.target_triangles = custom
  }
  return {
    kind: 'canonicalize',
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    settings,
  }
}

// What the finished artifact says about its own density: what was asked for,
// what came out, and the band the gate judged it against. Null for an artifact
// canonicalized before tiers existed, so the card can stay silent rather than
// inventing a "standard" label for a run that never chose one.
export type CanonicalizeDensityReport = {
  tier: string
  requestedTier: string | null
  targetTriangles: number | null
  achievedTriangles: number | null
  band: [number, number] | null
  atlasSize: number | null
}

// A finite number from loose metadata, or null -- the shared reader every
// metadata report below uses for its optional numeric fields.
function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function canonicalizeDensityReport(
  metadata?: Record<string, unknown> | null,
): CanonicalizeDensityReport | null {
  const tier = metadata?.quality_tier
  if (typeof tier !== 'string' || !tier) return null
  const band = metadata?.triangle_budget_band
  return {
    tier,
    requestedTier: typeof metadata?.quality_tier_requested === 'string' ? metadata.quality_tier_requested : null,
    targetTriangles: numberOrNull(metadata?.target_triangles),
    achievedTriangles: numberOrNull(metadata?.achieved_triangles),
    band: Array.isArray(band) && band.length === 2 && typeof band[0] === 'number' && typeof band[1] === 'number'
      ? [band[0], band[1]]
      : null,
    atlasSize: numberOrNull(metadata?.bake_texture_size),
  }
}

export type CanonicalizeVerdict = Pick<QualityDecision, 'status' | 'findings' | 'recommended_actions'> & {
  policy_version?: number
}

// The canonicalize gate's decision as the backend wrote it onto the output
// artifact's metadata (`quality_decisions["canonicalize"]`). Returns null when
// the artifact never went through the stage, so the panel can tell "not run
// yet" from "ran and was blocked". `findings` is normalised to an array so a
// malformed/partial decision still renders its status instead of throwing.
export function canonicalizeVerdict(
  metadata?: Record<string, unknown> | null,
): CanonicalizeVerdict | null {
  const decisions = metadata?.quality_decisions as Record<string, Partial<QualityDecision>> | undefined
  const decision = decisions?.canonicalize
  if (!decision || typeof decision.status !== 'string') return null
  return {
    status: decision.status,
    findings: Array.isArray(decision.findings) ? (decision.findings as QualityFinding[]) : [],
    recommended_actions: Array.isArray(decision.recommended_actions) ? decision.recommended_actions : [],
    policy_version: decision.policy_version,
  }
}

// A blocked/needs_review candidate is retained but never auto-promoted
// (main.py sets job.promoted_output from the gate), so the panel has to offer
// the pin explicitly rather than assuming the run advanced.
export function canonicalizeCandidatePinnable(
  verdict: CanonicalizeVerdict | null,
  promotedOutput?: boolean,
): boolean {
  return verdict !== null && promotedOutput !== true
}

// --- View projection: T-pose conditioning views baked over the S2 texture -

export type ProjectionVerdict = {
  status: 'green' | 'needs_review' | 'blocked'
  findings: QualityFinding[]
  recommended_actions: string[]
  policy_version?: string
}

// The project-views gate's decision, wherever the backend wrote it: a
// standalone `project-views` job stamps it directly, and `canonicalize`
// stamps the same key when it ran the projection pass in place (native.py's
// `run_canonicalize(view_images=...)`). Returns null when neither ran it, so
// the card can distinguish "not attempted" from "ran and needs a look".
export function projectionVerdict(
  metadata?: Record<string, unknown> | null,
): ProjectionVerdict | null {
  const decisions = metadata?.quality_decisions as Record<string, Partial<ProjectionVerdict>> | undefined
  const decision = decisions?.project_views
  if (!decision || typeof decision.status !== 'string') return null
  return {
    status: decision.status as ProjectionVerdict['status'],
    findings: Array.isArray(decision.findings) ? (decision.findings as QualityFinding[]) : [],
    recommended_actions: Array.isArray(decision.recommended_actions) ? decision.recommended_actions : [],
    policy_version: decision.policy_version,
  }
}

export type ProjectionReport = {
  viewsUsed: string[]
  perRoleCoverage: Record<string, number>
  projectedTexelFraction: number | null
  fallbackTexelFraction: number | null
  meanColorDeltaVsBake: number | null
  atlasSize: number | null
  caveats: string[]
}

// The projection pass's own coverage report (`metadata.view_projection` --
// `view_projection.projection_report`'s shape), independent of the verdict
// above: a project-views job merges the report directly onto the artifact's
// metadata (no `view_projection` wrapper key), while canonicalize nests it
// under `metadata.view_projection`. Checked in that order so the card reads
// correctly from either job kind.
export function projectionReport(
  metadata?: Record<string, unknown> | null,
): ProjectionReport | null {
  const nested = metadata?.view_projection as Record<string, unknown> | undefined
  const source = (nested && typeof nested === 'object' && !('skipped' in nested))
    ? nested
    : (typeof metadata?.projected_texel_fraction === 'number' ? metadata : null)
  if (!source) return null
  const perRoleCoverage = source.per_role_coverage
  return {
    viewsUsed: Array.isArray(source.views_used) ? (source.views_used as string[]) : [],
    perRoleCoverage: (perRoleCoverage && typeof perRoleCoverage === 'object')
      ? (perRoleCoverage as Record<string, number>)
      : {},
    projectedTexelFraction: numberOrNull(source.projected_texel_fraction),
    fallbackTexelFraction: numberOrNull(source.fallback_texel_fraction),
    meanColorDeltaVsBake: numberOrNull(source.mean_color_delta_vs_bake),
    atlasSize: numberOrNull(source.atlas_size),
    caveats: Array.isArray(source.caveats) ? (source.caveats as string[]) : [],
  }
}

// --- Source fidelity: how closely a generate/canonicalize/project-views
// output matches the source photo (backend/app/source_fidelity.py) ---

export type SourceFidelityVerdict = {
  status: 'green' | 'needs_review' | 'not_evaluable'
  findings: QualityFinding[]
  recommended_actions: string[]
  policy_version?: string
  summary: {
    silhouetteIou: number | null
    maskedSsim: number | null
    faceSsim: number | null
    perceptualDistance: number | null
    maskedLumaDelta: number | null
    detailRatio: number | null
    faceDetailRatio: number | null
  }
}

// The source-fidelity gate's decision off the output artifact's metadata
// (`quality_decisions.source_fidelity`), mirroring projectionVerdict above.
// Returns null when the artifact never went through the pass.
export function sourceFidelityVerdict(
  metadata?: Record<string, unknown> | null,
): SourceFidelityVerdict | null {
  const decisions = metadata?.quality_decisions as Record<string, Partial<SourceFidelityVerdict> & { summary?: Record<string, unknown> }> | undefined
  const decision = decisions?.source_fidelity
  if (!decision || typeof decision.status !== 'string') return null
  const summary: Record<string, unknown> = (decision.summary && typeof decision.summary === 'object') ? decision.summary : {}
  return {
    status: decision.status as SourceFidelityVerdict['status'],
    findings: Array.isArray(decision.findings) ? (decision.findings as QualityFinding[]) : [],
    recommended_actions: Array.isArray(decision.recommended_actions) ? decision.recommended_actions : [],
    policy_version: decision.policy_version,
    summary: {
      silhouetteIou: numberOrNull(summary.silhouette_iou),
      maskedSsim: numberOrNull(summary.masked_ssim),
      faceSsim: numberOrNull(summary.face_ssim),
      perceptualDistance: numberOrNull(summary.perceptual_distance),
      maskedLumaDelta: numberOrNull(summary.masked_luma_delta),
      detailRatio: numberOrNull(summary.detail_ratio),
      faceDetailRatio: numberOrNull(summary.face_detail_ratio),
    },
  }
}

export type SourceFidelityReport = {
  skipped: string | null
  sideBySideArtifactId: string | null
  photoArtifactId: string | null
  bboxWidthRatio: number | null
  cameraKind: 'perspective' | 'orthographic' | null
}

// The source-fidelity pass's own report (`metadata.source_fidelity`) --
// either `{skipped: reason}` when the pass never ran, or the full report with
// the side-by-side comparison image and alignment/camera detail. Independent
// of the verdict above, same as projectionReport is of projectionVerdict.
export function sourceFidelityReport(
  metadata?: Record<string, unknown> | null,
): SourceFidelityReport | null {
  const source = metadata?.source_fidelity as Record<string, unknown> | undefined
  if (!source || typeof source !== 'object') return null
  const alignment = source.alignment as Record<string, unknown> | undefined
  const camera = source.camera as Record<string, unknown> | undefined
  const cameraKind = camera?.kind === 'perspective' || camera?.kind === 'orthographic' ? camera.kind : null
  return {
    skipped: typeof source.skipped === 'string' ? source.skipped : null,
    sideBySideArtifactId: typeof source.side_by_side_artifact_id === 'string' ? source.side_by_side_artifact_id : null,
    photoArtifactId: typeof source.photo_artifact_id === 'string' ? source.photo_artifact_id : null,
    bboxWidthRatio: numberOrNull(alignment?.bbox_width_ratio),
    cameraKind,
  }
}

export type CleanMethod = 'canonicalize' | 'ai-refine' | 'instant-meshes' | 'rebuild-views'

// Which Clean method the panel opens on when a stage output is restored
// (journey navigation or History's "use this artifact"). Anything that is not
// itself a legacy repair output lands on the S2 canonical path.
export function cleanMethodForStage(stage?: string): CleanMethod {
  if (stage === 'retopologize') return 'ai-refine'
  if (stage === 'retopo') return 'instant-meshes'
  if (stage === 'rebuild-views') return 'rebuild-views'
  return 'canonicalize'
}

// --- Rebuild from views: Pixal3D multi-view rebuild + real-pixel overlay ---

export type RebuildViewsRoleScore = { ssim: number | null; face: number | null }
export type RebuildViewsReport = {
  meanIou: number | null
  front: RebuildViewsRoleScore | null
  back: RebuildViewsRoleScore | null
}

// `rebuild-views` metadata: `mv_iou` is approach9's own report.json (how well
// the rebuild matches the views it was painted from) and `gate` is
// scripts/source_fidelity_multiview.py's per-role table measured against the
// REAL reference photos. Tolerant of every field being absent -- a job whose
// gate never ran still has a mean IoU worth showing.
export function rebuildViewsReport(
  metadata?: Record<string, unknown> | null,
): RebuildViewsReport {
  const mvIou = metadata?.mv_iou as Record<string, unknown> | undefined
  const gate = metadata?.gate as Record<string, Record<string, unknown>> | undefined
  const role = (name: 'front' | 'back'): RebuildViewsRoleScore | null => {
    const entry = gate?.[name]
    if (!entry || typeof entry !== 'object') return null
    return { ssim: numberOrNull(entry.masked_ssim), face: numberOrNull(entry.face_ssim) }
  }
  return { meanIou: numberOrNull(mvIou?.mean_iou), front: role('front'), back: role('back') }
}

export function rebuildViewsJobPayload(sourceArtifactId: string, runId: string | null, seed: number) {
  return {
    kind: 'rebuild-views' as const,
    source_artifact_id: sourceArtifactId,
    run_id: runId || undefined,
    seed,
    settings: {},
  }
}

export type CleanRecommendationChoice = 'skip' | 'ai-refine' | 'instant-meshes'
export type CleanRecommendation = { choice: CleanRecommendationChoice; reason: string }

const TRELLIS_BACKENDS = new Set(['trellis', 'trellis2', 'trellis-cpp', 'trellis-refiner'])

// Computed from the working-on artifact's mesh_quality metadata and its
// provenance (backend/stage) — never network state, so it can render
// instantly and be unit-tested directly.
export function cleanRecommendation(artifact: {
  metadata?: { mesh_quality?: MeshQuality; backend?: string }
} | null): CleanRecommendation | null {
  const quality = artifact?.metadata?.mesh_quality
  if (!quality) return null
  const backend = artifact?.metadata?.backend
  const fromTrellis = backend === undefined || TRELLIS_BACKENDS.has(backend)
  const components = quality.components ?? 1
  const faces = quality.faces ?? 0
  const nonManifold = quality.non_manifold_edges ?? 0
  const boundary = quality.boundary_edges ?? 0

  if (nonManifold > 500 || boundary > 0 || components > 1) {
    return {
      choice: 'ai-refine',
      reason: 'Fallback if canonical clean is blocked: AI Refine re-reconstructs geometry and texture with TRELLIS.2 (GPU). It can regress the character\'s appearance, so it is not the default path.',
    }
  }
  if (fromTrellis && components === 1 && faces <= 250_000) {
    return {
      choice: 'skip',
      reason: 'No legacy repair needed: this generated mesh is a single component with no open boundaries, so AI Refine and Instant Meshes have nothing to fix. Canonical clean is still required before rigging (policy S2).',
    }
  }
  return {
    choice: 'instant-meshes',
    reason: 'Fallback if canonical clean is blocked: Instant Meshes (legacy) does a quad-dominant remesh for dense scans; enable Repair holes if the scan has gaps.',
  }
}

export async function pollJobUntilTerminal<T extends { id: string; status: string }>(
  initial: T,
  read: (id: string) => Promise<T>,
  pause: () => Promise<unknown>,
  shouldContinue: () => boolean = () => true,
): Promise<T> {
  let current = initial
  while (isActiveJobStatus(current.status) && shouldContinue()) {
    await pause()
    if (!shouldContinue()) break
    try {
      current = await read(current.id)
    } catch (error) {
      if (!(error instanceof TypeError)) throw error
    }
  }
  return current
}


// --- Active-job resync (tab-visibility recovery) ---------------------------
// The studio keeps the in-flight job in volatile React state only. Anything
// that remounts the app while a long job (auto-rig) is running -- a reload,
// Chrome discarding a backgrounded tab under Memory Saver, a dev-server
// reconnect -- used to drop that state while the job kept running
// server-side, leaving a UI that looks idle and a completion that never
// lands. The active job id is therefore persisted next to
// ACTIVE_RUN_STORAGE_KEY and re-read whenever the tab comes back.
export const ACTIVE_JOB_STORAGE_KEY = 'aismith-active-job-id'

export type RememberedJob = { jobId: string; runId: string | null }

export function serializeRememberedJob(jobId: string, runId: string | null): string {
  return JSON.stringify({ jobId, runId })
}

export function readRememberedJob(saved: string | null): RememberedJob | null {
  if (!saved) return null
  try {
    const parsed = JSON.parse(saved) as Partial<RememberedJob> | null
    if (!parsed || typeof parsed.jobId !== 'string' || !parsed.jobId) return null
    return { jobId: parsed.jobId, runId: typeof parsed.runId === 'string' ? parsed.runId : null }
  } catch {
    // A hand-edited or pre-format value is simply "nothing remembered" --
    // never an exception on mount.
    return null
  }
}

// Which job id the next resync should ask the server about: whatever this tab
// still holds in memory always wins (it is never staler than storage), and
// the persisted id is the fallback for a tab that just remounted.
export function activeJobResyncId(
  current: { id: string } | null | undefined,
  remembered: RememberedJob | null,
): string | null {
  return current?.id || remembered?.jobId || null
}

export type ActiveJobResync = 'restore' | 'reconcile' | 'ignore'

// What to do with the job the server hands back on a resync.
//  - 'ignore'    : nothing to adopt, this tab has since moved on to a
//                  different job, the job belongs to another character
//                  journey, or its terminal state is already on screen.
//                  Never clobber newer local state.
//  - 'restore'   : still queued/running -- adopt it so the progress panel
//                  comes back for a job that never stopped running.
//  - 'reconcile' : it finished while the tab was away -- adopt it so the
//                  existing terminal-hydration effect surfaces the output
//                  artifact and its gates exactly as if the tab had stayed
//                  open.
export function activeJobResyncDecision(
  fetched: { id: string; status: string; run_id?: string | null } | null | undefined,
  current: { id: string; status: string } | null | undefined,
  activeRunId: string | null = null,
): ActiveJobResync {
  if (!fetched) return 'ignore'
  if (current && current.id !== fetched.id) return 'ignore'
  if (fetched.run_id && activeRunId && fetched.run_id !== activeRunId) return 'ignore'
  if (isActiveJobStatus(fetched.status)) return 'restore'
  if (current && current.status === fetched.status) return 'ignore'
  return 'reconcile'
}

// A visibilitychange only resyncs when the tab actually came back; a window
// `focus` (and the initial mount) always does. The two are deliberately not
// both gated on document.visibilityState: this app is routinely driven in
// remote/automated Chrome sessions where visibilityState reads 'hidden' even
// for the focused window (the same observation StudioViewport.tsx's rAF
// postmortem records), and gating focus on it would make the whole recovery
// path dead code exactly there.
export function shouldResyncActiveJob(
  trigger: 'mount' | 'focus' | 'visibilitychange',
  visibilityState: DocumentVisibilityState,
): boolean {
  if (trigger === 'visibilitychange') return visibilityState === 'visible'
  return true
}

export const VISIBLE_JOB_POLL_INTERVAL_MS = 500
export const HIDDEN_JOB_POLL_INTERVAL_MS = 2_000

// Keep polling while the tab is hidden, just slower. A backgrounded tab has
// its timers clamped (1s, and once per minute once Chrome's intensive
// throttling kicks in), so a 500ms cadence buys nothing there while still
// costing a request per tick. What must never happen is the loop stopping:
// a running job's UI state would be orphaned with nothing to resume it.
export function jobPollIntervalMs(visibilityState: DocumentVisibilityState): number {
  return visibilityState === 'visible' ? VISIBLE_JOB_POLL_INTERVAL_MS : HIDDEN_JOB_POLL_INTERVAL_MS
}
