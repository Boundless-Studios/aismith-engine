export type ReviewState = 'unreviewed' | 'accepted' | 'rejected'

export type QualityFinding = {
  code: string
  severity: 'warning' | 'error'
  message: string
  metric: string
  actual: unknown
  expected: string
}

export type QualityDecision = {
  policy_version: number
  // Stage-gate decisions name the stage they gate; the `source_fidelity` and
  // `project_views` decisions (backend source_fidelity_policy /
  // projection_policy) carry no target stage and are labelled by their key.
  target_stage?: string
  status: 'ready' | 'green' | 'needs_review' | 'blocked'
  findings: QualityFinding[]
  recommended_actions: string[]
}

// The heading a quality decision renders under: its target stage when it
// gates one, otherwise the `quality_decisions` key it was stamped under.
// Never dereferences an optional field -- a decision without `target_stage`
// must not unmount the History panel.
export function qualityDecisionLabel(key: string, decision: Partial<QualityDecision> | null | undefined): string {
  const raw = typeof decision?.target_stage === 'string' && decision.target_stage ? decision.target_stage : key
  return raw.replace(/[-_]/g, ' ')
}

export function qualityDecisionPassed(status: unknown): boolean {
  return status === 'ready' || status === 'green'
}

export type Artifact = {
  id: string
  name: string
  kind: string
  size_bytes: number
  created_at: number
  metadata: {
    quality_decisions?: Record<string, QualityDecision>
    [key: string]: unknown
  }
}

export type DeformationReviewArtifact = {
  artifact_id: string
  label: string
  frame_ratio: number
  rest_pose: boolean
  focus_bone?: string | null
}

export function deformationReviewArtifacts(artifact?: Artifact): DeformationReviewArtifact[] {
  const review = artifact?.metadata.deformation_review
  if (!review || typeof review !== 'object') return []
  const references = (review as { review_artifacts?: unknown }).review_artifacts
  if (!Array.isArray(references)) return []
  return references.filter((item): item is DeformationReviewArtifact => {
    if (!item || typeof item !== 'object') return false
    const candidate = item as Record<string, unknown>
    return typeof candidate.artifact_id === 'string'
      && typeof candidate.label === 'string'
      && typeof candidate.frame_ratio === 'number'
      && typeof candidate.rest_pose === 'boolean'
      && (candidate.focus_bone == null || typeof candidate.focus_bone === 'string')
  })
}

export type SkeletonReviewArtifact = {
  artifact_id: string
  label: string
}

export function skeletonReviewArtifacts(artifact?: Artifact): SkeletonReviewArtifact[] {
  const review = artifact?.metadata.skeleton_review
  if (!review || typeof review !== 'object') return []
  const references = (review as { review_artifacts?: unknown }).review_artifacts
  if (!Array.isArray(references)) return []
  return references.filter((item): item is SkeletonReviewArtifact => {
    if (!item || typeof item !== 'object') return false
    const candidate = item as Record<string, unknown>
    return typeof candidate.artifact_id === 'string' && typeof candidate.label === 'string'
  })
}

export type BilateralSymmetrySummary = {
  p95: number
  band: number
  pairsOutsideBand: number
  worstPair?: string
}

export function bilateralSymmetrySummary(artifact?: Artifact): BilateralSymmetrySummary | null {
  const rigQuality = artifact?.metadata.rig_quality
  if (!rigQuality || typeof rigQuality !== 'object') return null
  const rigQualityRecord = rigQuality as Record<string, unknown>
  const p95 = rigQualityRecord.symmetry_mirror_error_p95
  if (typeof p95 !== 'number') return null
  const symmetry = rigQualityRecord.bilateral_symmetry
  const symmetryRecord = symmetry && typeof symmetry === 'object' ? symmetry as Record<string, unknown> : {}
  const band = typeof symmetryRecord.band === 'number' ? symmetryRecord.band : 0
  const pairsOutsideBand = typeof rigQualityRecord.symmetry_pairs_outside_band_count === 'number'
    ? rigQualityRecord.symmetry_pairs_outside_band_count
    : 0
  const worstOffenders = Array.isArray(symmetryRecord.worst_offenders) ? symmetryRecord.worst_offenders : []
  const worst = worstOffenders[0] as Record<string, unknown> | undefined
  const worstPair = worst && typeof worst.left === 'string' && typeof worst.right === 'string'
    ? `${worst.left}/${worst.right}`
    : undefined
  return { p95, band, pairsOutsideBand, worstPair }
}

export type StageAttempt = {
  id: string
  stage: string
  status: string
  input_artifact_id: string
  output_artifact_id?: string | null
  settings: Record<string, unknown>
  logs: string[]
  warnings: string[]
  error?: string | null
  promoted?: boolean
  created_at: number
  completed_at?: number | null
}

export type RunManifest = {
  id: string
  name: string
  status: string
  source_artifact_id: string
  source_artifact_kind: string
  current_artifact_id?: string | null
  review_status: ReviewState
  review_note: string
  reviewed_artifact_id?: string | null
  checkpoints?: Record<string, { note: string; completed_at: number; artifact_id: string }>
  attempts: StageAttempt[]
  created_at: number
  updated_at: number
  // H: subject-agnostic generation. Optional so an older run (or a backend
  // build that hasn't landed the field yet) defaults to "auto" / unrestricted.
  subject_kind?: 'auto' | 'object' | 'creature' | 'biped'
  applicable_stages?: string[]
}

export type StudioTab = 'generate' | 'retopologize' | 'rig' | 'paint'

export function artifactForHistoryUse(
  artifacts: Record<string, Artifact>,
  artifactId: string,
): Artifact | null {
  return artifacts[artifactId] || null
}

export async function waitForRunTerminal<T extends { attempts: Array<{ status: string }> }>(
  load: () => Promise<T>,
  wait: () => Promise<void>,
): Promise<T> {
  while (true) {
    const run = await load()
    if (!run.attempts.some((attempt) => ['queued', 'running'].includes(attempt.status))) return run
    await wait()
  }
}

export function isImageArtifactKind(kind: string): boolean {
  return ['unknown', 'image', 'png', 'jpg', 'jpeg', 'webp'].includes(kind)
}

export function studioTabForSourceKind(kind: string): StudioTab {
  if (isImageArtifactKind(kind)) return 'generate'
  return kind === 'glb' ? 'rig' : 'retopologize'
}

export function studioTabForStage(stage: string): StudioTab {
  if (stage === 'prepare-image') return 'generate'
  if (stage === 'create-tpose-set') return 'generate'
  if (stage === 'generate') return 'retopologize'
  if (stage === 'canonicalize') return 'retopologize'
  if (stage === 'retopologize') return 'retopologize'
  if (stage === 'retopo') return 'retopologize'
  if (stage === 'paint-bake') return 'paint'
  if (stage === 'project-views') return 'retopologize'
  if (stage === 'rebuild-views') return 'retopologize'
  return 'rig'
}

export function shouldRestoreRefineImage(stage: string, sourceKind: string, isSource: boolean): boolean {
  return !isSource && stage === 'retopologize' && isImageArtifactKind(sourceKind)
}

export function formatSetting(value: unknown): string {
  if (value !== null && typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function stageForQualityAction(action: string): string {
  if (action === 'refine-reconstruct') return 'retopologize'
  if (['retopologize', 'repair-topology', 'inspect-boundaries'].includes(action)) return 'retopo'
  if (['inspect-weight-paint', 'repair-weights', 'fit-bind-pose', 'rerig', 'attach-animation', 'reattach-animation'].includes(action)) return 'rig-edit'
  return 'retopologize'
}

const automatedQualityActions = new Set(['retry-generation', 'refine-reconstruct', 'repair-topology', 'retopologize', 'repair-weights'])

export function isAutomatedQualityAction(action: string): boolean {
  return automatedQualityActions.has(action)
}

export function qualityActionsForDisplay(actions: string[], compact: boolean): string[] {
  return compact ? actions.filter(isAutomatedQualityAction) : actions
}

export function isRigPlaybackStage(stage?: string): boolean {
  return stage === 'rig' || stage === 'auto-rig' || stage === 'rig-adjust' || stage === 'repair-weights' || stage === 'rig-transfer' || stage === 'target-proxy-rig' || stage === 'retarget-motion' || stage === 'generate-motion' || stage === 'apply-clip'
}

export function isRetainedCandidate(
  attempt: Pick<StageAttempt, 'status' | 'output_artifact_id' | 'promoted'>,
  currentArtifactId?: string | null,
  artifactKind = 'glb',
): boolean {
  return (artifactKind === 'glb' || isImageArtifactKind(artifactKind))
    && attempt.status === 'completed'
    && Boolean(attempt.output_artifact_id)
    && attempt.output_artifact_id !== currentArtifactId
    && attempt.promoted === false
}

export function canPreviewArtifact(kind?: string): boolean {
  return kind === 'glb' || (kind !== undefined && isImageArtifactKind(kind))
}

export function canBakeMotionChild(
  motion?: { kind?: string },
  current?: { kind?: string; metadata?: { quality_decisions?: Record<string, { status?: string; policy_version?: number }> } },
  blenderAvailable = false,
): boolean {
  const decisions = current?.metadata?.quality_decisions
  const certificate = decisions?.['rig-certificate']
  const required = ['rig', 'rig-certificate', 'animation-review', 'deformation-review']
  return blenderAvailable
    && motion?.kind === 'bvh'
    && current?.kind === 'glb'
    && required.every((name) => decisions?.[name]?.status === 'ready')
    && certificate?.status === 'ready'
    && (certificate.policy_version ?? 0) >= 2
}

export function hasActiveMotionBake(
  attempts: Array<Pick<StageAttempt, 'stage' | 'status'>>,
): boolean {
  return attempts.some((attempt) => attempt.stage === 'retarget-motion' && ['queued', 'running'].includes(attempt.status))
}

export function isRunPipelineLocked(
  attempts: Array<Pick<StageAttempt, 'stage' | 'status'>>,
  busyAction: string,
): boolean {
  return Boolean(busyAction) || attempts.some((attempt) => ['queued', 'running'].includes(attempt.status))
}

export function currentArtifactStage(run: {
  current_artifact_id?: string | null
  attempts: Array<Pick<StageAttempt, 'stage' | 'output_artifact_id'>>
}): string {
  const inspectionStages = new Set([
    'promote-artifact',
    'mesh-diagnostics',
    'rig-diagnostics',
    'deformation-review',
  ])
  for (let index = run.attempts.length - 1; index >= 0; index -= 1) {
    const attempt = run.attempts[index]
    if (
      attempt?.output_artifact_id === run.current_artifact_id
      && !inspectionStages.has(attempt.stage)
    ) return attempt.stage
  }
  return 'source'
}
