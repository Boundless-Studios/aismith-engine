import type { ReviewState, StudioTab } from './history'

export type JourneyState = 'complete' | 'active' | 'ready' | 'checkpoint' | 'failed' | 'upcoming' | 'skipped' | 'not-applicable'
export type JourneyOwner = 'agent' | 'you' | 'shared'

export type JourneyStep = {
  id: string
  label: string
  owner: JourneyOwner
  state: JourneyState
  agentAction: string
  humanAction: string
  completion: string
  // Present only when state is 'upcoming' -- why this step isn't reachable
  // yet, so the journey strip can say so instead of just disabling the
  // button with no explanation (the operator's ask: a legitimate gate must
  // say why rather than dead-click).
  blockedReason?: string
}

type JourneyRun = {
  source_artifact_kind?: string
  review_status: ReviewState
  reviewed_artifact_id?: string | null
  current_artifact_id?: string | null
  checkpoints?: Record<string, { note: string; completed_at: number; artifact_id: string }>
  attempts: Array<{ stage: string; status: string; output_artifact_id?: string | null }>
  // Backend-computed set of journey step ids this run's subject_kind actually
  // uses (H) — e.g. an "object" run excludes skeleton/animate. Missing
  // entirely (older runs, or the field not landed yet) means "everything applies".
  applicable_stages?: string[]
}

// Clean is a recommendation, never a gate: a generated mesh can go straight
// to Skeleton. It still shows up in the strip as its own step — "ready" while
// reachable, "skipped" once a later stage completed without it ever running.
const OPTIONAL_STAGE_IDS = new Set(['clean'])

type StepDefinition = Omit<JourneyStep, 'state'> & { stages: string[]; tab: StudioTab }

const definitions: StepDefinition[] = [
  {
    // `create-tpose-set` (bead 4vquq) is this step done as one composite job:
    // the T-pose front plus the three views fanned off it. Both stages belong
    // to Prepare, so a run built either way resolves the same journey step.
    id: 'prepare', label: 'Prepare', owner: 'shared', stages: ['prepare-image', 'create-tpose-set'], tab: 'generate',
    agentAction: 'Create the canonical T-pose set: a gated T-pose front plus matching back, left, and right views.',
    humanAction: 'Confirm the full body is visible, limbs are separated, and identity is correct.',
    completion: 'Approved rig-friendly references are retained with settings and seed.',
  },
  {
    id: 'generate', label: 'Generate', owner: 'agent', stages: ['generate'], tab: 'generate',
    agentAction: 'Generate textured geometry locally and retain the native output.',
    humanAction: 'Only intervene when silhouette or hidden-body inference is visibly wrong.',
    completion: 'A textured mesh passes basic geometry diagnostics.',
  },
  {
    id: 'clean', label: 'Clean', owner: 'agent', stages: ['canonicalize', 'retopologize', 'retopo', 'repair-topology', 'paint-bake', 'project-views', 'rebuild-views'], tab: 'retopologize',
    agentAction: 'Repair holes, normalize scale and materials, and produce deformation-friendly topology.',
    humanAction: 'Inspect face, hands, hair, clothing, and disconnected parts when flagged.',
    completion: 'Topology and material gates pass before any skeleton is added.',
  },
  {
    id: 'skeleton', label: 'Skeleton', owner: 'shared', stages: ['rig', 'auto-rig', 'rig-adjust', 'rig-transfer', 'target-proxy-rig', 'repair-weights'], tab: 'rig',
    agentAction: 'Predict and certify the humanoid skeleton with UniRig; hand-adjust joints in the viewport and rebind when needed.',
    humanAction: 'Inspect the applied skeleton, hand-adjust joints if needed, then approve the exact rig. Mesh2Motion template rigging remains available as the manual fallback.',
    completion: 'The current rig artifact passes validation and has your durable approval.',
  },
  {
    // Animate now also carries the accept/reject checkpoint that used to be
    // its own "Review" step (the operator's ask: no separate Review tab,
    // accept/save/export happens inline in Animate) — see the completion
    // override below, which requires run.reviewed_artifact_id to name one of
    // this stage's own completed outputs before painting the step complete.
    id: 'animate', label: 'Animate', owner: 'agent', stages: ['generate-motion', 'retarget-motion', 'apply-clip'], tab: 'rig',
    agentAction: 'Attach a genuine Sword Attack, retarget it, and repair contacts and joint limits.',
    humanAction: 'Choose and accept a clip; only pick a different motion if the attack intent is wrong.',
    completion: 'The accepted clip has real duration and passes motion diagnostics.',
  },
]

// A stage counts as satisfied for both the "complete" painting and the
// readiness check the same way: it has its own completed attempt, OR a
// downstream stage already completed past it (furthestCompletedIndex — the
// index-position shortcut a direct-generate run relies on for Prepare, which
// never gets its own attempt), OR it's a pre-rig stage on a rig-only run.
// The two checks used to diverge (readiness only looked at the stage's own
// attempts), which left an earlier stage painted "complete" via the shortcut
// while still counting as unmet for readiness — this keeps them in lockstep.
function stageSatisfied(
  definition: StepDefinition,
  index: number,
  attempts: JourneyRun['attempts'],
  furthestCompletedIndex: number,
  rigOnlyRun: boolean,
  preRigIds: Set<string>,
): boolean {
  if (rigOnlyRun && preRigIds.has(definition.id)) return true
  if (index < furthestCompletedIndex) return true
  return attempts.some((attempt) => definition.stages.includes(attempt.stage) && attempt.status === 'completed')
}

export function characterJourney(run: JourneyRun | null, activeTab: StudioTab): JourneyStep[] {
  const attempts = run?.attempts || []
  const furthestCompletedIndex = attempts.reduce((furthest, attempt) => {
    if (attempt.status !== 'completed') return furthest
    return Math.max(furthest, definitions.findIndex((definition) => definition.stages.includes(attempt.stage)))
  }, -1)
  const activeAttempt = [...attempts].reverse().find((attempt) => ['queued', 'running'].includes(attempt.status))
  const activeIndex = activeAttempt
    ? definitions.findIndex((definition) => definition.stages.includes(activeAttempt.stage))
    : -1
    // a run whose source is an imported GLB never ran the image pipeline;
  // those stages are satisfied by the import rather than permanently blocking
  const rigOnlyRun = run?.source_artifact_kind === 'glb'
  const preRigIds = new Set(['prepare', 'generate', 'clean'])
  // the durable skeleton checkpoint is judged against the run current rig
  // artifact (promotion can move it past older attempts)
  const currentRigAttempt = [...attempts].reverse().find((attempt) => (
    attempt.status === 'completed'
    && attempt.output_artifact_id
    && attempt.output_artifact_id === run?.current_artifact_id
    && definitions[3].stages.includes(attempt.stage)
  ))
  const latestRigArtifactId = currentRigAttempt?.output_artifact_id
    ?? [...attempts].reverse().find((attempt) => definitions[3].stages.includes(attempt.stage) && attempt.status === 'completed')?.output_artifact_id
  let blocked = false
  // Tracks the label of whichever step's own state (failed / checkpoint)
  // is cascading the "upcoming, and here's why" state to everything after
  // it, so the fallback reason below can name it instead of staying silent.
  let blockingStepLabel: string | null = null
  const isApplicable = (id: string) => !run?.applicable_stages || run.applicable_stages.includes(id)

  return definitions.map((definition, index) => {
    if (!isApplicable(definition.id)) return { ...definition, state: 'not-applicable' }
    const matches = attempts.filter((attempt) => definition.stages.includes(attempt.stage))
    const latest = matches[matches.length - 1]
    const isOptional = OPTIONAL_STAGE_IDS.has(definition.id)
    // Optional stages never get the "a downstream stage already passed me"
    // shortcut for their OWN completeness — that shortcut is exactly the
    // "ran without me" case that should render as skipped, not complete.
    let complete = isOptional
      ? matches.some((attempt) => attempt.status === 'completed') || (rigOnlyRun && preRigIds.has(definition.id))
      : stageSatisfied(definition, index, attempts, furthestCompletedIndex, rigOnlyRun, preRigIds)
    const skipped = isOptional && !complete && !blocked && furthestCompletedIndex > index
    // A failed attempt on an optional stage stops mattering once something
    // downstream has already succeeded without it — that reads as skipped,
    // not failed, even though the stage's own last attempt didn't land.
    const failed = !skipped && latest?.status === 'failed' && !complete
    // Animate absorbed the former Review step's accept/reject checkpoint: an
    // animated output existing at all is only ever a checkpoint (not
    // complete) until run.reviewed_artifact_id names one of THIS stage's own
    // completed outputs — a stale or foreign reviewed_artifact_id (e.g. left
    // over from a prior run) must not paint the step complete.
    const animatedOutput = definition.id === 'animate' && matches.some((attempt) => attempt.status === 'completed')
    if (definition.id === 'animate') {
      const acceptedArtifactIsAnimated = Boolean(run?.reviewed_artifact_id)
        && matches.some((attempt) => attempt.status === 'completed' && attempt.output_artifact_id === run?.reviewed_artifact_id)
      complete = acceptedArtifactIsAnimated && run?.review_status === 'accepted'
    }
    if (definition.id === 'skeleton') complete = Boolean(latestRigArtifactId && run?.checkpoints?.skeleton?.artifact_id === latestRigArtifactId)
    if (definition.id === 'skeleton' && !complete) {
      const rigComplete = attempts.some((attempt) => definitions[3].stages.includes(attempt.stage) && attempt.status === 'completed')
      if (rigComplete && !blocked) {
        blocked = true
        blockingStepLabel = definition.label
        return { ...definition, state: 'checkpoint' }
      }
    }
    if (failed && !blocked) {
      blocked = true
      blockingStepLabel = definition.label
      return { ...definition, state: 'failed' }
    }
    if (skipped) return { ...definition, state: 'skipped' }
    if (complete && !blocked) return { ...definition, state: 'complete' }
    if (animatedOutput && !blocked) return { ...definition, state: 'checkpoint' }
    if (index === activeIndex && !blocked) return { ...definition, state: 'active' }
    if (!blocked) {
      const blockingEarlier = definitions.slice(0, index).find((earlier, earlierIndex) => {
        // An optional or not-applicable earlier stage never blocks reaching
        // a later one — Skeleton is reachable as soon as Generate is done,
        // whether or not Clean ever ran.
        if (OPTIONAL_STAGE_IDS.has(earlier.id) || !isApplicable(earlier.id)) return false
        return !stageSatisfied(earlier, earlierIndex, attempts, furthestCompletedIndex, rigOnlyRun, preRigIds)
      })
      // Only decide active/ready/upcoming while nothing else is mid-flight —
      // an in-progress attempt elsewhere (activeIndex >= 0) leaves every other
      // step upcoming, matching the single "active" step it already owns above.
      if (activeIndex < 0) {
        const isOpenTab = (!run && definition.tab === activeTab && index === 0) || (!blockingEarlier && definition.tab === activeTab)
        if (isOpenTab) return { ...definition, state: definition.owner === 'you' ? 'checkpoint' : 'active' }
        // Reachable (every earlier stage is satisfied) but its tab isn't the
        // open one — without the removed tool tabs this is the only signal
        // that clicking it (or the Next button) will actually go somewhere,
        // so it must not read as a dead "upcoming" step.
        if (!blockingEarlier) return { ...definition, state: 'ready' }
      }
      const reason = activeIndex >= 0
        ? `${definitions[activeIndex].label} is still running.`
        : blockingEarlier ? `Complete ${blockingEarlier.label} first.` : undefined
      return { ...definition, state: 'upcoming', blockedReason: reason }
    }
    return {
      ...definition,
      state: 'upcoming',
      blockedReason: blockingStepLabel ? `${blockingStepLabel} needs your attention first.` : undefined,
    }
  })
}

// The step right after the furthest-completed one, for the "Next: <label>" affordance.
// Scans from the end so a revoked/regressed approval (e.g. a replaced rig) correctly
// points back at the step that needs attention again, rather than skipping ahead.
// Only offered when that immediate next step is actually reachable (ready, its own
// checkpoint, or already the active one) — never skips past a genuinely blocked
// (upcoming) step to some later one, and never re-offers a step already done.
export function nextJourneyStep(steps: JourneyStep[]): JourneyStep | null {
  const reachable: JourneyState[] = ['ready', 'checkpoint', 'active']
  const lastIndex = steps.length - 1
  for (let index = lastIndex; index >= 0; index -= 1) {
    if (steps[index].state !== 'complete') continue
    // The last step in the whole journey is a destination, not a waypoint —
    // once nothing remains but that step itself, there is nothing further to
    // advance "toward". The strip already shows it as reachable/current on
    // its own (clickable, non-upcoming); the "Next: <label>" button goes
    // away instead of pointing at the step you're already meant to be on.
    if (index + 1 >= lastIndex) return null
    const candidate = steps[index + 1]
    return reachable.includes(candidate.state) ? candidate : null
  }
  return null
}

export type JourneyDestination = StudioTab | 'history'

export type JourneyWorkspace = {
  destination: JourneyDestination
  artifactId?: string
  stage?: string
  isSource?: boolean
  mode?: 'rig' | 'animate'
  blocked?: boolean
}

export function journeyDestination(
  stepId: string,
  activeTab: StudioTab,
  attempts: Array<{ stage: string }>,
): JourneyDestination {
  if (stepId !== 'clean') return definitions.find((definition) => definition.id === stepId)?.tab || activeTab
  if (activeTab === 'retopologize' || activeTab === 'paint') return activeTab
  const latestCleanStage = [...attempts].reverse().find((attempt) => definitions[2].stages.includes(attempt.stage))?.stage
  if (latestCleanStage === 'paint-bake') return 'paint'
  return 'retopologize'
}

export type JourneyWorkspaceOptions = {
  // Set by the picker on the "Working on" card, or by History's Preview/
  // continue — a pinned artifact belonging to this run wins over whichever
  // artifact auto-resolution would otherwise pick, for the stages where more
  // than one compatible candidate can exist (I2).
  pinnedArtifactId?: string | null
  // Which motion-producing stages count as a playable "animated clip" for
  // Animate's default (pin-absent) resolution. retarget-motion always does —
  // a curated clip retargeted onto a humanoid rig. generate-motion does NOT
  // by default: on a humanoid rig its output is an intermediate BVH, not a
  // playable GLB. A chain (serpent) rig bakes motion directly onto the GLB
  // with no separate retarget step, so its generate-motion output IS the
  // playable clip — callers on a chain rig should pass
  // ['retarget-motion', 'generate-motion']. App.tsx derives this from the
  // active rig's rig-certificate profile (read from source.metadata), since
  // journeyWorkspace only sees the run's attempts, not artifact metadata.
  animatedOutputStages?: string[]
}

// Which prior stages' outputs a pin is allowed to satisfy, per journey step —
// Skeleton is handled specially below since a pin can target either the
// bare pre-rig mesh or the rig candidate itself. Animate accepts both a rig
// candidate (the "Working on" rig picker — which rig is being animated) and
// an animated clip (the Animations list — which clip is showing); both
// pickers share the same single pinnedArtifactId slot.
const PIN_COMPATIBLE_STAGES: Record<string, string[]> = {
  clean: definitions[1].stages,
  // apply-clip (N5 direct-attach) is a first-class animated output: without
  // it here, clicking one of its clips in the Animations list set the pin
  // but resolved back to the rig, so the "Selected clip" panel -- and with
  // it the only Export/Download affordance for an animation -- never
  // rendered.
  animate: [...definitions[3].stages, 'retarget-motion', 'generate-motion', 'apply-clip'],
}

export function journeyWorkspace(
  stepId: string,
  activeTab: StudioTab,
  run: {
    source_artifact_id: string
    current_artifact_id?: string | null
    checkpoints?: Record<string, { artifact_id: string }>
    attempts: Array<{ stage: string; status: string; completed_at?: number | null; created_at?: number; input_artifact_id?: string; output_artifact_id?: string | null }>
  } | null,
  options: JourneyWorkspaceOptions = {},
): JourneyWorkspace {
  const attempts = run?.attempts || []
  const destination = journeyDestination(stepId, activeTab, attempts)
  if (destination === 'history') return { destination }
  const normalizedStepId = stepId === 'rig' ? 'skeleton' : stepId
  const mode = normalizedStepId === 'animate' ? 'animate' : normalizedStepId === 'skeleton' ? 'rig' : undefined

  const definition = definitions.find((candidate) => candidate.id === normalizedStepId)
  const completedOutputFor = (stages: string[]) => {
    const eligible = attempts
      .map((candidate, index) => ({ candidate, index }))
      .filter(({ candidate }) => candidate.status === 'completed' && Boolean(candidate.output_artifact_id) && stages.includes(candidate.stage))
    if (!eligible.length) return undefined
    // newest by completion time; array position breaks ties (and fills in for
    // attempts without timestamps, preserving the historical reversed order)
    return eligible.reduce((latest, candidate) => {
      const latestTime = latest.candidate.completed_at ?? latest.candidate.created_at ?? -Infinity
      const candidateTime = candidate.candidate.completed_at ?? candidate.candidate.created_at ?? -Infinity
      if (candidateTime > latestTime) return candidate
      return candidateTime === latestTime && candidate.index > latest.index ? candidate : latest
    }).candidate
  }
  // approve always targets the run current rig artifact; fall back to the
  // newest completed rig when promotion hasnt happened yet
  const currentRigAttempt = [...attempts].reverse().find((candidate) => (
    candidate.status === 'completed'
    && candidate.output_artifact_id
    && candidate.output_artifact_id === run?.current_artifact_id
    && definitions[3].stages.includes(candidate.stage)
  ))
  const rigAttempt = currentRigAttempt || completedOutputFor(definitions[3].stages)

  const pinnedArtifactId = options.pinnedArtifactId
  if (pinnedArtifactId) {
    const pinnedAttempt = [...attempts].reverse().find((candidate) => (
      candidate.status === 'completed' && candidate.output_artifact_id === pinnedArtifactId
    ))
    const pinnedBelongsToRun = Boolean(pinnedAttempt) || pinnedArtifactId === run?.source_artifact_id
    if (pinnedBelongsToRun) {
      if (normalizedStepId === 'skeleton') {
        if (rigAttempt?.output_artifact_id) {
          // A rig candidate already exists: the pin only wins when it names
          // that exact rig — it must not silently swap in an unrelated mesh.
          if (pinnedArtifactId === rigAttempt.output_artifact_id) {
            return { destination, artifactId: pinnedArtifactId, stage: rigAttempt.stage, isSource: false, mode }
          }
        } else if (pinnedAttempt && [...definitions[1].stages, ...definitions[2].stages].includes(pinnedAttempt.stage)) {
          // No rig yet: pre-rig pin (a Generate or Clean mesh) wins outright.
          return { destination, artifactId: pinnedArtifactId, stage: pinnedAttempt.stage, isSource: false, mode }
        }
      } else {
        const compatibleStages = PIN_COMPATIBLE_STAGES[normalizedStepId]
        if (compatibleStages && pinnedAttempt && compatibleStages.includes(pinnedAttempt.stage)) {
          return { destination, artifactId: pinnedArtifactId, stage: pinnedAttempt.stage, isSource: false, mode }
        }
      }
    }
  }

  if (normalizedStepId === 'skeleton' && rigAttempt?.output_artifact_id) {
    return { destination, artifactId: rigAttempt.output_artifact_id, stage: rigAttempt.stage, isSource: false, mode }
  }
  const approvedRig = Boolean(rigAttempt?.output_artifact_id && run?.checkpoints?.skeleton?.artifact_id === rigAttempt.output_artifact_id)
  if (normalizedStepId === 'animate' && !approvedRig && rigAttempt?.output_artifact_id) {
    return { destination: 'rig', artifactId: rigAttempt.output_artifact_id, stage: rigAttempt.stage, isSource: false, mode: 'rig', blocked: true }
  }
  const animatedOutputStages = options.animatedOutputStages ?? ['retarget-motion', 'apply-clip']
  const playableStages = new Set([...definitions[3].stages, ...animatedOutputStages])
  const currentAttempt = [...attempts].reverse().find((candidate) => (
    candidate.status === 'completed'
    && candidate.output_artifact_id === run?.current_artifact_id
    && playableStages.has(candidate.stage)
  ))
  const attempt = normalizedStepId === 'animate'
    ? (run?.current_artifact_id
        ? currentAttempt || completedOutputFor(definitions[3].stages)
        : completedOutputFor(animatedOutputStages) || completedOutputFor(definitions[3].stages))
    // Skeleton with no rig candidate yet: auto-rig works from the cleaned mesh,
    // else the generated mesh — never from the run's source image.
    : normalizedStepId === 'skeleton' ? rigAttempt || completedOutputFor(definitions[2].stages) || completedOutputFor(definitions[1].stages)
    // Clean has nothing of its own to show yet (never run): work from the
    // Generate step's mesh output instead of falling all the way back to the
    // run's original source artifact (the uploaded image, for an image-sourced run).
    : normalizedStepId === 'clean' ? completedOutputFor(definition?.stages || []) || completedOutputFor(definitions[1].stages)
    : completedOutputFor(definition?.stages || [])
  if (attempt?.output_artifact_id) {
    return { destination, artifactId: attempt.output_artifact_id, stage: attempt.stage, isSource: false, mode }
  }
  if (run?.source_artifact_id) {
    return { destination, artifactId: run.source_artifact_id, stage: 'source', isSource: true, mode }
  }
  return { destination, mode }
}
