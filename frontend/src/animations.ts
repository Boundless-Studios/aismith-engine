import { readableAnimationName } from './animation'

export type AnimationReviewState = 'accepted' | 'rejected' | 'unreviewed'

export type JourneyAnimation = {
  artifactId: string
  label: string
  stage: string
  // Epoch SECONDS, straight from the run store's attempt.completed_at /
  // created_at (the backend stamps these with Python's time.time()) — NOT
  // milliseconds. Multiply by 1000 before handing this to `new Date(...)`
  // (see HistoryView.tsx's `new Date(run.updated_at * 1000)` for the same
  // convention elsewhere in this app).
  createdAt: number
  promoted?: boolean
  reviewState: AnimationReviewState
  // The rig this clip was produced on — a raw artifact id (never a label:
  // that would repeat the "never use a file name" mistake). The caller
  // resolves it to a rig name via detailsById, falling back to a short id,
  // for the row's secondary "on <rig> · <clip id>" caption.
  inputArtifactId: string
}

type JourneyAnimationAttempt = {
  stage: string
  status: string
  input_artifact_id?: string
  output_artifact_id?: string | null
  settings?: Record<string, unknown>
  promoted?: boolean
  completed_at?: number | null
  created_at?: number
}

type JourneyAnimationRun = {
  review_status?: string
  reviewed_artifact_id?: string | null
  attempts: JourneyAnimationAttempt[]
} | null

export type CuratedMotionClip = { id: string; label: string; animation: string }

// The curated donor library (vendor/mesh2motion-app/static/animations/human-base-animations.glb)
// is pinned by sha256 on the backend. Only list a clip here once its exact Blender action
// name has been confirmed against that donor — an unmatched name silently falls back to the
// donor's longest action, which would attach the wrong motion without any error. Shared
// between App.tsx (the retarget-motion job's default clip) and AnimatePanel.tsx (the picker),
// and doubles as the picker's fallback list — rendered while GET /api/animations/curated is
// loading, or if it errors — so the select is never empty.
export const CURATED_MOTION_CLIPS: CuratedMotionClip[] = [
  { id: 'sword-attack', label: 'Sword Attack', animation: 'Sword_Attack_Armature' },
]

type CuratedAnimationsResponse = {
  donor: string
  clips: { animation: string; label: string; frames: number; duration_seconds: number }[]
}

// Blender's glTF importer always appends "_Armature" to a curated-library
// action name on import (backend/app/curated_motion.py's module docstring
// has the confirmation) — strip it and kebab-case the remainder for a
// stable picker id, e.g. "Sword_Attack_Armature" -> "sword-attack" (matching
// this file's own hardcoded fallback id for the same clip).
export function curatedClipId(exactActionName: string): string {
  return exactActionName.replace(/_Armature$/, '').replace(/_/g, '-').toLowerCase()
}

// The `settings.animation` App.tsx sends alongside a retarget-motion job:
// backend/workers/blender_worker.py's retarget-bvh op only uses it as the
// display name of the new `bpy.data.actions.new(animation)` datablock it
// bakes onto the retargeted output — it does not need to match the BVH's
// own source action name, since the motion itself is read from the BVH file
// regardless of what this is called. Strips the "_Armature" import suffix
// (same rule as curatedClipId, underscores kept) so the baked output's
// action is named after whichever clip was actually attached instead of
// always "Sword_Attack".
export function curatedActionLabel(exactActionName: string): string {
  return exactActionName.replace(/_Armature$/, '')
}

// The name of the AnimationClip inside the donor GLB's own `gltf.animations`
// list for a given curated clip's exact action name — ClipBrowser.tsx uses
// this to find which clip to play when previewing on the loaded donor
// character. The donor's raw glTF JSON animation names never carry the
// "_Armature" suffix (only Blender's post-import action names do, per
// backend/app/curated_motion.py's module docstring) — kept as its own named
// function from curatedActionLabel even though the string operation is
// identical, since the two are strings for different targets (a freshly
// baked retarget output's action name vs. a clip already inside this donor
// file).
export function curatedDonorActionName(exactActionName: string): string {
  return exactActionName.replace(/_Armature$/, '')
}

// Pure mapping from the backend payload to the picker's clip shape — kept
// separate from the fetch below so it is unit-testable without a network.
export function curatedMotionClipsFromResponse(payload: CuratedAnimationsResponse): CuratedMotionClip[] {
  return payload.clips.map((clip) => ({ id: curatedClipId(clip.animation), label: clip.label, animation: clip.animation }))
}

// Fetches the full curated clip list on mount (AnimatePanel.tsx). Throws on
// any network/HTTP failure — the caller is expected to keep showing
// CURATED_MOTION_CLIPS rather than an empty picker.
export async function fetchCuratedMotionClips(apiBase: string): Promise<CuratedMotionClip[]> {
  const response = await fetch(`${apiBase}/animations/curated`)
  if (!response.ok) throw new Error(`Could not load curated clips (${response.status})`)
  return curatedMotionClipsFromResponse(await response.json() as CuratedAnimationsResponse)
}

// The stages that can ever produce a playable animated clip on this run (I):
// retarget-motion (a curated clip retargeted onto a humanoid rig),
// generate-motion (a chain/serpent rig bakes its motion straight onto the
// GLB with no separate retarget step, so its own output is already a clip),
// and apply-clip (N5: the same curated clip direct-attached by exact bone
// name onto a template-conformant rig, no retarget math).
const ANIMATE_ATTEMPT_STAGES = ['retarget-motion', 'generate-motion', 'apply-clip']

// generate-motion is two different producers wearing one stage id
// (backend/app/main.py's `is_slither` check): a chain/serpent rig bakes a
// procedural "Slither" cycle straight onto a playable GLB with Blender; any
// other rig instead asks NVIDIA Kimodo for a text-prompted motion clip,
// whose *raw* output is a `.bvh` file -- not a standalone playable asset (it
// only becomes one once a later retarget-motion attempt embeds it onto the
// rig, which shows up here as its own row). journeyAnimations filters those
// non-chain generate-motion attempts out of the visible list entirely below
// (7zdn5b) -- but the label for one is still computed (and independently
// tested) before that filter runs, so it never regresses back to claiming a
// Kimodo prompt clip is "Slither" (7zdn5a) even for the few callers that
// might one day need the label without the visibility filter applied.
export function generateMotionFallbackLabel(isChainRig: boolean, promptSetting: unknown): string {
  if (isChainRig) return 'Slither'
  const prompt = typeof promptSetting === 'string' ? promptSetting.trim() : ''
  return prompt || 'Generated Motion'
}

function fallbackLabel(stage: string, isChainRig: boolean, promptSetting: unknown): string {
  if (stage === 'generate-motion') return generateMotionFallbackLabel(isChainRig, promptSetting)
  return readableAnimationName(stage.replace(/-/g, '_'))
}

// Newest-first list of every embedded animated clip this run has produced,
// for the Animate tab's "Animations" picker (no separate Review tab: this
// is also where accept/reject state surfaces per clip). Pure and
// unit-testable — no network, no React. `isChainRig` mirrors the same
// chain-vs-humanoid rig-certificate check AnimatePanel.tsx already derives
// for this run (App.tsx's `isChainRig`) -- a run's subject/rig type doesn't
// change across its own attempts, so it applies uniformly here too.
export function journeyAnimations(run: JourneyAnimationRun, isChainRig: boolean = false): JourneyAnimation[] {
  if (!run) return []
  return run.attempts
    .filter((attempt): attempt is JourneyAnimationAttempt & { output_artifact_id: string } => (
      attempt.status === 'completed'
      && Boolean(attempt.output_artifact_id)
      && ANIMATE_ATTEMPT_STAGES.includes(attempt.stage)
      // A non-chain rig's generate-motion attempt is always Kimodo's raw
      // (unplayable) BVH -- see fallbackLabel's comment above.
      && (attempt.stage !== 'generate-motion' || isChainRig)
    ))
    .map((attempt) => {
      // apply-clip (N5) names its clip via settings.clip (the exact family
      // clip_name, e.g. "Sword_Attack_Armature" -- backend/app/main.py's
      // apply-clip validation) rather than settings.animation, since it has
      // no separate "action label" the way a retarget-motion bake does.
      const animationSetting = typeof attempt.settings?.animation === 'string'
        ? attempt.settings.animation
        : typeof attempt.settings?.clip === 'string' ? attempt.settings.clip : undefined
      // settings.animation/settings.clip always wins when present; otherwise
      // the per-stage fallback — never the artifact's own file name, which is
      // typically a generic upload/export name like "rigged-animated-model
      // .glb" or "Generate Motion Result" and tells the operator nothing.
      const label = animationSetting ? readableAnimationName(animationSetting) : fallbackLabel(attempt.stage, isChainRig, attempt.settings?.prompt)
      const reviewState: AnimationReviewState = run.reviewed_artifact_id === attempt.output_artifact_id
        ? (run.review_status === 'accepted' ? 'accepted' : run.review_status === 'rejected' ? 'rejected' : 'unreviewed')
        : 'unreviewed'
      return {
        artifactId: attempt.output_artifact_id,
        label,
        stage: attempt.stage,
        createdAt: attempt.completed_at ?? attempt.created_at ?? 0,
        promoted: attempt.promoted,
        reviewState,
        inputArtifactId: attempt.input_artifact_id ?? '',
      }
    })
    .sort((a, b) => b.createdAt - a.createdAt)
}
