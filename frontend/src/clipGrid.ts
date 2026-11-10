// Pure helpers backing ClipBrowser.tsx's visual grid picker -- kept out of
// the React component so batching/caching behavior is unit-testable without
// a WebGL context (same split as retargetPreview.ts's pure helpers vs.
// StudioViewport.tsx's RetargetPreview component).

// How many clips' thumbnails get rendered per idle-callback tick. Small
// enough that an 87-clip donor library doesn't jank the UI thread with one
// synchronous render burst, per the operator's grid-picker request.
export const THUMBNAIL_BATCH_SIZE = 8

// Fraction of a clip's own duration used as its thumbnail pose -- lands past
// a clip's rest/wind-up frames without hand-picking a keyframe per clip
// (mesh2motion parity: one consistent rule across all 87 curated clips).
export const THUMBNAIL_POSE_FRACTION = 0.4

// Fixed square render resolution used for every cached thumbnail, so cells
// stay visually consistent regardless of the live grid's current layout.
export const THUMBNAIL_RENDER_SIZE = 160

// Splits `ids` into `batchSize`-sized chunks, preserving order, so the
// caller can yield to the event loop (rAF/idle callback) between chunks
// instead of rendering every clip in one synchronous burst.
export function batchClipIds(ids: readonly string[], batchSize: number = THUMBNAIL_BATCH_SIZE): string[][] {
  if (batchSize <= 0) return ids.length ? [ids.slice()] : []
  const batches: string[][] = []
  for (let index = 0; index < ids.length; index += batchSize) batches.push(ids.slice(index, index + batchSize))
  return batches
}

// The ids from `clipIds` not already present in `cache` -- the clips a
// fresh generation pass still needs to render, in `clipIds`'s own order so
// batching stays deterministic and reopening the panel only regenerates
// what's actually missing (module-scope cache survives ClipBrowser remounts).
export function pendingClipIds(clipIds: readonly string[], cache: ReadonlyMap<string, string>): string[] {
  return clipIds.filter((id) => !cache.has(id))
}

// The pose time (seconds) to freeze a clip's mixer at for its thumbnail.
// Guards against a missing/zero/garbage duration (e.g. an action whose clip
// failed to resolve) by holding at time 0 rather than producing NaN, which
// would silently no-op AnimationMixer's time-setting.
export function poseTimeForClip(durationSeconds: number, fraction: number = THUMBNAIL_POSE_FRACTION): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return 0
  return durationSeconds * fraction
}
