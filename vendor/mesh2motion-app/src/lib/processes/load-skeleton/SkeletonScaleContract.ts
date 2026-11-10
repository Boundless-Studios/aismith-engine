const MIN_SKELETON_SCALE = 0.1
const MAX_SKELETON_SCALE = 2

export function scale_to_match_height (model_height: number, skeleton_height: number): number {
  if (!Number.isFinite(model_height) || !Number.isFinite(skeleton_height) || model_height <= 0 || skeleton_height <= 0) {
    return 1
  }

  return Math.min(MAX_SKELETON_SCALE, Math.max(MIN_SKELETON_SCALE, model_height / skeleton_height))
}

export function vertical_span (positions: number[]): number {
  if (positions.length === 0) return 0
  return Math.max(...positions) - Math.min(...positions)
}
