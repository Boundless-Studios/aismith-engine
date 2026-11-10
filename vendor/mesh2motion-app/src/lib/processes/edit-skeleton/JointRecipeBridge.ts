export type JointRecipeTransform = {
  name: string
  position: [number, number, number]
  quaternion: [number, number, number, number]
}

export type JointRecipePayload = {
  schema_version: 1
  clean_mesh_artifact_id: string
  skeleton_template: string
  bone_naming: string
  mirror_enabled: boolean
  transforms: JointRecipeTransform[]
  editor_version: string
}

export type JointRecipeContext = {
  run: string
  cleanMeshArtifactId: string
  session: string
}

export function jointRecipeContextFromLocation (search: string): JointRecipeContext | null {
  const params = new URLSearchParams(search)
  const run = params.get('run')?.trim() ?? ''
  const cleanMeshArtifactId = params.get('input')?.trim() ?? ''
  const session = params.get('session')?.trim() ?? ''
  return run && cleanMeshArtifactId && session ? { run, cleanMeshArtifactId, session } : null
}

export function acceptsJointRecipeMessage (
  data: unknown,
  context: JointRecipeContext
): data is { run: string, cleanMeshArtifactId: string, session: string } {
  if (data === null || typeof data !== 'object') return false
  const candidate = data as Record<string, unknown>
  return candidate.run === context.run &&
    candidate.cleanMeshArtifactId === context.cleanMeshArtifactId &&
    candidate.session === context.session
}

export function isJointRecipeCaptureMessage (
  data: unknown,
  context: JointRecipeContext
): boolean {
  if (!acceptsJointRecipeMessage(data, context)) return false
  const candidate = data as Record<string, unknown>
  return candidate.source === 'aismith-studio' && candidate.type === 'joint-recipe-capture'
}
