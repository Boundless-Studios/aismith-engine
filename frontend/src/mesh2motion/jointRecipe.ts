export type JointTransform = {
  name: string
  position: [number, number, number]
  quaternion: [number, number, number, number]
}

export type JointRecipe = {
  schema_version: 1
  clean_mesh_artifact_id: string
  skeleton_template: string
  bone_naming: string
  mirror_enabled: boolean
  transforms: JointTransform[]
  editor_version: string
}

export function normalizeJointRecipe(recipe: JointRecipe): JointRecipe {
  const names = recipe.transforms.map((transform) => transform.name)
  if (new Set(names).size !== names.length) throw new Error('Joint transform names must be unique')
  const finite = recipe.transforms.every((transform) => (
    [...transform.position, ...transform.quaternion].every(Number.isFinite)
  ))
  if (!finite) throw new Error('Joint transforms must contain finite values')
  return { ...recipe, transforms: recipe.transforms.map((transform) => ({ ...transform })) }
}
