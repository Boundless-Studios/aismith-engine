import { describe, expect, it } from 'vitest'
import { normalizeJointRecipe, type JointRecipe } from './jointRecipe'

const recipe = (): JointRecipe => ({
  schema_version: 1,
  clean_mesh_artifact_id: 'mesh-1',
  skeleton_template: 'Human',
  bone_naming: 'preserve',
  mirror_enabled: true,
  transforms: [{ name: 'pelvis', position: [0, 1, 0], quaternion: [0, 0, 0, 1] }],
  editor_version: 'mesh2motion-v1',
})

describe('normalizeJointRecipe', () => {
  it('accepts a finite recipe', () => expect(normalizeJointRecipe(recipe())).toEqual(recipe()))
  it('rejects duplicate names', () => {
    const value = recipe(); value.transforms.push({ ...value.transforms[0] })
    expect(() => normalizeJointRecipe(value)).toThrow(/unique/)
  })
  it('rejects non-finite transforms', () => {
    const value = recipe(); value.transforms[0].position[0] = Number.NaN
    expect(() => normalizeJointRecipe(value)).toThrow(/finite/)
  })
})
