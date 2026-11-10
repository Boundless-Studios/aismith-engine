import { describe, expect, it } from 'vitest'

import {
  acceptsJointRecipeMessage,
  jointRecipeContextFromLocation,
} from '../../../vendor/mesh2motion-app/src/lib/processes/edit-skeleton/JointRecipeBridge'
import * as JointRecipeBridge from '../../../vendor/mesh2motion-app/src/lib/processes/edit-skeleton/JointRecipeBridge'

describe('JointRecipeBridge', () => {
  const context = { run: 'run-1', cleanMeshArtifactId: 'mesh-1', session: 'session-1' }

  it('requires a complete editor context', () => {
    expect(jointRecipeContextFromLocation('?run=run-1&input=mesh-1&session=session-1')).toEqual(context)
    expect(jointRecipeContextFromLocation('?run=run-1&input=mesh-1')).toBeNull()
  })

  it('rejects stale runs, meshes, and sessions', () => {
    expect(acceptsJointRecipeMessage(context, context)).toBe(true)
    expect(acceptsJointRecipeMessage({ ...context, run: 'old-run' }, context)).toBe(false)
    expect(acceptsJointRecipeMessage({ ...context, cleanMeshArtifactId: 'old-mesh' }, context)).toBe(false)
    expect(acceptsJointRecipeMessage({ ...context, session: 'old-session' }, context)).toBe(false)
  })

  it('accepts a baseline capture request only for the active editor', () => {
    const capture = (JointRecipeBridge as unknown as {
      isJointRecipeCaptureMessage?: (data: unknown, activeContext: typeof context) => boolean
    }).isJointRecipeCaptureMessage
    expect(capture).toBeTypeOf('function')
    expect(capture?.({ source: 'aismith-studio', type: 'joint-recipe-capture', ...context }, context)).toBe(true)
    expect(capture?.({ source: 'aismith-studio', type: 'joint-recipe-capture', ...context, session: 'stale' }, context)).toBe(false)
    expect(capture?.({ source: 'other', type: 'joint-recipe-capture', ...context }, context)).toBe(false)
  })
})
