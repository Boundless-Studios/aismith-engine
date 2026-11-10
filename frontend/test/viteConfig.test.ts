import { describe, expect, it } from 'vitest'

import config from '../vite.config'

describe('Vite application routing', () => {
  it('does not fall back to the studio shell for missing Mesh2Motion pages', () => {
    expect(config.appType).toBe('mpa')
  })
})
