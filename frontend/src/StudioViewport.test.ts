import { describe, expect, it } from 'vitest'

import { readableAnimationName } from './animation'


describe('readableAnimationName', () => {
  it('turns embedded clip identifiers into labels', () => {
    expect(readableAnimationName('sword_attack')).toBe('Sword Attack')
    expect(readableAnimationName('SwordAttack')).toBe('Sword Attack')
  })
})
