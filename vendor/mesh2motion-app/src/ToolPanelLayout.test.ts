import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('joint editor tool panel', () => {
  it('reserves a compact fixed-width inspection panel', () => {
    const styles = readFileSync(`${process.cwd()}/src/styles.css`, 'utf8')
    document.head.innerHTML = `<style>${styles}</style>`
    document.body.innerHTML = '<aside id="tool-panel"></aside>'

    expect(getComputedStyle(document.getElementById('tool-panel')!).width).toBe('300px')
  })
})
