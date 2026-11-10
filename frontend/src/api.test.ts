import { describe, expect, it } from 'vitest'

import { buildApiBase, createSubmissionId } from './api'


describe('buildApiBase', () => {
  it('uses the browser hostname with the configured API port', () => {
    expect(buildApiBase('http:', '192.0.2.10', '8100')).toBe('http://192.0.2.10:8100/api')
  })

  it('defaults to port 8000', () => {
    expect(buildApiBase('http:', 'localhost', undefined)).toBe('http://localhost:8000/api')
  })
})

describe('createSubmissionId', () => {
  it('falls back to getRandomValues when randomUUID is unavailable on an HTTP origin', () => {
    const cryptoApi = {
      getRandomValues: (bytes: Uint8Array) => {
        bytes.set(Array.from({ length: 16 }, (_, index) => index))
        return bytes
      },
    }

    expect(createSubmissionId(cryptoApi)).toBe('00010203-0405-4607-8809-0a0b0c0d0e0f')
  })
})
