export function buildApiBase(protocol: string, hostname: string, port?: string): string {
  return `${protocol}//${hostname}:${port || '8000'}/api`
}

type SubmissionCrypto = {
  randomUUID?: () => string
  getRandomValues: (bytes: Uint8Array) => Uint8Array
}

export function createSubmissionId(
  cryptoApi: SubmissionCrypto = globalThis.crypto as unknown as SubmissionCrypto,
): string {
  if (typeof cryptoApi.randomUUID === 'function') return cryptoApi.randomUUID()
  const bytes = cryptoApi.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

const browserLocation = typeof window === 'undefined'
  ? { protocol: 'http:', hostname: 'localhost' }
  : window.location

export const API = buildApiBase(browserLocation.protocol, browserLocation.hostname, import.meta.env.VITE_API_PORT)
