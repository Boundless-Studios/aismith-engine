export type RetargetLaunch = {
  source: string
  extension: 'glb'
  autoOpenLibrary: boolean
}

export function resolveRetargetLaunch(search: string, currentHostname = 'localhost'): RetargetLaunch | null {
  const params = new URLSearchParams(search)
  const source = params.get('source')
  if (!source || params.get('extension')?.toLowerCase() !== 'glb' || !isLocalArtifactDownload(source, currentHostname)) return null
  return { source, extension: 'glb', autoOpenLibrary: params.get('library') !== 'manual' }
}

function isLocalArtifactDownload(source: string, currentHostname: string): boolean {
  const artifactPath = /^\/api\/assets\/[^/]+\/download$/
  if (source.startsWith('/')) return artifactPath.test(source)
  try {
    const url = new URL(source)
    return [currentHostname, '127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) && artifactPath.test(url.pathname)
  } catch {
    return false
  }
}
