import { Upload } from 'lucide-react'
import { runAssetOptions, type WorkingOnStage } from './workspace'

export type WorkingOnAsset = { id: string; name: string; kind?: string; size_bytes?: number }
export type WorkingOnAssetDetails = { name?: string; vertices?: number; size_bytes?: number }

type WorkingOnRun = { source_artifact_id: string; source_artifact_kind: string; attempts: Array<{ stage: string; status: string; output_artifact_id?: string | null; completed_at?: number | null; created_at?: number }> } | null

function formatBytes(bytes?: number): string | null {
  if (!bytes) return null
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`
}

function formatDuration(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

// Every stage panel shows one of these at the top: what artifact the stage
// action will act on, and — when the run has more than one compatible
// candidate — a picker to pin a different one (I1/I2).
export function WorkingOn({
  stage, stageLabel, run, current, detailsById, pinnedArtifactId, onPin, onUpload, uploadLabel = 'Upload / Replace', elapsedMs, hideUploadButton = false,
}: {
  stage: WorkingOnStage
  stageLabel: string
  run: WorkingOnRun
  current: WorkingOnAsset | null
  detailsById: Record<string, WorkingOnAssetDetails>
  pinnedArtifactId: string | null
  onPin: (artifactId: string) => void
  onUpload: () => void
  uploadLabel?: string
  elapsedMs?: number | null
  // Generate already has its own upload affordance in the source-image grid
  // right below this card — showing a second "Upload / Replace" here would
  // just be a redundant control for the same action.
  hideUploadButton?: boolean
}) {
  const baseOptions = runAssetOptions(run, stage)
  // The select must always have an option matching whatever the header
  // above it actually shows — `current` (the artifact the stage action will
  // act on) does not always appear in the run's stage-input options (e.g.
  // Clean's own options are Generate outputs, but `current` is already a
  // Clean output once cleaning has run once). Without this, the browser
  // falls back to its own first option instead, silently mismatching the
  // header. The current artifact is always listed first.
  const matchedCurrentOption = current ? baseOptions.find((option) => option.artifactId === current.id) : undefined
  const restOptions = baseOptions.filter((option) => option.artifactId !== current?.id)
  const options = current
    ? [{
      artifactId: current.id,
      stage: matchedCurrentOption?.stage ?? 'current',
      completedAt: Infinity,
      isSource: matchedCurrentOption?.isSource ?? false,
    }, ...restOptions]
    : baseOptions
  const selectedId = pinnedArtifactId && options.some((option) => option.artifactId === pinnedArtifactId)
    ? pinnedArtifactId
    : (current?.id ?? '')
  const currentDetails = current ? detailsById[current.id] : undefined
  const sizeLabel = currentDetails?.vertices !== undefined
    ? `${currentDetails.vertices.toLocaleString()} verts`
    : formatBytes(currentDetails?.size_bytes ?? current?.size_bytes) || '—'

  return <div className="working-on-card">
    <div className="working-on-header">
      <span className="working-on-badge">{stageLabel}</span>
      <strong className="working-on-name">{current ? current.name : 'No artifact selected'}</strong>
    </div>
    <div className="working-on-meta">
      <span>{sizeLabel}</span>
      {elapsedMs != null && <span>· {formatDuration(elapsedMs)}</span>}
    </div>
    {options.length > 1 && <select
      className="working-on-picker"
      value={selectedId}
      onChange={(event) => onPin(event.target.value)}
    >
      {options.map((option) => <option key={option.artifactId} value={option.artifactId}>
        {option.artifactId === current?.id ? current.name : (detailsById[option.artifactId]?.name || (option.isSource ? 'Run source' : option.stage))}
      </option>)}
    </select>}
    {!hideUploadButton && <button type="button" className="secondary-button working-on-upload" onClick={onUpload}><Upload size={13} /> {uploadLabel}</button>}
  </div>
}
