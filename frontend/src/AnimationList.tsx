import type { JourneyAnimation } from './animations'

// `createdAt` is epoch SECONDS (see JourneyAnimation.createdAt) — the same
// unit HistoryView.tsx converts with `new Date(run.updated_at * 1000)`, so
// this multiplies by 1000 before doing any Date math, the same way.
function formatRelativeTime(createdAtSeconds: number): string {
  if (!createdAtSeconds) return ''
  const timestampMs = createdAtSeconds * 1000
  const diffMinutes = Math.round((Date.now() - timestampMs) / 60000)
  if (diffMinutes < 1) return 'just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  const diffHours = Math.round(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.round(diffHours / 24)
  if (diffDays < 30) return `${diffDays}d ago`
  return new Date(timestampMs).toLocaleDateString()
}

const shortId = (id: string): string => id.slice(0, 8)

// The rig name when it's already in the details cache (fetched for the
// "Working on" rig picker — same artifact ids), else a short id: never
// blocks on a network round trip just to label a row.
function rigLabelFor(inputArtifactId: string, detailsById: Record<string, { name?: string }>): string {
  return detailsById[inputArtifactId]?.name || shortId(inputArtifactId)
}

// The Animate tab's inline replacement for the old separate Review tab: a
// flat, selectable list of every embedded clip this run has produced.
// Selecting a row pins that clip as the workspace's current animate source
// (via the same onPinArtifact mechanism the "Working on" picker uses);
// accept/reject/export for the selected clip lives in the "Selected clip"
// panel below this list, not here.
export function AnimationList({
  animations, selectedArtifactId, onSelect, detailsById = {},
}: {
  animations: JourneyAnimation[]
  selectedArtifactId?: string | null
  onSelect: (artifactId: string) => void
  detailsById?: Record<string, { name?: string }>
}) {
  return <div className="animation-list">
    <div className="rig-curated-copy"><span className="eyebrow">ANIMATIONS</span><strong>Embedded clips</strong></div>
    {!animations.length && <small className="motion-hint">No animated clips yet — attach or bake motion below.</small>}
    {animations.length > 0 && <ul className="animation-list-rows">
      {animations.map((animation) => {
        const selected = animation.artifactId === selectedArtifactId
        const chip = animation.reviewState === 'accepted' ? { label: 'Accepted', className: 'ok' }
          : animation.reviewState === 'rejected' ? { label: 'Rejected', className: 'bad' }
          : animation.promoted === false ? { label: 'Retained', className: 'warn' }
          : null
        return <li key={animation.artifactId}>
          <button
            type="button"
            className={`animation-list-row ${selected ? 'selected' : ''}`}
            aria-pressed={selected}
            onClick={() => onSelect(animation.artifactId)}
          >
            <span className="animation-list-row-main">
              <strong>{animation.label}</strong>
              <small>{formatRelativeTime(animation.createdAt)}</small>
              {/* Six "Sword Attack" rows otherwise read identically — this
                  ties each one to the exact rig and output artifact it is. */}
              <small className="animation-list-row-caption">
                on {animation.inputArtifactId ? rigLabelFor(animation.inputArtifactId, detailsById) : '—'} · {shortId(animation.artifactId)}
              </small>
            </span>
            {chip && <span className={`status-chip ${chip.className}`}>{chip.label}</span>}
          </button>
        </li>
      })}
    </ul>}
  </div>
}
