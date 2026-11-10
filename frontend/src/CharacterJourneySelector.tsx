import type { RunManifest } from './history'
import { groupRunsByRecency, isAutoCreatedRun, isRunArchived } from './runOrganization'

type Props = {
  runs: RunManifest[]
  activeRunId: string | null
  loading: boolean
  onSelect: (runId: string) => void
  onNew: () => void
  // §3-C7: archived runs (client-side today) never clutter the selector —
  // the currently-open run is always kept visible regardless.
  archivedRunIds?: Set<string>
}

export default function CharacterJourneySelector({ runs, activeRunId, loading, onSelect, onNew, archivedRunIds = new Set() }: Props) {
  const visible = runs.filter((run) => !isRunArchived(run, archivedRunIds) || run.id === activeRunId)
  const groups = groupRunsByRecency(visible)
  return <section className="journey-selector" aria-label="Character journey selector">
    <label>
      <span>Character journey</span>
      <select
        aria-label="Character journey"
        disabled={loading}
        value={activeRunId || ''}
        onChange={(event) => event.target.value && onSelect(event.target.value)}
      >
        {!activeRunId && <option value="">Select a character…</option>}
        {groups.map((group) => <optgroup label={group.label} key={group.label}>
          {group.runs.map((run) => <option key={run.id} value={run.id}>{run.name}{isAutoCreatedRun(run) ? ' · auto' : ''}</option>)}
        </optgroup>)}
      </select>
    </label>
    <span className="journey-selector-status">{loading ? 'Loading journey…' : activeRunId ? 'All stages use this character' : 'No journey selected'}</span>
    <button className="secondary-button" disabled={loading} onClick={onNew}>New character</button>
  </section>
}
