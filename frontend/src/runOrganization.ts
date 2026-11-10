// Run-list housekeeping for the character journey selector and History (plan
// §3-C7): the run list accumulates every agent test run ever created, most
// of it debris ("* experiment" one-shots with no promoted output). None of
// this needs the backend — it is pure client-side sorting, grouping, and an
// archive flag kept in localStorage until the backend contract grows a real
// `archived` field on the run manifest.

export type RunForOrganization = {
  id: string
  name: string
  updated_at: number
  archived?: boolean
  attempts: Array<{ promoted?: boolean }>
}

// Newest first — the natural "most relevant" order once a run list gets
// long enough that alphabetical/insertion order stops being useful.
export function sortRunsByRecency<T extends { updated_at: number }>(runs: T[]): T[] {
  return [...runs].sort((a, b) => b.updated_at - a.updated_at)
}

export type RunGroup<T> = { label: 'Recent' | 'Older'; runs: T[] }

// Groups an unsorted run list into "Recent" (the newest N, default 10) and
// "Older" (everything else), sorted newest-first within each group. A short
// list collapses to a single "Recent" group — no empty "Older" heading.
export function groupRunsByRecency<T extends { updated_at: number }>(runs: T[], recentCount = 10): RunGroup<T>[] {
  const sorted = sortRunsByRecency(runs)
  const groups: RunGroup<T>[] = []
  const recent = sorted.slice(0, recentCount)
  const older = sorted.slice(recentCount)
  if (recent.length) groups.push({ label: 'Recent', runs: recent })
  if (older.length) groups.push({ label: 'Older', runs: older })
  return groups
}

// A run is "auto-created" debris when it carries the auto-generated
// "<subject> experiment" name and never actually promoted anything — one
// attempt, no promotion. Flagged with a muted badge, never hidden outright:
// the operator may still want to look at exactly one of these.
export function isAutoCreatedRun(run: { name: string; attempts: Array<{ promoted?: boolean }> }): boolean {
  return run.name.trim().toLowerCase().endsWith('experiment')
    && run.attempts.length === 1
    && !run.attempts.some((attempt) => attempt.promoted)
}

// --- Client-side archive (localStorage fallback) --------------------------
//
// `PATCH /api/runs/{id}` does not accept an `archived` field yet (the
// review-request shape is a strict either/or of review fields or
// subject_kind — see backend/app/main.py's RunReviewRequest). Until the
// backend contract grows a real `archived` flag, "archiving" a run is
// purely a client-side hide, persisted per-browser.

export const ARCHIVED_RUNS_STORAGE_KEY = 'aismith.archivedRuns'

export function readArchivedRunIds(raw: string | null): Set<string> {
  if (!raw) return new Set()
  try {
    const parsed: unknown = JSON.parse(raw)
    return new Set(Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [])
  } catch {
    return new Set()
  }
}

export function withArchivedRunId(current: Set<string>, runId: string, archived: boolean): Set<string> {
  const next = new Set(current)
  if (archived) next.add(runId)
  else next.delete(runId)
  return next
}

// A run counts as archived once the backend says so OR (today, always) its
// id is in the client-side archive set — either is sufficient, so a future
// backend-native `archived: true` keeps working unchanged.
export function isRunArchived(run: { archived?: boolean; id: string }, localArchivedIds: Set<string>): boolean {
  return Boolean(run.archived) || localArchivedIds.has(run.id)
}

// The run list a view should actually render: everything when `showArchived`
// is set, otherwise every non-archived run plus (defensively) whichever run
// is currently open, so archiving the run you're looking at cannot yank it
// out from under you.
export function visibleRuns<T extends { archived?: boolean; id: string }>(
  runs: T[],
  localArchivedIds: Set<string>,
  showArchived: boolean,
  keepRunId?: string | null,
): T[] {
  if (showArchived) return runs
  return runs.filter((run) => !isRunArchived(run, localArchivedIds) || run.id === keepRunId)
}
