import { describe, expect, it } from 'vitest'
import {
  groupRunsByRecency, isAutoCreatedRun, isRunArchived, readArchivedRunIds, sortRunsByRecency,
  visibleRuns, withArchivedRunId,
} from './runOrganization'

describe('sortRunsByRecency', () => {
  it('orders newest updated_at first without mutating the input', () => {
    const runs = [{ id: 'a', updated_at: 1 }, { id: 'b', updated_at: 3 }, { id: 'c', updated_at: 2 }]
    const sorted = sortRunsByRecency(runs)
    expect(sorted.map((run) => run.id)).toEqual(['b', 'c', 'a'])
    expect(runs.map((run) => run.id)).toEqual(['a', 'b', 'c'])
  })
})

describe('groupRunsByRecency', () => {
  it('splits into Recent (newest N) and Older, both newest-first', () => {
    const runs = Array.from({ length: 14 }, (_, index) => ({ id: `run-${index}`, updated_at: index }))
    const groups = groupRunsByRecency(runs, 10)

    expect(groups).toHaveLength(2)
    expect(groups[0].label).toBe('Recent')
    expect(groups[0].runs).toHaveLength(10)
    expect(groups[0].runs[0].id).toBe('run-13')
    expect(groups[1].label).toBe('Older')
    expect(groups[1].runs).toHaveLength(4)
    expect(groups[1].runs[0].id).toBe('run-3')
  })

  it('collapses to a single Recent group when there are not enough runs for Older', () => {
    const runs = [{ id: 'a', updated_at: 2 }, { id: 'b', updated_at: 1 }]
    const groups = groupRunsByRecency(runs, 10)

    expect(groups).toEqual([{ label: 'Recent', runs: [{ id: 'a', updated_at: 2 }, { id: 'b', updated_at: 1 }] }])
  })

  it('returns no groups for an empty run list', () => {
    expect(groupRunsByRecency([], 10)).toEqual([])
  })
})

describe('isAutoCreatedRun', () => {
  it('flags a single-attempt, never-promoted "* experiment" run', () => {
    expect(isAutoCreatedRun({ name: 'Serpent F3 test experiment', attempts: [{ promoted: false }] })).toBe(true)
    expect(isAutoCreatedRun({ name: 'mary-rig.glb Experiment', attempts: [{}] })).toBe(true)
  })

  it('does not flag a run that promoted an output', () => {
    expect(isAutoCreatedRun({ name: 'mary-rig.glb experiment', attempts: [{ promoted: true }] })).toBe(false)
  })

  it('does not flag a run with more than one attempt', () => {
    expect(isAutoCreatedRun({ name: 'mary-rig.glb experiment', attempts: [{}, {}] })).toBe(false)
  })

  it('does not flag a deliberately named run', () => {
    expect(isAutoCreatedRun({ name: 'Mary — final delivery', attempts: [{}] })).toBe(false)
  })
})

describe('archived-run localStorage helpers', () => {
  it('parses a stored id list defensively', () => {
    expect(readArchivedRunIds(null)).toEqual(new Set())
    expect(readArchivedRunIds('not json')).toEqual(new Set())
    expect(readArchivedRunIds('[]')).toEqual(new Set())
    expect(readArchivedRunIds('["run-1","run-2"]')).toEqual(new Set(['run-1', 'run-2']))
    expect(readArchivedRunIds('["run-1", 42, null]')).toEqual(new Set(['run-1']))
  })

  it('adds and removes ids without mutating the input set', () => {
    const original = new Set(['run-1'])
    const added = withArchivedRunId(original, 'run-2', true)
    expect(added).toEqual(new Set(['run-1', 'run-2']))
    expect(original).toEqual(new Set(['run-1']))

    const removed = withArchivedRunId(added, 'run-1', false)
    expect(removed).toEqual(new Set(['run-2']))
  })

  it('treats a backend-flagged run as archived even without a local id', () => {
    expect(isRunArchived({ id: 'run-1', archived: true }, new Set())).toBe(true)
    expect(isRunArchived({ id: 'run-1' }, new Set(['run-1']))).toBe(true)
    expect(isRunArchived({ id: 'run-1' }, new Set())).toBe(false)
  })
})

describe('visibleRuns', () => {
  const runs = [{ id: 'a' }, { id: 'b', archived: true }, { id: 'c' }]

  it('hides archived runs by default', () => {
    expect(visibleRuns(runs, new Set(), false).map((run) => run.id)).toEqual(['a', 'c'])
  })

  it('shows everything when showArchived is set', () => {
    expect(visibleRuns(runs, new Set(), true).map((run) => run.id)).toEqual(['a', 'b', 'c'])
  })

  it('hides a locally archived run too', () => {
    expect(visibleRuns(runs, new Set(['a']), false).map((run) => run.id)).toEqual(['c'])
  })

  it('never hides the run currently open, even if archived', () => {
    expect(visibleRuns(runs, new Set(), false, 'b').map((run) => run.id)).toEqual(['a', 'b', 'c'])
  })
})
