import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, ArrowLeft, Archive, ArchiveRestore, Box, Check, Download, Eye, FolderClock, RotateCcw, ShieldCheck, Sparkles, X } from 'lucide-react'

import { artifactForHistoryUse, bilateralSymmetrySummary, canBakeMotionChild, canPreviewArtifact, currentArtifactStage, deformationReviewArtifacts, formatSetting, hasActiveMotionBake, isAutomatedQualityAction, isImageArtifactKind, isRetainedCandidate, isRunPipelineLocked, qualityActionsForDisplay, qualityDecisionLabel, qualityDecisionPassed, skeletonReviewArtifacts, stageForQualityAction, waitForRunTerminal, type Artifact, type QualityDecision, type ReviewState, type RunManifest } from './history'
import { API, createSubmissionId } from './api'
import { canRestoreHistoryArtifact, pollJobUntilTerminal, retryNetworkPost } from './workspace'
import { ARCHIVED_RUNS_STORAGE_KEY, isAutoCreatedRun, isRunArchived, readArchivedRunIds, sortRunsByRecency, visibleRuns, withArchivedRunId } from './runOrganization'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error(`History request failed (${response.status})`)
  return response.json()
}

type Props = {
  onUseArtifact: (artifact: Artifact, stage: string, run: RunManifest, isSource?: boolean) => void
  blenderAvailable: boolean
  initialRunId?: string | null
}

export default function HistoryView({ onUseArtifact, blenderAvailable, initialRunId }: Props) {
  const [runs, setRuns] = useState<RunManifest[]>([])
  const [selected, setSelected] = useState<RunManifest | null>(null)
  const [artifacts, setArtifacts] = useState<Record<string, Artifact>>({})
  const [error, setError] = useState('')
  const [busyAction, setBusyAction] = useState('')
  const [busyRunId, setBusyRunId] = useState('')
  // §3-C7: client-side archive (localStorage) until the backend contract
  // grows a real `archived` field on the run manifest.
  const [archivedRunIds, setArchivedRunIds] = useState<Set<string>>(() => (
    typeof window === 'undefined' ? new Set() : readArchivedRunIds(window.localStorage.getItem(ARCHIVED_RUNS_STORAGE_KEY))
  ))
  const [showArchived, setShowArchived] = useState(false)
  const toggleArchived = (run: RunManifest, event: React.MouseEvent) => {
    event.stopPropagation()
    const archived = !isRunArchived(run, archivedRunIds)
    const next = withArchivedRunId(archivedRunIds, run.id, archived)
    setArchivedRunIds(next)
    window.localStorage.setItem(ARCHIVED_RUNS_STORAGE_KEY, JSON.stringify([...next]))
  }
  const selectedBusyAction = selected?.id === busyRunId ? busyAction : ''
  const mountedRef = useRef(true)
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false } }, [])
  const load = () => request<RunManifest[]>('/runs').then(setRuns).catch((reason) => setError(reason.message))
  useEffect(() => { load() }, [])
  useEffect(() => {
    if (!initialRunId) return
    request<RunManifest>(`/runs/${initialRunId}`).then(setSelected).catch((reason) => setError(reason.message))
  }, [initialRunId])
  useEffect(() => {
    if (!runs.some((run) => ['queued', 'running'].includes(run.status))) return
    const timer = window.setInterval(load, 1000)
    return () => window.clearInterval(timer)
  }, [runs])
  useEffect(() => {
    if (!selected || !['queued', 'running'].includes(selected.status)) return
    const timer = window.setInterval(() => request<RunManifest>(`/runs/${selected.id}`).then(setSelected).catch((reason) => setError(reason.message)), 1000)
    return () => window.clearInterval(timer)
  }, [selected?.id, selected?.status])
  useEffect(() => {
    if (!selected) { setArtifacts({}); return }
    const ids = [...new Set([selected.source_artifact_id, ...selected.attempts.map((attempt) => attempt.output_artifact_id).filter((id): id is string => Boolean(id))])]
    Promise.all(ids.map((id) => request<Artifact>(`/assets/${id}`).catch(() => null))).then((loaded) => {
      setArtifacts(Object.fromEntries(loaded.filter((artifact): artifact is Artifact => artifact !== null).map((artifact) => [artifact.id, artifact])))
    })
  }, [selected?.id, selected?.updated_at])

  const review = async (status: ReviewState) => {
    if (!selected?.current_artifact_id) return
    const prompted = window.prompt(status === 'rejected' ? 'What failed visually?' : 'What did you verify across the full deformation suite?', selected.review_note)
    if (prompted === null) return
    const note = prompted.trim()
    if (!note) { setError('Describe what you inspected or what failed before saving this decision.'); return }
    const updated = await request<RunManifest>(`/runs/${selected.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: status, review_note: note, artifact_id: selected.current_artifact_id }),
    })
    setSelected(updated)
    load()
  }

  const executeQualityAction = async (action: string) => {
    if (!selected || isRunPipelineLocked(selected.attempts, selectedBusyAction)) return
    setBusyAction(action)
    setBusyRunId(selected.id)
    setError('')
    try {
      if (action === 'render-deformation-review') {
        const artifactId = selected.current_artifact_id || selected.source_artifact_id
        await request(`/assets/${artifactId}/diagnose-deformation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ run_id: selected.id }),
        })
        const updated = await request<RunManifest>(`/runs/${selected.id}`)
        setSelected(updated)
        load()
        return
      }
      const meshRecovery = ['refine-reconstruct', 'repair-topology', 'retopologize'].includes(action)
      await request(`/runs/${selected.id}/quality-actions/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          submission_id: `history-${action}-${createSubmissionId()}`,
          ...(meshRecovery ? { target_vertices: 50000, settings: { repair_holes: false, gpu_first: false } } : {}),
        }),
      })
      const updated = await waitForRunTerminal(
        () => request<RunManifest>(`/runs/${selected.id}`),
        () => new Promise((resolve) => window.setTimeout(resolve, 500)),
      )
      setSelected(updated)
      load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusyAction('')
      setBusyRunId('')
    }
  }

  const promoteCandidate = async (artifactId: string) => {
    if (!selected || isRunPipelineLocked(selected.attempts, selectedBusyAction)) return
    const reason = window.prompt('Why should this candidate become current?', 'Accepted after visual comparison')
    if (!reason?.trim()) return
    setBusyAction(artifactId)
    setBusyRunId(selected.id)
    setError('')
    try {
      const updated = await request<RunManifest>(`/runs/${selected.id}/artifacts/${artifactId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: reason.trim() }),
      })
      setSelected(updated)
      load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusyAction('')
      setBusyRunId('')
    }
  }

  const bakeMotion = async (motionArtifactId: string) => {
    if (!selected?.current_artifact_id || isRunPipelineLocked(selected.attempts, selectedBusyAction)) return
    const selectedRunId = selected.id
    const payload = {
      kind: 'retarget-motion',
      submission_id: `history-retarget-motion-${createSubmissionId()}`,
      source_artifact_id: selected.current_artifact_id,
      run_id: selectedRunId,
      settings: { motion_artifact_id: motionArtifactId, animation: 'Sword_Attack' },
    }
    setBusyAction(motionArtifactId)
    setBusyRunId(selectedRunId)
    setError('')
    try {
      let created = await retryNetworkPost(() => request<{ id: string; status: string }>('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }))
      created = await pollJobUntilTerminal(
        created,
        (jobId) => request<{ id: string; status: string }>(`/jobs/${jobId}`),
        () => new Promise((resolve) => window.setTimeout(resolve, 500)),
        () => mountedRef.current,
      )
      if (!mountedRef.current) return
      const updated = await request<RunManifest>(`/runs/${selectedRunId}`)
      setSelected((current) => current?.id === selectedRunId ? updated : current)
      load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusyAction('')
      setBusyRunId('')
    }
  }

  if (!selected) {
    const archivedCount = runs.filter((run) => isRunArchived(run, archivedRunIds)).length
    const displayedRuns = sortRunsByRecency(visibleRuns(runs, archivedRunIds, showArchived))
    return <main className="history-page">
      <div className="history-heading"><div><span className="eyebrow">LOCAL EXPERIMENTS</span><h1>Run history</h1><p>Every source, retry, parameter, warning, and artifact remains available.</p></div>
        {archivedCount > 0 && <button className="secondary-button" onClick={() => setShowArchived((current) => !current)}>{showArchived ? 'Hide archived' : `Show archived (${archivedCount})`}</button>}
        <FolderClock size={28} /></div>
      {error && <div className="error-box">{error}</div>}
      <section className="run-grid">
        {displayedRuns.map((run) => {
          const archived = isRunArchived(run, archivedRunIds)
          return <div className={`run-card ${run.review_status}${archived ? ' archived' : ''}`} role="button" tabIndex={0} onClick={() => request<RunManifest>(`/runs/${run.id}`).then(setSelected)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') request<RunManifest>(`/runs/${run.id}`).then(setSelected) }} key={run.id}>
            <span className="run-card-thumb">
              {isImageArtifactKind(run.source_artifact_kind)
                ? <img src={`${API}/assets/${run.source_artifact_id}/download`} alt="" loading="lazy" decoding="async" onError={(event) => { event.currentTarget.style.display = 'none'; const fallback = event.currentTarget.nextElementSibling as HTMLElement | null; if (fallback) fallback.style.display = 'grid' }} />
                : null}
              <span className="run-card-thumb-icon" style={{ display: isImageArtifactKind(run.source_artifact_kind) ? 'none' : 'grid' }}><Box size={22} /></span>
            </span>
            <div className="run-card-body">
              <div className="run-card-title"><strong>{run.name}</strong>{isAutoCreatedRun(run) && <span className="auto-created-badge">auto-created</span>}<span className={`review-badge ${run.review_status}`}>{run.review_status}</span></div>
              <p>{run.attempts.length ? `${currentArtifactStage(run)} · ${run.status}` : 'Source ready'}</p>
              <small>{run.attempts.length} stage attempt{run.attempts.length === 1 ? '' : 's'} · {new Date(run.updated_at * 1000).toLocaleString()}</small>
            </div>
            <button type="button" className="secondary-button run-card-archive" onClick={(event) => toggleArchived(run, event)} title={archived ? 'Restore this run to the active list' : 'Archive this run'}>
              {archived ? <ArchiveRestore size={14} /> : <Archive size={14} />}
            </button>
          </div>
        })}
        {!displayedRuns.length && !error && <div className="history-empty"><FolderClock size={32} /><strong>{runs.length ? 'No runs to show' : 'No runs yet'}</strong><span>{runs.length ? 'Every run is archived — toggle "Show archived" above to see them.' : 'Upload an image and start Generate. Its full lineage will appear here.'}</span></div>}
      </section>
    </main>
  }

  const attempts = [...selected.attempts].reverse()
  const currentAttempt = attempts.find((attempt) => attempt.output_artifact_id === selected.current_artifact_id)
  const currentStage = currentArtifactStage(selected)
  const currentArtifact = currentAttempt?.output_artifact_id
    ? artifactForHistoryUse(artifacts, currentAttempt.output_artifact_id)
    : null
  const sourceArtifact = artifactForHistoryUse(artifacts, selected.source_artifact_id)
  const motionBakeActive = hasActiveMotionBake(selected.attempts)
  const runLocked = isRunPipelineLocked(selected.attempts, selectedBusyAction)
  return <main className="history-page">
    <div className="history-heading"><button className="secondary-button" onClick={() => setSelected(null)}><ArrowLeft size={15} /> All runs</button><div><span className="eyebrow">EXPERIMENT</span><h1>{selected.name}</h1></div><span className={`review-badge ${selected.review_status}`}>{selected.review_status}</span></div>
    {error && <div className="error-box">{error}</div>}
    {currentAttempt?.output_artifact_id && <section className="history-current"><div><span className="eyebrow">CURRENT ARTIFACT</span><h2>{currentStage} artifact</h2><p>The best promoted output is retained with its measured readiness and recovery options.</p></div><div className="attempt-actions"><a href={`${API}/assets/${currentAttempt.output_artifact_id}/download`}><Download size={14} /> Download</a><button disabled={runLocked || !currentArtifact} onClick={() => currentArtifact && onUseArtifact(currentArtifact, currentStage, selected)}><Eye size={14} /> {motionBakeActive ? 'Bake in progress…' : 'Preview / continue'}</button></div></section>}
    {currentAttempt?.output_artifact_id && <QualityPanels artifact={artifacts[currentAttempt.output_artifact_id]} sourceArtifact={sourceArtifact || undefined} run={selected} onUseArtifact={onUseArtifact} onExecuteAction={executeQualityAction} busyAction={runLocked ? selectedBusyAction || 'pipeline-active' : ''} />}
    <div className="history-review"><strong>Visual verdict</strong><span>{selected.review_note || 'Not reviewed yet.'}</span><div><button onClick={() => review('accepted')}><Check size={14} /> Accept</button><button onClick={() => review('rejected')}><X size={14} /> Reject</button></div></div>
    <section className="attempt-timeline">
      <article className="attempt-card"><div className="attempt-index">0</div><div className="attempt-content"><div className="attempt-title"><strong>Source {isImageArtifactKind(selected.source_artifact_kind) ? 'image' : 'model'}</strong><span>retained</span></div>{isImageArtifactKind(selected.source_artifact_kind) && <img className="history-source-large" src={`${API}/assets/${selected.source_artifact_id}/download`} alt="Source" />}<div className="attempt-actions"><a href={`${API}/assets/${selected.source_artifact_id}/download`}><Download size={14} /> Download</a><button disabled={runLocked || !sourceArtifact} onClick={() => sourceArtifact && onUseArtifact(sourceArtifact, 'generate', selected, true)}><RotateCcw size={14} /> Use</button></div></div></article>
      {attempts.map((attempt, index) => { const artifact = attempt.output_artifact_id ? artifacts[attempt.output_artifact_id] : undefined; const selectedCurrentArtifact = selected.current_artifact_id ? artifacts[selected.current_artifact_id] : undefined; const previewable = canPreviewArtifact(artifact?.kind); const bakeableMotion = canBakeMotionChild(artifact, selectedCurrentArtifact, blenderAvailable); const retained = isRetainedCandidate(attempt, selected.current_artifact_id, artifact?.kind); const isCurrent = attempt.output_artifact_id === selected.current_artifact_id; const restorable = Boolean(artifact && canRestoreHistoryArtifact(attempt.stage, artifact.id, selected.current_artifact_id, attempt.promoted)); return <article className={`attempt-card ${attempt.status}${retained ? ' retained' : ''}`} key={attempt.id}><div className="attempt-index">{selected.attempts.length - index}</div><div className="attempt-content"><div className="attempt-title"><strong>{attempt.stage}</strong><span>{attempt.status}{isCurrent ? ' · current' : retained ? ' · retained candidate' : artifact?.kind === 'bvh' ? ' · motion child' : ''}</span></div><dl className="attempt-settings">{Object.entries(attempt.settings).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatSetting(value)}</dd></div>)}</dl>{attempt.error && <div className="attempt-error">{attempt.error}</div>}{attempt.logs.length > 0 && <details><summary>Logs ({attempt.logs.length})</summary><pre>{attempt.logs.join('\n')}</pre></details>}{attempt.warnings.map((warning) => <div className="attempt-warning" key={warning}>{warning}</div>)}{attempt.output_artifact_id && <QualityPanels artifact={artifact} sourceArtifact={sourceArtifact || undefined} run={selected} onUseArtifact={onUseArtifact} onExecuteAction={isCurrent ? executeQualityAction : undefined} busyAction={runLocked ? selectedBusyAction || 'pipeline-active' : ''} compact />}{attempt.output_artifact_id && <div className="attempt-actions"><a href={`${API}/assets/${attempt.output_artifact_id}/download`}><Download size={14} /> {artifact?.kind === 'bvh' ? 'Download BVH motion' : 'Download'}</a>{bakeableMotion && <button disabled={runLocked} onClick={() => bakeMotion(attempt.output_artifact_id!)}><Sparkles size={14} /> {motionBakeActive ? 'Bake in progress…' : selectedBusyAction === attempt.output_artifact_id ? 'Starting…' : 'Bake onto current rig'}</button>}{previewable && artifact && <button disabled={runLocked || !restorable} onClick={() => restorable && onUseArtifact(artifact, attempt.stage, selected)}><Eye size={14} /> {restorable ? 'Preview / use' : 'Retained · inspect only'}</button>}{retained && <button disabled={runLocked} onClick={() => promoteCandidate(attempt.output_artifact_id!)}><ShieldCheck size={14} /> Promote candidate</button>}</div>}</div></article> })}
    </section>
  </main>
}

function actionLabel(action: string): string {
  return action.split('-').map((word) => word[0]?.toUpperCase() + word.slice(1)).join(' ')
}

function DeformationReviewGallery({ artifact }: { artifact?: Artifact }) {
  const reviewArtifacts = deformationReviewArtifacts(artifact)
  if (!reviewArtifacts.length) return null
  return <section id={`deformation-review-${artifact?.id}`} className="deformation-review-gallery" aria-label="Deformation review frames">
    <div><span className="eyebrow">DEFORMATION REVIEW</span><h3>Rest and Sword Attack frames</h3><p>Durable evidence captured from this exact GLB.</p></div>
    <div className="deformation-review-grid">{reviewArtifacts.map((review) => <a href={`${API}/assets/${review.artifact_id}/download`} target="_blank" rel="noreferrer" key={review.artifact_id}><img src={`${API}/assets/${review.artifact_id}/download`} alt={`${review.label} deformation review`} /><span>{review.label.replace('-', ' ')}</span></a>)}</div>
  </section>
}

export function SkeletonReviewGallery({ artifact }: { artifact?: Artifact }) {
  const reviewArtifacts = skeletonReviewArtifacts(artifact)
  if (!reviewArtifacts.length) return null
  const ordered = [...reviewArtifacts].sort((a, b) => (a.label === 'skeleton-sheet' ? -1 : b.label === 'skeleton-sheet' ? 1 : 0))
  return <section id={`skeleton-review-${artifact?.id}`} className="skeleton-review-gallery" aria-label="Skeleton contact sheet">
    <div><span className="eyebrow">SKELETON REVIEW</span><h3>Front / right / left / back bone sheet</h3><p>Left/right joint placement, rendered for a visual check the numbers alone can't give.</p></div>
    <div className="skeleton-review-grid">{ordered.map((review) => <a href={`${API}/assets/${review.artifact_id}/download`} target="_blank" rel="noreferrer" key={review.artifact_id}><img src={`${API}/assets/${review.artifact_id}/download`} alt={`${review.label} skeleton review`} /><span>{review.label.replace('-', ' ')}</span></a>)}</div>
  </section>
}

export function QualityPanels({ artifact, sourceArtifact, run, onUseArtifact, onExecuteAction, busyAction = '', compact = false }: { artifact?: Artifact; sourceArtifact?: Artifact; run: RunManifest; onUseArtifact: Props['onUseArtifact']; onExecuteAction?: (action: string) => void; busyAction?: string; compact?: boolean }) {
  // Keyed entries, not values: the `source_fidelity` / `project_views`
  // decisions carry no `target_stage`, so the key is their only label.
  const decisions = Object.entries(artifact?.metadata.quality_decisions || {})
    .filter((entry): entry is [string, QualityDecision] => Boolean(entry[1]) && typeof entry[1] === 'object')
    .map(([key, decision]) => ({ key, label: qualityDecisionLabel(key, decision), decision, statusLabel: String(decision.status ?? '').replace('_', ' ') }))
  const reviewArtifacts = deformationReviewArtifacts(artifact)
  const skeletonArtifacts = skeletonReviewArtifacts(artifact)
  const symmetry = bilateralSymmetrySummary(artifact)
  if (!decisions.length && !reviewArtifacts.length && !skeletonArtifacts.length && !symmetry) return null
  const symmetryLine = symmetry && <p className="symmetry-advisory">Bilateral symmetry p95: {symmetry.p95.toFixed(4)} (band {symmetry.band}) — {symmetry.pairsOutsideBand} pairs outside band{symmetry.worstPair ? ` (worst: ${symmetry.worstPair})` : ''}</p>
  const recover = (action: string) => {
    if (action === 'retry-generation') {
      if (sourceArtifact) onUseArtifact(sourceArtifact, 'generate', run, true)
      return
    }
    if (artifact) onUseArtifact(artifact, stageForQualityAction(action), run)
  }
  const cards = <>
    {decisions.map(({ key, label, decision, statusLabel }) => <article className={`quality-card ${decision.status ?? ''}`} key={key}>
      <div className="quality-title">{qualityDecisionPassed(decision.status) ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}<strong>{label}</strong><span>{statusLabel}</span></div>
      {(decision.findings ?? []).length > 0 && <div className="quality-findings">{(decision.findings ?? []).map((finding) => <div className={finding.severity} key={finding.code}><p>{finding.message}</p><small>{finding.metric}: <b>{formatSetting(finding.actual)}</b> · expected {finding.expected}</small></div>)}</div>}
      {qualityActionsForDisplay(decision.recommended_actions ?? [], compact).length > 0 && <div className="quality-actions">{qualityActionsForDisplay(decision.recommended_actions ?? [], compact).map((action) => action === 'manual-blender-cleanup' ? <a href={`${API}/assets/${artifact?.id}/download`} key={action}><Download size={13} /> Download for Blender</a> : action === 'inspect-deformation-review' ? <a href={`#deformation-review-${artifact?.id}`} key={action}><Eye size={13} /> Inspect deformation review</a> : (isAutomatedQualityAction(action) || action === 'render-deformation-review') && onExecuteAction ? <button disabled={Boolean(busyAction)} onClick={() => onExecuteAction(action)} key={action}><RotateCcw size={13} /> {busyAction === action ? 'Starting…' : `Run ${actionLabel(action)}`}</button> : <button disabled={Boolean(busyAction)} onClick={() => recover(action)} key={action}><Eye size={13} /> {actionLabel(action)}</button>)}</div>}
    </article>)}
  </>
  if (compact) return <details className="quality-compact"><summary>Quality evidence{decisions.length > 0 ? ` · ${decisions.map(({ label, statusLabel }) => `${label}: ${statusLabel}`).join(' · ')}` : ''}</summary>{decisions.length > 0 && <section className="quality-panels compact" aria-label="Quality gates">{cards}</section>}{symmetryLine}<DeformationReviewGallery artifact={artifact} /><SkeletonReviewGallery artifact={artifact} /></details>
  return <>
    {decisions.length > 0 && <section className="quality-panels" aria-label="Quality gates">{cards}</section>}
    {symmetryLine}
    <DeformationReviewGallery artifact={artifact} />
    <SkeletonReviewGallery artifact={artifact} />
  </>
}
