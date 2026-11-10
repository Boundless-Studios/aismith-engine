import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Bot, Check, ChevronDown, ChevronRight, Hand, UserRound } from 'lucide-react'
import { characterJourney, nextJourneyStep } from './journey'
import type { RunManifest, StudioTab } from './history'

type Props = { run: RunManifest | null; activeTab: StudioTab; onNavigate: (stepId: string) => void }

export default function CharacterJourney({ run, activeTab, onNavigate }: Props) {
  const steps = useMemo(() => characterJourney(run, activeTab), [run, activeTab])
  const next = nextJourneyStep(steps)
  // Never default the disclosure to the bare last step in the array — on a
  // direct-generate run nothing is active/checkpoint/failed yet, so that used
  // to land on Review while it's still "upcoming" and unreachable. Prefer the
  // actionable Next step, then the furthest actually-complete one.
  const lastComplete = [...steps].reverse().find((step) => step.state === 'complete')
  const suggested = steps.find((step) => ['active', 'checkpoint', 'failed'].includes(step.state))
    || next || lastComplete || steps[steps.length - 1]
  const [selectedId, setSelectedId] = useState(suggested.id)
  useEffect(() => setSelectedId(suggested.id), [suggested.id])
  const selected = steps.find((step) => step.id === selectedId) || suggested

  return <section className="character-journey" aria-label="Character pipeline">
    <div className="journey-steps">{steps.map((step, index) => <button disabled={step.state === 'upcoming' || step.state === 'not-applicable'} title={step.state === 'upcoming' ? step.blockedReason : undefined} className={`journey-step ${step.state} ${selected.id === step.id ? 'selected' : ''}`} onClick={() => { setSelectedId(step.id); onNavigate(step.id) }} key={step.id}>
      <span className="journey-index">{step.state === 'complete' || step.state === 'skipped' ? <Check size={13} /> : step.state === 'failed' ? <AlertTriangle size={13} /> : index + 1}</span>
      <span><strong>{step.label}{step.state === 'skipped' ? ' · skipped' : step.state === 'not-applicable' ? ' · n/a' : ''}</strong><small>{step.owner === 'agent' ? <Bot size={11} /> : step.owner === 'you' ? <UserRound size={11} /> : <Hand size={11} />}{step.owner}</small></span>
    </button>)}</div>
    {next && <button className="primary-action next-step-button" disabled={next.state === 'upcoming'} title={next.state === 'upcoming' ? next.blockedReason : undefined} onClick={() => { setSelectedId(next.id); onNavigate(next.id) }}>Next: {next.label}<ChevronRight size={14} /></button>}
    <details className={`journey-detail ${selected.state}`}>
      <summary><strong>{selected.label}</strong><ChevronDown size={14} /></summary>
      <div className="journey-detail-grid"><div><span>Agent does</span><p>{selected.agentAction}</p></div><div><span>You do</span><p>{selected.humanAction}</p></div><div><span>Done when</span><p>{selected.completion}</p></div></div>
      {selected.state === 'upcoming' && selected.blockedReason && <p className="journey-detail-blocked">{selected.blockedReason}</p>}
    </details>
  </section>
}
