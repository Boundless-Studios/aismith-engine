// Shared job-progress readout (percent bar + last log line + optional
// Cancel). Lives in its own module so both App.tsx and SkeletonWorkspace.tsx
// can render it without an App <-> SkeletonWorkspace circular import.
export default function JobProgress({ label, progress, logLine, onCancel }: { label: string; progress: number; logLine?: string; onCancel?: () => void }) {
  return <div className="motion-progress">
    <strong>{label}</strong>
    <div className="motion-progress-track"><i style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
    <small>{progress}% · {logLine || 'Working…'}</small>
    {onCancel && <button className="secondary-button cancel-task" onClick={onCancel}>Cancel</button>}
  </div>
}
