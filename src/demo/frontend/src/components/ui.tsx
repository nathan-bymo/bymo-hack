import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { errorText } from '../api/client'
import type { BatchInfo, ModelSummary, RecordedInfo } from '../api/types'
import { useI18n } from '../i18n'
import { S } from '../strings'

/* ---- confidenza: numero + icona + colore, mai solo colore (D.3) ---- */
export function bin(value: number, bins = { low_below: 0.6, high_from: 0.9 }): 'low' | 'mid' | 'high' {
  return value < bins.low_below ? 'low' : value >= bins.high_from ? 'high' : 'mid'
}
const CHIP = {
  high: { cls: 'bg-conf-high-bg text-conf-high', icon: '✓', word: 'high' },
  mid: { cls: 'bg-conf-mid-bg text-conf-mid', icon: '●', word: 'medium' },
  low: { cls: 'bg-conf-low-bg text-conf-low', icon: '▲', word: 'low' },
} as const

export function ConfidenceChip({ value, method, onClick }: { value: number | null; method?: string | null; onClick?: () => void }) {
  const { t } = useI18n()
  if (value === null || value === undefined)
    return <span className="rounded bg-toolbtn px-1.5 py-0.5 text-xs text-muted" title={t('No confidence available for this value')}>{t('n/a')}</span>
  const c = CHIP[bin(value)]
  const word = c.word === 'high' ? t('high') : c.word === 'medium' ? t('medium') : t('low')
  const tip = `${t('{word} confidence', { word })}${method ? ` — ${S.confidence[method] ?? method}` : ''}`
  return (
    <button type="button" onClick={onClick} title={tip} aria-label={t('Confidence {value} ({word})', { value: value.toFixed(2), word })}
      data-testid="confidence-chip"
      className={`inline-flex items-center gap-1 rounded border-0 px-1.5 py-0.5 text-xs font-semibold ${c.cls}`}>
      <span aria-hidden="true">{c.icon}</span>{value.toFixed(2)}
    </button>
  )
}

export function StatusPill({ status }: { status: 'ai_suggested' | 'reviewed' | 'corrected' | 'queued' | 'none' }) {
  if (status === 'none') return <span className="text-muted">—</span>
  const style = {
    ai_suggested: 'bg-toolbtn text-ink', reviewed: 'bg-conf-high-bg text-conf-high',
    corrected: 'bg-conf-mid-bg text-conf-mid', queued: 'bg-toolbtn text-muted',
  }[status]
  const label = status === 'queued' ? 'Queued' : S.status[status]
  return <span data-testid="status-pill" className={`whitespace-nowrap px-2 py-0.5 text-xs font-semibold ${style}`}>{label}</span>
}

export function recordedLabel(r: RecordedInfo | null | undefined): string {
  if (!r) return 'Recorded'
  const when = r.recorded_at ? new Date(r.recorded_at).toLocaleString('en-GB', { dateStyle: 'short', timeStyle: 'short' }) : ''
  return `Recorded ${when} on ${r.gpu_name ?? 'unknown GPU'}`.replace('  ', ' ')
}

export function ModeBadge({ mode, recorded, compact }: { mode: 'live' | 'recorded'; recorded?: RecordedInfo | null; compact?: boolean }) {
  return mode === 'recorded'
    ? <span data-testid="mode-badge" title={recordedLabel(recorded)} className="rounded bg-conf-mid-bg px-2 py-0.5 text-xs font-semibold text-conf-mid">{compact ? 'Recorded' : recordedLabel(recorded)}</span>
    : <span data-testid="mode-badge" className="rounded bg-conf-high-bg px-2 py-0.5 text-xs font-semibold text-conf-high">{S.badge.live}</span>
}

export type BadgeKind = 'held_out' | 'phase1_internal' | 'seen_in_training' | 'na' | 'reported_only' | 'recomputed' | 'offline_script' | 'kaggle_user'
export function MetricBadge({ kind }: { kind: BadgeKind }) {
  const cls: Record<BadgeKind, string> = {
    held_out: 'bg-toolbtn text-ink', phase1_internal: 'bg-conf-mid-bg text-conf-mid',
    seen_in_training: 'bg-toolbtn text-muted', na: 'bg-toolbtn text-muted',
    reported_only: 'bg-toolbtn text-muted', recomputed: 'bg-toolbtn text-ink', offline_script: 'bg-toolbtn text-ink',
    kaggle_user: 'bg-sky-100 text-blue-800',
  }
  return (
    <span data-testid={`badge-${kind}`} title={kind === 'seen_in_training' ? S.seenTooltip : undefined}
      className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${cls[kind]}`}>
      {S.badge[kind]}
    </span>
  )
}

export function ConfidenceMethodNote({ method }: { method: string | null | undefined }) {
  if (!method) return null
  return (
    <p className="m-0 text-xs text-muted">
      ⓘ Confidence: {S.confidence[method] ?? method}. {S.confidence.notComparable}
    </p>
  )
}

export function ThroughputBadge({ batch, target }: { batch: BatchInfo | null; target: number }) {
  if (!batch) return null
  const ips = batch.images_per_second
  const kind = batch.recorded ? 'recorded run' : batch.gpu_name
  return (
    <span data-testid="throughput" className="text-xs text-muted">
      {ips ? `${ips.toFixed(2)} img/s measured` : 'speed n/a'} · {kind}
      {batch.batch_size ? ` · batch ${batch.batch_size}` : ''} · target ≥ {target} img/s
      {ips !== null && ips >= target ? ' ✓' : ''}
    </span>
  )
}

export function ProgressBar({ done, total, label }: { done: number; total: number; label?: string }) {
  const pct = total ? Math.round((done / total) * 100) : 0
  return (
    <div className="flex items-center gap-2" data-testid="progress">
      <div className="h-2 w-48 overflow-hidden rounded bg-toolbtn" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-conf-high" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs">{done}/{total}{label ? ` · ${label}` : ''}</span>
    </div>
  )
}

/* ---- stati comuni ---- */
export function EmptyState({ children }: { children: ReactNode }) {
  return <div data-testid="empty-state" className="px-4 py-8 text-center text-muted">{children}</div>
}
export function ErrorState({ error, action }: { error: unknown; action?: ReactNode }) {
  return (
    <div role="alert" data-testid="error-state" className="m-4 rounded border border-invalid bg-invalid-bg p-3 text-invalid">
      {errorText(error)} {action}
    </div>
  )
}
export function modelUnavailableText(m: ModelSummary): string {
  if (m.status_reason?.includes('LLM_API_KEY')) return S.unavailable.key
  if (m.status_reason?.includes('GPU worker')) return S.unavailable.gpu
  return `Unavailable: ${m.status_reason ?? 'unknown reason'}`
}
export function ModelUnavailableState({ model }: { model: ModelSummary }) {
  return (
    <div data-testid="model-unavailable" className="rounded border border-line bg-toolbtn p-2 text-sm">
      {modelUnavailableText(model)}.{' '}
      <Link to="/image-classification/performance">How does this model perform on test sets →</Link>
    </div>
  )
}

/* ---- toast ---- */
interface ToastItem { id: number; kind: 'ok' | 'error'; text: string }
const ToastContext = createContext<(kind: 'ok' | 'error', text: string) => void>(() => {})
export const useToast = () => useContext(ToastContext)
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const push = useCallback((kind: 'ok' | 'error', text: string) => {
    const id = Date.now() + Math.random()
    setItems((x) => [...x, { id, kind, text }])
    window.setTimeout(() => setItems((x) => x.filter((i) => i.id !== id)), 5000)
  }, [])
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} data-testid="toast" role="status"
            className={`rounded px-3 py-2 text-sm shadow ${t.kind === 'ok' ? 'bg-conf-high-bg text-conf-high' : 'bg-invalid-bg text-invalid'}`}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function RobotIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" className="inline-block align-middle">
      <rect x="5" y="8" width="14" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="9.5" cy="13" r="1.3" fill="currentColor" /><circle cx="14.5" cy="13" r="1.3" fill="currentColor" />
      <line x1="12" y1="8" x2="12" y2="4.5" stroke="currentColor" strokeWidth="1.6" /><circle cx="12" cy="3.8" r="1.2" fill="currentColor" />
    </svg>
  )
}
