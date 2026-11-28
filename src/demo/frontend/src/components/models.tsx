import type { ModelSummary } from '../api/types'

const DOT: Record<string, string> = {
  ready: '#047857', not_loaded: '#2563eb', loading: '#b45309', unloading: '#b45309', error: '#b91c1c', unavailable: '#94a3b8',
}

export function statusDot(m: ModelSummary, mode: 'live' | 'recorded') {
  if (mode === 'recorded' && m.modes.includes('recorded')) return '◆'
  return m.modes.includes('live') ? '●' : '○'
}

export function ModelSelect({ models, value, onChange, id = 'model-select' }: {
  models: ModelSummary[]; value: string | null; onChange: (id: string) => void; id?: string
}) {
  return (
    <label className="inline-flex items-center gap-1 text-sm">
      <span className="text-muted">Model:</span>
      <select id={id} data-testid="model-select" value={value ?? ''} onChange={(e) => onChange(e.target.value)}
        className="border border-graphite bg-white px-3 py-1.5 text-[15px]">
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.short_name}
            {m.modes.includes('live') ? ` — ${m.status === 'unavailable' ? 'ready' : m.status.replace('_', ' ')}` : m.modes.includes('recorded') ? ' — recorded' : ' — unavailable'}
          </option>
        ))}
      </select>
    </label>
  )
}
export { DOT }
