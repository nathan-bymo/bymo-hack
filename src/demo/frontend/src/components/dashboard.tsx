import { useMemo, useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ConfusionData, MetricResult, PerClassEntry } from '../api/types'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'
import { MetricBadge, type BadgeKind } from './ui'

export const MODEL_COLORS = ['#1d4ed8', '#047857', '#b45309', '#7c3aed', '#0e7490', '#be185d', '#4d7c0f']

export function CoverageWarningBanner() {
  return (
    <div role="note" data-testid="coverage-banner" className="rounded border border-conf-mid bg-conf-mid-bg px-3 py-2 text-sm text-conf-mid">
      ⚠ {S.bannerCoverage}
    </div>
  )
}

export function SourceFootnote({ source, set, trainedOn }: { source: string; set?: string; trainedOn?: string | null }) {
  return (
    <p className="m-0 mt-1 text-xs text-muted" data-testid="source-footnote">
      Source: {source}{set ? ` · set: ${set}` : ''}{trainedOn ? ` · training: ${trainedOn}` : ''}
    </p>
  )
}

export function ResultBadges({ result }: { result: Pick<MetricResult, 'validity' | 'provenance'> }) {
  return (
    <span className="inline-flex flex-wrap gap-1">
      <MetricBadge kind={result.validity as BadgeKind} />
      {result.provenance !== 'na' && <MetricBadge kind={result.provenance as BadgeKind} />}
    </span>
  )
}

/* ---- barre raggruppate: macro F1 per dimensione e modello + linea di riferimento ---- */
export interface BarSeries { id: string; name: string; color: string; values: Record<string, number | null> }
export function DimensionBarChart({ series, reference }: { series: BarSeries[]; reference: { value: number; label: string } | null }) {
  const data = DIMENSIONS.map((d) => ({ dimension: DIMENSION_NAMES[d], ...Object.fromEntries(series.map((s) => [s.id, s.values[d] ?? null])) }))
  return (
    <div data-testid="dimension-chart" style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="dimension" />
          <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : 'n/a')} />
          <Legend />
          {reference && (
            <ReferenceLine y={reference.value} stroke="#b91c1c" strokeDasharray="6 4"
              label={{ value: `${reference.value} ${reference.label} — other set`, position: 'insideTopRight', fill: '#b91c1c', fontSize: 11 }} />
          )}
          {series.map((s) => (
            <Bar key={s.id} dataKey={s.id} name={s.name} fill={s.color} isAnimationActive={false}>
              {data.map((_, i) => <Cell key={i} fill={s.color} />)}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ---- tabella per classe ---- */
const RARE = 30
export function PerClassTable({ perClass }: { perClass: Record<string, Record<string, PerClassEntry>> }) {
  const [dimension, setDimension] = useState(DIMENSIONS[0])
  const rows = Object.entries(perClass[dimension] ?? {}).sort((a, b) => (b[1].support ?? 0) - (a[1].support ?? 0))
  const fmt = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(2))
  return (
    <div data-testid="per-class">
      <div role="tablist" className="mb-2 flex flex-wrap gap-1">
        {DIMENSIONS.map((d) => (
          <button key={d} role="tab" aria-selected={d === dimension} type="button" onClick={() => setDimension(d)}
            className={`rounded border px-2 py-1 text-xs ${d === dimension ? 'border-link bg-imagetype font-semibold' : 'border-line bg-panel'}`}>
            {DIMENSION_NAMES[d]}
          </button>
        ))}
      </div>
      <table className="w-full max-w-2xl border-collapse text-left text-sm">
        <thead><tr className="border-b border-line text-xs text-muted"><th className="py-1">Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>
          {rows.map(([name, e]) => (
            <tr key={name} className="border-b border-line" data-rare={(e.support ?? 0) < RARE ? '1' : '0'}>
              <td className="py-0.5">{name}{(e.support ?? 0) < RARE && <span title={`support < ${RARE}`} className="ml-1 text-conf-mid">★</span>}</td>
              <td>{fmt(e.precision)}</td><td>{fmt(e.recall)}</td><td>{fmt(e.f1)}</td><td>{e.support ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="m-0 mt-1 text-xs text-muted">★ = support below {RARE} (the per-class target of the coverage set).</p>
    </div>
  )
}

/* ---- matrice di confusione ---- */
export function ConfusionMatrix({ data }: { data: ConfusionData }) {
  const [normalize, setNormalize] = useState(false)
  const rowSums = useMemo(() => data.matrix.map((r) => r.reduce((a, b) => a + b, 0)), [data])
  return (
    <div data-testid="confusion" className="overflow-x-auto">
      <label className="mb-2 block text-xs"><input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Normalise by row</label>
      <table className="border-collapse text-xs">
        <thead><tr><th className="p-1 text-left text-muted">label \ predicted</th>{data.columns.map((c) => <th key={c} className="p-1 font-normal" style={{ writingMode: 'vertical-rl' }}>{c}</th>)}</tr></thead>
        <tbody>
          {data.matrix.map((row, i) => (
            <tr key={data.labels[i]}>
              <th className="whitespace-nowrap p-1 text-left font-normal">{data.labels[i]}</th>
              {row.map((v, j) => {
                const share = rowSums[i] ? v / rowSums[i] : 0
                const intensity = normalize ? share : Math.min(1, v / Math.max(1, ...data.matrix.flat()))
                return (
                  <td key={j} title={`${data.labels[i]} → ${data.columns[j]}: ${v} (${(share * 100).toFixed(0)}% of row)`}
                    className="h-7 w-9 text-center" style={{ background: `rgba(29,78,216,${intensity.toFixed(2)})`, color: intensity > 0.5 ? '#fff' : 'var(--c-text)' }}>
                    {normalize ? (share ? share.toFixed(2) : '') : v || ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Section({ title, children, testId }: { title: string; children: ReactNode; testId?: string }) {
  return (
    <section className="mt-4" data-testid={testId}>
      <h2 className="section-title m-0 mb-2 text-sm">▼ {title}</h2>
      {children}
    </section>
  )
}
