import { useMemo, useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ConfusionData, MetricResult, PerClassEntry } from '../api/types'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'
import { MetricBadge, type BadgeKind } from './ui'

export const MODEL_COLORS = ['#222222', '#6b7079', '#a3a8b0', '#cfd3d8', '#45494f', '#8a9099', '#babec4']

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
          <CartesianGrid vertical={false} stroke="#dfe2e5" />
          <XAxis dataKey="dimension" tick={{ fontSize: 14, fill: '#222' }} axisLine={{ stroke: '#222' }} tickLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} tick={{ fontSize: 13, fill: '#646a73' }} axisLine={false} tickLine={false} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : 'n/a')} />
          <Legend iconType="square" wrapperStyle={{ fontSize: 14, color: '#222' }} />
          {reference && (
            <ReferenceLine y={reference.value} stroke="#646a73" strokeDasharray="6 4"
              label={{ value: `${reference.value} ${reference.label} — other set`, position: 'insideTopRight', fill: '#646a73', fontSize: 12 }} />
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
            className={`border px-3 py-1.5 text-[14px] ${d === dimension ? 'border-graphite bg-graphite font-bold text-white' : 'border-rule bg-white text-graphite hover:border-graphite'}`}>
            {DIMENSION_NAMES[d]}
          </button>
        ))}
      </div>
      <table className="w-full max-w-3xl border-collapse text-left text-[15px]">
        <thead><tr className="border-b-2 border-graphite text-[13px] text-slate"><th className="py-2 font-normal">Class</th><th className="font-normal">Precision</th><th className="font-normal">Recall</th><th className="font-normal">F1</th><th className="font-normal">Images</th></tr></thead>
        <tbody>
          {rows.map(([name, e]) => (
            <tr key={name} className="border-b border-rule" data-rare={(e.support ?? 0) < RARE ? '1' : '0'}>
              <td className="py-1.5">{name.replace(/_/g, ' ')}{(e.support ?? 0) < RARE && <span title={`fewer than ${RARE} images`} className="ml-1 text-slate">*</span>}</td>
              <td>{fmt(e.precision)}</td><td>{fmt(e.recall)}</td><td className="font-bold">{fmt(e.f1)}</td><td>{e.support ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="m-0 mt-2 text-[13px] text-slate">* fewer than {RARE} images of this class: the number moves a lot with a few mistakes.</p>
    </div>
  )
}

/* ---- matrice di confusione ---- */
export function ConfusionMatrix({ data }: { data: ConfusionData }) {
  const [normalize, setNormalize] = useState(false)
  const rowSums = useMemo(() => data.matrix.map((r) => r.reduce((a, b) => a + b, 0)), [data])
  return (
    <div data-testid="confusion" className="overflow-x-auto">
      <label className="mb-3 block text-[14px]"><input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Show shares of each row</label>
      <table className="border-collapse text-[13px]">
        <thead><tr><th className="p-1 text-left font-normal text-slate">true view ↓ / answer →</th>{data.columns.map((c) => <th key={c} className="p-1 font-normal" style={{ writingMode: 'vertical-rl' }}>{c}</th>)}</tr></thead>
        <tbody>
          {data.matrix.map((row, i) => (
            <tr key={data.labels[i]}>
              <th className="whitespace-nowrap p-1 text-left font-normal">{data.labels[i]}</th>
              {row.map((v, j) => {
                const share = rowSums[i] ? v / rowSums[i] : 0
                const intensity = normalize ? share : Math.min(1, v / Math.max(1, ...data.matrix.flat()))
                return (
                  <td key={j} title={`${data.labels[i]} → ${data.columns[j]}: ${v} (${(share * 100).toFixed(0)}% of row)`}
                    className="h-7 w-9 text-center" style={{ background: `rgba(34,34,34,${intensity.toFixed(2)})`, color: intensity > 0.5 ? '#fff' : 'var(--c-text)' }}>
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
