import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useModels } from '../api/hooks'
import type { KaggleReference, ModelSummary } from '../api/types'
import { DemoTabs } from '../components/nav'
import { ErpShell, TitleBar } from '../components/shell'
import { ErrorState } from '../components/ui'
import { SCORE_SET_LABEL, SET_INFO, availability, bestSpeed, headlineScore, memoryText, usd, weightsDetail, weightsText } from '../lib/models'

const BOARDS = ['kaggle_dev', 'prompt-test', 'coverage'] as const

const VALIDITY: Record<string, { label: string; cls: string; tip: string }> = {
  kaggle_public: { label: 'Kaggle public', cls: 'bg-sky-100 text-blue-800', tip: 'Score of a team submission on the public dev leaderboard.' },
  held_out: { label: 'held-out', cls: 'bg-conf-high-bg text-conf-high', tip: 'Never seen in training or for any choice.' },
  phase1_internal: { label: 'internal', cls: 'bg-conf-mid-bg text-conf-mid', tip: 'Internal phase-1 set: valid for comparing methods on the same set.' },
  reference: { label: 'reference', cls: 'bg-toolbtn text-muted', tip: 'Organisers’ reference on the same leaderboard.' },
}

function ValidityChip({ kind }: { kind: string }) {
  const v = VALIDITY[kind] ?? { label: kind, cls: 'bg-toolbtn text-muted', tip: '' }
  return <span title={v.tip} className={`whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-semibold ${v.cls}`}>{v.label}</span>
}

interface BoardRow { key: string; name: string; score: number; validity: string; source: string; ours: boolean; highlight: boolean; secondary: boolean }

function ScoreBoard({ set, models, references, showSecondary }: {
  set: (typeof BOARDS)[number]; models: ModelSummary[]; references: KaggleReference[]; showSecondary: boolean
}) {
  const info = SET_INFO[set]
  const rows: BoardRow[] = models.flatMap((m) => {
    const s = m.scores.find((x) => x.set === set && x.macro_f1 !== null)
    if (!s || (!m.featured && !showSecondary)) return []
    return [{ key: m.model_id, name: m.short_name, score: s.macro_f1!, validity: s.validity, source: s.source_label ? `${s.source_label} · ${s.source}` : s.source, ours: true, highlight: m.model_id === models[0]?.model_id, secondary: !m.featured }]
  })
  if (set === 'kaggle_dev')
    rows.push(...references.map((r) => ({ key: r.name, name: r.name, score: r.score, validity: 'reference', source: r.source, ours: false, highlight: false, secondary: false })))
  rows.sort((a, b) => b.score - a.score)
  const invalid = models.filter((m) => m.featured && m.seen_sets.includes(set) && !m.scores.some((x) => x.set === set))
  const noScore = set === 'kaggle_dev' ? [] : models.filter((m) => m.featured && !m.seen_sets.includes(set) && !m.scores.some((x) => x.set === set))
  const min = 0.4
  return (
    <section className="flex flex-col border border-line bg-panel" data-testid={`board-${set}`}>
      <header className="border-b border-line px-3 py-2" style={{ borderTop: '3px solid var(--c-panel-frame)' }}>
        <h2 className="m-0 text-sm font-bold text-sect">{info.title}</h2>
        <p className="m-0 text-xs text-muted">{info.images} · macro F1 · {info.what}</p>
      </header>
      <ol className="m-0 flex-1 list-none space-y-1.5 p-3">
        {rows.map((r, i) => (
          <li key={r.key} data-testid="board-row" title={`Source: ${r.source}`} className={r.secondary ? 'opacity-70' : ''}>
            <div className="flex items-baseline gap-2 text-xs">
              <span className="w-4 text-right text-muted">{i + 1}</span>
              <span className={`flex-1 truncate ${r.ours ? 'font-semibold text-ink' : 'text-muted'}`}>{r.name}</span>
              <ValidityChip kind={r.validity} />
              <span className={`w-12 text-right font-mono text-sm ${r.highlight ? 'font-bold' : ''}`}>{r.score.toFixed(3)}</span>
            </div>
            <div className="ml-6 mt-0.5 h-2 rounded-sm bg-app">
              <div className="h-full rounded-sm" style={{
                width: `${Math.max(2, ((r.score - min) / (1 - min)) * 100)}%`,
                background: r.ours ? (r.highlight ? 'var(--c-titlebar-bg)' : '#5b8a74') : '#b8bcc4',
              }} />
            </div>
          </li>
        ))}
      </ol>
      {(invalid.length > 0 || noScore.length > 0) && (
        <footer className="space-y-0.5 border-t border-line px-3 py-2 text-xs">
          {invalid.map((m) => (
            <p key={m.model_id} className="m-0 text-invalid" data-testid="board-invalid">⛔ {m.short_name}: seen during training — no valid score here</p>
          ))}
          {noScore.map((m) => <p key={m.model_id} className="m-0 text-muted">— {m.short_name}: not measured on this set</p>)}
        </footer>
      )}
      <p className="m-0 border-t border-line px-3 py-1 text-[11px] text-muted">Bars start at 0.40. Hover a row for its source file.</p>
    </section>
  )
}

function ModelsTable({ models, showSecondary }: { models: ModelSummary[]; showSecondary: boolean }) {
  const list = models.filter((m) => m.featured || showSecondary)
  return (
    <div className="overflow-x-auto border border-line bg-panel">
      <table className="w-full border-collapse text-left text-xs" data-testid="models-table">
        <thead>
          <tr className="border-b border-line bg-app text-muted" style={{ fontSize: 'var(--fs-label)' }}>
            <th className="px-2 py-1.5">Model</th>
            <th className="px-2 py-1.5">Best evidence</th>
            <th className="px-2 py-1.5">Other scores</th>
            <th className="px-2 py-1.5">Speed</th>
            <th className="px-2 py-1.5">GPU memory</th>
            <th className="px-2 py-1.5">Cost / 100,000 images</th>
            <th className="px-2 py-1.5">Trained on</th>
            <th className="px-2 py-1.5">Weights</th>
            <th className="px-2 py-1.5">In this demo</th>
          </tr>
        </thead>
        <tbody>
          {list.map((m) => {
            const head = headlineScore(m)
            const others = m.scores.filter((s) => s !== head && s.macro_f1 !== null)
            const speed = bestSpeed(m)
            const av = availability(m)
            return (
              <tr key={m.model_id} className={`border-b border-line align-top ${m.featured ? '' : 'text-muted'}`} data-testid="models-row" data-model={m.model_id}>
                <td className="min-w-48 px-2 py-1.5" title={m.display_name}><div className="font-semibold text-ink">{m.short_name}</div></td>
                <td className="whitespace-nowrap px-2 py-1.5">
                  {head ? <><span className="font-mono text-sm font-bold">{head.macro_f1!.toFixed(3)}</span> <span className="text-muted">on</span> <b>{SCORE_SET_LABEL[head.set] ?? head.set}</b><div className="mt-0.5"><ValidityChip kind={head.validity} /></div></> : <span className="text-muted">no valid score</span>}
                </td>
                <td className="px-2 py-1.5">
                  {others.length ? others.map((s) => <div key={s.set} className="whitespace-nowrap">{SCORE_SET_LABEL[s.set] ?? s.set}: <span className="font-mono">{s.macro_f1!.toFixed(3)}</span></div>) : <span className="text-muted">—</span>}
                  {m.seen_sets.includes('coverage') && <div className="text-invalid" title={m.contamination_note ?? ''}>coverage: seen in training</div>}
                </td>
                <td className="px-2 py-1.5">
                  {m.speeds.length ? m.speeds.map((s, i) => (
                    <div key={i} className="whitespace-nowrap" title={`${s.what}${s.conditions ? ` · ${s.conditions}` : ''} · ${s.source}`}>
                      <span className="font-mono">{s.images_per_second}</span> img/s · {s.gpu_label?.replace(' (HF a100-large)', '').replace(' (HF)', '')}{s.what.includes('solo generate') ? ' (generate only)' : ''}
                    </div>
                  )) : <span className="text-muted">not measured</span>}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5" title={m.memory ? `${m.memory.what} · ${m.memory.source}` : undefined}>{memoryText(m)}</td>
                <td className="min-w-28 px-2 py-1.5">
                  {speed?.cost_per_100k_usd != null
                    ? <><span className="font-mono text-sm font-bold">{usd(speed.cost_per_100k_usd)}</span><div className="text-[11px] text-muted">{speed.gpu_label} at ${speed.usd_per_hour}/h</div></>
                    : <span className="text-muted">n/a{speed?.cost_note ? ` — ${speed.cost_note}` : ''}</span>}
                </td>
                <td className="min-w-36 px-2 py-1.5">{m.training_label ?? 'n/a'}</td>
                <td className="min-w-32 px-2 py-1.5" title={weightsDetail(m)}>{weightsText(m)}</td>
                <td className="whitespace-nowrap px-2 py-1.5" title={m.status_reason ?? undefined}>
                  <span className={`rounded px-1.5 py-0.5 font-semibold ${av.tone === 'live' ? 'bg-conf-high-bg text-conf-high' : av.tone === 'recorded' ? 'bg-conf-mid-bg text-conf-mid' : 'bg-toolbtn text-muted'}`}>{av.text}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelsPage() {
  const models = useModels()
  const [showSecondary, setShowSecondary] = useState(false)
  const list = models.data?.models ?? []
  const refs = models.data?.kaggle_references ?? []
  const target = models.data?.throughput_target_images_per_second ?? 2
  return (
    <ErpShell>
      <TitleBar title="Apertus models — scores, speed and cost" />
      <DemoTabs />
      <div className="space-y-4 p-4">
        {models.isError && <ErrorState error={models.error} />}
        <div role="note" data-testid="sets-note" className="rounded border border-conf-mid bg-conf-mid-bg px-3 py-2 text-sm text-conf-mid">
          ⚠ One leaderboard per evaluation set. Scores from different sets are <b>not comparable</b> and are never ranked together:
          only “Kaggle dev” is the official leaderboard shared with the organisers’ references.
        </div>
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label><input type="checkbox" checked={showSecondary} onChange={(e) => setShowSecondary(e.target.checked)} data-testid="show-secondary" /> show secondary experiments (P10, P8, P9)</label>
          <Link to="/image-classification/try">Try a model on one image →</Link>
        </div>
        <div className="grid gap-3 xl:grid-cols-3">
          {BOARDS.map((s) => <ScoreBoard key={s} set={s} models={list} references={refs} showSecondary={showSecondary} />)}
        </div>
        <section>
          <h2 className="section-title m-0 mb-2">All models: speed, memory and cost</h2>
          <ModelsTable models={list} showSecondary={showSecondary} />
          <p className="m-0 mt-1 text-xs text-muted">
            Cost per 100,000 images = GPU price per hour ÷ (images per second × 3,600) × 100,000, only from end-to-end speeds
            measured on a known GPU (prices in models.yaml; Hugging Face list prices). Speeds were measured on different GPUs: compare
            the cost column, not the img/s column. Challenge target ≥ {target} img/s; cost per 100,000 images weighs 20% of the score.
            Zero-shot models run on the shared Keera gateway, whose price is not public.
          </p>
        </section>
      </div>
    </ErpShell>
  )
}
