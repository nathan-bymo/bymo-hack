import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'
import { useModels } from '../api/hooks'
import type { MetricBlock, ModelSummary } from '../api/types'
import { AppShell } from '../components/app'
import ExpertScores from '../components/ExpertScores'
import { QUESTIONS } from '../lib/labels'
import { bestSpeed, memoryText, sortForShow, usd } from '../lib/models'
import { DIMENSIONS } from '../strings'

interface Row { key: string; name: string; note: string; score: number; ours: boolean }

/** Barre orizzontali di un solo set: i numeri di set diversi non finiscono mai nello stesso grafico. */
function ScoreChart({ rows, testId }: { rows: Row[]; testId: string }) {
  const min = 0.4
  const sorted = [...rows].sort((a, b) => b.score - a.score)
  return (
    <ol className="m-0 mt-6 list-none space-y-3 p-0" data-testid={testId}>
      {sorted.map((r) => (
        <li key={r.key} data-testid="board-row" className="grid grid-cols-[minmax(0,15rem)_minmax(0,1fr)_4.5rem] items-center gap-4">
          <span className="leading-tight">
            <span className={`block text-[16px] ${r.ours ? 'font-bold' : 'text-slate'}`}>{r.name}</span>
            <span className="block text-[13px] text-slate">{r.note}</span>
          </span>
          <span className="h-3 bg-canvas">
            <span className={`grow block h-full ${r.ours ? 'bg-graphite' : 'bg-bar-alt'}`} style={{ width: `${((r.score - min) / (1 - min)) * 100}%` }} />
          </span>
          <span className={`text-right text-[18px] ${r.ours ? 'font-bold' : 'text-slate'}`}>{r.score.toFixed(3)}</span>
        </li>
      ))}
    </ol>
  )
}

function scoreOn(m: ModelSummary, set: string) {
  return m.scores.find((s) => s.set === set && s.macro_f1 !== null)?.macro_f1 ?? null
}

/** In quale gruppo di valutazione cade un punteggio: la challenge confronta per gruppi, non per decimali. */
function groupSentence(best: Row | undefined, flash: Row | undefined, qwen: Row | undefined, argon: Row | undefined): string | null {
  if (!best || !flash || !qwen) return null
  if (best.score > qwen.score) return `${best.name} is in the top assessment group: above the organisers’ fine-tuned Qwen (${qwen.score.toFixed(3)}).`
  if (best.score > flash.score) {
    const argonPart = argon && best.score > argon.score ? `above both Gemini references (${argon.score.toFixed(3)} and ${flash.score.toFixed(3)})` : `above Gemini 3.8 Flash (${flash.score.toFixed(3)})`
    return `${best.name} (${best.score.toFixed(3)}) is in the middle assessment group: ${argonPart}, below the organisers’ fine-tuned Qwen (${qwen.score.toFixed(3)}).`
  }
  return `${best.name} is below Gemini 3.8 Flash (${flash.score.toFixed(3)}).`
}

/** Le sei domande una per una, sullo stesso set held-out: dove ogni modello è forte e dove no. */
function PerQuestion({ models }: { models: ModelSummary[] }) {
  const ids = models.map((m) => m.model_id).join(',')
  const metrics = useQuery({
    queryKey: ['metrics', 'prompt-test', ids], enabled: !!ids,
    queryFn: () => api<{ models: Record<string, MetricBlock> }>(`/metrics/compare?models=${ids}&set=prompt-test&include_invalid=false`),
  })
  const result = (m: ModelSummary) => metrics.data?.models[m.model_id]?.results.find((r) => r.validity !== 'seen_in_training')
  const cols = models.filter((m) => result(m))
  if (!cols.length) return <p className="m-0 mt-5 text-slate">{metrics.isLoading ? 'Loading the scores…' : 'No scores on this test yet.'}</p>

  const rows: { key: string; label: string; sub?: string; value: (m: ModelSummary) => number | null | undefined; strong?: boolean }[] = [
    ...DIMENSIONS.map((d) => ({ key: d, label: QUESTIONS[d].question, sub: d === 'perspective' ? 'Bonus question in the challenge' : undefined, value: (m: ModelSummary) => result(m)?.per_dimension[d]?.macro_f1 })),
    { key: 'macro', label: 'Average of the six (macro F1)', sub: 'The challenge metric', value: (m) => result(m)?.macro_f1, strong: true },
    { key: 'weighted', label: 'Weighted F1', sub: 'Frequent classes count more, as in the shop', value: (m) => result(m)?.weighted_f1, strong: true },
  ]
  return (
    <div className="mt-6 overflow-x-auto">
      <table className="w-full border-collapse text-left" data-testid="per-question">
        <thead>
          <tr className="border-b-2 border-graphite text-[13px] text-slate">
            <th className="py-2 pr-4 font-normal">Question</th>
            {cols.map((m) => <th key={m.model_id} className="py-2 pl-4 text-right font-normal"><span className="font-bold text-graphite">{m.short_name}</span></th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const values = cols.map((m) => { const v = r.value(m); return v == null ? null : Math.round(v * 1000) / 1000 })
            const top = Math.max(...values.map((v) => v ?? -1))   // confronto sul valore mostrato: i pari merito sono tutti in grassetto
            return (
              <tr key={r.key} data-testid={`pq-${r.key}`} className={`align-baseline ${r.key === 'macro' ? 'border-t-2 border-graphite' : 'border-b border-rule'}`}>
                <td className="py-2.5 pr-4">
                  <span className={`block text-[16px] ${r.strong ? 'font-bold' : ''}`}>{r.label}</span>
                  {r.sub && <span className="block text-[13px] text-slate">{r.sub}</span>}
                </td>
                {values.map((v, i) => (
                  <td key={cols[i].model_id} className={`py-2.5 pl-4 text-right text-[18px] tabular-nums ${v === top ? 'font-bold' : 'text-slate'}`}>{v == null ? '—' : v.toFixed(3)}</td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function ResultsPage() {
  const models = useModels()
  const [details, setDetails] = useState(false)
  const list = sortForShow(models.data?.models ?? []).filter((m) => m.featured)
  const refs = models.data?.kaggle_references ?? []
  const target = models.data?.throughput_target_images_per_second ?? 2

  const kaggleRows: Row[] = [
    ...list.flatMap((m) => { const s = scoreOn(m, 'kaggle_dev'); return s === null ? [] : [{ key: m.model_id, name: m.short_name, note: m.runtime === 'none' ? 'Our best Kaggle submission so far' : 'Our Apertus model', score: s, ours: true }] }),
    ...refs.map((r) => ({ key: r.name, name: r.name.replace(" (organisers' baseline)", ''), note: r.name.includes('Qwen') ? 'Organisers’ own fine-tuned model' : 'Organisers’ reference, no training', score: r.score, ours: false })),
  ]
  const bestOurs = kaggleRows.filter((r) => r.ours).sort((a, b) => b.score - a.score)[0]
  const group = groupSentence(bestOurs, kaggleRows.find((r) => r.name.includes('3.8 Flash')), kaggleRows.find((r) => r.name.includes('Qwen')), kaggleRows.find((r) => r.name.includes('Argon')))
  const heldOut = list.filter((m) => scoreOn(m, 'prompt-test') !== null)
  const costRows = list.filter((m) => m.speeds.length > 0)

  return (
    <AppShell>
      <h1>How good, how fast, how cheap</h1>
      <p className="lead mt-3">
        The numbers behind the demo. Every score comes from a report in our repository, and scores from different test sets are
        never ranked together.
      </p>

      <section className="mt-12">
        <h2>Score on the official leaderboard</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Kaggle dev set: 5,261 product images labelled by Galaxus, scored with the challenge metric (macro F1, higher is better).
          Black bars are our Apertus models; grey bars are the organisers’ references.
        </p>
        <ScoreChart rows={kaggleRows} testId="board-kaggle_dev" />
        {group && <p className="m-0 mt-4 max-w-[70ch] text-[15px]" data-testid="assessment-group">{group}</p>}
      </section>

      <section className="mt-16">
        <h2>Question by question</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Our held-out test: 1,193 labelled images that none of these models saw while learning. The Apertus models without
          training were not submitted to Kaggle, so this is where all our models can be compared. Bold is the best in each row.
        </p>
        <PerQuestion models={heldOut} />
      </section>

      <section className="mt-16">
        <h2>Speed and cost</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Measured from image file to the six answers: trained models on one GPU, the others through the Hack Apertus API.
          The challenge asks for at least {target} images per second (✓) and more than 100,000 images a day.
        </p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left" data-testid="models-table">
            <thead>
              <tr className="border-b-2 border-graphite text-[13px] text-slate">
                <th className="py-2 pr-4 font-normal">Model</th>
                <th className="py-2 pr-4 text-right font-normal">Images per second</th>
                <th className="py-2 pr-4 text-right font-normal">100,000 images take</th>
                <th className="py-2 pr-4 font-normal">GPU</th>
                <th className="py-2 pr-4 text-right font-normal">Cost per 100,000 images</th>
                <th className="py-2 font-normal">GPU memory</th>
              </tr>
            </thead>
            <tbody>
              {costRows.map((m) => {
                const s = bestSpeed(m)
                const zeroShot = m.runtime === 'remote_openai'
                return (
                  <tr key={m.model_id} className="border-b border-rule align-baseline" data-testid="models-row" data-model={m.model_id}>
                    <td className="py-3 pr-4 text-[16px] font-bold">{m.short_name}</td>
                    <td className="py-3 pr-4 text-right text-[18px] tabular-nums">
                      {s?.images_per_second}
                      {s && <span className={`ml-1.5 text-[13px] ${s.images_per_second >= target ? 'text-slate' : 'font-bold text-signal'}`}>{s.images_per_second >= target ? '✓' : `below ${target}`}</span>}
                    </td>
                    <td className="py-3 pr-4 text-right text-[16px] tabular-nums" data-testid="hours-100k">{s ? `${(100_000 / s.images_per_second / 3600).toFixed(1)} h` : '—'}</td>
                    <td className="py-3 pr-4 text-[15px] text-slate">{zeroShot ? 'Hack Apertus API' : s?.gpu_label?.replace(' (HF a100-large)', '').replace(' (HF)', '')}</td>
                    <td className="py-3 pr-4 text-right text-[18px] font-bold tabular-nums">
                      {s?.cost_per_100k_usd != null ? usd(s.cost_per_100k_usd) : <span className="text-[14px] font-normal text-slate">{zeroShot ? 'API: no GPU of ours' : '—'}</span>}
                    </td>
                    <td className="py-3 text-[15px] text-slate">{memoryText(m)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="m-0 mt-3 max-w-[70ch] text-[13px] text-slate">
          Cost = GPU price per hour ÷ images per hour × 100,000, with Hugging Face list prices (A100 $2.50/h, L40S $1.80/h, H200 $5/h).
          Where a model was measured on two GPUs, the cheaper one is shown; the mixed model is shown in its delivery setup on an H100, whose price is not in this list. Every model gets through 100,000 images in less than a day.
        </p>
      </section>

      <section className="mt-16 border-t-2 border-graphite pt-4">
        <button type="button" onClick={() => setDetails((d) => !d)} aria-expanded={details} data-testid="toggle-details"
          className="flex w-full items-baseline justify-between border-0 bg-transparent p-0 text-left">
          <h2>Every class, every test set</h2>
          <span className="text-[15px] text-slate">{details ? 'Hide' : 'Show'}</span>
        </button>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">For experts: scores per class, a confusion matrix for perspective, and our other internal tests.</p>
        {details && <div className="mt-6"><ExpertScores /></div>}
      </section>
    </AppShell>
  )
}
