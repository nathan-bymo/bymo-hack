import { useState } from 'react'
import { useModels } from '../api/hooks'
import type { ModelSummary } from '../api/types'
import { AppShell } from '../components/app'
import ExpertScores from '../components/ExpertScores'
import { DID_NOT_HELP, MATTERED, type Insight } from '../lib/insights'
import { bestSpeed, memoryText, sortForShow, usd } from '../lib/models'

interface Row { key: string; name: string; note: string; score: number; ours: boolean }

function InsightList({ items, testId }: { items: Insight[]; testId: string }) {
  return (
    <ul className="m-0 mt-5 grid list-none gap-x-12 gap-y-6 p-0 md:grid-cols-2" data-testid={testId}>
      {items.map((i) => (
        <li key={i.title} className="border-t border-graphite pt-3">
          <h3 className="m-0 text-[17px] font-bold leading-snug tracking-tight">{i.title}</h3>
          <p className="m-0 mt-1.5 text-[15px] leading-relaxed">{i.body}</p>
          <p className="m-0 mt-1.5 text-[12px] text-slate">Source: {i.source}</p>
        </li>
      ))}
    </ul>
  )
}

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

export default function OverviewPage() {
  const models = useModels()
  const [showMore, setShowMore] = useState(false)
  const [details, setDetails] = useState(false)
  const list = sortForShow(models.data?.models ?? []).filter((m) => m.featured || showMore)
  const refs = models.data?.kaggle_references ?? []
  const target = models.data?.throughput_target_images_per_second ?? 2

  const kaggleRows: Row[] = [
    ...list.flatMap((m) => { const s = scoreOn(m, 'kaggle_dev'); return s === null ? [] : [{ key: m.model_id, name: m.short_name, note: 'Our Apertus model', score: s, ours: true }] }),
    ...refs.map((r) => ({ key: r.name, name: r.name.replace(" (organisers' baseline)", ''), note: r.name.includes('Qwen') ? 'Organisers’ own fine-tuned model' : 'Organisers’ reference, no training', score: r.score, ours: false })),
  ]
  const testRows: Row[] = list.flatMap((m) => {
    const s = scoreOn(m, 'prompt-test')
    const w = m.scores.find((x) => x.set === 'prompt-test')?.weighted_f1
    const kind = m.training_label === 'no training (zero-shot)' ? 'No training' : 'Trained by the team'
    return s === null ? [] : [{ key: m.model_id, name: m.short_name, note: w != null ? `${kind}, weighted F1 ${w.toFixed(3)}` : kind, score: s, ours: true }]
  })
  const notOnLeaderboard = list.filter((m) => scoreOn(m, 'kaggle_dev') === null && scoreOn(m, 'prompt-test') !== null)
  const costRows = list.filter((m) => m.speeds.length > 0)

  return (
    <AppShell>
      <h1>How good, how fast, how cheap</h1>
      <p className="lead mt-3">
        We tried several ways to teach Apertus to label product images: using it as it comes, adding a light adapter, retraining it
        completely, and adding an image encoder. Here is how they compare.
      </p>

      <section className="mt-14">
        <h2>Score on the official leaderboard</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Kaggle dev set: 5,261 product images labelled by Galaxus, scored with the challenge metric (macro F1, higher is better).
          Black bars are our Apertus models; grey bars are the organisers’ references.
        </p>
        <ScoreChart rows={kaggleRows} testId="board-kaggle_dev" />
      </section>

      <section className="mt-16">
        <h2>Speed and cost</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Measured from image file to the six answers: trained models on one GPU, the others through the shared Apertus gateway. The challenge asks for at least {target} images per second (✓) and
          more than 100,000 images a day: every model gets through them in less than a day.
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
                    <td className="py-3 pr-4 text-right text-[18px]">
                      {s?.images_per_second}
                      {s && <span className={`ml-1.5 text-[13px] ${s.images_per_second >= target ? 'text-slate' : 'font-bold text-signal'}`}>{s.images_per_second >= target ? '✓' : `below ${target}`}</span>}
                    </td>
                    <td className="py-3 pr-4 text-right text-[16px]" data-testid="hours-100k">{s ? `${(100_000 / s.images_per_second / 3600).toFixed(1)} h` : '—'}</td>
                    <td className="py-3 pr-4 text-[15px] text-slate">{zeroShot ? 'Shared Apertus gateway' : s?.gpu_label?.replace(' (HF a100-large)', '').replace(' (HF)', '')}</td>
                    <td className="py-3 pr-4 text-right text-[18px] font-bold">
                      {s?.cost_per_100k_usd != null ? usd(s.cost_per_100k_usd) : <span className="text-[14px] font-normal text-slate">{zeroShot ? 'gateway price not public' : 'not measured end to end'}</span>}
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
          Where a model was measured on two GPUs, the cheaper one is shown.
        </p>
      </section>

      <section className="mt-16">
        <h2>Our own test</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          1,193 labelled images that these models never saw while learning. They are different images from the leaderboard, so
          compare these numbers only with each other. The bar is the macro F1, where rare classes count as much as frequent ones;
          the weighted F1 under each name follows how often each class appears in the shop.
          {notOnLeaderboard.length > 0 && ' The Apertus models without training were not submitted to Kaggle: this is where they appear.'}
        </p>
        <ScoreChart rows={testRows} testId="board-prompt-test" />
      </section>

      <label className="mt-6 block text-[14px] text-slate"><input type="checkbox" checked={showMore} onChange={(e) => setShowMore(e.target.checked)} data-testid="show-secondary" /> Also show early experiments</label>

      <section className="mt-16">
        <h2>What mattered</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">The changes that moved the score most, measured on the same images before and after.</p>
        <InsightList items={MATTERED} testId="insights-mattered" />
      </section>

      <section className="mt-14">
        <h2>What did not help</h2>
        <InsightList items={DID_NOT_HELP} testId="insights-not" />
      </section>

      <section className="mt-16" data-testid="runtime-models">
        <h2>What runs at inference</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Only Apertus and open-weights helper models: no closed model and no commercial API. The trained models run on a
          single GPU, so they can be installed on-premises, on an air-gapped server or in a Swiss cloud.
        </p>
        <table className="mt-5 w-full border-collapse text-left">
          <thead>
            <tr className="border-b-2 border-graphite text-[13px] text-slate">
              <th className="py-2 pr-4 font-normal">Model</th><th className="py-2 pr-4 font-normal">Role</th><th className="py-2 font-normal">Used by</th>
            </tr>
          </thead>
          <tbody className="text-[15px]">
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">Apertus v1.5 8B</td><td className="py-2.5 pr-4">Classifies the image: with six classifiers on top and an adapter or retrained layers in our models, as it comes in the zero-shot one</td><td className="py-2.5 text-slate">Final model, full fine-tune, P11 on our GPU; 8B without training through the Hack Apertus inference API</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">Apertus v1.5 70B</td><td className="py-2.5 pr-4">Describes the image and answers the six questions</td><td className="py-2.5 text-slate">70B without training, through the Hack Apertus inference API</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">SigLIP 2 base</td><td className="py-2.5 pr-4">Image encoder that feeds Apertus 49 extra image tokens</td><td className="py-2.5 text-slate">Final model</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">docTR</td><td className="py-2.5 pr-4">Reads the text printed on the image (OCR)</td><td className="py-2.5 text-slate">Final model</td></tr>
          </tbody>
        </table>
        <p className="m-0 mt-2 text-[12px] text-slate">Sources: src/training/apertus-8B-vadapter-lora-heads/RISULTATI.md:119-120 (google/siglip2-base-patch16-224; docTR fast_base + Felix92/doctr-torch-parseq-multilingual-v1).</p>
      </section>

      <section className="mt-16 border-t-2 border-graphite pt-4">
        <button type="button" onClick={() => setDetails((d) => !d)} aria-expanded={details} data-testid="toggle-details"
          className="flex w-full items-baseline justify-between border-0 bg-transparent p-0 text-left">
          <h2>Details per question and class</h2>
          <span className="text-[15px] text-slate">{details ? 'Hide' : 'Show'}</span>
        </button>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">For experts: how each model does on each of the six questions and on every class, on our internal tests.</p>
        {details && <div className="mt-6"><ExpertScores /></div>}
      </section>
    </AppShell>
  )
}
