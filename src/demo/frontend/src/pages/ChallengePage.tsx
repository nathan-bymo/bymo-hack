import { Link } from 'react-router-dom'
import { useModels } from '../api/hooks'
import type { ModelSummary, SpeedInfo } from '../api/types'
import { AppShell } from '../components/app'
import { Eyebrow } from '../components/catalog-ui'
import { T, useI18n } from '../i18n'
import { questionText } from '../lib/labels'
import { DIMENSIONS } from '../strings'

/** Numero di classi per dimensione: `challenge_description.md`, tabella "Classes" (41 in tutto). */
const CLASS_COUNT: Record<string, number> = {
  subject: 13, background: 3, perspective: 10, individuals: 2, text_classification: 6, text_language: 7,
}
/** Pesi della valutazione: `challenge_description.md`, "How we assess". */
const WEIGHTS: [string, string][] = [
  [T('Performance on the final images'), T('about 30%')], [T('Innovation beyond training longer'), '20%'],
  [T('Speed and resource use'), '20%'], [T('Insights: what mattered and what did not'), '15%'], [T('Code quality and practicality'), '15%'],
]

interface Row { key: string; name: string; score: number; ours: boolean; source: string }

function ScoreChart({ rows }: { rows: Row[] }) {
  const sorted = [...rows].sort((a, b) => b.score - a.score)
  return (
    <ol className="m-0 mt-5 list-none space-y-2.5 p-0" data-testid="challenge-chart">
      {sorted.map((r) => (
        <li key={r.key} data-testid="challenge-row" title={r.source}
          className="grid grid-cols-[minmax(0,17rem)_minmax(0,1fr)_4.5rem] items-center gap-4">
          <span className={`text-[15px] leading-tight ${r.ours ? 'font-bold' : 'text-slate'}`}>{r.name}</span>
          <span className="h-3 bg-canvas">
            <span className={`grow block h-full ${r.ours ? 'bg-graphite' : 'bg-bar-alt'}`} style={{ width: `${r.score * 100}%` }} />
          </span>
          <span className={`text-right text-[17px] tabular-nums ${r.ours ? 'font-bold' : 'text-slate'}`}>{r.score.toFixed(3)}</span>
        </li>
      ))}
    </ol>
  )
}

const kaggleScore = (m: ModelSummary) => m.scores.find((s) => s.set === 'kaggle_dev' && s.macro_f1 !== null)
const find = (speeds: SpeedInfo[], gpu: string) => speeds.find((s) => s.gpu_label?.startsWith(gpu))

export default function ChallengePage() {
  const { t } = useI18n()
  const models = useModels()
  const data = models.data

  const ours: Row[] = (data?.models ?? [])
    .filter((m) => m.featured && kaggleScore(m))
    .map((m) => ({ key: m.model_id, name: t(m.short_name), score: kaggleScore(m)!.macro_f1!, ours: true, source: kaggleScore(m)!.source }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)   // i quattro migliori: il resto è nella pagina dei risultati del paper
  const references: Row[] = (data?.kaggle_references ?? []).map((r) => ({ key: r.name, name: t(r.name), score: r.score, ours: false, source: r.source }))
  const best = ours[0]
  const qwen = references.find((r) => r.name.startsWith('Qwen'))
  const gemini = references.filter((r) => r.name.startsWith('Gemini'))

  const sentence = (() => {
    if (!best || !qwen) return null
    const args = { name: best.name, score: best.score.toFixed(3), qwen: qwen.score.toFixed(3) }
    if (best.score > qwen.score) return t('Our best, {name} ({score}), is above the organisers’ fine-tuned Qwen ({qwen}).', args)
    if (gemini.length && best.score > Math.max(...gemini.map((g) => g.score)))
      return t('Our best, {name} ({score}), is above both Gemini zero-shot references and below the organisers’ fine-tuned Qwen ({qwen}).', args)
    return t('Our best, {name} ({score}), is below the organisers’ fine-tuned Qwen ({qwen}).', args)
  })()

  const finalModel = data?.models.find((m) => m.model_id === 'apertus-8b-vadapter-lora-heads-final')
  const mixed = data?.models.find((m) => m.model_id === 'apertus-8b-mixed-lora-heads')
  const l40s = finalModel && find(finalModel.speeds, 'L40S')
  const h100 = mixed && mixed.speeds.find((s) => s.gpu_label?.startsWith('H100'))
  const target = data?.throughput_target_images_per_second

  return (
    <AppShell>
      <div className="max-w-[820px] border-b border-graphite pb-5">
        <Eyebrow>{t('The challenge · Hack Apertus × Galaxus')}</Eyebrow>
        <h1 className="mt-2">{t('Teach Apertus to read every product photo')}</h1>
        <p className="lead mt-2 max-w-none">
          {t('Galaxus shows millions of product images. What each image shows decides which image comes first in the gallery, which images are hidden, and where text overlays or packaging shots belong. Today a fine-tuned Qwen 3.5 9B labels every new image: the goal is an Apertus v1.5 method that can replace it.')}
        </p>
      </div>

      <div className="mt-8 grid gap-10 md:grid-cols-2">
        <section data-testid="challenge-task">
          <h2 className="m-0 text-[20px] font-bold tracking-tight">{t('The task')}</h2>
          <p className="m-0 mt-1 text-[14px] text-slate">{t('For every image, one class in each of six dimensions: 41 classes in total.')}</p>
          <ul className="m-0 mt-3 list-none divide-y divide-rule border-y border-rule p-0">
            {DIMENSIONS.map((d) => (
              <li key={d} className="flex items-baseline justify-between gap-4 py-2">
                <span className="text-[15px] font-bold leading-snug">{questionText(d)}</span>
                <span className="shrink-0 text-[13px] tabular-nums text-slate">{t('{n} classes', { n: CLASS_COUNT[d] })}</span>
              </li>
            ))}
          </ul>
          <ul className="m-0 mt-4 list-disc space-y-1.5 pl-5 text-[14px] leading-relaxed text-slate">
            <li>{t('Apertus does the classifying at runtime. Other models may only help (image encoder, OCR) and must be open-weights.')}</li>
            <li>{t('The product text (title, brand, category) can be missing in production, so the method must work without it.')}</li>
            <li>{t('Fast and cheap enough for more than 100,000 images a day, and deployable on-premises or in a Swiss cloud.')}</li>
          </ul>
        </section>

        <section data-testid="challenge-judging">
          <h2 className="m-0 text-[20px] font-bold tracking-tight">{t('How it is judged')}</h2>
          <p className="m-0 mt-1 text-[14px] text-slate">{t('Score: macro F1 for each dimension, averaged over the six. Rare classes count as much as frequent ones.')}</p>
          <ul className="m-0 mt-3 list-none divide-y divide-rule border-y border-rule p-0">
            {WEIGHTS.map(([label, weight]) => (
              <li key={label} className="flex items-baseline justify-between gap-4 py-2">
                <span className="text-[15px]">{t(label)}</span>
                <span className="shrink-0 text-[15px] font-bold tabular-nums">{t(weight)}</span>
              </li>
            ))}
          </ul>
          <p className="m-0 mt-4 text-[14px] leading-relaxed text-slate">
            {t('Data: 11,876 labelled training photos, 5,261 dev photos scored on the public leaderboard, and a hidden final set.')}
          </p>
        </section>
      </div>

      <section className="mt-12 border-t-2 border-graphite pt-5" data-testid="challenge-results">
        <h2 className="m-0 text-[20px] font-bold tracking-tight">{t('Where we stand')}</h2>
        <p className="m-0 mt-1 text-[14px] text-slate">{t('Public Kaggle leaderboard on the 5,261 dev photos: mean macro F1 over the six dimensions.')}</p>
        {models.isLoading && <p className="mt-4 text-[15px] text-slate">{t('Loading…')}</p>}
        {data && <ScoreChart rows={[...ours, ...references]} />}
        <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-slate">
          <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-graphite" />{t('Our submissions')}</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-bar-alt" />{t('References from the organisers')}</span>
        </div>
        {sentence && <p className="m-0 mt-4 text-[16px] font-bold leading-snug" data-testid="challenge-sentence">{sentence}</p>}

        <div className="mt-7 grid gap-6 md:grid-cols-2" data-testid="challenge-speed">
          {l40s && (
            <div className="border-t-2 border-graphite pt-2">
              <div className="text-[13px] uppercase tracking-[0.1em] text-slate">{t('Final model')}</div>
              <div className="text-[26px] font-bold leading-tight tabular-nums">{t('{n} images per second', { n: l40s.images_per_second })}</div>
              <div className="text-[14px] text-slate">
                {t('on one {gpu}', { gpu: 'L40S' })}
                {l40s.cost_per_100k_usd !== null && ` · ${t('about {usd} per 100,000 images', { usd: `$${l40s.cost_per_100k_usd.toFixed(0)}` })}`}
              </div>
            </div>
          )}
          {h100 && (
            <div className="border-t-2 border-graphite pt-2">
              <div className="text-[13px] uppercase tracking-[0.1em] text-slate">{t('Two image encoders (mixed)')}</div>
              <div className="text-[26px] font-bold leading-tight tabular-nums">{t('{n} images per second', { n: h100.images_per_second })}</div>
              <div className="text-[14px] text-slate">{t('on one {gpu}', { gpu: 'H100' })} · {t('fast path')}</div>
            </div>
          )}
        </div>
        {target !== undefined && <p className="m-0 mt-3 text-[13px] text-slate">{t('The target is at least {n} images per second.', { n: target })}</p>}
        <p className="m-0 mt-5 text-[12px] text-slate">{t('Sources: Kaggle public leaderboard (our submissions), challenge description (references), team reports.')}</p>
      </section>

      <div className="mt-10">
        <Link to="/" className="inline-block bg-signal px-5 py-2.5 text-[15px] font-bold text-white no-underline hover:bg-graphite" data-testid="challenge-try">
          {t('Try it on a photo →')}
        </Link>
      </div>
    </AppShell>
  )
}
