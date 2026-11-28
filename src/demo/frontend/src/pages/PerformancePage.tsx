import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { useModels } from '../api/hooks'
import type { MetricBlock, ModelSummary } from '../api/types'
import { ConfusionMatrix, DimensionBarChart, MODEL_COLORS, PerClassTable } from '../components/dashboard'
import { ErpShell, TitleBar } from '../components/shell'
import { ErrorState } from '../components/ui'
import { sortForShow } from '../lib/models'
import { DIMENSION_NAMES } from '../strings'

/** Set interni con una descrizione semplice. I numeri si confrontano solo dentro lo stesso set. */
const SETS: { key: string; label: string; about: string }[] = [
  { key: 'coverage', label: 'Rare-class test, 2,532 images', about: 'Our internal test with extra examples of the rare classes. It is harder than the leaderboard, on purpose.' },
  { key: 'prompt-test', label: 'Held-out test, 1,193 images', about: 'Labelled images never used to train or tune these models, with the natural mix of classes.' },
  { key: 'base', label: 'Natural test, 2,363 images', about: 'The same images as the rare-class test, without the extra rare examples.' },
  { key: 'validation_450', label: 'Small balanced test, 450 images', about: 'The same number of images for every class. Used for the early experiments.' },
]

function ModelCard({ model, block }: { model: ModelSummary; block?: MetricBlock }) {
  const r = block?.results.find((x) => x.validity !== 'seen_in_training')
  if (!r) return null
  return (
    <article data-testid="model-card" data-model={model.model_id} className="border-t-2 border-graphite pt-3" title={`Source: ${r.source}`}>
      <h3 className="m-0 text-[16px] font-bold leading-snug">{model.short_name}</h3>
      <div className="mt-1 text-[32px] font-bold leading-none tracking-tight" data-testid="card-macro">{r.macro_f1?.toFixed(3)}</div>
      <div className="mt-1 text-[13px] text-slate">macro F1{r.weighted_f1 != null ? `, weighted ${r.weighted_f1.toFixed(3)}` : ''}</div>
      <div className="mt-2 text-[13px] text-slate">{model.training_label}</div>
    </article>
  )
}

function Block({ title, about, children, testId }: { title: string; about?: string; children: React.ReactNode; testId?: string }) {
  return (
    <section className="mt-12" data-testid={testId}>
      <h2>{title}</h2>
      {about && <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">{about}</p>}
      <div className="mt-5">{children}</div>
    </section>
  )
}

export default function PerformancePage() {
  const [params, setParams] = useSearchParams()
  const set = SETS.find((s) => s.key === params.get('set')) ?? SETS[0]
  const models = useModels()
  const list = useMemo(() => sortForShow(models.data?.models ?? []).filter((m) => m.scores.length > 0), [models.data])
  const ids = list.map((m) => m.model_id).join(',')
  const metrics = useQuery({
    queryKey: ['metrics', set.key, ids], enabled: !!ids,
    queryFn: () => api<{ models: Record<string, MetricBlock> }>(`/metrics/compare?models=${ids}&set=${set.key}&include_invalid=false`),
  })
  const blocks = metrics.data?.models ?? {}
  const withScore = list.filter((m) => blocks[m.model_id]?.results.some((r) => r.validity !== 'seen_in_training'))

  const series = withScore.flatMap((m, i) => {
    const r = blocks[m.model_id]?.results[0]
    if (!r || !Object.keys(r.per_dimension).length) return []
    return [{ id: m.model_id, name: m.short_name, color: MODEL_COLORS[i % MODEL_COLORS.length],
      values: Object.fromEntries(Object.entries(r.per_dimension).map(([d, v]) => [d, v.macro_f1])) }]
  })
  const withClasses = withScore.filter((m) => blocks[m.model_id]?.results.some((r) => r.per_class))
  const [pcModel, setPcModel] = useState<string | null>(null)
  const pcId = withClasses.some((m) => m.model_id === pcModel) ? pcModel! : withClasses[0]?.model_id
  const pcResult = pcId ? blocks[pcId]?.results.find((r) => r.per_class) : undefined
  const cmModel = withScore.find((m) => blocks[m.model_id]?.results.some((r) => r.confusion))
  const cmResult = cmModel ? blocks[cmModel.model_id]?.results.find((r) => r.confusion) : undefined

  return (
    <ErpShell>
      <TitleBar title="Scores per class" />
      <p className="m-0 max-w-[70ch] text-[15px] text-slate" data-testid="coverage-banner">
        A closer look for experts: how each model does on every question and every class, on our internal test sets.
        For the official comparison, see the Kaggle leaderboard on the Models page.
      </p>
      <div className="mt-6 flex flex-wrap items-center gap-3 text-[15px]">
        <label htmlFor="set-select" className="font-bold">Test set</label>
        <select id="set-select" data-testid="set-select" value={set.key} onChange={(e) => setParams({ set: e.target.value })}
          className="border border-graphite bg-white px-3 py-1.5 text-[15px]">
          {SETS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
        </select>
        <span className="text-[14px] text-slate">{set.about}</span>
      </div>
      {metrics.isError && <ErrorState error={metrics.error} />}

      <div className="mt-8 grid gap-x-8 gap-y-6 sm:grid-cols-2 xl:grid-cols-4" data-testid="model-cards">
        {withScore.map((m) => <ModelCard key={m.model_id} model={m} block={blocks[m.model_id]} />)}
      </div>
      {withScore.length === 0 && <p className="mt-6 text-slate">No scores on this test set.</p>}

      {series.length > 0 && (
        <Block title="Score for each question" about="Macro F1 for each of the six questions, on this test set only." testId="section-dimensions">
          <DimensionBarChart series={series} reference={null} />
        </Block>
      )}

      {withClasses.length > 0 && pcResult?.per_class && (
        <Block title="Score for each class" about="Precision: how often the model is right when it gives this answer. Recall: how many images of this class it finds." testId="section-per-class">
          <label className="mb-4 flex items-center gap-3 text-[15px]">
            <span className="font-bold">Model</span>
            <select value={pcId} onChange={(e) => setPcModel(e.target.value)} className="border border-graphite bg-white px-3 py-1.5 text-[15px]">
              {withClasses.map((m) => <option key={m.model_id} value={m.model_id}>{m.short_name}</option>)}
            </select>
          </label>
          <PerClassTable perClass={pcResult.per_class} />
        </Block>
      )}

      {cmResult?.confusion?.perspective && (
        <Block title="Where the perspective answers go wrong" testId="section-confusion"
          about={`${cmModel!.short_name}. Each row is the true view, each column the model’s answer: the darker the cell, the more images. Left and right are the hardest part.`}>
          <ConfusionMatrix data={cmResult.confusion.perspective} />
        </Block>
      )}
      <p className="mt-12 text-[13px] text-slate">Every number comes from the team’s result files in the repository; hover a model to see the file. {DIMENSION_NAMES.text_classification} = added text on the image.</p>
    </ErpShell>
  )
}
