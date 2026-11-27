import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, errorText } from '../api/client'
import { useModels, useSettings } from '../api/hooks'
import type { MetricBlock, ModelSummary, Settings, Throughput } from '../api/types'
import {
  ConfusionMatrix, CoverageWarningBanner, DimensionBarChart, MODEL_COLORS, PerClassTable, ResultBadges, Section, SourceFootnote,
} from '../components/dashboard'
import { ErpShell, SubLinks, TitleBar } from '../components/shell'
import { ErrorState, MetricBadge } from '../components/ui'
import { S } from '../strings'

const SETS = ['coverage', 'base', 'prompt-test', 'prompt-dev-600', 'validation_450']

function ModelCard({ model, block, thr, target }: { model: ModelSummary; block?: MetricBlock; thr?: Throughput; target: number }) {
  const valid = block?.results.find((r) => r.validity !== 'seen_in_training')
  const invalid = block?.results.find((r) => r.validity === 'seen_in_training')
  const speed = thr?.reported.find((r) => r.model_id === model.model_id)
  const measured = thr?.measured_in_demo.find((m) => m.model_id === model.model_id)
  const kaggle = model.kaggle_dev?.score
  return (
    <article data-testid="model-card" data-model={model.model_id} className="rounded border border-line bg-panel p-3 text-sm" style={{ borderRadius: 'var(--radius-card)' }}>
      <h3 className="m-0 mb-1 text-sm font-semibold">{model.display_name}</h3>
      {valid ? (
        <>
          <div className="text-xl font-bold" data-testid="card-macro">{valid.macro_f1?.toFixed(3)}</div>
          <div className="text-xs text-muted">macro F1 · weighted {valid.weighted_f1?.toFixed(3)}{valid.checkpoint ? ` · checkpoint ${valid.checkpoint}` : ''}</div>
          <div className="my-1"><ResultBadges result={valid} /></div>
        </>
      ) : invalid ? (
        <div className="my-1" data-testid="card-invalid">
          <MetricBadge kind="seen_in_training" />
          <p className="m-0 mt-1 text-xs text-invalid">{block?.contamination_note}</p>
        </div>
      ) : (
        <div className="my-1 text-muted">n/a on this set</div>
      )}
      <dl className="m-0 mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
        <dt className="text-muted">Training</dt><dd className="m-0">{model.training?.data ?? 'n/a'}</dd>
        <dt className="text-muted">Weights</dt><dd className="m-0">{!model.weights ? 'remote (gateway)' : model.weights.repo.includes('DA VERIFICARE') ? 'not accessible yet (repo to be confirmed)' : `${model.weights.repo}@${model.weights.revision?.slice(0, 7)} · ${model.weights.subfolder}`}</dd>
        <dt className="text-muted">Speed</dt>
        <dd className="m-0">
          {measured ? `${measured.images_per_second.toFixed(2)} img/s measured (${measured.mode === 'recorded' ? 'recorded run' : measured.gpu_name})`
            : speed ? `${speed.value} img/s reported (${speed.gpu_label ?? speed.conditions ?? 'see source'})` : 'n/a'}
          {(measured?.images_per_second ?? speed?.value ?? 0) >= target ? ' ✓' : ''}
        </dd>
        <dt className="text-muted">Kaggle dev</dt>
        <dd className="m-0">{kaggle != null ? <span data-testid="card-kaggle">{kaggle.toFixed(3)} <MetricBadge kind="kaggle_user" /> <span className="text-muted" title={model.kaggle_dev.note}>({model.kaggle_dev.submitted_at?.slice(0, 10)})</span></span> : 'n/a'}</dd>
      </dl>
      {(valid ?? invalid) && <SourceFootnote source={(valid ?? invalid)!.source} set={block?.set} />}
    </article>
  )
}

function CostPanel({ thr, models }: { thr?: Throughput; models: ModelSummary[] }) {
  const settings = useSettings()
  const qc = useQueryClient()
  const save = useMutation({
    mutationFn: (body: Partial<Settings>) => api<Settings>('/settings', { method: 'PUT', json: body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
  if (!thr || !settings.data) return null
  const target = thr.target_images_per_second
  const chart = [
    ...thr.reported.map((r) => ({ name: `${r.model_id.replace('apertus-', '')} (${r.what.includes('solo generate') ? 'generate only, reported' : 'reported'})`, ips: r.value })),
    ...thr.measured_in_demo.map((m) => ({ name: `${m.model_id} (measured, ${m.mode})`, ips: m.images_per_second })),
  ]
  const gpuOptions = Object.entries(thr.gpu_prices)
  return (
    <div data-testid="cost-panel">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-sm">
        <label>GPU price list:{' '}
          <select value={gpuOptions.some(([g]) => g === settings.data.gpu_label) ? settings.data.gpu_label : ''}
            onChange={(e) => save.mutate({ gpu_label: e.target.value, usd_per_hour: thr.gpu_prices[e.target.value] })} className="rounded border border-line bg-panel px-2 py-1">
            <option value="" disabled>custom</option>
            {gpuOptions.map(([g, p]) => <option key={g} value={g}>{g} — {p.toFixed(2)} $/h</option>)}
          </select>
        </label>
        <label>$/h: <input type="number" step="0.05" min="0" value={settings.data.usd_per_hour} data-testid="usd-per-hour"
          onChange={(e) => save.mutate({ usd_per_hour: Number(e.target.value) })} className="w-20 rounded border border-line px-2 py-1" /></label>
        {save.isError && <span className="text-invalid">{errorText(save.error)}</span>}
      </div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <BarChart data={chart} layout="vertical" margin={{ left: 160, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" /><YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
            <Tooltip /><ReferenceLine x={target} stroke="#b91c1c" strokeDasharray="5 4" label={{ value: `target ≥ ${target} img/s`, fill: '#b91c1c', fontSize: 11, position: 'top' }} />
            <Bar dataKey="ips" fill="#1d4ed8" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="mt-2 w-full max-w-3xl border-collapse text-left text-sm" data-testid="cost-table">
        <thead><tr className="border-b border-line text-xs text-muted"><th>Model</th><th>img/s</th><th>Measured on</th><th>Cost per 1,000 images</th></tr></thead>
        <tbody>
          {thr.reported.map((r, i) => {
            const valid = r.cost_per_1k_images_usd
            const price = settings.data.usd_per_hour
            const same = r.gpu_label === settings.data.gpu_label
            const other = valid !== null && !same ? price / (r.value * 3600) * 1000 : null
            return (
              <tr key={i} className="border-b border-line">
                <td>{r.model_id.replace('apertus-', '')}</td>
                <td>{r.value} <span className="text-xs text-muted">({r.what})</span></td>
                <td>{r.gpu_label ?? r.conditions}</td>
                <td data-testid="cost-cell">{valid === null ? <span className="text-muted">n/a (not end-to-end)</span> :
                  <>{valid.toFixed(3)} $ <span className="text-xs text-muted">= {(valid * 100).toFixed(1)} $ per 100,000 (price of {r.gpu_label}: {r.usd_per_hour} $/h)</span>
                    {other !== null && <span className="ml-1 text-xs text-invalid" data-testid="cost-invalid"> · at {settings.data.gpu_label} price: {other.toFixed(3)} $ — estimate not valid, the speed was measured on {r.gpu_label}</span>}</>}</td>
              </tr>
            )
          })}
          {thr.measured_in_demo.map((m) => (
            <tr key={m.job_id} className="border-b border-line">
              <td>{m.model_id}</td><td>{m.images_per_second.toFixed(2)}</td><td>{m.mode === 'recorded' ? `recorded run (${m.gpu_name})` : m.gpu_name}</td>
              <td>{m.cost_per_1k_images_usd === null ? <span className="text-muted">n/a — {m.cost_note}</span> : `${(settings.data.usd_per_hour / (m.images_per_second * 3600) * 1000).toFixed(3)} $`}</td>
            </tr>
          ))}
          <tr className="border-b border-line"><td>Zero-shot 8B / 70B (gateway)</td><td>3.8 / 2.8 <span className="text-xs text-muted">(reported, shared gateway)</span></td><td>gateway</td><td className="text-muted">n/a (API cost not public)</td></tr>
        </tbody>
      </table>
      <p className="m-0 mt-1 text-xs text-muted">Cost = price per hour ÷ (images per second × 3,600) × 1,000, using only measured or reported speeds with a known GPU. Score weight of cost per 100,000 images: 20% (challenge_description.md:220).</p>
    </div>
  )
}

export default function PerformancePage() {
  const [params, setParams] = useSearchParams()
  const set = SETS.includes(params.get('set') ?? '') ? params.get('set')! : 'coverage'
  const [includeInvalid, setIncludeInvalid] = useState(false)
  const models = useModels()
  const list = useMemo(() => (models.data?.models ?? []).filter((m) => m.headline_eval || m.training?.seen_sets?.length), [models.data])
  const ids = list.map((m) => m.model_id).join(',')
  const metrics = useQuery({
    queryKey: ['metrics', set, ids], enabled: !!ids,
    queryFn: () => api<{ models: Record<string, MetricBlock> }>(`/metrics/compare?models=${ids}&set=${set}&include_invalid=true`),
  })
  const thr = useQuery({ queryKey: ['throughput'], queryFn: () => api<Throughput>('/throughput') })
  const blocks = metrics.data?.models ?? {}
  const target = models.data?.throughput_target_images_per_second ?? 2

  const series = list.flatMap((m, i) => {
    const r = blocks[m.model_id]?.results.find((x) => includeInvalid ? true : x.validity !== 'seen_in_training')
    if (!r || !Object.keys(r.per_dimension).length) return []
    return [{ id: m.model_id, name: `${m.display_name.split('·').slice(0, 2).join('·').trim()}${r.validity === 'seen_in_training' ? ' (invalid)' : ''}`,
      color: MODEL_COLORS[i % MODEL_COLORS.length], values: Object.fromEntries(Object.entries(r.per_dimension).map(([d, v]) => [d, v.macro_f1])) }]
  })
  const withClasses = list.filter((m) => blocks[m.model_id]?.results.some((r) => r.per_class))
  const withConfusion = list.filter((m) => blocks[m.model_id]?.results.some((r) => r.confusion))
  const [pcModel, setPcModel] = useState<string | null>(null)
  const pcId = pcModel ?? withClasses[0]?.model_id
  const pcResult = pcId ? blocks[pcId]?.results.find((r) => r.per_class) : undefined
  const cmId = withConfusion[0]?.model_id
  const cmResult = cmId ? blocks[cmId]?.results.find((r) => r.confusion) : undefined

  return (
    <ErpShell>
      <TitleBar title="Performance — Apertus models" />
      <SubLinks links={[{ to: '/image-classification', label: 'Queue' }, { to: '/image-classification/performance?set=coverage', label: 'Performance' }]} />
      <div className="space-y-3 p-4">
        <CoverageWarningBanner />
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label>Set:{' '}
            <select data-testid="set-select" value={set} onChange={(e) => setParams({ set: e.target.value })} className="rounded border border-line bg-panel px-2 py-1">
              {SETS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label><input type="checkbox" checked={includeInvalid} onChange={(e) => setIncludeInvalid(e.target.checked)} /> include invalid results in the chart</label>
          <span className="text-xs text-muted">Numbers shown are reported by the team's own files, not recomputed here (badge on every number).</span>
        </div>
        {metrics.isError && <ErrorState error={metrics.error} />}
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4" data-testid="model-cards">
          {list.filter((m) => (blocks[m.model_id]?.results.length ?? 0) > 0).map((m) => <ModelCard key={m.model_id} model={m} block={blocks[m.model_id]} thr={thr.data} target={target} />)}
        </div>

        <Section title="Macro F1 per dimension" testId="section-dimensions">
          {series.length ? <DimensionBarChart series={series} reference={models.data ? { value: models.data.reference_line.value, label: S.referenceLine } : null} /> : <p className="text-sm text-muted">No numbers on this set.</p>}
          <p className="m-0 text-xs text-muted" data-testid="reference-note">{S.referenceNote} Reference line: {S.referenceLine} = 0.917 (challenge_description.md:143).</p>
          {series.length > 0 && <SourceFootnote source={list.map((m) => blocks[m.model_id]?.results[0]?.source).filter(Boolean).join('; ')} set={set} />}
        </Section>

        <Section title="F1 per class" testId="section-per-class">
          {withClasses.length && pcResult?.per_class ? (
            <>
              <label className="mb-2 block text-sm">Model:{' '}
                <select value={pcId} onChange={(e) => setPcModel(e.target.value)} className="rounded border border-line bg-panel px-2 py-1">
                  {withClasses.map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}
                </select> <ResultBadges result={pcResult} />
              </label>
              <PerClassTable perClass={pcResult.per_class} />
              <SourceFootnote source={pcResult.per_class_source ?? pcResult.source} set={set} trainedOn={pcResult.trained_on} />
            </>
          ) : <p className="text-sm text-muted">No per-class numbers for this set.</p>}
        </Section>

        <Section title="Confusion matrix" testId="section-confusion">
          {cmResult?.confusion?.perspective ? (
            <>
              <p className="m-0 mb-1 text-sm">{list.find((m) => m.model_id === cmId)?.display_name} · perspective · {cmResult.confusion.perspective.set} <ResultBadges result={cmResult} /></p>
              <ConfusionMatrix data={cmResult.confusion.perspective} />
              <SourceFootnote source={cmResult.confusion.perspective.source} set={cmResult.confusion.perspective.set} trainedOn={cmResult.trained_on} />
            </>
          ) : <p className="text-sm text-muted">Not available on this set: per-image predictions are needed (<code>src/demo/scripts/evaluate.py</code>: GPU, only with explicit approval).</p>}
        </Section>

        <Section title="Confidence: correct vs wrong, and calibration" testId="section-calibration">
          <p className="text-sm text-muted" data-testid="calibration-na">n/a: no per-image predictions on a labelled set are available yet (<code>evaluate.py</code>, GPU or gateway key, only with explicit approval). Confidence of uploaded images is never aggregated here.</p>
        </Section>

        <Section title="Speed and cost" testId="section-cost"><CostPanel thr={thr.data} models={models.data?.models ?? []} /></Section>
      </div>
    </ErpShell>
  )
}
