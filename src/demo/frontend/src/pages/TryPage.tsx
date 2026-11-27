import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Link, useSearchParams } from 'react-router-dom'
import { ApiClientError, api, errorText } from '../api/client'
import { useImage, useImages, useModels, useSessionId, useTaxonomy } from '../api/hooks'
import { subscribeJob } from '../api/sse'
import type { ApiErrorBody, ModelSummary, PredictResponse, PredictionResult, UploadResponse } from '../api/types'
import { DemoTabs } from '../components/nav'
import { ErpShell, TitleBar } from '../components/shell'
import { ModeBadge, RobotIcon, useToast } from '../components/ui'
import { SCORE_SET_LABEL, availability, headlineScore } from '../lib/models'
import { defaultMode } from '../lib/predictions'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'

type RunStatus = { state: 'running' | 'done' | 'error'; error?: ApiErrorBody; started: number; seconds?: number }

const MODEL_TINT = ['#2f5e4a', '#1f4fa8', '#b45309', '#7c3aed', '#0e7490', '#be185d']

function probColor(p: number) {
  return p >= 0.9 ? 'var(--c-conf-high)' : p >= 0.6 ? 'var(--c-conf-mid)' : 'var(--c-conf-low)'
}

function Bar({ p, strong }: { p: number; strong?: boolean }) {
  return (
    <div className={`${strong ? 'h-2.5' : 'h-1.5'} w-full overflow-hidden rounded-sm bg-app`}>
      <div className="h-full rounded-sm" style={{ width: `${Math.max(1, Math.round(p * 100))}%`, background: strong ? probColor(p) : '#9aa3b2' }} />
    </div>
  )
}

function errorMessage(e: ApiErrorBody | undefined): string {
  if (!e) return 'error'
  if (e.code === 'RECORDED_MISSING') return 'No recorded prediction for this image yet (record it on a GPU host first).'
  if (e.code === 'MODEL_UNAVAILABLE') return 'Not available in this demo session.'
  return `${e.message} (${e.code})`
}

/** Una dimensione: per ogni modello la classe scelta con la sua probabilità e le alternative. */
function DimensionCard({ dimension, models, preds, labelOf }: {
  dimension: string; models: ModelSummary[]; preds: Map<string, PredictionResult>; labelOf: (d: string, v: string) => string
}) {
  const labels = models.map((m) => preds.get(m.model_id)?.dimensions[dimension]).filter((d) => d && d.label && d.applicable).map((d) => d!.label)
  const agree = labels.length >= 2 && new Set(labels).size === 1
  const disagree = labels.length >= 2 && new Set(labels).size > 1
  return (
    <section data-testid={`try-dim-${dimension}`} className={`border bg-panel ${disagree ? 'border-conf-mid' : 'border-line'}`}>
      <header className="flex items-center justify-between border-b border-line px-3 py-1.5" style={{ background: 'var(--c-imagetype-bg)' }}>
        <h3 className="m-0 text-xs font-bold uppercase tracking-wide text-sect">{DIMENSION_NAMES[dimension]}</h3>
        {agree && <span className="text-[11px] font-semibold text-conf-high">✓ models agree</span>}
        {disagree && <span className="text-[11px] font-semibold text-conf-mid" data-testid="try-disagree">≠ models disagree</span>}
      </header>
      <div className="divide-y divide-line">
        {models.map((m, i) => {
          const p = preds.get(m.model_id)
          const d = p?.dimensions[dimension]
          return (
            <div key={m.model_id} className="px-3 py-2">
              {models.length > 1 && (
                <div className="mb-0.5 flex items-center gap-1 text-[11px] text-muted">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: MODEL_TINT[i % MODEL_TINT.length] }} />{m.short_name}
                </div>
              )}
              {!p || !d ? <div className="text-xs text-muted">—</div> : d.label === null ? (
                <div className="text-sm font-semibold text-invalid">{S.invalidOutput}</div>
              ) : (
                <>
                  <div className="flex items-baseline justify-between gap-2">
                    <span className={`text-base font-semibold ${d.applicable ? 'text-ink' : 'text-muted'}`} title={`internal value: ${d.label}`}>{d.erp_label === '-' ? 'None' : d.erp_label}</span>
                    <span className="font-mono text-sm font-bold" style={{ color: d.confidence !== null ? probColor(d.confidence) : undefined }}>
                      {d.confidence !== null ? `${Math.round(d.confidence * 100)}%` : 'n/a'}
                    </span>
                  </div>
                  {d.confidence !== null && <Bar p={d.confidence} strong />}
                  {!d.applicable && <div className="mt-0.5 text-[11px] text-muted">{S.notApplicable}</div>}
                  {d.top_k && d.top_k.length > 1 && (
                    <div className="mt-1.5 space-y-0.5">
                      {d.top_k.slice(1).map((t) => (
                        <div key={t.label} className="grid grid-cols-[1fr_5rem_2.5rem] items-center gap-2 text-[11px] text-muted">
                          <span className="truncate">{labelOf(dimension, t.label)}</span><Bar p={t.p} /><span className="text-right font-mono">{Math.round(t.p * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default function TryPage() {
  const { sessionId } = useSessionId()
  const [params, setParams] = useSearchParams()
  const qc = useQueryClient()
  const toast = useToast()
  const models = useModels()
  const taxonomy = useTaxonomy()
  const images = useImages(sessionId)
  const imageId = params.get('image') ?? images.data?.[images.data.length - 1]?.image_id ?? null
  const image = useImage(imageId ?? undefined)
  const [selected, setSelected] = useState<string[] | null>(null)
  const [allRuns, setAllRuns] = useState<Record<string, RunStatus>>({})   // chiave: immagine|modello
  const [uploading, setUploading] = useState(false)
  const [showSecondary, setShowSecondary] = useState(false)

  const all = models.data?.models ?? []
  const visible = all.filter((m) => m.featured || showSecondary).filter((m) => m.runtime !== 'none')
  const usable = (m: ModelSummary) => m.modes.length > 0
  const chosen = (selected ?? visible.filter(usable).map((m) => m.model_id)).filter((id) => visible.some((m) => m.model_id === id))
  const chosenModels = visible.filter((m) => chosen.includes(m.model_id))

  const runs = useMemo(() => Object.fromEntries(Object.entries(allRuns).filter(([k]) => k.startsWith(`${imageId}|`))
    .map(([k, v]) => [k.slice(k.indexOf('|') + 1), v])), [allRuns, imageId])
  const setRun = (image: string, model: string, value: RunStatus) => setAllRuns((r) => ({ ...r, [`${image}|${model}`]: value }))

  const preds = useMemo(() => {
    const map = new Map<string, PredictionResult>()
    for (const p of image.data?.predictions ?? []) map.set(p.model_id, p)
    return map
  }, [image.data])

  const labelOf = useCallback((d: string, v: string) => taxonomy.data?.dimensions.find((x) => x.key === d)?.classes.find((c) => c.value === v)?.erp_label ?? v, [taxonomy.data])

  const pick = (id: string) => setParams((p) => { p.set('image', id); return p })

  const upload = useCallback(async (files: File[]) => {
    if (!sessionId || !files.length) return
    setUploading(true)
    const form = new FormData()
    form.set('session_id', sessionId)
    form.set('source_kind', 'own_photo')
    files.forEach((f) => form.append('files', f))
    try {
      const res = await api<UploadResponse>('/images', { method: 'POST', body: form })
      await qc.invalidateQueries({ queryKey: ['images'] })
      if (res.images[0]) pick(res.images[0].image_id)
      if (res.rejected.length) toast('error', `${res.rejected.length} file(s) rejected: ${res.rejected.map((r) => r.error.message).join('; ')}`)
    } catch (e) {
      toast('error', e instanceof ApiClientError ? errorText(e) : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [sessionId, qc, toast]) // eslint-disable-line react-hooks/exhaustive-deps
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ onDrop: upload, noClick: true, noKeyboard: true, accept: { 'image/*': [] } })

  const classifyOne = async (m: ModelSummary) => {
    if (!imageId) return
    const started = performance.now()
    setRun(imageId, m.model_id, { state: 'running', started })
    try {
      const res = await api<PredictResponse>('/predict', { method: 'POST', json: { model_id: m.model_id, image_ids: [imageId], mode: defaultMode(m), use_context: true } })
      let failure: ApiErrorBody | undefined
      if (res.queued > 0) {
        await new Promise<void>((resolve) => {
          subscribeJob(res.job_id, res.total, { onError: (_, error) => { failure = error }, onDone: () => resolve() })
        })
      }
      await qc.invalidateQueries({ queryKey: ['image', imageId] })
      const seconds = (performance.now() - started) / 1000
      setRun(imageId, m.model_id, failure ? { state: 'error', error: failure, started, seconds } : { state: 'done', started, seconds })
    } catch (e) {
      const err = e instanceof ApiClientError ? { code: e.code, message: e.message } : { code: 'ERROR', message: errorText(e) }
      setRun(imageId, m.model_id, { state: 'error', error: err, started })
    }
  }
  const classify = () => Promise.all(chosenModels.filter(usable).map(classifyOne))
  const running = Object.values(runs).some((r) => r.state === 'running')
  const toggle = (id: string, on: boolean) => setSelected(on ? [...chosen, id] : chosen.filter((x) => x !== id))
  const shown = chosenModels.filter((m) => preds.has(m.model_id) || runs[m.model_id])
  const methods = new Set(shown.map((m) => preds.get(m.model_id)).flatMap((p) => p ? Object.values(p.dimensions).map((d) => d.confidence_method) : []).filter(Boolean))

  return (
    <ErpShell>
      <TitleBar title="Try a model — one image, six classes" />
      <DemoTabs />
      <div className="grid gap-4 p-4 lg:grid-cols-[minmax(280px,4fr)_minmax(0,8fr)]">
        {/* ---- immagine ---- */}
        <div className="space-y-3">
          <div {...getRootProps()} data-testid="try-dropzone"
            className={`relative flex min-h-72 items-center justify-center bg-app p-3 ${isDragActive ? 'ring-4 ring-link' : ''}`}
            style={{ border: '8px solid var(--c-panel-frame)' }}>
            <input {...getInputProps()} data-testid="try-file-input" />
            {image.data ? (
              <img src={image.data.file_url} alt={image.data.original_filename} data-testid="try-image" className="max-h-[420px] max-w-full object-contain" />
            ) : (
              <div className="text-center text-sm text-muted">Drop a product photo here<br />(JPEG, PNG or WebP; never Kaggle dev or final images)</div>
            )}
            {isDragActive && <div className="absolute inset-0 flex items-center justify-center bg-white/80 text-sm font-semibold">Drop to upload</div>}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <button type="button" onClick={open} className="erp-cancel" disabled={uploading} data-testid="try-upload">{uploading ? 'Uploading…' : '⬆ Upload image'}</button>
            {image.data && <span className="truncate text-xs text-muted" title={image.data.original_filename}>{image.data.original_filename} · {image.data.width}×{image.data.height}</span>}
          </div>
          {(images.data?.length ?? 0) > 1 && (
            <div>
              <div className="mb-1 text-xs text-muted">Images in this session</div>
              <div className="flex flex-wrap gap-1.5" data-testid="try-thumbs">
                {images.data!.map((i) => (
                  <button key={i.image_id} type="button" onClick={() => pick(i.image_id)} title={i.original_filename}
                    className={`h-14 w-14 overflow-hidden rounded border-2 bg-panel p-0 ${i.image_id === imageId ? 'border-link' : 'border-transparent'}`}>
                    <img src={i.thumbnail_url} alt="" className="h-full w-full object-cover" />
                  </button>
                ))}
              </div>
            </div>
          )}
          <p className="m-0 text-xs text-muted">{S.neverDevFinal} Uploaded photos have no product context (title, brand, category): the models see only the image.</p>
        </div>

        {/* ---- modelli e risultati ---- */}
        <div className="space-y-3">
          <section className="border border-line bg-panel p-3" style={{ borderTop: '3px solid var(--c-panel-frame)' }}>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h2 className="m-0 flex items-center gap-1 text-base font-bold">Models <RobotIcon /></h2>
              <label className="ml-auto text-xs text-muted"><input type="checkbox" checked={showSecondary} onChange={(e) => setShowSecondary(e.target.checked)} /> secondary experiments</label>
            </div>
            <div className="grid gap-1.5 sm:grid-cols-2" data-testid="try-models">
              {visible.map((m) => {
                const av = availability(m)
                const head = headlineScore(m)
                const run = runs[m.model_id]
                const pred = preds.get(m.model_id)
                return (
                  <label key={m.model_id} title={usable(m) ? m.display_name : `${m.display_name} — ${m.status_reason ?? ''}`}
                    className={`flex cursor-pointer items-start gap-2 rounded border px-2 py-1.5 text-xs ${chosen.includes(m.model_id) ? 'border-link bg-imagetype' : 'border-line'} ${usable(m) ? '' : 'cursor-not-allowed opacity-60'}`}>
                    <input type="checkbox" className="mt-0.5" disabled={!usable(m)} checked={chosen.includes(m.model_id) && usable(m)} onChange={(e) => toggle(m.model_id, e.target.checked)} data-testid={`try-model-${m.model_id}`} />
                    <span className="flex-1">
                      <span className="block font-semibold text-ink">{m.short_name}</span>
                      <span className="text-muted">{head ? `${head.macro_f1!.toFixed(3)} on ${SCORE_SET_LABEL[head.set] ?? head.set}` : 'no valid score'}</span>
                    </span>
                    <span className="flex flex-col items-end gap-0.5">
                      <span className={`rounded px-1.5 py-0.5 font-semibold ${av.tone === 'live' ? 'bg-conf-high-bg text-conf-high' : av.tone === 'recorded' ? 'bg-conf-mid-bg text-conf-mid' : 'bg-toolbtn text-muted'}`}>{av.text}</span>
                      {run?.state === 'running' && <span className="text-link">running…</span>}
                      {run?.state === 'done' && run.seconds !== undefined && <span className="text-muted">{run.seconds.toFixed(1)} s</span>}
                      {!run && pred && <span className="text-muted">cached</span>}
                    </span>
                  </label>
                )
              })}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <button type="button" onClick={classify} disabled={!imageId || running || chosenModels.filter(usable).length === 0}
                data-testid="try-classify"
                className="rounded border-0 px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50" style={{ background: 'var(--c-titlebar-bg)' }}>
                {running ? 'Classifying…' : `▶ Classify with ${chosenModels.filter(usable).length} model${chosenModels.filter(usable).length === 1 ? '' : 's'}`}
              </button>
              <span className="text-xs text-muted">Live zero-shot on the shared gateway: about 7 s (8B) and 24 s (70B) per image in our test of 28/11 — 7 requests each (description + 6 questions).</span>
            </div>
            {Object.entries(runs).filter(([, r]) => r.state === 'error').map(([id, r]) => (
              <p key={id} className="m-0 mt-1 text-xs text-invalid" data-testid="try-error">{all.find((m) => m.model_id === id)?.short_name}: {errorMessage(r.error)}</p>
            ))}
          </section>

          {shown.length > 0 && taxonomy.data ? (
            <>
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
                {shown.map((m, i) => {
                  const p = preds.get(m.model_id)
                  return (
                    <span key={m.model_id} className="inline-flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: MODEL_TINT[i % MODEL_TINT.length] }} />
                      <b className="text-ink">{m.short_name}</b>
                      {p && <ModeBadge mode={p.mode} recorded={p.recorded} compact />}
                      {p?.timing.total_ms != null && <span>{(p.timing.total_ms / 1000).toFixed(1)} s</span>}
                    </span>
                  )
                })}
              </div>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3" data-testid="try-results">
                {DIMENSIONS.map((d) => <DimensionCard key={d} dimension={d} models={shown} preds={preds} labelOf={labelOf} />)}
              </div>
              <p className="m-0 text-xs text-muted">
                Percentages are the model’s own probability for each class. {[...methods].map((x) => S.confidence[x as string] ?? x).join(' · ')}.{' '}
                {methods.size > 1 && <b>{S.confidence.notComparable}</b>}{' '}
                {imageId && <Link to={`/image-classification/${imageId}?session=${sessionId ?? ''}`}>Review and correct in the ERP panel →</Link>}
              </p>
            </>
          ) : (
            <div className="border border-dashed border-line bg-panel px-4 py-10 text-center text-sm text-muted" data-testid="try-empty">
              {imageId ? 'Choose the models and press Classify: the six classes appear here, with each model’s probability.' : 'Upload a product photo to start.'}
            </div>
          )}
        </div>
      </div>
    </ErpShell>
  )
}
