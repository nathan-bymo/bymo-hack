import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useSearchParams } from 'react-router-dom'
import { ApiClientError, api, errorText } from '../api/client'
import { useImage, useImages, useModels, useSessionId } from '../api/hooks'
import { subscribeJob } from '../api/sse'
import type { ApiErrorBody, DimensionResult, ImageItem, ModelSummary, PredictResponse, PredictionResult, UploadResponse } from '../api/types'
import { AppShell } from '../components/app'
import { QUESTIONS, plainLabel } from '../lib/labels'
import { availability, headlineScore, sortForShow } from '../lib/models'
import { defaultMode } from '../lib/predictions'
import { DIMENSIONS } from '../strings'

type RunStatus = { state: 'running' | 'done' | 'error'; error?: ApiErrorBody; seconds?: number }
const UNSURE = 0.6

const SET_PHRASE: Record<string, string> = {
  kaggle_dev: 'Kaggle leaderboard',
  'prompt-test': 'our held-out test',
  coverage: 'our rare-class test',
  base: 'our internal test',
  validation_450: 'a small balanced test',
}

function errorMessage(e: ApiErrorBody | undefined): string {
  if (!e) return 'Something went wrong.'
  if (e.code === 'RECORDED_MISSING') return 'There is no recorded answer for this photo yet.'
  if (e.code === 'MODEL_UNAVAILABLE') return 'This model is not connected in this demo.'
  if (e.code === 'UPSTREAM_ERROR') return 'The Apertus gateway did not answer. Try again in a moment.'
  return e.message
}

function Bar({ p }: { p: number }) {
  return (
    <div className="h-1 w-full bg-canvas">
      <div className={`grow h-full ${p < UNSURE ? 'bg-signal' : 'bg-graphite'}`} style={{ width: `${Math.max(1, Math.round(p * 100))}%` }} />
    </div>
  )
}

/** Una cella della tabella: la risposta in parole semplici, quanto è sicuro il modello, e (con un solo modello) le alternative. */
function AnswerCell({ dimension, d, single }: { dimension: string; d: DimensionResult; single: boolean }) {
  const next = (d.top_k ?? []).slice(1).filter((t) => t.p >= 0.01)
  const nextText = next.map((t) => `${plainLabel(dimension, t.label)} ${Math.round(t.p * 100)}%`).join(', ')
  return (
    <div title={[d.label ? `Official class: ${d.label}` : '', nextText ? `Next guesses: ${nextText}` : ''].filter(Boolean).join('\n') || undefined}>
      <div className="flex items-baseline justify-between gap-3">
        <span className={`${single ? 'text-[19px]' : 'text-[16px]'} font-bold leading-snug tracking-tight ${d.applicable ? '' : 'text-slate'}`}>
          {plainLabel(dimension, d.label)}
        </span>
        {d.confidence !== null && (
          <span className={`shrink-0 tabular-nums ${single ? 'text-[17px]' : 'text-[15px]'} ${d.confidence < UNSURE ? 'font-bold text-signal' : 'text-graphite'}`}>{Math.round(d.confidence * 100)}%</span>
        )}
      </div>
      {d.confidence !== null && <div className="mt-1"><Bar p={d.confidence} /></div>}
      {!d.applicable && <div className="mt-1 text-[12px] text-slate">Not scored for this kind of image</div>}
      {single && d.applicable && next.length > 0 && <div className="mt-1 text-[13px] text-slate">Next: {nextText}</div>}
    </div>
  )
}

function columnNote(run: RunStatus | undefined, pred: PredictionResult | undefined): string {
  if (run?.state === 'running') return 'Working…'
  if (run?.state === 'error') return 'No answer'
  if (pred?.mode === 'recorded') {
    const r = pred.recorded
    return `Recorded${r?.gpu_name ? ` on ${r.gpu_name}` : ''}${r?.images_per_second ? `, ${r.images_per_second} img/s` : ''}`
  }
  if (run?.state === 'done' && run.seconds !== undefined) return `Answered in ${run.seconds.toFixed(1)} s`
  if (pred) return 'Answered earlier'
  return 'Waiting'
}

/** Le sei domande in righe, i modelli scelti in colonne: si confrontano le risposte sulla stessa foto con un colpo d'occhio. */
function AnswerTable({ models, preds, runs }: { models: ModelSummary[]; preds: Map<string, PredictionResult>; runs: Record<string, RunStatus> }) {
  const single = models.length === 1
  const anyAnswer = models.some((m) => preds.has(m.model_id))
  return (
    <table className="mt-5 w-full table-fixed border-collapse text-left" data-testid={anyAnswer ? 'try-results' : 'try-pending'}>
      <thead>
        <tr className="border-b-2 border-graphite align-bottom">
          <th className="w-[34%] py-2 pr-3 text-[13px] font-normal text-slate">Question</th>
          {models.map((m) => (
            <th key={m.model_id} className="py-2 pl-3 font-normal" data-testid={`col-${m.model_id}`}>
              <span className="block text-[15px] font-bold leading-tight">{m.short_name}</span>
              <span className="block text-[12px] text-slate">{columnNote(runs[m.model_id], preds.get(m.model_id))}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {DIMENSIONS.map((dim) => {
          const answers = models.map((m) => preds.get(m.model_id)?.dimensions[dim])
          const labels = answers.filter((d): d is DimensionResult => !!d && !!d.label && d.applicable).map((d) => d.label)
          const disagree = labels.length >= 2 && new Set(labels).size > 1
          return (
            <tr key={dim} data-testid={`try-dim-${dim}`} className="border-b border-rule align-top">
              <td className={`py-2.5 pr-3 ${disagree ? 'border-l-2 border-signal pl-2.5' : ''}`}>
                <span className="block text-[15px] font-bold leading-snug" title={QUESTIONS[dim].name}>{QUESTIONS[dim].question}</span>
                {disagree && <span data-testid="try-disagree" className="block text-[12px] font-bold text-signal">Models disagree</span>}
              </td>
              {models.map((m, i) => (
                <td key={m.model_id} className="py-2.5 pl-3">
                  {answers[i] ? <AnswerCell dimension={dim} d={answers[i]!} single={single} />
                    : <span className="text-[15px] text-bar-alt">{runs[m.model_id]?.state === 'running' ? '…' : '—'}</span>}
                </td>
              ))}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

const CONTEXT_FIELDS: [string, string][] = [
  ['product_title', 'Product title'], ['product_type', 'Product type'], ['brand', 'Brand'], ['category', 'Category'], ['subcategory', 'Subcategory'],
]

/** Contesto prodotto facoltativo: in produzione può mancare, quindi la demo funziona anche senza. */
function ProductDetails({ image, onSaved }: { image: ImageItem; onSaved: () => void }) {
  const [open, setOpen] = useState(Object.keys(image.context).length > 0)
  const [form, setForm] = useState<Record<string, string>>(image.context)
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const filled = Object.values(image.context).filter(Boolean).length
  const save = async () => {
    setState('saving')
    try {
      await api(`/images/${image.image_id}/context`, { method: 'PUT', json: form })
      setState('saved')
      onSaved()
    } catch {
      setState('error')
    }
  }
  return (
    <div className="mt-5 border-t border-rule pt-3" data-testid="product-details">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-baseline justify-between border-0 bg-transparent p-0 text-left">
        <span className="text-[16px] font-bold">Product details <span className="font-normal text-slate">(optional)</span></span>
        <span className="text-[14px] text-slate">{filled ? `${filled} of 5 filled` : 'None given'}</span>
      </button>
      <p className="m-0 mt-1 text-[14px] text-slate">The catalog text that comes with a photo. It can help, but in production it is often missing, so every model also answers from the photo alone.</p>
      {open && (
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {CONTEXT_FIELDS.map(([key, label]) => (
            <label key={key} className="flex flex-col text-[13px] text-slate">{label}
              <input value={form[key] ?? ''} onChange={(e) => { setForm({ ...form, [key]: e.target.value }); setState('idle') }} data-testid={`ctx-${key}`}
                className="mt-0.5 border border-rule px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite" />
            </label>
          ))}
          <div className="flex items-center gap-3 sm:col-span-2">
            <button type="button" onClick={save} disabled={state === 'saving'} data-testid="save-context"
              className="border border-graphite bg-white px-4 py-1.5 text-[14px] font-bold hover:bg-canvas">Save details</button>
            <span className="text-[14px] text-slate" role="status">
              {state === 'saved' ? 'Saved. Press Classify again to use them.' : state === 'error' ? 'Not saved: check that the demo server is running.' : ''}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function statusText(m: ModelSummary): string {
  const av = availability(m)
  if (av.tone === 'live') return 'Runs live'
  if (av.tone === 'recorded') return 'Recorded on a GPU'
  if (m.runtime === 'worker_gpu') return 'Needs a GPU'
  if (m.runtime === 'remote_openai') return 'Needs the API key'
  return 'Scores only'
}

export default function ClassifyPage() {
  const { sessionId } = useSessionId()
  const [params, setParams] = useSearchParams()
  const qc = useQueryClient()
  const models = useModels()
  const images = useImages(sessionId)
  const imageId = params.get('image') ?? images.data?.[images.data.length - 1]?.image_id ?? null
  const image = useImage(imageId ?? undefined)
  const [selected, setSelected] = useState<string[] | null>(null)
  const [allRuns, setAllRuns] = useState<Record<string, RunStatus>>({})   // chiave: immagine|modello
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const all = useMemo(() => sortForShow(models.data?.models ?? []), [models.data])
  const visible = all.filter((m) => m.featured && m.runtime !== 'none')
  const usable = (m: ModelSummary) => m.modes.length > 0
  const chosen = (selected ?? visible.filter(usable).map((m) => m.model_id)).filter((id) => visible.some((m) => m.model_id === id && usable(m)))
  const chosenModels = visible.filter((m) => chosen.includes(m.model_id))
  const runs = useMemo(() => Object.fromEntries(Object.entries(allRuns).filter(([k]) => k.startsWith(`${imageId}|`))
    .map(([k, v]) => [k.slice(k.indexOf('|') + 1), v])), [allRuns, imageId])
  const setRun = (image: string, model: string, value: RunStatus) => setAllRuns((r) => ({ ...r, [`${image}|${model}`]: value }))

  const preds = useMemo(() => {
    const map = new Map<string, PredictionResult>()
    for (const p of image.data?.predictions ?? []) map.set(p.model_id, p)
    return map
  }, [image.data])

  const pick = useCallback((id: string) => setParams((p) => { p.set('image', id); return p }), [setParams])

  const upload = useCallback(async (files: File[]) => {
    if (!sessionId || !files.length) return
    setUploading(true)
    setUploadError(null)
    const form = new FormData()
    form.set('session_id', sessionId)
    form.set('source_kind', 'own_photo')
    files.forEach((f) => form.append('files', f))
    try {
      const res = await api<UploadResponse>('/images', { method: 'POST', body: form })
      await qc.invalidateQueries({ queryKey: ['images'] })
      if (res.images[0]) pick(res.images[0].image_id)
      if (res.rejected.length) setUploadError(`Not added: ${res.rejected.map((r) => `${r.original_filename} (${r.error.message})`).join('; ')}`)
    } catch (e) {
      setUploadError(e instanceof ApiClientError ? errorText(e) : 'The upload failed. Check that the demo server is running.')
    } finally {
      setUploading(false)
    }
  }, [sessionId, qc, pick])
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ onDrop: upload, noClick: true, noKeyboard: true, accept: { 'image/*': [] } })

  const classifyOne = async (m: ModelSummary) => {
    if (!imageId) return
    const started = performance.now()
    setRun(imageId, m.model_id, { state: 'running' })
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
      setRun(imageId, m.model_id, failure ? { state: 'error', error: failure, seconds } : { state: 'done', seconds })
    } catch (e) {
      setRun(imageId, m.model_id, { state: 'error', error: e instanceof ApiClientError ? { code: e.code, message: e.message } : { code: 'ERROR', message: errorText(e) } })
    }
  }
  const classify = () => Promise.all(chosenModels.map(classifyOne))
  const running = Object.values(runs).some((r) => r.state === 'running')
  const toggle = (id: string, on: boolean) => setSelected(on ? [...chosen, id] : chosen.filter((x) => x !== id))
  const shown = chosenModels.filter((m) => preds.has(m.model_id))
  const hasRecorded = shown.some((m) => preds.get(m.model_id)?.mode === 'recorded')

  const liveAndTrained = shown.some((m) => preds.get(m.model_id)?.mode === 'live') && shown.some((m) => m.runtime !== 'remote_openai')

  return (
    <AppShell>
      <h1>Six answers for every product photo</h1>
      <p className="lead mt-2 max-w-none">Drop a product photo and pick the Apertus models: each answers the six questions Galaxus asks about every image.</p>

      <div className="mt-7 grid items-start gap-10 lg:grid-cols-[400px_minmax(0,1fr)]">
        {/* ---- foto ---- */}
        <div className="lg:sticky lg:top-6">
          <div {...getRootProps()} data-testid="try-dropzone"
            className={`relative flex aspect-square items-center justify-center bg-canvas p-5 ${isDragActive ? 'outline-2 outline-signal' : ''}`}>
            <input {...getInputProps()} data-testid="try-file-input" />
            {image.data ? (
              <img src={image.data.file_url} alt="The photo being classified" data-testid="try-image" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="max-w-[26ch] text-center text-slate">
                <p className="m-0 text-[19px] font-bold text-graphite">Drop a photo here</p>
                <p className="m-0 mt-1 text-[15px]">JPEG, PNG or WebP. Use your own photos, never the Kaggle dev or final images.</p>
              </div>
            )}
            {isDragActive && <div className="absolute inset-0 flex items-center justify-center bg-white/85 text-[19px] font-bold">Release to add the photo</div>}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button type="button" onClick={open} disabled={uploading} data-testid="try-upload"
              className="border border-graphite bg-white px-4 py-2 text-[15px] font-bold hover:bg-canvas disabled:opacity-50">
              {uploading ? 'Adding the photo…' : 'Choose a photo'}
            </button>
            {(images.data?.length ?? 0) > 1 && (
              <div className="flex flex-wrap gap-1.5" data-testid="try-thumbs" aria-label="Earlier photos">
                {images.data!.map((i) => (
                  <button key={i.image_id} type="button" onClick={() => pick(i.image_id)} title={i.original_filename}
                    aria-pressed={i.image_id === imageId}
                    className={`h-10 w-10 overflow-hidden border-2 bg-canvas p-0 ${i.image_id === imageId ? 'border-signal' : 'border-transparent'}`}>
                    <img src={i.thumbnail_url} alt="" className="h-full w-full object-cover" />
                  </button>
                ))}
              </div>
            )}
          </div>
          {uploadError && <p className="m-0 mt-2 text-[14px] text-signal" role="alert">{uploadError}</p>}
          {image.data && <ProductDetails key={image.data.image_id} image={image.data} onSaved={() => qc.invalidateQueries({ queryKey: ['image', imageId] })} />}
        </div>

        {/* ---- modelli e risposte ---- */}
        <div>
          <fieldset className="m-0 border-0 p-0" data-testid="try-models">
            <legend className="p-0 text-[13px] text-slate">
              {visible.some((m) => m.runtime === 'worker_gpu' && m.modes.includes('recorded'))
                ? 'Models: the trained ones answer from recordings made on a GPU, the others run live'
                : 'Models: the trained ones need a GPU, not connected to this laptop; the others run live through the Apertus API'}
            </legend>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              {visible.map((m) => {
                const head = headlineScore(m)
                const ok = usable(m)
                const on = chosen.includes(m.model_id)
                return (
                  <label key={m.model_id} title={m.description ?? undefined}
                    className={`flex gap-2.5 border px-3 py-2.5 ${on ? 'border-graphite bg-white shadow-[inset_0_0_0_1px_var(--color-graphite)]' : 'border-rule'} ${ok ? 'cursor-pointer hover:border-graphite' : 'cursor-not-allowed opacity-50'}`}>
                    <input type="checkbox" className="mt-0.5 h-4 w-4 shrink-0 accent-signal" disabled={!ok} checked={on}
                      onChange={(e) => toggle(m.model_id, e.target.checked)} data-testid={`try-model-${m.model_id}`} />
                    <span className="min-w-0 leading-tight">
                      <span className="block text-[15px] font-bold">{m.short_name}</span>
                      {head && <span className="mt-0.5 block text-[12px] text-slate">{head.macro_f1!.toFixed(3)} on {SET_PHRASE[head.set] ?? head.set}</span>}
                      <span className={`mt-0.5 block text-[12px] ${ok ? 'font-bold' : 'text-slate'}`}>{statusText(m)}</span>
                    </span>
                  </label>
                )
              })}
              <div className="flex flex-col justify-center gap-1.5">
                <button type="button" onClick={classify} disabled={!imageId || running || chosenModels.length === 0} data-testid="try-classify"
                  className="h-full min-h-12 bg-signal px-5 text-[16px] font-bold text-white hover:bg-[#b82216] disabled:cursor-not-allowed disabled:opacity-40">
                  {running ? 'Classifying…' : chosenModels.length > 1 ? `Classify with ${chosenModels.length} models` : 'Classify'}
                </button>
              </div>
            </div>
          </fieldset>
          {shown.length === 0 && (
            <p className="m-0 mt-3 text-[15px] text-slate" data-testid="try-empty">
              {!imageId ? 'Add a photo to start.' : chosenModels.length === 0 ? 'Pick at least one model.' : 'Press Classify: the answers fill the table.'}
            </p>
          )}
          {Object.entries(runs).filter(([, r]) => r.state === 'error').map(([id, r]) => (
            <p key={id} className="m-0 mt-2 text-[14px] text-signal" data-testid="try-error" role="alert">{all.find((m) => m.model_id === id)?.short_name}: {errorMessage(r.error)}</p>
          ))}

          <section aria-live="polite">
            {chosenModels.length > 0 && <AnswerTable models={chosenModels} preds={preds} runs={runs} />}
            {shown.length > 0 && (
              <p className="m-0 mt-3 text-[13px] leading-relaxed text-slate" data-testid="try-legend">
                The percentage is how sure the model is; red means below {UNSURE * 100}%, or models that disagree.
                {liveAndTrained && ' Models without training and trained models measure certainty differently: compare the answers, not the percentages.'}
                {hasRecorded && ' Recorded answers were computed earlier on a GPU by the same model.'}
                {image.data && Object.keys(image.data.context).length === 0 && ' These answers use the photo only, without product details.'}
              </p>
            )}
          </section>
        </div>
      </div>
    </AppShell>
  )
}
