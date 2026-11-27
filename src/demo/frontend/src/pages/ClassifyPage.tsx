import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useSearchParams } from 'react-router-dom'
import { ApiClientError, api, errorText } from '../api/client'
import { useImage, useImages, useModels, useSessionId } from '../api/hooks'
import { subscribeJob } from '../api/sse'
import type { ApiErrorBody, DimensionResult, ModelSummary, PredictResponse, PredictionResult, UploadResponse } from '../api/types'
import { AppShell } from '../components/app'
import { QUESTIONS, plainLabel } from '../lib/labels'
import { availability, headlineScore, sortForShow } from '../lib/models'
import { defaultMode } from '../lib/predictions'
import { DIMENSIONS } from '../strings'

type RunStatus = { state: 'running' | 'done' | 'error'; error?: ApiErrorBody; seconds?: number }
const UNSURE = 0.6

const SET_PHRASE: Record<string, string> = {
  kaggle_dev: 'on the Kaggle leaderboard',
  'prompt-test': 'on our held-out test',
  coverage: 'on our rare-class test',
  base: 'on our internal test',
  validation_450: 'on a small balanced test',
}

function errorMessage(e: ApiErrorBody | undefined): string {
  if (!e) return 'Something went wrong.'
  if (e.code === 'RECORDED_MISSING') return 'There is no recorded answer for this photo yet.'
  if (e.code === 'MODEL_UNAVAILABLE') return 'This model is not connected in this demo.'
  if (e.code === 'UPSTREAM_ERROR') return 'The Apertus gateway did not answer. Try again in a moment.'
  return e.message
}

function Bar({ p, alt }: { p: number; alt?: boolean }) {
  return (
    <div className={`${alt ? 'h-1' : 'h-2'} w-full bg-canvas`}>
      <div className={`grow h-full ${alt ? 'bg-bar-alt' : p < UNSURE ? 'bg-signal' : 'bg-graphite'}`} style={{ width: `${Math.max(1, Math.round(p * 100))}%` }} />
    </div>
  )
}

function Percent({ p }: { p: number | null }) {
  if (p === null) return <span className="text-slate">n/a</span>
  return <span className={`font-bold ${p < UNSURE ? 'text-signal' : ''}`}>{Math.round(p * 100)}%</span>
}

function Answer({ dimension, d, big }: { dimension: string; d: DimensionResult; big: boolean }) {
  const next = (d.top_k ?? []).slice(1).filter((t) => t.p >= 0.01)
  return (
    <div>
      <div className="flex items-baseline justify-between gap-4">
        <span className={`${big ? 'text-[26px] leading-tight' : 'text-[18px]'} font-bold tracking-tight ${d.applicable ? '' : 'text-slate'}`}
          title={d.label ? `Official class: ${d.label}` : undefined}>
          {plainLabel(dimension, d.label)}
        </span>
        <span className={big ? 'text-[20px]' : 'text-[16px]'}><Percent p={d.confidence} /></span>
      </div>
      {d.confidence !== null && <div className="mt-1.5"><Bar p={d.confidence} /></div>}
      <div className="mt-1.5 text-[13px] text-slate">
        {!d.applicable ? 'Not counted for this kind of image (official rule).'
          : d.confidence !== null && d.confidence < UNSURE ? 'The model is unsure. '
            : ''}
        {d.applicable && next.length > 0 && <>Next guesses: {next.map((t) => `${plainLabel(dimension, t.label)} ${Math.round(t.p * 100)}%`).join(', ')}.</>}
        {d.applicable && next.length === 0 && d.confidence !== null && d.confidence >= UNSURE && 'No other answer above 1%.'}
      </div>
    </div>
  )
}

function QuestionBlock({ dimension, models, preds }: { dimension: string; models: ModelSummary[]; preds: Map<string, PredictionResult> }) {
  const answers = models.map((m) => preds.get(m.model_id)?.dimensions[dimension]).filter((d): d is DimensionResult => !!d)
  const labels = answers.filter((d) => d.label && d.applicable).map((d) => d.label)
  const disagree = labels.length >= 2 && new Set(labels).size > 1
  const q = QUESTIONS[dimension]
  return (
    <section data-testid={`try-dim-${dimension}`} className={`border-t-2 pt-4 ${disagree ? 'border-signal' : 'border-graphite'}`}>
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <h3 className="m-0 text-[19px] font-bold leading-snug tracking-tight">{q.question}</h3>
        {disagree
          ? <span data-testid="try-disagree" className="shrink-0 text-[13px] font-bold text-signal">Models disagree</span>
          : <span className="shrink-0 text-[13px] text-slate">{q.name}</span>}
      </div>
      {models.length === 1 ? (
        answers[0] ? <Answer dimension={dimension} d={answers[0]} big /> : <p className="m-0 text-slate">No answer.</p>
      ) : (
        <div className="space-y-4">
          {models.map((m) => {
            const d = preds.get(m.model_id)?.dimensions[dimension]
            return (
              <div key={m.model_id}>
                <div className="mb-0.5 text-[13px] text-slate">{m.short_name}</div>
                {d ? <Answer dimension={dimension} d={d} big={false} /> : <p className="m-0 text-slate">No answer yet.</p>}
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function statusText(m: ModelSummary): string {
  const av = availability(m)
  if (av.tone === 'live') return 'Runs live'
  if (av.tone === 'recorded') return 'Recorded answers'
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
  const [showMore, setShowMore] = useState(false)

  const all = useMemo(() => sortForShow(models.data?.models ?? []), [models.data])
  const visible = all.filter((m) => (m.featured || showMore) && m.runtime !== 'none')
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

  return (
    <AppShell>
      <h1>Classify a product image</h1>
      <p className="lead mt-3">Drop a product photo. Each Apertus model answers the same six questions Galaxus asks about every image, and shows how sure it is.</p>

      <div className="mt-10 grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        {/* ---- foto ---- */}
        <div>
          <div {...getRootProps()} data-testid="try-dropzone"
            className={`relative flex aspect-square items-center justify-center bg-canvas p-6 ${isDragActive ? 'outline-2 outline-signal' : ''}`}>
            <input {...getInputProps()} data-testid="try-file-input" />
            {image.data ? (
              <img src={image.data.file_url} alt="The photo being classified" data-testid="try-image" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="max-w-[28ch] text-center text-slate">
                <p className="m-0 text-[19px] font-bold text-graphite">Drop a photo here</p>
                <p className="m-0 mt-1">JPEG, PNG or WebP. Use your own photos, never the Kaggle dev or final images.</p>
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
                    className={`h-11 w-11 overflow-hidden border-2 bg-canvas p-0 ${i.image_id === imageId ? 'border-signal' : 'border-transparent'}`}>
                    <img src={i.thumbnail_url} alt="" className="h-full w-full object-cover" />
                  </button>
                ))}
              </div>
            )}
          </div>
          {uploadError && <p className="m-0 mt-2 text-[14px] text-signal" role="alert">{uploadError}</p>}
        </div>

        {/* ---- modelli ---- */}
        <div>
          <h2>Choose the models</h2>
          <p className="m-0 mt-1 text-[15px] text-slate">Models marked “Runs live” answer now. The trained ones run on a GPU, which is not connected to this laptop.</p>
          <ul className="m-0 mt-5 list-none divide-y divide-rule border-y border-rule p-0" data-testid="try-models">
            {visible.map((m) => {
              const head = headlineScore(m)
              const run = runs[m.model_id]
              const ok = usable(m)
              return (
                <li key={m.model_id}>
                  <label className={`flex gap-3 py-3 ${ok ? 'cursor-pointer' : 'cursor-not-allowed'}`}>
                    <input type="checkbox" className="mt-1 h-4 w-4 accent-signal" disabled={!ok} checked={chosen.includes(m.model_id)}
                      onChange={(e) => toggle(m.model_id, e.target.checked)} data-testid={`try-model-${m.model_id}`} />
                    <span className={`flex-1 ${ok ? '' : 'opacity-55'}`}>
                      <span className="flex flex-wrap items-baseline justify-between gap-x-4">
                        <span className="text-[16px] font-bold">{m.short_name}</span>
                        <span className={`text-[13px] ${ok ? 'font-bold' : 'text-slate'}`}>
                          {run?.state === 'running' ? 'Working…' : run?.state === 'done' && run.seconds !== undefined ? `Answered in ${run.seconds.toFixed(1)} s` : statusText(m)}
                        </span>
                      </span>
                      <span className="mt-0.5 block text-[14px] leading-snug text-slate">{m.description}</span>
                      {head && <span className="mt-0.5 block text-[13px] text-slate">Score {head.macro_f1!.toFixed(3)} {SET_PHRASE[head.set] ?? head.set}</span>}
                    </span>
                  </label>
                </li>
              )
            })}
          </ul>
          <label className="mt-2 block text-[13px] text-slate"><input type="checkbox" checked={showMore} onChange={(e) => setShowMore(e.target.checked)} /> Also show early experiments</label>
          <button type="button" onClick={classify} disabled={!imageId || running || chosenModels.length === 0} data-testid="try-classify"
            className="mt-5 bg-signal px-6 py-3 text-[16px] font-bold text-white hover:bg-[#b82216] disabled:cursor-not-allowed disabled:opacity-40">
            {running ? 'Classifying…' : chosenModels.length > 1 ? `Classify with ${chosenModels.length} models` : 'Classify'}
          </button>
          {!imageId && <p className="m-0 mt-2 text-[14px] text-slate">Add a photo first.</p>}
          {Object.entries(runs).filter(([, r]) => r.state === 'error').map(([id, r]) => (
            <p key={id} className="m-0 mt-2 text-[14px] text-signal" data-testid="try-error" role="alert">{all.find((m) => m.model_id === id)?.short_name}: {errorMessage(r.error)}</p>
          ))}
        </div>
      </div>

      {/* ---- risposte ---- */}
      <section className="mt-16" aria-live="polite">
        {shown.length > 0 ? (
          <>
            <h2>The six answers</h2>
            <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
              The bar shows how sure the model is. Red means the model is unsure (below {UNSURE * 100}%) or that the models give different answers.
              {shown.length > 1 && ' Live models without training and trained models measure certainty in different ways: compare the answers, not the percentages.'}
              {hasRecorded && ' Answers marked as recorded were computed earlier on a GPU, with the same model.'}
            </p>
            <div className="mt-8 grid gap-x-12 gap-y-10 md:grid-cols-2" data-testid="try-results">
              {DIMENSIONS.map((d) => <QuestionBlock key={d} dimension={d} models={shown} preds={preds} />)}
            </div>
          </>
        ) : (
          <p className="m-0 border-t-2 border-graphite pt-4 text-[17px] text-slate" data-testid="try-empty">
            {imageId ? 'Press Classify. The six answers appear here.' : 'Add a photo to start.'}
          </p>
        )}
      </section>
    </AppShell>
  )
}
