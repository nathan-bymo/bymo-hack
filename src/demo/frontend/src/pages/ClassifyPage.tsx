import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useSearchParams } from 'react-router-dom'
import { ApiClientError, api, apiMessage, errorText } from '../api/client'
import { useImage, useImages, useModels, useSamples, useSessionId } from '../api/hooks'
import { subscribeJob } from '../api/sse'
import type { ApiErrorBody, DimensionResult, ImageItem, ModelSummary, PredictResponse, PredictionResult, SamplePhoto, UploadResponse } from '../api/types'
import { AppShell } from '../components/app'
import { PublishPanel } from '../components/publish'
import { T, tr, useI18n } from '../i18n'
import { plainLabel, questionName, questionText } from '../lib/labels'
import { availability, sortForShow } from '../lib/models'
import { defaultMode } from '../lib/predictions'
import { DIMENSIONS } from '../strings'

type RunStatus = { state: 'running' | 'done' | 'error'; error?: ApiErrorBody; seconds?: number; cached?: boolean }
/** Orologio per misurare il tempo di risposta (solo nei gestori degli eventi). */
const clock = () => performance.now()

/** Foto tolte dalla pagina: si ricordano nel browser, il backend non viene toccato (la foto resta nel database). */
const REMOVED_KEY = 'demo.removed-images'
function loadRemoved(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(REMOVED_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : []
  } catch {
    return []
  }
}
/** Scelta iniziale: il modello finale, se risponde; altrimenti il primo modello che risponde. */
const DEFAULT_MODEL = 'apertus-8b-vadapter-lora-heads-final'

function Bar({ p, strong }: { p: number; strong: boolean }) {
  return (
    <div className="mt-0.5 h-1 w-full bg-canvas">
      <div className={`grow h-full ${strong ? 'bg-graphite' : 'bg-bar-alt'}`} style={{ width: `${Math.max(1, Math.round(p * 100))}%` }} />
    </div>
  )
}

/** Una cella della tabella: le classi più probabili per questa foto, ognuna con la sua percentuale (la prima è la risposta). */
function AnswerCell({ dimension, d }: { dimension: string; d: DimensionResult }) {
  const { t } = useI18n()
  const classes = (d.top_k && d.top_k.length ? d.top_k : d.label ? [{ label: d.label, p: d.confidence ?? 0 }] : [])
    .filter((t, i) => i === 0 || t.p >= 0.01)
  return (
    <ul className={`m-0 list-none space-y-1.5 p-0 ${d.applicable ? '' : 'text-slate'}`} data-testid={`classes-${dimension}`}>
      {classes.map((t, i) => (
        <li key={t.label} data-testid="class-row" title={tr('Official class: {label}', { label: t.label })}>
          <div className="flex items-baseline justify-between gap-3">
            <span className={`break-words leading-snug ${i === 0 ? 'text-[19px] font-bold tracking-tight' : 'text-[14px] text-slate'}`}>{plainLabel(dimension, t.label)}</span>
            <span className={`shrink-0 tabular-nums ${i === 0 ? 'text-[16px]' : 'text-[13px] text-slate'}`} data-testid="class-percent">{Math.round(t.p * 100)}%</span>
          </div>
          <Bar p={t.p} strong={i === 0} />
        </li>
      ))}
      {!d.applicable && <li className="text-[12px] text-slate">{t('Not scored for this kind of image')}</li>}
    </ul>
  )
}

function answerNote(run: RunStatus | undefined, pred: PredictionResult | undefined): string {
  if (run?.state === 'running') return tr('Working…')
  if (pred?.mode === 'recorded') {
    const r = pred.recorded
    const speed = r?.images_per_second ? `, ${r.images_per_second} ${tr('img/s')}` : ''
    return (r?.gpu_name ? tr('Recorded on {gpu}', { gpu: r.gpu_name }) : tr('Recorded')) + speed
  }
  if (run?.state === 'done' && !run.cached && run.seconds !== undefined) return tr('Answered in {s} s', { s: run.seconds.toFixed(1) })
  if (pred) return tr('Answered earlier')
  if (run?.state === 'error') {
    return run.error?.code === 'RECORDED_MISSING' ? tr('No recorded answer for this photo') : tr('No answer for this photo')
  }
  return tr('Waiting')
}

type Reference = NonNullable<SamplePhoto['reference_labels']>

/** Le sei domande in righe: l'etichetta di training (se la foto è di train) e la risposta del modello scelto, in colonne separate. */
function AnswerTable({ model, pred, run, reference }: {
  model: ModelSummary | null; pred: PredictionResult | undefined; run: RunStatus | undefined; reference: Reference | null
}) {
  const { t } = useI18n()
  return (
    <div className="mt-5 overflow-x-auto">
      <table className="w-full table-fixed border-collapse text-left" data-testid={pred ? 'try-results' : 'try-pending'}>
        <thead>
          <tr className="border-b-2 border-graphite align-bottom">
            <th className="w-[30%] py-2 pr-3 text-[13px] font-normal text-slate">{t('Question')}</th>
            {reference && (
              <th className="py-2 pl-3 font-normal" data-testid="col-reference">
                <span className="block text-[15px] font-bold leading-tight">{t('Training label')}</span>
                <span className="block text-[12px] text-slate">{t('From train.csv, not a model answer')}</span>
              </th>
            )}
            {model && (
              <th className="py-2 pl-3 font-normal" data-testid={`col-${model.model_id}`}>
                <span className="block text-[15px] font-bold leading-tight">{t(model.short_name)}</span>
                <span className="block text-[12px] text-slate">{answerNote(run, pred)}</span>
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {DIMENSIONS.map((dim) => {
            const answer = pred?.dimensions[dim]
            const ref = reference?.[dim]
            return (
              <tr key={dim} data-testid={`try-dim-${dim}`} className="border-b border-rule align-top">
                <td className="py-2.5 pr-3">
                  <span className="block text-[15px] font-bold leading-snug" title={questionName(dim)}>{questionText(dim)}</span>
                </td>
                {reference && (
                  <td className="py-2.5 pl-3" data-testid={`reference-${dim}`}>
                    <div className={`break-words text-[19px] font-bold leading-snug tracking-tight ${ref?.human ? '' : 'text-slate'}`}>{ref ? plainLabel(dim, ref.label) : '—'}</div>
                    {ref && !ref.human && <div className="mt-1 text-[12px] text-slate">{t('Not a human label')}</div>}
                  </td>
                )}
                {model && (
                  <td className="py-2.5 pl-3">
                    {answer ? <AnswerCell dimension={dim} d={answer} />
                      : <span className="text-[15px] text-bar-alt">{run?.state === 'running' ? '…' : '—'}</span>}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

const CONTEXT_FIELDS: [string, string][] = [
  ['product_title', T('Product title')], ['product_type', T('Product type')], ['brand', T('Brand')], ['category', T('Category')], ['subcategory', T('Subcategory')],
]

/** Contesto prodotto facoltativo: in produzione può mancare, quindi la demo funziona anche senza. */
function ProductDetails({ image, onSaved }: { image: ImageItem; onSaved: () => void }) {
  const { t } = useI18n()
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
        <span className="text-[16px] font-bold">{t('Product details')} <span className="font-normal text-slate">{t('(optional)')}</span></span>
        <span className="text-[14px] text-slate">{filled ? t('{n} of 5 filled', { n: filled }) : t('None given')}</span>
      </button>
      <p className="m-0 mt-1 text-[14px] text-slate">{t('The catalog text that comes with a photo. It can help, but in production it is often missing, so every model also answers from the photo alone.')}</p>
      {open && (
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {CONTEXT_FIELDS.map(([key, label]) => (
            <label key={key} className="flex flex-col text-[13px] text-slate">{t(label)}
              <input value={form[key] ?? ''} onChange={(e) => { setForm({ ...form, [key]: e.target.value }); setState('idle') }} data-testid={`ctx-${key}`}
                className="mt-0.5 border border-rule px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite" />
            </label>
          ))}
          <div className="flex items-center gap-3 sm:col-span-2">
            <button type="button" onClick={save} disabled={state === 'saving'} data-testid="save-context"
              className="border border-graphite bg-white px-4 py-1.5 text-[14px] font-bold hover:bg-canvas">{t('Save details')}</button>
            <span className="text-[14px] text-slate" role="status">
              {state === 'saved' ? t('Saved. Pick the photo again to use them.') : state === 'error' ? t('Not saved: check that the demo server is running.') : ''}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function statusText(m: ModelSummary): string {
  const av = availability(m)
  if (av.tone === 'live') return tr('Runs live')
  if (av.tone === 'recorded') return tr('Recorded on GPU')
  return tr('Not available here')
}

export default function ClassifyPage() {
  const { t } = useI18n()
  const { sessionId } = useSessionId()
  const [params, setParams] = useSearchParams()
  const qc = useQueryClient()
  const models = useModels()
  const images = useImages(sessionId)
  const samples = useSamples()
  const [openingSample, setOpeningSample] = useState<string | null>(null)
  const [removed, setRemoved] = useState<string[]>(loadRemoved)
  const updateRemoved = (next: string[]) => {
    setRemoved(next)
    localStorage.setItem(REMOVED_KEY, JSON.stringify(next))
  }
  const shownImages = useMemo(() => (images.data ?? []).filter((i) => !removed.includes(i.image_id)), [images.data, removed])
  const fromUrl = params.get('image')
  const imageId = (fromUrl && !removed.includes(fromUrl) ? fromUrl : null) ?? shownImages[shownImages.length - 1]?.image_id ?? null
  const image = useImage(imageId ?? undefined)
  const [selected, setSelected] = useState<string | null>(null)
  const [allRuns, setAllRuns] = useState<Record<string, RunStatus>>({})   // chiave: immagine|modello
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const all = useMemo(() => sortForShow(models.data?.models ?? []), [models.data])
  const usable = (m: ModelSummary) => m.modes.length > 0
  const visible = all.filter((m) => m.featured)   // tutti i modelli; chi non risponde in questa demo è grigio e non si può scegliere
  const chosenModel: ModelSummary | null =
    visible.find((m) => m.model_id === selected && usable(m))
    ?? visible.find((m) => m.model_id === DEFAULT_MODEL && usable(m))
    ?? visible.find(usable)
    ?? null
  const runs = useMemo(() => Object.fromEntries(Object.entries(allRuns).filter(([k]) => k.startsWith(`${imageId}|`))
    .map(([k, v]) => [k.slice(k.indexOf('|') + 1), v])), [allRuns, imageId])
  const setRun = (image: string, model: string, value: RunStatus) => setAllRuns((r) => ({ ...r, [`${image}|${model}`]: value }))

  const preds = useMemo(() => {
    const map = new Map<string, PredictionResult>()
    for (const p of image.data?.predictions ?? []) map.set(p.model_id, p)
    return map
  }, [image.data])

  const pick = useCallback((id: string) => setParams((p) => { p.set('image', id); return p }), [setParams])
  /** Toglie la foto dalla pagina (solo nel browser) e passa alla precedente, se c'è. */
  const removeCurrent = () => {
    if (!imageId) return
    const next = shownImages.filter((i) => i.image_id !== imageId).pop()
    updateRemoved([...removed, imageId])
    if (next) choose(next.image_id)
    else setParams((p) => { p.delete('image'); return p })
  }


  // Classificazione automatica: parte quando si aggiunge o si sceglie una foto e quando si spunta un modello.
  const inFlight = useRef(new Set<string>())
  const classifyOne = async (m: ModelSummary, img: string) => {
    const key = `${img}|${m.model_id}`
    if (inFlight.current.has(key)) return
    inFlight.current.add(key)
    const started = clock()
    setRun(img, m.model_id, { state: 'running' })
    try {
      const res = await api<PredictResponse>('/predict', { method: 'POST', json: { model_id: m.model_id, image_ids: [img], mode: defaultMode(m), use_context: true } })
      let failure: ApiErrorBody | undefined
      if (res.queued > 0) {
        await new Promise<void>((resolve) => {
          subscribeJob(res.job_id, res.total, { onError: (_, error) => { failure = error }, onDone: () => resolve() })
        })
      }
      await qc.invalidateQueries({ queryKey: ['image', img] })
      const seconds = (clock() - started) / 1000
      setRun(img, m.model_id, failure ? { state: 'error', error: failure, seconds } : { state: 'done', seconds, cached: res.queued === 0 })
    } catch (e) {
      setRun(img, m.model_id, { state: 'error', error: e instanceof ApiClientError ? { code: e.code, message: e.message } : { code: 'ERROR', message: errorText(e) } })
    } finally {
      inFlight.current.delete(key)
    }
  }
  const running = Object.values(runs).some((r) => r.state === 'running')
  const selectModel = (m: ModelSummary) => {
    setSelected(m.model_id)
    if (usable(m) && imageId) void classifyOne(m, imageId)
  }
  const choose = (id: string) => { pick(id); if (chosenModel && usable(chosenModel)) void classifyOne(chosenModel, id) }
  /** `replace`: la foto da sostituire (un sample prende il posto della foto corrente invece di aggiungersi alla lista). */
  const upload = async (files: File[], replace?: string | null) => {
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
      const added = res.images.map((i) => i.image_id)
      const replaced = replace && !added.includes(replace) ? [replace] : []
      const next = [...removed.filter((id) => !added.includes(id)), ...replaced]   // la stessa foto ricaricata torna visibile
      if (next.length !== removed.length || next.some((id, i) => id !== removed[i])) updateRemoved(next)
      if (res.images[0]) choose(res.images[0].image_id)
      if (res.rejected.length) setUploadError(t('Not added: {list}', { list: res.rejected.map((r) => `${r.original_filename} (${apiMessage(r.error.code, r.error.message)})`).join('; ') }))
    } catch (e) {
      setUploadError(e instanceof ApiClientError ? errorText(e) : t('The upload failed. Check that the demo server is running.'))
    } finally {
      setUploading(false)
    }
  }
  /** Una foto di esempio entra come se fosse caricata: stesso percorso, quindi stessa guardia e stessi modelli. */
  const openSample = async (sample: SamplePhoto) => {
    setOpeningSample(sample.sample_id)
    try {
      const blob = await (await fetch(sample.file_url)).blob()
      await upload([new File([blob], sample.filename, { type: blob.type || 'image/jpeg' })], imageId)   // sovrascrive la foto corrente
    } catch {
      setUploadError(t('The sample photo could not be loaded. Check that the demo server is running.'))
    } finally {
      setOpeningSample(null)
    }
  }
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ onDrop: (files) => upload(files), noClick: true, noKeyboard: true, accept: { 'image/*': [] } })
  const pred = chosenModel ? preds.get(chosenModel.model_id) : undefined
  const reference = (image.data && samples.data?.find((x) => x.sha256 === image.data!.sha256)?.reference_labels) || null
  const run = chosenModel ? runs[chosenModel.model_id] : undefined
  const hasRecorded = pred?.mode === 'recorded'

  const answers = (chosenModel || reference) ? (
    <section aria-live="polite">
      <AnswerTable model={chosenModel} pred={pred} run={run} reference={reference} />
      {(pred || reference) && (
        <p className="m-0 mt-3 text-[13px] leading-relaxed text-slate" data-testid="try-legend">
          {[
            pred && t('For each question, the three most likely classes and how sure the model is about each.'),
            pred && hasRecorded && t('This answer was computed earlier on a GPU by the same model.'),
            pred && image.data && Object.keys(image.data.context).length === 0 && t('It uses the photo only, without product details.'),
            reference && t('The training label is the answer the dataset gives for this photo; it is shown for reference.'),
          ].filter(Boolean).join(' ')}
        </p>
      )}
      {!chosenModel && <p className="m-0 mt-3 text-[15px] text-slate" data-testid="try-no-answers">{t('No model answers photos in this demo yet.')}</p>}
    </section>
  ) : (
    <p className="m-0 mt-5 text-[15px] text-slate" data-testid="try-no-answers">{t('No model answers photos in this demo yet.')}</p>
  )

  return (
    <AppShell>
      <h1>{t('Six answers for every product photo')}</h1>
      <p className="lead mt-2 max-w-none">{t('Drop a product photo and pick an Apertus model: it answers the six questions Galaxus asks about every image.')}</p>

      <div className="mt-7 grid items-start gap-10 lg:grid-cols-[360px_minmax(0,1fr)]">
        {/* ---- foto ---- */}
        <div className="lg:sticky lg:top-6">
          <div {...getRootProps()} data-testid="try-dropzone"
            className={`relative flex aspect-square items-center justify-center bg-canvas p-5 ${isDragActive ? 'outline-2 outline-signal' : ''}`}>
            <input {...getInputProps()} data-testid="try-file-input" />
            {image.data ? (
              <img src={image.data.file_url} alt={t('The photo being classified')} data-testid="try-image" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="max-w-[26ch] text-center text-slate">
                <p className="m-0 text-[19px] font-bold text-graphite">{t('Drop a photo here')}</p>
                <p className="m-0 mt-1 text-[15px]">{t('JPEG, PNG or WebP. Use your own photos, never the Kaggle dev or final images.')}</p>
              </div>
            )}
            {image.data && (
              <button type="button" onClick={removeCurrent} data-testid="try-remove" aria-label={t('Remove photo')} title={t('Remove photo')}
                className="absolute right-2 top-2 flex h-9 w-9 items-center justify-center border border-rule bg-white p-0 text-slate hover:border-signal hover:text-signal">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="square" aria-hidden="true">
                  <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
                </svg>
              </button>
            )}
            {isDragActive && <div className="absolute inset-0 flex items-center justify-center bg-white/85 text-[19px] font-bold">{t('Release to add the photo')}</div>}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button type="button" onClick={open} disabled={uploading} data-testid="try-upload"
              className="border border-graphite bg-white px-4 py-2 text-[15px] font-bold hover:bg-canvas disabled:opacity-50">
              {uploading ? t('Adding the photo…') : t('Choose a photo')}
            </button>
            {shownImages.length > 1 && (
              <div className="flex flex-wrap gap-1.5" data-testid="try-thumbs" aria-label={t('Earlier photos')}>
                {shownImages.map((i) => (
                  <button key={i.image_id} type="button" onClick={() => choose(i.image_id)} title={i.original_filename}
                    aria-pressed={i.image_id === imageId}
                    className={`h-10 w-10 overflow-hidden border-2 bg-canvas p-0 ${i.image_id === imageId ? 'border-signal' : 'border-transparent'}`}>
                    <img src={i.thumbnail_url} alt="" className="h-full w-full object-cover" />
                  </button>
                ))}
              </div>
            )}
          </div>
          {uploadError && <p className="m-0 mt-2 text-[14px] text-signal" role="alert">{uploadError}</p>}
          {(samples.data?.length ?? 0) > 0 && (
            <div className="mt-5 border-t border-rule pt-3" data-testid="samples">
              <span className="block text-[16px] font-bold">{t('Sample photos')}</span>
              <p className="m-0 mt-1 text-[13px] leading-snug text-slate">{t('Pick one to see the answers right away. Or add your own photo: the model then runs on it, which takes a moment.')}</p>
              <div className="mt-2 grid grid-cols-4 gap-1.5">
                {samples.data!.map((s, index) => {
                  const ready = !!chosenModel && s.recorded_models.includes(chosenModel.model_id)
                  // il titolo non viene dal nome del file (inglese): è la vista riconosciuta, oppure "Esempio n"
                  const sampleTitle = s.reference_labels?.perspective ? plainLabel('perspective', s.reference_labels.perspective.label) : t('Sample {n}', { n: index + 1 })
                  return (
                    <button key={s.sample_id} type="button" onClick={() => openSample(s)} disabled={uploading || openingSample !== null}
                      title={ready ? sampleTitle : t('{title} (this model answers live)', { title: sampleTitle })} data-testid="sample" data-ready={ready}
                      className="group relative block aspect-square border border-rule bg-canvas p-0.5 hover:border-graphite disabled:opacity-60">
                      <img src={s.file_url} alt={sampleTitle} className="h-full w-full object-contain" />
                      {ready
                        ? <span className="absolute bottom-0 right-0 bg-graphite px-1 text-[10px] font-bold uppercase tracking-wide text-white">{t('Ready')}</span>
                        : s.reference_labels && <span className="absolute bottom-0 right-0 bg-white px-1 text-[10px] font-bold uppercase tracking-wide text-slate">{t('Label')}</span>}
                    </button>
                  )
                })}
              </div>
            </div>
          )}
          {image.data && <ProductDetails key={image.data.image_id} image={image.data} onSaved={() => qc.invalidateQueries({ queryKey: ['image', imageId] })} />}
        </div>

        {/* ---- modelli e risposte ---- */}
        <div>
          <fieldset className="m-0 border-0 p-0" data-testid="try-models">
            <legend className="p-0 text-[13px] text-slate">{t('Model')}</legend>
            <div className="mt-2 grid gap-2 sm:grid-cols-3 xl:grid-cols-4">
              {visible.map((m) => {
                const on = chosenModel?.model_id === m.model_id
                const ok = usable(m)
                return (
                  <label key={m.model_id} title={ok ? (m.description ? t(m.description) : undefined) : t('This model does not answer in this demo.')}
                    className={`flex gap-2 border px-2.5 py-2 ${on ? 'border-graphite bg-white shadow-[inset_0_0_0_1px_var(--color-graphite)]' : 'border-rule'} ${ok ? 'cursor-pointer hover:border-graphite' : 'cursor-not-allowed opacity-50'}`}>
                    <input type="radio" name="model" className="mt-0.5 h-4 w-4 shrink-0 accent-signal" checked={on} disabled={!ok}
                      onChange={() => selectModel(m)} data-testid={`try-model-${m.model_id}`} />
                    <span className="min-w-0 leading-tight">
                      <span className="block text-[14px] font-bold leading-snug">{t(m.short_name)}</span>
                      <span className={`mt-0.5 block text-[12px] ${usable(m) ? 'font-bold' : 'text-slate'}`}>{statusText(m)}</span>
                    </span>
                  </label>
                )
              })}
            </div>
          </fieldset>
          {!pred && chosenModel && !reference && run?.state !== 'error' && (
            <p className="m-0 mt-3 text-[15px] text-slate" data-testid="try-empty">
              {!imageId ? t('Add a photo to start.') : running ? t('Working on the answers…') : t('Pick the photo again to get the answers.')}
            </p>
          )}
          {answers}
        </div>
      </div>

      {image.data && (
        <PublishPanel
          key={`${image.data.image_id}-${image.data.context_hash}`}
          image={image.data}
          hasAnswer={(image.data.predictions?.length ?? 0) > 0}
        />
      )}
    </AppShell>
  )
}
