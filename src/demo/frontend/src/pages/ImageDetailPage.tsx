import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, errorText } from '../api/client'
import { useImage, useImages, useModels, useSettings, useTaxonomy } from '../api/hooks'
import type { DimensionResult, ImageItem, PredictionResult, Taxonomy } from '../api/types'
import { useRun } from '../api/useRun'
import { ModelSelect } from '../components/models'
import { CollapsibleSection, ErpShell, LabelValueGrid, SubLinks, TitleBar, ToolbarButton } from '../components/shell'
import {
  ConfidenceChip, ConfidenceMethodNote, ErrorState, ModeBadge, ModelUnavailableState, RobotIcon, StatusPill, useToast,
} from '../components/ui'
import { defaultMode, isRunnable } from '../lib/predictions'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'

function TopKPopover({ dim, method }: { dim: DimensionResult; method: string | null }) {
  if (!dim.top_k) return <div className="p-2 text-xs text-muted">No alternatives available for this method.</div>
  return (
    <div className="w-60 rounded border border-line bg-panel p-2 text-xs shadow" role="dialog" aria-label="Top 3 classes" data-testid="topk-popover">
      <div className="mb-1 font-semibold">Top 3</div>
      {dim.top_k.map((t) => (
        <div key={t.label} className="mb-1">
          <div className="flex justify-between"><span>{t.label}</span><span>{t.p.toFixed(2)}</span></div>
          <div className="h-1.5 rounded bg-toolbtn"><div className="h-full rounded bg-conf-high" style={{ width: `${Math.round(t.p * 100)}%` }} /></div>
        </div>
      ))}
      <ConfidenceMethodNote method={method} />
    </div>
  )
}

function DimensionDropdown({ dimension, taxonomy, pred, value, onChange, selectRef }: {
  dimension: string; taxonomy: Taxonomy; pred: PredictionResult; value: string; onChange: (v: string) => void
  selectRef: (el: HTMLSelectElement | null) => void
}) {
  const [popover, setPopover] = useState(false)
  const dim = pred.dimensions[dimension]
  const classes = taxonomy.dimensions.find((d) => d.key === dimension)!.classes
  const labelOf = (v: string) => classes.find((c) => c.value === v)?.erp_label ?? v
  const top = (dim.top_k ?? []).map((t) => t.label)
  const edited = value !== (dim.label ?? '')
  const notApplicable = !dim.applicable
  return (
    <div className="relative flex items-center gap-2">
      <label className="w-24 shrink-0 text-muted" style={{ fontSize: 'var(--fs-label)' }} htmlFor={`dd-${dimension}`}>{DIMENSION_NAMES[dimension]}</label>
      <select id={`dd-${dimension}`} ref={selectRef} data-testid={`dd-${dimension}`} aria-label={DIMENSION_NAMES[dimension]}
        value={value} onChange={(e) => onChange(e.target.value)}
        title={notApplicable ? S.notApplicable : value ? `internal value: ${value}` : S.invalidOutput}
        className={`erp-select min-w-44 px-2 ${edited ? 'ring-2 ring-conf-mid' : ''} ${notApplicable ? 'bg-toolbtn text-muted' : ''}`}>
        {value === '' && <option value="">— choose —</option>}
        {top.length > 0 && (
          <optgroup label="Model top 3">
            {dim.top_k!.map((t) => <option key={`t-${t.label}`} value={t.label}>{labelOf(t.label)} ({t.p.toFixed(2)})</option>)}
          </optgroup>
        )}
        <optgroup label={top.length ? 'Other classes' : 'All classes'}>
          {classes.filter((c) => !top.includes(c.value)).map((c) => <option key={c.value} value={c.value}>{c.erp_label}</option>)}
        </optgroup>
      </select>
      <ConfidenceChip value={dim.confidence} method={dim.confidence_method} onClick={() => setPopover((p) => !p)} />
      {notApplicable && <span className="text-xs text-muted" title={S.notApplicable}>(not scored)</span>}
      {popover && <div className="absolute left-28 top-full z-20"><TopKPopover dim={dim} method={dim.confidence_method} /></div>}
    </div>
  )
}

function ContextForm({ image, onSaved }: { image: ImageItem; onSaved: () => void }) {
  const [form, setForm] = useState<Record<string, string>>(image.context)
  const toast = useToast()
  const fields: [string, string][] = [['product_title', 'Title'], ['product_type', 'Type'], ['brand', 'Brand'], ['category', 'Category'], ['subcategory', 'Subcategory']]
  const save = async () => {
    try {
      await api(`/images/${image.image_id}/context`, { method: 'PUT', json: form })
      toast('ok', 'Context saved. It changes the model input: re-run the classification.')
      onSaved()
    } catch (e) { toast('error', errorText(e)) }
  }
  return (
    <div className="grid grid-cols-2 gap-2 text-sm" data-testid="context-form">
      {fields.map(([k, label]) => (
        <label key={k} className="flex flex-col text-xs text-muted">{label}
          <input value={form[k] ?? ''} onChange={(e) => setForm({ ...form, [k]: e.target.value })} className="rounded border border-line px-2 py-1 text-sm text-ink" />
        </label>
      ))}
      <div className="col-span-2"><ToolbarButton onClick={save} testId="save-context">Save context</ToolbarButton></div>
    </div>
  )
}

export default function ImageDetailPage() {
  const { imageId } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const toast = useToast()
  const image = useImage(imageId)
  const taxonomy = useTaxonomy()
  const models = useModels()
  const settings = useSettings()
  const sessionId = image.data?.session_id ?? params.get('session')
  const siblings = useImages(sessionId ?? null)
  const { state: run, run: startRun } = useRun(sessionId ?? null)

  const modelList = models.data?.models ?? []
  const modelId = params.get('model') || image.data?.predictions?.[0]?.model_id || modelList.find((m) => m.modes.length)?.model_id || null
  const model = modelList.find((m) => m.model_id === modelId)
  const mode = (image.data?.predictions?.find((p) => p.model_id === modelId)?.mode) ?? defaultMode(model)
  const pred = image.data?.predictions?.find((p) => p.model_id === modelId)
  const review = image.data?.review && image.data.review.prediction_id === pred?.prediction_id ? image.data.review : null

  const [values, setValues] = useState<Record<string, string>>({})
  const [zoom, setZoom] = useState(false)
  const selects = useRef<Record<string, HTMLSelectElement | null>>({})
  useEffect(() => {
    if (!pred) { setValues({}); return }
    setValues(review ? { ...review.values } : Object.fromEntries(DIMENSIONS.map((d) => [d, pred.dimensions[d].label ?? ''])))
  }, [pred?.prediction_id, review?.review_id]) // eslint-disable-line react-hooks/exhaustive-deps

  const changed = useMemo(() => (pred ? DIMENSIONS.filter((d) => (values[d] ?? '') !== (pred.dimensions[d].label ?? '')) : []), [pred, values])
  const complete = DIMENSIONS.every((d) => values[d])

  const reset = useCallback(() => {
    if (pred) setValues(Object.fromEntries(DIMENSIONS.map((d) => [d, pred.dimensions[d].label ?? ''])))
  }, [pred])
  const save = useCallback(async () => {
    if (!pred || !imageId || !complete) return
    try {
      const res = await api<{ status: string; changed_dimensions: string[] }>(`/images/${imageId}/review`, {
        method: 'PUT', json: { prediction_id: pred.prediction_id, model_id: pred.model_id, values },
      })
      toast('ok', res.status === 'corrected' ? `Saved: corrected (${res.changed_dimensions.map((d) => DIMENSION_NAMES[d]).join(', ')})` : 'Saved: reviewed, no changes')
      qc.invalidateQueries({ queryKey: ['image', imageId] })
      qc.invalidateQueries({ queryKey: ['images'] })
    } catch (e) { toast('error', errorText(e)) }
  }, [pred, imageId, values, complete, qc, toast])

  const order = siblings.data ?? []
  const index = order.findIndex((i) => i.image_id === imageId)
  const go = useCallback((delta: number) => {
    const target = order[index + delta]
    if (target) navigate(`/image-classification/${target.image_id}?model=${modelId ?? ''}&session=${sessionId}`)
  }, [order, index, navigate, modelId, sessionId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return
      if (t.tagName === 'SELECT' && e.key !== 'Escape') return
      if (e.key === 'j' || e.key === 'J') go(1)
      else if (e.key === 'k' || e.key === 'K') go(-1)
      else if (e.key === 's' || e.key === 'S') save()
      else if (e.key === 'Escape') reset()
      else if (/^[1-6]$/.test(e.key)) selects.current[DIMENSIONS[Number(e.key) - 1]]?.focus()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, save, reset])

  const classify = async () => {
    if (!model || !imageId) return
    try { await startRun(model.model_id, [imageId], mode, settings.data?.default_use_context ?? true) } catch (e) { toast('error', errorText(e)) }
  }
  const setModel = (id: string) => setParams((p) => { p.set('model', id); return p })

  if (image.isError) return <ErpShell><ErrorState error={image.error} action={<Link to="/image-classification">Back to queue</Link>} /></ErpShell>
  const img = image.data
  const status = review ? review.status : pred ? 'ai_suggested' : 'none'
  const backToQueue = `/image-classification?session=${sessionId ?? ''}`

  return (
    <ErpShell>
      <TitleBar title={`Processed image — ${img?.original_filename ?? '…'}`} id={img ? img.image_id.slice(0, 17) : undefined}>
        <StatusPill status={status} />
      </TitleBar>
      <SubLinks links={[{ to: backToQueue, label: '← Queue' }]} />
      <div className="flex items-center gap-3 border-b border-line bg-panel px-4 py-1 text-sm">
        <button type="button" onClick={() => go(-1)} disabled={index <= 0} className="border-0 bg-transparent">‹ Previous (K)</button>
        <button type="button" onClick={() => go(1)} disabled={index < 0 || index >= order.length - 1} className="border-0 bg-transparent">Next (J) ›</button>
        <span className="text-muted">{index >= 0 ? `${index + 1}/${order.length}` : ''}</span>
      </div>

      {img && (
        <div className="grid gap-4 p-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,3fr)]">
          <div className="flex items-start justify-center bg-app p-4" style={{ border: '8px solid var(--c-panel-frame)' }}>
            <img src={img.file_url} alt={img.original_filename} data-testid="detail-image" onClick={() => setZoom((z) => !z)}
              className={`${zoom ? 'max-h-none cursor-zoom-out' : 'max-h-96 cursor-zoom-in'} max-w-full object-contain`} />
          </div>
          <div className="border border-line bg-panel p-3">
            <div className="section-title mb-2">Metadata</div>
            <LabelValueGrid rows={[
              { label: 'Manually uploaded', value: 'Yes (uploaded in the demo)', info: 'The image was uploaded to the demo session by a person.' },
              { label: 'Used by provider products', value: 'n/a', info: 'No data source in the demo.' },
              { label: 'CDN URL', value: 'n/a (local file)' },
              { label: 'Width (px)', value: img.width },
              { label: 'Height (px)', value: img.height },
              { label: 'Quality Score', value: 'n/a', info: 'No data source in the demo.' },
              { label: 'Format · size', value: `${img.format} · ${(img.bytes / 1024).toFixed(0)} KB` },
              { label: 'SHA-256', value: <span>{img.sha256.slice(0, 8)}… <button type="button" className="border-0 bg-transparent text-link" onClick={() => navigator.clipboard?.writeText(img.sha256)}>copy</button></span> },
            ]} />
          </div>
        </div>
      )}

      {img && taxonomy.data && (
        <section className="mx-4 mb-6 rounded border p-3" data-testid="image-type-panel"
          style={{ background: 'var(--c-imagetype-bg)', border: `1px solid ${pred && !pred.valid ? 'var(--c-invalid)' : 'var(--c-imagetype-border)'}`, borderTop: `3px solid ${pred && !pred.valid ? 'var(--c-invalid)' : 'var(--c-panel-frame)'}` }}>
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <h2 className="m-0 flex items-center gap-1 font-bold" style={{ fontSize: 16 }}>Image type <RobotIcon /></h2>
            {modelList.length > 0 && <ModelSelect id="detail-model" models={modelList} value={modelId} onChange={setModel} />}
            {pred && <ModeBadge mode={pred.mode} recorded={pred.recorded} />}
            {pred?.mode === 'live' && model && !model.parity.verified && <span data-testid="parity-badge" title={model.parity.scope} className="rounded bg-conf-low-bg px-2 py-0.5 text-xs font-semibold text-conf-low">{S.badge.parityUnverified}</span>}
            {pred?.timing.total_ms != null && <span className="text-xs text-muted">{Math.round(pred.timing.total_ms)} ms</span>}
          </div>

          {!pred && model && (
            <div className="space-y-2" data-testid="no-prediction">
              {!isRunnable(model, mode) ? <ModelUnavailableState model={model} /> : (
                <>
                  <p className="m-0 text-sm text-muted">No prediction from this model for this image yet.</p>
                  <ToolbarButton onClick={classify} disabled={run.running} testId="classify-one">{run.running ? 'Classifying…' : `Classify with ${model.display_name}`}</ToolbarButton>
                </>
              )}
              {Object.values(run.errors).map((e, i) => <p key={i} className="m-0 text-sm text-invalid">{e.message} ({e.code})</p>)}
            </div>
          )}

          {pred && (
            <>
              {!pred.valid && (
                <div role="alert" className="mb-2 rounded border border-invalid bg-invalid-bg p-2 text-sm text-invalid" data-testid="invalid-output">
                  {S.invalidOutput}: some values are empty — fill them in by hand.
                  {pred.raw_output && <pre className="m-0 mt-1 whitespace-pre-wrap text-xs">{pred.raw_output}</pre>}
                </div>
              )}
              <div className="grid gap-x-8 gap-y-2 md:grid-cols-2">
                <div className="space-y-2">
                  {['subject', 'background', 'perspective'].map((d, i) => (
                    <DimensionDropdown key={d} dimension={d} taxonomy={taxonomy.data} pred={pred} value={values[d] ?? ''}
                      onChange={(v) => setValues((s) => ({ ...s, [d]: v }))} selectRef={(el) => { selects.current[DIMENSIONS[i]] = el }} />
                  ))}
                </div>
                <div className="space-y-2">
                  {['individuals', 'text_classification', 'text_language'].map((d, i) => (
                    <DimensionDropdown key={d} dimension={d} taxonomy={taxonomy.data} pred={pred} value={values[d] ?? ''}
                      onChange={(v) => setValues((s) => ({ ...s, [d]: v }))} selectRef={(el) => { selects.current[DIMENSIONS[i + 3]] = el }} />
                  ))}
                </div>
              </div>
              {changed.length > 0 && (
                <p className="mb-0 mt-2 text-sm text-conf-mid" data-testid="edited-marker">✎ Edited by a person: {changed.map((d) => DIMENSION_NAMES[d]).join(', ')}</p>
              )}
              <div className="mt-2 space-y-1">
                <CollapsibleSection title="Product context (optional)" defaultOpen={false}><ContextForm image={img} onSaved={() => qc.invalidateQueries({ queryKey: ['image', imageId] })} /></CollapsibleSection>
                {pred.raw_output && pred.valid && (
                  <details className="px-4 text-sm" data-testid="raw-output"><summary className="cursor-pointer">Model text (description / raw output)</summary>
                    <pre className="whitespace-pre-wrap text-xs">{pred.raw_output}</pre></details>
                )}
                {Object.keys(img.context).length === 0 && <p className="m-0 px-4 text-xs text-muted">{S.contextMissing}</p>}
                <div className="px-4"><ConfidenceMethodNote method={Object.values(pred.dimensions).find((d) => d.confidence_method)?.confidence_method} /></div>
              </div>
              <div className="mt-3 flex items-center justify-between">
                <button type="button" onClick={reset} data-testid="cancel" className="erp-cancel">✕ Cancel</button>
                <ToolbarButton onClick={save} disabled={!complete} testId="save">Save</ToolbarButton>
              </div>
            </>
          )}
          <p className="mb-0 mt-2 text-xs text-muted">Shortcuts: J/K next/previous · 1–6 focus a dropdown · S save · Esc cancel.</p>
        </section>
      )}
    </ErpShell>
  )
}
