import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Link, useNavigate } from 'react-router-dom'
import { ApiClientError, api, errorText } from '../api/client'
import { useImages, useModels, usePredictions, useSessionId, useSettings } from '../api/hooks'
import type { ApiErrorBody, ImageItem, PredictionResult, UploadResponse } from '../api/types'
import { useRun } from '../api/useRun'
import { ModelSelect } from '../components/models'
import { ActionToolbar, ErpShell, SubLinks, TitleBar, ToolbarButton, useGlobalSearch } from '../components/shell'
import {
  ConfidenceChip, EmptyState, ErrorState, ModeBadge, ModelUnavailableState, ProgressBar, StatusPill, ThroughputBadge, useToast,
} from '../components/ui'
import { defaultMode, isRunnable, latestByImage, minConfidence } from '../lib/predictions'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'

function Cell({ pred, dim, error, queued }: { pred?: PredictionResult; dim: string; error?: ApiErrorBody; queued: boolean }) {
  if (error) return <td className="px-2 py-1 text-invalid" title={error.message}>⚠ {error.code}</td>
  if (!pred) return <td className="px-2 py-1 text-muted">{queued ? '⏳ queued' : ''}</td>
  const d = pred.dimensions[dim]
  if (d.label === null) return <td className="px-2 py-1 text-invalid" title={S.invalidOutput}>⚠ invalid</td>
  if (!d.applicable)
    return <td className="px-2 py-1 text-muted" title={`${S.notApplicable}. Internal value: ${d.label}`} data-testid={`cell-${dim}`}>- <span className="text-xs">n/a</span></td>
  return (
    <td className="px-2 py-1" data-testid={`cell-${dim}`}>
      <span title={`internal value: ${d.label}`}>{d.erp_label}</span>{' '}
      <ConfidenceChip value={d.confidence} method={d.confidence_method} />
    </td>
  )
}

export default function QueuePage() {
  const { sessionId, error: sessionError } = useSessionId()
  const models = useModels()
  const settings = useSettings()
  const images = useImages(sessionId)
  const qc = useQueryClient()
  const navigate = useNavigate()
  const toast = useToast()
  const { query } = useGlobalSearch()

  const modelList = models.data?.models ?? []
  const [modelId, setModelId] = useState<string | null>(null)
  const [mode, setMode] = useState<'live' | 'recorded'>('live')
  useEffect(() => {
    if (!modelId && modelList.length) {
      const first = modelList.find((m) => m.modes.length) ?? modelList[0]
      setModelId(first.model_id)
      setMode(defaultMode(first))
    }
  }, [modelId, modelList])
  const model = modelList.find((m) => m.model_id === modelId)
  const preds = usePredictions(sessionId, modelId)
  const latest = useMemo(() => latestByImage(preds.data), [preds.data])
  const { state: run, run: startRun } = useRun(sessionId)

  const [sourceKind, setSourceKind] = useState('own_photo')
  const [uploading, setUploading] = useState(false)
  const [rejected, setRejected] = useState<{ original_filename: string; error: ApiErrorBody }[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [sortByConf, setSortByConf] = useState(false)

  const upload = useCallback(async (files: File[]) => {
    if (!sessionId || !files.length) return
    setUploading(true)
    setRejected([])
    const form = new FormData()
    form.set('session_id', sessionId)
    form.set('source_kind', sourceKind)
    files.forEach((f) => form.append('files', f))
    try {
      const res = await api<UploadResponse>('/images', { method: 'POST', body: form })
      setRejected(res.rejected)
      toast('ok', `${res.images.length} image(s) added${res.rejected.length ? `, ${res.rejected.length} rejected` : ''}`)
    } catch (e) {
      if (e instanceof ApiClientError && Array.isArray(e.details.rejected)) setRejected(e.details.rejected as typeof rejected)
      else setRejected([{ original_filename: files.map((f) => f.name).join(', '), error: { code: (e as ApiClientError).code ?? 'ERROR', message: errorText(e) } }])
      toast('error', errorText(e))
    } finally {
      setUploading(false)
      qc.invalidateQueries({ queryKey: ['images'] })
    }
  }, [sessionId, sourceKind, qc, toast])
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ onDrop: upload, noClick: true, noKeyboard: true })

  const rows = useMemo(() => {
    let items = (images.data ?? []).filter((i) => i.original_filename.toLowerCase().includes(query.toLowerCase()))
    if (sortByConf)
      items = [...items].sort((a, b) => (minConfidence(latest.get(a.image_id)) ?? 9) - (minConfidence(latest.get(b.image_id)) ?? 9))
    return items
  }, [images.data, query, sortByConf, latest])

  const runnable = isRunnable(model, mode)
  const onRun = async () => {
    if (!model || !images.data?.length) return
    const all = images.data.map((i) => i.image_id)
    const lacking = all.filter((id) => !latest.has(id))
    const ids = selected.size ? [...selected] : lacking.length ? lacking : all
    try {
      await startRun(model.model_id, ids, mode, settings.data?.default_use_context ?? true)
    } catch (e) {
      toast('error', errorText(e))
    }
  }

  const statusOf = (img: ImageItem): 'ai_suggested' | 'reviewed' | 'corrected' | 'queued' | 'none' => {
    if (img.review_status) return img.review_status
    if (latest.has(img.image_id)) return 'ai_suggested'
    return run.running && run.modelId === modelId ? 'queued' : 'none'
  }
  const noContext = (images.data ?? []).filter((i) => Object.keys(i.context).length === 0).length
  const queueParam = `?session=${sessionId ?? ''}`
  const compareIds = modelList.filter((m) => m.modes.length).map((m) => m.model_id).join(',')

  return (
    <ErpShell>
      <TitleBar title="Image classification (AI)" id={sessionId ? undefined : undefined}>
        <span className="text-sm opacity-90">Session {sessionId ? sessionId.slice(4, 12) : '…'}</span>
      </TitleBar>
      <SubLinks links={[
        { to: `/image-classification${queueParam}`, label: 'Queue' },
        { to: `/image-classification/compare${queueParam}&models=${compareIds}`, label: 'Compare models' },
        { to: '/image-classification/performance?set=coverage', label: 'Performance' },
        { to: `/storefront-preview${queueParam}${modelId ? `&model=${modelId}` : ''}`, label: 'Storefront preview' },
      ]} />
      <ActionToolbar>
        {modelList.length > 0 && <ModelSelect models={modelList} value={modelId} onChange={(id) => { setModelId(id); setMode(defaultMode(modelList.find((m) => m.model_id === id))) }} />}
        {model && model.modes.length > 1 && (
          <select aria-label="Mode" value={mode} onChange={(e) => setMode(e.target.value as 'live' | 'recorded')} className="rounded border border-line bg-panel px-2 py-1 text-sm">
            <option value="live">Live</option><option value="recorded">Recorded</option>
          </select>
        )}
        <ToolbarButton testId="run-classification" onClick={onRun} disabled={!runnable || run.running || !images.data?.length}
          title={!runnable ? 'Model not available in this mode' : undefined}>
          ▶ Run classification
        </ToolbarButton>
        <Link to={`/image-classification/compare${queueParam}&models=${compareIds}`} className="rounded border border-line bg-toolbtn px-3 py-1 text-sm no-underline">Compare models</Link>
        <a href={`/api/export.csv?session_id=${sessionId}${modelId ? `&model_id=${modelId}` : ''}`} className="rounded border border-line bg-toolbtn px-3 py-1 text-sm no-underline">Export CSV</a>
      </ActionToolbar>

      <div className="space-y-3 p-4">
        {model && !runnable && <ModelUnavailableState model={model} />}
        {model && model.family === 'zeroshot_generative' && runnable && (
          <p className="m-0 text-xs text-muted">Zero-shot via the shared gateway: about 8–11 s per image (measured on the P4 evaluation). Trained models are much faster.</p>
        )}
        {sessionError != null && <ErrorState error={sessionError} />}

        <div {...getRootProps()} data-testid="dropzone"
          className={`rounded border-2 border-dashed p-4 text-sm ${isDragActive ? 'border-link bg-imagetype' : 'border-line bg-panel'}`}>
          <input {...getInputProps()} data-testid="file-input" />
          <div className="flex flex-wrap items-center gap-3">
            <span>Drop JPEG / PNG / WebP images here (max 20 MB, max 50 at a time) or</span>
            <ToolbarButton onClick={open}>Choose files</ToolbarButton>
            {uploading && <span className="text-muted">Uploading…</span>}
          </div>
          <fieldset className="mt-2 flex flex-wrap items-center gap-3 border-0 p-0 text-xs">
            <legend className="sr-only">Source</legend>
            <span className="text-muted">Source:</span>
            {[['own_photo', 'our photos'], ['train_coverage', 'train / coverage'], ['other', 'other']].map(([v, l]) => (
              <label key={v}><input type="radio" name="source" checked={sourceKind === v} onChange={() => setSourceKind(v)} /> {l}</label>
            ))}
            <span className="font-semibold text-invalid">⚠ {S.neverDevFinal}</span>
          </fieldset>
          {rejected.length > 0 && (
            <ul data-testid="rejected-list" className="m-0 mt-2 list-none p-0 text-xs text-invalid">
              {rejected.map((r, i) => <li key={i}>✕ {r.original_filename}: {r.error.message} ({r.error.code})</li>)}
            </ul>
          )}
        </div>

        {(run.running || run.batch || run.total > 0) && (
          <div className="flex flex-wrap items-center gap-3" data-testid="run-status">
            {run.running || run.status === null ? <ProgressBar done={run.done} total={run.total} label={run.running ? 'classifying…' : undefined} /> : <ProgressBar done={run.done} total={run.total} label={run.status} />}
            <ThroughputBadge batch={run.batch} target={models.data?.throughput_target_images_per_second ?? 2} />
            {run.batch?.recorded && <ModeBadge mode="recorded" recorded={run.batch.recorded} />}
            {run.jobError && <span className="text-xs text-invalid">{run.jobError.message} ({run.jobError.code})</span>}
          </div>
        )}

        {images.isLoading ? null : rows.length === 0 ? (
          <EmptyState>No images yet: drag your photos here.</EmptyState>
        ) : (
          <>
            <table className="w-full border-collapse bg-panel text-left" data-testid="results-table" style={{ fontSize: 'var(--fs-value)' }}>
              <thead>
                <tr className="border-b border-line text-muted" style={{ fontSize: 'var(--fs-label)' }}>
                  <th className="px-2 py-1"><input type="checkbox" aria-label="Select all" checked={selected.size === rows.length} onChange={(e) => setSelected(e.target.checked ? new Set(rows.map((r) => r.image_id)) : new Set())} /></th>
                  <th className="px-2 py-1">Image</th>
                  {DIMENSIONS.map((d) => <th key={d} className="px-2 py-1">{DIMENSION_NAMES[d]}</th>)}
                  <th className="px-2 py-1"><button type="button" data-testid="sort-minconf" onClick={() => setSortByConf((s) => !s)} className="border-0 bg-transparent p-0 font-semibold">Min conf {sortByConf ? '▲' : '↕'}</button></th>
                  <th className="px-2 py-1">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((img) => {
                  const p = latest.get(img.image_id)
                  const mc = minConfidence(p)
                  return (
                    <tr key={img.image_id} data-testid="result-row" className="cursor-pointer border-b border-line hover:bg-imagetype"
                      onClick={() => navigate(`/image-classification/${img.image_id}?model=${modelId ?? ''}&session=${sessionId}`)}>
                      <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" aria-label={`Select ${img.original_filename}`} checked={selected.has(img.image_id)}
                          onChange={(e) => setSelected((s) => { const n = new Set(s); if (e.target.checked) n.add(img.image_id); else n.delete(img.image_id); return n })} />
                      </td>
                      <td className="px-2 py-1">
                        <div className="flex items-center gap-2">
                          <img src={img.thumbnail_url} alt="" width={40} height={40} className="h-10 w-10 rounded object-cover" />
                          <span className="max-w-32 truncate" title={img.original_filename}>{img.original_filename}</span>
                          {p && <ModeBadge mode={p.mode} recorded={p.recorded} compact />}
                        </div>
                      </td>
                      {DIMENSIONS.map((d) => <Cell key={d} pred={p} dim={d} error={run.errors[img.image_id]} queued={run.running && run.modelId === modelId} />)}
                      <td className="px-2 py-1">{mc === null ? <span className="text-muted">—</span> : mc.toFixed(2)}</td>
                      <td className="px-2 py-1"><StatusPill status={statusOf(img)} /></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {noContext > 0 && (
              <p className="m-0 text-xs text-muted" data-testid="context-warning">
                Product context missing for {noContext}/{images.data?.length} images — {S.contextMissing}{' '}
                <Link to={`/image-classification/${rows[0].image_id}?model=${modelId ?? ''}&session=${sessionId}`}>Add context</Link>
              </p>
            )}
            <p className="m-0 text-xs text-muted">Metrics for this model live elsewhere: <Link to="/image-classification/performance?set=coverage">how does this model perform on test sets →</Link></p>
          </>
        )}
      </div>
    </ErpShell>
  )
}
