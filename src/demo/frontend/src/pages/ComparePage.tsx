import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, errorText } from '../api/client'
import { useImages, useModels, usePredictions, useSessionId } from '../api/hooks'
import type { PredictResponse, PredictionResult, Throughput } from '../api/types'
import { subscribeJob } from '../api/sse'
import { ActionToolbar, ErpShell, SubLinks, TitleBar, ToolbarButton, useGlobalSearch } from '../components/shell'
import { ConfidenceChip, ConfidenceMethodNote, EmptyState, ModeBadge, modelUnavailableText, useToast } from '../components/ui'
import { defaultMode } from '../lib/predictions'
import { DIMENSIONS, DIMENSION_NAMES, S } from '../strings'

export default function ComparePage() {
  const { sessionId } = useSessionId()
  const [params, setParams] = useSearchParams()
  const models = useModels()
  const images = useImages(sessionId)
  const preds = usePredictions(sessionId)
  const thr = useQuery({ queryKey: ['throughput'], queryFn: () => api<Throughput>('/throughput') })
  const navigate = useNavigate()
  const qc = useQueryClient()
  const toast = useToast()
  const { query } = useGlobalSearch()
  const [onlyDisagreements, setOnlyDisagreements] = useState(false)
  const [dimFilter, setDimFilter] = useState<string>('all')
  const [busy, setBusy] = useState<string | null>(null)

  const modelList = models.data?.models ?? []
  const requested = (params.get('models') ?? '').split(',').filter(Boolean)
  const selected = requested.length ? requested : modelList.filter((m) => m.modes.length).map((m) => m.model_id)
  const setSelected = (ids: string[]) => setParams((p) => { p.set('models', ids.join(',')); return p })

  const byKey = useMemo(() => {
    const map = new Map<string, PredictionResult>()
    for (const p of preds.data ?? []) map.set(`${p.image_id}|${p.model_id}`, p)
    return map
  }, [preds.data])

  const activeModels = modelList.filter((m) => selected.includes(m.model_id) && m.modes.length > 0)
  const withResults = activeModels.filter((m) => (images.data ?? []).some((i) => byKey.has(`${i.image_id}|${m.model_id}`)))
  const dims = dimFilter === 'all' ? DIMENSIONS : [dimFilter]

  const disagreement = (imageId: string, dim: string) => {
    const labels = withResults.map((m) => byKey.get(`${imageId}|${m.model_id}`)?.dimensions[dim]?.label).filter((l) => l !== undefined && l !== null)
    return new Set(labels).size >= 2
  }
  const rows = (images.data ?? [])
    .filter((i) => i.original_filename.toLowerCase().includes(query.toLowerCase()))
    .map((i) => ({ image: i, disagreeing: DIMENSIONS.filter((d) => disagreement(i.image_id, d)) }))
    .filter((r) => !onlyDisagreements || r.disagreeing.some((d) => dims.includes(d)))

  const computeMissing = async () => {
    for (const m of activeModels) {
      const missing = (images.data ?? []).filter((i) => !byKey.has(`${i.image_id}|${m.model_id}`)).map((i) => i.image_id)
      if (!missing.length) continue
      setBusy(m.display_name)
      try {
        const res = await api<PredictResponse>('/predict', { method: 'POST', json: { model_id: m.model_id, image_ids: missing, mode: defaultMode(m), use_context: true } })
        if (res.queued > 0) await new Promise<void>((resolve) => { subscribeJob(res.job_id, res.total, { onDone: () => resolve() }) })
        await qc.invalidateQueries({ queryKey: ['predictions'] })
      } catch (e) {
        toast('error', `${m.display_name}: ${errorText(e)}`)
      }
    }
    setBusy(null)
    qc.invalidateQueries({ queryKey: ['throughput'] })
  }

  const latency = (id: string) => {
    const m = thr.data?.measured_in_demo.find((x) => x.model_id === id)
    if (!m) return 'no measurement yet'
    const sec = m.ms_per_image / 1000
    return `${sec >= 1 ? `${sec.toFixed(1)} s/img` : `${Math.round(m.ms_per_image)} ms/img`} (${m.mode === 'recorded' ? 'recorded run' : m.gpu_name})`
  }
  const back = `?session=${sessionId ?? ''}`
  const methods = new Set(withResults.map((m) => m.confidence?.method))

  return (
    <ErpShell>
      <TitleBar title="Compare models" />
      <SubLinks links={[{ to: `/image-classification${back}`, label: 'Queue' }, { to: `/image-classification/compare${back}`, label: 'Compare models' }, { to: `/image-classification/try${back}`, label: 'Try a model' }, { to: `/image-classification/models${back}`, label: 'Models' }, { to: '/image-classification/performance?set=coverage', label: 'Performance' }]} />
      <ActionToolbar>
        <div className="flex flex-wrap items-center gap-3" data-testid="model-toggles">
          {modelList.map((m) => {
            const usable = m.modes.length > 0
            return (
              <label key={m.model_id} className={`text-sm ${usable ? '' : 'text-muted'}`} title={usable ? undefined : modelUnavailableText(m)}>
                <input type="checkbox" disabled={!usable} checked={selected.includes(m.model_id) && usable}
                  onChange={(e) => setSelected(e.target.checked ? [...selected, m.model_id] : selected.filter((x) => x !== m.model_id))} /> {m.display_name.split('·').slice(0, 2).join('·').trim()}
                {!usable && <span className="text-xs"> (unavailable)</span>}
              </label>
            )
          })}
        </div>
        <ToolbarButton testId="compute-missing" onClick={computeMissing} disabled={!!busy || activeModels.length === 0}>{busy ? `Computing ${busy}…` : '▶ Compute missing'}</ToolbarButton>
      </ActionToolbar>

      <div className="space-y-3 p-4">
        <div data-testid="latency-strip" className="text-xs text-muted">
          Latency measured here: {activeModels.map((m) => `${m.display_name.split('·').slice(0, 2).join('·').trim()}: ${latency(m.model_id)}`).join(' · ') || 'no model selected'}
        </div>
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label><input type="radio" checked={!onlyDisagreements} onChange={() => setOnlyDisagreements(false)} /> all</label>
          <label><input type="radio" data-testid="only-disagreements" checked={onlyDisagreements} onChange={() => setOnlyDisagreements(true)} /> only disagreements</label>
          <label>Dimension:{' '}
            <select value={dimFilter} onChange={(e) => setDimFilter(e.target.value)} className="rounded border border-line bg-panel px-2 py-1">
              <option value="all">all</option>{DIMENSIONS.map((d) => <option key={d} value={d}>{DIMENSION_NAMES[d]}</option>)}
            </select>
          </label>
        </div>

        {withResults.length < 2 ? (
          <EmptyState>Select at least two models with predictions (use “Compute missing”).</EmptyState>
        ) : rows.length === 0 ? (
          <EmptyState>No images match the filter.</EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="border-collapse bg-panel text-left" data-testid="comparison-grid" style={{ fontSize: 'var(--fs-value)' }}>
              <thead>
                <tr className="border-b border-line text-muted" style={{ fontSize: 'var(--fs-label)' }}>
                  <th rowSpan={2} className="px-2 py-1">Image</th>
                  {dims.map((d) => <th key={d} colSpan={withResults.length} className="border-l border-line px-2 py-1 text-center">{DIMENSION_NAMES[d]}</th>)}
                  <th rowSpan={2} className="border-l border-line px-2 py-1">Disagreements</th>
                </tr>
                <tr className="border-b border-line text-muted" style={{ fontSize: 'var(--fs-label)' }}>
                  {dims.flatMap((d) => withResults.map((m, i) => (
                    <th key={`${d}-${m.model_id}`} className={`px-2 py-1 font-normal ${i === 0 ? 'border-l border-line' : ''}`} title={m.display_name}>{m.display_name.split('·')[0].replace('Apertus ', '').trim()} {m.display_name.includes('zero-shot') ? 'zs' : m.display_name.includes('full fine-tune') ? 'FT' : m.display_name.includes('P11') ? 'P11' : m.display_name.includes('P10') ? 'P10' : ''}</th>
                  )))}
                </tr>
              </thead>
              <tbody>
                {rows.map(({ image, disagreeing }) => (
                  <tr key={image.image_id} data-testid="compare-row" className="border-b border-line">
                    <td className="px-2 py-1"><div className="flex items-center gap-2"><img src={image.thumbnail_url} alt="" width={36} height={36} className="h-9 w-9 rounded object-cover" /><span className="max-w-28 truncate" title={image.original_filename}>{image.original_filename}</span></div></td>
                    {dims.flatMap((d) => withResults.map((m, i) => {
                      const p = byKey.get(`${image.image_id}|${m.model_id}`)
                      const dim = p?.dimensions[d]
                      const hot = disagreeing.includes(d)
                      return (
                        <td key={`${d}-${m.model_id}`} data-disagree={hot ? '1' : '0'} data-testid="compare-cell"
                          onClick={() => navigate(`/image-classification/${image.image_id}?model=${m.model_id}&session=${sessionId}`)}
                          className={`cursor-pointer px-2 py-1 ${i === 0 ? 'border-l border-line' : ''} ${hot ? 'bg-conf-mid-bg' : ''}`}
                          title={dim?.label ? `internal value: ${dim.label}` : undefined}>
                          {!p ? <span className="text-muted">—</span> : dim?.label === null ? <span className="text-invalid">invalid</span> :
                            dim && !dim.applicable ? <span className="text-muted" title={S.notApplicable}>-</span> :
                            <span>{dim?.erp_label} <ConfidenceChip value={dim?.confidence ?? null} method={dim?.confidence_method} /></span>}
                        </td>
                      )
                    }))}
                    <td className="border-l border-line px-2 py-1" data-testid="disagreement-count">{disagreeing.length}/6</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          {withResults.map((m) => {
            const p = (preds.data ?? []).find((x) => x.model_id === m.model_id)
            return p ? <ModeBadge key={m.model_id} mode={p.mode} recorded={p.recorded} compact /> : null
          })}
          <span>{S.confidence.notComparable}</span>
          {methods.size > 0 && [...methods].map((x) => <ConfidenceMethodNote key={x} method={x ?? undefined} />)}
        </div>
        <p className="m-0 text-xs text-muted">Metrics for each model: <Link to="/image-classification/performance?set=coverage">how do these models perform on test sets →</Link></p>
      </div>
    </ErpShell>
  )
}
