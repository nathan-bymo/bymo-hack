import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useImages, useModels, usePredictions, useSessionId, useSettings } from '../api/hooks'
import type { StorefrontRule } from '../api/types'
import { DemoBadge, Wordmark } from '../components/shell'
import { latestByImage } from '../lib/predictions'
import { applyRule } from '../lib/storefront'

export default function StorefrontPage() {
  const { sessionId } = useSessionId()
  const [params, setParams] = useSearchParams()
  const models = useModels()
  const settings = useSettings()
  const images = useImages(sessionId)
  const modelList = models.data?.models ?? []
  const modelId = params.get('model') ?? modelList.find((m) => m.modes.length)?.model_id ?? null
  const preds = usePredictions(sessionId, modelId)
  const [override, setOverride] = useState<StorefrontRule | null>(null)
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState('')
  const [parseError, setParseError] = useState<string | null>(null)
  const rule = override ?? settings.data?.storefront_rule
  const placed = useMemo(
    () => (rule && images.data ? applyRule(rule, images.data, latestByImage(preds.data)) : []),
    [rule, images.data, preds.data],
  )
  const main = placed.find((p) => p.position === 1)
  const apply = () => {
    try { setOverride(JSON.parse(text)); setParseError(null) } catch (e) { setParseError((e as Error).message) }
  }

  return (
    <div className="min-h-screen bg-white text-ink" data-testid="storefront">
      <div aria-hidden="true" style={{ height: 3, background: 'linear-gradient(90deg,#e5533d 0 25%,#5db85c 25% 50%,#8e6fd0 50% 75%,#4a90d9 75%)' }} />
      <header className="flex items-center gap-4 bg-white px-4 py-3">
        <Wordmark variant="store" />
        <input aria-label="Search" placeholder="Wonach suchst du?" className="w-full max-w-2xl rounded-full border border-fieldline bg-white px-4 py-1.5 text-sm" />
        <div className="ml-auto"><DemoBadge /></div>
      </header>
      <div className="mx-auto max-w-6xl p-6">
        <nav className="mb-4 text-sm text-muted" aria-label="Categories">Categories › Demo product › <strong className="text-ink">Bilder</strong>
          {' · '}<Link to={`/image-classification?session=${sessionId ?? ''}`}>← Queue</Link></nav>
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <h1 className="m-0 text-xl font-semibold">Proposed gallery</h1>
          <span data-testid="rule-label" className="rounded bg-conf-mid-bg px-2 py-0.5 text-xs font-semibold text-conf-mid">{rule?.label ?? 'Illustrative rule'}</span>
          <label className="text-sm">Model:{' '}
            <select value={modelId ?? ''} onChange={(e) => setParams((p) => { p.set('model', e.target.value); return p })} className="rounded border border-line px-2 py-1">
              {modelList.filter((m) => m.modes.length).map((m) => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}
            </select>
          </label>
          <button type="button" onClick={() => { setText(JSON.stringify(rule, null, 2)); setEditing((e) => !e) }} className="rounded border border-line bg-toolbtn px-3 py-1 text-sm">Edit rule</button>
        </div>
        <p className="m-0 mb-4 text-xs text-muted">Not the Galaxus rule: an illustration of what the six Image-type values could drive in a shop (idea: {rule?.idea_source}). Human-reviewed values win over the model's.</p>
        {editing && (
          <div className="mb-4" data-testid="rule-editor">
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={10} className="w-full rounded border border-line p-2 font-mono text-xs" aria-label="Rule (JSON)" />
            <button type="button" onClick={apply} className="rounded border border-line bg-toolbtn px-3 py-1 text-sm">Apply (this page only)</button>
            {parseError && <span className="ml-2 text-xs text-invalid">{parseError}</span>}
          </div>
        )}
        {placed.length === 0 ? <p className="text-muted" data-testid="empty-state">No images yet: upload and classify some in the queue.</p> : (
          <>
            <ol className="m-0 flex list-none flex-wrap gap-3 p-0" data-testid="gallery">
              {placed.map((p) => (
                <li key={p.image.image_id} data-position={p.position ?? 'hidden'} data-testid="gallery-item"
                  className={`w-40 border-r border-line p-3 ${p.position === null ? 'opacity-50 grayscale' : ''}`}>
                  <div className="mb-1 text-xs font-semibold">{p.position === 1 ? '#1 MAIN' : p.position ? `#${p.position}` : 'hidden'}</div>
                  <img src={p.image.thumbnail_url} alt={p.image.original_filename} className="h-32 w-full object-cover" />
                  <div className="mt-1 text-xs text-muted">{p.group ?? '—'}</div>
                </li>
              ))}
            </ol>
            <div className="mt-4 text-sm">
              <div className="text-muted">Brand · Price n/a · Demo product · ☆☆☆☆☆ n/a</div>
              <button type="button" disabled className="mt-2 rounded-full px-6 py-2 text-white opacity-90" style={{ background: 'var(--c-store-pill)' }} title="Not active in this demo">Mehr anzeigen</button>
            </div>
            <section className="mt-6" data-testid="rule-explainer">
              <h2 className="section-title m-0 mb-1 text-sm">Why this order</h2>
              <ul className="m-0 pl-5 text-sm">
                {placed.map((p) => <li key={p.image.image_id}><strong>{p.position ? `#${p.position}` : 'hidden'}</strong> {p.image.original_filename}: {p.reason}</li>)}
              </ul>
              {main && <p className="text-sm" data-testid="why-main">Why #1: {main.reason}</p>}
            </section>
          </>
        )}
      </div>
    </div>
  )
}
