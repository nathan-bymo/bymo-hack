import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useImages, useModels, usePredictions, useSessionId, useSettings } from '../api/hooks'
import type { StorefrontRule } from '../api/types'
import { ErpShell, SubLinks, TitleBar, ToolbarButton } from '../components/shell'
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
    <ErpShell>
      <div data-testid="storefront">
        <TitleBar title="Shop preview">
          <span data-testid="rule-label" className="bg-canvas px-2 py-1 text-[13px] font-bold">{rule?.label ?? 'Illustrative rule'}</span>
        </TitleBar>
        <SubLinks links={[{ to: `/image-classification?session=${sessionId ?? ''}`, label: '← Queue' }]} />
        <p className="m-0 max-w-[70ch] text-[15px] text-slate">
          What the six answers could do in a shop: choose the main photo and the order of the gallery automatically.
          This is our illustration, not the Galaxus rule. Answers checked by a person win over the model’s.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-3 text-[15px]">
          <label className="flex items-center gap-3"><span className="font-bold">Model</span>
            <select value={modelId ?? ''} onChange={(e) => setParams((p) => { p.set('model', e.target.value); return p })} className="border border-graphite bg-white px-3 py-1.5 text-[15px]">
              {modelList.filter((m) => m.modes.length).map((m) => <option key={m.model_id} value={m.model_id}>{m.short_name}</option>)}
            </select>
          </label>
          <ToolbarButton onClick={() => { setText(JSON.stringify(rule, null, 2)); setEditing((e) => !e) }}>{editing ? 'Close the rule' : 'Edit the rule'}</ToolbarButton>
        </div>
        {editing && (
          <div className="mt-4" data-testid="rule-editor">
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={10} className="w-full border border-rule p-3 font-mono text-[13px]" aria-label="Rule (JSON)" />
            <div className="mt-2 flex items-center gap-3">
              <ToolbarButton onClick={apply}>Apply on this page</ToolbarButton>
              {parseError && <span className="text-[14px] text-signal">{parseError}</span>}
            </div>
          </div>
        )}
        {placed.length === 0 ? <p className="mt-8 text-slate" data-testid="empty-state">No photos yet: add and classify some in the review queue.</p> : (
          <>
            <ol className="m-0 mt-8 grid list-none grid-cols-2 gap-6 p-0 sm:grid-cols-3 lg:grid-cols-5" data-testid="gallery">
              {placed.map((p) => (
                <li key={p.image.image_id} data-position={p.position ?? 'hidden'} data-testid="gallery-item"
                  className={p.position === null ? 'opacity-40 grayscale' : ''}>
                  <div className={`mb-2 text-[14px] ${p.position === 1 ? 'font-bold' : 'text-slate'}`}>{p.position === 1 ? 'Main photo' : p.position ? `Position ${p.position}` : 'Hidden'}</div>
                  <div className={`aspect-square bg-canvas ${p.position === 1 ? 'outline-2 outline-graphite' : ''}`}>
                    <img src={p.image.thumbnail_url} alt={p.image.original_filename} className="h-full w-full object-contain" />
                  </div>
                  <div className="mt-1.5 text-[13px] text-slate">{p.group ?? '—'}</div>
                </li>
              ))}
            </ol>
            <section className="mt-10 border-t border-rule pt-4" data-testid="rule-explainer">
              <h2 style={{ fontSize: 20 }}>Why this order</h2>
              <ul className="m-0 mt-3 list-none space-y-1 p-0 text-[15px]">
                {placed.map((p) => <li key={p.image.image_id}><b>{p.position ? `${p.position}.` : 'Hidden:'}</b> {p.image.original_filename}: {p.reason}</li>)}
              </ul>
              {main && <p className="mt-3 text-[15px] text-slate" data-testid="why-main">Main photo: {main.reason}</p>}
            </section>
          </>
        )}
      </div>
    </ErpShell>
  )
}
