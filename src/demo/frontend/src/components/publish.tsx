import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAttachImage, useCatalog, useCreateProduct } from '../api/catalog'
import { errorText } from '../api/client'
import type { ImageItem } from '../api/types'
import { ruleBlockers } from '../lib/catalog'
import { useToast } from './ui'

/** Dalla risposta del modello allo scaffale: crea o riusa un prodotto, pubblica la foto, porta in vetrina. */
export function PublishPanel({ image, hasAnswer }: { image: ImageItem; hasAnswer: boolean }) {
  const catalog = useCatalog('', null)
  const create = useCreateProduct()
  const attach = useAttachImage()
  const toast = useToast()
  const [mode, setMode] = useState<'new' | 'existing'>('new')
  const [existingId, setExistingId] = useState('')
  const [title, setTitle] = useState(image.context.product_title || image.original_filename.replace(/\.[^.]+$/, ''))
  const [brand, setBrand] = useState(image.context.brand ?? '')
  const [category, setCategory] = useState(image.context.category ?? '')
  const [result, setResult] = useState<{ productId: string; title: string; hidden: string[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const busy = create.isPending || attach.isPending
  const existing = catalog.data?.products ?? []
  const canSubmit = hasAnswer && !busy && (mode === 'existing' ? !!existingId : !!title.trim())

  const submit = async () => {
    if (!canSubmit) return
    setError(null)
    try {
      let productId: string
      let productTitle: string
      let isCover: boolean
      if (mode === 'existing') {
        const product = existing.find((p) => p.product_id === existingId)
        if (!product) return
        productId = product.product_id
        productTitle = product.title
        isCover = product.published_count === 0
      } else {
        const created = await create.mutateAsync({
          title: title.trim(),
          brand: brand.trim() || undefined,
          category: category.trim() || undefined,
          subcategory: image.context.subcategory || undefined,
          product_type: image.context.product_type || undefined,
        })
        productId = created.product.product_id
        productTitle = created.product.title
        isCover = created.images.length === 0
      }
      const detail = await attach.mutateAsync({ imageId: image.image_id, productId, isCover })
      const attached = detail.images.find((i) => i.image_id === image.image_id)
      const hidden = attached ? ruleBlockers(attached.values, attached.confidence, detail.filters) : []
      setResult({ productId, title: productTitle, hidden })
      toast('ok', `“${productTitle}” is now in the catalog.`)
    } catch (e) {
      setError(errorText(e))
    }
  }

  return (
    <section className="mt-8 border-t-2 border-signal pt-4" data-testid="publish-panel">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 text-[20px] font-bold tracking-tight">Add this photo to the storefront</h2>
        <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">From answer to shelf</span>
      </div>
      <p className="m-0 mt-1 max-w-[70ch] text-[14px] leading-relaxed text-slate">
        The six answers become the catalog values. Give the product a name, or add the photo to an existing product as another view.
      </p>

      {result ? (
        <div className="mt-4 border border-graphite bg-white p-4" data-testid="publish-done">
          <p className="m-0 text-[16px] font-bold">“{result.title}” is in the catalog.</p>
          {result.hidden.length > 0 ? (
            <p className="m-0 mt-1 text-[14px] leading-relaxed text-signal" data-testid="publish-hidden">
              It is published, but the current shop rules hide it: {result.hidden.join(', ')}.
              {' '}<Link to="/catalog" className="font-bold">Change the rules</Link> to put it on the shelf.
            </p>
          ) : (
            <p className="m-0 mt-1 text-[14px] text-slate" data-testid="publish-visible">
              It is published and visible in the storefront right now.
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-3">
            <Link to={`/shop/${result.productId}`} data-testid="publish-open-shop"
              className="bg-graphite px-4 py-2 text-[14px] font-bold text-white no-underline hover:bg-signal">
              Open the product page →
            </Link>
            <Link to={`/shop?q=${encodeURIComponent(result.title)}`} data-testid="publish-search-shop"
              className="border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">
              Search the shop
            </Link>
            <Link to={`/catalog/${result.productId}`}
              className="border border-rule px-4 py-2 text-[14px] font-bold no-underline hover:border-graphite">
              Admin view
            </Link>
          </div>
        </div>
      ) : !hasAnswer ? (
        <p className="m-0 mt-3 border border-rule px-3 py-2 text-[14px] text-slate" data-testid="publish-needs-answer">
          Classify the photo first: the shop uses the model’s answers as catalog values.
        </p>
      ) : (
        <div className="mt-3 space-y-3">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Where to add the photo">
            <button type="button" onClick={() => setMode('new')} aria-pressed={mode === 'new'}
              className={`border px-3 py-1.5 text-[13px] font-bold ${mode === 'new' ? 'border-graphite bg-graphite text-white' : 'border-rule bg-white hover:border-graphite'}`}>
              New product
            </button>
            <button type="button" onClick={() => setMode('existing')} aria-pressed={mode === 'existing'}
              disabled={existing.length === 0}
              className={`border px-3 py-1.5 text-[13px] font-bold disabled:opacity-40 ${mode === 'existing' ? 'border-graphite bg-graphite text-white' : 'border-rule bg-white hover:border-graphite'}`}>
              Another view of an existing product ({existing.length})
            </button>
          </div>

          {mode === 'new' ? (
            <div className="grid gap-2 sm:grid-cols-3">
              <label className="flex flex-col text-[12px] text-slate sm:col-span-3">Product title
                <input value={title} onChange={(e) => setTitle(e.target.value)} data-testid="publish-title"
                  className="mt-0.5 border border-rule px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite" />
              </label>
              <label className="flex flex-col text-[12px] text-slate">Brand
                <input value={brand} onChange={(e) => setBrand(e.target.value)} data-testid="publish-brand"
                  className="mt-0.5 border border-rule px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite" />
              </label>
              <label className="flex flex-col text-[12px] text-slate sm:col-span-2">Category
                <input value={category} onChange={(e) => setCategory(e.target.value)} data-testid="publish-category"
                  className="mt-0.5 border border-rule px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite" />
              </label>
            </div>
          ) : (
            <label className="flex flex-col text-[12px] text-slate">Existing product
              <select value={existingId} onChange={(e) => setExistingId(e.target.value)} data-testid="publish-existing"
                className="mt-0.5 border border-rule bg-white px-2.5 py-1.5 text-[15px] text-graphite focus:border-graphite">
                <option value="">Choose a product…</option>
                {existing.map((p) => (
                  <option key={p.product_id} value={p.product_id}>
                    {p.title}{p.brand ? ` — ${p.brand}` : ''} ({p.image_count} {p.image_count === 1 ? 'photo' : 'photos'})
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <button type="button" onClick={submit} disabled={!canSubmit} data-testid="publish-submit"
              className="bg-signal px-5 py-2 text-[14px] font-bold text-white hover:bg-graphite disabled:cursor-not-allowed disabled:opacity-40">
              {busy ? 'Publishing…' : mode === 'existing' ? 'Add as another view' : 'Publish to the storefront'}
            </button>
            <span className="text-[12px] text-slate">
              {mode === 'new' ? 'The first photo of a product becomes its cover.' : 'The photo keeps its own perspective in the gallery.'}
            </span>
          </div>
          {error && <p className="m-0 text-[14px] text-signal" role="alert">{error}</p>}
        </div>
      )}
    </section>
  )
}
