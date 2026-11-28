import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useStorefrontProduct } from '../api/catalog'
import { ApiClientError } from '../api/client'
import { AppShell } from '../components/app'
import { Breadcrumbs, Eyebrow, ValueTable } from '../components/catalog-ui'
import { VIEW_ORDER_RULE, galleryOrder, perspectiveLabel } from '../lib/catalog'

export default function ShopProductPage() {
  const { productId } = useParams()
  const detail = useStorefrontProduct(productId)
  const images = useMemo(() => galleryOrder(detail.data?.images ?? []), [detail.data])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = images.find((i) => i.image_id === selectedId) ?? images[0]
  const product = detail.data?.product

  /** Frecce: passano alla vista precedente o successiva, e dall'ultima si torna alla prima. */
  const go = useCallback((delta: number) => {
    if (images.length < 2) return
    const current = Math.max(0, images.findIndex((i) => i.image_id === selectedId))
    setSelectedId(images[(current + delta + images.length) % images.length].image_id)
  }, [images, selectedId])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return
      if (e.key === 'ArrowLeft') go(-1)
      else if (e.key === 'ArrowRight') go(1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go])

  if (detail.isError) {
    const hidden = detail.error instanceof ApiClientError && detail.error.code === 'NOT_FOUND'
    return (
      <AppShell>
        <Breadcrumbs items={[{ label: 'Storefront', to: '/shop' }, { label: 'Product' }]} />
        <div className="mt-6 border border-rule px-6 py-14 text-center" data-testid="shop-product-missing">
          <p className="m-0 text-[22px] font-bold tracking-tight">
            {hidden ? 'This product is not on the shelf right now.' : 'The product could not be loaded.'}
          </p>
          <p className="mx-auto mt-2 max-w-[52ch] text-[15px] text-slate">
            {hidden
              ? 'Its photos are published, but the current shop rules hide every one of them.'
              : 'Check that the demo server is running and try again.'}
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-3">
            <Link to="/shop" className="border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">Back to the storefront</Link>
            <Link to="/catalog" className="border border-rule px-4 py-2 text-[14px] font-bold no-underline hover:border-graphite">Change the rules</Link>
          </div>
        </div>
      </AppShell>
    )
  }

  if (detail.isLoading || !product || !selected) {
    return <AppShell><p className="mt-8 text-[15px] text-slate">Loading the product…</p></AppShell>
  }

  return (
    <AppShell>
      <Breadcrumbs items={[
        { label: 'Storefront', to: '/shop' },
        ...(product.category ? [{ label: product.category, to: `/shop?category=${encodeURIComponent(product.category)}` }] : []),
        { label: product.title },
      ]} />

      <div className="mt-5 grid items-start gap-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        {/* ---- galleria: una foto per prospettiva ---- */}
        <div>
          <figure className="m-0">
            <div className="relative flex aspect-square items-center justify-center bg-canvas p-8" data-testid="shop-main-image">
              <img src={selected.file_url} alt={`${product.title}, ${perspectiveLabel(selected.values.perspective)}`}
                className="max-h-full max-w-full object-contain" />
              <span className="absolute left-0 top-0 bg-white px-2.5 py-1 text-[12px] font-bold tabular-nums" data-testid="shop-counter">
                {images.indexOf(selected) + 1} / {images.length}
              </span>
              {images.length > 1 && (
                <>
                  <button type="button" onClick={() => go(-1)} aria-label="Previous view" title="Previous view (left arrow key)" data-testid="shop-prev"
                    className="absolute left-2 top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center border border-rule bg-white p-0 text-graphite hover:border-graphite">
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square" aria-hidden="true"><path d="M15 4l-8 8 8 8" /></svg>
                  </button>
                  <button type="button" onClick={() => go(1)} aria-label="Next view" title="Next view (right arrow key)" data-testid="shop-next"
                    className="absolute right-2 top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center border border-rule bg-white p-0 text-graphite hover:border-graphite">
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square" aria-hidden="true"><path d="M9 4l8 8-8 8" /></svg>
                  </button>
                </>
              )}
            </div>
            <figcaption className="mt-2 flex flex-wrap items-baseline justify-between gap-2 border-b border-rule pb-2">
              <span className="text-[15px] font-bold">{perspectiveLabel(selected.values.perspective)}</span>
              <span className="text-[13px] text-slate">
                {selected.values.subject ? selected.values.subject.replace(/_/g, ' ') : 'subject not read'}
              </span>
            </figcaption>
          </figure>

          <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-5" data-testid="shop-views">
            {images.map((image, index) => {
              const active = image.image_id === selected.image_id
              return (
                <button key={image.image_id} type="button" onClick={() => setSelectedId(image.image_id)}
                  aria-pressed={active} aria-label={`View ${index + 1}: ${perspectiveLabel(image.values.perspective)}`}
                  data-testid="shop-view"
                  className={`group border-2 bg-white p-0 text-left ${active ? 'border-signal' : 'border-rule hover:border-graphite'}`}>
                  <span className="flex aspect-square items-center justify-center bg-canvas p-1.5">
                    <img src={image.thumbnail_url} alt="" className="max-h-full max-w-full object-contain" />
                  </span>
                  <span className="block px-1.5 py-1 text-[11px] font-bold leading-tight">
                    {perspectiveLabel(image.values.perspective)}
                  </span>
                </button>
              )
            })}
          </div>
          <p className="m-0 mt-2 text-[12px] text-slate">
            Each photo is a different perspective of the same product. The model named the view. {VIEW_ORDER_RULE}
          </p>
        </div>

        {/* ---- scheda ---- */}
        <div>
          <Eyebrow>{product.brand ?? 'No brand'}</Eyebrow>
          <h1 className="mt-1.5 text-[36px] leading-[1.05]">{product.title}</h1>
          <p className="m-0 mt-2 text-[15px] text-slate">
            {[product.category, product.subcategory, product.product_type].filter(Boolean).join(' · ') || 'Category not given'}
          </p>

          <div className="mt-5 grid grid-cols-2 gap-4 border-y border-rule py-3">
            <div>
              <div className="text-[12px] uppercase tracking-[0.1em] text-slate">Photos</div>
              <div className="text-[24px] font-bold tabular-nums">{images.length}</div>
            </div>
            <div>
              <div className="text-[12px] uppercase tracking-[0.1em] text-slate">Perspectives</div>
              <div className="text-[24px] font-bold tabular-nums">{product.perspectives.length}</div>
            </div>
          </div>

          <div className="mt-5">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="m-0 text-[17px] font-bold tracking-tight">What the model read on this view</h2>
              <span className="text-[12px] text-slate">six answers</span>
            </div>
            <ValueTable values={selected.values} confidence={null} />
            <p className="m-0 mt-2 text-[12px] leading-snug text-slate">
              These are the values the catalog rules use. Change a rule and this page can gain or lose a photo without anyone re-labelling it.
            </p>
          </div>


          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/" className="border border-graphite bg-white px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">
              Classify another photo
            </Link>
            <Link to={`/catalog/${product.product_id}`} className="border border-rule px-4 py-2 text-[14px] font-bold no-underline hover:border-graphite">
              Admin view of this product
            </Link>
          </div>
        </div>
      </div>

      {/* ---- tutte le viste, in tabella ---- */}
      <section className="mt-16">
        <div className="flex items-baseline justify-between border-b border-graphite pb-2">
          <h2 className="m-0 text-[20px] font-bold tracking-tight">Every view in this listing</h2>
          <span className="text-[13px] text-slate">{images.length} {images.length === 1 ? 'photo' : 'photos'}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left" data-testid="shop-views-table">
            <thead>
              <tr className="border-b border-rule text-[12px] uppercase tracking-[0.08em] text-slate">
                <th className="w-16 py-2 pr-3 font-normal">Photo</th>
                <th className="py-2 pr-4 font-normal">Perspective</th>
                <th className="py-2 pr-4 font-normal">Subject</th>
                <th className="py-2 pr-4 font-normal">Background</th>
                <th className="py-2 pr-4 font-normal">People</th>
              </tr>
            </thead>
            <tbody>
              {images.map((image) => (
                <tr key={image.image_id} className="border-b border-rule align-middle">
                  <td className="py-2 pr-3">
                    <button type="button" onClick={() => setSelectedId(image.image_id)} aria-label="Show this view"
                      className="block h-10 w-10 border border-rule bg-canvas p-0.5 hover:border-graphite">
                      <img src={image.thumbnail_url} alt="" className="h-full w-full object-contain" />
                    </button>
                  </td>
                  <td className="py-2 pr-4 text-[14px] font-bold">{perspectiveLabel(image.values.perspective)}</td>
                  <td className="py-2 pr-4 text-[14px]">{image.values.subject?.replace(/_/g, ' ') ?? '—'}</td>
                  <td className="py-2 pr-4 text-[14px]">{image.values.background?.replace(/_/g, ' ') ?? '—'}</td>
                  <td className="py-2 pr-4 text-[14px]">{image.values.individuals?.replace(/_/g, ' ') ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  )
}
