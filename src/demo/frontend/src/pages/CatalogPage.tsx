import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useCatalog, useSaveFilters } from '../api/catalog'
import type { StorefrontFilters } from '../api/types'
import { AppShell } from '../components/app'
import { Eyebrow, FilterPanel, SearchForm, Stat, StatusTag } from '../components/catalog-ui'
import { ConfidenceMeter } from '../components/catalog-ui'
import { ErrorState } from '../components/ui'
import { percent } from '../lib/catalog'

const STATUSES = [
  { key: null, label: 'All' },
  { key: 'live', label: 'Live' },
  { key: 'draft', label: 'Drafts' },
  { key: 'empty', label: 'Empty' },
] as const

export default function CatalogPage() {
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const q = params.get('q') ?? ''
  const status = params.get('status')
  const [draft, setDraft] = useState(q)
  useEffect(() => setDraft(q), [q])
  const catalog = useCatalog(q, status)
  const saveFilters = useSaveFilters()

  const apply = (next: { q?: string; status?: string | null }) => {
    setParams((p) => {
      const query = next.q ?? q
      const nextStatus = next.status === undefined ? status : next.status
      if (query.trim()) p.set('q', query.trim())
      else p.delete('q')
      if (nextStatus) p.set('status', nextStatus)
      else p.delete('status')
      return p
    })
  }

  const stats = catalog.data?.stats
  const products = catalog.data?.products ?? []
  const filters = catalog.data?.filters
  const onFilters = (next: StorefrontFilters) => saveFilters.mutate(next)

  return (
    <AppShell>
      <div className="flex flex-wrap items-end justify-between gap-6 border-b border-graphite pb-4">
        <div className="max-w-[760px]">
          <Eyebrow>Admin view · catalog workspace</Eyebrow>
          <h1 className="mt-2">The catalog behind the shop</h1>
          <p className="lead mt-2 max-w-none">
            Products, photos and the rules that decide what customers see. Every value here comes from the model, with a human able to step in.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link to="/" data-testid="catalog-add-photo"
            className="bg-signal px-4 py-2 text-[14px] font-bold text-white no-underline hover:bg-graphite">
            Add a photo
          </Link>
          <Link to="/shop" className="border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">
            Open the storefront →
          </Link>
        </div>
      </div>

      <div className="mt-7 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <Stat label="Products" value={stats?.products ?? '—'} />
        <Stat label="Live" value={stats?.live_products ?? '—'} hint="on the shelf" />
        <Stat label="Drafts" value={stats?.draft_products ?? '—'} hint="not published" />
        <Stat label="Photos" value={stats?.images ?? '—'} />
        <Stat label="Hidden by rules" value={stats?.hidden_by_rules ?? '—'} hint="published but filtered" />
        <Stat label="To check" value={stats?.needs_review ?? '—'} hint="low confidence or no answer" />
      </div>

      <div className="mt-9 grid items-start gap-8 lg:grid-cols-[290px_minmax(0,1fr)]">
        {filters && (
          <div className="lg:sticky lg:top-6">
            <FilterPanel filters={filters} hidden={stats?.hidden_by_rules ?? 0} visible={stats?.visible_images ?? 0}
              saving={saveFilters.isPending} onChange={onFilters} onOpenShop={() => navigate('/shop')} />
          </div>
        )}

        <div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="min-w-[240px] flex-1">
              <SearchForm value={draft} onChange={setDraft} onSubmit={() => apply({ q: draft })}
                placeholder="Search the catalog" testId="catalog-search" button="Find" />
            </div>
            <div className="flex border border-rule bg-white" role="group" aria-label="Filter by status">
              {STATUSES.map((s) => (
                <button key={s.label} type="button" onClick={() => apply({ status: s.key })}
                  aria-pressed={status === s.key} data-testid={`catalog-status-${s.key ?? 'all'}`}
                  className={`border-0 px-3 py-2 text-[13px] font-bold ${status === s.key ? 'bg-graphite text-white' : 'bg-white hover:bg-canvas'}`}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {catalog.isError && <div className="mt-4"><ErrorState error={catalog.error} /></div>}

          <div className="mt-4 flex items-baseline justify-between border-b border-graphite pb-2">
            <span className="text-[13px] text-slate">{catalog.isLoading ? 'Loading…' : `${products.length} ${products.length === 1 ? 'product' : 'products'}`}</span>
            <span className="text-[13px] text-slate">{stats ? `${stats.visible_images} photos visible · ${stats.hidden_by_rules} hidden by rules` : ''}</span>
          </div>

          {catalog.isSuccess && products.length === 0 ? (
            <div className="mt-5 border border-rule px-6 py-12 text-center" data-testid="catalog-empty">
              <p className="m-0 text-[20px] font-bold">No product matches this view.</p>
              <p className="mt-2 text-[14px] text-slate">Clear the search or the status filter.</p>
              <button type="button" onClick={() => apply({ q: '', status: null })}
                className="mt-4 border border-graphite px-4 py-2 text-[14px] font-bold hover:bg-canvas">Show everything</button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left" data-testid="catalog-table">
                <thead>
                  <tr className="border-b border-rule text-[11px] uppercase tracking-[0.08em] text-slate">
                    <th className="w-16 py-2 pr-3 font-normal">Photo</th>
                    <th className="py-2 pr-4 font-normal">Product</th>
                    <th className="py-2 pr-4 font-normal">Category</th>
                    <th className="py-2 pr-4 text-right font-normal">Views</th>
                    <th className="w-36 py-2 pr-4 font-normal">Lowest confidence</th>
                    <th className="py-2 pr-3 font-normal">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {products.map((product) => (
                    <tr key={product.product_id} className="group border-b border-rule align-middle hover:bg-canvas"
                      data-testid="catalog-row" data-product={product.product_id}>
                      <td className="py-2 pr-3">
                        <Link to={`/catalog/${product.product_id}`} className="block h-12 w-12 border border-rule bg-white p-1">
                          {product.cover
                            ? <img src={product.cover.thumbnail_url} alt="" className="h-full w-full object-contain" />
                            : <span className="flex h-full w-full items-center justify-center text-[10px] text-slate">none</span>}
                        </Link>
                      </td>
                      <td className="py-2 pr-4">
                        <Link to={`/catalog/${product.product_id}`} className="text-[15px] font-bold no-underline group-hover:underline">
                          {product.title}
                        </Link>
                        <div className="text-[12px] text-slate">
                          {[product.brand, product.product_type].filter(Boolean).join(' · ') || 'No brand'}
                          {product.needs_review > 0 && (
                            <span className="ml-2 font-bold text-signal">{product.needs_review} to check</span>
                          )}
                        </div>
                      </td>
                      <td className="py-2 pr-4 text-[14px]">{product.category ?? '—'}</td>
                      <td className="py-2 pr-4 text-right text-[14px] tabular-nums">
                        {product.published_count} / {product.image_count}
                      </td>
                      <td className="py-2 pr-4"><ConfidenceMeter value={product.min_confidence} compact /></td>
                      <td className="py-2 pr-3"><StatusTag status={product.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="m-0 mt-3 text-[12px] text-slate">
            “Views” is published photos over all photos in the product. A published photo can still be hidden by the shop rules; the product page shows why.
          </p>
        </div>
      </div>

      <section className="mt-14 border-t-2 border-graphite pt-5">
        <h2 className="m-0 text-[20px] font-bold tracking-tight">From one photo to a product page</h2>
        <div className="mt-4 grid gap-6 md:grid-cols-3">
          <div className="border-t border-rule pt-3">
            <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">Add</span>
            <p className="m-0 mt-1 text-[14px] leading-relaxed">
              Upload a photo on the demo page. The model answers the six questions; the answers become the catalog values.
            </p>
            <Link to="/" className="mt-2 inline-block text-[14px] font-bold">Add a photo →</Link>
          </div>
          <div className="border-t border-rule pt-3">
            <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">Check</span>
            <p className="m-0 mt-1 text-[14px] leading-relaxed">
              Open a product, publish the views you trust and pick the cover. Low confidence and missing answers are flagged.
            </p>
          </div>
          <div className="border-t border-rule pt-3">
            <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">Show</span>
            <p className="m-0 mt-1 text-[14px] leading-relaxed">
              The storefront applies the rules on every request. No redeploy, no retraining: the same catalog, a different shelf.
            </p>
            <Link to="/shop" className="mt-2 inline-block text-[14px] font-bold">Open the storefront →</Link>
          </div>
        </div>
        <p className="m-0 mt-4 text-[12px] text-slate">
          Current rule: minimum confidence {filters ? percent(filters.min_confidence) : '—'}.
        </p>
      </section>
    </AppShell>
  )
}
