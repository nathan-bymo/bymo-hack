import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useStorefront } from '../api/catalog'
import { AppShell } from '../components/app'
import { Eyebrow, ProductCard, SearchForm, Stat } from '../components/catalog-ui'
import { ErrorState } from '../components/ui'

export default function ShopPage() {
  const [params, setParams] = useSearchParams()
  const q = params.get('q') ?? ''
  const category = params.get('category')
  const [draft, setDraft] = useState(q)
  useEffect(() => setDraft(q), [q])
  const shop = useStorefront(q, category)

  const apply = (next: { q?: string; category?: string | null }) => {
    setParams((p) => {
      const query = next.q ?? q
      const cat = next.category === undefined ? category : next.category
      if (query.trim()) p.set('q', query.trim())
      else p.delete('q')
      if (cat) p.set('category', cat)
      else p.delete('category')
      return p
    })
  }

  const stats = shop.data?.stats
  const products = shop.data?.products ?? []
  const categories = shop.data?.categories ?? []
  const noResults = shop.isSuccess && products.length === 0

  return (
    <AppShell>
      <div className="flex flex-wrap items-end justify-between gap-6 border-b border-graphite pb-4">
        <div className="max-w-[720px]">
          <Eyebrow>Client view · Galaxus demo store</Eyebrow>
          <h1 className="mt-2">Find the product. See every side.</h1>
          <p className="lead mt-2 max-w-none">
            A shop built from the catalog: Apertus has read every photo, and the catalog rules decide what goes on the shelf.
          </p>
        </div>
        <Link to="/catalog" data-testid="shop-admin-link"
          className="border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">
          Admin: catalog and rules →
        </Link>
      </div>

      <div className="mt-6">
        <SearchForm value={draft} onChange={setDraft} onSubmit={() => apply({ q: draft })}
          placeholder="Search a product, a brand or a category" testId="shop-search" />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2" data-testid="shop-categories">
        <span className="mr-1 text-[12px] uppercase tracking-[0.1em] text-slate">Category</span>
        <button type="button" onClick={() => apply({ category: null })} aria-pressed={!category}
          className={`border px-3 py-1 text-[13px] font-bold ${!category ? 'border-graphite bg-graphite text-white' : 'border-rule bg-white hover:border-graphite'}`}>
          All
        </button>
        {categories.map((c) => (
          <button key={c.name} type="button" onClick={() => apply({ category: c.name })} aria-pressed={category === c.name}
            data-testid="shop-category"
            className={`border px-3 py-1 text-[13px] ${category === c.name ? 'border-graphite bg-graphite font-bold text-white' : 'border-rule bg-white hover:border-graphite'}`}>
            {c.name} <span className="tabular-nums opacity-60">{c.count}</span>
          </button>
        ))}
      </div>

      <div className="mt-7 grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Products" value={shop.data?.total ?? '—'} hint="on the shelf now" />
        <Stat label="Photos" value={stats?.visible_images ?? '—'} hint="published and passing the rules" />
        <Stat label="Views" value={products.reduce((n, p) => n + p.image_count, 0)} hint="one photo per perspective" />
        <Stat label="Hidden by rules" value={stats?.hidden_by_rules ?? '—'} hint="admin choice, not a model error" />
      </div>

      <div className="mt-9 flex items-baseline justify-between border-b border-graphite pb-2">
        <h2 className="m-0 text-[18px] font-bold tracking-tight">
          {q ? <>Results for “{q}”</> : category ? category : 'All products'}
        </h2>
        <span className="text-[13px] text-slate" data-testid="shop-count">
          {shop.isLoading ? 'Loading…' : `${products.length} ${products.length === 1 ? 'product' : 'products'}`}
        </span>
      </div>

      {shop.isError && <div className="mt-4"><ErrorState error={shop.error} /></div>}

      {noResults ? (
        <div className="mt-6 border border-rule px-6 py-14 text-center" data-testid="shop-empty">
          <p className="m-0 text-[22px] font-bold tracking-tight">Nothing matches {q ? `“${q}”` : 'these filters'}.</p>
          <p className="mx-auto mt-2 max-w-[52ch] text-[15px] text-slate">
            Try a brand, a category or a simpler word. {stats?.hidden_by_rules ?? 0} published photos are hidden by the current shop rules.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-3">
            <button type="button" onClick={() => apply({ q: '', category: null })}
              className="border border-graphite bg-white px-4 py-2 text-[14px] font-bold hover:bg-canvas">Show every product</button>
            <Link to="/catalog" className="border border-rule px-4 py-2 text-[14px] font-bold no-underline hover:border-graphite">Change the rules</Link>
          </div>
        </div>
      ) : (
        <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" data-testid="shop-grid">
          {products.map((product) => <ProductCard key={product.product_id} product={product} />)}
        </div>
      )}

      <section className="mt-16 grid gap-6 border-t-2 border-graphite pt-5 md:grid-cols-3">
        <div>
          <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">1 · The photo</span>
          <h3 className="m-0 mt-1 text-[18px] font-bold tracking-tight">Every product starts with a picture</h3>
          <p className="m-0 mt-1 text-[14px] leading-relaxed text-slate">
            A supplier sends a photo. Nobody types six fields by hand.
          </p>
          <Link to="/" className="mt-2 inline-block text-[14px] font-bold">Try it on a photo →</Link>
        </div>
        <div>
          <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">2 · The model</span>
          <h3 className="m-0 mt-1 text-[18px] font-bold tracking-tight">Apertus answers six questions</h3>
          <p className="m-0 mt-1 text-[14px] leading-relaxed text-slate">
            Subject, background, perspective, people, added text, language: one class and one probability each.
          </p>
          <Link to="/" className="mt-2 inline-block text-[14px] font-bold">See the six answers →</Link>
        </div>
        <div>
          <span className="text-[12px] font-bold uppercase tracking-[0.1em] text-signal">3 · The rules</span>
          <h3 className="m-0 mt-1 text-[18px] font-bold tracking-tight">The catalog decides what is shown</h3>
          <p className="m-0 mt-1 text-[14px] leading-relaxed text-slate">
            The admin publishes photos and sets the shop rules. The same photo can be on the shelf or hidden, without retraining anything.
          </p>
          <Link to="/catalog" className="mt-2 inline-block text-[14px] font-bold">Open the catalog →</Link>
        </div>
      </section>
    </AppShell>
  )
}
