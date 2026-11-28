import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { CatalogProduct, StorefrontFilters, StorefrontProductCard } from '../api/types'
import { confidenceTone, FILTER_LABELS, percent, perspectiveLabel, STATUS_LABEL } from '../lib/catalog'
import { ConfidenceChip } from './ui'

/* ---- piccoli elementi condivisi fra negozio e catalogo ---- */

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="m-0 text-[12px] font-bold uppercase tracking-[0.14em] text-signal">{children}</p>
}

export function Breadcrumbs({ items }: { items: { label: string; to?: string }[] }) {
  return (
    <nav aria-label="Breadcrumb" className="flex flex-wrap items-center gap-2 text-[13px] text-slate">
      {items.map((item, i) => (
        <span key={item.label} className="flex items-center gap-2">
          {i > 0 && <span aria-hidden="true">/</span>}
          {item.to ? <Link to={item.to} className="no-underline hover:underline">{item.label}</Link> : <span>{item.label}</span>}
        </span>
      ))}
    </nav>
  )
}

export function SearchForm({
  value, onChange, onSubmit, placeholder, testId, button = 'Search',
}: {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  placeholder: string
  testId: string
  button?: string
}) {
  const submit = (e: FormEvent) => { e.preventDefault(); onSubmit() }
  return (
    <form onSubmit={submit} role="search" className="flex border border-graphite bg-white">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
        data-testid={testId}
        className="min-w-0 flex-1 border-0 bg-transparent px-4 py-3 text-[17px] text-graphite outline-none placeholder:text-slate"
      />
      <button type="submit" data-testid={`${testId}-submit`}
        className="border-0 bg-signal px-6 text-[15px] font-bold text-white hover:bg-graphite">
        {button}
      </button>
    </form>
  )
}

export function ConfidenceMeter({ value, compact }: { value: number | null; compact?: boolean }) {
  const tone = confidenceTone(value)
  const width = value === null ? 0 : Math.max(2, Math.round(value * 100))
  const bar = tone === 'low' ? 'bg-signal' : tone === 'none' ? 'bg-bar-alt' : 'bg-graphite'
  return (
    <div className={`flex items-center gap-2 ${compact ? 'text-[12px]' : 'text-[13px]'}`}>
      <div className={`${compact ? 'h-1' : 'h-1.5'} min-w-0 flex-1 bg-canvas`} aria-hidden="true">
        <div className={`h-full ${bar}`} style={{ width: `${width}%` }} />
      </div>
      <span className={`shrink-0 tabular-nums ${tone === 'low' ? 'font-bold text-signal' : 'text-slate'}`}>
        {value === null ? 'not read' : percent(value)}
      </span>
    </div>
  )
}

export function PerspectiveLine({ perspectives }: { perspectives: string[] }) {
  if (!perspectives.length) return <span className="text-slate">No view read yet</span>
  return <span>{perspectives.map((p) => perspectiveLabel(p)).join(' · ')}</span>
}

export function StatusTag({ status }: { status: CatalogProduct['status'] }) {
  const cls = status === 'live' ? 'bg-conf-high-bg text-conf-high'
    : status === 'draft' ? 'bg-conf-mid-bg text-conf-mid' : 'bg-toolbtn text-muted'
  return (
    <span data-testid={`status-${status}`} className={`inline-block whitespace-nowrap px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide ${cls}`}>
      {STATUS_LABEL[status]}
    </span>
  )
}

/** Scheda prodotto della vetrina: foto grande, marca, titolo, quante viste e quanto è sicuro il modello. */
export function ProductCard({ product, testId = 'shop-card' }: { product: StorefrontProductCard; testId?: string }) {
  return (
    <Link to={`/shop/${product.product_id}`} data-testid={testId} data-product={product.product_id}
      className="group flex flex-col border border-rule bg-white no-underline transition-colors hover:border-graphite">
      <div className="relative aspect-[4/3] overflow-hidden bg-canvas p-4">
        <img src={product.cover.file_url} alt={product.title}
          className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-[1.03]" />
        <span className="absolute left-0 top-0 bg-white/90 px-2 py-1 text-[11px] font-bold tabular-nums text-graphite">
          {product.image_count} {product.image_count === 1 ? 'view' : 'views'}
        </span>
      </div>
      <div className="flex flex-1 flex-col border-t border-rule px-3 py-2.5">
        <span className="text-[11px] font-bold uppercase tracking-[0.12em] text-slate">{product.brand ?? 'No brand'}</span>
        <span className="mt-0.5 text-[16px] font-bold leading-snug tracking-tight text-graphite group-hover:underline">{product.title}</span>
        <span className="mt-1 text-[13px] leading-snug text-slate"><PerspectiveLine perspectives={product.perspectives} /></span>
        <div className="mt-auto pt-2"><ConfidenceMeter value={product.min_confidence} compact /></div>
      </div>
    </Link>
  )
}

/** Riquadro numerico: usato nelle testate di negozio e catalogo. */
export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="border-t-2 border-graphite pt-2">
      <div className="text-[26px] font-bold leading-none tracking-tight tabular-nums">{value}</div>
      <div className="mt-1 text-[12px] uppercase tracking-[0.1em] text-slate">{label}</div>
      {hint && <div className="mt-0.5 text-[12px] text-slate">{hint}</div>}
    </div>
  )
}

/** I sei valori di una foto, con la confidenza. Serve sia all'admin sia alla scheda prodotto. */
export function ValueTable({
  values, confidence, compact,
}: {
  values: Record<string, string | null>
  confidence: number | null
  compact?: boolean
}) {
  const rows: [string, string][] = [
    ['subject', 'Subject'], ['background', 'Background'], ['perspective', 'Perspective'],
    ['individuals', 'People'], ['text_classification', 'Added text'], ['text_language', 'Language'],
  ]
  return (
    <div className={compact ? '' : 'mt-1'}>
      <table className="w-full border-collapse text-left">
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key} className="border-b border-rule align-baseline">
              <th className="w-[42%] py-1.5 pr-3 text-[13px] font-normal text-slate">{label}</th>
              <td className="py-1.5 text-[14px] font-bold leading-snug text-graphite">
                {values[key] ? values[key]!.replace(/_/g, ' ') : <span className="font-normal text-slate">not read yet</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex items-center justify-between gap-3">
        <span className="text-[12px] text-slate">Lowest confidence</span>
        <ConfidenceChip value={confidence} />
      </div>
    </div>
  )
}

/** Pannello dei filtri: sono le regole che decidono cosa vede il cliente. */
export function FilterPanel({
  filters, hidden, visible, saving, onChange, onOpenShop,
}: {
  filters: StorefrontFilters
  hidden: number
  visible: number
  saving: boolean
  onChange: (filters: StorefrontFilters) => void
  onOpenShop: () => void
}) {
  const [minConf, setMinConf] = useState(filters.min_confidence)
  useEffect(() => setMinConf(filters.min_confidence), [filters.min_confidence])
  const commitMin = () => {
    if (Math.abs(minConf - filters.min_confidence) > 0.001) onChange({ ...filters, min_confidence: minConf })
  }
  const toggles = (['hide_non_photos', 'only_white_background', 'hide_people', 'hide_added_text'] as const)
  return (
    <aside className="border border-rule bg-white" data-testid="filter-panel">
      <div className="flex items-baseline justify-between border-b-2 border-graphite px-4 py-3">
        <h2 className="m-0 text-[18px] font-bold tracking-tight">Shop rules</h2>
        <span className="text-[12px] text-slate">{saving ? 'Saving…' : 'Live'}</span>
      </div>
      <div className="px-4 py-3">
        <p className="m-0 text-[13px] leading-snug text-slate">
          Every switch changes the customer storefront at once. {visible} photos are visible, {hidden} hidden by these rules.
        </p>
        <div className="mt-3 divide-y divide-rule border-t border-rule">
          {toggles.map((key) => (
            <label key={key} className="flex cursor-pointer items-start gap-3 py-2.5" title={FILTER_LABELS[key].hint}>
              <input type="checkbox" checked={filters[key]} data-testid={`filter-${key}`}
                onChange={(e) => onChange({ ...filters, [key]: e.target.checked })}
                className="mt-0.5 h-4 w-4 shrink-0 accent-signal" />
              <span className="min-w-0">
                <span className="block text-[14px] font-bold leading-snug">{FILTER_LABELS[key].title}</span>
                <span className="block text-[12px] leading-snug text-slate">{FILTER_LABELS[key].hint}</span>
              </span>
            </label>
          ))}
          <div className="py-3">
            <div className="flex items-baseline justify-between gap-3">
              <label htmlFor="min-confidence" className="text-[14px] font-bold">{FILTER_LABELS.min_confidence.title}</label>
              <span className="text-[14px] font-bold tabular-nums text-signal">{Math.round(minConf * 100)}%</span>
            </div>
            <input id="min-confidence" type="range" min={0} max={0.95} step={0.05} value={minConf}
              data-testid="filter-min-confidence"
              onChange={(e) => setMinConf(Number(e.target.value))}
              onPointerUp={commitMin} onKeyUp={commitMin} onBlur={commitMin}
              className="mt-2 w-full accent-signal" />
            <p className="m-0 mt-1 text-[12px] leading-snug text-slate">{FILTER_LABELS.min_confidence.hint}</p>
          </div>
        </div>
        <button type="button" onClick={onOpenShop} data-testid="open-shop"
          className="mt-2 w-full border border-graphite bg-white px-3 py-2 text-[14px] font-bold hover:bg-canvas">
          Open the storefront →
        </button>
      </div>
    </aside>
  )
}
