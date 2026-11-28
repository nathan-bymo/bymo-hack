import { createContext, useContext, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation, useSearchParams } from 'react-router-dom'
import { useHealth } from '../api/hooks'
import { S } from '../strings'
import { AppShell } from './app'

/* Pagine "In the ERP": stesso stile dell'app di presentazione (bianco, grafite, accento rosso).
   Le firme dei componenti restano quelle di prima, così le pagine non cambiano. */

/* ---- ricerca: filtra per nome file le immagini della sessione ---- */
const SearchContext = createContext<{ query: string; setQuery: (q: string) => void }>({ query: '', setQuery: () => {} })
export const useGlobalSearch = () => useContext(SearchContext)

export function DemoBadge() {
  return <span data-testid="demo-badge" className="text-[13px] text-slate">{S.badgeDemo}</span>
}

function useSessionQuery() {
  const [params] = useSearchParams()
  const session = params.get('session') ?? localStorage.getItem('demo.session')
  return session ? `?session=${session}` : ''
}

function ErpNav() {
  const { pathname } = useLocation()
  const q = useSessionQuery()
  const { query, setQuery } = useGlobalSearch()
  const tabs = [
    { to: `/image-classification${q}`, label: 'Review queue', active: pathname === '/image-classification' || /^\/image-classification\/img/.test(pathname) },
    { to: `/image-classification/compare${q}`, label: 'Compare models', active: pathname.startsWith('/image-classification/compare') },
    { to: '/image-classification/performance?set=coverage', label: 'Scores per class', active: pathname.startsWith('/image-classification/performance') },
    { to: `/storefront-preview${q}`, label: 'Shop preview', active: pathname.startsWith('/storefront-preview') },
    { to: '/product/DEMO-0001', label: 'Product page', active: pathname.startsWith('/product') },
  ]
  return (
    <div className="mb-8">
      <p className="lead">How Galaxus staff would use the classifier inside their ERP: add the photos of a product, let a model fill in the six fields, then check and save.</p>
      <div className="mt-6 flex flex-wrap items-end gap-x-7 gap-y-3 border-b border-rule">
        <nav aria-label="ERP" className="flex flex-wrap gap-x-7 text-[15px]">
          {tabs.map((t) => (
            <NavLink key={t.label} to={t.to}
              className={`-mb-px border-b-2 pb-2 no-underline ${t.active ? 'border-graphite font-bold text-graphite' : 'border-transparent text-slate hover:text-graphite'}`}>
              {t.label}
            </NavLink>
          ))}
        </nav>
        <input aria-label="Search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter photos by name"
          className="mb-2 ml-auto w-56 border border-rule bg-white px-3 py-1.5 text-[14px] placeholder:text-slate" />
      </div>
    </div>
  )
}

/* ---- titolo, link secondari, toolbar, sezioni ---- */
export function TitleBar({ title, id, prefix, children }: { title: string; id?: string; prefix?: string; children?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-baseline gap-x-4 gap-y-2">
      <h1 style={{ fontSize: 32 }} className="min-w-0 break-words">{prefix ? <span className="text-slate">{prefix} / </span> : null}{title}</h1>
      {id && <span className="text-[14px] text-slate">Id {id}</span>}
      <div className="ml-auto flex items-center gap-3 text-[14px]">{children}</div>
    </div>
  )
}

/** Le voci principali sono già nelle schede: qui restano solo i link "indietro" (es. "← Queue"). */
export function SubLinks({ links }: { links: { to: string; label: string }[] }) {
  const back = links.filter((l) => l.label.startsWith('←'))
  if (!back.length) return null
  return (
    <div className="mb-4 flex gap-4 text-[14px]">
      {back.map((l) => <Link key={l.to} to={l.to}>{l.label}</Link>)}
    </div>
  )
}

export function ToolbarButton({ children, onClick, disabled, title, testId }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; title?: string; testId?: string
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title} data-testid={testId}
      className="border border-graphite bg-white px-3.5 py-1.5 text-[14px] font-bold text-graphite hover:bg-canvas disabled:cursor-not-allowed disabled:opacity-40">
      {children}
    </button>
  )
}

export function ToolbarDropdown({ label, items }: {
  label: string
  items: { label: string; to?: string; disabled?: boolean; testId?: string }[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <ToolbarButton onClick={() => setOpen((o) => !o)} testId={`menu-${label.toLowerCase().replace(/\s+/g, '-')}`}>{label} ▾</ToolbarButton>
      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 min-w-56 border border-graphite bg-white py-1">
          {items.map((it) =>
            it.to && !it.disabled ? (
              <Link key={it.label} to={it.to} onClick={() => setOpen(false)} data-testid={it.testId}
                className="block px-3 py-1.5 text-[14px] no-underline hover:bg-canvas">{it.label}</Link>
            ) : (
              <div key={it.label} title={S.inactive} className="cursor-not-allowed px-3 py-1.5 text-[14px] text-slate">{it.label}</div>
            ),
          )}
        </div>
      )}
    </div>
  )
}

export function ActionToolbar({ children }: { children: ReactNode }) {
  return <div className="mb-6 flex flex-wrap items-center gap-3">{children}</div>
}

export function CollapsibleSection({ title, defaultOpen = true, inactive, children }: {
  title: string; defaultOpen?: boolean; inactive?: boolean; children?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen && !inactive)
  if (inactive) return null   // le sezioni segnaposto dell'ERP finto non servono alla presentazione
  return (
    <section className="border-t border-rule py-4">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center justify-between border-0 bg-transparent p-0 text-left text-[17px] font-bold text-graphite">
        {title}<span aria-hidden="true" className="text-[15px] font-normal text-slate">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && <div className="mt-3">{children}</div>}
    </section>
  )
}

export function LabelValueGrid({ rows }: { rows: { label: string; value: ReactNode; info?: string }[] }) {
  return (
    <dl className="m-0 grid grid-cols-[minmax(9rem,13rem)_1fr] gap-x-4 gap-y-1.5">
      {rows.map((r) => (
        <div key={r.label} className="contents">
          <dt className="text-[14px] text-slate" title={r.info}>{r.label}</dt>
          <dd className="m-0 text-[15px]">{r.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function WidgetCard({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-3 border-t border-rule pt-2">
      <div className="text-[14px] font-bold">{title}</div>
      {children && <div className="mt-1 text-[13px] text-slate">{children}</div>}
    </div>
  )
}

/* ---- guscio ---- */
export function ErpShell({ children }: { children: ReactNode; rightColumn?: ReactNode }) {
  const health = useHealth()
  const [query, setQuery] = useState('')
  return (
    <SearchContext.Provider value={{ query, setQuery }}>
      <AppShell>
        {health.isError && (
          <div role="alert" data-testid="backend-down" className="mb-6 bg-signal-soft px-4 py-3 font-bold text-signal">
            {S.backendDown}
          </div>
        )}
        <ErpNav />
        {children}
      </AppShell>
    </SearchContext.Provider>
  )
}
