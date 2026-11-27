import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useHealth } from '../api/hooks'
import { S } from '../strings'

/* ---- ricerca globale: filtra per nome file le immagini della sessione (unica funzione attiva) ---- */
const SearchContext = createContext<{ query: string; setQuery: (q: string) => void }>({ query: '', setQuery: () => {} })
export const useGlobalSearch = () => useContext(SearchContext)

export function DemoBadge() {
  return (
    <span
      data-testid="demo-badge"
      className="inline-flex items-center rounded-full bg-titlebar px-3 py-1 text-xs font-semibold text-titlebar-fg"
    >
      {S.badgeDemo}
    </span>
  )
}

export function Wordmark({ variant = 'erp' }: { variant?: 'erp' | 'store' }) {
  // Testo + icona SVG generica: nessun logo né asset Galaxus.
  return (
    <Link to="/product/DEMO-0001" className="flex items-center gap-2 no-underline" style={{ color: variant === 'erp' ? 'var(--c-text)' : 'var(--c-text)' }}>
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="5" fill="#e5533d" />
        {Array.from({ length: 8 }, (_, i) => (
          <line key={i} x1="12" y1="2" x2="12" y2="5" stroke="#e5533d" strokeWidth="2" strokeLinecap="round"
            transform={`rotate(${i * 45} 12 12)`} />
        ))}
      </svg>
      <span className="text-lg font-bold tracking-wide">{variant === 'erp' ? 'ERP' : 'GALAXUS'}</span>
    </Link>
  )
}

function GlobalSearch() {
  const { query, setQuery } = useGlobalSearch()
  return (
    <input
      aria-label="Search"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Search for products, orders, customers, and more"
      className="w-full max-w-3xl border border-line bg-app px-4 text-sm text-ink placeholder:text-muted"
      style={{ height: 36, borderRadius: 'var(--radius-search)' }}
    />
  )
}

function DensityToggle() {
  const [presentation, setPresentation] = useState(() => document.documentElement.dataset.density === 'presentation')
  useEffect(() => {
    document.documentElement.dataset.density = presentation ? 'presentation' : 'normal'
  }, [presentation])
  return (
    <button
      type="button"
      onClick={() => setPresentation((p) => !p)}
      aria-pressed={presentation}
      className="rounded border border-line bg-toolbtn px-2 py-1 text-xs text-ink"
      title="Switch between laptop and presentation (TV) density"
    >
      {presentation ? 'Presentation' : 'Normal'}
    </button>
  )
}

export function ErpHeader() {
  return (
    <header className="flex items-center gap-4 bg-panel px-4" style={{ height: 56 }}>
      <Wordmark />
      <div className="flex-1"><GlobalSearch /></div>
      <DensityToggle />
      <DemoBadge />
    </header>
  )
}

export const AccentLine = () => <div className="bg-accent" style={{ height: 3 }} aria-hidden="true" />

/* ---- barra laterale ---- */
function Inactive({ children, indent = 0 }: { children: ReactNode; indent?: number }) {
  return (
    <div className="cursor-not-allowed py-1 text-muted" style={{ paddingLeft: 12 + indent * 12 }} title={S.inactive}>
      {children}
    </div>
  )
}

export function SwitchNowButton() {
  return (
    <button type="button" disabled title={S.inactive}
      className="mx-3 mt-6 cursor-not-allowed bg-switch px-3 py-2 text-left text-sm font-semibold text-white opacity-90" style={{ borderRadius: 'var(--radius-switch)' }}>
      New design<br /><span className="font-normal">Switch now →</span>
    </button>
  )
}

export function Sidebar() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState({ products: true, data: true })
  const active = pathname.startsWith('/image-classification')
  return (
    <nav aria-label="Main" className="w-56 shrink-0 bg-app py-2" style={{ fontSize: 'var(--fs-value)' }}>
      <Inactive>▸ Order management</Inactive>
      <button type="button" className="block w-full py-1 pl-3 text-left" onClick={() => setOpen((o) => ({ ...o, products: !o.products }))}>
        {open.products ? '▾' : '▸'} Products
      </button>
      {open.products && (
        <>
          <button type="button" className="block w-full py-1 pl-6 text-left" onClick={() => setOpen((o) => ({ ...o, data: !o.data }))}>
            {open.data ? '▾' : '▸'} Product data
          </button>
          {open.data && (
            <NavLink to="/image-classification"
              className={`block py-1 pl-10 no-underline ${active ? 'bg-toolbtn font-semibold' : ''}`}>
              · Image classification (AI)
            </NavLink>
          )}
        </>
      )}
      <Inactive>▸ Human Resources</Inactive>
      <Inactive>★ Favourites</Inactive>
      <SwitchNowButton />
    </nav>
  )
}

/* ---- titolo, sotto-link, toolbar, sezioni ---- */
export function TitleBar({ title, id, prefix, children }: { title: string; id?: string; prefix?: string; children?: ReactNode }) {
  return (
    <div className="flex items-center gap-3 bg-titlebar px-4 text-titlebar-fg" style={{ fontSize: 'var(--fs-title)', minHeight: 32 }}>
      {prefix && <span className="font-bold underline" style={{ fontSize: 'var(--fs-title-main)' }}>{prefix}</span>}
      <h1 className="m-0 font-bold" style={{ fontSize: 'inherit' }}>{title}</h1>
      {id && <span className="text-sm opacity-90">(Id {id})</span>}
      <span aria-hidden="true" title={S.inactive}>☆</span>
      <div className="ml-auto flex items-center gap-2">{children}</div>
    </div>
  )
}

export function SubLinks({ links }: { links: { to: string; label: string }[] }) {
  return (
    <div className="flex gap-3 border-b border-line bg-panel px-4 py-1 text-sm">
      {links.map((l) => <NavLink key={l.to} to={l.to} end className={({ isActive }) => (isActive ? 'font-semibold' : '')}>{l.label}</NavLink>)}
    </div>
  )
}

export function ToolbarButton({ children, onClick, disabled, title, testId }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; title?: string; testId?: string
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title} data-testid={testId}
      className="border bg-toolbtn px-3 text-ink disabled:cursor-not-allowed disabled:opacity-50"
      style={{ borderRadius: 'var(--radius-btn)', borderColor: 'var(--c-toolbar-btn-border)', height: 24, fontSize: 'var(--fs-value)' }}>
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
        <div className="absolute left-0 top-full z-20 mt-1 min-w-52 rounded border border-line bg-panel py-1 shadow">
          {items.map((it) =>
            it.to && !it.disabled ? (
              <Link key={it.label} to={it.to} onClick={() => setOpen(false)} data-testid={it.testId}
                className="block px-3 py-1 text-sm no-underline hover:bg-toolbtn">{it.label}</Link>
            ) : (
              <div key={it.label} title={S.inactive} className="cursor-not-allowed px-3 py-1 text-sm text-muted">{it.label}</div>
            ),
          )}
        </div>
      )}
    </div>
  )
}

export function ActionToolbar({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center justify-end gap-2 border-b border-line bg-panel px-4 py-2">{children}</div>
}

export function CollapsibleSection({ title, defaultOpen = true, inactive, children }: {
  title: string; defaultOpen?: boolean; inactive?: boolean; children?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen && !inactive)
  return (
    <section className={`px-4 py-3 ${inactive ? 'text-muted' : ''}`} style={{ borderTop: '2px solid var(--c-section-rule)' }} title={inactive ? S.inactive : undefined}>
      <button type="button" disabled={inactive} onClick={() => setOpen((o) => !o)}
        className={`section-title border-0 bg-transparent p-0 ${inactive ? 'cursor-not-allowed text-muted' : ''}`}>
        {open ? '▼' : '▶'} {title}
      </button>
      {open && !inactive && <div className="mt-2">{children}</div>}
    </section>
  )
}

export function LabelValueGrid({ rows }: { rows: { label: string; value: ReactNode; info?: string }[] }) {
  return (
    <dl className="m-0 grid grid-cols-[minmax(10rem,14rem)_1fr] gap-x-4 gap-y-1">
      {rows.map((r) => (
        <div key={r.label} className="contents">
          <dt className="text-muted" style={{ fontSize: 'var(--fs-label)' }}>{r.label}</dt>
          <dd className="m-0" style={{ fontSize: 'var(--fs-value)' }}>
            {r.value}{r.info && <span title={r.info} className="ml-1 cursor-help text-muted">ⓘ</span>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function WidgetCard({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-2 border border-line bg-panel p-2">
      <div className="section-title">▾ {title}</div>
      <div className="mt-1 text-xs text-muted">{children ?? 'Placeholder'}</div>
    </div>
  )
}

/* ---- guscio ---- */
export function ErpShell({ children, rightColumn }: { children: ReactNode; rightColumn?: ReactNode }) {
  const health = useHealth()
  const [query, setQuery] = useState('')
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get('presentation') === '1')
      document.documentElement.dataset.density = 'presentation'
  }, [])
  return (
    <SearchContext.Provider value={{ query, setQuery }}>
      <div className="flex min-h-screen flex-col bg-app">
        <AccentLine />
        <ErpHeader />
        {health.isError && (
          <div role="alert" data-testid="backend-down" className="bg-invalid-bg px-4 py-2 font-semibold text-invalid">
            {S.backendDown}
          </div>
        )}
        <div className="flex flex-1">
          <Sidebar />
          <main className="min-w-0 flex-1 bg-panel" style={{ border: '2px solid var(--c-titlebar-bg)' }}>{children}</main>
          {rightColumn && <aside className="hidden w-64 shrink-0 p-2 xl:block">{rightColumn}</aside>}
        </div>
      </div>
    </SearchContext.Provider>
  )
}
