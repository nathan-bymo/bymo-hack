import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { S } from '../strings'

const LINKS = [
  { to: '/shop', label: 'Storefront', end: false },
  { to: '/catalog', label: 'Catalog', end: false },
  { to: '/', label: 'Classify', end: true },
]

/** Guscio dell'app: marchio, le due facce della demo (vetrina e catalogo) e la classificazione. */
export function AppShell({ children }: { children: ReactNode }) {
  const link = ({ isActive }: { isActive: boolean }) =>
    `border-b-2 pb-1 no-underline ${isActive ? 'border-signal text-graphite' : 'border-transparent text-slate hover:text-graphite'}`
  return (
    <div className="swiss min-h-screen">
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-x-9 gap-y-3 px-6 py-4">
          <NavLink to="/" className="flex items-center gap-2.5 no-underline">
            <span aria-hidden="true" className="relative inline-block h-6 w-6 bg-signal">
              <span className="absolute left-[10px] top-[4px] h-4 w-1 bg-white" />
              <span className="absolute left-[4px] top-[10px] h-1 w-4 bg-white" />
            </span>
            <span>
              <span className="block text-[17px] font-bold leading-tight tracking-tight">Apertus image classifier</span>
              <span className="block text-[10px] uppercase leading-tight tracking-[0.13em] text-slate">From photo to shop shelf</span>
            </span>
          </NavLink>
          <nav aria-label="Main" className="flex flex-wrap gap-6 text-[15px]">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={link}>{l.label}</NavLink>
            ))}
          </nav>
          <span data-testid="demo-badge" className="ml-auto text-[13px] text-slate">{S.badgeDemo}</span>
        </div>
      </header>
      <main className="mx-auto max-w-[1180px] px-6 pb-20 pt-8">{children}</main>
    </div>
  )
}
