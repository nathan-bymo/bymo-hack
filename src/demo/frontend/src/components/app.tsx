import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { LANGS, T, useI18n } from '../i18n'
import { S } from '../strings'

const LINKS = [
  { to: '/shop', label: T('Storefront'), end: false },
  { to: '/catalog', label: T('Catalog'), end: false },
  { to: '/', label: T('Classify'), end: true },
]

/** Guscio dell'app: marchio, le due facce della demo (vetrina e catalogo), la classificazione e la scelta della lingua. */
export function AppShell({ children }: { children: ReactNode }) {
  const { t, lang, setLang } = useI18n()
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
              <span className="block text-[17px] font-bold leading-tight tracking-tight">{t('Apertus image classifier')}</span>
              <span className="block text-[10px] uppercase leading-tight tracking-[0.13em] text-slate">{t('From photo to shop shelf')}</span>
            </span>
          </NavLink>
          <nav aria-label={t('Main')} className="flex flex-wrap gap-6 text-[15px]">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={link}>{t(l.label)}</NavLink>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-2">
            <span data-testid="demo-badge" className="text-[13px] text-slate">{S.badgeDemo}</span>
            <div role="group" aria-label={t('Language')} className="flex border border-rule bg-white" data-testid="language-switch">
              {LANGS.map((l) => (
                <button key={l.code} type="button" onClick={() => setLang(l.code)} aria-pressed={lang === l.code}
                  lang={l.code} title={l.name} data-testid={`lang-${l.code}`}
                  className={`border-0 px-2.5 py-1 text-[12px] font-bold ${lang === l.code ? 'bg-graphite text-white' : 'bg-white text-slate hover:bg-canvas hover:text-graphite'}`}>
                  {l.short}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1180px] px-6 pb-20 pt-8">{children}</main>
    </div>
  )
}
