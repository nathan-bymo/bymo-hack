/**
 * Traduzioni dell'interfaccia: inglese (predefinito), italiano, francese, tedesco.
 *
 * Il testo inglese è la chiave: `tr('Add a photo')` restituisce la stessa frase in inglese e la traduzione dai file
 * `it.json`, `fr.json`, `de.json` nelle altre lingue. `T('…')` serve solo a segnare una frase che si traduce più tardi
 * (tabelle di etichette): `scripts/check-i18n.mjs` legge `tr(`, `trn(` e `T(` e verifica che ogni frase sia in tutte le lingue.
 * I dati (titoli, marche, categorie dei prodotti) non si traducono: sono quelli del catalogo.
 */
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import de from './de.json'
import fr from './fr.json'
import it from './it.json'

export type Lang = 'en' | 'it' | 'fr' | 'de'
export const LANGS: { code: Lang; short: string; name: string }[] = [
  { code: 'en', short: 'EN', name: 'English' },
  { code: 'it', short: 'IT', name: 'Italiano' },
  { code: 'fr', short: 'FR', name: 'Français' },
  { code: 'de', short: 'DE', name: 'Deutsch' },
]
const DICTIONARIES: Record<Lang, Record<string, string>> = { en: {}, it, fr, de }
const STORAGE_KEY = 'demo.lang'

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && saved in DICTIONARIES) return saved as Lang
  } catch {
    /* localStorage non disponibile: resta l'inglese */
  }
  return 'en'
}

let current: Lang = initialLang()

/** Segna una frase da tradurre senza tradurla subito. */
export const T = (text: string): string => text

type Params = Record<string, string | number>

export function tr(text: string, params?: Params): string {
  const base = DICTIONARIES[current][text] ?? text
  return params ? base.replace(/\{(\w+)\}/g, (_, key: string) => String(params[key] ?? `{${key}}`)) : base
}

/** Singolare o plurale: le due frasi contengono `{n}`. */
export function trn(n: number, one: string, other: string, params: Params = {}): string {
  return tr(n === 1 ? one : other, { n, ...params })
}

const Context = createContext<{ lang: Lang; setLang: (lang: Lang) => void }>({ lang: 'en', setLang: () => {} })

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(current)
  const setLang = (next: Lang) => {
    current = next   // prima della nuova visualizzazione: `tr` legge questa variabile
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* non salvata: vale per questa visita */
    }
    setLangState(next)
  }
  useEffect(() => {
    document.documentElement.lang = lang
    document.title = tr('Galaxus × Apertus — Image classification')
  }, [lang])
  return <Context.Provider value={{ lang, setLang }}>{children}</Context.Provider>
}

/** Ogni componente con testo lo chiama: così si ridisegna quando cambia la lingua. */
export function useI18n() {
  const { lang, setLang } = useContext(Context)
  return { lang, setLang, t: tr, tn: trn }
}
