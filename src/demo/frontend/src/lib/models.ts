import type { ModelSummary, ScoreInfo, SpeedInfo } from '../api/types'

/** Descrizione di ogni set di valutazione: i numeri si confrontano solo dentro lo stesso set. */
export const SET_INFO: Record<string, { title: string; images: string; what: string }> = {
  kaggle_dev: {
    title: 'Kaggle dev — public leaderboard',
    images: '5,261 images',
    what: 'Official metric on the hidden dev labels. The only set shared with the organisers’ references.',
  },
  'prompt-test': {
    title: 'prompt-test — internal held-out',
    images: '1,193 images',
    what: 'Part of train.csv never used for training or prompt choice by the models marked held-out. Natural class distribution.',
  },
  coverage: {
    title: 'coverage — internal, rare classes enriched',
    images: '2,532 images',
    what: 'Phase-1 test set with extra rare classes, made to compare methods. Harder than the leaderboard: not an estimate of it.',
  },
}

export const SCORE_SET_LABEL: Record<string, string> = {
  kaggle_dev: 'Kaggle dev',
  'prompt-test': 'prompt-test',
  coverage: 'coverage',
  base: 'base',
  validation_450: 'validation_450',
}

export function weightsText(model: ModelSummary): string {
  const w = model.weights
  if (!w) return model.runtime === 'remote_openai' ? 'remote (gateway)' : 'not on this machine'
  if (w.location_label) return w.location_label
  if (!w.repo || w.repo.includes('DA VERIFICARE')) return 'not accessible yet (repo to be confirmed)'
  return `HF ${w.repo.split('/')[1] ?? w.repo}${w.size_gb ? `, ${w.size_gb} GB` : ''}`
}

/** Percorso completo dei pesi, per il tooltip. */
export function weightsDetail(model: ModelSummary): string | undefined {
  const w = model.weights
  if (!w) return undefined
  return w.local_path ?? (w.repo ? `${w.repo}@${w.revision?.slice(0, 7)} · ${w.subfolder}` : undefined)
}

/** Il punteggio "di riferimento" di un modello: Kaggle dev se c'è, altrimenti prompt-test, poi coverage. */
export function headlineScore(model: ModelSummary): ScoreInfo | null {
  const order = ['kaggle_dev', 'prompt-test', 'coverage', 'base', 'validation_450']
  for (const set of order) {
    const s = model.scores.find((x) => x.set === set && x.macro_f1 !== null)
    if (s) return s
  }
  return null
}

/** La velocità end to end più economica (costo per 100.000 immagini più basso), altrimenti la prima riportata. */
export function bestSpeed(model: ModelSummary): SpeedInfo | null {
  const preferred = model.speeds.find((s) => s.preferred)   // configurazione di consegna indicata dal team
  if (preferred) return preferred
  const costed = model.speeds.filter((s) => s.cost_per_100k_usd !== null)
  if (costed.length) return [...costed].sort((a, b) => (a.cost_per_100k_usd ?? 0) - (b.cost_per_100k_usd ?? 0))[0]
  return model.speeds[0] ?? null
}

export function memoryText(model: ModelSummary): string {
  if (model.runtime === 'remote_openai') return 'none on our side (API)'
  const m = model.memory
  if (!m) return '—'
  if (m.gb === null) return m.what.charAt(0).toUpperCase() + m.what.slice(1)
  return `${m.gb} GB ${m.kind === 'measured' ? 'measured' : 'estimate'}`
}

export function usd(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : `$${value.toFixed(2)}`
}

/** Etichetta breve dello stato: cosa può fare oggi questo modello nella demo. */
export function availability(model: ModelSummary): { text: string; tone: 'live' | 'recorded' | 'off' } {
  if (model.modes.includes('live')) return { text: 'Live', tone: 'live' }
  if (model.modes.includes('recorded')) return { text: 'Recorded', tone: 'recorded' }
  return { text: 'Scores', tone: 'off' }
}

/** Ordine di presentazione: prima i modelli con punteggio Kaggle (dal migliore), poi gli altri per punteggio interno. */
export function sortForShow(models: ModelSummary[]): ModelSummary[] {
  const key = (m: ModelSummary) => {
    const kaggle = m.scores.find((s) => s.set === 'kaggle_dev')?.macro_f1
    if (kaggle != null) return 2 + kaggle
    const internal = m.scores.find((s) => s.set === 'prompt-test')?.macro_f1 ?? m.scores[0]?.macro_f1 ?? 0
    return (m.featured ? 1 : 0) + internal
  }
  return [...models].sort((a, b) => key(b) - key(a))
}
