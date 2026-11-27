import type { ModelSummary, PredictionResult } from '../api/types'
import { DIMENSIONS } from '../strings'

/** Ultima predizione per immagine (la lista arriva in ordine di creazione). */
export function latestByImage(preds: PredictionResult[] | undefined): Map<string, PredictionResult> {
  const map = new Map<string, PredictionResult>()
  for (const p of preds ?? []) map.set(p.image_id, p)
  return map
}

/** Confidenza minima sulle dimensioni applicabili; null se non c'è nessuna confidenza. */
export function minConfidence(p: PredictionResult | undefined): number | null {
  if (!p) return null
  const values = DIMENSIONS.map((d) => p.dimensions[d]).filter((d) => d.applicable && d.confidence !== null).map((d) => d.confidence as number)
  return values.length ? Math.min(...values) : null
}

export function defaultMode(model: ModelSummary | undefined): 'live' | 'recorded' {
  return model?.modes.includes('live') ? 'live' : 'recorded'
}
export function isRunnable(model: ModelSummary | undefined, mode: 'live' | 'recorded'): boolean {
  return !!model && model.modes.includes(mode)
}
