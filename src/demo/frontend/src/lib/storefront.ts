import type { ImageItem, PredictionResult, StorefrontRule } from '../api/types'
import { DIMENSIONS } from '../strings'
import { minConfidence } from './predictions'

export interface Placed {
  image: ImageItem
  position: number | null      // null = nascosta
  group: string | null
  reason: string
  values: Record<string, string>
  reviewed: boolean
}

const matches = (when: Record<string, string[]>, values: Record<string, string>) =>
  Object.entries(when).every(([d, allowed]) => allowed.includes(values[d]))
const describe = (when: Record<string, string[]>) => Object.entries(when).map(([d, v]) => `${d} ${v.join('/')}`).join(', ')

/** Applica la regola illustrativa ai valori rivisti (se ci sono) o a quelli predetti. */
export function applyRule(rule: StorefrontRule, images: ImageItem[], preds: Map<string, PredictionResult>): Placed[] {
  const items = images.map((image) => {
    const pred = preds.get(image.image_id)
    const reviewed = !!image.review_values
    const values: Record<string, string> = image.review_values ?? Object.fromEntries(DIMENSIONS.map((d) => [d, pred?.dimensions[d]?.label ?? '']))
    return { image, pred, values, reviewed }
  })
  const visible: (typeof items[number] & { rank: number; group: string; conf: number })[] = []
  const out: Placed[] = []
  for (const it of items) {
    if (!it.pred && !it.reviewed) {
      out.push({ image: it.image, position: null, group: null, reason: 'No prediction yet: run a model first.', values: it.values, reviewed: false })
      continue
    }
    if (matches(rule.hide_if, it.values)) {
      out.push({ image: it.image, position: null, group: null, reason: `Hidden by the rule (${describe(rule.hide_if)}).`, values: it.values, reviewed: it.reviewed })
      continue
    }
    let rank = rule.rank.findIndex((r) => matches(r.when, it.values))
    const group = rank >= 0 ? rule.rank[rank].name : 'other'
    if (rank < 0) rank = rule.rank.length
    visible.push({ ...it, rank, group, conf: minConfidence(it.pred) ?? 0 })
  }
  visible.sort((a, b) => a.rank - b.rank || b.conf - a.conf)
  visible.forEach((v, i) => {
    const r = rule.rank[v.rank]
    out.push({ image: v.image, position: i + 1, group: v.group, values: v.values, reviewed: v.reviewed,
      reason: `Rule "${v.group}"${r ? ` (${describe(r.when)})` : ' (no group matched)'}${v.reviewed ? ' — using the human-reviewed values' : ''}.` })
  })
  return out.sort((a, b) => (a.position ?? 999) - (b.position ?? 999))
}
