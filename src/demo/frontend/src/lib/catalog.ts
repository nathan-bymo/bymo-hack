import type { CompactImage, StorefrontFilters } from '../api/types'
import { T, tr } from '../i18n'
import { plainLabel } from './labels'

/**
 * Regola unica per l'ordine delle viste (la stessa del backend, `catalog.py`): prima le viste ad angolo a sinistra, poi
 * quelle ad angolo a destra (davanti prima del retro), poi tutte le altre; le foto senza vista letta vanno in fondo.
 * La copertina scelta dall'admin va sempre per prima.
 */
export const PERSPECTIVE_ORDER = [
  'front_angle_left', 'back_angle_left', 'front_angle_right', 'back_angle_right',
  'front_head_on', 'left_side', 'right_side', 'back_head_on', 'top_down', 'bottom_up',
] as const

export const viewOrderRule = (): string =>
  tr('Angled-left views first, then angled-right views, then every other view; photos without a read view come last. The cover always leads.')


export function perspectiveRank(value: string | null | undefined): number {
  const index = PERSPECTIVE_ORDER.indexOf(value as (typeof PERSPECTIVE_ORDER)[number])
  return index === -1 ? PERSPECTIVE_ORDER.length : index
}

export function perspectiveLabel(value: string | null | undefined): string {
  return value ? plainLabel('perspective', value) : tr('View not read yet')
}

/** Galleria: prima la copertina, poi le altre viste secondo la regola dell'ordine. */
export function galleryOrder(images: CompactImage[]): CompactImage[] {
  return [...images].sort((a, b) => {
    if (a.is_cover !== b.is_cover) return a.is_cover ? -1 : 1
    return perspectiveRank(a.values.perspective) - perspectiveRank(b.values.perspective) || a.position - b.position
  })
}

const NON_PHOTOS = new Set(['placeholder', 'logo', 'energy_label', 'infographic'])

/** Le stesse regole del backend: l'admin vede subito se una foto passa i filtri del negozio. */
export function passesShopRules(
  values: Record<string, string | null>,
  confidence: number | null,
  filters: StorefrontFilters,
): boolean {
  if (filters.hide_non_photos && NON_PHOTOS.has(values.subject ?? '')) return false
  if (filters.only_white_background && values.background !== 'white') return false
  if (filters.hide_people && values.individuals !== 'none') return false
  if (filters.hide_added_text && values.text_classification !== 'none') return false
  if (filters.min_confidence > 0 && (confidence === null || confidence < filters.min_confidence)) return false
  return true
}

/** Perché una foto non passa i filtri, in una riga sola. */
export function ruleBlockers(
  values: Record<string, string | null>,
  confidence: number | null,
  filters: StorefrontFilters,
): string[] {
  const out: string[] = []
  if (filters.hide_non_photos && NON_PHOTOS.has(values.subject ?? '')) out.push(tr('not a product photo'))
  if (filters.only_white_background && values.background !== 'white') out.push(tr('background is not white'))
  if (filters.hide_people && values.individuals !== 'none') out.push(tr('someone is in the photo'))
  if (filters.hide_added_text && values.text_classification !== 'none') out.push(tr('text was added on top'))
  if (filters.min_confidence > 0 && (confidence === null || confidence < filters.min_confidence))
    out.push(confidence === null ? tr('no confidence yet') : tr('confidence below {n}%', { n: Math.round(filters.min_confidence * 100) }))
  return out
}

export function confidenceTone(value: number | null): 'low' | 'mid' | 'high' | 'none' {
  if (value === null) return 'none'
  return value < 0.6 ? 'low' : value >= 0.9 ? 'high' : 'mid'
}

export function percent(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

export const statusLabel = (status: 'live' | 'draft' | 'empty'): string =>
  status === 'live' ? tr('Live') : status === 'draft' ? tr('Draft') : tr('Empty')

const FILTER_LABELS: Record<keyof StorefrontFilters, { title: string; hint: string }> = {
  hide_non_photos: { title: T('Only product photos'), hint: T('Hide placeholders, logos, energy labels and infographics.') },
  only_white_background: { title: T('White background only'), hint: T('Keep photos that can sit on a clean product page.') },
  hide_people: { title: T('No people or animals'), hint: T('Hide photos with a person or an animal in them.') },
  hide_added_text: { title: T('No text added on top'), hint: T('Hide promotional text, seals and spec sheets burnt into the image.') },
  min_confidence: { title: T('Lowest confidence allowed'), hint: T('Hide photos where any of the six answers is below this.') },
}

export const FILTER_KEYS = Object.keys(FILTER_LABELS) as (keyof StorefrontFilters)[]

export const filterLabel = (key: keyof StorefrontFilters): { title: string; hint: string } => ({
  title: tr(FILTER_LABELS[key].title),
  hint: tr(FILTER_LABELS[key].hint),
})
