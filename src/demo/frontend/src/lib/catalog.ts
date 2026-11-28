import type { CompactImage, StorefrontFilters } from '../api/types'
import { plainLabel } from './labels'

/** Ordine delle viste: lo stesso della tassonomia, così la galleria segue il giro del prodotto. */
export const PERSPECTIVE_ORDER = [
  'front_head_on', 'front_angle_left', 'front_angle_right', 'left_side', 'right_side',
  'back_head_on', 'back_angle_left', 'back_angle_right', 'top_down', 'bottom_up',
] as const

export function perspectiveRank(value: string | null | undefined): number {
  const index = PERSPECTIVE_ORDER.indexOf(value as (typeof PERSPECTIVE_ORDER)[number])
  return index === -1 ? PERSPECTIVE_ORDER.length : index
}

export function perspectiveLabel(value: string | null | undefined): string {
  return value ? plainLabel('perspective', value) : 'View not read yet'
}

/** Galleria: prima la copertina, poi le altre viste in ordine di prospettiva. */
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
  if (filters.hide_non_photos && NON_PHOTOS.has(values.subject ?? '')) out.push('not a product photo')
  if (filters.only_white_background && values.background !== 'white') out.push('background is not white')
  if (filters.hide_people && values.individuals !== 'none') out.push('someone is in the photo')
  if (filters.hide_added_text && values.text_classification !== 'none') out.push('text was added on top')
  if (filters.min_confidence > 0 && (confidence === null || confidence < filters.min_confidence))
    out.push(confidence === null ? 'no confidence yet' : `confidence below ${Math.round(filters.min_confidence * 100)}%`)
  return out
}

export function confidenceTone(value: number | null): 'low' | 'mid' | 'high' | 'none' {
  if (value === null) return 'none'
  return value < 0.6 ? 'low' : value >= 0.9 ? 'high' : 'mid'
}

export function percent(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

export const STATUS_LABEL: Record<'live' | 'draft' | 'empty', string> = {
  live: 'Live',
  draft: 'Draft',
  empty: 'Empty',
}

export const FILTER_LABELS: Record<keyof StorefrontFilters, { title: string; hint: string }> = {
  hide_non_photos: { title: 'Only product photos', hint: 'Hide placeholders, logos, energy labels and infographics.' },
  only_white_background: { title: 'White background only', hint: 'Keep photos that can sit on a clean product page.' },
  hide_people: { title: 'No people or animals', hint: 'Hide photos with a person or an animal in them.' },
  hide_added_text: { title: 'No text added on top', hint: 'Hide promotional text, seals and spec sheets burnt into the image.' },
  min_confidence: { title: 'Lowest confidence allowed', hint: 'Hide photos where any of the six answers is below this.' },
}
