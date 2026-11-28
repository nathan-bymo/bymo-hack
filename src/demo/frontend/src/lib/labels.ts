/** Testi semplici per la presentazione: una domanda per dimensione e un nome leggibile per ogni classe.
 *  Il nome ufficiale della classe resta visibile accanto (piccolo), così nessuna parafrasi nasconde il valore vero. */

export const QUESTIONS: Record<string, { question: string; name: string }> = {
  subject: { question: 'What does the image show?', name: 'Subject' },
  background: { question: 'What is behind the product?', name: 'Background' },
  perspective: { question: 'From where do we see the product?', name: 'Perspective' },
  individuals: { question: 'Is a person or an animal in the picture?', name: 'People' },
  text_classification: { question: 'Was any text added on top?', name: 'Added text' },
  text_language: { question: 'In which language is that text?', name: 'Language' },
}

const PLAIN: Record<string, Record<string, string>> = {
  subject: {
    full_product: 'The whole product', detail_closeup: 'A close-up detail', accessories: 'Accessories',
    packaging: 'Only the packaging', package_and_product: 'Product with its box', multipack_trade_unit: 'A multipack',
    delivery_scope: 'Everything in the box', product_grouping: 'Several products together', multiview: 'Several views in one',
    energy_label: 'An energy label', infographic: 'An infographic', logo: 'A logo', placeholder: 'A placeholder image',
  },
  background: { white: 'White background', color: 'Coloured background', scene_lifestyle: 'A real-life scene' },
  perspective: {
    front_head_on: 'Straight from the front', front_angle_left: 'Front, turned to the left', front_angle_right: 'Front, turned to the right',
    back_head_on: 'Straight from the back', back_angle_left: 'Back, turned to the left', back_angle_right: 'Back, turned to the right',
    left_side: 'From the left side', right_side: 'From the right side', top_down: 'From above', bottom_up: 'From below',
  },
  individuals: { individual_present: 'Yes, someone is there', none: 'No one' },
  text_classification: {
    certification_seal: 'A certification seal', technical_specs: 'Technical specs', promotional_text: 'Promotional text',
    nutritional_info: 'Nutrition facts', other: 'Some other text', none: 'No added text',
  },
  text_language: { German: 'German', English: 'English', French: 'French', Italian: 'Italian', Dutch: 'Dutch', other: 'Another language', none: 'No text' },
}

export function plainLabel(dimension: string, value: string | null | undefined): string {
  if (!value) return 'No valid answer'
  return PLAIN[dimension]?.[value] ?? value.replace(/_/g, ' ')
}
