/** Testi semplici per la presentazione: una domanda per dimensione e un nome leggibile per ogni classe.
 *  Il nome ufficiale della classe resta visibile accanto (piccolo), così nessuna parafrasi nasconde il valore vero. */

import { T, tr } from '../i18n'

export const QUESTIONS: Record<string, { question: string; name: string }> = {
  subject: { question: T('What does the image show?'), name: T('Subject') },
  background: { question: T('What is behind the product?'), name: T('Background') },
  perspective: { question: T('From where do we see the product?'), name: T('Perspective') },
  individuals: { question: T('Is a person or an animal in the picture?'), name: T('People') },
  text_classification: { question: T('Was any text added on top?'), name: T('Added text') },
  text_language: { question: T('In which language is that text?'), name: T('Language') },
}

const PLAIN: Record<string, Record<string, string>> = {
  subject: {
    full_product: T('The whole product'), detail_closeup: T('A close-up detail'), accessories: T('Accessories'),
    packaging: T('Only the packaging'), package_and_product: T('Product with its box'), multipack_trade_unit: T('A multipack'),
    delivery_scope: T('Everything in the box'), product_grouping: T('Several products together'), multiview: T('Several views in one'),
    energy_label: T('An energy label'), infographic: T('An infographic'), logo: T('A logo'), placeholder: T('A placeholder image'),
  },
  background: { white: T('White background'), color: T('Coloured background'), scene_lifestyle: T('A real-life scene') },
  perspective: {
    front_head_on: T('Straight from the front'), front_angle_left: T('Front, turned to the left'), front_angle_right: T('Front, turned to the right'),
    back_head_on: T('Straight from the back'), back_angle_left: T('Back, turned to the left'), back_angle_right: T('Back, turned to the right'),
    left_side: T('From the left side'), right_side: T('From the right side'), top_down: T('From above'), bottom_up: T('From below'),
  },
  individuals: { individual_present: T('Yes, someone is there'), none: T('No one') },
  text_classification: {
    certification_seal: T('A certification seal'), technical_specs: T('Technical specs'), promotional_text: T('Promotional text'),
    nutritional_info: T('Nutrition facts'), other: T('Some other text'), none: T('No added text'),
  },
  text_language: { German: T('German'), English: T('English'), French: T('French'), Italian: T('Italian'), Dutch: T('Dutch'), other: T('Another language'), none: T('No text') },
}

export function plainLabel(dimension: string, value: string | null | undefined): string {
  if (!value) return tr(T('No valid answer'))
  const plain = PLAIN[dimension]?.[value]
  return plain ? tr(plain) : value.replace(/_/g, ' ')
}

/** Le categorie del dataset (fisse, in inglese; "Unkategorisiert" è già tedesco): si traducono come i testi dell'interfaccia. */
export const CATEGORY_NAMES = [
  T('Apparel & Baby'), T('Beauty & Supermarket'), T('Computing'), T('Consumer Electronics'), T('Do It & Garden'),
  T('Domestic Appliances (SDA/MDA)'), T('Furniture'), T('Household'), T('Mobiles'), T('Office & Gaming'), T('Sport & Toys'),
  T('Unkategorisiert'),
]
export const categoryLabel = (name: string): string => tr(name)   // una categoria creata a mano resta com'è

export const questionText = (dimension: string): string => tr(QUESTIONS[dimension].question)
export const questionName = (dimension: string): string => tr(QUESTIONS[dimension].name)
