import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const files = (...names: string[]) => names.map((n) => join(process.env.E2E_ROOT!, 'files', n))
const shot = (page: import('@playwright/test').Page, name: string, fullPage = true) => page.screenshot({ path: `e2e/screenshots/${name}.png`, fullPage })
const DIMENSIONS = ['subject', 'background', 'perspective', 'individuals', 'text_classification', 'text_language']

test('percorso completo della demo', async ({ page }) => {
  // 1. Demo: pagina iniziale, nessuna foto
  await page.goto('/')
  await expect(page.getByTestId('demo-badge')).toHaveText('Team Bymo, Hack Apertus × Galaxus')
  await expect(page).toHaveURL(/session=ses_/)
  await expect(page.getByTestId('try-empty')).toContainText('Add a photo')
  await shot(page, '01-empty')

  // 1b. foto di esempio: un clic e la risposta registrata compare subito, senza caricare nulla
  await expect(page.getByTestId('samples')).toBeVisible()
  await page.getByTestId('sample').first().click()
  await expect(page.getByTestId('try-image')).toBeVisible()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('sample').first()).toHaveAttribute('data-ready', 'true')
  await expect(page.getByTestId('col-reference')).toContainText('Training label')   // l'etichetta di train.csv, separata dal modello
  await expect(page.getByTestId('reference-subject')).toContainText('The whole product')
  await expect(page.getByTestId('reference-text_language')).toContainText('Not a human label')
  await page.getByTestId('sample').nth(1).click()   // un altro sample SOVRASCRIVE la foto corrente: la lista non cresce
  await expect(page.getByTestId('try-thumbs')).toHaveCount(0)
  await expect(page.getByTestId('col-reference')).toHaveCount(0)   // il secondo sample non ha etichette: la colonna del primo è sparita
  await page.getByTestId('try-remove').click()   // si toglie dalla pagina per ripartire da vuoto
  await expect(page.getByTestId('try-empty')).toContainText('Add a photo')

  // 2. upload di 3 immagini sintetiche + un file non valido
  await page.getByTestId('try-file-input').setInputFiles(files('synthetic-1.jpg', 'synthetic-2.jpg', 'synthetic-3.jpg', 'bad.png'))
  await expect(page.getByTestId('try-thumbs').locator('button')).toHaveCount(3)
  await expect(page.getByRole('alert')).toContainText('bad.png')
  await expect(page.getByTestId('try-image')).toBeVisible()

  // 3. un modello alla volta: si sceglie e compaiono le risposte e il suo punteggio
  await expect(page.getByTestId('try-models').locator('input[type=radio]:checked')).toHaveCount(1)   // un solo modello scelto
  await expect(page.getByTestId('try-model-apertus-8b-zeroshot-v2')).toBeDisabled()   // senza chiave non risponde: si vede ma non si sceglie
  await page.getByTestId('try-model-apertus-8b-fullft-heads').check()
  await page.getByTestId('try-thumbs').locator('button').first().click()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })   // nessun pulsante: le risposte arrivano da sole
  await expect(page.getByTestId('try-classify')).toHaveCount(0)
  await expect(page.getByTestId('col-apertus-8b-fullft-heads')).toContainText('Recorded', { timeout: 30_000 })
  for (const d of DIMENSIONS) {
    await expect(page.getByTestId(`try-dim-${d}`)).toBeVisible()
    await expect(page.getByTestId(`classes-${d}`).getByTestId('class-percent').first()).toHaveText(/^\d+%$/)   // ogni domanda: classe e percentuale
  }
  await expect(page.getByTestId('model-score')).toHaveCount(0)   // nessuna macro F1 nella pagina di classificazione
  await expect(page.getByTestId('try-disagree')).toHaveCount(0)
  await expect(page.getByTestId('try-error')).toHaveCount(0)
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await expect(page.getByText('It uses the photo only')).toBeVisible()
  await shot(page, '02-classify-one-model')
  await shot(page, '02b-video-frame', false)   // quello che si vede nel video, senza scorrere

  // 4. cambio modello sulla stessa foto: cambiano risposte e punteggio, non c'è nessuna media tra modelli
  await page.getByTestId('try-model-apertus-8b-lora-heads-p11').check()
  await expect(page.getByTestId('col-apertus-8b-lora-heads-p11')).toContainText('Recorded', { timeout: 30_000 })
  await expect(page.getByTestId('col-apertus-8b-fullft-heads')).toHaveCount(0)
  await page.getByTestId('try-thumbs').locator('button').nth(1).click()
  await expect(page.getByTestId('try-dim-subject')).toContainText('The whole product', { timeout: 30_000 })
  await shot(page, '03-classify-other-photo')

  // 4a. dettagli prodotto facoltativi: si salvano e restano sulla foto (cambiano il contesto, quindi per ultimi)
  await page.getByTestId('product-details').getByRole('button').first().click()
  await page.getByTestId('ctx-product_title').fill('Ceramic mug 350 ml')
  await page.getByTestId('ctx-brand').fill('Demo brand')
  await page.getByTestId('save-context').click()
  await expect(page.getByTestId('product-details')).toContainText('2 of 5 filled')

  // 4b. un modello che non risponde in questa demo non si può scegliere
  await expect(page.getByTestId('try-model-apertus-team-best-submission')).toBeDisabled()

  // 4c. si può togliere una foto dalla pagina (solo nel browser) e resta tolta dopo il refresh
  const thumbs = page.getByTestId('try-thumbs').locator('button')
  await expect(thumbs).toHaveCount(3)
  await page.getByTestId('try-remove').click()
  await expect(thumbs).toHaveCount(2)
  await page.reload()
  await expect(thumbs).toHaveCount(2)
  await page.getByTestId('try-remove').click()
  await page.getByTestId('try-remove').click()
  await expect(page.getByTestId('try-image')).toHaveCount(0)
  await expect(page.getByTestId('try-remove')).toHaveCount(0)
  await expect(page.getByTestId('try-empty')).toContainText('Add a photo')
  await page.reload()
  await expect(page.getByTestId('try-image')).toHaveCount(0)   // dopo il refresh non ricompare
  await page.getByTestId('try-file-input').setInputFiles(files('synthetic-1.jpg'))   // la stessa foto ricaricata torna visibile
  await expect(page.getByTestId('try-image')).toBeVisible()

  // 5. Results e How it works non esistono più: i vecchi indirizzi portano alla pagina Demo
  for (const old of ['/models', '/results', '/how']) {
    await page.goto(old)
    await expect(page).toHaveURL(/\/(\?.*)?$/)
  }
  await page.goto('/image-classification')
  await expect(page.getByRole('heading', { name: 'Six answers for every product photo' })).toBeVisible()
})

/** Lingue: inglese di partenza, italiano / francese / tedesco a scelta; nessuna frase inglese deve restare visibile. */
test('lingue dell\'interfaccia', async ({ page }) => {
  const dict = (lang: string): Record<string, string> =>
    JSON.parse(readFileSync(join(import.meta.dirname, '..', 'src', 'i18n', `${lang}.json`), 'utf8'))
  const visibleTexts = () => page.evaluate(() => {
    const out = new Set<string>()
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    while (walker.nextNode()) { const t = walker.currentNode.textContent?.trim(); if (t) out.add(t) }
    document.querySelectorAll('[title],[aria-label],[placeholder],[alt]').forEach((el) => {
      if (el.closest('[data-testid=language-switch]')) return   // i nomi delle lingue si scrivono nella lingua stessa
      ['title', 'aria-label', 'placeholder', 'alt'].forEach((a) => { const v = el.getAttribute(a); if (v) out.add(v.trim()) })
    })
    return [...out]
  })
  const expected: Record<string, { nav: string; h1: string; title: string }> = {
    it: { nav: 'Classifica', h1: 'Sei risposte per ogni foto di prodotto', title: 'Classificazione delle immagini' },
    fr: { nav: 'Classifier', h1: 'Six réponses pour chaque photo de produit', title: "Classification d'images" },
    de: { nav: 'Klassifizieren', h1: 'Sechs Antworten für jedes Produktfoto', title: 'Bildklassifizierung' },
  }

  await page.goto('/')   // inglese di partenza
  await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Six answers for every product photo')
  await expect(page.getByTestId('lang-en')).toHaveAttribute('aria-pressed', 'true')

  for (const lang of ['it', 'fr', 'de']) {
    await page.goto('/')
    await page.getByTestId(`lang-${lang}`).click()
    await expect(page.locator('html')).toHaveAttribute('lang', lang)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(expected[lang].h1)
    await expect(page.getByRole('link', { name: expected[lang].nav, exact: true })).toBeVisible()
    await expect(page).toHaveTitle(new RegExp(expected[lang].title))
    await page.reload()   // la scelta resta dopo il refresh
    await expect(page.locator('html')).toHaveAttribute('lang', lang)
    for (const path of ['/', '/catalog', '/shop', '/shop?q=nulla-del-genere']) {
      await page.goto(path)
      await expect(page.getByTestId(`lang-${lang}`)).toHaveAttribute('aria-pressed', 'true')
      await page.waitForTimeout(400)
      const leftovers = (await visibleTexts()).filter((t) => dict(lang)[t] !== undefined && dict(lang)[t] !== t)
      expect(leftovers, `${lang} ${path}: frasi rimaste in inglese`).toEqual([])
    }
  }

  await page.goto('/')
  await page.getByTestId('lang-en').click()   // si torna all'inglese
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Six answers for every product photo')
})
