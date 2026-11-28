import { expect, test } from '@playwright/test'
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

  // 2. upload di 3 immagini sintetiche + un file non valido
  await page.getByTestId('try-file-input').setInputFiles(files('synthetic-1.jpg', 'synthetic-2.jpg', 'synthetic-3.jpg', 'bad.png'))
  await expect(page.getByTestId('try-thumbs').locator('button')).toHaveCount(3)
  await expect(page.getByRole('alert')).toContainText('bad.png')
  await expect(page.getByTestId('try-image')).toBeVisible()

  // 3. due modelli registrati (registrazioni segnaposto del test) a confronto sulla stessa foto
  await expect(page.getByTestId('try-model-apertus-8b-lora-heads-p11')).toBeChecked()   // scelta iniziale
  await expect(page.getByTestId('try-model-apertus-8b-zeroshot-v2')).toHaveCount(0)   // senza chiave non compare
  await expect(page.getByTestId('try-model-apertus-8b-vadapter-lora-heads-final')).toHaveCount(0)   // nessuna registrazione nel test
  await expect(page.getByText(/needs a gpu/i)).toHaveCount(0)
  await page.getByTestId('try-model-apertus-8b-fullft-heads').check()
  await page.getByTestId('try-thumbs').locator('button').first().click()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })   // nessun pulsante: le risposte arrivano da sole
  await expect(page.getByTestId('try-classify')).toHaveCount(0)
  await expect(page.getByTestId('col-apertus-8b-fullft-heads')).toContainText('Recorded', { timeout: 30_000 })
  for (const d of DIMENSIONS) await expect(page.getByTestId(`try-dim-${d}`)).toBeVisible()
  await expect(page.getByTestId('try-dim-perspective').getByTestId('try-disagree')).toBeVisible()
  await expect(page.getByTestId('try-error')).toHaveCount(0)
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await expect(page.getByText('These answers use the photo only')).toBeVisible()
  await expect(page.getByTestId('col-apertus-8b-lora-heads-p11')).toContainText('Recorded')
  await shot(page, '02-classify-two-models')
  await shot(page, '02b-video-frame', false)   // quello che si vede nel video, senza scorrere

  // 3b. dettagli prodotto facoltativi: si salvano e restano sulla foto
  await page.getByTestId('product-details').getByRole('button').first().click()
  await page.getByTestId('ctx-product_title').fill('Ceramic mug 350 ml')
  await page.getByTestId('ctx-brand').fill('Demo brand')
  await page.getByTestId('save-context').click()
  await expect(page.getByTestId('product-details')).toContainText('2 of 5 filled')

  // 4. un solo modello su un'altra foto
  await page.getByTestId('try-model-apertus-8b-fullft-heads').uncheck()
  await page.getByTestId('try-thumbs').locator('button').nth(1).click()
  await expect(page.getByTestId('try-dim-subject')).toContainText('The whole product', { timeout: 30_000 })
  await expect(page.getByTestId('try-disagree')).toHaveCount(0)
  await shot(page, '03-classify-one-model')

  // 5. Results e How it works non esistono più: i vecchi indirizzi portano alla pagina Demo
  for (const old of ['/models', '/results', '/how']) {
    await page.goto(old)
    await expect(page).toHaveURL(/\/(\?.*)?$/)
  }
  await page.goto('/image-classification')
  await expect(page.getByRole('heading', { name: 'Six answers for every product photo' })).toBeVisible()
})
