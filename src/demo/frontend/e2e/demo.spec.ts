import { expect, test } from '@playwright/test'
import { join } from 'node:path'

const files = (...names: string[]) => names.map((n) => join(process.env.E2E_ROOT!, 'files', n))
const shot = (page: import('@playwright/test').Page, name: string) => page.screenshot({ path: `e2e/screenshots/${name}.png`, fullPage: true })
const DIMENSIONS = ['subject', 'background', 'perspective', 'individuals', 'text_classification', 'text_language']

test('percorso completo della demo', async ({ page }) => {
  // 1. Classify: pagina iniziale, nessuna foto
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
  await expect(page.getByTestId('try-model-apertus-8b-lora-heads-p11')).toBeChecked()
  await expect(page.getByTestId('try-model-apertus-8b-fullft-heads')).toBeChecked()
  await expect(page.getByTestId('try-model-apertus-8b-zeroshot-v2')).toBeDisabled()   // nessuna chiave nel test
  await page.getByTestId('try-thumbs').locator('button').first().click()
  await page.getByTestId('try-classify').click()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })
  for (const d of DIMENSIONS) await expect(page.getByTestId(`try-dim-${d}`)).toBeVisible()
  await expect(page.getByTestId('try-dim-perspective').getByTestId('try-disagree')).toBeVisible()
  await expect(page.getByTestId('try-error')).toHaveCount(0)
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await expect(page.getByText('These answers use the photo only')).toBeVisible()
  await shot(page, '02-classify-two-models')

  // 3b. dettagli prodotto facoltativi: si salvano e restano sulla foto
  await page.getByTestId('product-details').getByRole('button').first().click()
  await page.getByTestId('ctx-product_title').fill('Ceramic mug 350 ml')
  await page.getByTestId('ctx-brand').fill('Demo brand')
  await page.getByTestId('save-context').click()
  await expect(page.getByTestId('product-details')).toContainText('2 of 5 filled')

  // 4. un solo modello su un'altra foto
  await page.getByTestId('try-model-apertus-8b-fullft-heads').uncheck()
  await page.getByTestId('try-thumbs').locator('button').nth(1).click()
  await page.getByTestId('try-classify').click()
  await expect(page.getByTestId('try-dim-subject')).toContainText('The whole product', { timeout: 30_000 })
  await expect(page.getByTestId('try-disagree')).toHaveCount(0)
  await shot(page, '03-classify-one-model')

  // 5. Models: classifica Kaggle, velocità e costo, test interno; nessuna scritta "invalid"
  await page.getByRole('link', { name: 'Models', exact: true }).click()
  const kaggle = page.getByTestId('board-kaggle_dev')
  await expect(kaggle.getByTestId('board-row')).toHaveCount(6)   // 3 modelli del team + 3 riferimenti
  await expect(kaggle.getByTestId('board-row').first()).toContainText('0.917')
  await expect(kaggle).toContainText('0.832')
  await expect(page.getByTestId('board-prompt-test')).toContainText('0.821')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-fullft-heads"]')).toContainText('$7.27')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-lora-heads-p11"]')).toContainText('22 GB measured')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-fullft-heads"]').getByTestId('hours-100k')).toHaveText('2.9 h')
  await expect(page.getByTestId('insights-mattered').locator('li')).toHaveCount(5)
  await expect(page.getByTestId('insights-not').locator('li')).toHaveCount(4)
  await expect(page.getByTestId('runtime-models')).toContainText('SigLIP 2')
  await expect(page.getByTestId('board-prompt-test')).toContainText('weighted F1 0.917')
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '04-models')

  // 6. dettagli per esperti, chiusi di default
  await expect(page.getByTestId('expert-scores')).toHaveCount(0)
  await page.getByTestId('toggle-details').click()
  const final = page.locator('[data-testid="model-card"][data-model="apertus-8b-vadapter-lora-heads-final"]')
  await expect(final.getByTestId('card-macro')).toHaveText('0.821')
  await page.getByTestId('set-select').selectOption('coverage')
  await expect(page.getByTestId('model-card')).toHaveCount(4)   // solo i modelli con un punteggio su questo set
  await expect(page.locator('[data-testid="model-card"][data-model="apertus-8b-lora-heads-p11"]').getByTestId('card-macro')).toHaveText('0.781')
  await expect(page.getByTestId('dimension-chart')).toBeVisible()
  await expect(page.getByTestId('per-class')).toBeVisible()
  await expect(page.getByTestId('confusion')).toBeVisible()
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '05-models-details')

  // 7. i vecchi indirizzi portano alla pagina iniziale
  await page.goto('/image-classification')
  await expect(page.getByRole('heading', { name: 'Classify a product image' })).toBeVisible()
})
