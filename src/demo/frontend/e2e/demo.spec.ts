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
  await page.getByTestId('try-classify').click()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })
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
  await page.getByTestId('try-classify').click()
  await expect(page.getByTestId('try-dim-subject')).toContainText('The whole product', { timeout: 30_000 })
  await expect(page.getByTestId('try-disagree')).toHaveCount(0)
  await shot(page, '03-classify-one-model')

  // 5. Results: classifica Kaggle con il gruppo di valutazione, domanda per domanda, velocità e costo
  await page.getByRole('link', { name: 'Results', exact: true }).click()
  const kaggle = page.getByTestId('board-kaggle_dev')
  await expect(kaggle.getByTestId('board-row')).toHaveCount(10)   // 7 punteggi del team + 3 riferimenti
  await expect(kaggle.getByTestId('board-row').first()).toContainText('0.917')
  await expect(kaggle).toContainText('0.832')
  await expect(page.getByTestId('assessment-group')).toContainText('Best team submission (0.847) is in the middle assessment group: above both Gemini references')
  await expect(kaggle.getByTestId('board-row').nth(1)).toContainText('0.847')
  await expect(page.getByTestId('pq-macro')).toContainText('0.821')
  await expect(page.getByTestId('pq-weighted')).toContainText('0.917')
  await expect(page.getByTestId('pq-perspective')).toContainText('0.755')
  await expect(page.getByTestId('pq-perspective')).toContainText('0.198')
  await expect(page.getByTestId('pq-macro')).toContainText('0.847')   // P13
  await expect(kaggle).toContainText('0.829')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-fullft-heads"]')).toContainText('$7.27')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-lora-heads-p11"]')).toContainText('22 GB measured')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-fullft-heads"]').getByTestId('hours-100k')).toHaveText('2.9 h')
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '04-results')

  // 6. dettagli per esperti, chiusi di default
  await expect(page.getByTestId('expert-scores')).toHaveCount(0)
  await page.getByTestId('toggle-details').click()
  const final = page.locator('[data-testid="model-card"][data-model="apertus-8b-vadapter-lora-heads-final"]')
  await expect(final.getByTestId('card-macro')).toHaveText('0.821')
  await page.getByTestId('set-select').selectOption('coverage')
  await expect(page.getByTestId('model-card')).toHaveCount(5)   // solo i modelli con un punteggio su questo set
  await expect(page.locator('[data-testid="model-card"][data-model="apertus-8b-lora-heads-p11"]').getByTestId('card-macro')).toHaveText('0.781')
  await expect(page.getByTestId('dimension-chart')).toBeVisible()
  await expect(page.getByTestId('per-class')).toBeVisible()
  await expect(page.getByTestId('confusion')).toBeVisible()
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '05-results-details')

  // 7. How it works: schema, passi di training, cosa ha contato, modelli a runtime
  await page.getByRole('link', { name: 'How it works', exact: true }).click()
  await expect(page.getByTestId('pipeline')).toContainText('SigLIP 2')
  await expect(page.getByTestId('walkthrough').locator('li')).toHaveCount(4)
  await expect(page.getByTestId('experiments').getByTestId('experiment')).toHaveCount(19)   // P0–P15, full fine-tune, modello finale, mixed
  await expect(page.getByTestId('experiments')).toContainText('P13')
  await expect(page.getByTestId('training-steps').locator('tbody tr')).toHaveCount(3)
  await expect(page.getByTestId('insights-mattered').locator('li')).toHaveCount(5)
  await expect(page.getByTestId('insights-not').locator('li')).toHaveCount(4)
  await expect(page.getByTestId('runtime-models')).toContainText('docTR')
  await shot(page, '06-how')
  await shot(page, '06b-how-frame', false)

  // 8. i vecchi indirizzi portano alle pagine nuove
  await page.goto('/models')
  await expect(page).toHaveURL(/\/results/)
  await page.goto('/image-classification')
  await expect(page.getByRole('heading', { name: 'Six answers for every product photo' })).toBeVisible()
})
