import { expect, test } from '@playwright/test'
import { join } from 'node:path'

const files = (...names: string[]) => names.map((n) => join(process.env.E2E_ROOT!, 'files', n))
const shot = (page: import('@playwright/test').Page, name: string) => page.screenshot({ path: `e2e/screenshots/${name}.png`, fullPage: true })

const MODEL_A = 'apertus-8b-lora-heads-p11'
const MODEL_B = 'apertus-8b-fullft-heads'

test('percorso completo della demo', async ({ page }) => {
  const uploads: string[] = []
  page.on('request', (r) => { if (r.method() === 'POST' && r.url().endsWith('/api/images')) uploads.push(r.url()) })

  // 1. pagina prodotto e badge
  await page.goto('/product/DEMO-0001')
  await expect(page.getByTestId('demo-badge')).toHaveText('Team Bymo, Hack Apertus × Galaxus')
  await shot(page, '01-product')

  // 2. Enrich Product ▾ → Image classification (AI)
  await page.getByTestId('menu-enrich-product').click()
  await page.getByTestId('menu-image-classification').click()
  await expect(page).toHaveURL(/\/image-classification\?session=ses_/)
  await expect(page.getByTestId('demo-badge')).toBeVisible()

  // 3. upload di 3 immagini sintetiche + un file non valido
  await page.getByTestId('file-input').setInputFiles(files('synthetic-1.jpg', 'synthetic-2.jpg', 'synthetic-3.jpg', 'bad.png'))
  await expect(page.getByTestId('result-row')).toHaveCount(3)
  await expect(page.getByTestId('rejected-list')).toContainText('bad.png')
  await expect(page.getByTestId('rejected-list')).toContainText('UNSUPPORTED_MEDIA_TYPE')
  await shot(page, '02-uploaded')

  // 4. modello A (registrato), Run classification, 6 celle per riga
  await page.getByTestId('model-select').selectOption(MODEL_A)
  await page.getByTestId('run-classification').click()
  await expect(page.getByTestId('status-pill').filter({ hasText: 'AI suggested' })).toHaveCount(3, { timeout: 30_000 })
  for (const row of await page.getByTestId('result-row').all()) {
    await expect(row.locator('[data-testid^="cell-"]')).toHaveCount(6)
    await expect(row.getByTestId('mode-badge')).toContainText('Recorded')
  }
  await expect(page.getByTestId('throughput')).toContainText('target ≥ 2 img/s')
  await shot(page, '03-model-a')

  // 5. cambio modello senza nuovo upload
  const uploadsBefore = uploads.length
  await page.getByTestId('model-select').selectOption(MODEL_B)
  await expect(page.getByTestId('status-pill').filter({ hasText: 'AI suggested' })).toHaveCount(0)
  await page.getByTestId('run-classification').click()
  await expect(page.getByTestId('status-pill').filter({ hasText: 'AI suggested' })).toHaveCount(3, { timeout: 30_000 })
  expect(uploads.length).toBe(uploadsBefore)
  await shot(page, '04-model-b')

  // ordinamento per confidenza minima
  await page.getByTestId('sort-minconf').click()
  await shot(page, '05-sorted')

  // 6. confronto: disaccordi evidenziati (la Perspective della prima immagine differisce tra i due modelli)
  await page.getByRole('link', { name: 'Compare models' }).first().click()
  await expect(page.getByTestId('comparison-grid')).toBeVisible()
  await expect(page.getByTestId('compare-row')).toHaveCount(3)
  await expect(page.locator('[data-testid="compare-cell"][data-disagree="1"]').first()).toBeVisible()
  await expect(page.getByTestId('disagreement-count').first()).toHaveText('1/6')
  await shot(page, '05b-compare')
  await page.getByTestId('only-disagreements').check()
  await expect(page.getByTestId('compare-row')).toHaveCount(1)
  await page.getByRole('link', { name: 'Queue' }).first().click()
  await expect(page.getByTestId('result-row')).toHaveCount(3)

  // 7. dettaglio: cambia Perspective, Save → Corrected nella coda
  await page.getByTestId('result-row').first().click()
  await expect(page.getByTestId('image-type-panel')).toBeVisible()
  await expect(page.getByTestId('demo-badge')).toBeVisible()
  await shot(page, '06-detail')
  await page.getByTestId('dd-perspective').selectOption('top_down')
  await expect(page.getByTestId('edited-marker')).toContainText('Perspective')
  await page.getByTestId('save').click()
  await expect(page.getByTestId('toast').first()).toContainText('corrected')
  await shot(page, '07-detail-saved')
  await page.getByRole('link', { name: '← Queue' }).click()
  await expect(page.getByTestId('status-pill').filter({ hasText: 'Corrected' })).toHaveCount(1)

  // 7b. anteprima storefront: regola illustrativa, la correzione umana pesa più della predizione
  await page.getByRole('link', { name: 'Shop preview' }).click()
  await expect(page.getByTestId('rule-label')).toContainText('Illustrative rule (not the Galaxus rule)')
  await expect(page.getByTestId('gallery-item')).toHaveCount(3)
  await expect(page.locator('[data-testid="gallery-item"][data-position="1"]')).toBeVisible()
  await expect(page.getByTestId('why-main')).toContainText('Rule "whole product"')
  await shot(page, '07b-storefront')
  await page.getByRole('link', { name: '← Queue' }).click()
  await page.getByRole('link', { name: 'Scores per class' }).first().click()

  // 8. punteggi per classe: solo modelli con un punteggio sul set, nessuna scritta "invalid"
  await page.getByRole('link', { name: 'Scores per class' }).first().click()
  await expect(page.getByTestId('coverage-banner')).toContainText('Kaggle leaderboard')
  await expect(page.getByTestId('model-card')).toHaveCount(4)
  const p11 = page.locator('[data-testid="model-card"][data-model="apertus-8b-lora-heads-p11"]')
  await expect(p11.getByTestId('card-macro')).toHaveText('0.781')
  await expect(page.getByTestId('dimension-chart')).toBeVisible()
  await expect(page.getByTestId('per-class')).toBeVisible()
  await expect(page.getByTestId('confusion')).toBeVisible()
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await page.getByTestId('set-select').selectOption('prompt-test')
  await expect(page.locator('[data-testid="model-card"][data-model="apertus-8b-vadapter-lora-heads-final"]').getByTestId('card-macro')).toHaveText('0.821')
  await shot(page, '08-performance')

  // 9. app di presentazione, pagina Models: classifica Kaggle, velocità e costo, test interno; nessuna scritta "invalid"
  await page.goto('/models')
  const kaggle = page.getByTestId('board-kaggle_dev')
  await expect(kaggle.getByTestId('board-row')).toHaveCount(6)   // 3 modelli del team + 3 riferimenti
  await expect(kaggle.getByTestId('board-row').first()).toContainText('0.917')
  await expect(kaggle).toContainText('0.832')
  await expect(page.getByTestId('board-prompt-test')).toContainText('0.821')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-fullft-heads"]')).toContainText('$7.27')
  await expect(page.locator('[data-testid="models-row"][data-model="apertus-8b-lora-heads-p11"]')).toContainText('22 GB measured')
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '09-models')

  // 10. Classify: una foto → sei risposte, due modelli registrati a confronto
  await page.goto('/image-classification')
  await expect(page).toHaveURL(/session=ses_/)
  const session = new URL(page.url()).searchParams.get('session')
  await page.goto(`/?session=${session}`)
  await page.getByTestId('try-thumbs').locator('button').first().click()
  await page.getByTestId('try-classify').click()
  await expect(page.getByTestId('try-results')).toBeVisible({ timeout: 30_000 })
  for (const d of ['subject', 'background', 'perspective', 'individuals', 'text_classification', 'text_language'])
    await expect(page.getByTestId(`try-dim-${d}`)).toBeVisible()
  await expect(page.getByTestId('try-dim-perspective').getByTestId('try-disagree')).toBeVisible()
  await expect(page.getByTestId('try-error')).toHaveCount(0)
  await expect(page.getByText(/invalid/i)).toHaveCount(0)
  await shot(page, '10-classify')
})
