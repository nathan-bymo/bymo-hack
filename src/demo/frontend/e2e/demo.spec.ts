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
  await expect(page.getByTestId('demo-badge')).toHaveText('Demo — Team Bymo · Hack Apertus')
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
  await page.getByRole('link', { name: 'Storefront preview' }).click()
  await expect(page.getByTestId('rule-label')).toContainText('Illustrative rule (not the Galaxus rule)')
  await expect(page.getByTestId('gallery-item')).toHaveCount(3)
  await expect(page.locator('[data-testid="gallery-item"][data-position="1"]')).toBeVisible()
  await expect(page.getByTestId('why-main')).toContainText('Rule "whole product"')
  await shot(page, '07b-storefront')
  await page.getByRole('link', { name: '← Queue' }).click()
  await page.getByRole('link', { name: 'Performance' }).first().click()

  // 8. dashboard: banner coverage, linea 0.917, badge rosso del full FT, Kaggle dev solo dove c'è
  await page.getByRole('link', { name: 'Performance' }).first().click()
  await expect(page.getByTestId('coverage-banner')).toContainText('not to estimate the Kaggle leaderboard')
  await expect(page.getByTestId('model-card')).toHaveCount(5)
  const full = page.locator('[data-testid="model-card"][data-model="apertus-8b-fullft-heads"]')
  await expect(full.getByTestId('badge-seen_in_training')).toContainText('seen during training — invalid')
  await expect(full.getByTestId('card-macro')).toHaveCount(0)
  const p11 = page.locator('[data-testid="model-card"][data-model="apertus-8b-lora-heads-p11"]')
  await expect(p11.getByTestId('card-macro')).toHaveText('0.781')
  await expect(p11.getByTestId('badge-phase1_internal')).toBeVisible()
  await expect(p11.getByTestId('badge-reported_only')).toBeVisible()
  await expect(p11.getByTestId('card-kaggle')).toContainText('0.796')
  await expect(full.getByTestId('card-kaggle')).toHaveCount(0)
  await expect(page.getByText("Organisers' reference (Qwen 3.5 9B FT, dev)").first()).toBeVisible()
  await expect(page.getByTestId('reference-note')).toContainText('Different set (Kaggle dev)')
  await expect(page.getByTestId('confusion')).toBeVisible()
  await expect(page.getByTestId('calibration-na')).toContainText('n/a')
  const cost = page.getByTestId('cost-cell').filter({ hasText: 'per 100,000' })
  await expect(cost).toContainText('0.105 $')
  await expect(cost).toContainText('10.5 $ per 100,000')
  await expect(page.getByTestId('cost-invalid')).toContainText('estimate not valid')
  await expect(page.getByTestId('source-footnote').first()).toContainText('Source:')
  await shot(page, '08-performance')
})
