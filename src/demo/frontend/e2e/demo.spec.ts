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
})
