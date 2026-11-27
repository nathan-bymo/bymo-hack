import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { defineConfig } from '@playwright/test'

// Fixture solo per i test: immagini sintetiche e registrazioni segnaposto in una cartella temporanea.
const root = process.env.E2E_ROOT ?? mkdtempSync(join(tmpdir(), 'demo-e2e-'))
process.env.E2E_ROOT = root
const repo = resolve(import.meta.dirname, '../../..')
if (!process.env.E2E_FIXTURE_DONE) {
  execFileSync(join(repo, '.venv/bin/python'), [join(repo, 'src/demo/backend/tests/make_e2e_fixture.py'), root], { stdio: 'inherit' })
  process.env.E2E_FIXTURE_DONE = '1'
}
const PORT = 8110

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  outputDir: 'test-results',
  use: { baseURL: `http://127.0.0.1:${PORT}`, channel: 'chrome', viewport: { width: 1440, height: 900 }, screenshot: 'off' },
  webServer: {
    command: `${join(repo, '.venv/bin/python')} -m uvicorn demo_api.main:app --port ${PORT}`,
    cwd: join(repo, 'src/demo/backend'),
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    env: { DEMO_DATA_DIR: join(root, 'data'), LLM_API_KEY: '' },
    timeout: 60_000,
  },
})
