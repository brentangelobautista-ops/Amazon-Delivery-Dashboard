import { spawn } from 'node:child_process'

const apiHost = process.env.LOGITRACK_API_HOST || '127.0.0.1'
const apiPort = process.env.LOGITRACK_API_PORT || '8000'
const healthUrl = process.env.LOGITRACK_API_HEALTH_URL || `http://${apiHost}:${apiPort}/health`
const timeoutMs = 120_000

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))

async function waitForHealthyApi() {
  const startedAt = Date.now()
  console.log(`[LogiTrack web] Waiting for a healthy prediction API at ${healthUrl}...`)

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(healthUrl)
      const health = await response.json()
      if (response.ok && health.ready === true && typeof health.model === 'string') {
        console.log(`[LogiTrack web] Prediction API is ready (${health.model}). Starting Vite...`)
        return
      }
    } catch {
      // FastAPI may still be loading the joblib pipeline. Keep polling while its own logs stay visible.
    }
    await delay(750)
  }

  throw new Error(`Prediction API did not become ready within ${timeoutMs / 1000} seconds: ${healthUrl}`)
}

try {
  await waitForHealthyApi()
  const vite = spawn('vite --host 127.0.0.1', { shell: true, stdio: 'inherit' })
  vite.on('exit', (code) => process.exit(code ?? 1))
} catch (error) {
  console.error(`[LogiTrack web] ${error instanceof Error ? error.message : error}`)
  process.exit(1)
}
