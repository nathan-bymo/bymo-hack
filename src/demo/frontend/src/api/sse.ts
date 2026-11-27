import { api } from './client'
import type { ApiErrorBody, BatchInfo, JobSnapshot, PredictionResult } from './types'

export interface JobHandlers {
  onResult?: (r: PredictionResult) => void
  onProgress?: (done: number, total: number) => void
  onError?: (imageId: string | null, error: ApiErrorBody) => void
  onDone?: (status: string, batch: BatchInfo | null) => void
}

/** SSE con ripiego a polling di GET /api/jobs/{id} ogni secondo se il flusso si chiude (allegato C, C.5). */
export function subscribeJob(jobId: string, total: number, handlers: JobHandlers): () => void {
  let finished = false
  let closed = false
  const source = new EventSource(`/api/jobs/${jobId}/events`)
  let poll: number | undefined

  const finish = (status: string, batch: BatchInfo | null) => {
    if (finished) return
    finished = true
    source.close()
    if (poll) window.clearInterval(poll)
    handlers.onDone?.(status, batch)
  }
  const parse = (e: MessageEvent) => JSON.parse(e.data)
  source.addEventListener('result', (e) => handlers.onResult?.(parse(e as MessageEvent)))
  source.addEventListener('progress', (e) => {
    const d = parse(e as MessageEvent)
    handlers.onProgress?.(d.done, d.total)
  })
  source.addEventListener('error', (e) => {
    const me = e as MessageEvent
    if (me.data) {
      const d = JSON.parse(me.data)
      handlers.onError?.(d.image_id, d.error)
    }
  })
  source.addEventListener('done', (e) => {
    const d = parse(e as MessageEvent)
    finish(d.status, d.batch)
  })
  source.onerror = () => {
    if (finished || closed || poll) return
    source.close()
    poll = window.setInterval(async () => {
      try {
        const job = await api<JobSnapshot>(`/jobs/${jobId}`)
        handlers.onProgress?.(job.done, total)
        if (job.status !== 'running') {
          job.results.forEach((r) => handlers.onResult?.(r))
          job.errors.forEach((x) => handlers.onError?.(x.image_id, x.error))
          finish(job.status, job.batch)
        }
      } catch {
        /* backend momentaneamente irraggiungibile: riprova al prossimo giro */
      }
    }, 1000)
  }
  return () => {
    closed = true
    source.close()
    if (poll) window.clearInterval(poll)
  }
}
