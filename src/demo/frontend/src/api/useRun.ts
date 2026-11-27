import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './client'
import { subscribeJob } from './sse'
import type { ApiErrorBody, BatchInfo, JobSnapshot, PredictResponse } from './types'

export interface RunState {
  running: boolean
  jobId: string | null
  modelId: string | null
  done: number
  total: number
  batch: BatchInfo | null
  errors: Record<string, ApiErrorBody>
  jobError: ApiErrorBody | null
  status: string | null
}
const IDLE: RunState = { running: false, jobId: null, modelId: null, done: 0, total: 0, batch: null, errors: {}, jobError: null, status: null }

/** Lancia POST /api/predict e segue il job via SSE; aggiorna le query di predizioni mentre arrivano i risultati. */
export function useRun(sessionId: string | null) {
  const qc = useQueryClient()
  const [state, setState] = useState<RunState>(IDLE)
  const stop = useRef<(() => void) | null>(null)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => { stop.current?.(); window.clearTimeout(timer.current) }, [])

  const refresh = useCallback(() => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      qc.invalidateQueries({ queryKey: ['predictions'] })
      qc.invalidateQueries({ queryKey: ['image'] })
    }, 150)
  }, [qc])

  const run = useCallback(async (modelId: string, imageIds: string[], mode: 'live' | 'recorded', useContext: boolean) => {
    stop.current?.()
    setState({ ...IDLE, running: true, modelId })
    try {
      const res = await api<PredictResponse>('/predict', {
        method: 'POST', json: { model_id: modelId, image_ids: imageIds, use_context: useContext, mode },
      })
      setState((s) => ({ ...s, jobId: res.job_id, total: res.total, done: res.cached }))
      if (res.queued === 0) {
        const job = await api<JobSnapshot>(`/jobs/${res.job_id}`)
        setState((s) => ({ ...s, running: false, done: job.done, batch: job.batch, status: job.status }))
        refresh()
        return res
      }
      stop.current = subscribeJob(res.job_id, res.total, {
        onProgress: (done, total) => setState((s) => ({ ...s, done, total })),
        onResult: refresh,
        onError: (imageId, error) =>
          setState((s) => (imageId ? { ...s, errors: { ...s.errors, [imageId]: error } } : { ...s, jobError: error })),
        onDone: (status, batch) => {
          setState((s) => ({ ...s, running: false, batch, status }))
          qc.invalidateQueries({ queryKey: ['predictions'] })
          qc.invalidateQueries({ queryKey: ['images'] })
          qc.invalidateQueries({ queryKey: ['image'] })
        },
      })
      return res
    } catch (e) {
      setState((s) => ({ ...s, running: false, jobError: { code: (e as { code?: string }).code ?? 'ERROR', message: (e as Error).message } }))
      throw e
    }
  }, [qc, refresh])

  const reset = useCallback(() => setState(IDLE), [])
  void sessionId
  return { state, run, reset }
}
