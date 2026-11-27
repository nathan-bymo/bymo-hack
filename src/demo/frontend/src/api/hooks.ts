import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from './client'
import type {
  Health, ImageItem, ModelsResponse, PredictionResult, Session, Settings, Taxonomy,
} from './types'

export const useHealth = () =>
  useQuery({ queryKey: ['health'], queryFn: () => api<Health>('/health'), refetchInterval: 5000, retry: false })
export const useTaxonomy = () => useQuery({ queryKey: ['taxonomy'], queryFn: () => api<Taxonomy>('/taxonomy'), staleTime: Infinity })
export const useModels = () => useQuery({ queryKey: ['models'], queryFn: () => api<ModelsResponse>('/models'), refetchInterval: 10000 })
export const useSettings = () => useQuery({ queryKey: ['settings'], queryFn: () => api<Settings>('/settings') })
export const useImages = (sessionId: string | null) =>
  useQuery({
    queryKey: ['images', sessionId],
    queryFn: () => api<{ images: ImageItem[] }>(`/images?session_id=${sessionId}`).then((r) => r.images),
    enabled: !!sessionId,
  })
export const useImage = (imageId: string | undefined) =>
  useQuery({ queryKey: ['image', imageId], queryFn: () => api<ImageItem>(`/images/${imageId}`), enabled: !!imageId })
export const usePredictions = (sessionId: string | null, modelId?: string | null) =>
  useQuery({
    queryKey: ['predictions', sessionId, modelId ?? 'all'],
    queryFn: () =>
      api<{ predictions: PredictionResult[] }>(
        `/predictions?session_id=${sessionId}${modelId ? `&model_id=${modelId}` : ''}`,
      ).then((r) => r.predictions),
    enabled: !!sessionId,
  })

const STORAGE_KEY = 'demo.session'

/** Sessione corrente: ?session=… nell'URL, altrimenti l'ultima usata, altrimenti se ne crea una. */
export function useSessionId(): { sessionId: string | null; error: unknown } {
  const [params, setParams] = useSearchParams()
  const fromUrl = params.get('session')
  const [error, setError] = useState<unknown>(null)
  const qc = useQueryClient()
  useEffect(() => {
    if (fromUrl) {
      localStorage.setItem(STORAGE_KEY, fromUrl)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const stored = localStorage.getItem(STORAGE_KEY)
        let id: string | null = null
        if (stored) {
          try {
            id = (await api<Session>(`/sessions/${stored}`)).session_id
          } catch {
            id = null
          }
        }
        if (!id) {
          const name = `Demo ${new Date().toLocaleDateString('en-GB')}`
          id = (await api<Session>('/sessions', { method: 'POST', json: { name } })).session_id
          localStorage.setItem(STORAGE_KEY, id)
        }
        if (!cancelled) {
          setParams((p) => { p.set('session', id!); return p }, { replace: true })
          qc.invalidateQueries({ queryKey: ['images'] })
        }
      } catch (e) {
        if (!cancelled) setError(e)
      }
    })()
    return () => { cancelled = true }
  }, [fromUrl, setParams, qc])
  return { sessionId: fromUrl, error }
}

export const useSession = (sessionId: string | null) =>
  useQuery({ queryKey: ['session', sessionId], queryFn: () => api<Session>(`/sessions/${sessionId}`), enabled: !!sessionId })
