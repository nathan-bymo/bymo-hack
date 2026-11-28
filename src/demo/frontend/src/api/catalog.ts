import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { api } from './client'
import type {
  CatalogProductDetail, CatalogResponse, StorefrontFilters, StorefrontProductDetail, StorefrontResponse,
} from './types'

/** Tutto ciò che cambia quando l'admin pubblica o cambia una regola. */
function invalidateShop(qc: QueryClient) {
  for (const key of ['storefront', 'storefront-product', 'catalog', 'catalog-product']) {
    void qc.invalidateQueries({ queryKey: [key] })
  }
}

export const useStorefront = (q: string, category: string | null) =>
  useQuery({
    queryKey: ['storefront', q, category],
    queryFn: () => {
      const params = new URLSearchParams()
      if (q.trim()) params.set('q', q.trim())
      if (category) params.set('category', category)
      const query = params.toString()
      return api<StorefrontResponse>(`/storefront${query ? `?${query}` : ''}`)
    },
  })

export const useStorefrontProduct = (productId: string | undefined) =>
  useQuery({
    queryKey: ['storefront-product', productId],
    queryFn: () => api<StorefrontProductDetail>(`/storefront/products/${productId}`),
    enabled: !!productId,
  })

export const useCatalog = (q: string, status: string | null) =>
  useQuery({
    queryKey: ['catalog', q, status],
    queryFn: () => {
      const params = new URLSearchParams()
      if (q.trim()) params.set('q', q.trim())
      if (status) params.set('status', status)
      const query = params.toString()
      return api<CatalogResponse>(`/catalog${query ? `?${query}` : ''}`)
    },
  })

export const useCatalogProduct = (productId: string | undefined) =>
  useQuery({
    queryKey: ['catalog-product', productId],
    queryFn: () => api<CatalogProductDetail>(`/catalog/products/${productId}`),
    enabled: !!productId,
  })

export const useStorefrontFilters = () =>
  useQuery({
    queryKey: ['storefront-settings'],
    queryFn: () => api<StorefrontFilters>('/storefront/settings'),
  })

export const useSaveFilters = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (filters: StorefrontFilters) =>
      api<StorefrontFilters>('/storefront/settings', { method: 'PUT', json: filters }),
    onSuccess: (filters) => {
      qc.setQueryData(['storefront-settings'], filters)
      invalidateShop(qc)
    },
  })
}

export const useUpdateCatalogImage = (productId: string | undefined) => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ imageId, ...body }: {
      imageId: string
      published?: boolean
      is_cover?: boolean
      position?: number
      note?: string | null
    }) => api<CatalogProductDetail>(`/catalog/images/${imageId}`, { method: 'PUT', json: body }),
    onSuccess: (detail) => {
      if (productId) qc.setQueryData(['catalog-product', productId], detail)
      invalidateShop(qc)
    },
  })
}

export const useUpdateCatalogProduct = (productId: string | undefined) => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: Record<string, string | null>) =>
      api<CatalogProductDetail>(`/catalog/products/${productId}`, { method: 'PUT', json: body }),
    onSuccess: (detail) => {
      if (productId) qc.setQueryData(['catalog-product', productId], detail)
      invalidateShop(qc)
    },
  })
}

export interface CreatedProduct {
  product: { product_id: string; title: string }
  images: unknown[]
  created: boolean
}

export const useCreateProduct = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { title: string; brand?: string; category?: string; subcategory?: string; product_type?: string }) =>
      api<CreatedProduct>('/catalog/products', { method: 'POST', json: body }),
    onSuccess: () => invalidateShop(qc),
  })
}

export const useAttachImage = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ imageId, productId, isCover }: { imageId: string; productId: string; isCover: boolean }) =>
      api<CatalogProductDetail>(`/catalog/images/${imageId}`, {
        method: 'PUT',
        json: { product_id: productId, published: true, is_cover: isCover },
      }),
    onSuccess: () => invalidateShop(qc),
  })
}
