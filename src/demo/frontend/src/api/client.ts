import type { ApiErrorBody } from './types'

export class ApiClientError extends Error {
  code: string
  status: number
  details: Record<string, unknown>
  constructor(status: number, body: ApiErrorBody) {
    super(body.message)
    this.code = body.code
    this.status = status
    this.details = body.details ?? {}
  }
}

export async function api<T>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const { json, ...rest } = init ?? {}
  const headers = new Headers(rest.headers)
  let body = rest.body
  if (json !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(json)
  }
  let response: Response
  try {
    response = await fetch(`/api${path}`, { ...rest, headers, body })
  } catch {
    throw new ApiClientError(0, { code: 'NETWORK', message: 'Backend unreachable' })
  }
  if (!response.ok) {
    let parsed: { error?: ApiErrorBody } = {}
    try {
      parsed = await response.json()
    } catch {
      /* corpo non JSON */
    }
    throw new ApiClientError(response.status, parsed.error ?? { code: 'INTERNAL_ERROR', message: response.statusText })
  }
  return (await response.json()) as T
}

export function errorText(error: unknown): string {
  if (error instanceof ApiClientError) return error.code === 'NETWORK' ? error.message : `${error.message} (${error.code})`
  return error instanceof Error ? error.message : 'Unexpected error'
}
