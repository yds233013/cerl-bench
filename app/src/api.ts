import type { Demo, ReviewDetail, ReviewRow, Session } from './types'

export class ApiError extends Error {
  constructor(readonly status: number, message: string, readonly detail?: unknown) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the workspace API. Is `cerl serve` running?')
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new ApiError(response.status, body.error ?? response.statusText, body.detail)
  }
  return body as T
}

export const api = {
  demos: () => request<{ demos: Demo[] }>('/api/demos').then((r) => r.demos),
  createSession: (demoId: string) =>
    request<Session>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ demo_id: demoId }),
    }),
  session: (id: string) => request<Session>(`/api/sessions/${id}`),
  reset: (id: string) =>
    request<Session>(`/api/sessions/${id}/reset`, { method: 'POST' }),
  // `submissionId` is generated once per user gesture by the caller, so a
  // retried request, a double click or a re-render cannot dispatch twice.
  act: (id: string, action: Record<string, unknown>, submissionId: string) =>
    request<{ record: unknown; session: Session }>(`/api/sessions/${id}/actions`, {
      method: 'POST',
      body: JSON.stringify({ submission_id: submissionId, action }),
    }),
}

export const reviewApi = {
  episodes: () =>
    request<{ episodes: ReviewRow[] }>('/review/episodes').then((r) => r.episodes),
  detail: (id: string) => request<ReviewDetail>(`/review/episodes/${id}`),
}
