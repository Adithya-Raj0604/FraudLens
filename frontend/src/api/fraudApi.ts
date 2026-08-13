import type { TransactionInput, SSEEvent, ExplainResponse } from "../types"

// Dev: talk to the local uvicorn server. Production build: empty string = same
// origin, since CloudFront serves the React app and the API on one domain (no CORS).
const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000"

/** Pings /health to confirm the backend is up and the model is loaded. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { method: "GET" })
    if (!res.ok) return false
    const data = await res.json()
    return data.status === "ok" && data.model_loaded === true
  } catch {
    return false
  }
}

/** Gateway/transient statuses worth retrying — the Lambda is cold-starting and
 *  CloudFront gave up before the agent finished initialising (504), or the origin
 *  was briefly unavailable (502/503). A retry usually lands on a now-warm container. */
const RETRYABLE_STATUS = new Set([502, 503, 504])
export const MAX_RETRIES = 3

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** Signals the caller about retry attempts so the UI can explain the wait. */
export interface InvestigateHooks {
  /** Called before attempt N (0-indexed). attempt 0 = first try. */
  onAttempt?: (attempt: number) => void
  /** Called after a retryable failure, before the next attempt. */
  onRetry?: (attempt: number, reason: string) => void
}

/**
 * Streams the /investigate SSE endpoint via fetch + ReadableStream.
 * EventSource doesn't support POST, so we use fetch manually.
 *
 * On a cold start the agent import (torch + sentence-transformers + faiss +
 * langgraph) can exceed CloudFront's 60s origin timeout, yielding a 504 before
 * any bytes stream. We retry the whole request a few times with backoff — safe
 * because a gateway timeout means the response body never started.
 */
export async function* streamInvestigation(
  tx: TransactionInput,
  hooks: InvestigateHooks = {}
): AsyncGenerator<SSEEvent> {
  let res: Response | undefined

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    hooks.onAttempt?.(attempt)
    try {
      res = await fetch(`${BASE}/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(tx),
      })
    } catch (e) {
      // Network-level failure (connection dropped, DNS, TLS) — also transient
      // during a cold start. Retry until we're out of attempts.
      if (attempt < MAX_RETRIES) {
        hooks.onRetry?.(attempt + 1, e instanceof Error ? e.message : String(e))
        await sleep(2000 * (attempt + 1))
        continue
      }
      throw e
    }

    if (res.ok) break

    if (RETRYABLE_STATUS.has(res.status) && attempt < MAX_RETRIES) {
      hooks.onRetry?.(attempt + 1, `${res.status} ${res.statusText}`)
      await sleep(2000 * (attempt + 1))
      continue
    }

    throw new Error(`Backend error: ${res.status} ${res.statusText}`)
  }

  if (!res || !res.ok) {
    throw new Error(
      `Backend error: ${res?.status ?? "no response"} ${res?.statusText ?? ""}`.trim()
    )
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const raw = line.slice(6).trim()
        if (!raw) continue
        try {
          yield JSON.parse(raw) as SSEEvent
        } catch {
          // malformed chunk — skip
        }
      }
    }
  }
}

export async function fetchExplanation(tx: TransactionInput): Promise<ExplainResponse> {
  const res = await fetch(`${BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(tx),
  })
  if (!res.ok) throw new Error(`Explain error: ${res.status}`)
  return res.json()
}
