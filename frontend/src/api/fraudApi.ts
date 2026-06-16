import type { TransactionInput, SSEEvent, ExplainResponse } from "../types"

const BASE = "http://localhost:8000"

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

/**
 * Streams the /investigate SSE endpoint via fetch + ReadableStream.
 * EventSource doesn't support POST, so we use fetch manually.
 */
export async function* streamInvestigation(
  tx: TransactionInput
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${BASE}/investigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(tx),
  })

  if (!res.ok) {
    throw new Error(`Backend error: ${res.status} ${res.statusText}`)
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
