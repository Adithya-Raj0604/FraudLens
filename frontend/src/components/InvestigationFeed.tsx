import { useEffect, useRef } from "react"
import { CheckCircle, Loader, AlertCircle, ChevronRight } from "lucide-react"
import type { SSEEvent } from "../types"

const TOOL_LABELS: Record<string, string> = {
  tool_run_fraud_model: "Run Fraud Model",
  tool_explain_prediction: "Explain Prediction",
  tool_retrieve_regulations: "Retrieve Regulations",
  tool_check_account_velocity: "Check Account Velocity",
}

interface Props {
  events: SSEEvent[]
  status: "idle" | "running" | "done" | "error"
  errorMessage?: string
}

export default function InvestigationFeed({ events, status, errorMessage }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [events.length])

  if (status === "idle") return null

  // Pair tool_call with its matching tool_result
  const paired: Array<{
    call: SSEEvent & { type: "tool_call" }
    result?: SSEEvent & { type: "tool_result" }
  }> = []

  const resultMap = new Map<string, SSEEvent & { type: "tool_result" }>()
  for (const e of events) {
    if (e.type === "tool_result") resultMap.set(e.name, e)
  }
  for (const e of events) {
    if (e.type === "tool_call") {
      paired.push({ call: e, result: resultMap.get(e.name) })
    }
  }

  return (
    <div data-testid="investigation-feed" data-status={status} className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-6 space-y-3">
      <h2 className="font-mono text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
        <span>Live Investigation</span>
        {status === "running" && (
          <span className="inline-flex h-2 w-2 rounded-full bg-accent animate-pulse" />
        )}
      </h2>

      <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
        {paired.map(({ call, result }, i) => {
          const label = TOOL_LABELS[call.name] ?? call.name
          const done = result !== undefined

          return (
            <div
              key={i}
              className="rounded-xl border border-white/8 bg-surface-2/60 p-3 space-y-1.5"
            >
              {/* Tool call header */}
              <div className="flex items-center gap-2 text-sm">
                {done ? (
                  <CheckCircle size={14} className="text-accent shrink-0" />
                ) : (
                  <Loader size={14} className="text-slate-400 shrink-0 animate-spin" />
                )}
                <span className="font-mono font-medium text-slate-200">{label}</span>
                <ChevronRight size={12} className="text-slate-600 ml-auto shrink-0" />
              </div>

              {/* Tool result snippet */}
              {result && (
                <p className="text-xs text-slate-400 leading-relaxed line-clamp-2 pl-5">
                  {result.content.slice(0, 200)}
                  {result.content.length > 200 ? "…" : ""}
                </p>
              )}
            </div>
          )
        })}

        {status === "running" && paired.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Loader size={14} className="animate-spin" />
            <span>Agent is initialising…</span>
          </div>
        )}

        {status === "error" && (
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-danger">
              <AlertCircle size={14} className="shrink-0" />
              <span>Investigation failed.</span>
            </div>
            {errorMessage && (
              <pre className="text-xs text-slate-400 font-mono pl-5 whitespace-pre-wrap break-all">
                {errorMessage}
              </pre>
            )}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
