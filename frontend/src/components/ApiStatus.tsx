import { useEffect, useState } from "react"
import { checkHealth } from "../api/fraudApi"

type State = "checking" | "online" | "offline"

export default function ApiStatus() {
  const [state, setState] = useState<State>("checking")

  useEffect(() => {
    let active = true

    const ping = async () => {
      const ok = await checkHealth()
      if (active) setState(ok ? "online" : "offline")
    }

    ping()
    const id = setInterval(ping, 5000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  const config = {
    checking: { dot: "bg-slate-400", text: "Checking API…", pulse: true },
    online: { dot: "bg-accent", text: "API Connected", pulse: true },
    offline: { dot: "bg-danger", text: "API Offline", pulse: false },
  }[state]

  return (
    <div
      data-testid="api-status"
      data-status={state}
      className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 backdrop-blur-md px-3 py-1.5"
      title={
        state === "offline"
          ? "Backend not reachable on http://localhost:8000 — start uvicorn"
          : "Backend health check on /health"
      }
    >
      <span className="relative flex h-2 w-2">
        {config.pulse && (
          <span
            className={`absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping ${config.dot}`}
          />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${config.dot}`} />
      </span>
      <span className="font-mono text-xs text-slate-300">{config.text}</span>
    </div>
  )
}
