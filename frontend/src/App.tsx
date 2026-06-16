import { useState, useCallback } from "react"
import { Shield } from "lucide-react"
import TransactionForm from "./components/TransactionForm"
import InvestigationFeed from "./components/InvestigationFeed"
import InvestigatorReport from "./components/InvestigatorReport"
import ShapChart from "./components/ShapChart"
import ApiStatus from "./components/ApiStatus"
import { streamInvestigation, fetchExplanation } from "./api/fraudApi"
import type { TransactionInput, SSEEvent, SHAPFeature, InvestigationStatus } from "./types"

export default function App() {
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [report, setReport] = useState<string | null>(null)
  const [shapFeatures, setShapFeatures] = useState<SHAPFeature[]>([])
  const [status, setStatus] = useState<InvestigationStatus>("idle")
  const [errorMessage, setErrorMessage] = useState<string | undefined>()

  const handleSubmit = useCallback(async (tx: TransactionInput) => {
    setEvents([])
    setReport(null)
    setShapFeatures([])
    setErrorMessage(undefined)
    setStatus("running")

    try {
      for await (const event of streamInvestigation(tx)) {
        if (event.type === "report") {
          setReport(event.content)
          setStatus("done")
          fetchExplanation(tx)
            .then(res => setShapFeatures(res.features))
            .catch(() => {})
        } else if (event.type === "error") {
          setErrorMessage(event.content)
          setStatus("error")
        } else {
          setEvents(prev => [...prev, event])
        }
      }
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : String(e))
      setStatus("error")
    }
  }, [])

  return (
    <div className="min-h-screen text-slate-100">
      <header className="border-b border-white/8 bg-surface/40 backdrop-blur-md sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-3">
          <Shield size={22} className="text-accent" />
          <span className="font-mono font-semibold text-lg tracking-tight">FraudLens</span>
          <span className="text-xs text-slate-500 ml-1">Agentic Investigation System</span>
          <div className="ml-auto">
            <ApiStatus />
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-5 grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6 items-start">
        <TransactionForm onSubmit={handleSubmit} disabled={status === "running"} />

        <div className="space-y-6">
          <InvestigationFeed events={events} status={status} errorMessage={errorMessage} />
          {report && <InvestigatorReport content={report} />}
          {shapFeatures.length > 0 && <ShapChart features={shapFeatures} />}
        </div>
      </main>
    </div>
  )
}
