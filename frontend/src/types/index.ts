export interface TransactionInput {
  step: number
  type: "TRANSFER" | "CASH_OUT"
  amount: number
  oldbalanceOrg: number
  newbalanceOrig: number
  oldbalanceDest: number
  newbalanceDest: number
  velocity_cumcount: number
  velocity_1hr: number
  velocity_3hr: number
  velocity_24hr: number
}

export interface SSEToolCallEvent {
  type: "tool_call"
  name: string
  input: Record<string, unknown>
}

export interface SSEToolResultEvent {
  type: "tool_result"
  name: string
  content: string
}

export interface SSEReportEvent {
  type: "report"
  content: string
}

export interface SSEErrorEvent {
  type: "error"
  content: string
}

export type SSEEvent =
  | SSEToolCallEvent
  | SSEToolResultEvent
  | SSEReportEvent
  | SSEErrorEvent

export interface SHAPFeature {
  feature: string
  label: string
  shap_value: number
  raw_value: number
  direction: "increases_fraud" | "decreases_fraud"
}

export interface ExplainResponse {
  risk_score: number
  base_value: number
  features: SHAPFeature[]
  top_driver: string
  summary: string
}

export type InvestigationStatus = "idle" | "running" | "done" | "error"
