import type { TransactionInput } from "../types"

export type Severity = "legit" | "suspicious" | "fraud"

export interface Preset {
  id: string
  label: string
  description: string
  severity: Severity
  tx: TransactionInput
}

export const SEVERITY_COLOR: Record<Severity, string> = {
  legit: "#22C55E",
  suspicious: "#F59E0B",
  fraud: "#EF4444",
}

export const PRESETS: Preset[] = [
  {
    id: "legit",
    label: "Legitimate Transfer",
    description:
      "A normal transfer that retains most of the account balance with no velocity spikes. The model should score this LOW risk.",
    severity: "legit",
    tx: {
      step: 200,
      type: "TRANSFER",
      amount: 1500,
      oldbalanceOrg: 8400,
      newbalanceOrig: 6900,
      oldbalanceDest: 12000,
      newbalanceDest: 13500,
      velocity_cumcount: 1,
      velocity_1hr: 1,
      velocity_3hr: 1,
      velocity_24hr: 1,
    },
  },
  {
    id: "drain",
    label: "Account Drain",
    description:
      "Origin balance emptied to $0 into a previously empty account, with elevated velocity (8 txns/24hr). The realistic model excludes the balance-to-zero leakage features, so it scores this ~MEDIUM (amount-driven) rather than HIGH — yet the agent still ESCALATES on the velocity flag + full depletion. A live example of why the leakage fix matters.",
    severity: "fraud",
    tx: {
      step: 1,
      type: "TRANSFER",
      amount: 9000,
      oldbalanceOrg: 9000,
      newbalanceOrig: 0,
      oldbalanceDest: 0,
      newbalanceDest: 9000,
      velocity_cumcount: 7,
      velocity_1hr: 3,
      velocity_3hr: 5,
      velocity_24hr: 8,
    },
  },
  {
    id: "mule",
    label: "Mule Cash-Out",
    description:
      "Large CASH_OUT draining the account into a pass-through mule account with very high transaction velocity. Layering behaviour — expect HIGH risk.",
    severity: "fraud",
    tx: {
      step: 50,
      type: "CASH_OUT",
      amount: 64000,
      oldbalanceOrg: 64000,
      newbalanceOrig: 0,
      oldbalanceDest: 0,
      newbalanceDest: 0,
      velocity_cumcount: 12,
      velocity_1hr: 4,
      velocity_3hr: 8,
      velocity_24hr: 14,
    },
  },
  {
    id: "structuring",
    label: "Structuring",
    description:
      "Amount deliberately set just under the $10,000 CAD reporting threshold, repeated several times in a day (smurfing). Partial drain but suspicious amount and velocity — typically MEDIUM risk.",
    severity: "suspicious",
    tx: {
      step: 300,
      type: "TRANSFER",
      amount: 9500,
      oldbalanceOrg: 47500,
      newbalanceOrig: 38000,
      oldbalanceDest: 5000,
      newbalanceDest: 14500,
      velocity_cumcount: 5,
      velocity_1hr: 2,
      velocity_3hr: 3,
      velocity_24hr: 6,
    },
  },
]
