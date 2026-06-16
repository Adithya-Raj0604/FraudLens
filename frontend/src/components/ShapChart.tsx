import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts"
import type { SHAPFeature } from "../types"

interface Props {
  features: SHAPFeature[]
}

const FRAUD_RED = "#EF4444"
const SAFE_GREEN = "#22C55E"

export default function ShapChart({ features }: Props) {
  // Top 8 by absolute SHAP, already sorted by backend
  const data = features.slice(0, 8).map(f => ({
    label: f.label,
    value: f.shap_value,
    direction: f.direction,
  }))

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-6 space-y-4">
      <div className="space-y-0.5">
        <h2 className="font-mono text-base font-semibold text-slate-100 tracking-tight">
          SHAP Feature Attribution
        </h2>
        <p className="text-xs text-slate-400">
          Red bars increase fraud probability · Green bars decrease it
        </p>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
        >
          <XAxis
            type="number"
            tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "Fira Code" }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            tickFormatter={v => v.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={148}
            tick={{ fill: "#CBD5E1", fontSize: 11, fontFamily: "Fira Sans" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "#0F172A",
              border: "1px solid rgba(255,255,255,0.10)",
              borderRadius: 10,
              fontSize: 12,
              fontFamily: "Fira Code",
              color: "#F8FAFC",
            }}
            formatter={(val) => [typeof val === "number" ? val.toFixed(5) : val, "SHAP value"]}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.direction === "increases_fraud" ? FRAUD_RED : SAFE_GREEN}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
