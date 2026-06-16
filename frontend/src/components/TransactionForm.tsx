import { useState } from "react"
import { Search } from "lucide-react"
import type { TransactionInput } from "../types"
import { PRESETS, SEVERITY_COLOR, type Preset } from "../data/presets"
import NumberField from "./NumberField"

interface Props {
  onSubmit: (tx: TransactionInput) => void
  disabled: boolean
}

export default function TransactionForm({ onSubmit, disabled }: Props) {
  const [form, setForm] = useState<TransactionInput>(PRESETS[0].tx)
  const [activeId, setActiveId] = useState<string | null>(PRESETS[0].id)

  function set<K extends keyof TransactionInput>(key: K, value: TransactionInput[K]) {
    setForm(f => ({ ...f, [key]: value }))
    setActiveId(null) // manual edit clears the active preset highlight
  }

  function applyPreset(p: Preset) {
    setForm(p.tx)
    setActiveId(p.id)
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onSubmit(form)
  }

  const labelClass = "block text-xs font-medium text-slate-400 mb-1"
  const activePreset = PRESETS.find(p => p.id === activeId)

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-5 space-y-3.5"
    >
      <h2 className="font-mono text-base font-semibold text-slate-100 tracking-tight">
        Transaction Input
      </h2>

      {/* Preset examples */}
      <div className="space-y-1.5">
        <p className={labelClass}>Quick Examples</p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => {
            const active = p.id === activeId
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p)}
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium cursor-pointer transition-colors duration-150 ${
                  active
                    ? "border-white/30 bg-white/10 text-slate-100"
                    : "border-white/10 bg-surface-2/60 text-slate-400 hover:text-slate-200 hover:border-white/20"
                }`}
              >
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ background: SEVERITY_COLOR[p.severity] }}
                />
                {p.label}
              </button>
            )
          })}
        </div>
        {activePreset && (
          <p className="text-xs text-slate-500 leading-snug pt-0.5">
            {activePreset.description}
          </p>
        )}
      </div>

      <div className="h-px bg-white/8" />

      {/* Type + Step */}
      <div className="grid grid-cols-2 gap-2.5">
        <div>
          <label className={labelClass}>Type</label>
          <select
            value={form.type}
            onChange={e => set("type", e.target.value as "TRANSFER" | "CASH_OUT")}
            className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-slate-100 cursor-pointer focus:outline-none focus:border-accent/60 transition-colors duration-150"
          >
            <option value="TRANSFER">TRANSFER</option>
            <option value="CASH_OUT">CASH_OUT</option>
          </select>
        </div>
        <NumberField
          label="Step (hour 1–743)"
          value={form.step}
          onChange={v => set("step", v)}
          min={1}
          max={743}
        />
      </div>

      {/* Amount */}
      <NumberField
        label="Amount ($)"
        value={form.amount}
        onChange={v => set("amount", v)}
        min={0}
        step={100}
      />

      {/* Origin balances */}
      <div>
        <p className={labelClass + " mb-1.5"}>Origin Account</p>
        <div className="grid grid-cols-2 gap-2.5">
          <NumberField
            label="Balance before"
            value={form.oldbalanceOrg}
            onChange={v => set("oldbalanceOrg", v)}
            min={0}
            step={100}
          />
          <NumberField
            label="Balance after"
            value={form.newbalanceOrig}
            onChange={v => set("newbalanceOrig", v)}
            min={0}
            step={100}
          />
        </div>
      </div>

      {/* Destination balances */}
      <div>
        <p className={labelClass + " mb-1.5"}>Destination Account</p>
        <div className="grid grid-cols-2 gap-2.5">
          <NumberField
            label="Balance before"
            value={form.oldbalanceDest}
            onChange={v => set("oldbalanceDest", v)}
            min={0}
            step={100}
          />
          <NumberField
            label="Balance after"
            value={form.newbalanceDest}
            onChange={v => set("newbalanceDest", v)}
            min={0}
            step={100}
          />
        </div>
      </div>

      {/* Velocity */}
      <div>
        <p className={labelClass + " mb-1.5"}>Velocity</p>
        <div className="grid grid-cols-2 gap-2.5">
          <NumberField
            label="Cumulative count"
            value={form.velocity_cumcount}
            onChange={v => set("velocity_cumcount", v)}
            min={0}
          />
          <NumberField
            label="Same hour"
            value={form.velocity_1hr}
            onChange={v => set("velocity_1hr", v)}
            min={0}
          />
          <NumberField
            label="Same 3hr window"
            value={form.velocity_3hr}
            onChange={v => set("velocity_3hr", v)}
            min={0}
          />
          <NumberField
            label="Same day"
            value={form.velocity_24hr}
            onChange={v => set("velocity_24hr", v)}
            min={0}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={disabled}
        className="w-full flex items-center justify-center gap-2 bg-accent text-sm font-semibold text-slate-900 py-2 rounded-xl cursor-pointer hover:bg-green-400 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-accent/60"
      >
        <Search size={16} />
        {disabled ? "Investigating…" : "Investigate"}
      </button>
    </form>
  )
}
