import { Plus, Minus } from "lucide-react"

interface Props {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}

export default function NumberField({ label, value, onChange, min, max, step = 1 }: Props) {
  const clamp = (v: number) => {
    if (min !== undefined && v < min) return min
    if (max !== undefined && v > max) return max
    return v
  }

  const bump = (dir: 1 | -1) => onChange(clamp(Number((value + dir * step).toFixed(4))))

  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-0.5">{label}</label>
      <div className="flex items-stretch rounded-lg border border-white/10 bg-surface-2 overflow-hidden focus-within:border-accent/60 transition-colors duration-150">
        <button
          type="button"
          tabIndex={-1}
          onClick={() => bump(-1)}
          className="px-2 flex items-center justify-center text-slate-400 hover:text-accent hover:bg-white/5 cursor-pointer transition-colors duration-150 border-r border-white/10"
          aria-label={`Decrease ${label}`}
        >
          <Minus size={13} />
        </button>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(clamp(Number(e.target.value)))}
          className="w-full bg-transparent px-2 py-1.5 text-sm text-slate-100 text-center focus:outline-none"
        />
        <button
          type="button"
          tabIndex={-1}
          onClick={() => bump(1)}
          className="px-2 flex items-center justify-center text-slate-400 hover:text-accent hover:bg-white/5 cursor-pointer transition-colors duration-150 border-l border-white/10"
          aria-label={`Increase ${label}`}
        >
          <Plus size={13} />
        </button>
      </div>
    </div>
  )
}
