import { FileText } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

interface Props {
  content: string
}

export default function InvestigatorReport({ content }: Props) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-6 space-y-4">
      <h2 className="font-mono text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
        <FileText size={16} className="text-accent" />
        Investigation Report
      </h2>

      <div className="max-h-[32rem] overflow-y-auto prose-report">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => (
              <h1 className="font-mono text-sm font-bold text-slate-100 uppercase tracking-widest mb-3 mt-0 border-b border-white/10 pb-2">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="font-mono text-xs font-semibold text-accent uppercase tracking-wider mt-5 mb-2">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="font-sans text-sm font-semibold text-slate-200 mt-4 mb-1">
                {children}
              </h3>
            ),
            p: ({ children }) => (
              <p className="font-sans text-sm text-slate-300 leading-relaxed mb-3">
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="space-y-1 mb-3 pl-4">{children}</ul>
            ),
            ol: ({ children }) => (
              <ol className="space-y-1 mb-3 pl-4 list-decimal">{children}</ol>
            ),
            li: ({ children }) => (
              <li className="font-sans text-sm text-slate-300 leading-relaxed list-disc">
                {children}
              </li>
            ),
            strong: ({ children }) => (
              <strong className="font-semibold text-slate-100">{children}</strong>
            ),
            code: ({ children }) => (
              <code className="font-mono text-xs bg-white/8 text-accent px-1.5 py-0.5 rounded">
                {children}
              </code>
            ),
            hr: () => <hr className="border-white/10 my-4" />,
            table: ({ children }) => (
              <div className="my-3 overflow-x-auto rounded-lg border border-white/10">
                <table className="w-full border-collapse text-xs">{children}</table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-white/5">{children}</thead>
            ),
            th: ({ children }) => (
              <th className="border-b border-white/10 px-3 py-2 text-left font-mono font-semibold text-slate-200 whitespace-nowrap">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border-b border-white/5 px-3 py-2 font-sans text-slate-300 align-top">
                {children}
              </td>
            ),
            tr: ({ children }) => (
              <tr className="hover:bg-white/5 transition-colors duration-150">{children}</tr>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}
