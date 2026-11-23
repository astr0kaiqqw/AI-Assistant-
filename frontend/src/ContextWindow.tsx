import { ChevronDown, Layers3 } from 'lucide-react'
import type { Usage } from './api'

const formatter = new Intl.NumberFormat()
const percentage = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 })
export const tokens = (value?: number | null) => value == null ? 'Usage unavailable' : formatter.format(value)
function percent(fraction: number) {
  if (fraction > 0 && fraction < .001) return '<0.1%'
  if (fraction < 1 && fraction > .999) return '>99.9%'
  return percentage.format(fraction)
}

export function ContextWindow({ usage, limit }: { usage?: Usage; limit?: number | null }) {
  const peak = usage?.peak_input_tokens
  const max = usage?.input_token_limit ?? limit
  const fraction = peak != null && max != null && max > 0 ? peak / max : null
  const filled = Math.min(fraction ?? 0, 1)

  return (
    <details className="card context-card" open>
      <summary className="card-heading">
        <span className="heading-icon violet"><Layers3 size={17} /></span>
        <span>Context Window</span><span className="context-scope">Last reply</span><ChevronDown size={17} className="context-chevron" />
      </summary>
      <div className="context-body">
        <p className="context-value">{tokens(peak)}</p>
        <p className="muted">{max == null ? 'Limit unknown' : `of ${tokens(max)} input tokens · peak per model call`}</p>
        {fraction != null && max != null && peak != null && <>
          <div className="context-meter" role="meter" aria-label="Peak request context usage"
            aria-valuemin={0} aria-valuemax={max} aria-valuenow={Math.min(peak, max)}
            aria-valuetext={`${tokens(peak)} of ${tokens(max)} input tokens`}>
            <span style={{ width: `${filled * 100}%` }} />
            <span style={{ width: `${100 - filled * 100}%` }} />
          </div>
          <div className="meter-caption"><span>{percent(fraction)} used</span><span>{percent(1 - filled)} available</span></div>
        </>}
        <dl className="token-pair">
          <div><dt>Input tokens</dt><dd>{tokens(usage?.input_tokens)}</dd></div>
          <div><dt>Output tokens</dt><dd>{tokens(usage?.output_tokens)}</dd></div>
        </dl>
        <p className="context-note">Totals include all model calls for the last reply, including tool rounds.</p>
      </div>
    </details>
  )
}
