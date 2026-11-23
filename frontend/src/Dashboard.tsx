import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUpFromLine, Calculator, Clock3, Cpu, MessageSquare, Plus, Zap,
} from 'lucide-react'
import { conversationTitle, type Conversation, type Usage, type Workspace } from './api'
import { ContextWindow, tokens } from './ContextWindow'

const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })
const toolNames: Record<string, string> = {
  calculator: 'Calculator',
  get_current_datetime: 'Date & time',
}

export function Dashboard({ workspace, rows, usage, busy, loading, error, create, select }: {
  workspace?: Workspace; rows: Conversation[]; usage?: Usage; loading: boolean; error: string;
  busy: boolean; create: () => void; select: (id: string) => void;
}) {
  return (
    <div className="dashboard">
      <section className="hero" aria-labelledby="dashboard-title">
        <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><div className="orbit-core"><Activity size={44} /></div></div>
        <div className="hero-content">
          <div className="hero-status"><span className={workspace?.model ? 'status-dot ready' : 'status-dot'} />{workspace?.model ? 'Model configured' : 'Model unavailable'}</div>
          <h1 id="dashboard-title">Your AI workspace</h1>
          <p>Pick up a conversation, start a new one, or see how much context your last reply used.</p>
          <button className="primary" onClick={create} disabled={busy}><Plus size={17} /> New chat <ArrowRight size={16} /></button>
        </div>
      </section>

      <div className="section-heading"><h2>Workspace overview</h2><span>Last reply in this session</span></div>
      <div className="stats-grid">
        <div className="card stat-card indigo"><div><span className="heading-icon violet"><Cpu size={17} /></span><h3>Configured model</h3></div><p className="model-value">{workspace?.model ?? 'Unavailable'}</p></div>
        <div className="card stat-card teal"><div><span className="heading-icon mint"><ArrowDownToLine size={17} /></span><h3>Total input tokens</h3></div><p>{tokens(usage?.input_tokens)}</p></div>
        <div className="card stat-card blue"><div><span className="heading-icon sky"><ArrowUpFromLine size={17} /></span><h3>Total output tokens</h3></div><p>{tokens(usage?.output_tokens)}</p></div>
      </div>

      <div className="dashboard-grid">
        <section className="card activity-card">
          <div className="card-heading"><span className="heading-icon mint"><MessageSquare size={17} /></span><h2>Recent conversations</h2></div>
          {rows.length > 0 ? <ul className="activity-list">
            {rows.slice(0, 5).map(row => (
              <li key={row.id}><button onClick={() => select(row.id)} disabled={busy}>
                <span className="activity-icon"><MessageSquare size={15} /></span>
                <span className="activity-title"><strong className="truncate">{conversationTitle(row)}</strong><small>{dateFormat.format(new Date(row.updated_at))}</small></span>
                <ArrowRight size={17} className="activity-arrow" />
              </button></li>
            ))}
          </ul> : <div className="activity-empty">
            <MessageSquare size={25} />
            <strong>{loading ? 'Loading conversations…' : error ? 'Conversation history unavailable' : 'No conversations yet'}</strong>
            {!loading && !error && <button className="text-button" onClick={create} disabled={busy}>Start a conversation <ArrowRight size={15} /></button>}
          </div>}
        </section>
        <div className="dashboard-rail">
          <ContextWindow usage={usage} limit={workspace?.input_token_limit} />
          <section className="card tools-card">
            <div className="card-heading"><span className="heading-icon sky"><Zap size={17} /></span><h2>Registered tools</h2></div>
            <div className="tool-list">
              {workspace?.tools.length ? workspace.tools.map(tool => <span key={tool}>{tool === 'calculator' ? <Calculator size={14} /> : tool === 'get_current_datetime' ? <Clock3 size={14} /> : <Zap size={14} />}{toolNames[tool] ?? tool}</span>) : <p className="muted">Tool information unavailable</p>}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
