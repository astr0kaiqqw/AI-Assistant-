import {
  FileText, LayoutDashboard, MessageSquare, PanelLeftClose, Plus, Settings2, Sparkles,
} from 'lucide-react'
import { conversationTitle, type Conversation } from './api'

export type View = 'dashboard' | 'chat' | 'settings'

type Props = {
  view: View; selected: string | null; rows: Conversation[]; more: boolean;
  loading: boolean; error: string; busy: boolean; collapsed: boolean;
  navigate: (view: View) => void; select: (id: string) => void;
  create: () => void; loadMore: () => void; toggle: () => void;
}

export function Sidebar({ view, selected, rows, more, loading, error, busy, collapsed,
  navigate, select, create, loadMore, toggle }: Props) {
  return (
    <aside className="sidebar" id="workspace-sidebar" inert={collapsed}>
      <div className="brand">
        <span className="brand-mark"><Sparkles size={20} /></span>
        <div><strong>AI Assistant</strong><small>Local workspace</small></div>
        <button className="icon-button sidebar-close" onClick={toggle} aria-label="Collapse sidebar" title="Collapse sidebar"><PanelLeftClose size={18} /></button>
      </div>
      <button className="new-chat" onClick={create} disabled={busy}><Plus size={17} />New chat</button>
      <nav aria-label="Workspace navigation">
        <p className="section-label">Workspace</p>
        <button className="nav-item" aria-current={view === 'dashboard' ? 'page' : undefined} onClick={() => navigate('dashboard')}><LayoutDashboard size={17} />Dashboard</button>
        <button className="nav-item" aria-current={view === 'chat' ? 'page' : undefined} onClick={() => navigate('chat')}><MessageSquare size={17} />AI Chat</button>
        <div className="nav-item future" aria-disabled="true"><FileText size={17} />Documents <span>Planned</span></div>
      </nav>
      <div className="sidebar-history">
        <p className="section-label">Conversation history</p>
        {rows.map(row => (
          <button className="nav-item history-item" key={row.id}
            aria-current={view === 'chat' && selected === row.id ? 'true' : undefined}
            onClick={() => select(row.id)} disabled={busy} title={conversationTitle(row)}>
            <MessageSquare size={15} /><span className="truncate">{conversationTitle(row)}</span>
          </button>
        ))}
        {loading && <p className="sidebar-note" role="status">Loading conversations…</p>}
        {error && <p className="sidebar-note" role="alert">{error}</p>}
        {!loading && !error && rows.length === 0 && <p className="sidebar-note">No saved conversations</p>}
        {more && <button className="text-button sidebar-note" onClick={loadMore} disabled={loading || busy}>Load more</button>}
      </div>
      <div className="sidebar-footer">
        <button className="nav-item" aria-current={view === 'settings' ? 'page' : undefined} onClick={() => navigate('settings')}><Settings2 size={17} />Settings</button>
      </div>
    </aside>
  )
}
