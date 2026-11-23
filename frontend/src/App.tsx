import {
  ChevronRight, Cpu, LayoutDashboard, Menu, MessageSquare, Monitor, Moon, PanelLeftOpen, Settings2, Sun, Trash2,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, conversationTitle, errorText, pageSize, type Conversation, type Message, type Usage, type Workspace } from './api'
import { Chat } from './Chat'
import { Dashboard } from './Dashboard'
import { Sidebar, type View } from './Sidebar'

type Theme = 'dark' | 'light' | 'system'
const themes = [
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'system', label: 'System', Icon: Monitor },
] as const

function currentView(): View {
  const hash = location.hash.slice(1)
  return hash === 'chat' || hash === 'settings' ? hash : 'dashboard'
}

function savedTheme(): Theme {
  const value = localStorage.getItem('assistant-theme')
  return value === 'light' || value === 'system' ? value : 'dark'
}

export function App() {
  const [view, setView] = useState(currentView)
  const [theme, setTheme] = useState(savedTheme)
  const [collapsed, setCollapsed] = useState(() => matchMedia('(max-width: 760px)').matches)
  const [workspace, setWorkspace] = useState<Workspace>()
  const [rows, setRows] = useState<Conversation[]>([])
  const [historyOffset, setHistoryOffset] = useState(0)
  const [historyMore, setHistoryMore] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [hasOlder, setHasOlder] = useState(false)
  const [loading, setLoading] = useState(false)
  const [olderLoading, setOlderLoading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [pending, setPending] = useState('')
  const [draft, setDraft] = useState('')
  const [chatError, setChatError] = useState('')
  const [usageById, setUsageById] = useState<Record<string, Usage | undefined>>({})
  const [latest, setLatest] = useState<{ id: string; usage?: Usage }>()
  const listRequest = useRef(0)
  const busy = loading || olderLoading || deleting || !!pending

  async function getConversations(offset = 0) {
    const request = ++listRequest.current
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const page = await api.conversations(offset)
      if (request !== listRequest.current) return
      setRows(previous => offset === 0 ? page : [...new Map([...previous, ...page].map(row => [row.id, row])).values()])
      setHistoryOffset(offset + page.length)
      setHistoryMore(page.length === pageSize)
    } catch (error) {
      if (request === listRequest.current) setHistoryError(errorText(error))
    } finally {
      if (request === listRequest.current) setHistoryLoading(false)
    }
  }

  useEffect(() => {
    void getConversations()
    void api.workspace().then(setWorkspace).catch(() => setWorkspace(undefined))
    const onHash = () => setView(currentView())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const system = matchMedia('(prefers-color-scheme: dark)')
    function apply() {
      document.documentElement.dataset.theme = theme === 'system' ? (system.matches ? 'dark' : 'light') : theme
    }
    apply()
    localStorage.setItem('assistant-theme', theme)
    system.addEventListener('change', apply)
    return () => system.removeEventListener('change', apply)
  }, [theme])

  function navigate(next: View) {
    setView(next)
    location.hash = next
    if (matchMedia('(max-width: 760px)').matches) setCollapsed(true)
  }

  function create() {
    if (busy) return
    setSelected(null)
    setMessages([])
    setHasOlder(false)
    setDraft('')
    setChatError('')
    navigate('chat')
  }

  async function select(id: string) {
    if (busy) return
    navigate('chat')
    if (selected === id && !chatError) return
    setLoading(true)
    setChatError('')
    setDraft('')
    try {
      const history = await api.messages(id)
      setSelected(id)
      setMessages(history)
      setHasOlder(history.length === pageSize)
    } catch (error) {
      setChatError(errorText(error))
    } finally {
      setLoading(false)
    }
  }

  async function loadOlder() {
    if (!selected || busy) return
    setOlderLoading(true)
    setChatError('')
    try {
      const history = await api.messages(selected, messages.length)
      setMessages(previous => [...history, ...previous])
      setHasOlder(history.length === pageSize)
    } catch (error) {
      setChatError(errorText(error))
    } finally {
      setOlderLoading(false)
    }
  }

  async function send() {
    const text = draft.trim()
    if (!text || busy) return
    setPending(text)
    setDraft('')
    setChatError('')
    try {
      const result = await api.chat(text, selected)
      const id = result.conversation_id
      setSelected(id)
      setMessages(previous => [...previous,
        { id: crypto.randomUUID(), role: 'user', content: text },
        { id: crypto.randomUUID(), role: 'assistant', content: result.answer },
      ])
      setUsageById(previous => ({ ...previous, [id]: result.usage }))
      setLatest({ id, usage: result.usage })
      void getConversations()
    } catch (error) {
      setChatError(errorText(error))
      setDraft(previous => previous || text)
      if (selected) setUsageById(previous => ({ ...previous, [selected]: undefined }))
    } finally {
      setPending('')
    }
  }

  async function remove() {
    if (!selected || busy || !window.confirm('Delete this conversation and all its messages?')) return
    setDeleting(true)
    setChatError('')
    try {
      await api.remove(selected)
      setRows(previous => previous.filter(row => row.id !== selected))
      setUsageById(previous => {
        const remaining = { ...previous }
        delete remaining[selected]
        return remaining
      })
      if (latest?.id === selected) setLatest(undefined)
      create()
      void getConversations()
    } catch (error) {
      setChatError(errorText(error))
    } finally {
      setDeleting(false)
    }
  }

  const current = rows.find(row => row.id === selected)
  const chatTitle = selected ? conversationTitle(current) : 'New conversation'
  const ViewIcon = view === 'dashboard' ? LayoutDashboard : view === 'chat' ? MessageSquare : Settings2

  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {!collapsed && <button className="mobile-overlay" aria-label="Close sidebar" onClick={() => setCollapsed(true)} />}
      <Sidebar view={view} selected={selected} rows={rows} more={historyMore} loading={historyLoading} error={historyError}
        busy={busy} collapsed={collapsed} navigate={navigate} select={select} create={create}
        loadMore={() => void getConversations(historyOffset)} toggle={() => setCollapsed(value => !value)} />
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            {collapsed && <button className="icon-button open-sidebar" onClick={() => setCollapsed(false)} aria-label="Expand sidebar" aria-controls="workspace-sidebar" aria-expanded={false}><PanelLeftOpen size={19} /><Menu size={20} /></button>}
            <span className="breadcrumb-root">Workspace</span><ChevronRight size={14} className="breadcrumb-divider" />
            <ViewIcon size={17} /><strong className="truncate">{view === 'dashboard' ? 'Dashboard' : view === 'settings' ? 'Settings' : chatTitle}</strong>
          </div>
          <div className="topbar-actions">
            {workspace?.model && <span className="model-chip" title={workspace.model}><Cpu size={14} /><span className="truncate">{workspace.model}</span></span>}
            {view === 'chat' && selected && <button className="icon-button delete-button" disabled={busy} onClick={remove} aria-label="Delete conversation" title="Delete conversation"><Trash2 size={18} /></button>}
          </div>
        </header>
        <main className={`main-view ${view === 'chat' ? 'chat-view' : ''}`}>
          {view === 'dashboard' && <Dashboard workspace={workspace} rows={rows} usage={latest?.usage} busy={busy}
            loading={historyLoading} error={historyError} create={create} select={select} />}
          {view === 'chat' && <Chat workspace={workspace} usage={selected ? usageById[selected] : undefined}
            messages={messages} pending={pending} draft={draft} error={chatError} loading={loading || deleting}
            olderLoading={olderLoading} hasOlder={hasOlder} setDraft={setDraft} send={send} loadOlder={loadOlder} />}
          {view === 'settings' && <div className="settings-view"><h1>Settings</h1><section className="card settings-card">
            <div className="card-heading"><span className="heading-icon violet"><Sun size={17} /></span><h2>Appearance</h2></div>
            <p className="muted">Choose a theme for this browser.</p>
            <div className="theme-options">{themes.map(({ value, label, Icon }) =>
              <button key={value} onClick={() => setTheme(value)} aria-pressed={theme === value}><Icon size={18} />{label}</button>,
            )}</div>
          </section></div>}
        </main>
      </div>
    </div>
  )
}
