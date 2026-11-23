import { ArrowUp, Calculator, Clock3, Code2, LoaderCircle, Sparkles } from 'lucide-react'
import { useEffect, useRef, type FormEvent } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message, Usage, Workspace } from './api'
import { ContextWindow } from './ContextWindow'

const starters = [
  { Icon: Code2, label: 'Review an API design', prompt: 'What should I check when reviewing a REST API design?' },
  { Icon: Calculator, label: 'Run a calculation', prompt: 'Use the calculator to multiply 847 by 39.' },
  { Icon: Clock3, label: 'Check the local time', prompt: 'Use the date and time tool to tell me the current time in Asia/Almaty.' },
]

function ChatMessage({ message }: { message: Message }) {
  return (
    <article className={`message ${message.role}`}>
      {message.role === 'assistant' && <span className="assistant-avatar" aria-hidden="true"><Sparkles size={16} /></span>}
      <div className="message-text">
        <h2 className="sr-only">{message.role === 'assistant' ? 'Assistant' : 'You'}</h2>
        {message.role === 'assistant'
          ? <Markdown skipHtml remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          : <p>{message.content}</p>}
      </div>
    </article>
  )
}

type Props = {
  workspace?: Workspace; usage?: Usage; messages: Message[]; pending: string; draft: string;
  error: string; loading: boolean; olderLoading: boolean; hasOlder: boolean;
  setDraft: (value: string) => void; send: () => void; loadOlder: () => void;
}

export function Chat({ workspace, usage, messages, pending, draft, error, loading,
  olderLoading, hasOlder, setDraft, send, loadOlder }: Props) {
  const log = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const lastMessage = messages.at(-1)?.id
  const previousLast = useRef<typeof lastMessage>(undefined)
  useEffect(() => {
    if (log.current && (previousLast.current !== lastMessage || pending)) {
      log.current.scrollTop = log.current.scrollHeight
    }
    previousLast.current = lastMessage
  }, [lastMessage, pending])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (draft.trim() && !pending && !loading) send()
  }

  return (
    <div className="chat-workspace">
      <section className="chat-pane" aria-label="AI chat">
        <div className="chat-scroll" ref={log}>
          {loading ? <p className="chat-loading" role="status"><LoaderCircle size={18} className="spinner" /> Loading conversation…</p> :
            <div className="thread">
              {hasOlder && <button className="text-button older-button" onClick={loadOlder} disabled={olderLoading || !!pending}>{olderLoading ? 'Loading…' : 'Load earlier messages'}</button>}
              {!messages.length && !pending && !error && <div className="chat-empty">
                <span className="empty-mark"><Sparkles size={27} /></span>
                <h1>What are we working on?</h1>
                <p>Ask a question, work through an idea, or try a tool.</p>
                <div className="starters">{starters.map(({ Icon, label, prompt }) =>
                  <button key={label} onClick={() => { setDraft(prompt); input.current?.focus() }}><Icon size={16} />{label}</button>,
                )}</div>
              </div>}
              <div role="log" aria-label="Conversation messages">
                {messages.map(message => <ChatMessage key={message.id} message={message} />)}
                {pending && <ChatMessage message={{ id: 'pending', role: 'user', content: pending }} />}
              </div>
              {pending && <p className="chat-loading" role="status"><LoaderCircle size={18} className="spinner" /> Waiting for the assistant…</p>}
            </div>}
        </div>
        <div className="composer-wrap">
          {error && <div className="chat-error" role="alert">{error}</div>}
          <form className="composer" onSubmit={submit}>
            <textarea ref={input} value={draft} onChange={event => setDraft(event.target.value)}
              aria-label="Message" placeholder="Message AI Assistant…" rows={2}
              onKeyDown={event => {
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }} />
            <div className="composer-footer">
              <span>Shift + Enter for a new line</span>
              <button className="send-button" type="submit" aria-label="Send message" title="Send message" disabled={!draft.trim() || !!pending || loading}>
                {pending ? <LoaderCircle size={17} className="spinner" /> : <ArrowUp size={20} />}
              </button>
            </div>
          </form>
        </div>
      </section>
      <aside className="chat-rail" aria-label="Request details">
        <ContextWindow usage={usage} limit={workspace?.input_token_limit} />
      </aside>
    </div>
  )
}
