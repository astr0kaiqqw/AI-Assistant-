export type Usage = {
  input_tokens: number | null
  output_tokens: number | null
  peak_input_tokens: number | null
  input_token_limit: number | null
}
export type Conversation = {
  id: string
  title: string | null
  created_at: string
  updated_at: string
}
export type Message = {
  id: string | number
  role: 'user' | 'assistant'
  content: string
}
export type Workspace = {
  model: string | null
  input_token_limit: number | null
  tools: string[]
}
type ChatResponse = {
  answer: string
  conversation_id: string
  usage?: Usage
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status})`)
  }
  return response.status === 204 ? undefined as T : response.json()
}

export const pageSize = 30
export const api = {
  workspace: () => request<Workspace>('/workspace'),
  conversations: (offset = 0) =>
    request<Conversation[]>(`/conversations?include_title=true&limit=${pageSize}&offset=${offset}`),
  messages: (id: string, offset = 0) =>
    request<Message[]>(`/conversations/${id}/messages?limit=${pageSize}&offset=${offset}`),
  remove: (id: string) => request<void>(`/conversations/${id}`, { method: 'DELETE' }),
  chat: (message: string, conversation_id: string | null) =>
    request<ChatResponse>('/chat?include_usage=true', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, conversation_id }),
    }),
}

export function conversationTitle(conversation?: Conversation) {
  return conversation?.title?.replace(/\s+/g, ' ').trim() || 'Untitled conversation'
}

export function errorText(error: unknown) {
  return error instanceof Error ? error.message : 'The workspace could not complete this request.'
}
