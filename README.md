# Universal AI Agent

A FastAPI backend that sends chat messages to an OpenAI-compatible Chat Completions provider.

## Run locally

Use Python 3.11 or later.

```sh
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

For Gemini, set `GEMINI_API_KEY` in your local `.env`. When `LLM_API_KEY`
is unset or blank, the app sends the Gemini key to Google's OpenAI-compatible
endpoint. It selects the model from `GEMINI_MODEL`, then the legacy
`LLM_MODEL`, then defaults to `gemini-3.1-flash-lite`. To select the default
model for the current launch without editing an older `.env`, run:

```sh
GEMINI_MODEL=gemini-3.1-flash-lite python -m uvicorn app.main:app --reload
```

For another OpenAI-compatible provider, set all three of `LLM_API_KEY`,
`LLM_BASE_URL`, and `LLM_MODEL`. Use the API root, for example
`https://api.openai.com/v1`. The SDK appends `/chat/completions`. The provider
and model must support function calling. If using the model from `.env`
directly, run `python -m uvicorn app.main:app --reload`. Exported environment
variables override values from `.env`.

## Workspace frontend

Build the React workspace with Node.js 22.12+ or 24 and npm:

```sh
npm --prefix frontend ci
npm --prefix frontend run build
GEMINI_MODEL=gemini-3.1-flash-lite python -m uvicorn app.main:app
```

Open `http://127.0.0.1:8000/` for the dashboard. For frontend development,
keep FastAPI running on port 8000 and run `npm --prefix frontend run dev`.
Vite serves the workspace on port 5173 and proxies API calls to FastAPI.
When the production build is absent, FastAPI serves the basic chat page.

Dashboard shows the configured model, registered tools, recently updated
conversations, and the latest reply's usage in the current browser session.
AI Chat supports Markdown, code, saved conversation switching, and deletion.
Use Enter to send or Shift+Enter for a new line. Configuration for the model
stays on the server. Settings lets you choose Dark, Light, or System
appearance. The Documents screen remains marked as planned. Use Swagger or the API to manage documents.

### Context Window monitor

The collapsible monitor appears on Dashboard and next to the chat on wider
screens. It moves below the chat on narrow screens. When opening a saved
conversation, usage becomes available after a reply in the current session.

Send a message to see usage for that request. The chat page opts in with
`POST /chat?include_usage=true`, which adds a `usage` object to the existing
response. Its input and output totals sum the provider's reported
`prompt_tokens` and `completion_tokens` across all model calls for the reply.
The bar compares the largest input count of any one call with the model's
input token limit. Tool rounds can resend context, so those totals measure
tokens processed for one reply rather than the size of the conversation.

The configured limit for `gemini-3.1-flash-lite` is 1,048,576 input tokens,
from [Google's model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite).
Other models display **Limit unknown** unless their limit is verified in the
backend. A total displays **Usage unavailable** when any model call in the
request omits the corresponding count. The API represents missing values
with `null`.

## Conversation memory

Send `{"message": "My name is Mira and I'm studying coral reefs."}` to
`POST /chat` to start a conversation. The response includes `answer` and a
UUID `conversation_id`. Pass that ID with your next `message` to continue
the same conversation. Omit the ID to start a separate conversation.

The app sends the latest 20 saved messages (up to 10 previous turns) to the
provider in chronological order. After a successful reply, it saves the user
message and final assistant reply together. Provider failures leave saved
history intact.

| Endpoint | Behavior |
| --- | --- |
| `GET /conversations` | List conversations by most recent activity. |
| `GET /conversations/{conversation_id}/messages` | Retrieve the latest page of messages, oldest first within that page. |
| `DELETE /conversations/{conversation_id}` | Delete the conversation and its messages. Returns HTTP 204. |

Both GET endpoints accept `limit` (1–100, default 50) and `offset` (default
0). For message history, increase `offset` to page backward from the newest
messages. Invalid UUIDs return HTTP 422. Unknown conversation IDs return
HTTP 404.

SQLAlchemy creates `conversations.sqlite3` in the project root on startup.
The `conversations` table stores UUIDs and UTC creation and update times.
The `messages` table stores message IDs, conversation IDs, roles, content,
and UTC creation times. Each conversation can have many messages. The
database file is ignored by Git. Set `AI_ASSISTANT_DB_PATH` to use a
different file (use a `.sqlite3` extension to keep it ignored).

**This API is for local single-user use. Do not expose it publicly without
authentication and conversation ownership checks.**

## Local document search

Start the backend as shown above and open Swagger at `http://127.0.0.1:8000/docs`.
The first upload loads the multilingual MiniLM sentence-transformers model on
CPU and downloads its weights from Hugging Face when they are absent from the
local cache. The embedding model runs locally. For document content, Gemini
receives the excerpts selected for the current chat turn.

| Method and path | Behavior |
| --- | --- |
| `POST /documents` | Upload PDF or UTF-8 TXT as multipart field `file`. Returns HTTP 201 and document metadata. |
| `GET /documents` | List documents in the local workspace. Accepts `limit` and `offset`. |
| `GET /documents/{document_id}` | Inspect metadata, including page count and indexed chunk count. |
| `GET /documents/search?q=...&top_k=4` | Return ranked excerpts with document ID, name, page and cosine similarity. |
| `DELETE /documents/{document_id}` | Remove a document, its source file and searchable chunks. Returns HTTP 204. |

Upload files up to 10 MiB. PDFs may contain up to 250 pages and must have
extractable text. Image-only scans need OCR before upload. Identical file
contents return HTTP 409, even when renamed. PDF page numbers are 1-based,
and TXT page numbers are `null`. Search returns an empty list when the
workspace has no searchable documents. A similarity score describes semantic
proximity. It is not a probability that an answer is correct.

Original files live under the ignored `.rag-data/` directory, or the directory
specified by `RAG_STORAGE_DIR`. Keep custom storage outside Git. The same SQLite
file used for conversations holds document metadata, overlapping chunks and the
serialized FAISS index. Keep the SQLite file and source directory together
when moving or backing up the workspace. Deleting a document removes it from
future retrieval. Answers already saved in chat history retain their text.

Gemini decides when to call `search_documents`. The backend validates the
arguments, returns bounded excerpts, and resolves the model's citation markers
to actual document names and pages. Chat responses include optional
`citations` when sources are used and show source names in `answer` so the
existing dashboard can display them. When a search yields no usable evidence or
the model uses invalid markers, chat returns an insufficient-evidence response.
The server validates source references. The model still interprets the excerpts.

Set `RAG_CONTEXT_TOKEN_BUDGET` to adjust the per-turn retrieved-context budget
(default `4500`, range `256–50000`). The app uses serialized UTF-8 byte counts
as a conservative proxy for tokens and caps tool rounds through the existing
chat limit. Very small budgets may omit whole chunks.

**This is a single-user local workspace. Public deployment requires
authentication and per-workspace authorization before exposing the endpoints.**

### Manually verify the document flow

1. Make a two-page PDF with a distinctive fact on page 2. For example, use
   “The Cedar Lantern experiment needs 73 amber tiles.” Upload it in Swagger
   using **POST /documents** and copy the returned `id`.
2. Call **GET /documents/{document_id}** to inspect the counts. Search for
   `Cedar Lantern amber tiles` with **GET /documents/search** and verify the
   matching excerpt has `page: 2`.
3. Send `{"message":"Search my uploaded PDF. How many amber tiles does Cedar Lantern need?"}`
   to **POST /chat**. Verify the answer contains 73 and the citation names your
   PDF and page 2. Try a question the document cannot answer and check that it
   reports insufficient evidence. Send a normal chat and the calculator prompt
   below to check the existing tools.
4. Stop and restart FastAPI using the same SQLite file and storage directory.
   Repeat the search and verify the excerpt still appears.
5. Delete the document using its ID. Repeat the search and verify none of the
   results belong to that ID. Inspecting the deleted ID should return HTTP 404.

## Try tool calling in Swagger

Open [Swagger UI](http://127.0.0.1:8000/docs), expand **POST /chat**, and select
**Try it out**. Run each request separately.

```json
{"message": "Use the calculator tool to multiply 847 by 39, then tell me the result."}
```

```json
{"message": "Use the date and time tool with its default timezone. Tell me the current local date and time."}
```

Expect an `answer` with `33033` for the calculator and the current time in
`Asia/Almaty` for the datetime tool. You can also request an explicit IANA
timezone, such as `Europe/London`.

The model can request registered tools for up to three rounds. The backend
validates the tool name and arguments, runs the tool, and sends its result
back to the model for the final answer. The endpoint returns HTTP 503 for
missing configuration, 504 for provider timeouts, and 502 for other provider
failures or unusable responses.

## Run tests

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The tests use a mocked provider and a temporary SQLite database. They require
no live credentials.
