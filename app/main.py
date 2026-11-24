from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, APITimeoutError

from app.agent import InvalidProviderResponse, llm_client
from app.database import database_sessions
from app.document_evidence import DocumentEvidence, evidence_budget
from app.document_routes import router as document_router
from app.document_text import DocumentInputError
from app.documents import DocumentNotFound, DocumentSearchUnavailable, DocumentStore
from app.repository import ConversationNotFound, ConversationRepository
from app.schemas import (
    ChatRequest, ChatResponse, ConversationPreview, ConversationResponse, MessageResponse,
    WorkspaceInfo,
)
from app.service import ProviderNotConfigured, chat_with_memory
from app.tools import TOOL_DEFINITIONS
from app.usage import RequestUsage, verified_input_limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    with llm_client() as (client, model), database_sessions() as sessions:
        app.state.llm_client = client
        app.state.llm_model = model
        app.state.conversations = ConversationRepository(sessions)
        app.state.documents = DocumentStore(sessions)
        app.state.rag_context_budget = evidence_budget()
        yield


app = FastAPI(title="Universal AI Agent", lifespan=lifespan)
app.include_router(document_router)
STATIC_DIR = Path(__file__).with_name("static")
FRONTEND_DIST = STATIC_DIR.parent.parent / "frontend" / "dist"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount(
    "/workspace-assets",
    StaticFiles(directory=FRONTEND_DIST, check_dir=False),
    name="workspace-assets",
)


PageLimit = Annotated[int, Query(ge=1, le=100)]
PageOffset = Annotated[int, Query(ge=0, le=2**63 - 1)]


@app.exception_handler(ConversationNotFound)
async def conversation_not_found(_request: Request, _error: ConversationNotFound):
    return JSONResponse(status_code=404, content={"detail": "Conversation not found"})


@app.exception_handler(DocumentInputError)
async def invalid_document(_request: Request, error: DocumentInputError):
    return JSONResponse(status_code=error.status, content={"detail": str(error)})


@app.exception_handler(DocumentNotFound)
async def document_not_found(_request: Request, _error: DocumentNotFound):
    return JSONResponse(status_code=404, content={"detail": "Document not found"})


@app.exception_handler(DocumentSearchUnavailable)
async def document_search_unavailable(_request: Request, _error: DocumentSearchUnavailable):
    return JSONResponse(status_code=503, content={"detail": "Document search is currently unavailable"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def chat_page():
    index = FRONTEND_DIST / "index.html"
    return FileResponse(index if index.is_file() else STATIC_DIR / "index.html")


@app.get("/workspace", response_model=WorkspaceInfo)
def workspace_info(request: Request):
    state = request.app.state
    client = state.llm_client
    return WorkspaceInfo(
        model=state.llm_model if client else None,
        input_token_limit=verified_input_limit(client.base_url.host, state.llm_model)
        if client else None,
        tools=[tool["function"]["name"] for tool in TOOL_DEFINITIONS],
    )


@app.post("/chat", response_model=ChatResponse, response_model_exclude_unset=True)
def chat(request: ChatRequest, http_request: Request, include_usage: bool = False):
    state = http_request.app.state
    usage = None
    evidence = DocumentEvidence(state.documents, state.rag_context_budget)
    if include_usage and state.llm_client is not None:
        usage = RequestUsage(state.llm_client.base_url.host, state.llm_model)
    try:
        answer, conversation_id = chat_with_memory(
            request.message,
            request.conversation_id,
            state.conversations,
            state.llm_client,
            state.llm_model,
            usage=usage,
            evidence=evidence,
        )
    except ProviderNotConfigured:
        raise HTTPException(status_code=503, detail="AI provider is not configured") from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="AI provider timed out") from None
    except (APIError, InvalidProviderResponse):
        raise HTTPException(status_code=502, detail="AI provider request failed") from None

    result = ChatResponse(answer=answer, conversation_id=conversation_id)
    if evidence.used_citations:
        result.citations = evidence.used_citations
    if usage is not None:
        result.usage = usage.summary()
    return result


@app.get("/conversations", response_model=list[ConversationPreview | ConversationResponse])
def list_conversations(
    request: Request, limit: PageLimit = 50, offset: PageOffset = 0, include_title: bool = False,
):
    if include_title:
        return request.app.state.conversations.list_previews(limit, offset)
    return request.app.state.conversations.list_conversations(limit, offset)


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: UUID, request: Request, limit: PageLimit = 50, offset: PageOffset = 0
):
    return request.app.state.conversations.list_messages(conversation_id, limit, offset)


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: UUID, request: Request):
    request.app.state.conversations.delete_conversation(conversation_id)
    return Response(status_code=204)
