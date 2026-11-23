from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from openai import APIError, APITimeoutError

from app.agent import InvalidProviderResponse, llm_client
from app.database import database_sessions
from app.repository import ConversationNotFound, ConversationRepository
from app.schemas import ChatRequest, ChatResponse, ConversationResponse, MessageResponse
from app.service import ProviderNotConfigured, chat_with_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    with llm_client() as (client, model), database_sessions() as sessions:
        app.state.llm_client = client
        app.state.llm_model = model
        app.state.conversations = ConversationRepository(sessions)
        yield


app = FastAPI(title="Universal AI Agent", lifespan=lifespan)


PageLimit = Annotated[int, Query(ge=1, le=100)]
PageOffset = Annotated[int, Query(ge=0, le=2**63 - 1)]


@app.exception_handler(ConversationNotFound)
async def conversation_not_found(_request: Request, _error: ConversationNotFound):
    return JSONResponse(status_code=404, content={"detail": "Conversation not found"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    state = http_request.app.state
    try:
        answer, conversation_id = chat_with_memory(
            request.message,
            request.conversation_id,
            state.conversations,
            state.llm_client,
            state.llm_model,
        )
    except ProviderNotConfigured:
        raise HTTPException(status_code=503, detail="AI provider is not configured") from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="AI provider timed out") from None
    except (APIError, InvalidProviderResponse):
        raise HTTPException(status_code=502, detail="AI provider request failed") from None

    return ChatResponse(answer=answer, conversation_id=conversation_id)


@app.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(request: Request, limit: PageLimit = 50, offset: PageOffset = 0):
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
