from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import APIError, APITimeoutError
from pydantic import BaseModel

from app.agent import InvalidProviderResponse, ask_agent, llm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    with llm_client() as (client, model):
        app.state.llm_client = client
        app.state.llm_model = model
        yield


app = FastAPI(title="Universal AI Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    client = http_request.app.state.llm_client
    if client is None:
        raise HTTPException(status_code=503, detail="AI provider is not configured")

    try:
        answer = ask_agent(request.message, client, http_request.app.state.llm_model)
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="AI provider timed out") from None
    except (APIError, InvalidProviderResponse):
        raise HTTPException(status_code=502, detail="AI provider request failed") from None

    return ChatResponse(answer=answer)
