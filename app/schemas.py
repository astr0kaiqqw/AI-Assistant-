from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict


# Stored timestamps are UTC. Include the offset in API responses.
UtcTimestamp = Annotated[datetime, AfterValidator(lambda value: value.replace(tzinfo=UTC))]


class ChatRequest(BaseModel):
    message: str
    conversation_id: UUID | None = None


class ChatUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    peak_input_tokens: int | None
    input_token_limit: int | None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: UUID
    usage: ChatUsage | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: UtcTimestamp
    updated_at: UtcTimestamp


class ConversationPreview(ConversationResponse):
    title: str | None


class WorkspaceInfo(BaseModel):
    model: str | None
    input_token_limit: int | None
    tools: list[str]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: UtcTimestamp
