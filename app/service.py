from uuid import UUID

from openai import OpenAI

from app.agent import ask_agent
from app.repository import ConversationRepository


MAX_HISTORY_MESSAGES = 20


class ProviderNotConfigured(Exception):
    pass


def chat_with_memory(
    message: str,
    conversation_id: UUID | None,
    repository: ConversationRepository,
    client: OpenAI | None,
    model: str,
) -> tuple[str, UUID]:
    history = []
    if conversation_id is not None:
        history = repository.list_messages(conversation_id, MAX_HISTORY_MESSAGES)
    if client is None:
        raise ProviderNotConfigured

    context = [{"role": entry.role, "content": entry.content} for entry in history]
    answer = ask_agent(message, client, model, history=context)
    conversation_id = repository.save_turn(conversation_id, message, answer)
    return answer, conversation_id
