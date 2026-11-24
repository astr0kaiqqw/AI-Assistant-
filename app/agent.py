import os
from contextlib import contextmanager

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall

from app.document_evidence import GROUNDING_INSTRUCTION, DocumentEvidence
from app.tools import DOCUMENT_SEARCH_TOOL, TOOL_DEFINITIONS, execute_tool
from app.usage import RequestUsage


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_TOOL_ROUNDS = 3


class InvalidProviderResponse(Exception):
    pass


@contextmanager
def llm_client():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            base_url = GEMINI_BASE_URL
            model = os.getenv("GEMINI_MODEL", "").strip() or model or DEFAULT_GEMINI_MODEL

    if not (api_key and base_url and model):
        yield None, model
        return

    with OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=0,
    ) as client:
        yield client, model


def ask_agent(
    message: str,
    client: OpenAI,
    model: str,
    *,
    history: list[dict[str, str]] | None = None,
    usage: RequestUsage | None = None,
    evidence: DocumentEvidence | None = None,
) -> str:
    messages = []
    if evidence is not None and (history or evidence.documents.has_documents()):
        messages.append({"role": "system", "content": GROUNDING_INSTRUCTION})
    messages.extend([*(history or []), {"role": "user", "content": message}])

    for round_number in range(MAX_TOOL_ROUNDS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto" if round_number < MAX_TOOL_ROUNDS else "none",
            )
        except ValueError:
            raise InvalidProviderResponse from None

        choices = getattr(response, "choices", None)
        if not isinstance(response, ChatCompletion) or not isinstance(choices, list) or not choices:
            raise InvalidProviderResponse
        if usage is not None:
            usage.record(response)

        reply = getattr(choices[0], "message", None)
        calls = getattr(reply, "tool_calls", None)
        if not calls:
            content = getattr(reply, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise InvalidProviderResponse
            return content

        if round_number == MAX_TOOL_ROUNDS or not isinstance(calls, list):
            raise InvalidProviderResponse

        for call in calls:
            if (
                not isinstance(call, ChatCompletionMessageFunctionToolCall)
                or not isinstance(call.id, str)
                or not call.id
                or not isinstance(getattr(call.function, "name", None), str)
                or not isinstance(getattr(call.function, "arguments", None), str)
            ):
                raise InvalidProviderResponse

        # Replay the original calls, including Gemini thought signatures when present.
        messages.append({
            "role": "assistant",
            "content": getattr(reply, "content", None),
            "tool_calls": [call.model_dump(exclude_none=True) for call in calls],
        })
        for call in calls:
            if evidence is not None and call.function.name == DOCUMENT_SEARCH_TOOL:
                evidence.searched = True
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": execute_tool(
                    call.function.name, call.function.arguments,
                    document_search=evidence.search if evidence is not None else None,
                ),
            })

    raise InvalidProviderResponse
