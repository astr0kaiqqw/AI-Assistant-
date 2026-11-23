import os
from contextlib import contextmanager

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion


class InvalidProviderResponse(Exception):
    pass


@contextmanager
def llm_client():
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

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


def ask_agent(message: str, client: OpenAI, model: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
    except ValueError:
        raise InvalidProviderResponse from None

    choices = getattr(response, "choices", None)
    if not isinstance(response, ChatCompletion) or not isinstance(choices, list) or not choices:
        raise InvalidProviderResponse

    reply = getattr(choices[0], "message", None)
    content = getattr(reply, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise InvalidProviderResponse
    return content
