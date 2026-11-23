from openai.types.chat import ChatCompletion

from app.schemas import ChatUsage


GEMINI_HOST = "generativelanguage.googleapis.com"
# Google model specification: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite
GEMINI_INPUT_LIMITS = {"gemini-3.1-flash-lite": 1_048_576}


def verified_input_limit(provider_host: str, model: str) -> int | None:
    return GEMINI_INPUT_LIMITS.get(model) if provider_host == GEMINI_HOST else None


def token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def complete_total(counts: list[int | None]) -> int | None:
    if not counts or None in counts:
        return None
    return sum(count for count in counts if count is not None)


class RequestUsage:
    def __init__(self, provider_host: str, model: str):
        self.input_limit = verified_input_limit(provider_host, model)
        self.input_counts: list[int | None] = []
        self.output_counts: list[int | None] = []

    def record(self, response: ChatCompletion) -> None:
        usage = getattr(response, "usage", None)
        self.input_counts.append(token_count(getattr(usage, "prompt_tokens", None)))
        self.output_counts.append(token_count(getattr(usage, "completion_tokens", None)))

    def summary(self) -> ChatUsage:
        total_input = complete_total(self.input_counts)
        peak_input = None
        if total_input is not None:
            peak_input = max(count for count in self.input_counts if count is not None)
        return ChatUsage(
            input_tokens=total_input,
            output_tokens=complete_total(self.output_counts),
            peak_input_tokens=peak_input,
            input_token_limit=self.input_limit,
        )
