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

The tests use a mocked provider and require no live credentials.
