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

Set `LLM_API_KEY` to your provider key, `LLM_BASE_URL` to its API root (for
example, `https://api.openai.com/v1`), and `LLM_MODEL` to a Chat Completions
model ID available to your account. Use the API root without
`/chat/completions` at the end. Exported environment variables override
`.env`. Restart the server after changing these settings.

```sh
python -m uvicorn app.main:app --reload
```

Open [Swagger UI](http://127.0.0.1:8000/docs). Expand **POST /chat**, select
**Try it out**, enter the following JSON, and click **Execute**.

```json
{"message": "Explain what FastAPI is in one sentence."}
```

A successful response has the form `{"answer": "FastAPI is ..."}`.
The endpoint returns HTTP 503 if settings are missing, 504 on a provider
timeout, and 502 on other provider failures or unusable responses.

## Run tests

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The tests use a mocked provider and require no live credentials.
