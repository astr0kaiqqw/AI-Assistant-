import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from app.main import app


LLM_ENV = {
    "LLM_API_KEY": "mock-provider-key",
    "LLM_BASE_URL": "https://provider.example/v1",
    "LLM_MODEL": "mock-chat-model",
}


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("app.agent.load_dotenv"))
        self.enterContext(patch.dict(os.environ, LLM_ENV))

    def mock_provider(self, handler):
        transport = httpx.MockTransport(handler)

        def create_client(**options):
            return OpenAI(http_client=httpx.Client(transport=transport), **options)

        self.enterContext(patch("app.agent.OpenAI", side_effect=create_client))
        return self.enterContext(TestClient(app))

    def test_returns_answer_and_sends_expected_provider_request(self):
        def handler(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), "https://provider.example/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer mock-provider-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "mock-chat-model")
            self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-chat-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello there!"},
                }],
            })

        response = self.mock_provider(handler).post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "Hello there!"})

    def test_provider_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("mock timeout", request=request)

        response = self.mock_provider(handler).post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json(), {"detail": "AI provider timed out"})

    def test_provider_error_does_not_expose_provider_details(self):
        def handler(request):
            return httpx.Response(401, json={"error": {"message": LLM_ENV["LLM_API_KEY"]}})

        response = self.mock_provider(handler).post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "AI provider request failed"})

    def test_unusable_provider_responses(self):
        payloads = iter(({"choices": []}, {"choices": [None]}, {"choices": {"0": {}}}))

        def handler(request):
            return httpx.Response(200, json=next(payloads))

        client = self.mock_provider(handler)
        for _ in range(3):
            response = client.post("/chat", json={"message": "Hello"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json(), {"detail": "AI provider request failed"})

    def test_health_and_chat_without_configuration(self):
        with patch.dict(os.environ, LLM_API_KEY=""), TestClient(app) as client:
            health = client.get("/health")
            response = client.post("/chat", json={"message": "Hello"})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "AI provider is not configured"})

    def test_invalid_request_is_rejected_before_calling_provider(self):
        def handler(request):
            self.fail("Provider should not receive an invalid chat request")

        response = self.mock_provider(handler).post("/chat", json={})
        self.assertEqual(response.status_code, 422)
