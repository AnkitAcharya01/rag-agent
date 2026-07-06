#just for test

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import vapi_server


class VapiServerTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(vapi_server.app)

    def test_chat_completion_returns_openai_style_payload(self):
        with patch.object(vapi_server, "run_rag", return_value="Test answer"):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "Test answer")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_webhook_returns_assistant_message_for_user_transcript(self):
        with patch.object(vapi_server, "run_rag", return_value="Webhook answer"):
            response = self.client.post(
                "/vapi/webhook",
                json={
                    "message": {
                        "type": "transcript",
                        "role": "user",
                        "transcript": "hello",
                        "history": [],
                    }
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["message"]["content"], "Webhook answer")
        self.assertEqual(payload["message"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
