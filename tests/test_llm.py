import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.llm import (
    GroqAuthenticationError,
    GroqChatClient,
    GroqConfigurationError,
    GroqConnectionError,
    GroqRateLimitError,
    GroqRequestError,
    GroqResponseError,
    LLMResponse,
)


def make_response(
    content="Synthetic test answer.",
    *,
    prompt_tokens=12,
    completion_tokens=5,
    total_tokens=17,
):
    return SimpleNamespace(
        id="synthetic-request-id",
        model="synthetic-test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else make_response()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeGroqClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class GroqClientTests(unittest.TestCase):
    def make_client(self, *, response=None, error=None):
        completions = FakeCompletions(response=response, error=error)
        client = GroqChatClient(
            api_key="synthetic-test-key",
            model="synthetic-test-model",
            client=FakeGroqClient(completions),
        )
        return client, completions

    def test_chat_returns_text_and_token_usage(self):
        client, _completions = self.make_client()

        response = client.chat(
            [
                {"role": "system", "content": "Use only the supplied context."},
                {"role": "user", "content": "A synthetic test question?"},
            ],
        )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Synthetic test answer.")
        self.assertEqual(response.text, "Synthetic test answer.")
        self.assertEqual(response.model, "synthetic-test-model")
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.total_tokens, 17)

    def test_chat_passes_model_and_token_limit(self):
        client, completions = self.make_client()

        client.chat(
            [{"role": "user", "content": "A synthetic test question?"}],
            temperature=0.1,
            max_completion_tokens=64,
        )

        call = completions.calls[0]
        self.assertEqual(call["model"], "synthetic-test-model")
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["max_completion_tokens"], 64)

    def test_chat_makes_one_attempt(self):
        client, completions = self.make_client()

        client.chat([{"role": "user", "content": "A test question?"}])

        self.assertEqual(len(completions.calls), 1)

    def test_invalid_messages_are_rejected_before_api_call(self):
        client, completions = self.make_client()

        invalid_messages = [
            [],
            [{"role": "system", "content": "System only."}],
            [{"role": "user", "content": "   "}],
            [{"role": "tool", "content": "Unsupported role."}],
            ["not a message object"],
        ]

        for messages in invalid_messages:
            with self.subTest(messages=messages):
                with self.assertRaises(GroqRequestError):
                    client.chat(messages)

        self.assertEqual(completions.calls, [])

    def test_invalid_token_limit_is_rejected(self):
        client, completions = self.make_client()

        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(GroqRequestError):
                    client.chat(
                        [{"role": "user", "content": "A test question?"}],
                        max_completion_tokens=value,
                    )

        self.assertEqual(completions.calls, [])

    def test_invalid_temperature_is_rejected(self):
        client, completions = self.make_client()

        for value in (-0.1, 2.1, float("nan"), True):
            with self.subTest(value=value):
                with self.assertRaises(GroqRequestError):
                    client.chat(
                        [{"role": "user", "content": "A test question?"}],
                        temperature=value,
                    )

        self.assertEqual(completions.calls, [])

    def test_missing_environment_settings_are_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("dotenv.load_dotenv", return_value=False):
                with self.assertRaises(GroqConfigurationError):
                    GroqChatClient.from_env(
                        client=FakeGroqClient(FakeCompletions())
                    )

    def test_from_env_uses_configured_model(self):
        fake_client = FakeGroqClient(FakeCompletions())

        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "synthetic-test-key",
                "GROQ_MODEL": "synthetic-account-model",
            },
            clear=True,
        ):
            with patch("dotenv.load_dotenv", return_value=False):
                client = GroqChatClient.from_env(client=fake_client)

        self.assertEqual(client.model, "synthetic-account-model")

    def test_sdk_is_configured_with_zero_retries(self):
        captured = {}
        fake_client = FakeGroqClient(FakeCompletions())

        def fake_groq_constructor(**kwargs):
            captured.update(kwargs)
            return fake_client

        fake_module = types.ModuleType("groq")
        fake_module.Groq = fake_groq_constructor

        with patch.dict(sys.modules, {"groq": fake_module}):
            GroqChatClient(
                api_key="synthetic-test-key",
                model="synthetic-test-model",
                timeout_seconds=12,
            )

        self.assertEqual(captured["max_retries"], 0)
        self.assertEqual(captured["timeout"], 12.0)

    def test_authentication_error_is_translated(self):
        class AuthenticationError(Exception):
            status_code = 401

        client, _ = self.make_client(error=AuthenticationError("private detail"))

        with self.assertRaises(GroqAuthenticationError):
            client.chat([{"role": "user", "content": "A test question?"}])

    def test_rate_limit_error_is_translated(self):
        class RateLimitError(Exception):
            status_code = 429

        client, _ = self.make_client(error=RateLimitError("private detail"))

        with self.assertRaises(GroqRateLimitError):
            client.chat([{"role": "user", "content": "A test question?"}])

    def test_connection_error_is_translated(self):
        class APIConnectionError(Exception):
            pass

        client, _ = self.make_client(error=APIConnectionError("private detail"))

        with self.assertRaises(GroqConnectionError):
            client.chat([{"role": "user", "content": "A test question?"}])

    def test_other_provider_errors_are_translated(self):
        class BadRequestError(Exception):
            status_code = 400

        client, _ = self.make_client(error=BadRequestError("private detail"))

        with self.assertRaises(GroqRequestError):
            client.chat([{"role": "user", "content": "A test question?"}])

    def test_empty_provider_response_is_rejected(self):
        client, _ = self.make_client(response=SimpleNamespace(choices=[]))

        with self.assertRaises(GroqResponseError):
            client.chat([{"role": "user", "content": "A test question?"}])


if __name__ == "__main__":
    unittest.main()