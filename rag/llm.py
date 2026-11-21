"""Groq chat client with validation and safe, non-retrying error handling."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GroqClientError(RuntimeError):
    """Base exception for Groq client failures."""


class GroqConfigurationError(GroqClientError):
    """Raised when local SDK or environment configuration is missing."""


class GroqAuthenticationError(GroqClientError):
    """Raised when Groq rejects the API credentials."""


class GroqRateLimitError(GroqClientError):
    """Raised when the provider reports a rate limit."""


class GroqConnectionError(GroqClientError):
    """Raised when the SDK cannot connect or times out."""


class GroqRequestError(GroqClientError):
    """Raised when Groq reports a request or service error."""


class GroqResponseError(GroqClientError):
    """Raised when the provider response has no usable text."""


@dataclass(frozen=True)
class LLMResponse:
    """Text and optional usage details returned by the chat API."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None

    @property
    def text(self) -> str:
        """Alias for code that prefers a text field name."""

        return self.content

    @property
    def usage(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _member(value: Any, name: str, default: Any = None) -> Any:
    """Read a value from either an SDK object or a dictionary."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validate_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes, bytearray)):
        raise GroqRequestError(
            "messages must be a list of role/content message objects."
        )

    try:
        message_list = list(messages)
    except TypeError as exc:
        raise GroqRequestError(
            "messages must be an iterable of role/content message objects."
        ) from exc

    if not message_list:
        raise GroqRequestError("At least one chat message is required.")

    normalized: list[dict[str, str]] = []
    has_user_message = False
    allowed_roles = {"system", "user", "assistant"}

    for index, item in enumerate(message_list, start=1):
        if not isinstance(item, Mapping):
            raise GroqRequestError(
                f"Message {index} must be a dictionary with role and content."
            )

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            raise GroqRequestError(f"Message {index} needs a valid role.")
        role = role.strip().casefold()

        if role not in allowed_roles:
            raise GroqRequestError(
                f"Message {index} uses an unsupported role. "
                "Allowed roles are system, user and assistant."
            )

        if not isinstance(content, str) or not content.strip():
            raise GroqRequestError(
                f"Message {index} needs non-empty text content."
            )

        if role == "user":
            has_user_message = True

        # Copy only known fields so unexpected caller data is not sent.
        normalized.append({"role": role, "content": content})

    if not has_user_message:
        raise GroqRequestError("At least one user message is required.")

    return normalized


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)

    if isinstance(status, bool):
        return None

    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _translate_sdk_error(error: Exception) -> GroqClientError:
    """Translate SDK failures without including raw provider error details."""

    error_name = type(error).__name__.casefold()
    status = _status_code(error)

    if status in (401, 403) or "authentication" in error_name:
        return GroqAuthenticationError(
            "Groq rejected the API credentials. Check your local key and account access."
        )

    if status == 429 or "ratelimit" in error_name or "rate_limit" in error_name:
        return GroqRateLimitError(
            "Groq reported a rate limit. Check the current limits for your account "
            "and model before trying again."
        )

    if (
        "connection" in error_name
        or "timeout" in error_name
        or "network" in error_name
    ):
        return GroqConnectionError(
            "Could not connect to Groq. Check the network and try again later."
        )

    if status is not None and status >= 500:
        return GroqRequestError(
            "Groq reported a temporary server error. Try again later."
        )

    return GroqRequestError(
        "The Groq request failed. Check the model ID, request settings and "
        "current provider documentation."
    )


class GroqChatClient:
    """Small chat-completions client. API calls are not retried automatically."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise GroqConfigurationError("A non-empty Groq API key is required.")
        if not isinstance(model, str) or not model.strip():
            raise GroqConfigurationError(
                "A Groq model ID is required. Choose one available to your account."
            )
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise GroqConfigurationError("timeout_seconds must be a positive number.")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise GroqConfigurationError("timeout_seconds must be a positive number.")

        self.model = model.strip()

        if client is not None:
            self._client = client
            return

        try:
            from groq import Groq
        except ImportError as exc:
            raise GroqConfigurationError(
                "Groq SDK is missing. Install it with: "
                'python -m pip install "groq>=0.11,<2.0"'
            ) from exc

        try:
            # Explicitly disable SDK retries; this client makes one attempt
            # per chat() call.
            self._client = Groq(
                api_key=api_key.strip(),
                timeout=float(timeout_seconds),
                max_retries=0,
            )
        except Exception as exc:
            raise GroqConfigurationError(
                "Could not initialize the Groq SDK. Check the SDK installation "
                "and configuration."
            ) from None

    @classmethod
    def from_env(
        cls,
        *,
        dotenv_path: str | Path | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> GroqChatClient:
        """Load GROQ_API_KEY and GROQ_MODEL from environment or a local .env."""

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise GroqConfigurationError(
                "python-dotenv is missing. Install it with: "
                'python -m pip install "python-dotenv>=1.0,<2.0"'
            ) from exc

        load_dotenv(dotenv_path=dotenv_path, override=False)

        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        model = os.environ.get("GROQ_MODEL", "").strip()

        missing: list[str] = []
        if not api_key:
            missing.append("GROQ_API_KEY")
        if not model:
            missing.append("GROQ_MODEL")

        if missing:
            raise GroqConfigurationError(
                "Missing local setting(s): "
                + ", ".join(missing)
                + ". Add them to your local .env file. Never paste API keys "
                "into chat or commit them."
            )

        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            client=client,
        )

    def chat(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        temperature: float = 0.2,
        max_completion_tokens: int = 512,
    ) -> LLMResponse:
        """Send one chat-completion request and return its text and usage."""

        normalized_messages = _validate_messages(messages)

        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ):
            raise GroqRequestError("temperature must be a number from 0 to 2.")
        if not math.isfinite(float(temperature)) or not (
            0.0 <= float(temperature) <= 2.0
        ):
            raise GroqRequestError("temperature must be a number from 0 to 2.")

        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens < 1
        ):
            raise GroqRequestError(
                "max_completion_tokens must be a positive integer."
            )

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                temperature=float(temperature),
                max_completion_tokens=max_completion_tokens,
            )
        except GroqClientError:
            raise
        except Exception as exc:
            # Do not include the raw SDK exception text: it can contain
            # provider response details that should not be shown to users.
            raise _translate_sdk_error(exc) from None

        choices = _member(result, "choices")
        if not choices:
            raise GroqResponseError("Groq returned no answer choices.")

        choice = choices[0]
        message = _member(choice, "message")
        content = _member(message, "content")

        if not isinstance(content, str) or not content.strip():
            raise GroqResponseError("Groq returned an empty or unsupported answer.")

        usage = _member(result, "usage")
        prompt_tokens = _optional_non_negative_int(
            _member(usage, "prompt_tokens")
        )
        completion_tokens = _optional_non_negative_int(
            _member(usage, "completion_tokens")
        )
        total_tokens = _optional_non_negative_int(
            _member(usage, "total_tokens")
        )

        finish_reason = _member(choice, "finish_reason")
        request_id = _member(result, "id")
        returned_model = _member(result, "model") or self.model

        return LLMResponse(
            content=content.strip(),
            model=str(returned_model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=(
                str(finish_reason) if finish_reason is not None else None
            ),
            request_id=str(request_id) if request_id is not None else None,
        )