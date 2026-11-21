"""In-memory conversation history and follow-up support for PDF RAG."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_MISSING = object()


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dictionary or an object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _page_number(value: Any) -> int | None:
    """Convert common page-number formats into an integer."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
        if number.is_integer():
            return int(number)
    except (TypeError, ValueError):
        pass

    match = re.search(r"\d+", str(value))
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None

    return None


def _truncate(text: str, limit: int) -> str:
    """Shorten text to no more than limit characters."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1] + "…"


def is_follow_up_question(question: str) -> bool:
    """Heuristically detect questions that refer to earlier chat turns."""
    text = re.sub(r"\s+", " ", str(question or "").strip()).casefold()

    if not text:
        return False

    reference_words = re.compile(
        r"\b(?:it|they|them|this|that|these|those|its|their|theirs|"
        r"he|she|him|her|there|one|ones|former|latter|both)\b",
        re.IGNORECASE,
    )
    follow_up_start = re.compile(
        r"^(?:and\b|also\b|what about\b|how about\b|what if\b|"
        r"what else\b|same\b|instead\b|then\b|continue\b|why\b)",
        re.IGNORECASE,
    )

    if reference_words.search(text) or follow_up_start.search(text):
        return True

    return text.rstrip("?.! ") in {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
    }


@dataclass(frozen=True)
class ConversationSource:
    """Small source reference saved with a chat turn.

    PDF passage text is deliberately not kept in conversation history.
    """

    source_id: str
    document_name: str
    page_number: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
        }


@dataclass(frozen=True)
class ConversationTurn:
    """One user question and the answer returned by the RAG pipeline."""

    question: str
    answer: str
    cited_source_ids: tuple[str, ...]
    sources: tuple[ConversationSource, ...]
    abstained: bool
    abstention_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "cited_source_ids": list(self.cited_source_ids),
            "sources": [source.to_dict() for source in self.sources],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }


class ConversationSession:
    """Maintain a bounded, in-memory conversation for one user/session.

    `pipeline` must provide an `answer(question)` method, such as the
    project's RAGPipeline.
    """

    def __init__(
        self,
        pipeline: Any,
        max_turns: int = 20,
        context_turns: int = 3,
        max_history_chars: int = 3000,
        max_question_chars: int | None = None,
    ) -> None:
        if pipeline is None or not callable(getattr(pipeline, "answer", None)):
            raise ValueError("pipeline must provide a callable answer(question) method.")

        for name, value in (
            ("max_turns", max_turns),
            ("context_turns", context_turns),
            ("max_history_chars", max_history_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        pipeline_config = _get(pipeline, "config", default=None)
        configured_limit = _get(
            pipeline_config,
            "max_question_chars",
            default=None,
        )

        if max_question_chars is None:
            if isinstance(configured_limit, int) and not isinstance(
                configured_limit, bool
            ):
                max_question_chars = configured_limit
            else:
                max_question_chars = 4000

        if (
            isinstance(max_question_chars, bool)
            or not isinstance(max_question_chars, int)
            or max_question_chars <= 0
        ):
            raise ValueError("max_question_chars must be a positive integer.")

        # Do not allow this wrapper to pass a longer question to the pipeline
        # than the pipeline itself accepts.
        if (
            isinstance(configured_limit, int)
            and not isinstance(configured_limit, bool)
            and configured_limit > 0
        ):
            max_question_chars = min(max_question_chars, configured_limit)

        self.pipeline = pipeline
        self.max_turns = max_turns
        self.context_turns = context_turns
        self.max_history_chars = max_history_chars
        self.max_question_chars = max_question_chars
        self._history: list[ConversationTurn] = []

    @property
    def history(self) -> tuple[ConversationTurn, ...]:
        """Return saved turns as an immutable tuple."""
        return tuple(self._history)

    @property
    def turn_count(self) -> int:
        return len(self._history)

    def clear_history(self) -> None:
        """Clear this session's in-memory conversation."""
        self._history.clear()

    def history_for_ui(self) -> list[dict[str, Any]]:
        """Return chat messages suitable for a UI such as Streamlit."""
        messages: list[dict[str, Any]] = []

        for turn in self._history:
            messages.append(
                {
                    "role": "user",
                    "content": turn.question,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.answer,
                    "cited_source_ids": list(turn.cited_source_ids),
                    "sources": [source.to_dict() for source in turn.sources],
                    "abstained": turn.abstained,
                    "abstention_reason": turn.abstention_reason,
                }
            )

        return messages

    def _build_contextual_question(self, question: str) -> str:
        """Add recent chat as reference context without exceeding the limit."""
        recent_turns = self._history[-self.context_turns :]
        if not recent_turns:
            return question

        records: list[dict[str, Any]] = []

        for turn in recent_turns:
            if turn.abstained:
                answer_text = "[No grounded answer was returned in that turn.]"
            else:
                answer_text = turn.answer

            records.append(
                {
                    "user_question": _truncate(turn.question, 700),
                    "assistant_answer": _truncate(answer_text, 900),
                    "cited_source_ids": list(turn.cited_source_ids[:5]),
                }
            )

        prefix = (
            "Use recent conversation only to resolve references such as "
            "pronouns or names. It is not evidence. Answer only the current "
            "question; factual claims must use retrieved PDF passages.\n"
            "Recent conversation (JSON):\n"
        )
        suffix = "\n\nCurrent user question:\n"

        available_history_chars = min(
            self.max_history_chars,
            self.max_question_chars - len(prefix) - len(suffix) - len(question),
        )
        if available_history_chars <= 0:
            return question

        def encoded_records() -> str:
            return json.dumps(
                records,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        while records and len(encoded_records()) > available_history_chars:
            if len(records) > 1:
                # Keep the newest turns first.
                records.pop(0)
                continue

            record = records[0]
            record["cited_source_ids"] = []

            for field_name in ("assistant_answer", "user_question"):
                current_json = encoded_records()
                if len(current_json) <= available_history_chars:
                    break

                current_text = record[field_name]
                overflow = len(current_json) - available_history_chars
                new_limit = max(0, len(current_text) - overflow - 2)
                record[field_name] = _truncate(current_text, new_limit)

            if len(encoded_records()) > available_history_chars:
                records.clear()

        if not records:
            return question

        contextual_question = (
            prefix + encoded_records() + suffix + question
        )

        if len(contextual_question) > self.max_question_chars:
            return question

        return contextual_question

    def _make_turn(self, question: str, result: Any) -> ConversationTurn:
        """Save a compact record of a pipeline answer."""
        raw_answer = _get(result, "answer", default="")
        answer = str(raw_answer or "").strip()

        raw_cited_ids = _get(result, "cited_source_ids", default=[])
        if isinstance(raw_cited_ids, str):
            raw_cited_ids = [raw_cited_ids]
        elif raw_cited_ids is None:
            raw_cited_ids = []

        try:
            cited_ids_list = list(raw_cited_ids)
        except TypeError:
            cited_ids_list = []

        cited_ids: list[str] = []
        seen_ids: set[str] = set()

        for item in cited_ids_list:
            source_id = str(item).strip()
            key = source_id.casefold()
            if source_id and key not in seen_ids:
                seen_ids.add(key)
                cited_ids.append(source_id)

        raw_sources = _get(result, "sources", default=[])
        if raw_sources is None:
            raw_sources = []
        elif isinstance(raw_sources, Mapping):
            raw_sources = [raw_sources]
        elif isinstance(raw_sources, (str, bytes)):
            raw_sources = []
        else:
            try:
                raw_sources = list(raw_sources)
            except TypeError:
                raw_sources = []

        sources: list[ConversationSource] = []
        seen_source_ids: set[str] = set()

        for source in raw_sources:
            raw_source_id = _get(source, "source_id", default=None)
            if raw_source_id is None:
                continue

            source_id = str(raw_source_id).strip()
            source_key = source_id.casefold()
            if not source_id or source_key in seen_source_ids:
                continue
            seen_source_ids.add(source_key)

            document_name = _get(source, "document_name", default="unknown")
            if document_name is None or not str(document_name).strip():
                document_name = "unknown"

            raw_page = _get(
                source,
                "page_number",
                default=_get(source, "page", default=None),
            )

            sources.append(
                ConversationSource(
                    source_id=source_id,
                    document_name=str(document_name),
                    page_number=_page_number(raw_page),
                )
            )

        raw_reason = _get(result, "abstention_reason", default=None)
        reason = str(raw_reason) if raw_reason is not None else None

        return ConversationTurn(
            question=question,
            answer=answer,
            cited_source_ids=tuple(cited_ids),
            sources=tuple(sources),
            abstained=bool(_get(result, "abstained", default=False)),
            abstention_reason=reason,
        )

    def ask(self, question: str, use_history: bool | None = None) -> Any:
        """Ask the RAG pipeline and save its answer in this session.

        By default, history is added only when the question looks like a
        follow-up. Pass use_history=True or False to override that choice.
        """
        question_text = "" if question is None else str(question).strip()

        if not question_text:
            raise ValueError("Question cannot be empty.")

        if len(question_text) > self.max_question_chars:
            raise ValueError(
                "Question exceeds the configured limit of "
                f"{self.max_question_chars} characters."
            )

        if use_history is not None and not isinstance(use_history, bool):
            raise TypeError("use_history must be True, False, or None.")

        should_use_history = bool(self._history) and (
            is_follow_up_question(question_text)
            if use_history is None
            else use_history
        )

        pipeline_question = (
            self._build_contextual_question(question_text)
            if should_use_history
            else question_text
        )

        result = self.pipeline.answer(pipeline_question)

        self._history.append(self._make_turn(question_text, result))
        if len(self._history) > self.max_turns:
            del self._history[: -self.max_turns]

        return result