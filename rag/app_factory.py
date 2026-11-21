"""Build the local PDF RAG pipeline with Groq-backed chat."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _hinglish_retrieval_hint(question: str) -> str | None:
    """Add a lightweight English search hint for common Romanized Hindi queries.

    This only changes the local search vector. It does not call an API and
    does not change the user's original question or the answer language.
    """
    text = re.sub(r"\s+", " ", str(question or "").strip().casefold())
    words = set(re.findall(r"[a-z]+", text))

    hinglish_markers = {
        "kab", "kb", "khul", "khulta", "khulti", "khulte",
        "khulna", "band", "bandh", "baje", "samay", "waqt",
        "hai", "hain", "hoti", "hota",
    }

    if not words.intersection(hinglish_markers):
        return None

    open_words = {
        "khul", "khulta", "khulti", "khulte", "khulna",
        "open", "opens", "opening",
    }
    close_words = {
        "band", "bandh", "hoti", "hota",
        "close", "closes", "closed", "closing",
    }

    has_open = bool(words.intersection(open_words))
    has_close = bool(words.intersection(close_words))

    ignored_words = {
        "kab", "kb", "khul", "khulta", "khulti", "khulte", "khulna",
        "band", "bandh", "hoti", "hota", "hai", "hain", "h",
        "baje", "samay", "waqt", "kis", "kya", "ka", "ki", "ke",
        "mera", "meri", "mere", "please", "tell", "me", "when",
        "what", "time", "does", "do", "is", "the", "a", "an",
        "open", "opens", "opening", "close", "closes", "closed",
        "closing", "library", "pustakalay", "pustakalaya",
        "pustakghar", "kitabghar",
    }

    tokens = re.findall(r"[a-z]+", text)
    subject_words = [word for word in tokens if word not in ignored_words]

    subject_aliases = {
        "pustakalay": "library",
        "pustakalaya": "library",
        "pustakghar": "library",
        "kitabghar": "library",
        "ghar": "home",
    }

    subject_words = [
        subject_aliases.get(word, word)
        for word in subject_words
    ]

    # These words describe the subject, so keep them in the search hint.
    if "library" in words:
        subject_words.insert(0, "library")

    # Remove repeated words while preserving order.
    subject_words = list(dict.fromkeys(subject_words))
    subject = " ".join(subject_words).strip() or "place"
    topic = f"the {subject}"

    if has_open:
        return (
            f"When does {topic} open? What time does {topic} open? "
            f"{subject} opening time and opening hours."
        )

    if has_close:
        return (
            f"When does {topic} close? What time does {topic} close? "
            f"{subject} closing time and closing hours."
        )

    return None


class MultilingualE5EmbeddingModel:
    """CPU wrapper for intfloat/multilingual-e5-small."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("The embedding model name cannot be empty.")

        self.model_name = model_name.strip()
        self._model: Any | None = None
        self._model_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Any:
        """Load the local model once and reuse it."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                import torch

                requested_threads = int(
                    os.getenv("AURA_TORCH_THREADS", "4")
                )
                thread_count = max(
                    1,
                    min(requested_threads, os.cpu_count() or 1),
                )
                torch.set_num_threads(thread_count)
            except (ImportError, ValueError, RuntimeError):
                pass

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is missing. Install the project "
                    "dependencies before indexing or chatting."
                ) from exc

            # The first run may download this model; later runs use its cache.
            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
            )
            return self._model

    @staticmethod
    def _add_prefix(text: Any, prefix: str) -> str:
        value = str(text or "").strip()
        if value.casefold().startswith(prefix.casefold()):
            return value
        return f"{prefix}{value}"

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._get_model()
        model_inputs = [
            self._add_prefix(text, prefix)
            for text in texts
        ]

        vectors = model.encode(
            model_inputs,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = np.asarray(vectors, dtype=np.float32)

        if result.ndim != 2 or result.shape[0] != len(texts):
            raise ValueError(
                "The embedding model returned an unexpected vector shape: "
                f"{result.shape}."
            )

        if not np.isfinite(result).all():
            raise ValueError("The embedding model returned invalid numbers.")

        return result

    def warm_up(self) -> None:
        """Load and warm up the local model at app startup."""
        started = time.perf_counter()
        self._encode(["warm up"], "query: ")
        elapsed = time.perf_counter() - started
        print(
            f"[AURA timing] Local model startup + warm-up: {elapsed:.2f}s",
            flush=True,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts), "passage: ")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a question; add an English search hint for common Hinglish."""
        started = time.perf_counter()
        original = str(query or "").strip()
        hint = _hinglish_retrieval_hint(original)

        if hint:
            # Encode both locally in one model call; no extra API request.
            vectors = self._encode(
                [original, hint],
                "query: ",
            )

            # Give the English search hint more weight for English PDF text.
            combined = 0.30 * vectors[0] + 0.70 * vectors[1]
            norm = float(np.linalg.norm(combined))

            if norm > 0:
                combined = combined / norm

            result = np.asarray(combined, dtype=np.float32)
        else:
            result = self._encode([original], "query: ")[0]

        elapsed = time.perf_counter() - started
        print(
            f"[AURA timing] Local question embedding: {elapsed:.2f}s",
            flush=True,
        )
        return result

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(list(queries), "query: ")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self.embed_query(query)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.embed_queries(queries)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_question(self, query: str) -> np.ndarray:
        return self.embed_query(query)


class LanguageAwareGroqClient:
    """Ask Groq to answer briefly in the user's language and script."""

    LANGUAGE_INSTRUCTION = """
Answer in the same language and script as the user's current question.

If the user writes Hindi using Latin letters (Romanized Hindi/Hinglish), reply
in natural Romanized Hindi/Hinglish. If the user writes in Devanagari, reply
in Devanagari. For follow-up questions, use the language and script of the
text after "Current user question:".

Keep the answer concise. Use only the supplied PDF evidence. Preserve
citation markers exactly, for example [S1]. Do not invent citations.
""".strip()

    def __init__(self, client: Any) -> None:
        if client is None or not callable(getattr(client, "chat", None)):
            raise ValueError("A Groq-compatible chat client is required.")
        self._client = client

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        forwarded_messages = [
            dict(message)
            for message in messages
        ]

        system_index = next(
            (
                index
                for index, message in enumerate(forwarded_messages)
                if message.get("role") == "system"
            ),
            None,
        )

        if system_index is None:
            forwarded_messages.insert(
                0,
                {
                    "role": "system",
                    "content": self.LANGUAGE_INSTRUCTION,
                },
            )
        else:
            existing = str(
                forwarded_messages[system_index].get("content", "")
            )
            forwarded_messages[system_index]["content"] = (
                existing + "\n\n" + self.LANGUAGE_INSTRUCTION
            )

        started = time.perf_counter()
        try:
            return self._client.chat(
                messages=forwarded_messages,
                **kwargs,
            )
        finally:
            elapsed = time.perf_counter() - started
            print(
                f"[AURA timing] Groq request: {elapsed:.2f}s",
                flush=True,
            )


def _is_configured(value: str) -> bool:
    cleaned = str(value or "").strip().casefold()

    if not cleaned:
        return False

    return not any(
        marker in cleaned
        for marker in (
            "replace_with",
            "your_actual",
            "paste_your",
        )
    )


def create_pipeline():
    """Create and warm the local PDF pipeline with an optional Groq client."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "").strip()
    llm_client = None

    # Client creation does not send an API request.
    if _is_configured(api_key) and _is_configured(groq_model):
        from rag.llm import GroqChatClient

        llm_client = LanguageAwareGroqClient(
            GroqChatClient.from_env()
        )

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "intfloat/multilingual-e5-small",
    ).strip()

    from rag.pipeline import RAGConfig, RAGPipeline
    from rag.vector_store import SQLiteVectorStore

    embedding_model = MultilingualE5EmbeddingModel(model_name)
    vector_store = SQLiteVectorStore()

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_client=llm_client,
        config=RAGConfig(
            top_k=3,
            max_context_chars=4500,
            max_completion_tokens=160,
        ),
    )

    # Local model work happens at startup, not on the first chat question.
    embedding_model.warm_up()
    return pipeline