"""Local text embeddings using the multilingual E5 Sentence Transformer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.models import TextChunk


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class SentenceTransformerEmbedder:
    """Create normalized local embeddings for document chunks and queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name.strip()

        # Tests can provide a fake model. Normal application code loads
        # the real model from the Hugging Face model repository.
        if model is None:
            self._model = SentenceTransformer(self.model_name, device=device)
        else:
            self._model = model

        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("The selected model did not provide an embedding dimension.")

        self.embedding_dimension = int(dimension)
        if self.embedding_dimension <= 0:
            raise ValueError("The embedding dimension must be a positive integer.")

        self.max_sequence_length = getattr(self._model, "max_seq_length", None)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")

    @staticmethod
    def _prepare_texts(texts: Iterable[str], *, prefix: str) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "Pass an iterable of text strings, such as a list; "
                "do not pass a single string."
            )

        try:
            text_list = list(texts)
        except TypeError as error:
            raise TypeError("texts must be an iterable of strings.") from error

        prepared: list[str] = []

        for text in text_list:
            if not isinstance(text, str):
                raise TypeError("Every text value must be a string.")

            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError("Text values cannot be empty.")

            prepared.append(f"{prefix}{cleaned_text}")

        return prepared

    def _encode(
        self,
        prepared_texts: list[str],
        *,
        batch_size: int,
    ) -> np.ndarray:
        if not prepared_texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        model_output = self._model.encode(
            prepared_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(model_output, dtype=np.float32)
        expected_shape = (len(prepared_texts), self.embedding_dimension)

        if vectors.shape != expected_shape:
            raise RuntimeError(
                f"Model returned shape {vectors.shape}; "
                f"expected {expected_shape}."
            )

        if not np.isfinite(vectors).all():
            raise RuntimeError("Model returned an embedding with non-finite values.")

        return vectors

    def embed_documents(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Create one normalized 'passage' embedding per document text."""
        self._validate_batch_size(batch_size)

        prepared_texts = self._prepare_texts(
            texts,
            prefix=DOCUMENT_PREFIX,
        )

        return self._encode(
            prepared_texts,
            batch_size=batch_size,
        )

    def embed_chunks(
        self,
        chunks: Iterable[TextChunk],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Create embeddings for chunks, keeping their input order."""
        self._validate_batch_size(batch_size)

        if isinstance(chunks, (str, bytes)):
            raise TypeError("chunks must contain TextChunk objects.")

        try:
            chunk_list = list(chunks)
        except TypeError as error:
            raise TypeError("chunks must be an iterable of TextChunk objects.") from error

        for chunk in chunk_list:
            if not isinstance(chunk, TextChunk):
                raise TypeError("Every item in chunks must be a TextChunk.")

        return self.embed_documents(
            (chunk.text for chunk in chunk_list),
            batch_size=batch_size,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Create a normalized query embedding."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        query_vectors = self._encode(
            [f"{QUERY_PREFIX}{cleaned_query}"],
            batch_size=1,
        )

        return query_vectors[0]