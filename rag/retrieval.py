"""Query-embedding and semantic search over the SQLite vector store."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rag.vector_store import (
    SQLiteVectorStore,
    VectorSearchResult,
)


class RetrievalError(RuntimeError):
    """Raised when a question cannot be embedded or searched."""


_SINGLE_QUERY_METHODS = (
    "embed_query",
    "encode_query",
    "embed_question",
)

_BATCH_QUERY_METHODS = (
    "embed_queries",
    "encode_queries",
)


def _query_embedding(
    embedding_model: Any,
    question: str,
) -> np.ndarray:
    """Call a query-specific embedding method.

    Query prefixes such as E5's `query: ` are the embedding model's job. This
    function intentionally does not use document-embedding methods, which
    might apply the wrong prefix.
    """
    method = None
    is_batch_method = False

    for method_name in _SINGLE_QUERY_METHODS:
        candidate = getattr(embedding_model, method_name, None)
        if callable(candidate):
            method = candidate
            break

    if method is None:
        for method_name in _BATCH_QUERY_METHODS:
            candidate = getattr(embedding_model, method_name, None)
            if callable(candidate):
                method = candidate
                is_batch_method = True
                break

    if method is None:
        raise RetrievalError(
            "The embedding model needs a query-specific method such as "
            "'embed_query' or 'encode_query'. Do not use a document-embedding "
            "method for a search question."
        )

    try:
        output = method([question]) if is_batch_method else method(question)
        vector = np.asarray(output, dtype=np.float32)
    except Exception as exc:
        raise RetrievalError(
            "Could not create the query embedding. Check the Stage 4 embedding "
            "model and its local dependencies."
        ) from exc

    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]

    if vector.ndim != 1 or vector.size < 1:
        raise RetrievalError(
            "The embedding model must return one query vector."
        )

    if not np.isfinite(vector).all():
        raise RetrievalError(
            "The query embedding contains a non-finite value."
        )

    if float(np.linalg.norm(vector.astype(np.float64))) <= 1e-12:
        raise RetrievalError(
            "The query embedding has zero length and cannot be searched."
        )

    return np.ascontiguousarray(vector, dtype=np.float32)


class SemanticRetriever:
    """Embed questions and retrieve matching chunks from SQLite."""

    def __init__(
        self,
        embedding_model: Any,
        vector_store: SQLiteVectorStore,
        *,
        default_top_k: int = 5,
    ) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model is required.")
        if not callable(getattr(vector_store, "search", None)):
            raise ValueError(
                "vector_store must provide the SQLite store's search method."
            )
        if type(default_top_k) is not int or default_top_k < 1:
            raise ValueError("default_top_k must be a positive integer.")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return matching chunks in descending cosine-similarity order."""
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("Please provide a non-empty question.")

        if top_k is None:
            top_k = self.default_top_k
        if type(top_k) is not int or top_k < 1:
            raise RetrievalError("top_k must be a positive integer.")

        if where is not None and not isinstance(where, Mapping):
            raise RetrievalError("where must be a metadata-filter dictionary.")

        if document_name is not None:
            if not isinstance(document_name, str) or not document_name.strip():
                raise RetrievalError(
                    "document_name must be a non-empty string when provided."
                )
            document_name = document_name.strip()

        query_vector = _query_embedding(
            self.embedding_model,
            question.strip(),
        )

        # SQLiteVectorStore performs the dimension check and exact cosine search.
        return self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            where=where,
            document_name=document_name,
        )

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """Alias for retrieve."""
        return self.retrieve(
            question=question,
            top_k=top_k,
            where=where,
            document_name=document_name,
        )


# Short aliases for code that prefers simpler class names.
Retriever = SemanticRetriever
RetrievedChunk = VectorSearchResult
RetrievalResult = VectorSearchResult