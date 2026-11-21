from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rag.retrieval import SemanticRetriever
from rag.vector_store import SQLiteVectorStore


class SyntheticEmbedder:
    """Deterministic test embedder, not a production embedding model."""

    def embed_query(self, question: str) -> np.ndarray:
        if "deadline" in question.casefold():
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def main() -> None:
    # Temporary DB: don't modify the project's persistent vector store.
    with tempfile.TemporaryDirectory(prefix="sqlite-retrieval-test-") as temp:
        store = SQLiteVectorStore(
            db_path=Path(temp) / "synthetic_vectors.sqlite3",
        )

        chunks = [
            {
                "chunk_id": "synthetic-deadline-page-1",
                "document_name": "synthetic_policy.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Synthetic example: submit requests within 14 days.",
                "metadata": {"kind": "synthetic-test"},
            },
            {
                "chunk_id": "synthetic-mileage-page-2",
                "document_name": "synthetic_policy.pdf",
                "page_number": 2,
                "chunk_index": 0,
                "text": "Synthetic example: mileage uses a sample rate.",
                "metadata": {"kind": "synthetic-test"},
            },
        ]

        document_vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        store.upsert(chunks, document_vectors)

        retriever = SemanticRetriever(
            embedding_model=SyntheticEmbedder(),
            vector_store=store,
            default_top_k=2,
        )
        results = retriever.retrieve("What is the deadline?")

        for result in results:
            print(
                f"{result.chunk_id} | {result.document_name} | "
                f"page {result.page_number} | "
                f"similarity {result.similarity:.3f}"
            )

        if not results or results[0].chunk_id != "synthetic-deadline-page-1":
            raise SystemExit("Retrieval smoke test failed.")

        print("SQLite semantic-retrieval smoke test completed.")


if __name__ == "__main__":
    main()