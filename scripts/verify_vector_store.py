from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rag.vector_store import SQLiteVectorStore


def main() -> None:
    # Temporary DB: test project ka real vector-store data nahi badlega.
    with tempfile.TemporaryDirectory(prefix="sqlite-vector-test-") as temp:
        store = SQLiteVectorStore(persist_directory=Path(temp))

        chunks = [
            {
                "chunk_id": "synthetic-page-1",
                "document_name": "synthetic_policy.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Synthetic policy: requests are due within 14 days.",
                "metadata": {"kind": "synthetic-test"},
            },
            {
                "chunk_id": "synthetic-page-2",
                "document_name": "synthetic_policy.pdf",
                "page_number": 2,
                "chunk_index": 0,
                "text": "Synthetic policy: mileage uses a sample rate.",
                "metadata": {"kind": "synthetic-test"},
            },
        ]

        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        saved = store.upsert(chunks, embeddings)
        results = store.search([1.0, 0.0, 0.0], top_k=2)

        print(f"Saved chunks: {saved}")
        print(f"Search results: {len(results)}")
        for result in results:
            print(
                f"{result.chunk_id} | {result.document_name} | "
                f"page {result.page_number} | "
                f"similarity {result.similarity:.3f}"
            )

        if not results or results[0].chunk_id != "synthetic-page-1":
            raise SystemExit("Smoke test failed: expected result was not first.")

        print("SQLite vector-store smoke test completed.")


if __name__ == "__main__":
    main()