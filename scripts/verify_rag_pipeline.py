from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import numpy as np

from rag.llm import LLMResponse
from rag.pipeline import RAGConfig, RAGPipeline
from rag.vector_store import SQLiteVectorStore


class SyntheticEmbeddingModel:
    """Test-only embedder; not a replacement for the Stage 4 model."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        text = text.casefold()
        if any(word in text for word in ("deadline", "reimbursement", "14 days")):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ("mileage", "receipts")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, question):
        return np.asarray(self._vector(question), dtype=np.float32)


class SyntheticLLM:
    """Test-only fake response; it does not contact a provider."""

    def chat(
        self,
        messages,
        *,
        temperature=0.2,
        max_completion_tokens=512,
    ):
        return LLMResponse(
            content="Requests are due within 14 days. [S1]",
            model="synthetic-test-model",
        )


def make_synthetic_pdf() -> bytes:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Synthetic policy: reimbursement requests are due within 14 days.",
        )

        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Synthetic policy: mileage and receipts have separate rules.",
        )

        return document.tobytes()
    finally:
        document.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-pipeline-test-") as temp:
        store = SQLiteVectorStore(
            db_path=Path(temp) / "synthetic_vectors.sqlite3",
        )
        pipeline = RAGPipeline(
            embedding_model=SyntheticEmbeddingModel(),
            vector_store=store,
            llm_client=SyntheticLLM(),
            config=RAGConfig(top_k=2),
        )

        indexed = pipeline.index_pdf(
            make_synthetic_pdf(),
            "synthetic_policy.pdf",
        )
        answer = pipeline.answer(
            "When are reimbursement requests due?"
        )

        print(
            f"Indexed: {indexed.document_name} | "
            f"{indexed.page_count} pages | {indexed.chunk_count} chunks"
        )
        print(f"Answer: {answer.answer}")
        print("Sources:")
        for source in answer.sources:
            print(
                f"- [{source.source_id}] {source.document_name}, "
                f"page {source.page_number}: {source.excerpt}"
            )

        if not answer.sources or not answer.answer.strip():
            raise SystemExit("Synthetic RAG smoke test failed.")

        print(
            "Synthetic RAG pipeline smoke test completed. "
            "This is not a real-model quality result."
        )


if __name__ == "__main__":
    main()