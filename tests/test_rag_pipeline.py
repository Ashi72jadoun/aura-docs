import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np

from rag.llm import LLMResponse
from rag.pipeline import (
    DocumentAlreadyIndexedError,
    NO_RETRIEVAL_ANSWER,
    RAGConfig,
    RAGPipeline,
    RAGPipelineError,
)
from rag.vector_store import SQLiteVectorStore


def create_pdf(page_texts):
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


class FakeEmbeddingModel:
    """Deterministic test model; it downloads no real embedding model."""

    def __init__(self):
        self.document_texts = []
        self.queries = []

    @staticmethod
    def vector_for(text):
        normalized = text.casefold()
        if any(
            word in normalized
            for word in ("reimbursement", "deadline", "14 days", "due")
        ):
            return [1.0, 0.0, 0.0]
        if any(word in normalized for word in ("mileage", "receipt")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_texts.extend(texts)
        return np.asarray(
            [self.vector_for(text) for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, question):
        self.queries.append(question)
        return np.asarray(self.vector_for(question), dtype=np.float32)


class FakeLLM:
    def __init__(self, response_text="Requests are due within 14 days. [S1]"):
        self.response_text = response_text
        self.calls = []

    def chat(
        self,
        messages,
        *,
        temperature=0.2,
        max_completion_tokens=512,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        return LLMResponse(
            content=self.response_text,
            model="synthetic-test-model",
        )


class RAGPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "vectors.sqlite3"
        self.store = SQLiteVectorStore(db_path=self.database_path)
        self.embedder = FakeEmbeddingModel()
        self.llm = FakeLLM()
        self.config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=100,
            top_k=2,
            max_context_chars=2000,
            max_question_chars=1000,
            max_completion_tokens=64,
        )
        self.pipeline = RAGPipeline(
            embedding_model=self.embedder,
            vector_store=self.store,
            llm_client=self.llm,
            config=self.config,
        )
        self.pdf_bytes = create_pdf(
            [
                "Synthetic policy: reimbursement requests are due within "
                "14 days.",
                "Synthetic policy: mileage and receipts have separate rules.",
            ]
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_index_pdf_stores_page_metadata_and_chunks(self):
        result = self.pipeline.index_pdf(
            self.pdf_bytes,
            "synthetic_policy.pdf",
        )

        self.assertEqual(result.document_name, "synthetic_policy.pdf")
        self.assertEqual(result.page_count, 2)
        self.assertGreaterEqual(result.chunk_count, 2)
        self.assertEqual(self.store.count(), result.chunk_count)
        self.assertIn("synthetic_policy.pdf", self.store.list_document_names())
        self.assertTrue(self.embedder.document_texts)

    def test_duplicate_filename_is_rejected_unless_replacing(self):
        self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")

        with self.assertRaises(DocumentAlreadyIndexedError):
            self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")

        replacement = self.pipeline.index_pdf(
            self.pdf_bytes,
            "synthetic_policy.pdf",
            replace_existing=True,
        )

        self.assertEqual(
            replacement.chunk_count,
            self.store.count(),
        )

    def test_answer_uses_retrieved_source_and_calls_fake_llm(self):
        self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")

        answer = self.pipeline.answer(
            "When are reimbursement requests due?"
        )

        self.assertEqual(
            answer.answer,
            "Requests are due within 14 days. [S1]",
        )
        self.assertTrue(answer.llm_used)
        self.assertFalse(answer.abstained)
        self.assertEqual(answer.cited_source_ids, ("S1",))
        self.assertTrue(answer.sources)
        self.assertEqual(answer.sources[0].source_id, "S1")
        self.assertEqual(
            answer.sources[0].document_name,
            "synthetic_policy.pdf",
        )
        self.assertEqual(answer.sources[0].page_number, 1)
        self.assertEqual(len(self.llm.calls), 1)

    def test_prompt_includes_question_context_and_citation_instructions(self):
        self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")
        self.pipeline.answer("When are reimbursement requests due?")

        messages = self.llm.calls[0]["messages"]
        combined_content = "\n".join(message["content"] for message in messages)

        self.assertIn("When are reimbursement requests due?", combined_content)
        self.assertIn("Synthetic policy", combined_content)
        self.assertIn("source IDs", messages[0]["content"])

    def test_empty_store_abstains_without_calling_llm(self):
        empty_store = SQLiteVectorStore(
            db_path=Path(self.temp_dir.name) / "empty.sqlite3",
        )
        pipeline = RAGPipeline(
            embedding_model=self.embedder,
            vector_store=empty_store,
            llm_client=self.llm,
            config=self.config,
        )

        answer = pipeline.answer("An unrelated synthetic question?")

        self.assertEqual(answer.answer, NO_RETRIEVAL_ANSWER)
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.abstention_reason, "no_retrieved_passages")
        self.assertFalse(answer.llm_used)
        self.assertEqual(self.llm.calls, [])

    def test_missing_llm_is_only_an_error_when_answer_generation_is_needed(self):
        self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")
        pipeline_without_llm = RAGPipeline(
            embedding_model=self.embedder,
            vector_store=self.store,
            llm_client=None,
            config=self.config,
        )

        with self.assertRaises(RAGPipelineError):
            pipeline_without_llm.answer(
                "When are reimbursement requests due?"
            )

    def test_question_length_is_limited(self):
        short_config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=100,
            top_k=1,
            max_question_chars=10,
        )
        pipeline = RAGPipeline(
            embedding_model=self.embedder,
            vector_store=self.store,
            llm_client=self.llm,
            config=short_config,
        )

        with self.assertRaises(RAGPipelineError):
            pipeline.answer("A question longer than ten characters.")

    def test_ask_is_an_alias_for_answer(self):
        self.pipeline.index_pdf(self.pdf_bytes, "synthetic_policy.pdf")

        answer = self.pipeline.ask("When are reimbursement requests due?")

        self.assertTrue(answer.sources)
        self.assertTrue(answer.llm_used)

    def test_indexing_invalid_pdf_does_not_add_chunks(self):
        with self.assertRaises(Exception):
            self.pipeline.index_pdf(b"not a pdf", "invalid.pdf")

        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()