import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.retrieval import RetrievalError, SemanticRetriever
from rag.vector_store import SQLiteVectorStore, VectorStoreError


class FakeQueryEmbedder:
    """Small deterministic fake; it does not download a model."""

    def __init__(self, vector=None):
        self.vector = np.asarray(
            vector if vector is not None else [1.0, 0.0],
            dtype=np.float32,
        )
        self.questions = []

    def embed_query(self, question: str):
        self.questions.append(question)
        return self.vector


class FakeBatchQueryEmbedder:
    def __init__(self):
        self.questions = []

    def embed_queries(self, questions):
        self.questions.extend(questions)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class DocumentOnlyEmbedder:
    """A document-only API must not be used for a query."""

    def embed_documents(self, texts):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def make_chunk(
    chunk_id,
    *,
    document_name="policy.pdf",
    page_number=1,
    text="Synthetic policy passage.",
    metadata=None,
):
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": 0,
        "text": text,
        "metadata": metadata or {},
    }


class SemanticRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteVectorStore(
            db_path=Path(self.temp_dir.name) / "test_vectors.sqlite3",
        )

        self.store.upsert(
            [
                make_chunk(
                    "best",
                    document_name="policy.pdf",
                    page_number=1,
                    text="Best matching synthetic passage.",
                    metadata={"department": "HR"},
                ),
                make_chunk(
                    "second",
                    document_name="policy.pdf",
                    page_number=2,
                    text="Second matching synthetic passage.",
                    metadata={"department": "HR"},
                ),
                make_chunk(
                    "other-document",
                    document_name="guide.pdf",
                    page_number=1,
                    text="A synthetic passage from another document.",
                    metadata={"department": "Finance"},
                ),
            ],
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.8, 0.6],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_retrieve_ranks_matching_chunk_first(self):
        embedder = FakeQueryEmbedder()
        retriever = SemanticRetriever(embedder, self.store)

        results = retriever.retrieve("What does the policy say?", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "best")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=6)
        self.assertEqual(
            embedder.questions,
            ["What does the policy say?"],
        )

    def test_top_k_uses_configured_default(self):
        retriever = SemanticRetriever(
            FakeQueryEmbedder(),
            self.store,
            default_top_k=1,
        )

        results = retriever.retrieve("Find a matching passage.")

        self.assertEqual(len(results), 1)

    def test_document_name_filter(self):
        retriever = SemanticRetriever(FakeQueryEmbedder(), self.store)

        results = retriever.retrieve(
            "Find a matching passage.",
            top_k=5,
            document_name="guide.pdf",
        )

        self.assertEqual([item.document_name for item in results], ["guide.pdf"])

    def test_metadata_filter(self):
        retriever = SemanticRetriever(FakeQueryEmbedder(), self.store)

        results = retriever.retrieve(
            "Find a matching passage.",
            top_k=5,
            where={"department": "Finance"},
        )

        self.assertEqual([item.chunk_id for item in results], ["other-document"])

    def test_empty_question_is_rejected(self):
        retriever = SemanticRetriever(FakeQueryEmbedder(), self.store)

        with self.assertRaises(RetrievalError):
            retriever.retrieve("   ")

    def test_invalid_top_k_is_rejected(self):
        retriever = SemanticRetriever(FakeQueryEmbedder(), self.store)

        for value in (0, -1, True, 1.5):
            with self.subTest(top_k=value):
                with self.assertRaises(RetrievalError):
                    retriever.retrieve("A question?", top_k=value)

    def test_query_dimension_mismatch_is_reported(self):
        retriever = SemanticRetriever(
            FakeQueryEmbedder([1.0, 0.0, 0.0]),
            self.store,
        )

        with self.assertRaises(VectorStoreError):
            retriever.retrieve("A question?")

    def test_supports_batch_query_embedding_method(self):
        embedder = FakeBatchQueryEmbedder()
        retriever = SemanticRetriever(embedder, self.store)

        results = retriever.retrieve("A batch query?")

        self.assertTrue(results)
        self.assertEqual(embedder.questions, ["A batch query?"])

    def test_does_not_use_document_only_embedding_method(self):
        retriever = SemanticRetriever(DocumentOnlyEmbedder(), self.store)

        with self.assertRaises(RetrievalError):
            retriever.retrieve("A question?")

    def test_empty_store_returns_no_results(self):
        empty_store = SQLiteVectorStore(
            db_path=Path(self.temp_dir.name) / "empty.sqlite3",
        )
        retriever = SemanticRetriever(FakeQueryEmbedder(), empty_store)

        results = retriever.retrieve("A question?")

        self.assertEqual(results, [])

    def test_search_alias_works(self):
        retriever = SemanticRetriever(FakeQueryEmbedder(), self.store)

        results = retriever.search("Find a matching passage.", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "best")


if __name__ == "__main__":
    unittest.main()