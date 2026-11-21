import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.vector_store import SQLiteVectorStore, VectorStoreError


def make_chunk(
    chunk_id,
    *,
    document_name="policy.pdf",
    page_number=1,
    text="Synthetic policy text.",
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


class SQLiteVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteVectorStore(
            persist_directory=Path(self.temp_dir.name),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_returns_best_match_first(self):
        self.store.upsert(
            [
                make_chunk("best", text="Relevant synthetic passage."),
                make_chunk("other", page_number=2, text="Different passage."),
            ],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        results = self.store.search([1.0, 0.0], top_k=2)

        self.assertEqual(results[0].chunk_id, "best")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=6)
        self.assertAlmostEqual(results[0].distance, 0.0, places=6)
        self.assertEqual(results[0].page_number, 1)

    def test_search_normalizes_query_and_stored_vectors(self):
        self.store.upsert(
            [make_chunk("one")],
            np.asarray([[10.0, 0.0]], dtype=np.float32),
        )

        result = self.store.search([2.0, 0.0])[0]

        self.assertAlmostEqual(result.similarity, 1.0, places=6)

    def test_document_filter(self):
        self.store.upsert(
            [
                make_chunk("a", document_name="a.pdf"),
                make_chunk("b", document_name="b.pdf"),
            ],
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

        results = self.store.search([1.0, 0.0], document_name="b.pdf")

        self.assertEqual([result.chunk_id for result in results], ["b"])

    def test_metadata_filter(self):
        self.store.upsert(
            [
                make_chunk("hr", metadata={"department": "HR"}),
                make_chunk("finance", metadata={"department": "Finance"}),
            ],
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

        results = self.store.search(
            [1.0, 0.0],
            where={"department": "Finance"},
        )

        self.assertEqual([result.chunk_id for result in results], ["finance"])

    def test_data_persists_when_store_is_reopened(self):
        self.store.upsert(
            [make_chunk("saved")],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        reopened = SQLiteVectorStore(
            persist_directory=Path(self.temp_dir.name),
        )

        self.assertEqual(reopened.count(), 1)
        self.assertEqual(reopened.search([1.0, 0.0])[0].chunk_id, "saved")

    def test_upsert_updates_existing_id(self):
        self.store.upsert(
            [make_chunk("same-id", text="Old text.")],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        self.store.upsert(
            [make_chunk("same-id", text="New text.")],
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

        result = self.store.search([0.0, 1.0])[0]

        self.assertEqual(self.store.count(), 1)
        self.assertEqual(result.text, "New text.")

    def test_replace_document_keeps_other_documents(self):
        self.store.upsert(
            [
                make_chunk("old-a", document_name="a.pdf"),
                make_chunk("keep-b", document_name="b.pdf"),
            ],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        self.store.replace_document(
            "a.pdf",
            [
                make_chunk(
                    "new-a",
                    document_name="a.pdf",
                    text="Updated document.",
                )
            ],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        self.assertEqual(self.store.count(), 2)
        self.assertEqual(
            set(self.store.list_document_names()),
            {"a.pdf", "b.pdf"},
        )

    def test_delete_document(self):
        self.store.upsert(
            [
                make_chunk("a", document_name="a.pdf"),
                make_chunk("b", document_name="b.pdf"),
            ],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        deleted = self.store.delete_document("a.pdf")

        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.list_document_names(), ["b.pdf"])

    def test_mixed_embedding_dimensions_are_rejected(self):
        self.store.upsert(
            [make_chunk("first")],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        with self.assertRaises(VectorStoreError):
            self.store.upsert(
                [make_chunk("second")],
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            )

    def test_zero_vector_is_rejected(self):
        with self.assertRaises(VectorStoreError):
            self.store.upsert(
                [make_chunk("zero")],
                np.asarray([[0.0, 0.0]], dtype=np.float32),
            )

    def test_page_number_must_be_one_based(self):
        with self.assertRaises(VectorStoreError):
            self.store.upsert(
                [make_chunk("bad-page", page_number=0)],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

    def test_duplicate_ids_in_one_batch_are_rejected(self):
        with self.assertRaises(VectorStoreError):
            self.store.upsert(
                [
                    make_chunk("same"),
                    make_chunk("same", page_number=2),
                ],
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()