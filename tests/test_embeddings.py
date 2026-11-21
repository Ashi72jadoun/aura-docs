"""Unit tests for the embedding module, using a fake model."""

import unittest

import numpy as np

from rag.embeddings import SentenceTransformerEmbedder
from rag.models import TextChunk


class FakeSentenceTransformer:
    """Small stand-in for SentenceTransformer that does not download a model."""

    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls += 1
        self.sentences = list(sentences)

        vectors = np.array(
            [
                [1.0, float(len(sentence)), float(sentence.count("a") + 1)]
                for sentence in sentences
            ],
            dtype=np.float32,
        )

        if kwargs.get("normalize_embeddings"):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

        return vectors


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_model = FakeSentenceTransformer()
        self.embedder = SentenceTransformerEmbedder(model=self.fake_model)

    def test_document_embeddings_use_passage_prefix(self) -> None:
        chunk = TextChunk(
            chunk_id="test-chunk-1",
            text="Paid sick leave rules.",
            document_name="handbook.pdf",
            page_number=2,
            chunk_index=0,
        )

        vectors = self.embedder.embed_chunks([chunk])

        self.assertEqual(vectors.shape, (1, 3))
        self.assertEqual(
            self.fake_model.sentences,
            ["passage: Paid sick leave rules."],
        )
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1),
            [1.0],
            atol=1e-6,
        )

    def test_query_embedding_uses_query_prefix(self) -> None:
        vector = self.embedder.embed_query(" What is the leave policy? ")

        self.assertEqual(vector.shape, (3,))
        self.assertEqual(
            self.fake_model.sentences,
            ["query: What is the leave policy?"],
        )
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_empty_document_list_returns_correct_empty_shape(self) -> None:
        vectors = self.embedder.embed_documents([])

        self.assertEqual(vectors.shape, (0, 3))
        self.assertEqual(self.fake_model.calls, 0)

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.embedder.embed_documents(["   "])

        with self.assertRaises(ValueError):
            self.embedder.embed_query(" ")

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.embedder.embed_documents(["text"], batch_size=0)

    def test_non_string_document_text_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.embedder.embed_documents(["valid text", 42])  # type: ignore[list-item]

    def test_a_single_string_is_not_treated_as_many_texts(self) -> None:
        with self.assertRaises(TypeError):
            self.embedder.embed_documents("this is not a list of texts")


if __name__ == "__main__":
    unittest.main()