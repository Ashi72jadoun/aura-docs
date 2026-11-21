import json
from pathlib import Path
import tempfile
import unittest

from rag.evaluation import (
    EvaluationDataError,
    evaluate_retrieval,
    load_evaluation_cases,
)


class FakeRetrievalPipeline:
    """Synthetic pipeline for testing metric calculations only."""

    def __init__(self, results_by_question):
        self.results_by_question = results_by_question
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.results_by_question.get(question, [])[:top_k]


class EvaluationTests(unittest.TestCase):
    def test_precision_recall_mrr_and_hit_rate(self):
        case = {
            "id": "synthetic-1",
            "language": "en",
            "question": "Synthetic question",
            "relevant_sources": [
                {"document_name": "guide.pdf", "page_number": 2},
                {"document_name": "guide.pdf", "page_number": 3},
            ],
        }

        pipeline = FakeRetrievalPipeline(
            {
                "Synthetic question": [
                    {
                        "document_name": "guide.pdf",
                        "page_number": 9,
                    },
                    {
                        "document_name": "guide.pdf",
                        "page_number": 3,
                    },
                ]
            }
        )

        result = evaluate_retrieval(pipeline, [case], top_k=2)
        per_question = result["per_question"][0]

        self.assertEqual(result["evaluation_cases"], 1)
        self.assertEqual(result["top_k"], 2)
        self.assertEqual(per_question["precision_at_k"], 0.5)
        self.assertEqual(per_question["recall_at_k"], 0.5)
        self.assertEqual(per_question["reciprocal_rank_at_k"], 0.5)
        self.assertTrue(per_question["hit_at_k"])

    def test_duplicate_chunks_from_same_page_count_once(self):
        case = {
            "id": "synthetic-duplicate",
            "question": "Another synthetic question",
            "relevant_sources": [
                {"document_name": "guide.pdf", "page_number": 2}
            ],
        }

        same_page_chunk_1 = {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk": "Synthetic chunk one",
        }
        same_page_chunk_2 = {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk": "Synthetic chunk two",
        }

        pipeline = FakeRetrievalPipeline(
            {
                "Another synthetic question": [
                    same_page_chunk_1,
                    same_page_chunk_2,
                ]
            }
        )

        result = evaluate_retrieval(pipeline, [case], top_k=2)
        per_question = result["per_question"][0]

        self.assertEqual(per_question["retrieved_unique_page_count"], 1)
        self.assertEqual(per_question["precision_at_k"], 1.0)
        self.assertEqual(per_question["recall_at_k"], 1.0)

    def test_filename_paths_and_case_are_normalized(self):
        case = {
            "id": "synthetic-path",
            "question": "Path normalization question",
            "relevant_sources": [
                {
                    "document_name": r"C:\PrivatePDFs\Guide.PDF",
                    "page_number": 1,
                }
            ],
        }

        pipeline = FakeRetrievalPipeline(
            {
                "Path normalization question": [
                    {
                        "document_name": "guide.pdf",
                        "page_number": 1,
                    }
                ]
            }
        )

        result = evaluate_retrieval(pipeline, [case], top_k=1)

        self.assertTrue(result["per_question"][0]["hit_at_k"])

    def test_missing_page_metadata_is_reported(self):
        case = {
            "id": "synthetic-missing-page",
            "question": "Page metadata question",
            "relevant_sources": [
                {"document_name": "guide.pdf", "page_number": 2}
            ],
        }

        pipeline = FakeRetrievalPipeline(
            {
                "Page metadata question": [
                    {
                        "document_name": "guide.pdf",
                        "chunk": "Synthetic text without a page number",
                    }
                ]
            }
        )

        result = evaluate_retrieval(pipeline, [case], top_k=1)
        per_question = result["per_question"][0]

        self.assertEqual(per_question["missing_page_metadata_count"], 1)
        self.assertFalse(per_question["hit_at_k"])
        self.assertEqual(result["missing_page_metadata_total"], 1)

    def test_missing_relevant_sources_are_rejected(self):
        case = {
            "id": "synthetic-invalid",
            "question": "Invalid synthetic case",
            "relevant_sources": [],
        }

        pipeline = FakeRetrievalPipeline({})

        with self.assertRaises(EvaluationDataError):
            evaluate_retrieval(pipeline, [case], top_k=1)

    def test_empty_dataset_is_rejected(self):
        pipeline = FakeRetrievalPipeline({})

        with self.assertRaises(EvaluationDataError):
            evaluate_retrieval(pipeline, [], top_k=1)

    def test_top_k_must_be_positive(self):
        case = {
            "id": "synthetic-top-k",
            "question": "Top k question",
            "relevant_sources": [
                {"document_name": "guide.pdf", "page_number": 1}
            ],
        }

        pipeline = FakeRetrievalPipeline({})

        with self.assertRaises(ValueError):
            evaluate_retrieval(pipeline, [case], top_k=0)

    def test_load_jsonl_dataset(self):
        rows = [
            {
                "id": "synthetic-load",
                "language": "hi",
                "question": "Synthetic Hindi question",
                "relevant_sources": [
                    {"document_name": "guide.pdf", "page_number": 4}
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "questions.jsonl"
            file_path.write_text(
                json.dumps(rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            cases = load_evaluation_cases(file_path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["language"], "hi")
        self.assertEqual(cases[0]["id"], "synthetic-load")


if __name__ == "__main__":
    unittest.main()