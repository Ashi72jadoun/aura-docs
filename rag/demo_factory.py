"""Synthetic retrieval data for demonstrating the evaluation runner.

This is NOT the real embedding model or PDF retriever. It returns fixed,
pre-written results so the evaluation workflow can be demonstrated offline.
"""

from __future__ import annotations


HINDI_OPEN_QUESTION = (
    "\u092a\u0941\u0938\u094d\u0924\u0915\u093e\u0932\u092f "
    "\u0915\u093f\u0938 "
    "\u0938\u092e\u092f "
    "\u0916\u0941\u0932\u0924\u093e "
    "\u0939\u0948?"
)


class DemoRetrievalPipeline:
    """Return predetermined synthetic pages for three demo questions."""

    def __init__(self) -> None:
        page_1 = {
            "document_name": "demo_guide.pdf",
            "page_number": 1,
            "text": "Synthetic demo page: the library opens at 9 a.m.",
        }
        page_2 = {
            "document_name": "demo_guide.pdf",
            "page_number": 2,
            "text": "Synthetic demo page: the library is closed on Sundays.",
        }
        page_3 = {
            "document_name": "demo_guide.pdf",
            "page_number": 3,
            "text": "Synthetic demo page: library membership is free.",
        }

        self._results = {
            "at what time does the demo library open?": [page_1, page_2],
            "which day is the demo library closed?": [page_3, page_2],
            HINDI_OPEN_QUESTION.casefold(): [page_2, page_1],
        }

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        key = " ".join(str(question).strip().split()).casefold()
        results = self._results.get(key, [])
        return [dict(result) for result in results[:top_k]]


def create_pipeline() -> DemoRetrievalPipeline:
    """Factory used by the evaluation command."""
    return DemoRetrievalPipeline()