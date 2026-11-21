"""Tests for text cleaning and page-aware chunking."""

import unittest

from rag.chunking import clean_text, chunk_page_fixed, chunk_pages
from rag.models import PageText


class TextCleaningTests(unittest.TestCase):
    def test_clean_text_normalizes_spacing_and_line_endings(self) -> None:
        raw_text = (
            "\ufeffPolicy  title\r\n"
            "\r\n"
            "\r\n"
            "Line\twith   spaces\x00"
        )

        self.assertEqual(
            clean_text(raw_text),
            "Policy title\n\nLine with spaces",
        )


class ChunkingTests(unittest.TestCase):
    def test_fixed_chunks_keep_overlap_and_metadata(self) -> None:
        page = PageText(
            document_name="policy.pdf",
            page_number=4,
            text="0123456789abcdefghij",
        )

        chunks = chunk_page_fixed(page, chunk_size=10, overlap=3)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["0123456789", "789abcdefg", "efghij"],
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            [0, 1, 2],
        )
        self.assertEqual(
            chunks[0].metadata,
            {
                "document_name": "policy.pdf",
                "page_number": 4,
                "chunk_index": 0,
            },
        )

        record = chunks[0].to_record()
        self.assertEqual(record["id"], chunks[0].chunk_id)
        self.assertEqual(record["text"], chunks[0].text)

    def test_chunk_ids_are_deterministic(self) -> None:
        page = PageText(
            document_name="policy.pdf",
            page_number=1,
            text="Repeatable sample text.",
        )

        first_run = chunk_page_fixed(page, chunk_size=10, overlap=2)
        second_run = chunk_page_fixed(page, chunk_size=10, overlap=2)

        self.assertEqual(
            [chunk.chunk_id for chunk in first_run],
            [chunk.chunk_id for chunk in second_run],
        )

    def test_empty_page_returns_no_chunks(self) -> None:
        page = PageText(
            document_name="empty.pdf",
            page_number=1,
            text=" \t\r\n ",
        )

        self.assertEqual(chunk_page_fixed(page), [])

    def test_multiple_pages_keep_their_source_metadata(self) -> None:
        pages = [
            PageText("manual.pdf", 1, "A" * 12),
            PageText("manual.pdf", 2, "B" * 12),
        ]

        chunks = chunk_pages(pages, chunk_size=10, overlap=2)

        self.assertEqual(
            [chunk.page_number for chunk in chunks],
            [1, 1, 2, 2],
        )
        self.assertEqual(
            [chunk.document_name for chunk in chunks],
            ["manual.pdf", "manual.pdf", "manual.pdf", "manual.pdf"],
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            [0, 1, 0, 1],
        )

    def test_invalid_chunk_settings_raise_value_error(self) -> None:
        page = PageText("policy.pdf", 1, "Some sample text.")

        invalid_settings = [
            (0, 0),
            (10, -1),
            (10, 10),
        ]

        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_page_fixed(
                        page,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )

    def test_page_number_must_start_at_one(self) -> None:
        with self.assertRaises(ValueError):
            PageText(
                document_name="policy.pdf",
                page_number=0,
                text="Some text.",
            )


if __name__ == "__main__":
    unittest.main()