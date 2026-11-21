import unittest

import fitz

from rag.pdf_processor import (
    PDFProcessingError,
    ScannedOrEmptyPDFError,
    extract_pdf_pages,
)


def create_pdf(page_texts):
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


class PDFProcessorTests(unittest.TestCase):
    def test_extracts_page_text_and_one_based_page_numbers(self):
        pdf_bytes = create_pdf(
            [
                "Synthetic test text on page one.",
                "Synthetic test text on page two.",
            ]
        )

        pages = extract_pdf_pages(pdf_bytes, "synthetic.pdf")

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[0].document_name, "synthetic.pdf")
        self.assertIn("page one", pages[0].text)
        self.assertIn("page two", pages[1].text)

    def test_blank_page_is_kept_if_another_page_has_text(self):
        pdf_bytes = create_pdf(["Synthetic visible text.", ""])

        pages = extract_pdf_pages(pdf_bytes, "synthetic.pdf")

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].text.strip(), "")

    def test_rejects_empty_input(self):
        with self.assertRaises(PDFProcessingError):
            extract_pdf_pages(b"", "empty.pdf")

    def test_rejects_non_pdf_bytes(self):
        with self.assertRaises(PDFProcessingError):
            extract_pdf_pages(b"not a pdf", "invalid.pdf")

    def test_rejects_pdf_over_configured_size(self):
        pdf_bytes = create_pdf(["A small synthetic PDF."])

        with self.assertRaises(PDFProcessingError):
            extract_pdf_pages(
                pdf_bytes,
                "large.pdf",
                max_pdf_size_bytes=10,
            )

    def test_rejects_pdf_with_no_selectable_text(self):
        pdf_bytes = create_pdf(["", ""])

        with self.assertRaises(ScannedOrEmptyPDFError):
            extract_pdf_pages(pdf_bytes, "blank.pdf")

    def test_uses_only_the_uploaded_filename_not_a_path(self):
        pdf_bytes = create_pdf(["Synthetic path test."])

        pages = extract_pdf_pages(
            pdf_bytes,
            r"C:\private\uploads\synthetic.pdf",
        )

        self.assertEqual(pages[0].document_name, "synthetic.pdf")


if __name__ == "__main__":
    unittest.main()