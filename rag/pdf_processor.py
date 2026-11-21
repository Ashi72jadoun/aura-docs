"""Page-wise PDF text extraction using PyMuPDF; OCR is not included."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from rag.models import PageText


DEFAULT_MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be safely read or processed."""


class ScannedOrEmptyPDFError(PDFProcessingError):
    """Raised when a PDF contains no selectable text."""


def _safe_document_name(filename: str) -> str:
    """Keep only the final filename; never use an uploaded path."""
    normalized = filename.replace("\\", "/").strip()
    name = PurePosixPath(normalized).name
    return name or "uploaded.pdf"


def extract_pdf_pages(
    pdf_bytes: bytes,
    document_name: str,
    *,
    max_pdf_size_bytes: int = DEFAULT_MAX_PDF_SIZE_BYTES,
) -> list[PageText]:
    """Extract text from every PDF page and retain 1-based page numbers.

    Blank pages are retained in the returned list. The function raises
    ScannedOrEmptyPDFError if no page has selectable text. It does not run OCR.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
        raise PDFProcessingError("PDF data must be bytes.")
    payload = bytes(pdf_bytes)

    if not payload:
        raise PDFProcessingError("The uploaded PDF is empty.")

    if (
        type(max_pdf_size_bytes) is not int
        or max_pdf_size_bytes < 1
    ):
        raise PDFProcessingError(
            "max_pdf_size_bytes must be a positive integer."
        )

    if len(payload) > max_pdf_size_bytes:
        limit_mib = max_pdf_size_bytes / (1024 * 1024)
        raise PDFProcessingError(
            f"The PDF exceeds the configured size limit of {limit_mib:.1f} MiB."
        )

    if not isinstance(document_name, str) or not document_name.strip():
        raise PDFProcessingError("A non-empty PDF filename is required.")

    safe_name = _safe_document_name(document_name)

    try:
        import fitz
    except ImportError as exc:
        raise PDFProcessingError(
            "PyMuPDF is not installed. Install it with: "
            'python -m pip install "PyMuPDF>=1.24,<2.0"'
        ) from exc

    pages: list[PageText] = []

    try:
        with fitz.open(stream=payload, filetype="pdf") as document:
            if document.needs_pass:
                raise PDFProcessingError(
                    "This PDF is password-protected. A password handler is not "
                    "configured."
                )

            if document.page_count < 1:
                raise PDFProcessingError("The PDF contains no pages.")

            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_text = page.get_text("text")
                if not isinstance(page_text, str):
                    page_text = str(page_text)

                pages.append(
                    PageText(
                        document_name=safe_name,
                        page_number=page_index + 1,
                        text=page_text,
                    )
                )
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "The PDF could not be opened or its page text could not be extracted. "
            "Check that it is a valid, non-corrupt PDF."
        ) from exc

    if not any(page.text.strip() for page in pages):
        raise ScannedOrEmptyPDFError(
            "This PDF has no selectable text. It may be scanned or image-only; "
            "OCR is not included in this project."
        )

    return pages


# A short alias for callers that prefer a smaller function name.
extract_pages = extract_pdf_pages