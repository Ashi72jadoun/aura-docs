"""Text cleaning and fixed-size chunking utilities."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import re
import unicodedata

from rag.models import PageText, TextChunk


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Normalize common PDF text noise while preserving paragraph breaks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\x00", "").replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse repeated spaces and tabs within each line.
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in normalized.split("\n")
    ]

    cleaned = "\n".join(lines)

    # Keep paragraph breaks, but remove excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    """Validate the character size and overlap configuration."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def _make_chunk_id(
    page: PageText,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """Create a repeatable ID from a page and its chunk content."""
    id_source = (
        f"{page.document_name}\0"
        f"{page.page_number}\0"
        f"{chunk_index}\0"
        f"{chunk_text}"
    )
    return hashlib.sha256(id_source.encode("utf-8")).hexdigest()


def chunk_page_fixed(
    page: PageText,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Clean and split one page into fixed-size character chunks."""
    if not isinstance(page, PageText):
        raise TypeError("page must be a PageText instance.")

    _validate_chunk_settings(chunk_size, overlap)

    cleaned_text = clean_text(page.text)
    if not cleaned_text:
        return []

    stride = chunk_size - overlap
    chunks: list[TextChunk] = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk_text = cleaned_text[start:end].strip()

        if chunk_text:
            chunk_index = len(chunks)
            chunk_id = _make_chunk_id(page, chunk_index, chunk_text)

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )

        if end == len(cleaned_text):
            break

        start += stride

    return chunks


def chunk_pages(
    pages: Iterable[PageText],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Clean and chunk multiple pages while preserving each page's metadata."""
    _validate_chunk_settings(chunk_size, overlap)

    chunks: list[TextChunk] = []

    for page in pages:
        if not isinstance(page, PageText):
            raise TypeError("Every item in pages must be a PageText instance.")

        chunks.extend(
            chunk_page_fixed(
                page,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks