"""Shared data models for page-level PDF text and text chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one PDF page. Page numbers are one-based."""

    document_name: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_name, str):
            raise TypeError("document_name must be a string.")

        normalized_name = self.document_name.strip()
        if not normalized_name:
            raise ValueError("document_name cannot be empty.")

        object.__setattr__(self, "document_name", normalized_name)

        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
            raise TypeError("page_number must be an integer.")

        if self.page_number < 1:
            raise ValueError("page_number must be 1 or greater.")

        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One text chunk and its page-level source metadata."""

    chunk_id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int

    @property
    def metadata(self) -> dict[str, str | int]:
        """Return source metadata in a format suitable for vector storage."""
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }

    def to_record(self) -> dict[str, object]:
        """Return the chunk and metadata as a plain dictionary."""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }