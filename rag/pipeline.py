"""PDF RAG pipeline: PDF extraction, chunking, SQLite retrieval and answers."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag.chunking import chunk_pages
from rag.pdf_processor import extract_pdf_pages
from rag.reliability import (
    build_rag_messages,
    llm_response_text,
    validate_rag_answer,
)
from rag.retrieval import SemanticRetriever


@dataclass
class RAGConfig:
    """Settings for PDF indexing and question answering."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    max_context_chars: int = 12000
    max_question_chars: int = 4000
    max_completion_tokens: int = 512
    max_pdf_size_bytes: int = 50 * 1024 * 1024
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be at least zero and less than chunk_size."
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")
        if self.max_question_chars <= 0:
            raise ValueError("max_question_chars must be greater than zero.")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be greater than zero.")
        if self.max_pdf_size_bytes <= 0:
            raise ValueError("max_pdf_size_bytes must be greater than zero.")


@dataclass
class SourceCitation:
    """A retrieved PDF passage and its citation metadata."""

    source_id: str
    chunk: str
    excerpt: str
    document_name: str
    page_number: int | None
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """Answer, sources and citation-validation status."""

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    cited_source_ids: list[str] = field(default_factory=list)
    llm_used: bool = False
    abstained: bool = False
    abstention_reason: str | None = None


_MISSING = object()


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a mapping or object."""
    for name in names:
        if isinstance(obj, Mapping):
            result = obj.get(name, _MISSING)
        else:
            result = getattr(obj, name, _MISSING)

        if result is not _MISSING and result is not None:
            return result

    return default


def _extract_text(value: Any, depth: int = 0) -> str:
    """Extract text from a string, mapping, or chunk-like object."""
    if isinstance(value, str):
        return value.strip()

    if depth >= 4:
        return ""

    for name in (
        "text",
        "page_content",
        "content",
        "chunk",
        "document",
        "excerpt",
    ):
        nested = _value(value, name, default=_MISSING)

        if nested is _MISSING or nested is value:
            continue

        if isinstance(nested, str) and nested.strip():
            return nested.strip()

        nested_text = _extract_text(nested, depth + 1)
        if nested_text:
            return nested_text

    return ""


def _extract_metadata(value: Any) -> dict[str, Any]:
    """Return metadata as a plain dictionary."""
    raw_metadata = _value(value, "metadata", default={})

    if isinstance(raw_metadata, Mapping):
        return dict(raw_metadata)

    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            return {}

        if isinstance(parsed, Mapping):
            return dict(parsed)

    return {}


def _to_page_number(value: Any) -> int | None:
    """Convert a page value to a positive integer when possible."""
    if value is None or isinstance(value, bool):
        return None

    try:
        converted = int(value)
        return converted if converted >= 1 else None
    except (TypeError, ValueError):
        pass

    try:
        number = float(value)
        if number.is_integer() and number >= 1:
            return int(number)
    except (TypeError, ValueError):
        pass

    match = re.search(r"\d+", str(value))
    if match:
        try:
            converted = int(match.group(0))
            return converted if converted >= 1 else None
        except ValueError:
            return None

    return None


def _page_number(value: Any, metadata: Mapping[str, Any]) -> int | None:
    """Find a one-based page number in a chunk or its metadata."""
    for name in ("page_number", "page", "page_num", "page_no"):
        result = metadata.get(name)
        if result is None:
            result = _value(value, name)

        converted = _to_page_number(result)
        if converted is not None:
            return converted

    page_index = metadata.get("page_index")
    if page_index is None:
        page_index = _value(value, "page_index")

    if page_index is not None and not isinstance(page_index, bool):
        try:
            index_number = int(page_index)
            if index_number >= 0:
                return index_number + 1
        except (TypeError, ValueError):
            pass

    return None


def _clean_document_name(filename: str | None) -> str:
    """Keep a filename, not a local directory path."""
    raw_name = str(filename or "").strip().replace("\\", "/")
    clean_name = raw_name.rsplit("/", 1)[-1].strip()
    return clean_name or "uploaded.pdf"


def _call_pdf_extractor(
    pdf_bytes: bytes,
    document_name: str,
    max_pdf_size_bytes: int,
) -> list[Any]:
    """Extract PDF pages, including the required document name."""
    try:
        parameters = inspect.signature(extract_pdf_pages).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs: dict[str, Any] = {}

    size_parameter_names = (
        "max_pdf_size_bytes",
        "max_size_bytes",
        "max_file_size_bytes",
    )

    size_parameter_found = False

    for name in size_parameter_names:
        parameter = parameters.get(name)
        if parameter is not None:
            size_parameter_found = True
            if parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
                kwargs[name] = max_pdf_size_bytes
            break

    if not size_parameter_found:
        accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_extra_kwargs:
            kwargs["max_pdf_size_bytes"] = max_pdf_size_bytes

    document_parameter = None
    for name in ("document_name", "filename", "source_name"):
        if name in parameters:
            document_parameter = parameters[name]
            break

    if (
        document_parameter is not None
        and document_parameter.kind == inspect.Parameter.KEYWORD_ONLY
    ):
        kwargs[document_parameter.name] = document_name
        result = extract_pdf_pages(pdf_bytes, **kwargs)
    else:
        # document_name is required by this project's PDF processor.
        result = extract_pdf_pages(pdf_bytes, document_name, **kwargs)

    if isinstance(result, Mapping) and "pages" in result:
        result = result["pages"]

    if result is None or isinstance(result, (str, bytes)):
        raise ValueError("The PDF extractor did not return a list of pages.")

    try:
        return list(result)
    except TypeError as exc:
        raise ValueError(
            "The PDF extractor did not return iterable pages."
        ) from exc


def _call_chunker(
    pages: list[Any],
    config: RAGConfig,
    document_name: str,
) -> list[Any]:
    """Call chunk_pages with the argument names it supports."""
    try:
        parameters = inspect.signature(chunk_pages).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: dict[str, Any] = {}

    if "chunk_size" in parameters or accepts_extra_kwargs:
        kwargs["chunk_size"] = config.chunk_size

    if "chunk_overlap" in parameters or accepts_extra_kwargs:
        kwargs["chunk_overlap"] = config.chunk_overlap
    elif "overlap" in parameters:
        kwargs["overlap"] = config.chunk_overlap
    elif "overlap_size" in parameters:
        kwargs["overlap_size"] = config.chunk_overlap

    if "document_name" in parameters or accepts_extra_kwargs:
        kwargs["document_name"] = document_name
    elif "source_name" in parameters:
        kwargs["source_name"] = document_name
    elif "filename" in parameters:
        kwargs["filename"] = document_name
    elif "source" in parameters:
        kwargs["source"] = document_name

    result = chunk_pages(pages, **kwargs)

    if isinstance(result, Mapping) and "chunks" in result:
        result = result["chunks"]

    if result is None or isinstance(result, (str, bytes)):
        raise ValueError("The chunker did not return a list of chunks.")

    try:
        return list(result)
    except TypeError as exc:
        raise ValueError("The chunker did not return iterable chunks.") from exc


def _make_chunk_id(
    document_name: str,
    chunk_index: int,
    page_number: int | None,
    text: str,
) -> str:
    """Create a stable, non-empty ID for a document chunk."""
    identity = json.dumps(
        {
            "document_name": document_name.casefold(),
            "chunk_index": chunk_index,
            "page_number": page_number,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _normalise_chunks(
    raw_chunks: list[Any],
    document_name: str,
) -> list[dict[str, Any]]:
    """Add text, source metadata and required chunk IDs."""
    normalised: list[dict[str, Any]] = []

    for chunk_index, raw_chunk in enumerate(raw_chunks):
        text = _extract_text(raw_chunk)
        if not text:
            continue

        metadata = _extract_metadata(raw_chunk)
        page = _page_number(raw_chunk, metadata)

        chunk_id = _make_chunk_id(
            document_name=document_name,
            chunk_index=chunk_index,
            page_number=page,
            text=text,
        )

        metadata["document_name"] = document_name
        metadata["source"] = document_name
        metadata["chunk_id"] = chunk_id
        metadata["chunk_index"] = chunk_index

        if page is not None:
            metadata["page_number"] = page

        record: dict[str, Any] = {
            "chunk_id": chunk_id,
            "id": chunk_id,
            "text": text,
            "page_content": text,
            "document_name": document_name,
            "source": document_name,
            "metadata": metadata,
        }

        if page is not None:
            record["page_number"] = page

        normalised.append(record)

    return normalised


def _embedding_matrix(result: Any, expected_count: int) -> np.ndarray:
    """Convert embedding output to a finite float32 matrix."""
    if isinstance(result, Mapping):
        result = _value(result, "embeddings", "vectors", default=None)

    if result is None:
        raise ValueError("The embedding model returned no embeddings.")

    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "cpu") and hasattr(result, "numpy"):
        result = result.cpu().numpy()

    try:
        matrix = np.asarray(result, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding output could not be converted to numbers."
        ) from exc

    if matrix.ndim == 1 and expected_count == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix; got shape {matrix.shape}."
        )

    if matrix.shape[0] != expected_count:
        raise ValueError(
            "The embedding model returned a different number of vectors "
            f"({matrix.shape[0]}) than chunks ({expected_count})."
        )

    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embeddings are empty or contain invalid numbers.")

    return np.ascontiguousarray(matrix, dtype=np.float32)


def _search_hits(result: Any) -> list[Any]:
    """Unwrap common retrieval results into a list of hits."""
    if result is None:
        return []

    if isinstance(result, Mapping):
        for name in ("results", "matches", "hits", "items", "retrieved_chunks"):
            items = result.get(name)
            if isinstance(items, (list, tuple)):
                return list(items)

        return [result]

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], (int, float, np.number))
    ):
        return [result]

    if isinstance(result, (str, bytes)):
        return []

    try:
        return list(result)
    except TypeError:
        return [result]


def _safe_float(value: Any) -> float | None:
    """Convert a similarity value to a finite float."""
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    return converted if np.isfinite(converted) else None


class RAGPipeline:
    """Index PDFs locally and answer questions from retrieved passages."""

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        llm_client: Any | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model is required.")
        if vector_store is None:
            raise ValueError("vector_store is required.")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or RAGConfig()

        self.retriever = SemanticRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            default_top_k=self.config.top_k,
        )

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed PDF passages using the configured local model."""
        method_names = (
            "embed_documents",
            "encode_documents",
            "embed_passages",
            "embed_texts",
            "encode_texts",
            "encode",
        )

        for method_name in method_names:
            method = getattr(self.embedding_model, method_name, None)
            if not callable(method):
                continue

            inputs = texts
            if method_name == "encode":
                inputs = [f"passage: {text}" for text in texts]

            return _embedding_matrix(method(inputs), len(texts))

        raise AttributeError(
            "The embedding model must provide one of: "
            "embed_documents, encode_documents, embed_passages, "
            "embed_texts, encode_texts, or encode."
        )

    def _existing_document_names(self) -> list[str]:
        """Get indexed names if the vector store supports listing."""
        method = getattr(self.vector_store, "list_document_names", None)
        if not callable(method):
            return []

        result = method() or []

        if isinstance(result, Mapping):
            result = result.get(
                "document_names",
                result.get("names", list(result.keys())),
            )

        if isinstance(result, (str, bytes)):
            result = [result]

        try:
            items = list(result)
        except TypeError:
            return []

        names: list[str] = []

        for item in items:
            if isinstance(item, str):
                names.append(item)
                continue

            name = _value(
                item,
                "document_name",
                "filename",
                "name",
                default=None,
            )
            if name is not None:
                names.append(str(name))

        return names

    def _save_chunks(
        self,
        document_name: str,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        replace_existing: bool,
    ) -> None:
        """Save chunks, replacing the document when requested."""
        if not replace_existing:
            self.vector_store.upsert(chunks, embeddings)
            return

        replace_method = getattr(self.vector_store, "replace_document", None)
        if callable(replace_method):
            replace_method(document_name, chunks, embeddings)
            return

        delete_method = getattr(self.vector_store, "delete_document", None)
        if callable(delete_method):
            delete_method(document_name)

        self.vector_store.upsert(chunks, embeddings)

    def index_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Extract, chunk, embed and save a PDF in the local vector store."""
        document_name = _clean_document_name(filename)

        if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("pdf_bytes must be bytes.")

        pdf_data = bytes(pdf_bytes)

        if not pdf_data:
            raise ValueError("The PDF file is empty.")

        if len(pdf_data) > self.config.max_pdf_size_bytes:
            max_mb = self.config.max_pdf_size_bytes / (1024 * 1024)
            raise ValueError(
                f"PDF exceeds the configured {max_mb:.1f} MiB limit."
            )

        existing_names = self._existing_document_names()
        already_exists = any(
            name.casefold() == document_name.casefold()
            for name in existing_names
        )

        if already_exists and not replace_existing:
            raise FileExistsError(
                f"{document_name!r} is already indexed. "
                "Choose the replace option to index it again."
            )

        pages = _call_pdf_extractor(
            pdf_bytes=pdf_data,
            document_name=document_name,
            max_pdf_size_bytes=self.config.max_pdf_size_bytes,
        )

        if not pages:
            raise ValueError("No pages were extracted from this PDF.")

        if not any(_extract_text(page) for page in pages):
            raise ValueError(
                "No selectable text was found in this PDF. Scanned/image-only "
                "PDFs need OCR, which is not enabled here."
            )

        raw_chunks = _call_chunker(
            pages=pages,
            config=self.config,
            document_name=document_name,
        )

        chunks = _normalise_chunks(raw_chunks, document_name)

        if not chunks:
            raise ValueError("No text chunks were created from this PDF.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._embed_documents(texts)

        # Every item passed to SQLite has a non-empty top-level chunk_id.
        self._save_chunks(
            document_name=document_name,
            chunks=chunks,
            embeddings=embeddings,
            replace_existing=replace_existing,
        )

        return {
            "document_name": document_name,
            "pages_indexed": len(pages),
            "chunks_indexed": len(chunks),
            "replaced_existing": bool(already_exists and replace_existing),
        }

    def index_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Alias for index_pdf, useful for upload interfaces."""
        return self.index_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
            replace_existing=replace_existing,
        )

    def _call_retriever(
        self,
        question: str,
        top_k: int,
        document_name: str | None,
    ) -> Any:
        """Call the retriever using the arguments its method supports."""
        method = getattr(self.retriever, "retrieve", None)

        if not callable(method):
            method = getattr(self.retriever, "search", None)

        if not callable(method):
            raise AttributeError(
                "The semantic retriever must provide retrieve() or search()."
            )

        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}

        accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        kwargs: dict[str, Any] = {}

        if "top_k" in parameters or accepts_extra_kwargs:
            kwargs["top_k"] = top_k
        elif "k" in parameters:
            kwargs["k"] = top_k
        elif "n_results" in parameters:
            kwargs["n_results"] = top_k

        if document_name:
            if "document_name" in parameters or accepts_extra_kwargs:
                kwargs["document_name"] = document_name
            elif "where" in parameters:
                kwargs["where"] = {"document_name": document_name}

        return method(question, **kwargs)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_name: str | None = None,
    ) -> list[SourceCitation]:
        """Retrieve relevant passages from the local vector store."""
        question_text = str(question or "").strip()
        if not question_text:
            return []

        result_count = self.config.top_k if top_k is None else int(top_k)
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero.")

        raw_result = self._call_retriever(
            question=question_text,
            top_k=result_count,
            document_name=document_name,
        )

        hits = _search_hits(raw_result)
        citations: list[SourceCitation] = []

        for original_hit in hits:
            hit = original_hit
            tuple_score = None

            if (
                isinstance(original_hit, tuple)
                and len(original_hit) == 2
                and isinstance(original_hit[1], (int, float, np.number))
            ):
                hit = original_hit[0]
                tuple_score = original_hit[1]

            text = _extract_text(hit)
            if not text:
                continue

            nested_chunk = _value(hit, "chunk", "document", default=None)

            metadata = _extract_metadata(nested_chunk)
            metadata.update(_extract_metadata(hit))

            found_name = _value(
                metadata,
                "document_name",
                "filename",
                "source",
                default=None,
            )

            if found_name is None:
                found_name = _value(
                    hit,
                    "document_name",
                    "filename",
                    "source",
                    default=None,
                )

            if found_name is None:
                found_name = _value(
                    nested_chunk,
                    "document_name",
                    "filename",
                    "source",
                    default="unknown",
                )

            found_name = _clean_document_name(str(found_name or "unknown"))

            if document_name:
                requested_name = _clean_document_name(document_name)
                if (
                    found_name != "unknown"
                    and found_name.casefold() != requested_name.casefold()
                ):
                    continue

            page = _page_number(hit, metadata)
            if page is None:
                page = _page_number(nested_chunk, metadata)

            similarity_value = tuple_score

            if similarity_value is None:
                similarity_value = _value(
                    hit,
                    "similarity",
                    "cosine_similarity",
                    "score",
                    default=None,
                )

            if similarity_value is None:
                similarity_value = _value(
                    nested_chunk,
                    "similarity",
                    "cosine_similarity",
                    "score",
                    default=None,
                )

            similarity = _safe_float(similarity_value)
            source_id = f"S{len(citations) + 1}"

            metadata["document_name"] = found_name
            if page is not None:
                metadata["page_number"] = page

            citations.append(
                SourceCitation(
                    source_id=source_id,
                    chunk=text,
                    excerpt=text[:1000],
                    document_name=found_name,
                    page_number=page,
                    similarity=similarity,
                    metadata=metadata,
                )
            )

            if len(citations) >= result_count:
                break

        return citations

    def answer(self, question: str) -> RAGAnswer:
        """Answer using retrieved passages and validate citations."""
        question_text = str(question or "").strip()

        if not question_text:
            return RAGAnswer(
                answer="Please enter a question.",
                abstained=True,
                abstention_reason="empty_question",
            )

        if len(question_text) > self.config.max_question_chars:
            return RAGAnswer(
                answer=(
                    "Please shorten the question to at most "
                    f"{self.config.max_question_chars} characters."
                ),
                abstained=True,
                abstention_reason="question_too_long",
            )

        sources = self.retrieve(
            question_text,
            top_k=self.config.top_k,
        )

        if not sources:
            return RAGAnswer(
                answer=(
                    "I couldn't retrieve any document passages to answer this "
                    "question. Please index a PDF containing the information, "
                    "then try again."
                ),
                abstained=True,
                abstention_reason="no_retrieved_passages",
            )

        if self.llm_client is None:
            return RAGAnswer(
                answer=(
                    "Groq answer generation is not configured. Check the local "
                    "GROQ_API_KEY and GROQ_MODEL settings."
                ),
                sources=sources,
                abstained=True,
                abstention_reason="groq_client_not_configured",
            )

        try:
            messages = build_rag_messages(
                question=question_text,
                sources=sources,
                max_context_chars=self.config.max_context_chars,
            )

            response = self.llm_client.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_completion_tokens,
            )

            generated_text = llm_response_text(response)

        except Exception:
            # Keep provider details and credentials out of the UI message.
            return RAGAnswer(
                answer=(
                    "Groq could not generate an answer. Check the local API key "
                    "and model settings, and confirm that the selected model is "
                    "currently available."
                ),
                sources=sources,
                llm_used=False,
                abstained=True,
                abstention_reason="groq_request_failed",
            )

        result = RAGAnswer(
            answer=generated_text,
            sources=sources,
            cited_source_ids=[],
            llm_used=True,
            abstained=False,
            abstention_reason=None,
        )

        return validate_rag_answer(result)

    def ask(self, question: str) -> RAGAnswer:
        """Alias for answer()."""
        return self.answer(question)

    def query(self, question: str) -> RAGAnswer:
        """Alias for answer()."""
        return self.answer(question)