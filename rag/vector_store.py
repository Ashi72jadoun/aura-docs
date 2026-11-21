"""Local vector storage using the Python SQLite module and NumPy.

The store performs exact cosine similarity search over saved float32 vectors.
It does not use Chroma or a separate vector-database package.
"""

from __future__ import annotations

import json
import numbers
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class VectorStoreError(RuntimeError):
    """Raised for invalid vector-store input or database errors."""


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: StoredChunk
    similarity: float
    distance: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def content(self) -> str:
        return self.chunk.text

    @property
    def document_name(self) -> str:
        return self.chunk.document_name

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def chunk_index(self) -> int:
        return self.chunk.chunk_index

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata

    @property
    def score(self) -> float:
        return self.similarity

    @property
    def cosine_similarity(self) -> float:
        return self.similarity

    @property
    def cosine_distance(self) -> float:
        return self.distance

    def __getitem__(self, key: str) -> Any:
        """Support convenient dictionary-style access for search results."""
        fields = {
            "chunk": self.chunk,
            "chunk_id": self.chunk_id,
            "id": self.id,
            "text": self.text,
            "content": self.content,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "similarity": self.similarity,
            "score": self.score,
            "distance": self.distance,
            "cosine_similarity": self.cosine_similarity,
            "cosine_distance": self.cosine_distance,
        }
        if key not in fields:
            raise KeyError(key)
        return fields[key]


def _field(value: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either a dictionary or an object."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if not isinstance(value, Mapping) and hasattr(value, name):
            return getattr(value, name)
    return default


def _as_list(value: Iterable[Any], label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)):
        raise VectorStoreError(f"{label} must be an iterable, not text.")

    try:
        return list(value)
    except TypeError as exc:
        raise VectorStoreError(f"{label} must be iterable.") from exc


def _normalize_embeddings(embeddings: Any, expected_count: int) -> np.ndarray:
    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VectorStoreError("Embeddings must be a numeric array.") from exc

    if expected_count == 1 and matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise VectorStoreError(
            "Embeddings must be a 2D array: one row per chunk."
        )

    if matrix.shape[0] != expected_count:
        raise VectorStoreError(
            f"Received {matrix.shape[0]} embeddings for {expected_count} chunks."
        )

    if matrix.shape[1] < 1:
        raise VectorStoreError("An embedding must have at least one dimension.")

    if not np.isfinite(matrix).all():
        raise VectorStoreError("Embeddings must contain finite numbers.")

    # Store unit-length vectors so their dot product is cosine similarity.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12):
        raise VectorStoreError("Zero-length embeddings cannot be stored.")

    normalized = matrix.astype(np.float64) / norms[:, None]
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _normalize_query(query_embedding: Any) -> np.ndarray:
    try:
        vector = np.asarray(query_embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VectorStoreError("The query embedding must be numeric.") from exc

    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]

    if vector.ndim != 1 or vector.size < 1:
        raise VectorStoreError("The query embedding must be a 1D vector.")

    if not np.isfinite(vector).all():
        raise VectorStoreError("The query embedding must contain finite numbers.")

    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm <= 1e-12:
        raise VectorStoreError("A zero-length query embedding cannot be searched.")

    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class SQLiteVectorStore:
    """Persistent exact-cosine vector search backed by a local SQLite file."""

    def __init__(
        self,
        persist_directory: str | Path = "data/vector_store",
        collection_name: str = "document_chunks",
        *,
        db_path: str | Path | None = None,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise VectorStoreError("collection_name must be a non-empty string.")

        if db_path is not None:
            database_path = Path(db_path)
        else:
            requested_path = Path(persist_directory)
            if requested_path.suffix.casefold() in {
                ".db",
                ".sqlite",
                ".sqlite3",
            }:
                database_path = requested_path
            else:
                database_path = requested_path / "document_vectors.sqlite3"

        self.database_path = database_path
        self.persist_directory = database_path.parent
        self.collection_name = collection_name.strip()

        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VectorStoreError(
                f"Could not create vector-store directory: "
                f"{self.database_path.parent}"
            ) from exc

        self._initialize_database()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None

        try:
            connection = sqlite3.connect(str(self.database_path), timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise VectorStoreError(
                f"SQLite vector-store operation failed: {exc}"
            ) from exc
        except Exception:
            if connection is not None:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL CHECK (page_number >= 1),
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL CHECK (dimension >= 1),
                    PRIMARY KEY (collection_name, chunk_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vector_chunks_collection_document
                ON vector_chunks (collection_name, document_name)
                """
            )

    @staticmethod
    def _chunk_record(chunk: Any) -> tuple[str, str, int, int, str, str]:
        metadata_value = _field(chunk, "metadata", default={})
        if metadata_value is None:
            metadata_value = {}
        if not isinstance(metadata_value, Mapping):
            raise VectorStoreError("Chunk metadata must be a dictionary.")

        metadata = dict(metadata_value)

        # Preserve extra top-level dictionary fields as metadata.
        if isinstance(chunk, Mapping):
            reserved_fields = {
                "chunk_id",
                "id",
                "text",
                "content",
                "page_content",
                "document_name",
                "filename",
                "source_name",
                "source",
                "page_number",
                "page",
                "chunk_index",
                "metadata",
            }
            for key, value in chunk.items():
                if key not in reserved_fields:
                    metadata.setdefault(key, value)

        chunk_id_value = _field(
            chunk,
            "chunk_id",
            "id",
            default=_field(metadata, "chunk_id", "id"),
        )
        if chunk_id_value is None or not str(chunk_id_value).strip():
            raise VectorStoreError("Every chunk needs a non-empty chunk_id.")
        chunk_id = str(chunk_id_value).strip()

        text_value = _field(chunk, "text", "content", "page_content")
        if not isinstance(text_value, str) or not text_value.strip():
            raise VectorStoreError(f"Chunk '{chunk_id}' needs non-empty text.")
        text = text_value

        document_value = _field(
            chunk,
            "document_name",
            "filename",
            "source_name",
            "source",
            default=_field(
                metadata,
                "document_name",
                "filename",
                "source_name",
                "source",
            ),
        )
        if document_value is None or not str(document_value).strip():
            raise VectorStoreError(
                f"Chunk '{chunk_id}' needs a document_name."
            )
        document_name = str(document_value).strip()

        page_value = _field(
            chunk,
            "page_number",
            "page",
            default=_field(metadata, "page_number", "page"),
        )
        if isinstance(page_value, bool) or not isinstance(
            page_value, numbers.Integral
        ):
            raise VectorStoreError(
                f"Chunk '{chunk_id}' needs a positive, 1-based page number."
            )
        page_number = int(page_value)
        if page_number < 1:
            raise VectorStoreError(
                f"Chunk '{chunk_id}' needs a positive, 1-based page number."
            )

        index_value = _field(
            chunk,
            "chunk_index",
            default=_field(metadata, "chunk_index", default=0),
        )
        if isinstance(index_value, bool) or not isinstance(
            index_value, numbers.Integral
        ):
            raise VectorStoreError(
                f"Chunk '{chunk_id}' has an invalid chunk_index."
            )
        chunk_index = int(index_value)
        if chunk_index < 0:
            raise VectorStoreError(
                f"Chunk '{chunk_id}' has a negative chunk_index."
            )

        metadata.update(
            {
                "chunk_id": chunk_id,
                "document_name": document_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
            }
        )

        try:
            metadata_json = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise VectorStoreError(
                f"Metadata for chunk '{chunk_id}' must be JSON-serializable."
            ) from exc

        return (
            chunk_id,
            document_name,
            page_number,
            chunk_index,
            text,
            metadata_json,
        )

    def _prepare_rows(
        self,
        chunks: Iterable[Any],
        embeddings: Any,
    ) -> tuple[list[tuple[Any, ...]], int | None]:
        chunk_list = _as_list(chunks, "chunks")
        if not chunk_list:
            return [], None

        matrix = _normalize_embeddings(embeddings, len(chunk_list))
        dimension = int(matrix.shape[1])
        records = [self._chunk_record(chunk) for chunk in chunk_list]

        chunk_ids = [record[0] for record in records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorStoreError("Chunk IDs in a write batch must be unique.")

        rows: list[tuple[Any, ...]] = []
        for index, record in enumerate(records):
            (
                chunk_id,
                document_name,
                page_number,
                chunk_index,
                text,
                metadata_json,
            ) = record

            rows.append(
                (
                    self.collection_name,
                    chunk_id,
                    document_name,
                    page_number,
                    chunk_index,
                    text,
                    metadata_json,
                    matrix[index].tobytes(order="C"),
                    dimension,
                )
            )

        return rows, dimension

    def _collection_dimension(
        self,
        connection: sqlite3.Connection,
    ) -> int | None:
        rows = connection.execute(
            """
            SELECT DISTINCT dimension
            FROM vector_chunks
            WHERE collection_name = ?
            """,
            (self.collection_name,),
        ).fetchall()

        dimensions = {int(row["dimension"]) for row in rows}
        if len(dimensions) > 1:
            raise VectorStoreError(
                "This collection contains different embedding dimensions."
            )
        return next(iter(dimensions), None)

    def _check_dimension(
        self,
        connection: sqlite3.Connection,
        dimension: int,
    ) -> None:
        existing = self._collection_dimension(connection)
        if existing is not None and existing != dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: collection uses {existing}, "
                f"but received {dimension}."
            )

    @staticmethod
    def _write_rows(
        connection: sqlite3.Connection,
        rows: list[tuple[Any, ...]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO vector_chunks (
                collection_name,
                chunk_id,
                document_name,
                page_number,
                chunk_index,
                text,
                metadata_json,
                embedding,
                dimension
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (collection_name, chunk_id) DO UPDATE SET
                document_name = excluded.document_name,
                page_number = excluded.page_number,
                chunk_index = excluded.chunk_index,
                text = excluded.text,
                metadata_json = excluded.metadata_json,
                embedding = excluded.embedding,
                dimension = excluded.dimension
            """,
            rows,
        )

    def upsert(
        self,
        chunks: Iterable[Any],
        embeddings: Any,
    ) -> int:
        """Insert new chunks or update chunks with matching IDs."""
        rows, dimension = self._prepare_rows(chunks, embeddings)
        if not rows or dimension is None:
            return 0

        with self._connection() as connection:
            self._check_dimension(connection, dimension)
            self._write_rows(connection, rows)

        return len(rows)

    def add_chunks(
        self,
        chunks: Iterable[Any],
        embeddings: Any,
    ) -> int:
        """Convenience alias for upsert."""
        return self.upsert(chunks, embeddings)

    def replace_document(
        self,
        document_name: str,
        chunks: Iterable[Any],
        embeddings: Any,
    ) -> int:
        """Replace one document's chunks atomically."""
        if not isinstance(document_name, str) or not document_name.strip():
            raise VectorStoreError("document_name must be a non-empty string.")
        document_name = document_name.strip()

        rows, dimension = self._prepare_rows(chunks, embeddings)
        if not rows or dimension is None:
            self.delete_document(document_name)
            return 0

        if any(row[2] != document_name for row in rows):
            raise VectorStoreError(
                "All replacement chunks must match document_name."
            )

        with self._connection() as connection:
            connection.execute(
                """
                DELETE FROM vector_chunks
                WHERE collection_name = ? AND document_name = ?
                """,
                (self.collection_name, document_name),
            )
            self._check_dimension(connection, dimension)
            self._write_rows(connection, rows)

        return len(rows)

    def delete_document(self, document_name: str) -> int:
        """Delete chunks for one document and return how many were removed."""
        if not isinstance(document_name, str) or not document_name.strip():
            raise VectorStoreError("document_name must be a non-empty string.")

        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM vector_chunks
                WHERE collection_name = ? AND document_name = ?
                """,
                (self.collection_name, document_name.strip()),
            )
            deleted_count = cursor.rowcount

        return max(deleted_count, 0)

    def search(
        self,
        query_embedding: Any,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return the K most similar chunks, with the best match first."""
        if type(top_k) is not int or top_k < 1:
            raise VectorStoreError("top_k must be a positive integer.")

        query_vector = _normalize_query(query_embedding)
        metadata_filter: dict[str, Any] = {}

        if where is not None:
            if not isinstance(where, Mapping):
                raise VectorStoreError("where must be a dictionary.")
            metadata_filter.update(where)

        if document_name is not None:
            metadata_filter["document_name"] = document_name

        with self._connection() as connection:
            dimension = self._collection_dimension(connection)
            if dimension is None:
                return []
            if dimension != query_vector.size:
                raise VectorStoreError(
                    f"Query dimension mismatch: collection uses {dimension}, "
                    f"but query has {query_vector.size}."
                )

            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_name,
                    page_number,
                    chunk_index,
                    text,
                    metadata_json,
                    embedding,
                    dimension
                FROM vector_chunks
                WHERE collection_name = ?
                ORDER BY chunk_id
                """,
                (self.collection_name,),
            ).fetchall()

        candidates: list[tuple[Any, dict[str, Any]]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise VectorStoreError(
                    f"Could not read metadata for chunk '{row['chunk_id']}'."
                ) from exc

            if not isinstance(metadata, dict):
                raise VectorStoreError(
                    f"Metadata for chunk '{row['chunk_id']}' is invalid."
                )

            if all(
                metadata.get(key) == expected
                for key, expected in metadata_filter.items()
            ):
                candidates.append((row, metadata))

        if not candidates:
            return []

        vectors: list[np.ndarray] = []
        for row, _metadata in candidates:
            if int(row["dimension"]) != query_vector.size:
                raise VectorStoreError(
                    f"Stored chunk '{row['chunk_id']}' has a wrong dimension."
                )

            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != query_vector.size:
                raise VectorStoreError(
                    f"Stored vector for chunk '{row['chunk_id']}' is invalid."
                )
            vectors.append(vector)

        matrix = np.vstack(vectors)
        similarities = np.clip(matrix @ query_vector, -1.0, 1.0)

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(similarities[index]),
                str(candidates[index][0]["chunk_id"]),
            ),
        )[:top_k]

        results: list[VectorSearchResult] = []
        for index in ranked_indices:
            row, metadata = candidates[index]
            similarity = float(similarities[index])

            chunk = StoredChunk(
                chunk_id=str(row["chunk_id"]),
                text=str(row["text"]),
                document_name=str(row["document_name"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                metadata=metadata,
            )
            results.append(
                VectorSearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    distance=1.0 - similarity,
                )
            )

        return results

    def count(self) -> int:
        """Return the number of chunks in this collection."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM vector_chunks
                WHERE collection_name = ?
                """,
                (self.collection_name,),
            ).fetchone()
        return int(row["total"])

    def list_document_names(self) -> list[str]:
        """Return the distinct document names in this collection."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT document_name
                FROM vector_chunks
                WHERE collection_name = ?
                ORDER BY document_name
                """,
                (self.collection_name,),
            ).fetchall()
        return [str(row["document_name"]) for row in rows]

    def clear(self) -> int:
        """Delete this collection's rows, but keep the SQLite database file."""
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM vector_chunks
                WHERE collection_name = ?
                """,
                (self.collection_name,),
            )
            deleted_count = cursor.rowcount
        return max(deleted_count, 0)

    def close(self) -> None:
        """Compatibility no-op: each operation closes its own SQLite connection."""

    def __len__(self) -> int:
        return self.count()