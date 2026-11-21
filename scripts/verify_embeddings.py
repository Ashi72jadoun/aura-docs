"""Download/load the real model and verify example embeddings."""

import sys

import numpy as np

from rag.embeddings import SentenceTransformerEmbedder
from rag.models import TextChunk


def main() -> int:
    chunks = [
        TextChunk(
            chunk_id="sample-1",
            text=(
                "Employees may request paid sick leave to recover from illness, "
                "subject to the organization's leave policy."
            ),
            document_name="sample_policy.pdf",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="sample-2",
            text=(
                "Company laptops must be returned to the IT department "
                "when employment ends."
            ),
            document_name="sample_policy.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]

    query = "What does the policy say about paid sick leave?"

    print("Loading the local embedding model...")
    print("The first run may download model files from Hugging Face.")

    embedder = SentenceTransformerEmbedder(device="cpu")
    document_vectors = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(query)

    expected_document_shape = (len(chunks), embedder.embedding_dimension)
    expected_query_shape = (embedder.embedding_dimension,)

    if document_vectors.shape != expected_document_shape:
        print(
            f"[FAIL] Document vector shape: {document_vectors.shape}; "
            f"expected {expected_document_shape}."
        )
        return 1

    if query_vector.shape != expected_query_shape:
        print(
            f"[FAIL] Query vector shape: {query_vector.shape}; "
            f"expected {expected_query_shape}."
        )
        return 1

    document_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = float(np.linalg.norm(query_vector))

    if not np.isfinite(document_vectors).all() or not np.isfinite(query_vector).all():
        print("[FAIL] At least one vector contains a non-finite value.")
        return 1

    if not np.allclose(document_norms, 1.0, atol=1e-3):
        print(f"[FAIL] Document vectors are not normalized: {document_norms}")
        return 1

    if not np.isclose(query_norm, 1.0, atol=1e-3):
        print(f"[FAIL] Query vector is not normalized: {query_norm}")
        return 1

    # Since vectors are normalized, dot product is cosine similarity.
    similarity_scores = document_vectors @ query_vector

    print(f"[OK] Model: {embedder.model_name}")
    print(f"[OK] Embedding dimension: {embedder.embedding_dimension}")
    print(f"[OK] Model maximum sequence length: {embedder.max_sequence_length}")
    print(f"[OK] Document-vector shape: {document_vectors.shape}")
    print(f"[OK] Query-vector shape: {query_vector.shape}")
    print(f"[OK] Document-vector norms: {document_norms}")
    print(f"[OK] Query-vector norm: {query_norm:.4f}")
    print("[OK] Example similarity scores:")

    ranked_indices = np.argsort(similarity_scores)[::-1]
    for index in ranked_indices:
        chunk = chunks[int(index)]
        print(
            f"  {chunk.document_name}, page {chunk.page_number}: "
            f"{similarity_scores[index]:.4f}"
        )

    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())