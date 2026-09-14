"""Halo Vector — ChromaDB-backed persistent vector storage.

Stores and retrieves source-code chunks using cosine similarity over
bge-m3 embeddings produced by
:class:`~halo_vector.embeddings.EmbeddingService`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .embeddings import EmbeddingService


class VectorEngine:
    """Persistent vector database for indexed source-code chunks."""

    def __init__(
        self, persist_dir: str, embedding_service: EmbeddingService
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="apex_code_index",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def index_file(
        self,
        file_path: str,
        text: str,
        chunk_lines: int = 40,
        overlap_lines: int = 8,
    ) -> int:
        """Split a file into overlapping line chunks, embed and upsert
        them. Returns the number of chunks indexed."""
        lines = text.splitlines()
        if not lines:
            return 0

        chunks: list[str] = []
        ranges: list[tuple[int, int]] = []
        start = 0
        while True:
            end = min(start + chunk_lines, len(lines))
            chunks.append("\n".join(lines[start:end]))
            ranges.append((start + 1, end))
            if end >= len(lines):
                break
            start = max(end - overlap_lines, start + 1)

        embeddings = self.embedding_service.embed_batch(chunks)
        file_name = Path(file_path).name
        ids = [f"{file_name}::chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_path": str(file_path),
                "file_name": file_name,
                "chunk_index": i,
                "line_start": ranges[i][0],
                "line_end": ranges[i][1],
            }
            for i in range(len(chunks))
        ]
        self.collection.upsert(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )
        return len(chunks)

    # ------------------------------------------------------------------
    def query(self, query_text: str, n_results: int = 3) -> list[dict[str, Any]]:
        """Semantic code search using cosine similarity."""
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.embedding_service.embed_text(query_text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": 1.0 - float(dist),
                }
            )
        return hits

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop and recreate the collection (remove all documents)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="apex_code_index", metadata={"hnsw:space": "cosine"}
        )

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self.collection.count()