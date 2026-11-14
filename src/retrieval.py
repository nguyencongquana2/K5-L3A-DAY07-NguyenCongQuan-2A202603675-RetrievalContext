"""Retrievers backed by the CP1 corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


class DenseChromaRetriever:
    """Dense retriever compatible with the selected Day 6 embedding setup."""

    ALLOWED_FILTERS = {"category", "source_id", "page", "release"}

    def __init__(
        self,
        chunks_path: str | Path,
        persist_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "day7_dense_v1",
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.persist_path = Path(persist_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
        self._ensure_index()

    def _ensure_index(self) -> None:
        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        desired_ids = [chunk["id"] for chunk in chunks]
        current_ids = set(self.collection.get(include=[])["ids"])
        if current_ids == set(desired_ids):
            return
        if current_ids:
            self.collection.delete(ids=list(current_ids))
        vectors = self.model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source_id": chunk["source_id"],
                "source": chunk["source"],
                "page": chunk["page"],
            }
            metadata.update(
                {key: value for key, value in chunk.get("metadata", {}).items()
                 if isinstance(value, (str, int, float, bool))}
            )
            metadatas.append(metadata)
        self.collection.upsert(
            ids=desired_ids,
            documents=[chunk["text"] for chunk in chunks],
            embeddings=vectors,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        unknown = set(filters or {}) - self.ALLOWED_FILTERS
        if unknown:
            raise ValueError(f"unsupported filter fields: {sorted(unknown)}")
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            metadata = dict(metadata or {})
            rows.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source_id": metadata.pop("source_id"),
                    "source": metadata.pop("source"),
                    "page": metadata.pop("page"),
                    "metadata": metadata,
                    "score": round(1.0 - float(distance), 8),
                    "strategy": "dense_chroma",
                }
            )
        return rows
