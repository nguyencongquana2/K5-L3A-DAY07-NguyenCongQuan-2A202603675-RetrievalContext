"""Retrievers backed by the CP1 corpus."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
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


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class TfidfRetriever:
    """Offline lexical retriever with the same output contract as dense search."""

    ALLOWED_FILTERS = DenseChromaRetriever.ALLOWED_FILTERS

    def __init__(self, chunks_path: str | Path) -> None:
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        counts = [Counter(_tokens(chunk["text"])) for chunk in self.chunks]
        document_frequency = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(self.chunks)) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    @staticmethod
    def _matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        values = {"source_id": chunk["source_id"], "page": chunk["page"], **chunk.get("metadata", {})}
        return all(values.get(key) == value for key, value in filters.items())

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
        query_vector = self._vector(Counter(_tokens(query)))
        rows = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if not self._matches(chunk, filters):
                continue
            score = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            if score > 0:
                rows.append({**chunk, "score": score, "strategy": "tfidf"})
        return sorted(rows, key=lambda row: (-row["score"], row["id"]))[:top_k]


class HybridRrfRetriever:
    """Fuse dense and lexical rankings with reciprocal-rank fusion."""

    def __init__(
        self,
        dense: DenseChromaRetriever,
        lexical: TfidfRetriever,
        candidate_k: int = 10,
        rrf_constant: int = 60,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidate_k = max(top_k, self.candidate_k)
        dense_rows = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_rows = self.lexical.search(query, filters=filters, top_k=candidate_k)
        by_id: dict[str, dict[str, Any]] = {}
        scores: Counter[str] = Counter()
        ranks: dict[str, dict[str, int]] = {}
        component_scores: dict[str, dict[str, float]] = {}
        for name, rows in (("dense", dense_rows), ("lexical", lexical_rows)):
            for rank, row in enumerate(rows, start=1):
                by_id.setdefault(row["id"], row)
                scores[row["id"]] += 1.0 / (self.rrf_constant + rank)
                ranks.setdefault(row["id"], {})[name] = rank
                component_scores.setdefault(row["id"], {})[name] = float(row["score"])
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            {
                **by_id[chunk_id],
                "score": round(scores[chunk_id], 8),
                "answer_score": component_scores[chunk_id].get("dense", 0.0),
                "strategy": "hybrid_rrf",
                "ranks": ranks[chunk_id],
                "component_scores": component_scores[chunk_id],
            }
            for chunk_id in ordered
        ]
