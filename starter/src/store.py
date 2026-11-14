"""Persistent vector store: Chroma (local, on-disk) with cosine HNSW + metadata filter.

Docker was not available on the lab machine, so the pgvector starter was
replaced by Chroma as allowed in GUIDE.md §3. One collection per chunking
strategy (A, B); strategy C searches collection B with a `where` filter.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .models import Chunk

ALLOWED_FILTER_FIELDS = {"pep_type", "status", "created_year", "python_version", "pep", "section"}


class ChromaStore:
    def __init__(self, path: str, dimensions: int, model_name: str) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.dimensions = dimensions
        self.model_name = model_name

    def _collection(self, name: str):
        col = self.client.get_or_create_collection(
            name=f"day06_{name}",
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name,
                      "dimensions": self.dimensions},
            embedding_function=None)
        meta = col.metadata or {}
        if meta.get("embedding_model") != self.model_name or int(meta.get("dimensions", 0)) != self.dimensions:
            raise ValueError(f"collection {name} was built with {meta}; refusing to mix models")
        return col

    def count(self, name: str) -> int:
        return self._collection(name).count()

    def upsert(self, name: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Idempotent write with stable IDs; deletes stale chunks of re-ingested docs.

        Unchanged chunks (same id + same content hash) are skipped, so a second
        ingest of the same corpus version writes nothing and duplicates nothing.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension mismatch")
        col = self._collection(name)
        new_ids = {c.chunk_id for c in chunks}
        if len(new_ids) != len(chunks):
            raise ValueError("duplicate chunk ids in batch")
        docs = sorted({c.document_id for c in chunks})
        existing = col.get(where={"document_id": {"$in": docs}}, include=["metadatas"])
        old = {i: m.get("content_hash") for i, m in zip(existing["ids"], existing["metadatas"])}
        stale = [i for i in old if i not in new_ids]
        if stale:
            col.delete(ids=stale)
        ids, embs, texts, metas = [], [], [], []
        for c, v in zip(chunks, vectors):
            h = hashlib.sha256(c.text.encode("utf-8")).hexdigest()[:16]
            if old.get(c.chunk_id) == h:
                continue
            ids.append(c.chunk_id)
            embs.append(v)
            texts.append(c.text)
            metas.append({"document_id": c.document_id, "source": c.source, "page": c.page,
                          "content_hash": h, **{k: v2 for k, v2 in c.metadata.items()
                                                if isinstance(v2, (str, int, float, bool))}})
        for s in range(0, len(ids), 500):
            col.upsert(ids=ids[s:s + 500], embeddings=embs[s:s + 500],
                       documents=texts[s:s + 500], metadatas=metas[s:s + 500])
        return {"written": len(ids), "unchanged": len(chunks) - len(ids), "deleted_stale": len(stale)}

    def search(self, name: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where = None
        if filters:
            bad = set(filters) - ALLOWED_FILTER_FIELDS
            if bad:
                raise ValueError(f"filter fields not allowed: {bad}")
            clauses = [{k: {"$eq": v}} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._collection(name).query(query_embeddings=[query_vector], n_results=top_k,
                                           where=where, include=["documents", "metadatas", "distances"])
        return [{"rank": r + 1, "chunk_id": i, "score": round(1 - d, 4), "text": t, "metadata": m}
                for r, (i, t, m, d) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                     res["metadatas"][0], res["distances"][0]))]
