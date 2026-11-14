"""Persistent vector stores with cosine search + metadata filter.

- PgVectorStore (default, VECTOR_STORE=pgvector): PostgreSQL 16 + pgvector 0.8.7
  from starter/docker-compose.yml. Exact search (no ANN index, ~1.4k rows)
  ordered by cosine distance `<=>`; filter columns are whitelisted and values
  are bound parameters.
- ChromaStore (VECTOR_STORE=chroma): on-disk Chroma, HNSW (approximate). Used
  for the first run before Docker was available; kept for comparison.

A and B are stored separately (strategy column / collection); C searches B's
rows with a metadata filter.
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


class PgVectorStore:
    """PostgreSQL + pgvector. One table, rows keyed by (strategy, chunk_id)."""

    TABLE = "day06_chunks"
    COLUMNS = ("pep", "title", "pep_type", "status", "created_year", "python_version", "section",
               "char_start", "char_end")

    def __init__(self, database_url: str, dimensions: int, model_name: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self.url = database_url
        self.dimensions = dimensions
        self.model_name = model_name
        self.create_schema()

    def _conn(self):
        return self._psycopg.connect(self.url, connect_timeout=5)

    def create_schema(self) -> None:
        with self._conn() as con:
            con.execute("CREATE EXTENSION IF NOT EXISTS vector")
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    strategy        text    NOT NULL,
                    chunk_id        text    NOT NULL,
                    document_id     text    NOT NULL,
                    source          text    NOT NULL,
                    page            integer NOT NULL,
                    pep             integer,
                    title           text,
                    pep_type        text,
                    status          text,
                    created_year    integer,
                    python_version  text,
                    section         text,
                    char_start      integer,
                    char_end        integer,
                    text            text    NOT NULL,
                    content_hash    text    NOT NULL,
                    embedding_model text    NOT NULL,
                    embedding       vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id))""")
            con.execute(f"CREATE INDEX IF NOT EXISTS {self.TABLE}_filter_idx "
                        f"ON {self.TABLE} (strategy, pep_type, document_id)")
            row = con.execute("SELECT atttypmod FROM pg_attribute "
                              "WHERE attrelid = %s::regclass AND attname = 'embedding'", (self.TABLE,)).fetchone()
            if row and row[0] != self.dimensions:
                raise ValueError(f"table has vector({row[0]}), model gives {self.dimensions}")
            other = con.execute(f"SELECT DISTINCT embedding_model FROM {self.TABLE} WHERE embedding_model <> %s",
                                (self.model_name,)).fetchall()
            if other:
                raise ValueError(f"table already holds vectors from {other}; refusing to mix models")

    @staticmethod
    def _vec(v: list[float]) -> str:
        return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

    def count(self, name: str) -> int:
        with self._conn() as con:
            return con.execute(f"SELECT count(*) FROM {self.TABLE} WHERE strategy = %s", (name,)).fetchone()[0]

    def upsert(self, name: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Idempotent: unchanged (id, hash) rows skipped; stale rows of re-ingested docs deleted."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension mismatch")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids in batch")
        new_ids = set(ids)
        docs = sorted({c.document_id for c in chunks})
        cols = ["strategy", "chunk_id", "document_id", "source", "page", *self.COLUMNS,
                "text", "content_hash", "embedding_model", "embedding"]
        placeholders = ", ".join(["%s"] * (len(cols) - 1) + ["%s::vector"])
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[2:])
        sql = (f"INSERT INTO {self.TABLE} ({', '.join(cols)}) VALUES ({placeholders}) "
               f"ON CONFLICT (strategy, chunk_id) DO UPDATE SET {updates}")
        with self._conn() as con:
            with con.transaction():
                old = dict(con.execute(
                    f"SELECT chunk_id, content_hash FROM {self.TABLE} "
                    "WHERE strategy = %s AND document_id = ANY(%s)", (name, docs)).fetchall())
                stale = [i for i in old if i not in new_ids]
                if stale:
                    con.execute(f"DELETE FROM {self.TABLE} WHERE strategy = %s AND chunk_id = ANY(%s)",
                                (name, stale))
                rows = []
                for c, v in zip(chunks, vectors):
                    h = hashlib.sha256(c.text.encode("utf-8")).hexdigest()[:16]
                    if old.get(c.chunk_id) == h:
                        continue
                    rows.append((name, c.chunk_id, c.document_id, c.source, c.page,
                                 *[c.metadata.get(k) for k in self.COLUMNS],
                                 c.text, h, self.model_name, self._vec(v)))
                if rows:
                    with con.cursor() as cur:
                        cur.executemany(sql, rows)
        return {"written": len(rows), "unchanged": len(chunks) - len(rows), "deleted_stale": len(stale)}

    def search(self, name: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where, params = ["strategy = %s"], [name]
        for k, v in (filters or {}).items():
            if k not in ALLOWED_FILTER_FIELDS:  # column names whitelisted, values bound
                raise ValueError(f"filter field not allowed: {k}")
            where.append(f"{k} = %s")
            params.append(v)
        q = self._vec(query_vector)
        meta_cols = ["document_id", "source", "page", *self.COLUMNS, "content_hash"]
        sql = (f"SELECT chunk_id, text, 1 - (embedding <=> %s::vector) AS score, {', '.join(meta_cols)} "
               f"FROM {self.TABLE} WHERE {' AND '.join(where)} "
               "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self._conn() as con:
            rows = con.execute(sql, [q, *params, q, top_k]).fetchall()
        return [{"rank": r + 1, "chunk_id": row[0], "score": round(float(row[2]), 4), "text": row[1],
                 "metadata": dict(zip(meta_cols, row[3:]))} for r, row in enumerate(rows)]
