"""Local sentence-transformers embedder (real model, no mock fallback)."""
from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Recommended BGE v1.5 instruction for short query -> passage retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class LocalEmbedder:
    """Runs on CPU. Vectors are L2-normalized so cosine == dot product."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # fails loudly if missing

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self._model.max_seq_length)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, question: str) -> list[float]:
        vec = self._model.encode([QUERY_INSTRUCTION + question], normalize_embeddings=True,
                                 show_progress_bar=False)
        return vec[0].tolist()
