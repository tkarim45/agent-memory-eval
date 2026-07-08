"""Retrieval for the 'vector memory' strategy: embed the history turn-by-turn and fetch the turns
most similar to the question. A real sentence-transformer by default, a deterministic hashing
embedder offline.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-tokens embedder. No network — tests and CI use this."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, hf_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(hf_id)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)


def top_k(embedder, query: str, chunks: list[str], k: int) -> list[int]:
    """Return the indices of the k chunks most similar to the query (descending)."""
    if not chunks:
        return []
    doc = embedder.encode(chunks)
    q = embedder.encode([query])[0]
    sims = doc @ q
    return list(np.argsort(-sims)[:k])
