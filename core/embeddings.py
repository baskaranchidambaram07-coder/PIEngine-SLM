"""Embedding wrapper around fastembed (ONNX, CPU-only, no torch).

BAAI/bge-small-en-v1.5: 384-dim, ~130MB ONNX — small enough to ship on a
phone alongside the SLM, which is why the same model is referenced in the
agent bundle manifest.
"""
from __future__ import annotations

import numpy as np

from .paths import EMBEDDER_CACHE

EMBEDDER_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=EMBEDDER_ID, cache_dir=str(EMBEDDER_CACHE))
    return _model


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_passages(texts: list[str]) -> np.ndarray:
    mat = np.array(list(get_model().embed(texts)), dtype=np.float32)
    return _normalize(mat)


def embed_query(text: str) -> np.ndarray:
    mat = np.array(list(get_model().query_embed([text])), dtype=np.float32)
    return _normalize(mat)[0]
