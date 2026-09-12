"""Embedding wrapper around fastembed (ONNX, CPU-only, no torch), one loaded
model per catalogue entry (core/embedders.py).

The prefixes are applied HERE, explicitly, because fastembed adds none of its
own and the handset (src/llm.ts) must produce the same vector for the same
text: both sides prepend `query_prefix` to queries and `passage_prefix` to
chunks, then L2-normalise.

`EMBEDDER_ID` / `EMBEDDING_DIM` remain as the defaults for callers that never
choose a model — agents published before the catalogue existed.
"""
from __future__ import annotations

import numpy as np

from . import embedders
from .paths import EMBEDDER_CACHE

EMBEDDER_ID = embedders.default()["fastembed_model"]
EMBEDDING_DIM = embedders.default()["dim"]

_models: dict[str, object] = {}


def get_model(embedder_id: str | None = None):
    entry = embedders.get(embedder_id)
    key = entry["id"]
    if key not in _models:
        from fastembed import TextEmbedding

        _models[key] = TextEmbedding(model_name=entry["fastembed_model"], cache_dir=str(EMBEDDER_CACHE))
    return _models[key]


def dim(embedder_id: str | None = None) -> int:
    return embedders.get(embedder_id)["dim"]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_passages(texts: list[str], embedder_id: str | None = None) -> np.ndarray:
    entry = embedders.get(embedder_id)
    prefixed = [entry["passage_prefix"] + t for t in texts] if entry["passage_prefix"] else texts
    mat = np.array(list(get_model(entry["id"]).embed(prefixed)), dtype=np.float32)
    return _normalize(mat)


def embed_query(text: str, embedder_id: str | None = None) -> np.ndarray:
    entry = embedders.get(embedder_id)
    mat = np.array(list(get_model(entry["id"]).embed([entry["query_prefix"] + text])), dtype=np.float32)
    return _normalize(mat)[0]


def warm(embedder_id: str) -> dict:
    """Download (if needed) and load a model so the first upload is not slow;
    returns the entry's dimension as a smoke test."""
    vec = embed_query("warm-up", embedder_id)
    return {"id": embedders.get(embedder_id)["id"], "dim": int(vec.shape[0])}
