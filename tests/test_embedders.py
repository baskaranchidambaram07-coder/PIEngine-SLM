"""Embedding-model catalogue, dimension-aware KB store and the Studio's
per-agent embedder handling. Model-free: vectors are synthetic."""
from __future__ import annotations

import numpy as np
import pytest

from core import embedders, kbstore


def test_catalogue_entries_are_complete_and_unique():
    ids = [e["id"] for e in embedders.EMBEDDERS]
    assert len(ids) == len(set(ids))
    for e in embedders.EMBEDDERS:
        for k in ("fastembed_model", "dim", "pooling", "query_prefix", "passage_prefix",
                  "gguf_file", "gguf_url", "gguf_size_bytes", "max_tokens"):
            assert k in e, (e["id"], k)
        assert e["pooling"] in ("cls", "mean")
        assert e["gguf_url"].startswith("https://huggingface.co/")
    assert embedders.default()["id"] == "bge-small-en-v1.5"


def test_unknown_or_legacy_ids_fall_back_to_default():
    assert embedders.get(None)["id"] == "bge-small-en-v1.5"
    assert embedders.get("")["id"] == "bge-small-en-v1.5"
    assert embedders.get("does-not-exist")["id"] == "bge-small-en-v1.5"
    assert embedders.get("bge-base-en-v1.5")["dim"] == 768


def test_manifest_entry_carries_what_the_device_needs():
    m = embedders.manifest_entry(embedders.get("nomic-embed-text-v1.5"))
    assert m["file"].endswith(".gguf") and m["download_url"] and m["size_bytes"] > 0
    assert m["dim"] == 768 and m["pooling"] == "mean"
    assert m["query_prefix"] == "search_query: " and m["passage_prefix"] == "search_document: "


def _unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_kbstore_reads_dimension_from_the_vectors(tmp_path):
    kb = kbstore.connect(tmp_path / "kb.sqlite")
    vecs = _unit(np.random.RandomState(0).randn(3, 768).astype(np.float32))
    kbstore.add_document(kb, "doc", ["a", "b", "c"], vecs, embedder_id="bge-base-en-v1.5")
    assert kbstore.dimension(kb) == 768
    assert kbstore.embedder_of(kb) == "bge-base-en-v1.5"
    hits = kbstore.search(kb, vecs[1], top_k=1)
    assert hits[0]["text"] == "b" and hits[0]["score"] > 0.99


def test_kbstore_refuses_a_query_of_the_wrong_width(tmp_path):
    kb = kbstore.connect(tmp_path / "kb.sqlite")
    vecs = _unit(np.random.RandomState(1).randn(2, 384).astype(np.float32))
    kbstore.add_document(kb, "doc", ["a", "b"], vecs)
    assert kbstore.dimension(kb) == 384 and kbstore.embedder_of(kb) is None   # legacy KB
    with pytest.raises(ValueError, match="re-embedded"):
        kbstore.search(kb, np.zeros(768, dtype=np.float32), top_k=1)


def test_studio_reembeds_when_the_embedder_changes(tmp_path, monkeypatch):
    from studio import app as studio

    monkeypatch.setattr(studio, "kb_path", lambda aid: tmp_path / f"{aid}.sqlite")
    kb = kbstore.connect(studio.kb_path("acme"))
    kbstore.add_document(kb, "doc", ["alpha", "beta"], _unit(np.ones((2, 384), dtype=np.float32)))
    kb.close()

    calls = []

    def fake_passages(texts, embedder_id=None):
        calls.append((list(texts), embedder_id))
        d = embedders.get(embedder_id)["dim"]
        return _unit(np.random.RandomState(2).randn(len(texts), d).astype(np.float32))
    monkeypatch.setattr(studio.embeddings, "embed_passages", fake_passages)

    out = studio.reembed_kb("acme", "bge-base-en-v1.5")
    assert out == {"agent_id": "acme", "embedder": "bge-base-en-v1.5", "chunks": 2}
    assert calls == [(["alpha", "beta"], "bge-base-en-v1.5")]
    kb = kbstore.connect(studio.kb_path("acme"))
    assert kbstore.dimension(kb) == 768 and kbstore.embedder_of(kb) == "bge-base-en-v1.5"


def test_runtime_resolves_embedder_from_old_and_new_manifests():
    from runtime import app as rt

    assert rt.embedder_id_of({}) == "bge-small-en-v1.5"
    assert rt.embedder_id_of({"embedder": {"id": "BAAI/bge-small-en-v1.5", "dim": 384}}) == "bge-small-en-v1.5"
    assert rt.embedder_id_of({"embedder": {"id": "all-minilm-l6-v2"}}) == "all-minilm-l6-v2"
