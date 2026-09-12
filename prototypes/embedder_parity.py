"""Measure server/device embedding parity per catalogue entry.

The Studio embeds knowledge bases with fastembed (ONNX); the handset embeds
queries with llama.cpp (GGUF). If the two disagree, retrieval on the phone
silently degrades. This script embeds the same sentences both ways on this
box — fastembed via core/embeddings, llama.cpp via a throw-away llama-server
on port 8399 — and prints the cosine between the two vectors for each text,
plus the minimum. Anything below ~0.99 means pooling or prefix handling
differs and the entry should not ship.

    venv\\Scripts\\python prototypes\\embedder_parity.py            # all entries with a cached GGUF
    venv\\Scripts\\python prototypes\\embedder_parity.py bge-base-en-v1.5
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

import numpy as np
import requests

from core import embedders, embeddings
from core.paths import LLAMA_DIR, MODELS_DIR

PORT = 8399
TEXTS = [
    "What did we decide about the architecture?",
    "The Phase 2 rollout date moves from 22 September to 6 October 2026 so UAT can finish.",
    "Invoice INV-2026-0912 total due INR 35,542, payment terms 15 days net.",
    "Employees past probation may work remotely up to three days a week with line-manager approval.",
    "How many pilot sites are planned and where?",
]


def llama_embed(texts: list[str]) -> np.ndarray:
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/embeddings", json={"input": texts}, timeout=120)
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    mat = np.array([d["embedding"] for d in data], dtype=np.float32)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def run(entry: dict) -> float | None:
    gguf = MODELS_DIR / entry["gguf_file"]
    if not gguf.exists():
        print(f"{entry['id']:40s} GGUF not cached — skipped")
        return None
    proc = subprocess.Popen([str(LLAMA_DIR / "llama-server.exe"), "-m", str(gguf), "--embedding",
                             "--pooling", entry["pooling"], "--port", str(PORT), "-c", "512",
                             "-t", "4", "--no-webui"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            time.sleep(1)
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).json().get("status") == "ok":
                    break
            except requests.RequestException:
                pass
        else:
            print(f"{entry['id']:40s} llama-server did not start")
            return None
        # queries and passages exactly as both sides prepare them
        q_texts = [entry["query_prefix"] + t for t in TEXTS]
        p_texts = [entry["passage_prefix"] + t for t in TEXTS]
        fe_q = np.array([embeddings.embed_query(t, entry["id"]) for t in TEXTS])
        fe_p = embeddings.embed_passages(TEXTS, entry["id"])
        ll_q = llama_embed(q_texts)
        ll_p = llama_embed(p_texts)
        cos_q = (fe_q * ll_q).sum(axis=1)
        cos_p = (fe_p * ll_p).sum(axis=1)
        worst = float(min(cos_q.min(), cos_p.min()))
        print(f"{entry['id']:40s} dim={fe_q.shape[1]:4d} llama_dim={ll_q.shape[1]:4d} "
              f"query cos min/mean {cos_q.min():.4f}/{cos_q.mean():.4f}  "
              f"passage cos min/mean {cos_p.min():.4f}/{cos_p.mean():.4f}  "
              f"{'OK' if worst >= 0.99 else 'MISMATCH'}")
        return worst
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ids = sys.argv[1:] or [e["id"] for e in embedders.EMBEDDERS]
    results = {i: run(embedders.get(i)) for i in ids}
    bad = [i for i, v in results.items() if v is not None and v < 0.99]
    print("\nparity:", "all OK" if not bad else f"MISMATCH in {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
