"""The embedding-model catalogue.

An agent's knowledge base is embedded once in the Studio and searched on the
device, so the SAME model must exist in both worlds: as a fastembed ONNX model
on the server and as a llama.cpp GGUF on the handset. Every entry here names
both, plus the three things that make two implementations produce the same
vector for the same text:

  * `pooling`         — how token vectors become one vector (CLS or mean);
  * `query_prefix`    — instruction text prepended to a QUERY before embedding;
  * `passage_prefix`  — text prepended to each PASSAGE (chunk) at index time.

fastembed 0.8 adds no prefixes on its own (checked: none in its source), and
llama.cpp reads pooling from the GGUF header, so both sides apply exactly the
strings below through core/embeddings.py and the app's src/llm.ts. Change an
entry and the KB built with it must be re-embedded — the Studio does that when
an agent's embedder changes.

Parity between the two implementations is measured, not assumed:
prototypes/embedder_parity.py embeds the same sentences with fastembed and
with llama-server on this box and reports the cosine per model. Measured
2026-09-12: bge-small/base/large 0.9998, all-MiniLM 0.9997, nomic 0.998.
paraphrase-multilingual-MiniLM-L12-v2 was dropped: no GGUF of it loads in
llama.cpp (XLM-R tokenizer), so it cannot run on the handset.
"""
from __future__ import annotations

from .paths import EMBEDDER_CACHE, MODELS_DIR

DEFAULT_EMBEDDER_ID = "bge-small-en-v1.5"

BGE_QUERY = "Represent this sentence for searching relevant passages: "

EMBEDDERS: list[dict] = [
    {
        "id": "bge-small-en-v1.5",
        "name": "BGE small (English, 384-d)",
        "fastembed_model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "pooling": "cls",
        "query_prefix": BGE_QUERY,
        "passage_prefix": "",
        "max_tokens": 512,
        "server_size_mb": 67,
        "gguf_file": "bge-small-en-v1.5-q8_0.gguf",
        "gguf_url": "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        "gguf_size_bytes": 36806944,
        "languages": "English",
        "license": "MIT",
        "notes": "Default. Smallest and fastest; verified 0.9998 server/phone parity.",
    },
    {
        "id": "bge-base-en-v1.5",
        "name": "BGE base (English, 768-d)",
        "fastembed_model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "pooling": "cls",
        "query_prefix": BGE_QUERY,
        "passage_prefix": "",
        "max_tokens": 512,
        "server_size_mb": 210,
        "gguf_file": "bge-base-en-v1.5-q8_0.gguf",
        "gguf_url": "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q8_0.gguf",
        "gguf_size_bytes": 117998272,
        "languages": "English",
        "license": "MIT",
        "notes": "Better retrieval than small at 3x the size; 2x the vector storage.",
    },
    {
        "id": "bge-large-en-v1.5",
        "name": "BGE large (English, 1024-d)",
        "fastembed_model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "pooling": "cls",
        "query_prefix": BGE_QUERY,
        "passage_prefix": "",
        "max_tokens": 512,
        "server_size_mb": 1200,
        "gguf_file": "bge-large-en-v1.5-q8_0.gguf",
        "gguf_url": "https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5-q8_0.gguf",
        "gguf_size_bytes": 358155232,
        "languages": "English",
        "license": "MIT",
        "notes": "Highest quality of the BGE family; heavy for a handset (358 MB, slow per chunk).",
    },
    {
        "id": "all-minilm-l6-v2",
        "name": "all-MiniLM-L6-v2 (English, 384-d)",
        "fastembed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "pooling": "mean",
        "query_prefix": "",
        "passage_prefix": "",
        "max_tokens": 256,
        "server_size_mb": 90,
        "gguf_file": "all-MiniLM-L6-v2-q8_0.gguf",
        "gguf_url": "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
        "gguf_size_bytes": 24981152,
        "languages": "English",
        "license": "Apache-2.0",
        "notes": "Tiny and very fast. 256-token limit: chunks longer than ~1,000 characters are truncated.",
    },
    {
        "id": "nomic-embed-text-v1.5",
        "name": "Nomic Embed v1.5 (English, 768-d, long context)",
        "fastembed_model": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "pooling": "mean",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "max_tokens": 8192,
        "server_size_mb": 520,
        "gguf_file": "nomic-embed-text-v1.5-q8_0.gguf",
        "gguf_url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf",
        "gguf_size_bytes": 146132608,
        "languages": "English",
        "license": "Apache-2.0",
        "notes": "8k-token context; needs its query/document prefixes (applied automatically).",
    },
]

_BY_ID = {e["id"]: e for e in EMBEDDERS}


def get(embedder_id: str | None) -> dict:
    """Resolve an id; unknown or missing ids fall back to the default so an
    agent published before this catalogue existed keeps working."""
    return _BY_ID.get(embedder_id or DEFAULT_EMBEDDER_ID, _BY_ID[DEFAULT_EMBEDDER_ID])


def default() -> dict:
    return _BY_ID[DEFAULT_EMBEDDER_ID]


def server_cached(entry: dict) -> bool:
    """Has fastembed already downloaded this model into the local cache?

    fastembed stores models under hub-style folders whose repo may differ from
    the model name (e.g. models--qdrant--bge-small-en-v1.5-onnx-q for
    BAAI/bge-small-en-v1.5), so match on the model's own short name."""
    short = entry["fastembed_model"].split("/")[-1].lower()
    return any(p.is_dir() and short in p.name.lower() for p in EMBEDDER_CACHE.glob("models--*"))


def device_cached(entry: dict) -> bool:
    """Is the GGUF the handsets download cached on this portal?"""
    return (MODELS_DIR / entry["gguf_file"]).exists()


def describe(entry: dict) -> dict:
    """Public view for the Studio catalogue and the bundle manifest."""
    return {
        "id": entry["id"], "name": entry["name"], "dim": entry["dim"], "pooling": entry["pooling"],
        "query_prefix": entry["query_prefix"], "passage_prefix": entry["passage_prefix"],
        "max_tokens": entry["max_tokens"], "languages": entry["languages"], "license": entry["license"],
        "notes": entry["notes"], "fastembed_model": entry["fastembed_model"],
        "server_size_mb": entry["server_size_mb"],
        "file": entry["gguf_file"], "download_url": entry["gguf_url"], "size_bytes": entry["gguf_size_bytes"],
    }


def manifest_entry(entry: dict) -> dict:
    """What the bundle carries — everything the device needs to embed queries
    identically: file, where to fetch it, dimension, pooling and prefixes."""
    return {
        "id": entry["id"], "name": entry["name"], "dim": entry["dim"], "pooling": entry["pooling"],
        "query_prefix": entry["query_prefix"], "passage_prefix": entry["passage_prefix"],
        "max_tokens": entry["max_tokens"],
        "file": entry["gguf_file"], "download_url": entry["gguf_url"], "size_bytes": entry["gguf_size_bytes"],
    }
