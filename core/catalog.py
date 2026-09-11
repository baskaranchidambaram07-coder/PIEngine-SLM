"""Curated SLM catalog for the Studio model picker.

Every entry is a 4-bit GGUF that fits comfortably in the RAM budget of the
target devices (iPhone 14+, Galaxy S23+, i.e. >= 6-8GB with ~2-3GB usable
for an app). download_url lets the Studio (or the device runtime) pull the
file on demand; license notes flag gated repos.

`hf_repo` is the upstream (unquantised) Hugging Face repo the GGUF was
built from. Fine-tuning trains against that, never against the GGUF, so a
catalog entry without an hf_repo cannot be the base of an adapter —
finetune/pack.py refuses the job rather than guessing the mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import MODELS_DIR

MODEL_CATALOG = [
    {
        "id": "qwen3-1.7b-q4_k_m",
        "name": "Qwen3 1.7B (Q4_K_M)",
        "file": "Qwen3-1.7B-Q4_K_M.gguf",
        "hf_repo": "Qwen/Qwen3-1.7B",
        "download_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        "size_gb": 1.03,
        "size_bytes": 1107409472,
        "min_device_ram_gb": 4,
        "context_length": 8192,
        "license": "Apache-2.0 (ungated)",
        "notes": "Default. Strong tool-calling for its size; hybrid thinking mode (disabled via /no_think). Best quality/speed balance for iPhone 14 / S23 class devices.",
        "family": "qwen3",
    },
    {
        "id": "qwen3-0.6b-q4_k_m",
        "name": "Qwen3 0.6B (Q4_K_M)",
        "file": "Qwen3-0.6B-Q4_K_M.gguf",
        "hf_repo": "Qwen/Qwen3-0.6B",
        "download_url": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        "size_gb": 0.37,
        "size_bytes": 396705472,
        "min_device_ram_gb": 3,
        "context_length": 8192,
        "license": "Apache-2.0 (ungated)",
        "notes": "Fastest option — 30+ tok/s on recent phones. Good for classification / extraction agents; weaker at multi-step reasoning.",
        "family": "qwen3",
    },
    {
        "id": "qwen3-4b-instruct-2507-q4_k_m",
        "name": "Qwen3 4B Instruct 2507 (Q4_K_M)",
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "hf_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "download_url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "size_gb": 2.33,
        "size_bytes": 2497281120,
        "min_device_ram_gb": 8,
        "context_length": 16384,
        "license": "Apache-2.0 (ungated)",
        "notes": "Highest quality that still runs on 8GB-RAM flagships (iPhone 15 Pro+, S23 Ultra+). Non-thinking instruct variant.",
        "family": "qwen3",
    },
    {
        "id": "llama-3.2-3b-instruct-q4_k_m",
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "hf_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "download_url": "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.88,
        "size_bytes": 2019377600,
        "min_device_ram_gb": 6,
        "context_length": 8192,
        "license": "Llama 3.2 Community License",
        "notes": "Good general assistant; widely benchmarked on mobile (ExecuTorch reference model).",
        "family": "llama3",
    },
    {
        "id": "gemma-3-4b-it-q4_k_m",
        "name": "Gemma 3 4B IT (Q4_K_M)",
        "file": "gemma-3-4b-it-Q4_K_M.gguf",
        "hf_repo": "google/gemma-3-4b-it",
        "download_url": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.32,
        "size_bytes": 2489894016,
        "min_device_ram_gb": 8,
        "context_length": 8192,
        "license": "Gemma Terms (acceptance required)",
        "notes": "Strong multilingual + summarization. First-class on Android via Google AI Edge / MediaPipe.",
        "family": "gemma3",
    },
]


# ------------------------------------------------------------ onboarded models
# The curated list above ships in code. Models a user onboards at runtime — from
# a Hugging Face URL or an upload — cannot, so they live in a JSON side-car
# beside the GGUF files they describe. Keeping them in MODELS_DIR means the
# catalog travels with the model store and stays out of git.

CUSTOM_FIELDS = ("id", "name", "file", "hf_repo", "download_url", "size_gb",
                 "size_bytes", "min_device_ram_gb", "context_length", "license",
                 "notes", "family")


def custom_path() -> Path:
    return MODELS_DIR / "custom_catalog.json"


def load_custom() -> list[dict]:
    p = custom_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # A corrupt side-car must not take the whole catalog down; the curated
        # models still work and the file can be repaired.
        return []


def save_custom(entries: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    custom_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ------------------------------------------------------------- hidden models
# A curated entry cannot be deleted — it lives in code — but a deployment may
# not want it offered. Hiding suppresses it from the catalogue listing while
# leaving get_model() able to resolve it, so agents and published bundles that
# already reference the model keep working. Reversible, and per-deployment.

def hidden_path() -> Path:
    return MODELS_DIR / "hidden_models.json"


def load_hidden() -> set[str]:
    p = hidden_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError):
        return set()


def save_hidden(ids: set[str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hidden_path().write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")


def hide(model_id: str) -> bool:
    ids = load_hidden()
    if model_id in ids:
        return False
    ids.add(model_id)
    save_hidden(ids)
    return True


def unhide(model_id: str) -> bool:
    ids = load_hidden()
    if model_id not in ids:
        return False
    ids.discard(model_id)
    save_hidden(ids)
    return True


def all_models(include_hidden: bool = False) -> list[dict]:
    """Curated models first, then onboarded ones, each tagged with its source.

    Hidden models are left out unless asked for; `get_model` always searches
    them, because something already published may still depend on one.
    """
    hidden = load_hidden()
    out = [{**m, "source": "builtin", "hidden": m["id"] in hidden} for m in MODEL_CATALOG]
    known = {m["id"] for m in out}
    for m in load_custom():
        if m.get("id") not in known:
            out.append({**m, "source": "custom", "hidden": m.get("id") in hidden})
    return out if include_hidden else [m for m in out if not m["hidden"]]


def add_custom(entry: dict) -> dict:
    if any(m["id"] == entry["id"] for m in MODEL_CATALOG):
        raise ValueError(f"'{entry['id']}' is a built-in model id")
    entries = [m for m in load_custom() if m.get("id") != entry["id"]]
    entries.append(entry)
    save_custom(entries)
    return entry


def remove_custom(model_id: str) -> bool:
    entries = load_custom()
    kept = [m for m in entries if m.get("id") != model_id]
    if len(kept) == len(entries):
        return False
    save_custom(kept)
    return True


def get_model(model_id: str) -> dict | None:
    """Resolve a model id, hidden models included.

    Hiding removes a model from the picker, not from the system: agents and
    published bundles that already reference it must keep resolving, or
    hiding one would break everything built on it.
    """
    for m in all_models(include_hidden=True):
        if m["id"] == model_id:
            return m
    return None
