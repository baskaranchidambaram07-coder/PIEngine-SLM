"""The SLM catalog behind the Studio model picker.

Nothing ships here: the catalog is whatever this deployment has onboarded,
from a Hugging Face GGUF URL or an upload, stored in a JSON side-car next to
the model files. Adding and removing models is therefore an ordinary runtime
action rather than a code change.

An entry describes a 4-bit GGUF small enough for the target devices (iPhone
14+, Galaxy S23+, i.e. >= 6-8GB with ~2-3GB usable for an app). `download_url`
is where a device or the Studio fetches it; an uploaded model has none,
because this server holds the only copy.

`hf_repo` is the upstream (unquantised) Hugging Face repo the GGUF was built
from. Fine-tuning trains against that, never against the GGUF, so an entry
without an hf_repo cannot be the base of an adapter — finetune/pack.py refuses
the job rather than guessing the mapping. It cannot be inferred from a GGUF
URL, so onboarding leaves it blank for the user to fill in.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import MODELS_DIR

# No models ship with the product. Every model in the catalogue is one this
# deployment onboarded — from a Hugging Face GGUF URL or an upload — and can
# be removed again the same way. A curated list was previously baked in here;
# it was retired because entries nobody wanted could not be deleted, only
# hidden, and would accumulate as the list grew.
#
# Kept as an empty list rather than deleted outright: all_models() and
# add_custom() still consult it, so a deployment that wants opinionated
# defaults can reinstate them here without touching anything else.
MODEL_CATALOG: list[dict] = []


# ------------------------------------------------------------ onboarded models
# Every model in the catalogue lives here — a JSON side-car beside the GGUF
# files it describes. Keeping it in MODELS_DIR means the catalogue travels with
# the model store and stays out of git, so one deployment's models never become
# another's defaults.

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
