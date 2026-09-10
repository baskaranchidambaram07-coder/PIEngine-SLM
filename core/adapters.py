"""Registry of trained LoRA adapters.

An adapter is a small file that changes how a base model behaves without
changing which base model is on the device. That distinction drives the whole
deployment story here:

  * a merged fine-tune means every agent that wants tuned behaviour ships its
    own ~1 GB Q4_K_M, and a fleet of five tuned agents is five gigabytes over
    a metered tunnel;
  * an adapter is ~20-60 MB against the ONE base model every agent already
    shares, and llama.cpp (`--lora`, `POST /lora-adapters`) and llama.rn
    (`lora_list`, `applyLoraAdapters`) both apply it at load time.

So the bundle carries the adapter and the base stays shared. It also makes
rollback a config change rather than a re-download: drop the adapter from the
manifest, republish, devices go back to stock weights on the next update.

Adapters live in models/adapters/ next to the GGUFs they modify, and the
runtime serves them over the portal the same way it serves models.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import MODELS_DIR

ADAPTERS_DIR = MODELS_DIR / "adapters"
REGISTRY = ADAPTERS_DIR / "registry.json"

# Applying an adapter to a base it was not trained against produces garbage
# rather than an error, so the pairing is recorded and checked at publish time.
STATUS = ("imported", "promoted", "rejected")


def _ensure() -> None:
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


def load() -> list[dict]:
    _ensure()
    if not REGISTRY.exists():
        return []
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def save(entries: list[dict]) -> None:
    _ensure()
    REGISTRY.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def get(adapter_id: str) -> dict | None:
    return next((a for a in load() if a["id"] == adapter_id), None)


def path(adapter_id: str) -> Path | None:
    entry = get(adapter_id)
    if not entry:
        return None
    p = ADAPTERS_DIR / entry["file"]
    return p if p.exists() else None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def register(adapter_id: str, file: str, base_model_id: str, agent: str,
             spec_sha256: str | None = None, train_log: dict | None = None) -> dict:
    """Record a freshly imported adapter. Status starts at 'imported' — only a
    passing scorecard moves it to 'promoted', and only a promoted adapter may
    be attached to a published bundle."""
    _ensure()
    p = ADAPTERS_DIR / file
    if not p.exists():
        raise FileNotFoundError(f"adapter file missing: {p}")
    entry = {
        "id": adapter_id,
        "file": file,
        "base_model_id": base_model_id,
        "agent": agent,
        "size_bytes": p.stat().st_size,
        "sha256": sha256(p),
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_sha256": spec_sha256,
        "status": "imported",
        "scorecard": None,
        "train_log": {k: train_log.get(k) for k in
                      ("seconds", "gpu", "n_train", "n_val", "hyperparameters")} if train_log else None,
    }
    entries = [a for a in load() if a["id"] != adapter_id]
    entries.append(entry)
    save(entries)
    return entry


def set_status(adapter_id: str, status: str, scorecard: dict | None = None) -> dict:
    if status not in STATUS:
        raise ValueError(f"status must be one of {STATUS}")
    entries = load()
    entry = next((a for a in entries if a["id"] == adapter_id), None)
    if not entry:
        raise KeyError(f"unknown adapter {adapter_id!r}")
    entry["status"] = status
    entry["decided_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if scorecard is not None:
        entry["scorecard"] = {
            "score": scorecard.get("score"),
            "regression": scorecard.get("regression", {}).get("pass_rate"),
            "requirements": {r["id"]: r["pass_rate"] for r in scorecard.get("requirements", [])},
            "at": scorecard.get("at"),
        }
    save(entries)
    return entry


def manifest_entry(adapter_id: str, scale: float = 1.0) -> dict:
    """The `adapter` block a published bundle carries to devices."""
    entry = get(adapter_id)
    if not entry:
        raise KeyError(f"unknown adapter {adapter_id!r}")
    if entry["status"] != "promoted":
        raise ValueError(
            f"adapter {adapter_id!r} is '{entry['status']}', not 'promoted' — run "
            f"`python -m finetune evaluate --adapter {adapter_id}` and pass the gate first")
    return {
        "id": entry["id"],
        "file": entry["file"],
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
        "base_model_id": entry["base_model_id"],
        "scale": scale,
    }
