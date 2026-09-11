"""Agent Studio — Journey 1.

Web app where a solution team designs an on-device agent for a business
scenario: pick an SLM, author the persona/system prompt, build a lightweight
knowledge base (chunk + embed into a portable SQLite file), declare tools,
and publish a versioned bundle that handheld devices download and run
fully offline.

Run from repo root:  venv\\Scripts\\python -m uvicorn studio.app:app --port 8100
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import re as _re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

from core import (adapters, chunking, downloads, embeddings, ggufmeta, kbstore,
                  telemetry, versions)
from core import catalog as catalog_mod
from core.catalog import get_model
from core.paths import BUNDLES_DIR, MODELS_DIR, ROOT

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "studio.db"
KB_DIR = APP_DIR / "kb"
KB_DIR.mkdir(exist_ok=True)

BUNDLE_SCHEMA = "slm-agent-bundle/1"

app = FastAPI(title="SLM Agent Studio")


# ---------------------------------------------------------------- storage

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS agents (
               id TEXT PRIMARY KEY,
               config TEXT NOT NULL,
               version INTEGER NOT NULL DEFAULT 0,
               updated_at TEXT NOT NULL
           )"""
    )
    return conn


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "agent"


def kb_path(agent_id: str) -> Path:
    return KB_DIR / f"{agent_id}.sqlite"


def load_agent(conn: sqlite3.Connection, agent_id: str) -> dict:
    row = conn.execute("SELECT config, version, updated_at FROM agents WHERE id = ?",
                       (agent_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"agent '{agent_id}' not found")
    cfg = json.loads(row[0])
    cfg["id"] = agent_id
    cfg["version"] = row[1]
    cfg["updated_at"] = row[2]
    return cfg


def save_agent(conn: sqlite3.Connection, agent_id: str, cfg: dict, version: int | None = None) -> None:
    stored = {k: v for k, v in cfg.items() if k not in ("id", "version", "updated_at")}
    if version is None:
        conn.execute("UPDATE agents SET config = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(stored), now(), agent_id))
    else:
        conn.execute("UPDATE agents SET config = ?, version = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(stored), version, now(), agent_id))
    conn.commit()


# ---------------------------------------------------------------- models

DEFAULT_GENERATION = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 768}
DEFAULT_RAG = {"top_k": 4, "min_score": 0.45}


class AgentIn(BaseModel):
    name: str
    description: str = ""
    scenario: str = ""
    system_prompt: str = ""
    model_id: str = "qwen3-1.7b-q4_k_m"
    # A promoted LoRA adapter from finetune/ — behaviour on top of the shared
    # base model. Empty means stock weights.
    adapter_id: str = ""
    generation: dict = DEFAULT_GENERATION
    rag: dict = DEFAULT_RAG
    tools: list[dict] = []


class SearchIn(BaseModel):
    query: str
    top_k: int = 4


# ---------------------------------------------------------------- catalog

@app.get("/api/catalog")
def catalog():
    umap = usage_map()
    loaded = loaded_model_file()
    out = []
    for m in catalog_mod.all_models():
        entry = dict(m)
        path = MODELS_DIR / m["file"]
        entry["downloaded"] = path.exists()
        entry["file_bytes"] = path.stat().st_size if entry["downloaded"] else 0
        entry["usage"] = usage_of(m, umap)
        entry["loaded"] = (loaded == m["file"])
        # What the UI may offer: the file is reclaimable whenever it is here
        # and not open; the catalog entry only when the model was onboarded.
        entry["can_delete_file"] = entry["downloaded"] and not entry["loaded"]
        entry["can_remove_entry"] = (m.get("source") == "custom")
        out.append(entry)
    return out


# --------------------------------------------------- onboarding a new model
# Models can be brought in two ways: by pointing at a GGUF on Hugging Face, or
# by uploading one. Both derive their metadata from the GGUF header rather than
# asking the user to type it — a wrong context_length reaches the device
# manifest and llama-server then truncates or over-allocates against it.

HF_HOSTS = {"huggingface.co", "hf.co"}
_MODEL_DOWNLOADS: dict[str, dict] = {}   # file -> {total, done, error}


def _require_hf_url(url: str) -> str:
    """Only Hugging Face, only https, only .gguf.

    This endpoint makes the server fetch a user-supplied URL, so an open
    allowlist would turn the Studio into an SSRF proxy onto whatever the box
    can reach.
    """
    from urllib.parse import urlparse

    u = urlparse(url.strip())
    if u.scheme != "https":
        raise HTTPException(400, "URL must be https")
    if u.hostname not in HF_HOSTS:
        raise HTTPException(400, f"only Hugging Face URLs are accepted (got '{u.hostname}')")
    if not u.path.lower().endswith(".gguf"):
        raise HTTPException(400, "URL must point directly at a .gguf file")
    # a blob link is the human page for the same object
    return url.strip().replace("/blob/", "/resolve/")


def safe_gguf_name(name: str) -> str:
    """Filename only, no traversal, must be a .gguf."""
    base = Path(str(name)).name.strip()
    if not base.lower().endswith(".gguf"):
        raise HTTPException(400, "file must be a .gguf")
    if not _re.fullmatch(r"[A-Za-z0-9._+-]+", base):
        raise HTTPException(400, f"unsafe filename '{base}'")
    return base


def _derive_entry(file_name: str, meta: dict, size_bytes: int,
                  download_url: str) -> dict:
    size_gb = round(size_bytes / 1e9, 2) if size_bytes else 0.0
    stem = file_name[:-5] if file_name.lower().endswith(".gguf") else file_name
    quant = meta.get("quantization")
    pretty = meta.get("name") or stem
    return {
        "id": slugify(stem),
        "name": f"{pretty} ({quant})" if quant and quant not in pretty else pretty,
        "file": file_name,
        # The upstream *unquantised* repo cannot be inferred from a GGUF URL,
        # and finetune/pack.py refuses to guess it, so leave it for the user.
        "hf_repo": "",
        "download_url": download_url,
        "size_gb": size_gb,
        "size_bytes": size_bytes,
        # weights + KV cache + compute buffer runs to roughly twice the file.
        "min_device_ram_gb": max(3, int(round(size_gb * 2 + 1))),
        "context_length": meta.get("context_length") or 4096,
        "license": "check the source repo",
        "notes": short_note(meta),
        "family": meta.get("architecture") or "unknown",
    }


def short_note(meta: dict) -> str:
    """A few words at most: architecture, scale, quantisation.

    The table already shows size, RAM and context, so the note only has to
    carry what those columns do not. Prose here just wraps the row.
    """
    bits = [meta.get("architecture") or "gguf"]
    if meta.get("size_label"):
        bits.append(str(meta["size_label"]))
    if meta.get("quantization"):
        bits.append(str(meta["quantization"]))
    return " ".join(bits[:3])


class InspectIn(BaseModel):
    url: str


@app.post("/api/catalog/inspect")
def inspect_model(body: InspectIn):
    """Preview what onboarding this URL would add — nothing is saved."""
    url = _require_hf_url(body.url)
    file_name = safe_gguf_name(url.split("?")[0].rsplit("/", 1)[-1])
    try:
        meta = ggufmeta.from_url(url)
    except ggufmeta.GGUFError as exc:
        raise HTTPException(400, f"not a readable GGUF: {exc}")
    except requests.RequestException as exc:
        raise HTTPException(502, f"could not read the model header: {str(exc)[:200]}")

    size_bytes = 0
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        size_bytes = int(head.headers.get("content-length") or 0)
    except (requests.RequestException, ValueError):
        pass

    entry = _derive_entry(file_name, meta, size_bytes, url)
    return {"entry": entry, "gguf": meta,
            "already_in_catalog": catalog_mod.get_model(entry["id"]) is not None,
            "already_downloaded": (MODELS_DIR / file_name).exists()}


@app.get("/api/catalog/hf-files")
def hf_files(repo: str):
    """List the GGUFs in a Hugging Face repo, so a repo URL is enough."""
    repo = repo.strip().rstrip("/")
    for host in HF_HOSTS:
        repo = repo.replace(f"https://{host}/", "")
    repo = _re.sub(r"^(models/)", "", repo).split("/tree/")[0]
    if not _re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise HTTPException(400, f"'{repo}' is not a Hugging Face <owner>/<repo>")
    try:
        r = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
        if r.status_code == 404:
            raise HTTPException(404, f"repo '{repo}' not found (or gated)")
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise HTTPException(502, f"Hugging Face unreachable: {str(exc)[:200]}")
    files = [s["rfilename"] for s in data.get("siblings", [])
             if s.get("rfilename", "").lower().endswith(".gguf")]
    return {"repo": repo, "gated": bool(data.get("gated")),
            "files": sorted(files),
            "urls": {f: f"https://huggingface.co/{repo}/resolve/main/{f}" for f in files}}


class OnboardIn(BaseModel):
    url: str
    id: str = ""
    name: str = ""
    context_length: int = 0
    min_device_ram_gb: int = 0
    license: str = ""
    notes: str = ""
    hf_repo: str = ""
    download_now: bool = False


@app.post("/api/catalog/custom")
def onboard_from_url(body: OnboardIn):
    """Add a Hugging Face GGUF to the catalog, optionally caching it here."""
    preview = inspect_model(InspectIn(url=body.url))
    entry = preview["entry"]
    for field in ("id", "name", "license", "notes", "hf_repo"):
        if getattr(body, field):
            entry[field] = getattr(body, field)
    if body.context_length:
        entry["context_length"] = body.context_length
    if body.min_device_ram_gb:
        entry["min_device_ram_gb"] = body.min_device_ram_gb
    entry["id"] = slugify(entry["id"])

    if catalog_mod.get_model(entry["id"]):
        raise HTTPException(409, f"model id '{entry['id']}' already exists")
    try:
        catalog_mod.add_custom(entry)
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    if body.download_now and not (MODELS_DIR / entry["file"]).exists():
        _start_model_download(entry)
    return {"ok": True, "entry": entry,
            "downloading": bool(body.download_now),
            "gguf": preview["gguf"]}


@app.post("/api/catalog/upload")
async def onboard_upload(file: UploadFile = File(...)):
    """Onboard a GGUF uploaded from the browser.

    Streamed to disk in chunks — these files are gigabytes and must never be
    held in memory. Written to a .part first so a failed upload cannot leave a
    truncated GGUF that looks installable.
    """
    name = safe_gguf_name(file.filename or "")
    dest = MODELS_DIR / name
    if dest.exists():
        raise HTTPException(409, f"'{name}' is already on the server")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_DIR / (name + ".part")
    size = 0
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(4 << 20):
                fh.write(chunk)
                size += len(chunk)
        try:
            meta = ggufmeta.from_file(tmp)
        except ggufmeta.GGUFError as exc:
            raise HTTPException(400, f"not a valid GGUF: {exc}")
        tmp.rename(dest)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"could not store the upload: {str(exc)[:200]}")

    # No CDN for an uploaded file: devices fetch it from the portal, which
    # already serves /models/<file> and is the fallback source in the app.
    entry = _derive_entry(name, meta, size, "")
    if catalog_mod.get_model(entry["id"]):
        entry["id"] = f"{entry['id']}-{int(datetime.now(timezone.utc).timestamp())}"
    catalog_mod.add_custom(entry)
    return {"ok": True, "entry": entry, "gguf": meta, "bytes": size}


# ----------------------------------------------------- removing models
# Two different things get confused here, so they are separate operations:
#   * deleting the GGUF frees disk but keeps the catalog entry, which simply
#     reverts to "on demand" — the model can be fetched again later;
#   * removing the entry drops the model from the catalog, and only onboarded
#     models can be removed at all, since the curated ones live in code.
# A built-in's file was previously undeletable through the UI, so a 1 GB model
# nobody used could not be reclaimed without touching the filesystem by hand.

RUNTIME_URL = "http://127.0.0.1:8200"


def usage_map() -> dict[str, dict]:
    """Who depends on each model, computed in one pass over agents + bundles.

    Per-model lookups would reopen every bundle zip for every row of the
    catalog; this walks each source once and indexes by model id and by GGUF
    filename, because agents reference the id while published manifests carry
    the filename.
    """
    out: dict[str, dict] = {}

    def slot(key: str) -> dict:
        return out.setdefault(key, {"agents": [], "bundles": [], "installed": []})

    conn = db()
    for aid, cfg_json in conn.execute("SELECT id, config FROM agents"):
        mid = json.loads(cfg_json).get("model_id")
        if mid:
            slot(mid)["agents"].append(aid)

    for p in sorted(BUNDLES_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:
                model = json.loads(z.read("manifest.json"))["model"]
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError):
            continue
        for key in {model.get("id"), model.get("file")}:
            if key:
                slot(key)["bundles"].append(p.name)

    try:
        for a in requests.get(f"{RUNTIME_URL}/api/installed", timeout=3).json():
            for key in {a["model"].get("id"), a["model"].get("file")}:
                if key:
                    slot(key)["installed"].append(a["id"])
    except (requests.RequestException, ValueError, KeyError):
        pass   # the Runtime being down must not block catalog browsing
    return out


def usage_of(entry: dict, umap: dict[str, dict] | None = None) -> dict:
    umap = umap if umap is not None else usage_map()
    by_id = umap.get(entry["id"], {})
    by_file = umap.get(entry["file"], {})
    merged = {k: sorted(set(by_id.get(k, [])) | set(by_file.get(k, [])))
              for k in ("agents", "bundles", "installed")}
    merged["in_use"] = any(merged[k] for k in ("agents", "bundles", "installed"))
    return merged


def loaded_model_file() -> str | None:
    """The GGUF llama-server currently holds open, if any.

    Windows will not unlink a memory-mapped file, so deleting the loaded model
    fails with a permission error that says nothing useful. Ask first.
    """
    try:
        st = requests.get(f"{RUNTIME_URL}/api/llm/status", timeout=3).json()
        return st.get("model") if st.get("running") else None
    except (requests.RequestException, ValueError):
        return None


@app.delete("/api/catalog/{model_id}/file")
def delete_model_file(model_id: str, force: bool = False):
    """Delete a model's GGUF from this server, freeing its disk space.

    The catalog entry stays; the model becomes "on demand" again. Refused
    outright when the file cannot be recovered or cannot be unlinked, and
    behind `force` when deleting it would merely inconvenience something that
    can re-download.
    """
    entry = catalog_mod.get_model(model_id)
    if not entry:
        raise HTTPException(404, f"unknown model '{model_id}'")
    path = MODELS_DIR / entry["file"]
    if not path.exists():
        return {"ok": True, "state": "not on server", "freed_bytes": 0}

    if loaded_model_file() == entry["file"]:
        raise HTTPException(409, f"{entry['file']} is loaded by llama-server right now — "
                                 "chat with a different model first, or restart the runtime")

    use = usage_of(entry)
    recoverable = bool(entry.get("download_url"))
    if use["in_use"] and not recoverable:
        # An uploaded model has no CDN copy: this file is the only one there is.
        raise HTTPException(409, (
            f"{entry['file']} was uploaded, so this server holds the only copy, and it is "
            f"still used by {_describe_use(use)}. Delete those first."))
    if use["in_use"] and not force:
        raise HTTPException(409, (
            f"still used by {_describe_use(use)}. Deleting the file frees "
            f"{path.stat().st_size / 1e9:.2f} GB; it will be downloaded again when needed. "
            "Pass force=true to go ahead."))

    size = path.stat().st_size
    try:
        path.unlink()
    except PermissionError:
        raise HTTPException(409, f"{entry['file']} is open by another process and "
                                 "cannot be deleted right now")
    except OSError as exc:
        raise HTTPException(500, f"could not delete {entry['file']}: {str(exc)[:200]}")
    return {"ok": True, "state": "deleted", "freed_bytes": size,
            "file": entry["file"], "still_in_catalog": True}


def _describe_use(use: dict) -> str:
    parts = []
    if use["agents"]:
        parts.append(f"{len(use['agents'])} agent(s): {', '.join(use['agents'])}")
    if use["bundles"]:
        parts.append(f"{len(use['bundles'])} published bundle(s)")
    if use["installed"]:
        parts.append(f"{len(use['installed'])} agent(s) installed on the device")
    return "; ".join(parts) or "nothing"


@app.delete("/api/catalog/custom/{model_id}")
def remove_onboarded(model_id: str, delete_file: bool = False, force: bool = False):
    """Remove an onboarded model from the catalog, optionally with its file."""
    entry = catalog_mod.get_model(model_id)
    if not entry or entry.get("source") == "builtin":
        raise HTTPException(404, f"'{model_id}' is a curated model and cannot be "
                                 "removed from the catalog — delete its file instead")
    use = usage_of(entry)
    if use["in_use"] and not force:
        raise HTTPException(409, f"still used by {_describe_use(use)}")

    removed_bytes = 0
    if delete_file:
        path = MODELS_DIR / entry["file"]
        if path.exists():
            if loaded_model_file() == entry["file"]:
                raise HTTPException(409, f"{entry['file']} is loaded by llama-server right now")
            try:
                removed_bytes = path.stat().st_size
                path.unlink()
            except OSError as exc:
                raise HTTPException(409, f"could not delete {entry['file']}: {str(exc)[:200]}")
    catalog_mod.remove_custom(model_id)
    return {"ok": True, "id": model_id, "file_deleted": bool(removed_bytes),
            "freed_bytes": removed_bytes}


@app.delete("/api/catalog/{model_id}/entry")
def remove_catalog_entry(model_id: str, delete_file: bool = True, force: bool = False):
    """Drop a model from the catalogue, whichever kind it is.

    An onboarded model is deleted outright. A curated one cannot be — it lives
    in code — so it is hidden instead: gone from the picker, still resolvable
    for agents and bundles already built on it, and restorable later.
    """
    entry = catalog_mod.get_model(model_id)
    if not entry:
        raise HTTPException(404, f"unknown model '{model_id}'")

    use = usage_of(entry)
    if use["agents"] and not force:
        raise HTTPException(409, (
            f"'{model_id}' is selected by {len(use['agents'])} agent(s): "
            f"{', '.join(use['agents'])}. Point them at another model first, "
            "or pass force=true."))

    freed = 0
    if delete_file:
        path = MODELS_DIR / entry["file"]
        if path.exists() and loaded_model_file() != entry["file"]:
            try:
                freed = path.stat().st_size
                path.unlink()
            except OSError:
                freed = 0

    if entry.get("source") == "custom":
        catalog_mod.remove_custom(model_id)
        action = "removed"
    else:
        catalog_mod.hide(model_id)
        action = "hidden"
    return {"ok": True, "id": model_id, "action": action, "freed_bytes": freed,
            "still_resolvable": action == "hidden",
            "note": ("Curated models live in code, so this one is hidden from the "
                     "catalogue rather than deleted; agents and bundles already using "
                     "it keep working, and it can be restored."
                     if action == "hidden" else "Onboarded model deleted.")}


@app.post("/api/catalog/{model_id}/restore")
def restore_catalog_entry(model_id: str):
    """Put a hidden curated model back in the catalogue."""
    if not catalog_mod.unhide(model_id):
        raise HTTPException(404, f"'{model_id}' is not hidden")
    return {"ok": True, "id": model_id, "action": "restored"}


@app.get("/api/catalog/hidden")
def list_hidden():
    hidden = catalog_mod.load_hidden()
    umap = usage_map()
    out = []
    for m in catalog_mod.all_models(include_hidden=True):
        if m["id"] not in hidden:
            continue
        entry = dict(m)
        # Hidden models keep resolving, so what still depends on one matters.
        entry["usage"] = usage_of(m, umap)
        entry["downloaded"] = (MODELS_DIR / m["file"]).exists()
        out.append(entry)
    return out


@app.get("/api/catalog/disk")
def catalog_disk():
    """What the model store costs and what is left, for the catalog header."""
    files = [p for p in MODELS_DIR.glob("*.gguf") if p.is_file()]
    used = sum(p.stat().st_size for p in files)
    try:
        free = shutil.disk_usage(MODELS_DIR).free
    except OSError:
        free = 0
    return {"models_bytes": used, "model_files": len(files), "free_bytes": free}


def _start_model_download(entry: dict) -> None:
    import threading

    f = entry["file"]
    prog = _MODEL_DOWNLOADS.setdefault(f, {"total": 0, "done": 0, "error": None})
    prog.update({"done": 0, "error": None})

    def run():
        tmp = MODELS_DIR / (f + ".part")
        try:
            with requests.get(entry["download_url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                prog["total"] = int(r.headers.get("content-length") or 0)
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        prog["done"] += len(chunk)
                tmp.rename(MODELS_DIR / f)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            prog["error"] = str(exc)[:300]
            prog["reclaimed_bytes"] = downloads.discard_partial(tmp)

    threading.Thread(target=run, daemon=True).start()


@app.post("/api/catalog/{model_id}/download")
def cache_model_here(model_id: str):
    """Pull a catalog model onto this server so devices can use the portal."""
    entry = catalog_mod.get_model(model_id)
    if not entry:
        raise HTTPException(404, f"unknown model '{model_id}'")
    if (MODELS_DIR / entry["file"]).exists():
        return {"ok": True, "state": "already downloaded"}
    if not entry.get("download_url"):
        raise HTTPException(400, "this model has no source URL (it was uploaded)")
    # Refuse up front rather than filling the disk and dying part-way. 507 is
    # Insufficient Storage — the accurate status for this.
    short = downloads.space_shortfall(MODELS_DIR, int(entry.get("size_bytes") or 0))
    if short:
        raise HTTPException(507, downloads.describe_shortfall(
            short, int(entry["size_bytes"]), MODELS_DIR))
    _start_model_download(entry)
    return {"ok": True, "state": "downloading"}


@app.get("/api/catalog/downloads")
def model_download_progress():
    return _MODEL_DOWNLOADS


# ---------------------------------------------------------------- agents

@app.get("/api/agents")
def list_agents():
    conn = db()
    rows = conn.execute("SELECT id, config, version, updated_at FROM agents ORDER BY updated_at DESC").fetchall()
    out = []
    for aid, cfg_json, version, updated in rows:
        cfg = json.loads(cfg_json)
        kbp = kb_path(aid)
        kb_stats = kbstore.stats(kbstore.connect(kbp)) if kbp.exists() else {"docs": 0, "chunks": 0}
        out.append({
            "id": aid, "name": cfg.get("name"), "description": cfg.get("description"),
            "model_id": cfg.get("model_id"), "version": version, "updated_at": updated,
            "kb": kb_stats, "tools": len(cfg.get("tools", [])),
        })
    return out


@app.post("/api/agents")
def create_agent(body: AgentIn):
    conn = db()
    base = slugify(body.name)
    agent_id, n = base, 2
    while conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone():
        agent_id = f"{base}-{n}"
        n += 1
    cfg = body.model_dump()
    conn.execute("INSERT INTO agents (id, config, version, updated_at) VALUES (?, ?, 0, ?)",
                 (agent_id, json.dumps(cfg), now()))
    conn.commit()
    return load_agent(conn, agent_id)


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str):
    conn = db()
    cfg = load_agent(conn, agent_id)
    kbp = kb_path(agent_id)
    cfg["kb"] = kbstore.stats(kbstore.connect(kbp)) if kbp.exists() else {"docs": 0, "chunks": 0}
    return cfg


@app.put("/api/agents/{agent_id}")
def update_agent(agent_id: str, body: AgentIn):
    conn = db()
    load_agent(conn, agent_id)  # 404 check
    save_agent(conn, agent_id, body.model_dump())
    return load_agent(conn, agent_id)


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str):
    """Remove the agent from the server entirely.

    Works whether the agent was ever published or is still in the creation
    journey. Previously this deleted only the database row and the KB, leaving
    every published bundle on disk and still listed in the registry — so a
    "deleted" agent stayed installable from the store. It now takes the
    published artifacts with it.
    """
    conn = db()
    load_agent(conn, agent_id)

    removed = []
    for v in versions.bundle_versions(agent_id):
        path = BUNDLES_DIR / versions.bundle_name(agent_id, v)
        path.unlink(missing_ok=True)
        removed.append(v)
    write_registry([r for r in read_registry() if r["id"] != agent_id])
    versions.forget_agent(agent_id)

    conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
    conn.commit()
    kb_path(agent_id).unlink(missing_ok=True)
    return {"ok": True, "agent_id": agent_id, "bundles_removed": sorted(removed, reverse=True)}


# ---------------------------------------------------------------- knowledge base

@app.post("/api/agents/{agent_id}/docs")
async def upload_docs(agent_id: str, files: list[UploadFile] = File(...)):
    conn = db()
    load_agent(conn, agent_id)
    kb = kbstore.connect(kb_path(agent_id))
    results = []
    for f in files:
        data = await f.read()
        text = chunking.extract_text(f.filename, data)
        chunks = chunking.split_text(text)
        if not chunks:
            results.append({"name": f.filename, "error": "no extractable text"})
            continue
        vecs = embeddings.embed_passages(chunks)
        doc_id = kbstore.add_document(kb, f.filename, chunks, vecs,
                                      meta={"bytes": len(data)})
        results.append({"name": f.filename, "doc_id": doc_id, "chunks": len(chunks)})
    return {"uploaded": results, "stats": kbstore.stats(kb)}


@app.get("/api/agents/{agent_id}/docs")
def list_docs(agent_id: str):
    kbp = kb_path(agent_id)
    if not kbp.exists():
        return []
    return kbstore.list_documents(kbstore.connect(kbp))


@app.delete("/api/agents/{agent_id}/docs/{doc_id}")
def delete_doc(agent_id: str, doc_id: int):
    kbp = kb_path(agent_id)
    if not kbp.exists():
        raise HTTPException(404, "no knowledge base")
    kb = kbstore.connect(kbp)
    kbstore.delete_document(kb, doc_id)
    return {"ok": True, "stats": kbstore.stats(kb)}


@app.post("/api/agents/{agent_id}/search")
def test_search(agent_id: str, body: SearchIn):
    kbp = kb_path(agent_id)
    if not kbp.exists():
        return {"results": []}
    kb = kbstore.connect(kbp)
    qvec = embeddings.embed_query(body.query)
    return {"results": kbstore.search(kb, qvec, top_k=body.top_k)}


# ---------------------------------------------------------------- publish

def registry_path() -> Path:
    return BUNDLES_DIR / "registry.json"


def read_registry() -> list[dict]:
    if registry_path().exists():
        return json.loads(registry_path().read_text(encoding="utf-8"))
    return []


@app.post("/api/agents/{agent_id}/publish")
def publish(agent_id: str):
    conn = db()
    cfg = load_agent(conn, agent_id)
    model = get_model(cfg["model_id"])
    if not model:
        raise HTTPException(400, f"unknown model '{cfg['model_id']}'")

    version = cfg["version"] + 1
    kbp = kb_path(agent_id)
    kb_stats = kbstore.stats(kbstore.connect(kbp)) if kbp.exists() else {"docs": 0, "chunks": 0}

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "id": agent_id,
        "name": cfg["name"],
        "version": version,
        "description": cfg.get("description", ""),
        "scenario": cfg.get("scenario", ""),
        "published_at": now(),
        "system_prompt": cfg.get("system_prompt", ""),
        "generation": cfg.get("generation", DEFAULT_GENERATION),
        "rag": {**cfg.get("rag", DEFAULT_RAG), **kb_stats},
        "tools": cfg.get("tools", []),
        "model": {
            "id": model["id"], "name": model["name"], "file": model["file"],
            "download_url": model["download_url"], "size_gb": model["size_gb"],
            "size_bytes": model.get("size_bytes"),
            "context_length": model["context_length"], "family": model["family"],
        },
        "embedder": {"id": embeddings.EMBEDDER_ID, "dim": embeddings.EMBEDDING_DIM},
    }

    # A tuned agent carries its adapter descriptor, not the adapter file: the
    # device fetches ~30 MB once from /adapters/<file> and keeps using the base
    # GGUF it already has. Only a PROMOTED adapter is allowed through — see
    # core/adapters.manifest_entry.
    adapter_id = (cfg.get("adapter_id") or "").strip()
    if adapter_id:
        try:
            entry = adapters.manifest_entry(adapter_id)
        except (KeyError, ValueError) as exc:
            # str(KeyError) wraps the message in quotes; args[0] is the message
            raise HTTPException(400, exc.args[0] if exc.args else str(exc))
        if entry["base_model_id"] != model["id"]:
            raise HTTPException(
                400, f"adapter '{adapter_id}' was trained against "
                     f"{entry['base_model_id']}, but this agent runs {model['id']} — "
                     f"applying a LoRA to the wrong base produces garbage, not an error")
        manifest["adapter"] = entry

    bundle_name = f"{agent_id}-v{version}.zip"
    bundle_file = BUNDLES_DIR / bundle_name
    with zipfile.ZipFile(bundle_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        if kbp.exists():
            zf.write(kbp, "kb.sqlite")

    registry = [r for r in read_registry() if r["id"] != agent_id]
    registry.append({
        "id": agent_id, "name": cfg["name"], "description": cfg.get("description", ""),
        "version": version, "published_at": manifest["published_at"],
        "bundle": bundle_name, "bundle_bytes": bundle_file.stat().st_size,
        "model": manifest["model"], "adapter": manifest.get("adapter"),
        "kb": kb_stats, "tools": len(manifest["tools"]),
    })
    registry_path().write_text(json.dumps(registry, indent=2), encoding="utf-8")

    save_agent(conn, agent_id, {k: v for k, v in cfg.items() if k not in ("kb",)}, version=version)
    return {"ok": True, "version": version, "bundle": bundle_name}


def write_registry(entries: list[dict]) -> None:
    registry_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def registry_entry_from_bundle(agent_id: str, version: int) -> dict | None:
    """Rebuild a registry entry from a published bundle's own manifest.

    Promoting an older version after the current one is retired needs that
    version's metadata as it was at publish time — which lives in the bundle,
    not in the (since-edited) agent config.
    """
    path = BUNDLES_DIR / versions.bundle_name(agent_id, version)
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as zf:
        m = json.loads(zf.read("manifest.json"))
    rag = m.get("rag", {})
    return {
        "id": agent_id, "name": m.get("name", agent_id),
        "description": m.get("description", ""),
        "version": version, "published_at": m.get("published_at", ""),
        "bundle": path.name, "bundle_bytes": path.stat().st_size,
        "model": m.get("model"), "adapter": m.get("adapter"),
        "kb": {"docs": rag.get("docs", 0), "chunks": rag.get("chunks", 0)},
        "tools": len(m.get("tools", [])),
    }


def reconcile_registry(agent_id: str) -> int | None:
    """Point the store at this agent's newest ACTIVE version, or drop it.

    Called after any disable/enable/delete so devices never see a retired
    version. Returns the version now published, or None if the agent no longer
    has one.
    """
    registry = [r for r in read_registry() if r["id"] != agent_id]
    newest = versions.newest_active(agent_id)
    if newest is not None:
        entry = registry_entry_from_bundle(agent_id, newest)
        if entry:
            registry.append(entry)
        else:
            newest = None
    write_registry(registry)
    return newest


@app.get("/api/agents/{agent_id}/versions")
def list_versions(agent_id: str):
    conn = db()
    load_agent(conn, agent_id)  # 404 if the agent is gone
    current = next((r["version"] for r in read_registry() if r["id"] == agent_id), None)
    return {"agent_id": agent_id, "published_version": current,
            "versions": versions.describe(agent_id, current)}


class VersionState(BaseModel):
    state: str  # "active" | "disabled"


@app.post("/api/agents/{agent_id}/versions/{version}/state")
def set_version_state(agent_id: str, version: int, body: VersionState):
    """Disable (or re-enable) one published version.

    A disabled version stays on disk and can be re-enabled; it is simply not
    servable and cannot be run on device or web.
    """
    conn = db()
    load_agent(conn, agent_id)
    if body.state not in ("active", "disabled"):
        raise HTTPException(400, "state must be 'active' or 'disabled'")
    if versions.state_of(agent_id, version) == versions.DELETED:
        raise HTTPException(409, f"v{version} is deleted — deletion is not reversible")
    if version not in versions.bundle_versions(agent_id):
        raise HTTPException(404, f"v{version} has no bundle on disk")

    versions.set_state(agent_id, version, body.state)
    now_published = reconcile_registry(agent_id)
    return {"ok": True, "agent_id": agent_id, "version": version,
            "state": body.state, "published_version": now_published}


@app.delete("/api/agents/{agent_id}/versions/{version}")
def delete_version(agent_id: str, version: int):
    """Delete one published version: its bundle is removed from the server.

    The agent and its other versions are untouched — deleting an outdated v1
    leaves v9 installable. If the deleted version was the published one, the
    newest remaining active version is promoted in its place.
    """
    conn = db()
    load_agent(conn, agent_id)
    if version not in versions.known_versions(agent_id):
        raise HTTPException(404, f"'{agent_id}' has no v{version}")

    (BUNDLES_DIR / versions.bundle_name(agent_id, version)).unlink(missing_ok=True)
    versions.set_state(agent_id, version, versions.DELETED)
    now_published = reconcile_registry(agent_id)
    return {"ok": True, "agent_id": agent_id, "deleted_version": version,
            "published_version": now_published}


@app.get("/api/published")
def published():
    # Defensive: reconcile keeps this true, but never advertise a version whose
    # bundle has gone missing or been retired underneath us.
    return [r for r in read_registry()
            if versions.is_servable(r["id"], r["version"])]


@app.get("/bundles/{name}")
def download_bundle(name: str):
    path = (BUNDLES_DIR / name).resolve()
    if path.parent != BUNDLES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "bundle not found")
    return FileResponse(path, filename=name)


# ---------------------------------------------------------------- adapters
# Scenario fine-tunes, produced by the finetune/ pipeline. The Studio only
# reads this registry: adapters are imported and gated from the CLI, because
# promotion depends on a scorecard that takes minutes to produce.

@app.get("/api/adapters")
def list_adapters(base_model_id: str | None = None, promoted_only: bool = False):
    rows = adapters.load()
    if base_model_id:
        rows = [a for a in rows if a["base_model_id"] == base_model_id]
    if promoted_only:
        rows = [a for a in rows if a["status"] == "promoted"]
    return rows


@app.get("/adapters/{name}")
def download_adapter(name: str):
    path = (adapters.ADAPTERS_DIR / name).resolve()
    if path.parent != adapters.ADAPTERS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "adapter not found")
    return FileResponse(path, filename=name, media_type="application/octet-stream")


# ---------------------------------------------------------------- governance
# Read-only views over the shared telemetry store (written by the runtime +
# Android app). Usage metadata only — no chat content is ever collected.

@app.get("/api/governance/overview")
def governance_overview(days: int = 30):
    return {
        "summary": telemetry.summary(days),
        "per_agent": telemetry.per_agent(days),
        "per_model": telemetry.per_model(days),
        "daily": telemetry.daily_activity(min(days, 21)),
        "devices": telemetry.devices(days),
    }


@app.get("/api/governance/events")
def governance_events(limit: int = 120, event: str | None = None,
                      agent_id: str | None = None):
    return telemetry.recent_events(limit=limit, event=event, agent_id=agent_id)


# ---------------------------------------------------------------- landing page

CF_TUNNEL_JSON = ROOT / "cloudflare" / "tunnel.json"


def _named_tunnel() -> dict | None:
    """Stable hostnames of the Cloudflare *named* tunnel, once set up.

    Written by cloudflare/setup.ps1. Named-tunnel hostnames never rotate, so
    they always win over a quick-tunnel URL scraped out of the log."""
    try:
        cfg = json.loads(CF_TUNNEL_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cfg.get("portal_url") and cfg.get("studio_url"):
        return cfg
    return None


def _quick_tunnel_url() -> str | None:
    """Last URL minted by a cloudflare *quick* tunnel — rotates every restart."""
    for log in (ROOT / "cloudflared.log", ROOT / "logs" / "cloudflared.log"):
        if not log.exists():
            continue
        found = _re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com",
                            log.read_text(encoding="utf-8", errors="ignore"))
        if found:
            return found[-1]
    return None


def _runtime_public_url() -> str:
    """Current Device Runtime URL for the landing page's live-demo links."""
    named = _named_tunnel()
    if named:
        return named["portal_url"]
    quick = _quick_tunnel_url()
    if quick:
        return quick
    return "http://localhost:8200"


@app.get("/api/portal-url")
def api_portal_url():
    """Discovery: where is the Device Runtime portal right now?

    Handsets are pinned to THIS host (a stable domain) and re-resolve the
    portal URL from here whenever their saved one stops answering — so a
    rotating quick-tunnel URL no longer strands an installed app. Once the
    named tunnel is in place the answer is permanent and this just confirms it.
    """
    named = _named_tunnel()
    if named:
        return {"url": named["portal_url"], "source": "named-tunnel", "stable": True,
                "studio_url": named["studio_url"],
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    quick = _quick_tunnel_url()
    return {"url": quick or "http://localhost:8200",
            "source": "quick-tunnel" if quick else "local",
            "stable": False,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/welcome", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing():
    html = (APP_DIR / "static" / "landing.html").read_text(encoding="utf-8")
    html = html.replace("{{RUNTIME_URL}}", _runtime_public_url()).replace("{{STUDIO}}", "/")
    return HTMLResponse(html)


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
