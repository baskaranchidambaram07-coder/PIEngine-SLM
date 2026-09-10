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

from core import adapters, chunking, embeddings, kbstore, telemetry, versions
from core.catalog import MODEL_CATALOG, get_model
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
    out = []
    for m in MODEL_CATALOG:
        entry = dict(m)
        entry["downloaded"] = (MODELS_DIR / m["file"]).exists()
        out.append(entry)
    return out


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
