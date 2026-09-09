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

from core import chunking, embeddings, kbstore, telemetry
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
DEFAULT_RAG = {"top_k": 4, "min_score": 0.35}


class AgentIn(BaseModel):
    name: str
    description: str = ""
    scenario: str = ""
    system_prompt: str = ""
    model_id: str = "qwen3-1.7b-q4_k_m"
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
    conn = db()
    load_agent(conn, agent_id)
    conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
    conn.commit()
    kb_path(agent_id).unlink(missing_ok=True)
    return {"ok": True}


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
        "model": manifest["model"], "kb": kb_stats, "tools": len(manifest["tools"]),
    })
    registry_path().write_text(json.dumps(registry, indent=2), encoding="utf-8")

    save_agent(conn, agent_id, {k: v for k, v in cfg.items() if k not in ("kb",)}, version=version)
    return {"ok": True, "version": version, "bundle": bundle_name}


@app.get("/api/published")
def published():
    return read_registry()


@app.get("/bundles/{name}")
def download_bundle(name: str):
    path = (BUNDLES_DIR / name).resolve()
    if path.parent != BUNDLES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "bundle not found")
    return FileResponse(path, filename=name)


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
