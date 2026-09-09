"""Device Runtime — Journey 2.

Simulates the handheld/laptop app: browse the Agent Store (published by the
Studio), install an agent bundle, then chat fully offline — local SLM
(llama.cpp), local embeddings, local SQLite knowledge base, on-device tools.

Run from repo root:  venv\\Scripts\\python -m uvicorn runtime.app:app --port 8200
"""
from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import embeddings, kbstore, telemetry
from core.paths import BUNDLES_DIR, MODELS_DIR, ROOT

from . import export, llm, tools

APP_DIR = Path(__file__).resolve().parent
AGENTS_DIR = APP_DIR / "device_storage" / "agents"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)

STUDIO_URL = "http://127.0.0.1:8100"
MAX_TOOL_ROUNDS = 4

app = FastAPI(title="SLM Device Runtime")

# model_file -> {"total": int, "done": int, "error": str|None}
_model_downloads: dict[str, dict] = {}


# ---------------------------------------------------------------- installed agents

def installed_ids() -> list[str]:
    return sorted(p.name for p in AGENTS_DIR.iterdir() if (p / "manifest.json").exists())


def load_manifest(agent_id: str) -> dict:
    path = AGENTS_DIR / agent_id / "manifest.json"
    if not path.exists():
        raise HTTPException(404, f"agent '{agent_id}' is not installed")
    return json.loads(path.read_text(encoding="utf-8"))


def model_state(manifest: dict) -> str:
    f = manifest["model"]["file"]
    if (MODELS_DIR / f).exists():
        return "ready"
    dl = _model_downloads.get(f)
    if dl and not dl.get("error"):
        return "downloading"
    return "missing"


@app.get("/api/installed")
def api_installed():
    out = []
    for aid in installed_ids():
        m = load_manifest(aid)
        dl = _model_downloads.get(m["model"]["file"], {})
        out.append({
            "id": aid, "name": m["name"], "description": m["description"],
            "version": m["version"], "model": m["model"], "rag": m["rag"],
            "tools": [t["name"] for t in m.get("tools", [])],
            "model_state": model_state(m),
            "download": {"done": dl.get("done", 0), "total": dl.get("total", 0),
                          "error": dl.get("error")} if dl else None,
        })
    return out


@app.delete("/api/installed/{agent_id}")
def api_uninstall(agent_id: str):
    load_manifest(agent_id)
    import shutil
    shutil.rmtree(AGENTS_DIR / agent_id)
    return {"ok": True}


# ---------------------------------------------------------------- agent store

@app.get("/api/store")
def api_store():
    try:
        published = requests.get(f"{STUDIO_URL}/api/published", timeout=5).json()
    except requests.RequestException:
        return {"online": False, "agents": [],
                "note": "Studio unreachable — installed agents keep working offline."}
    installed = {aid: load_manifest(aid)["version"] for aid in installed_ids()}
    for p in published:
        p["installed_version"] = installed.get(p["id"])
        p["state"] = ("update" if installed.get(p["id"], 99999) < p["version"]
                      else "installed" if p["id"] in installed else "available")
    return {"online": True, "agents": published}


def _download_model(model: dict) -> None:
    f = model["file"]
    prog = _model_downloads.setdefault(f, {"total": 0, "done": 0, "error": None})
    try:
        with requests.get(model["download_url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            prog["total"] = int(r.headers.get("content-length", 0))
            tmp = MODELS_DIR / (f + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    prog["done"] += len(chunk)
            tmp.rename(MODELS_DIR / f)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        prog["error"] = str(exc)[:300]


class InstallIn(BaseModel):
    id: str


@app.post("/api/install")
def api_install(body: InstallIn):
    try:
        published = requests.get(f"{STUDIO_URL}/api/published", timeout=5).json()
    except requests.RequestException:
        raise HTTPException(503, "Agent Store unreachable (Studio offline)")
    entry = next((p for p in published if p["id"] == body.id), None)
    if not entry:
        raise HTTPException(404, f"'{body.id}' not in store")

    zdata = requests.get(f"{STUDIO_URL}/bundles/{entry['bundle']}", timeout=30).content
    target = AGENTS_DIR / body.id
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        zf.extractall(target)

    manifest = load_manifest(body.id)
    state = model_state(manifest)
    if state == "missing":
        threading.Thread(target=_download_model, args=(manifest["model"],), daemon=True).start()
        state = "downloading"
    telemetry.record(source="web-runtime", device_id="web-sim", event="install", ok=1,
                     agent_id=body.id, agent_version=manifest["version"],
                     model_id=manifest["model"]["id"])
    return {"ok": True, "version": manifest["version"], "model_state": state}


# ---------------------------------------------------------------- mobile export

@app.get("/api/export/{agent_id}/prompt.txt", response_class=PlainTextResponse)
def export_prompt(agent_id: str, kb: int = 1):
    manifest = load_manifest(agent_id)
    kb_path = AGENTS_DIR / agent_id / "kb.sqlite"
    return export.exported_prompt(manifest, kb_path, include_kb=bool(kb))


@app.get("/api/export/{agent_id}/card.json")
def export_card(agent_id: str, kb: int = 1):
    manifest = load_manifest(agent_id)
    kb_path = AGENTS_DIR / agent_id / "kb.sqlite"
    return export.character_card(manifest, kb_path, include_kb=bool(kb))


@app.get("/export/{agent_id}", response_class=HTMLResponse)
def export_guide(agent_id: str, request: Request):
    manifest = load_manifest(agent_id)
    # honour ngrok/proxy scheme so QR codes carry the public https URL
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    base_url = f"{scheme}://{request.url.netloc}/"
    return export.export_page(manifest, base_url, kb_chunks=manifest.get("rag", {}).get("chunks", 0))


# ------------------------------------------------------- portal API for phones
# The public tunnel points at THIS app, so it must expose the same catalog +
# bundle surface the Studio has — one portal URL serves the Android app:
# /api/published (store list), /bundles/<file> (agent bundles), /apk (the app).

@app.get("/api/published")
def published_catalog():
    reg = BUNDLES_DIR / "registry.json"
    return json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else []


@app.get("/bundles/{name}")
def bundle_file(name: str):
    path = (BUNDLES_DIR / name).resolve()
    if path.parent != BUNDLES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "bundle not found")
    telemetry.record(source="portal", event="bundle_download",
                     detail=name, bytes=path.stat().st_size, ok=1)
    return FileResponse(path, filename=name)


@app.get("/models/{name}")
def model_file(name: str):
    """Serve GGUF models cached on this server so phones don't depend on
    reaching Hugging Face (mobile/corporate networks often break large CDN
    downloads). The app falls back to the HF URL if a model isn't here."""
    path = (MODELS_DIR / name).resolve()
    if path.parent != MODELS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "model not cached on portal")
    telemetry.record(source="portal", event="model_download",
                     detail=name, bytes=path.stat().st_size, ok=1)
    return FileResponse(path, filename=name)


# ---------------------------------------------------------------- native app APK

APK_PATH = (ROOT / "android_app" / "AgentRuntime" / "android" / "app" /
            "build" / "outputs" / "apk" / "release" / "app-release.apk")


@app.get("/apk")
def download_apk():
    if not APK_PATH.exists():
        raise HTTPException(404, "APK not built yet — run gradlew assembleRelease")
    telemetry.record(source="portal", event="apk_download",
                     bytes=APK_PATH.stat().st_size, ok=1)
    return FileResponse(APK_PATH, filename="enterprise-agents.apk",
                        media_type="application/vnd.android.package-archive")


# ------------------------------------------------------- governance telemetry
# The Android app POSTs usage metadata here (NO chat content — see
# core/telemetry). The Studio's Governance dashboard reads the shared store.

class TelemetryIn(BaseModel):
    device_id: str = "unknown"
    device_model: str | None = None
    event: str                       # install | chat | model_load | error
    agent_id: str | None = None
    agent_version: int | None = None
    model_id: str | None = None
    tokens: int | None = None
    tok_per_sec: float | None = None
    prefill_tokens: int | None = None
    load_ms: int | None = None
    kb_hits: int | None = None
    duration_ms: int | None = None
    ok: int | None = None
    detail: str | None = None        # error text / note — system strings only


@app.post("/api/telemetry")
def ingest_telemetry(body: TelemetryIn):
    data = body.model_dump()
    data["source"] = "android"
    if data.get("detail"):
        data["detail"] = str(data["detail"])[:300]
    telemetry.record(**data)
    return {"ok": True}


# ---------------------------------------------------------------- device data / status

@app.get("/api/device")
def api_device():
    return tools.get_device_data()


@app.get("/api/llm/status")
def api_llm_status():
    return llm.status()


# ---------------------------------------------------------------- chat

QWEN_TOOLS_HEADER = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def build_system_prompt(manifest: dict) -> str:
    prompt = manifest.get("system_prompt") or "You are a helpful enterprise assistant."
    tool_defs = manifest.get("tools", [])
    if tool_defs:
        lines = "\n".join(json.dumps({
            "type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("parameters", {"type": "object", "properties": {}})},
        }) for t in tool_defs)
        prompt += QWEN_TOOLS_HEADER.format(tool_lines=lines)
    return prompt + " /no_think"


def retrieve(agent_id: str, manifest: dict, query: str) -> list[dict]:
    kb_path = AGENTS_DIR / agent_id / "kb.sqlite"
    if not kb_path.exists():
        return []
    rag = manifest.get("rag", {})
    qvec = embeddings.embed_query(query)
    return kbstore.search(kbstore.connect(kb_path), qvec,
                          top_k=int(rag.get("top_k", 4)),
                          min_score=float(rag.get("min_score", 0.35)))


class ChatIn(BaseModel):
    agent_id: str
    messages: list[dict]  # [{role: user|assistant, content: str}, ...]


@app.post("/api/chat")
def api_chat(body: ChatIn):
    manifest = load_manifest(body.agent_id)
    if model_state(manifest) != "ready":
        raise HTTPException(409, "Model not on device yet — check the store panel for download progress.")

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    model_id = manifest["model"]["id"]

    def generate():
        import time
        last_stats: dict = {}
        try:
            needs_load = llm.status().get("model") != manifest["model"]["file"]
            yield sse({"type": "status", "text": "loading model" if needs_load else "ready"})
            t_load = time.time()
            llm.ensure_model(manifest["model"]["file"],
                             context=min(int(manifest["model"].get("context_length", 4096)), 8192))
            if needs_load:
                telemetry.record(source="web-runtime", device_id="web-sim", event="model_load",
                                 ok=1, agent_id=body.agent_id, model_id=model_id,
                                 load_ms=int((time.time() - t_load) * 1000))

            user_query = next((m["content"] for m in reversed(body.messages) if m["role"] == "user"), "")
            sources = retrieve(body.agent_id, manifest, user_query)
            kb_hits = len(sources)
            if sources:
                yield sse({"type": "sources", "items": [
                    {"doc": s["doc_name"], "chunk": s["chunk_index"], "score": s["score"]} for s in sources]})

            context_block = ""
            if sources:
                context_block = "Context from the knowledge base:\n\n" + "\n\n".join(
                    f"[{s['doc_name']} #{s['chunk_index']}]\n{s['text']}" for s in sources
                ) + "\n\n---\n\n"

            messages = [{"role": "system", "content": build_system_prompt(manifest)}]
            for m in body.messages[:-1]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": context_block + user_query})

            tool_defs = {t["name"]: t for t in manifest.get("tools", [])}

            for _round in range(MAX_TOOL_ROUNDS):
                tool_called = False
                for ev in llm.stream_chat(messages, manifest.get("generation", {})):
                    if ev["type"] == "token":
                        yield sse({"type": "token", "text": ev["text"]})
                    elif ev["type"] == "stats":
                        last_stats = ev
                        yield sse(ev)
                    elif ev["type"] == "tool_call":
                        call = ev.get("call") or {}
                        name = call.get("name", "?")
                        args = call.get("arguments") or {}
                        tdef = tool_defs.get(name)
                        result = (tools.execute_tool(tdef, args) if tdef
                                  else {"error": f"tool '{name}' is not attached to this agent"})
                        yield sse({"type": "tool", "name": name, "args": args, "result": result})
                        messages.append({"role": "assistant",
                                         "content": f"<tool_call>\n{ev['raw']}\n</tool_call>"})
                        messages.append({"role": "user",
                                         "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>"})
                        tool_called = True
                if not tool_called:
                    break
            telemetry.record(source="web-runtime", device_id="web-sim", event="chat", ok=1,
                             agent_id=body.agent_id, agent_version=manifest.get("version"),
                             model_id=model_id, kb_hits=kb_hits,
                             tokens=last_stats.get("tokens"),
                             tok_per_sec=last_stats.get("tok_per_sec"),
                             prefill_tokens=last_stats.get("prefill_tokens"))
            yield sse({"type": "done"})
        except Exception as exc:  # noqa: BLE001 — stream the failure to the UI
            telemetry.record(source="web-runtime", device_id="web-sim", event="error", ok=0,
                             agent_id=body.agent_id, model_id=model_id,
                             detail=str(exc)[:300])
            yield sse({"type": "error", "text": str(exc)[:400]})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
