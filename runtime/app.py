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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import (adapters, chunking, downloads, embeddings, gating, kbstore,
                  telemetry, versions)
from core.paths import BUNDLES_DIR, MODELS_DIR, ROOT

from . import attachments, export, guard, llm, tools, vision

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


def adapter_path(manifest: dict) -> Path | None:
    """Local path to this agent's LoRA adapter, if it has one and it is here.

    A missing adapter is not fatal: the agent still runs on the shared base
    model, just with stock behaviour. Failing the chat instead would make a
    ~30 MB download a hard dependency of a 1 GB model that is already present.
    """
    ad = manifest.get("adapter")
    if not ad:
        return None
    p = adapters.ADAPTERS_DIR / ad["file"]
    return p if p.exists() else None


def embedder_id_of(manifest: dict) -> str:
    """The catalogue id of the embedder this agent's KB was built with."""
    from core import embedders
    emb = manifest.get("embedder") or {}
    eid = emb.get("id") or ""
    # pre-catalogue bundles stored the fastembed model name as the id
    if "/" in eid:
        for e in embedders.EMBEDDERS:
            if e["fastembed_model"] == eid:
                return e["id"]
    return embedders.get(eid)["id"]


def pii_enabled(manifest: dict) -> bool:
    """Per-agent guardrail switch, set in the Studio and shipped in the bundle.

    Off by default: an agent that never sees user documents should not pay a
    model-judge call per turn, and its author decides. Manifests published
    before the switch existed have no `guardrails` key and are treated as off.
    """
    return bool((manifest.get("guardrails") or {}).get("pii"))


def adapter_state(manifest: dict) -> str:
    ad = manifest.get("adapter")
    if not ad:
        return "none"
    if adapter_path(manifest):
        return "ready"
    dl = _model_downloads.get(ad["file"])
    if dl and not dl.get("error"):
        return "downloading"
    return "missing"


def _studio_agent_ids() -> set[str] | None:
    """Agents the Studio still knows about, via the shared bundle registry.
    The registry is rewritten on every publish and delete, so an installed
    agent missing from it was deleted in the Studio. None if unreadable."""
    reg = BUNDLES_DIR / "registry.json"
    try:
        return {e["id"] for e in json.loads(reg.read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return None


@app.get("/api/installed")
def api_installed():
    out = []
    known = _studio_agent_ids()
    for aid in installed_ids():
        m = load_manifest(aid)
        dl = _model_downloads.get(m["model"]["file"], {})
        out.append({
            "id": aid, "name": m["name"], "description": m["description"],
            "version": m["version"], "model": m["model"], "rag": m["rag"],
            "guardrails": {"pii": pii_enabled(m)},
            "embedder": {**(m.get("embedder") or {}), "id": embedder_id_of(m)},
            # False = the Studio deleted this agent after it was installed here
            "in_studio": (aid in known) if known is not None else True,
            "tools": [t["name"] for t in m.get("tools", [])],
            "model_state": model_state(m),
            "version_state": versions.state_of(aid, m["version"]),
            "adapter": m.get("adapter"), "adapter_state": adapter_state(m),
            "download": {"done": dl.get("done", 0), "total": dl.get("total", 0),
                          "error": dl.get("error")} if dl else None,
        })
    return out


@app.delete("/api/installed/{agent_id}")
def api_uninstall(agent_id: str):
    manifest = load_manifest(agent_id)
    import shutil
    shutil.rmtree(AGENTS_DIR / agent_id)
    # Recorded so the Studio's "who still holds this agent" check (which gates
    # Delete) sees the copy go, the same way it sees handsets report theirs.
    telemetry.record(source="web-runtime", device_id="web-sim", event="uninstall", ok=1,
                     agent_id=agent_id, agent_version=manifest.get("version"),
                     model_id=manifest["model"]["id"])
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
    tmp = MODELS_DIR / (f + ".part")
    # Check before writing a byte: a multi-GB model that fills the disk takes
    # more than itself down with it — in-flight SQLite writes included.
    need = int(model.get("size_bytes") or 0)
    short = downloads.space_shortfall(MODELS_DIR, need)
    if short:
        prog["error"] = downloads.describe_shortfall(short, need, MODELS_DIR)
        return
    try:
        with requests.get(model["download_url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            prog["total"] = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    prog["done"] += len(chunk)
            tmp.rename(MODELS_DIR / f)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        prog["error"] = str(exc)[:300]
        prog["reclaimed_bytes"] = downloads.discard_partial(tmp)


def _download_adapter(adapter: dict) -> None:
    """Adapters come from the portal, not Hugging Face — they are ours."""
    f = adapter["file"]
    prog = _model_downloads.setdefault(f, {"total": 0, "done": 0, "error": None})
    tmp = adapters.ADAPTERS_DIR / (f + ".part")
    need = int(adapter.get("size_bytes") or 0)
    short = downloads.space_shortfall(adapters.ADAPTERS_DIR.parent, need)
    if short:
        prog["error"] = downloads.describe_shortfall(short, need,
                                                     adapters.ADAPTERS_DIR.parent)
        return
    try:
        adapters.ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        url = adapter.get("download_url") or f"{STUDIO_URL}/adapters/{f}"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            prog["total"] = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    prog["done"] += len(chunk)
            tmp.rename(adapters.ADAPTERS_DIR / f)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        prog["error"] = str(exc)[:300]
        prog["reclaimed_bytes"] = downloads.discard_partial(tmp)


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
    if adapter_state(manifest) == "missing":
        threading.Thread(target=_download_adapter, args=(manifest["adapter"],),
                         daemon=True).start()
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
    entries = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else []
    # The Studio keeps the registry pointed at active versions, but the portal
    # must not advertise a retired one even if the two ever drift.
    return [e for e in entries if versions.is_servable(e["id"], e["version"])]


@app.get("/api/version-states")
def version_states():
    """Per-version lifecycle state, so an installed device can retire a version
    it already holds. Hiding a version from the store is not enough — a phone
    that installed v5 before it was disabled would otherwise keep running it."""
    return versions.read_states()


@app.get("/bundles/{name}")
def bundle_file(name: str):
    path = (BUNDLES_DIR / name).resolve()
    if path.parent != BUNDLES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "bundle not found")
    # Enforce on the direct URL too: the store listing is a hint, this is the
    # actual gate. A disabled version must not be installable by guessing.
    parsed = versions.parse_bundle(path.name)
    if parsed and not versions.is_servable(*parsed):
        state = versions.state_of(*parsed)
        raise HTTPException(410, f"'{path.name}' is {state} and is no longer available")
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


@app.get("/adapters/{name}")
def adapter_file(name: str):
    """Serve a LoRA adapter to a phone. Same portal-first pattern as /models,
    but there is no upstream fallback — adapters are ours, nobody mirrors them."""
    path = (adapters.ADAPTERS_DIR / name).resolve()
    if path.parent != adapters.ADAPTERS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "adapter not on portal")
    telemetry.record(source="portal", event="adapter_download",
                     detail=name, bytes=path.stat().st_size, ok=1)
    return FileResponse(path, filename=name, media_type="application/octet-stream")


# ---------------------------------------------------------------- native app APK

APK_PATH = (ROOT / "android_app" / "AgentRuntime" / "android" / "app" /
            "build" / "outputs" / "apk" / "release" / "app-release.apk")


APK_LABELS = {
    "enterprise-agents-v2.8.apk": "v2.8 — stable: attachments, OCR, PII Guard, uninstall, retry of blocked models",
    "enterprise-agents-v2.6-git.apk": "v2.6 — built from the last git commit (comparison build, no attachments)",
    "enterprise-agents-v2.9-exp-llamarn-0.12.9.apk": "v2.9-exp — v2.8 on the newer llama.rn 0.12.9 engine (experimental)",
}


def _apk_version(path: Path) -> str | None:
    """versionName from the APK's manifest via aapt, if the SDK is here."""
    aapt = ROOT / "tools" / "android-sdk" / "build-tools" / "35.0.0" / "aapt.exe"
    if not aapt.exists():
        return None
    try:
        import re
        import subprocess
        out = subprocess.run([str(aapt), "dump", "badging", str(path)], capture_output=True,
                             text=True, timeout=20, encoding="utf-8", errors="replace").stdout
        m = re.search(r"versionCode='(\d+)' versionName='([^']+)'", out)
        return f"{m.group(2)} (code {m.group(1)})" if m else None
    except (OSError, subprocess.SubprocessError):
        return None


@app.get("/api/apks")
def api_apks(limit: int = 2):
    """The APKs this portal offers: the current build at /apk plus any .apk
    placed in the models directory, newest first, trimmed to `limit` (default
    two — the current build and the one before it; older builds stay
    downloadable by direct URL but are not advertised)."""
    import datetime as dt
    out = []
    if APK_PATH.exists():
        st = APK_PATH.stat()
        out.append({"name": "enterprise-agents.apk", "url": "/apk", "bytes": st.st_size,
                    "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes"),
                    "version": _apk_version(APK_PATH),
                    "label": "Current build — install this one", "current": True})
    for p in sorted(MODELS_DIR.glob("*.apk"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = p.stat()
        out.append({"name": p.name, "url": f"/models/{p.name}", "bytes": st.st_size,
                    "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes"),
                    "version": _apk_version(p), "label": APK_LABELS.get(p.name, ""), "current": False,
                    "_mtime": st.st_mtime})
    # the current build is always first; the rest by build time, newest first
    rest = sorted([a for a in out if not a["current"]], key=lambda a: a["_mtime"], reverse=True)
    out = [a for a in out if a["current"]] + rest
    for a in out:
        a.pop("_mtime", None)
    return out[: max(1, limit)]


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


# ------------------------------------------------------- attachments + guard
# One file per turn (jpg/jpeg/pdf/txt/md, <= 5 MB) is uploaded here BEFORE the
# question is sent, so the expensive part — OCR, embedding, the PII Guard's
# review — happens once and its verdict is stored with the file. /api/chat
# then references the attachment by id. This is the web test surface for a
# capability that ships in the mobile app; see docs/attachments-and-pii-guard.md.

def _chat_text(messages: list[dict], generation: dict) -> str:
    """Non-streaming helper for the guard's judge: run the loaded agent model
    and return the reply text (think blocks and tool calls already filtered)."""
    return "".join(ev["text"] for ev in llm.stream_chat(messages, generation)
                   if ev["type"] == "token")


def _judge_fn_for(agent_id: str | None):
    """The judge runs on the agent's own model. Returns None if that model is
    not loadable right now — the guard then falls back to regex-only and says
    so in its note, rather than failing the upload."""
    if not agent_id:
        return None
    try:
        manifest = load_manifest(agent_id)
        if model_state(manifest) != "ready":
            return None
        llm.ensure_model(manifest["model"]["file"],
                         context=min(int(manifest["model"].get("context_length", 4096)), 8192),
                         lora=adapter_path(manifest))
        return _chat_text
    except Exception:  # noqa: BLE001 — advisory path
        return None


@app.get("/api/capabilities")
def api_capabilities():
    return {
        "attachments": {"max_bytes": attachments.MAX_BYTES, "max_files": 1,
                        "allowed": sorted(attachments.ALLOWED_EXT),
                        "retention_hours": attachments.RETENTION_HOURS,
                        "ocr": attachments.ocr_available()},
        "vision": vision.status(),
        "guard": {"name": guard.GUARD_NAME, "policy": guard.policy()},
    }


@app.get("/api/guard/policy")
def api_guard_policy():
    return guard.policy()


@app.put("/api/guard/policy")
def api_guard_policy_put(body: dict):
    try:
        return guard.save_policy(body or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/vision/status")
def api_vision_status():
    return vision.status()


@app.post("/api/attachments")
def api_attach(file: UploadFile = File(...), agent_id: str | None = Form(None)):
    # Plain `def`: OCR, embedding and the judge are CPU/blocking work and must
    # not stall the event loop that streams other users' chats.
    attachments.sweep_expired()
    data = file.file.read(attachments.MAX_BYTES + 1)
    try:
        meta = attachments.save(file.filename or "", data)
    except attachments.AttachmentError as exc:
        raise HTTPException(exc.status, str(exc))
    try:
        meta = attachments.extract(meta)
        # index with the selected agent's embedder so retrieval can join KB + file
        emb = None
        if agent_id:
            try:
                emb = embedder_id_of(load_manifest(agent_id))
            except HTTPException:
                emb = None
        meta = attachments.index(meta, emb)
    except Exception as exc:  # noqa: BLE001 — surface a readable reason, drop the file
        attachments.delete(meta["id"])
        raise HTTPException(422, f"could not read the file: {str(exc)[:200]}")

    # ---- PII Guard over the document — only if this agent has the guardrail
    # on. An upload with no agent (API use) gets the regex stage regardless.
    enabled = True
    if agent_id:
        try:
            enabled = pii_enabled(load_manifest(agent_id))
        except HTTPException:
            enabled = True
    if enabled:
        pol = guard.policy()
        text = attachments.full_text(meta["id"])
        chunks = chunking.split_text(text) or ([text] if text else [])
        chat_fn = _judge_fn_for(agent_id) if guard.judge_enabled_for("attachment", pol) else None
        verdict = guard.inspect_chunks(chunks, f"attachment:{meta['name']}", chat_fn, pol,
                                       judge_limit=int(pol.get("judge_upload_chunks", 2)) if chat_fn else 0)
        g = verdict.to_public()
        g["judged_chunks"] = list(range(min(len(chunks), int(pol.get("judge_upload_chunks", 2))))) if chat_fn else []
        g["message"] = (guard.refusal_message(f'the attached document "{meta["name"]}"', verdict.summary)
                        if verdict.blocked else None)
    else:
        verdict = guard.Verdict(blocked=False)
        g = {**verdict.to_public(), "judged_chunks": [], "message": None,
             "judge_note": "PII Guard is off for this agent"}
    g["enabled"] = enabled
    meta["guard"] = g
    meta["timings_ms"]["guard"] = verdict.ms
    attachments.save_meta(meta)

    telemetry.record(source="web-runtime", device_id="web-sim", event="attachment", ok=1,
                     agent_id=agent_id, bytes=meta["bytes"], kb_hits=meta["chunks"],
                     detail=f"{meta['kind']} {meta['status']} pages={meta['pages']} "
                            f"ocr={meta['ocr_pages']} guard={'blocked' if verdict.blocked else 'clear'}")
    if verdict.blocked:
        telemetry.record(source="web-runtime", device_id="web-sim", event="guard_block", ok=1,
                         agent_id=agent_id,
                         detail="attachment: " + ", ".join(s["label"] for s in verdict.summary)[:250])
    return attachments.public(meta)


@app.get("/api/attachments/{att_id}")
def api_attachment(att_id: str):
    try:
        return attachments.public(attachments.load_meta(att_id))
    except attachments.AttachmentError as exc:
        raise HTTPException(exc.status, str(exc))


@app.delete("/api/attachments/{att_id}")
def api_attachment_delete(att_id: str):
    try:
        return {"ok": attachments.delete(att_id)}
    except attachments.AttachmentError as exc:
        raise HTTPException(exc.status, str(exc))


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


def build_system_prompt(manifest: dict, with_tools: bool = True) -> str:
    prompt = manifest.get("system_prompt") or "You are a helpful enterprise assistant."
    tool_defs = manifest.get("tools", []) if with_tools else []
    if tool_defs:
        lines = "\n".join(json.dumps({
            "type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("parameters", {"type": "object", "properties": {}})},
        }) for t in tool_defs)
        prompt += QWEN_TOOLS_HEADER.format(tool_lines=lines)
    return prompt + " /no_think"


ATTACHMENT_RULES = ("\n\nThe user has attached a file. Text from it appears under "
                    "'Attached document'. Answer questions about the file from that text, "
                    "quote it where useful, and say plainly when the file does not contain "
                    "the answer.")


def retrieve(agent_id: str, manifest: dict, query: str) -> list[dict]:
    kb_path = AGENTS_DIR / agent_id / "kb.sqlite"
    if not kb_path.exists():
        return []
    # Gate BEFORE embedding: a greeting should cost nothing at all, not an
    # embedding plus a KB scan plus ~1000 prompt tokens of context.
    ok, _reason = gating.should_retrieve(query)
    if not ok:
        return []
    rag = manifest.get("rag", {})
    # The bundle names the embedder its KB was built with; older bundles have
    # only {id, dim} or nothing, and embedders.get() maps those to the default.
    qvec = embeddings.embed_query(query, embedder_id_of(manifest))
    return kbstore.search(kbstore.connect(kb_path), qvec,
                          top_k=int(rag.get("top_k", 4)),
                          min_score=float(rag.get("min_score", 0.45)))


class ChatIn(BaseModel):
    agent_id: str
    messages: list[dict]  # [{role: user|assistant, content: str}, ...]
    attachment_id: str | None = None


@app.post("/api/chat")
def api_chat(body: ChatIn):
    manifest = load_manifest(body.agent_id)
    # A retired version must stop RUNNING, not just stop being installable —
    # this device may have installed it before it was disabled.
    vstate = versions.state_of(body.agent_id, manifest["version"])
    if vstate != versions.ACTIVE:
        raise HTTPException(
            410, f"{manifest['name']} v{manifest['version']} has been {vstate} by the "
                 f"Studio and can no longer be run. Check the store for a newer version.")
    if model_state(manifest) != "ready":
        raise HTTPException(409, "Model not on device yet — check the store panel for download progress.")
    att: dict | None = None
    if body.attachment_id:
        try:
            att = attachments.load_meta(body.attachment_id)
        except attachments.AttachmentError as exc:
            raise HTTPException(exc.status, str(exc))

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    model_id = manifest["model"]["id"]

    def refuse(where: str, summary: list[dict], src: str):
        text = guard.refusal_message(where, summary)
        telemetry.record(source="web-runtime", device_id="web-sim", event="guard_block", ok=1,
                         agent_id=body.agent_id, agent_version=manifest.get("version"),
                         model_id=model_id,
                         detail=f"{src}: " + ", ".join(s["label"] for s in summary)[:250])
        yield sse({"type": "guard", "where": where, "text": text, "summary": summary})
        yield sse({"type": "done"})

    def generate():
        import time
        last_stats: dict = {}
        try:
            want_lora = adapter_path(manifest)
            st = llm.status()
            needs_load = (st.get("model") != manifest["model"]["file"]
                          or st.get("adapter") != (str(want_lora) if want_lora else None))
            yield sse({"type": "status", "text": "loading model" if needs_load else "ready"})
            t_load = time.time()
            llm.ensure_model(manifest["model"]["file"],
                             context=min(int(manifest["model"].get("context_length", 4096)), 8192),
                             lora=want_lora)
            if needs_load:
                telemetry.record(source="web-runtime", device_id="web-sim", event="model_load",
                                 ok=1, agent_id=body.agent_id, model_id=model_id,
                                 load_ms=int((time.time() - t_load) * 1000))

            user_query = next((m["content"] for m in reversed(body.messages) if m["role"] == "user"), "")
            pol = guard.policy()
            # The whole guard is a per-agent guardrail (Studio checkbox, shipped
            # in the manifest). Off = the normal flow: no scan of the message,
            # the file or the answer.
            pii = pii_enabled(manifest)

            # ---- PII Guard, stage 0: a file already judged unsafe at upload
            # refuses every question about it — checked first because it is
            # free, and the slow model judge below would reach the same answer.
            if pii and att and (att.get("guard") or {}).get("blocked"):
                yield from refuse(f'the attached document "{att["name"]}"',
                                  (att.get("guard") or {}).get("summary") or [], "attachment")
                return

            # ---- PII Guard, stage 1: the user's own message. Regex always; the
            # model judge per policy (default: only when a file is in play).
            if pii:
                use_judge = guard.judge_enabled_for("attachment" if att else "query", pol)
                if use_judge:
                    yield sse({"type": "status", "text": f"{guard.GUARD_NAME} reviewing your message"})
                verdict = guard.inspect(user_query, "query", use_judge=use_judge,
                                        chat_fn=_chat_text if use_judge else None, pol=pol)
                if verdict.blocked:
                    yield from refuse("your message", verdict.summary, "query")
                    return

            # ---- PII Guard, stage 2: the attachment. The upload-time verdict
            # is final for a blocked file; a clear file still gets its
            # retrieved chunks judged the first time they are used.
            att_sources: list[dict] = []
            if att:
                g = att.get("guard") or {}
                att_sources = attachments.context_for(att, user_query)
                if pii and att_sources and guard.judge_enabled_for("attachment", pol):
                    judged = set(g.get("judged_chunks") or [])
                    fresh = [s for s in att_sources if s["chunk_index"] not in judged]
                    if fresh:
                        yield sse({"type": "status", "text": f"{guard.GUARD_NAME} reviewing the document"})
                        v2 = guard.inspect_chunks([s["text"] for s in fresh],
                                                  f"attachment:{att['name']}", _chat_text, pol,
                                                  judge_limit=int(pol.get("judge_query_chunks", 4)))
                        g["judged_chunks"] = sorted(judged | {s["chunk_index"] for s in fresh})
                        if v2.blocked:
                            g.update(v2.to_public())
                            g["message"] = guard.refusal_message(
                                f'the attached document "{att["name"]}"', v2.summary)
                        att["guard"] = g
                        attachments.save_meta(att)
                        if v2.blocked:
                            yield from refuse(f'the attached document "{att["name"]}"',
                                              v2.summary, "attachment")
                            return

            sources = retrieve(body.agent_id, manifest, user_query)
            kb_hits = len(sources)
            shown = ([{"doc": s["doc_name"], "chunk": s["chunk_index"], "score": s["score"],
                       "kind": "attachment"} for s in att_sources]
                     + [{"doc": s["doc_name"], "chunk": s["chunk_index"], "score": s["score"],
                         "kind": "kb"} for s in sources])
            if shown:
                yield sse({"type": "sources", "items": shown})

            answer_parts: list[str] = []
            use_vision = bool(att and att["kind"] == "image" and vision.available())

            if use_vision:
                # ---- image turn: the vision model answers, seeing the pixels
                # plus the OCR transcript; the agent's persona still applies.
                yield sse({"type": "status", "text": "reading the image with the vision model"})
                jpeg, info = vision.prepare_image(attachments.file_path(att).read_bytes())
                yield sse({"type": "vision", "model": vision.VISION_MODEL["name"], **info})
                system = build_system_prompt(manifest, with_tools=False)
                ocr_text = attachments.full_text(att["id"])[:3000]
                for ev in vision.stream_answer(jpeg, user_query, system, extra_context=ocr_text,
                                               generation=manifest.get("generation", {})):
                    if ev["type"] == "token":
                        answer_parts.append(ev["text"])
                        yield sse({"type": "token", "text": ev["text"]})
                    elif ev["type"] == "stats":
                        last_stats = {**ev, "model": vision.VISION_MODEL["name"]}
                        yield sse(last_stats)
            else:
                context_block = ""
                if att_sources:
                    how = "full text" if att.get("small") else "relevant passages"
                    context_block += (f"Attached document \"{att['name']}\" ({how}):\n\n"
                                      + "\n\n".join(f"[{att['name']} #{s['chunk_index']}]\n{s['text']}"
                                                    for s in att_sources) + "\n\n---\n\n")
                elif att and not att_sources:
                    context_block += (f"Attached document \"{att['name']}\": no readable text could "
                                      f"be extracted from it.\n\n---\n\n")
                if sources:
                    context_block += "Context from the knowledge base:\n\n" + "\n\n".join(
                        f"[{s['doc_name']} #{s['chunk_index']}]\n{s['text']}" for s in sources
                    ) + "\n\n---\n\n"

                system = build_system_prompt(manifest)
                if att:
                    system = system.replace(" /no_think", ATTACHMENT_RULES + " /no_think")
                messages = [{"role": "system", "content": system}]
                for m in body.messages[:-1]:
                    messages.append({"role": m["role"], "content": m["content"]})
                messages.append({"role": "user", "content": context_block + user_query})

                tool_defs = {t["name"]: t for t in manifest.get("tools", [])}

                for _round in range(MAX_TOOL_ROUNDS):
                    tool_called = False
                    for ev in llm.stream_chat(messages, manifest.get("generation", {})):
                        if ev["type"] == "token":
                            answer_parts.append(ev["text"])
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
                            answer_parts = []
                    if not tool_called:
                        break

            # ---- PII Guard, stage 3: the answer itself. Defence in depth — the
            # inputs were clear, but a model can still complete a pattern.
            if pii and pol.get("redact_output", True):
                masked, summary = guard.redact_text("".join(answer_parts), pol)
                if summary:
                    yield sse({"type": "redact", "text": masked, "summary": summary})

            telemetry.record(source="web-runtime", device_id="web-sim", event="chat", ok=1,
                             agent_id=body.agent_id, agent_version=manifest.get("version"),
                             model_id=model_id, kb_hits=kb_hits + len(att_sources),
                             adapter_id=(manifest.get("adapter") or {}).get("id")
                                        if adapter_path(manifest) else None,
                             tokens=last_stats.get("tokens"),
                             tok_per_sec=last_stats.get("tok_per_sec"),
                             prefill_tokens=last_stats.get("prefill_tokens"),
                             detail=(f"attachment {att['kind']}" + (" vision" if use_vision else ""))
                                    if att else None)
            yield sse({"type": "done"})
        except Exception as exc:  # noqa: BLE001 — stream the failure to the UI
            telemetry.record(source="web-runtime", device_id="web-sim", event="error", ok=0,
                             agent_id=body.agent_id, model_id=model_id,
                             detail=str(exc)[:300])
            yield sse({"type": "error", "text": str(exc)[:400]})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.on_event("shutdown")
def _shutdown():
    vision.stop()


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
