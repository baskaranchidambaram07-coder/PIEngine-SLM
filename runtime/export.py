"""Mobile Export — package an installed agent for third-party on-device apps.

Existing Play-Store GGUF runners (PocketPal AI, ChatterUI, Maid, ...) can run
our model and a persona, but not our vector KB or tools. This module bridges
the gap:

- ``prompt.txt``   — system prompt (+ optional knowledge digest) to paste into
                     a PocketPal "Pal".
- ``card.json``    — SillyTavern chara_card_v2, directly importable by
                     ChatterUI / Maid as a character.
- knowledge digest — the agent's kb.sqlite compressed into a bounded text
                     block baked into the prompt, so the exported agent still
                     knows the project facts without RAG. Fine for small KBs;
                     the full RAG + tools experience needs our own runtime
                     (see android_app/PLAN.md).
"""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

DIGEST_BUDGET_CHARS = 8000       # ~2k tokens: safe inside an 8k context
PER_CHUNK_CAP = 700

MODEL_HINTS = {
    # catalog id -> in-app search string for PocketPal's Hugging Face search
    "qwen3-1.7b-q4_k_m": "unsloth/Qwen3-1.7B-GGUF",
    "qwen3-0.6b-q4_k_m": "unsloth/Qwen3-0.6B-GGUF",
    "qwen3-4b-instruct-2507-q4_k_m": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "llama-3.2-3b-instruct-q4_k_m": "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "gemma-3-4b-it-q4_k_m": "unsloth/gemma-3-4b-it-GGUF",
}


def knowledge_digest(kb_path: Path, budget: int = DIGEST_BUDGET_CHARS) -> str:
    """Compress kb.sqlite into a bounded, doc-grouped text digest."""
    if not kb_path.exists():
        return ""
    conn = sqlite3.connect(kb_path)
    rows = conn.execute(
        "SELECT doc_name, chunk_index, text FROM chunks ORDER BY doc_name, chunk_index"
    ).fetchall()
    if not rows:
        return ""

    out = io.StringIO()
    out.write("\n\n# EMBEDDED KNOWLEDGE BASE\n"
              "Facts below are your only source of truth. Cite the document "
              "name when you use one. If the answer is not here, say you don't "
              "have it in the knowledge base.\n")
    used = len(out.getvalue())
    current_doc = None
    for doc, _idx, text in rows:
        compact = " ".join(text.split())
        if len(compact) > PER_CHUNK_CAP:
            compact = compact[:PER_CHUNK_CAP].rsplit(" ", 1)[0] + " …"
        header = f"\n## {doc}\n" if doc != current_doc else "\n"
        piece = header + compact
        if used + len(piece) > budget:
            out.write("\n(…digest truncated to fit device context…)")
            break
        out.write(piece)
        used += len(piece)
        current_doc = doc
    return out.getvalue()


def exported_prompt(manifest: dict, kb_path: Path, include_kb: bool) -> str:
    prompt = manifest.get("system_prompt") or "You are a helpful enterprise assistant."
    if include_kb:
        prompt += knowledge_digest(kb_path)
    if manifest.get("model", {}).get("family") == "qwen3":
        prompt += "\n\n/no_think"
    return prompt


def character_card(manifest: dict, kb_path: Path, include_kb: bool) -> dict:
    """SillyTavern chara_card_v2 — the de-facto import format for mobile
    character apps (ChatterUI, Maid, Layla)."""
    name = manifest["name"]
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": manifest.get("description", ""),
            "personality": "",
            "scenario": manifest.get("scenario", ""),
            "first_mes": (f"Hi — I'm {name}, running fully on your device. "
                          "Ask me about your meetings, actions, risks or stakeholders."),
            "mes_example": "",
            "system_prompt": exported_prompt(manifest, kb_path, include_kb),
            "post_history_instructions": "",
            "creator_notes": (
                f"Exported from the Offline Enterprise Agent Runtime (bundle v{manifest['version']}). "
                f"Pair with model: {manifest['model']['name']} — "
                f"{MODEL_HINTS.get(manifest['model']['id'], manifest['model']['file'])}. "
                "Knowledge digest embedded in the system prompt."
                if include_kb else
                f"Exported from the Offline Enterprise Agent Runtime (bundle v{manifest['version']}). "
                f"Pair with model: {manifest['model']['name']}."
            ),
            "alternate_greetings": [],
            "tags": ["enterprise", "offline", "agent"],
            "creator": "Agent Studio",
            "character_version": f"v{manifest['version']}",
            "extensions": {},
        },
    }


def qr_svg(data: str) -> str:
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage,
                      box_size=9, border=2)
    return img.to_string().decode()


def export_page(manifest: dict, base_url: str, kb_chunks: int) -> str:
    """Mobile-friendly HTML guide with QR codes and copy buttons."""
    aid = manifest["id"]
    model = manifest["model"]
    hf_repo = MODEL_HINTS.get(model["id"], "")
    card_url = f"{base_url}api/export/{aid}/card.json?kb=1"
    prompt_url = f"{base_url}api/export/{aid}/prompt.txt?kb=1"
    model_url = model["download_url"]
    tools = [t["name"] for t in manifest.get("tools", [])]

    qr_card = qr_svg(card_url)
    qr_model = qr_svg(model_url)
    apk_url = f"{base_url}apk"
    qr_apk = qr_svg(apk_url)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Get {manifest['name']} on your phone</title>
<style>
 :root{{--bg:#0d1117;--panel:#161b22;--border:#2d333b;--text:#e6edf3;--muted:#8b949e;--accent:#4f8cff;--green:#3fb950;--warn:#d29922}}
 *{{box-sizing:border-box;margin:0;padding:0}}
 body{{background:var(--bg);color:var(--text);font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:720px;margin:0 auto;padding:24px 16px 60px}}
 h1{{font-size:20px;margin-bottom:4px}} h2{{font-size:16px;margin:0 0 10px}}
 .sub{{color:var(--muted);font-size:13px;margin-bottom:22px}}
 .card{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px}}
 .step{{display:flex;gap:10px;margin:10px 0}}
 .n{{background:var(--accent);color:#fff;border-radius:50%;width:22px;height:22px;flex-shrink:0;text-align:center;font-size:12px;font-weight:700;line-height:22px}}
 .qr{{background:#fff;border-radius:10px;padding:8px;width:150px;height:150px;margin:10px auto}}
 .qr svg{{width:100%;height:100%}}
 a{{color:var(--accent)}} code{{background:#1c2129;padding:1px 6px;border-radius:5px;font-size:13px;word-break:break-all}}
 button{{background:var(--accent);border:none;color:#fff;padding:9px 16px;border-radius:8px;font-weight:600;cursor:pointer;font-size:13.5px;width:100%;margin-top:8px}}
 .ghost{{background:transparent;border:1px solid var(--border)}}
 .pill{{display:inline-block;border:1px solid var(--border);border-radius:20px;padding:2px 10px;font-size:11.5px;color:var(--muted);margin:2px 3px 0 0}}
 .note{{border-left:3px solid var(--warn);padding:8px 12px;background:#1a1712;border-radius:0 8px 8px 0;font-size:13px;color:#d8c9a3;margin-top:12px}}
 .ok{{color:var(--green)}}
</style></head><body>
<h1>📲 {manifest['name']} — on your phone</h1>
<div class="sub">Bundle v{manifest['version']} · {model['name']} ({model['size_gb']} GB) · {kb_chunks} knowledge chunks baked into the exported persona</div>

<div class="card">
 <h2>Option A — Native app <span class="pill">full experience</span><span class="pill">recommended</span></h2>
 <div class="step"><div class="n">1</div><div>Scan (or <a href="{apk_url}">tap here</a>) to download <b>Enterprise Agents.apk</b>, open it, and allow "install unknown apps" when Android asks.</div></div>
 <div class="qr">{qr_apk}</div>
 <div class="step"><div class="n">2</div><div>In the app, the <b>Portal URL</b> (⚙, explained on screen) is preset to this portal — the one address all agents are published on. Tap <b>Install</b> on {manifest['name']} — it pulls the bundle, the model and the embedder, then works in airplane mode.</div></div>
 <div class="step"><div class="n">3</div><div>Full experience: live retrieval over the entire KB with citations{", tools (" + ", ".join(tools) + ")" if tools else ""}, and 📎 upload of your own files (≤2 MB) into an on-device inline KB.</div></div>
</div>

<div class="card">
 <h2>Option B — PocketPal AI <span class="pill">Play Store</span><span class="pill">open source</span></h2>
 <div class="step"><div class="n">1</div><div>Install <b>PocketPal AI</b> from the Play Store.</div></div>
 <div class="step"><div class="n">2</div><div><b>Models → + → Add from Hugging Face</b>, search <code>{hf_repo}</code> and download <code>{model['file']}</code> — or scan the model QR below with your phone.</div></div>
 <div class="step"><div class="n">3</div><div><b>Pals → + New Pal</b>: name it <b>{manifest['name']}</b>, pick the model from step 2, and paste the agent persona:</div></div>
 <button onclick="copyPrompt(this)">📋 Copy agent persona (system prompt + knowledge)</button>
 <a href="{prompt_url}" download="{aid}-persona.txt"><button class="ghost">⬇ Or download persona as .txt</button></a>
 <div class="step"><div class="n">4</div><div>Turn on <b>airplane mode</b> and chat. <span class="ok">Everything runs on the handset.</span></div></div>
</div>

<div class="card">
 <h2>Option C — ChatterUI <span class="pill">character import</span></h2>
 <div class="step"><div class="n">1</div><div>Install <b>ChatterUI</b> (GitHub releases APK) and download the same model as above.</div></div>
 <div class="step"><div class="n">2</div><div>Download the character card and import it via <b>Characters → import</b>:</div></div>
 <a href="{card_url}" download="{aid}-character.json"><button>⬇ Download character card (.json)</button></a>
</div>

<div class="card">
 <h2>Scan on your phone</h2>
 <table style="width:100%;text-align:center;font-size:12.5px;color:var(--muted)"><tr>
  <td><div class="qr">{qr_apk}</div>native app (Option A)</td>
  <td><div class="qr">{qr_card}</div>character card (Option C)</td>
  <td><div class="qr">{qr_model}</div>model GGUF ({model['size_gb']} GB)</td>
 </tr></table>
</div>

<div class="note"><b>What Options B/C give you vs. the native app:</b> the exported persona includes a compressed
knowledge digest, so the agent knows your project facts offline — but live retrieval over the full KB
{"and tools (" + ", ".join(tools) + ")" if tools else ""} work only in the native app (Option A).</div>

<script>
async function copyPrompt(btn){{
  const r = await fetch("{prompt_url}");
  await navigator.clipboard.writeText(await r.text());
  btn.textContent = "✓ Copied — paste into PocketPal";
}}
</script>
</body></html>"""
