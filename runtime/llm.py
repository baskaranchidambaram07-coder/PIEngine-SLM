"""llama.cpp server lifecycle + streaming client.

On a real handheld this layer is replaced by an embedded llama.cpp binding
(llama.rn / LLMFarm / MLC). Here we run llama-server as a managed subprocess
— one model loaded at a time, restarted when an agent needs a different GGUF.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from core.paths import LLAMA_DIR, MODELS_DIR

LLM_PORT = 8302
LLM_URL = f"http://127.0.0.1:{LLM_PORT}"

_proc: subprocess.Popen | None = None
_loaded_model: str | None = None
_loaded_lora: str | None = None


def status() -> dict:
    running = _proc is not None and _proc.poll() is None
    return {"running": running, "model": _loaded_model if running else None,
            "adapter": _loaded_lora if running else None}


def ensure_model(model_file: str, context: int = 4096, lora: Path | str | None = None) -> None:
    """Start (or restart) llama-server with the requested GGUF and adapter.

    A scenario fine-tune ships as a LoRA adapter rather than a merged model
    (core/adapters.py explains why), so the base GGUF stays shared across every
    agent on the device and only the ~20-60 MB adapter changes. Switching
    adapters restarts the server exactly like switching models does.
    """
    global _proc, _loaded_model, _loaded_lora
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"model file not on device: {model_file}")

    lora_path = Path(lora) if lora else None
    if lora_path and not lora_path.exists():
        raise FileNotFoundError(f"adapter file not on device: {lora_path.name}")
    lora_key = str(lora_path) if lora_path else None

    if (_proc is not None and _proc.poll() is None
            and _loaded_model == model_file and _loaded_lora == lora_key):
        return

    stop()
    threads = max(1, (os.cpu_count() or 4))
    cmd = [
        str(LLAMA_DIR / "llama-server.exe"),
        "-m", str(model_path),
        "--port", str(LLM_PORT),
        "-c", str(context),
        "-t", str(threads),
        "--jinja",
        "--no-webui",
    ]
    if lora_path:
        cmd += ["--lora", str(lora_path)]
    _proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    _loaded_model = model_file
    _loaded_lora = lora_key

    deadline = time.time() + 120
    while time.time() < deadline:
        if _proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            if requests.get(f"{LLM_URL}/health", timeout=2).json().get("status") == "ok":
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become healthy in 120s")


def stop() -> None:
    global _proc, _loaded_model, _loaded_lora
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
    _loaded_model = None
    _loaded_lora = None


atexit.register(stop)


def stream_chat(messages: list[dict], generation: dict, url: str = LLM_URL) -> Iterator[dict]:
    """Stream deltas from llama-server, filtering Qwen3 <think> blocks and
    surfacing <tool_call> payloads as structured events.

    `url` defaults to the agent's text model; runtime/vision.py passes its own
    server so image turns share this parser (message content may then be the
    OpenAI list form with an image_url part).

    Yields: {"type": "token", "text": str} | {"type": "tool_call", "raw": str, "call": dict}
            | {"type": "stats", ...}
    """
    payload = {
        "messages": messages,
        "stream": True,
        "temperature": generation.get("temperature", 0.7),
        "top_p": generation.get("top_p", 0.8),
        "max_tokens": generation.get("max_tokens", 768),
        "timings_per_token": True,  # ensures a timings object arrives on the final chunk
    }
    # min_p truncates the tail relative to the top token's probability, which
    # suppresses the low-confidence tokens fabricated detail is made of. Only
    # send it when an agent asks for it — llama.cpp defaults to 0.05, so passing
    # 0.0 unconditionally would silently DISABLE min-p sampling for every agent.
    if generation.get("min_p") is not None:
        payload["min_p"] = float(generation["min_p"])
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, stream=True, timeout=600)
    resp.raise_for_status()
    resp.encoding = "utf-8"  # SSE has no charset header; default latin-1 mangles UTF-8

    tags = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    holdback = max(len(t) for t in tags) - 1
    pending = ""
    mode = "text"  # text | think | tool
    n_tokens = 0
    t_start = time.time()
    timings: dict = {}

    def drain(final: bool = False) -> Iterator[dict]:
        nonlocal pending, mode
        while True:
            if mode == "think":
                idx = pending.find("</think>")
                if idx < 0:
                    if final:
                        pending = ""
                    return
                pending = pending[idx + len("</think>"):].lstrip("\n")
                mode = "text"
                continue
            if mode == "tool":
                idx = pending.find("</tool_call>")
                if idx < 0:
                    if final and pending.strip():
                        # model stopped mid tool-call; treat as text so nothing is lost
                        yield {"type": "token", "text": pending}
                        pending = ""
                    return
                raw = pending[:idx].strip()
                pending = pending[idx + len("</tool_call>"):]
                mode = "text"
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError:
                    call = None
                yield {"type": "tool_call", "raw": raw, "call": call}
                continue
            # text mode: find earliest opening tag
            i_think = pending.find("<think>")
            i_tool = pending.find("<tool_call>")
            candidates = [(i, m) for i, m in ((i_think, "think"), (i_tool, "tool")) if i >= 0]
            if candidates:
                idx, new_mode = min(candidates)
                if pending[:idx]:
                    yield {"type": "token", "text": pending[:idx]}
                pending = pending[idx + len("<think>" if new_mode == "think" else "<tool_call>"):]
                mode = new_mode
                continue
            safe = len(pending) if final else max(0, len(pending) - holdback)
            if safe > 0:
                yield {"type": "token", "text": pending[:safe]}
                pending = pending[safe:]
            return

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "timings" in chunk:
            timings = chunk["timings"]
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        piece = delta.get("content") or delta.get("reasoning_content")
        if delta.get("reasoning_content"):
            continue  # thinking tokens surfaced separately by --jinja; skip
        if piece:
            n_tokens += 1
            pending += piece
            yield from drain()

    yield from drain(final=True)
    elapsed = max(time.time() - t_start, 1e-6)
    stats = {"type": "stats", "tokens": n_tokens, "seconds": round(elapsed, 1),
             "tok_per_sec": round(n_tokens / elapsed, 1)}
    if timings:  # llama-server's own numbers: split prefill vs generation
        stats.update({
            "prefill_tokens": timings.get("prompt_n"),
            "prefill_per_sec": round(timings.get("prompt_per_second") or 0, 1),
            "tok_per_sec": round(timings.get("predicted_per_second") or 0, 1),
            "tokens": timings.get("predicted_n", n_tokens),
        })
    yield stats
