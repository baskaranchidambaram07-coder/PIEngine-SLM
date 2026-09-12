"""Vision model lifecycle: a second managed llama-server with a multimodal
projector, used for attached images.

Why a second server rather than swapping the agent's model: the agent's SLM
(Qwen3 text-only) has no vision tower, and restarting llama-server with a
different GGUF costs 5-10 s each way and throws away the KV cache. The vision
model therefore gets its own process on its own port, started on first use
and left running. Only one of the two generates at a time, so they do not
contend for the 4 vCPUs in practice; RAM is ~1.5 GB extra on a 16 GB box.

Model: Qwen3-VL-2B-Instruct (Apache-2.0, ungated). Chosen over the 7B the
Studio had onboarded because CPU-only prefill of image tokens is the cost that
matters here: a 900x220 receipt is ~230 image tokens and answers in ~15 s on
this box with the 2B; the 7B at Q2 would be several times slower and its
projector was never downloaded.

Images are downscaled before they reach the model: Qwen3-VL spends one token
per 32x32 px patch, so a 4000x3000 phone photo would be ~12,000 tokens and
minutes of prefill. 1024 px on the longest side caps that at ~800 tokens.
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from core.paths import LLAMA_DIR, MODELS_DIR

from . import llm

VL_PORT = 8303
VL_URL = f"http://127.0.0.1:{VL_PORT}"

VISION_MODEL = {
    "id": "qwen3-vl-2b-q4_k_m",
    "name": "Qwen3-VL 2B (Q4_K_M)",
    "file": "Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
    "mmproj": "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf",
    "download_url": "https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
    "mmproj_url": "https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf",
    "context_length": 4096,
    "license": "Apache-2.0 (ungated)",
}

MAX_IMAGE_SIDE = 1024      # px, longest side sent to the model
IMAGE_MAX_TOKENS = 1024    # hard cap enforced by llama-server as well
JPEG_QUALITY = 88

_proc: subprocess.Popen | None = None
_started_at: float | None = None


def model_path() -> Path:
    return MODELS_DIR / VISION_MODEL["file"]


def mmproj_path() -> Path:
    return MODELS_DIR / VISION_MODEL["mmproj"]


def available() -> bool:
    return model_path().exists() and mmproj_path().exists()


def running() -> bool:
    return _proc is not None and _proc.poll() is None


def status() -> dict:
    return {
        "available": available(),
        "running": running(),
        "model": VISION_MODEL["name"],
        "file": VISION_MODEL["file"],
        "mmproj": VISION_MODEL["mmproj"],
        "port": VL_PORT,
        "max_image_side": MAX_IMAGE_SIDE,
        "missing": [p.name for p in (model_path(), mmproj_path()) if not p.exists()],
    }


def ensure() -> None:
    """Start the vision server if it is not already answering."""
    global _proc, _started_at
    if not available():
        raise FileNotFoundError(
            "vision model not on device: " + ", ".join(status()["missing"]))
    if running():
        return
    # Another process (a previous runtime that was not shut down cleanly) may
    # already own the port with the same model; reuse it instead of failing.
    if _healthy():
        return
    threads = max(1, (os.cpu_count() or 4))
    cmd = [
        str(LLAMA_DIR / "llama-server.exe"),
        "-m", str(model_path()),
        "--mmproj", str(mmproj_path()),
        "--port", str(VL_PORT),
        "-c", str(VISION_MODEL["context_length"]),
        "-t", str(threads),
        "--image-max-tokens", str(IMAGE_MAX_TOKENS),
        "--jinja",
        "--no-webui",
    ]
    _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _started_at = time.time()
    deadline = time.time() + 180
    while time.time() < deadline:
        if _proc.poll() is not None:
            raise RuntimeError("vision llama-server exited during startup")
        if _healthy():
            return
        time.sleep(1.0)
    raise TimeoutError("vision llama-server did not become healthy in 180s")


def _healthy() -> bool:
    try:
        return requests.get(f"{VL_URL}/health", timeout=2).json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


# ------------------------------------------------------------------ images

def prepare_image(data: bytes, max_side: int = MAX_IMAGE_SIDE) -> tuple[bytes, dict]:
    """EXIF-orient, flatten and downscale a JPEG for the model. Returns the
    JPEG bytes and a small info dict (original and sent sizes)."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    orig = img.size
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), {"original": list(orig), "sent": list(img.size),
                            "sent_bytes": buf.tell()}


def image_message(text: str, jpeg: bytes) -> dict:
    """An OpenAI-style multimodal user message llama-server understands."""
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}


def stream_answer(jpeg: bytes, question: str, system_prompt: str,
                  extra_context: str = "", generation: dict | None = None) -> Iterator[dict]:
    """Answer `question` about the image, streaming llm.stream_chat events.

    `extra_context` is typically the OCR text of the same image: the model sees
    both the pixels and a machine-read transcript, which is markedly more
    reliable for serial numbers, amounts and codes than either alone.
    """
    ensure()
    gen = dict(generation or {})
    gen.setdefault("temperature", 0.3)
    gen.setdefault("max_tokens", 512)
    text = question
    if extra_context:
        text = (f"{question}\n\n(OCR transcript of the same image, for reference — "
                f"trust the image where they disagree:)\n{extra_context}")
    messages = [{"role": "system", "content": system_prompt},
                image_message(text, jpeg)]
    yield from llm.stream_chat(messages, gen, url=VL_URL)


def describe(jpeg: bytes, prompt: str, system_prompt: str = "You are a precise assistant.",
             max_tokens: int = 300) -> str:
    """Non-streaming convenience used by tests and the benchmark."""
    out = []
    for ev in stream_answer(jpeg, prompt, system_prompt, generation={"max_tokens": max_tokens}):
        if ev["type"] == "token":
            out.append(ev["text"])
    return "".join(out).strip()
