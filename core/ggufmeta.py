"""Read a GGUF file's metadata header, from disk or over HTTP.

Onboarding a model needs its context length, architecture and parameter scale.
Asking a user to type those invites silent mistakes — a wrong context_length
reaches the device manifest and llama-server then truncates or over-allocates.
The GGUF file already states them, so read them from the source.

Layout (GGUF v2/v3): magic 'GGUF', uint32 version, uint64 tensor_count,
uint64 kv_count, then kv_count entries of {string key, uint32 type, value}.
Metadata sits at the head of the file, so a ranged read of the first few MB is
enough — the tensor data behind it is never fetched.
"""
from __future__ import annotations

import struct

MAGIC = b"GGUF"

# GGUF value type ids
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# The general.* and <arch>.* keys occupy ~1-2 KB at the head of the file, and
# parsing stops as soon as they are in hand, so this never reaches the
# tokenizer arrays behind them (151,936 token strings in a Qwen3 GGUF). Keeping
# the probe small is what makes inspecting a remote model quick.
DEFAULT_PROBE_BYTES = 512 * 1024

# ggml file_type -> the quantisation label people recognise
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}


class GGUFError(ValueError):
    """The bytes are not a readable GGUF header."""


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise EOFError("ran past the probed range")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, t: int):
        fmt, size = _FIXED[t]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = self.scalar(U64)
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t: int):
        if t in _FIXED:
            return self.scalar(t)
        if t == STRING:
            return self.string()
        if t == ARRAY:
            elem = self.scalar(U32)
            count = self.scalar(U64)
            if elem == STRING:
                # Token vocabularies live here and are huge; skip, don't build.
                for _ in range(count):
                    self.take(self.scalar(U64))
                return f"<{count} strings>"
            if elem == ARRAY:
                raise GGUFError("nested arrays are not supported")
            _, size = _FIXED[elem]
            self.take(size * count)
            return f"<{count} values>"
        raise GGUFError(f"unknown GGUF value type {t}")


def _have_essentials(kv: dict) -> bool:
    """arch + its context_length — everything the catalog needs.

    Used to stop parsing early. `general.file_type` deliberately is NOT part of
    this: in real GGUFs it sits *after* the tokenizer vocabulary, so waiting for
    it would mean walking ~150k token strings on every inspect. Quantisation is
    recovered from the filename instead.
    """
    arch = kv.get("general.architecture")
    return bool(arch) and isinstance(kv.get(f"{arch}.context_length"), int)


def parse_header(buf: bytes, stop_early: bool = True) -> dict:
    """Parse metadata key/values out of the head of a GGUF file.

    A truncated probe is not an error: whatever was read is returned, so a
    partial range still yields the general.* keys that come first.
    """
    if len(buf) < 4 or buf[:4] != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    r = _Reader(buf)
    r.take(4)
    version = r.scalar(U32)
    if version not in (1, 2, 3):
        raise GGUFError(f"unsupported GGUF version {version}")
    tensor_count = r.scalar(U64)
    kv_count = r.scalar(U64)

    kv: dict = {}
    truncated = False
    for _ in range(kv_count):
        try:
            key = r.string()
            kv[key] = r.value(r.scalar(U32))
        except EOFError:
            truncated = True
            break
        if stop_early and _have_essentials(kv):
            break
    return {"version": version, "tensor_count": tensor_count,
            "kv_count": kv_count, "kv": kv, "truncated": truncated}


def summarise(header: dict) -> dict:
    """The handful of fields the catalog actually needs."""
    kv = header["kv"]
    arch = kv.get("general.architecture") or ""
    ctx = None
    for key in (f"{arch}.context_length", "llama.context_length"):
        if isinstance(kv.get(key), int):
            ctx = int(kv[key])
            break
    ft = kv.get("general.file_type")
    return {
        "architecture": arch,
        "name": kv.get("general.name") or "",
        "context_length": ctx,
        "quantization": FILE_TYPES.get(ft) if isinstance(ft, int) else None,
        "size_label": kv.get("general.size_label") or "",
        "parameter_count": kv.get("general.parameter_count"),
        "block_count": kv.get(f"{arch}.block_count"),
        "embedding_length": kv.get(f"{arch}.embedding_length"),
        "truncated": header["truncated"],
    }


def quant_from_filename(filename: str) -> str | None:
    """Recover the quantisation label from the conventional GGUF filename.

    `general.file_type` lives behind the tokenizer vocabulary, so reading it
    would cost megabytes. Publishers name the file for it instead —
    Qwen3-1.7B-Q4_K_M.gguf — which is both cheaper and what users recognise.
    """
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    for part in reversed(stem.split("-")):
        if part.upper() in _QUANT_LABELS:
            return part.upper()
    # Q4_K_M style suffixes survive an underscore split too
    upper = stem.upper()
    for label in sorted(_QUANT_LABELS, key=len, reverse=True):
        if upper.endswith(label) or f"-{label}" in upper or f".{label}" in upper:
            return label
    return None


_QUANT_LABELS = set(FILE_TYPES.values())


def from_file(path, probe_bytes: int = DEFAULT_PROBE_BYTES) -> dict:
    import os
    with open(path, "rb") as fh:
        out = summarise(parse_header(fh.read(probe_bytes)))
    out["quantization"] = out["quantization"] or quant_from_filename(os.path.basename(str(path)))
    return out


def from_url(url: str, probe_bytes: int = DEFAULT_PROBE_BYTES,
             timeout: int = 30) -> dict:
    """Range-read the head of a remote GGUF. Never downloads the tensors."""
    import requests

    with requests.get(url, headers={"Range": f"bytes=0-{probe_bytes - 1}"},
                      timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        # A server that ignores Range would otherwise stream the whole multi-GB
        # file into memory, so cap the read instead of trusting the response.
        chunks, got = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            got += len(chunk)
            if got >= probe_bytes:
                break
    out = summarise(parse_header(b"".join(chunks)[:probe_bytes]))
    out["quantization"] = out["quantization"] or quant_from_filename(url)
    return out
