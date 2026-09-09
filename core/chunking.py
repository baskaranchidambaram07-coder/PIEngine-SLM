"""Text extraction and chunking tuned for small on-device knowledge bases.

Chunks target ~1200 characters (~300 tokens) with sentence-boundary splits and
a small overlap, so retrieval context stays compact enough for a 1-4B SLM's
context window on mobile hardware.
"""
from __future__ import annotations

import io
import re

TARGET_CHARS = 1200
OVERLAP_CHARS = 150


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
    # txt / md / csv / anything text-like
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def split_text(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Paragraph-first recursive splitter with sentence fallback and overlap."""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target:
            pieces.append(para)
        else:
            sentences = _split_sentences(para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + len(sent) + 1 > target:
                    pieces.append(buf)
                    buf = buf[-overlap:] + " " + sent if overlap else sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                pieces.append(buf)

    # Merge small neighbouring paragraphs up to the target size.
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > target:
            chunks.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks
