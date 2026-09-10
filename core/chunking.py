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


# A heading line: markdown (## Foo), a numbered section (3. Foo, 2.1. Foo), or a
# short ALL-CAPS line. The length cap and trailing-punctuation test keep
# numbered LIST items ("1. Confirm the UAT plan for August.") from being read as
# headings — those are common in meeting minutes and splitting on them would
# shred the document.
MAX_HEADING_CHARS = 80

_HEADING = re.compile(
    r"^(?:"
    r"\#{1,6}\s+\S.*"                     # markdown heading
    r"|\d+(?:\.\d+)*\.\s+[^\d\s].*"       # 1. Section  /  2.3. Section
    r"|[A-Z][A-Z0-9 \-/&']{3,}"           # SHORT ALL-CAPS HEADING
    r")$"
)


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > MAX_HEADING_CHARS:
        return False
    if line[-1] in ".,;":
        return False          # a sentence, not a title
    return bool(_HEADING.match(line))


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] — heading is '' for text before the first one."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.split("\n"):
        if is_heading(line):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def split_text(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split into chunks that never straddle a section heading.

    Merging purely on character count put a document's section 8 (a table of
    engineering machines) and section 9 (approval gates) into ONE chunk, and the
    model then answered a question about gate G2 with a sentence about the
    Change machine. Sections are therefore hard boundaries, and every chunk is
    prefixed with its heading so both the retriever and the model can see which
    section the text came from.
    """
    text = re.sub(r"\r\n?", "\n", text)
    out: list[str] = []
    for heading, body in split_sections(text):
        prefix = f"{heading}\n\n" if heading else ""
        # keep the prefixed chunk within the target the caller asked for
        budget = max(200, target - len(prefix))
        for chunk in _split_block(body, budget, overlap):
            out.append(prefix + chunk)
    return out


def _split_block(text: str, target: int, overlap: int) -> list[str]:
    """Paragraph-first recursive splitter with sentence fallback and overlap."""
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
