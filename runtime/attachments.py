"""Per-conversation file attachments: validation, storage, text extraction
(with OCR), indexing for on-demand retrieval, and expiry.

Scope decided with the product: jpg/jpeg/pdf/txt/md only, one file at a time,
5 MB maximum. This web runtime is the test and benchmarking surface; the
feature ships to users in the mobile app afterwards, so everything here is
kept device-shaped — plain files on disk, SQLite, a subprocess OCR engine —
rather than leaning on anything server-only.

Pipeline for one upload:
  validate  -> extension AND magic bytes must agree; size cap; no NUL bytes
               in text files (a renamed binary is refused, not decoded).
  extract   -> txt/md: decode.  pdf: pypdf text per page, and any page with
               almost no text layer is rasterised (PyMuPDF, 200 dpi) and OCR'd.
               jpg: Tesseract OCR after light preprocessing.
  index     -> chunk (core.chunking, section-aware) + embed (bge-small) into a
               kb.sqlite beside the file, so retrieval at question time is the
               same code path as the agent's own knowledge base.
  guard     -> runtime/guard.py scans the extracted text; the verdict is stored
               in meta.json so every later question against this file is
               refused without re-scanning.

"On-demand RAG": a small document (<= FULLTEXT_CHARS) is injected whole at
question time — retrieval over three chunks only loses information. Larger
documents go through the lexical gate and vector search exactly like the KB.

Retention: attachments are conversation-scoped user data, not knowledge. They
are deleted on request and swept after RETENTION_HOURS regardless.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from core import chunking, embeddings, gating, kbstore

APP_DIR = Path(__file__).resolve().parent
ATTACH_DIR = APP_DIR / "device_storage" / "attachments"
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg": "image", ".jpeg": "image", ".pdf": "pdf", ".txt": "text", ".md": "text"}
MAX_BYTES = 5 * 1024 * 1024
RETENTION_HOURS = 24

FULLTEXT_CHARS = 3500        # inject whole; ~3 chunks / ~900 tokens
ATTACH_TOP_K = 4
ATTACH_MIN_SCORE = 0.30      # a floor only — the user chose this file, so lean inclusive
MIN_TEXT_LAYER_CHARS = 40    # a PDF page below this is treated as scanned -> OCR
OCR_DPI = 300                # Tesseract's recommended density; 200 misread "30" as "3@"
MAX_OCR_PAGES = 40           # 5 MB of scanned pages could be 100+; cap the wall-clock
OCR_TIMEOUT_S = 60

TESSERACT = os.environ.get("TESSERACT_EXE") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class AttachmentError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ validation

def validate(filename: str, data: bytes) -> tuple[str, str]:
    """Return (ext, kind) or raise AttachmentError. Type is decided by the
    extension AND the file's own bytes — a .txt that is really an executable,
    or a .jpg that is really a PDF, is refused rather than guessed at."""
    name = (filename or "").strip()
    if not name:
        raise AttachmentError("the file has no name")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise AttachmentError(
            f"'{ext or name}' is not supported — attach a jpg, jpeg, pdf, txt or md file")
    if not data:
        raise AttachmentError("the file is empty")
    if len(data) > MAX_BYTES:
        raise AttachmentError(
            f"file is {len(data) / (1024 * 1024):.1f} MB; the limit is 5 MB", status=413)
    kind = ALLOWED_EXT[ext]
    if kind == "image" and not data.startswith(b"\xff\xd8\xff"):
        raise AttachmentError("this is not a JPEG image (the bytes do not match the extension)")
    if kind == "pdf" and not data[:1024].lstrip().startswith(b"%PDF-"):
        raise AttachmentError("this is not a PDF (the bytes do not match the extension)")
    if kind == "text":
        sample = data[:65536]
        if b"\x00" in sample:
            raise AttachmentError("this is not a text file (binary content)")
    return ext, kind


# ------------------------------------------------------------------ storage

def _dir(att_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", att_id or ""):
        raise AttachmentError("unknown attachment", status=404)
    return ATTACH_DIR / att_id


def save(filename: str, data: bytes) -> dict:
    ext, kind = validate(filename, data)
    att_id = uuid.uuid4().hex
    d = _dir(att_id)
    d.mkdir(parents=True, exist_ok=False)
    (d / f"file{ext}").write_bytes(data)
    meta = {
        "id": att_id,
        "name": Path(filename).name[:120],
        "ext": ext,
        "kind": kind,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "created": time.time(),
        "status": "stored",
        "text_chars": 0, "pages": 0, "ocr_pages": 0, "chunks": 0, "small": False,
        "timings_ms": {},
        "guard": None,
    }
    save_meta(meta)
    return meta


def meta_path(att_id: str) -> Path:
    return _dir(att_id) / "meta.json"


def load_meta(att_id: str) -> dict:
    p = meta_path(att_id)
    if not p.exists():
        raise AttachmentError("attachment not found (it may have expired)", status=404)
    return json.loads(p.read_text(encoding="utf-8"))


def save_meta(meta: dict) -> None:
    meta_path(meta["id"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def file_path(meta: dict) -> Path:
    return _dir(meta["id"]) / f"file{meta['ext']}"


def text_path(att_id: str) -> Path:
    return _dir(att_id) / "text.txt"


def kb_path(att_id: str) -> Path:
    return _dir(att_id) / "kb.sqlite"


def delete(att_id: str) -> bool:
    d = _dir(att_id)
    if not d.exists():
        return False
    kbstore._cache.pop(str(kb_path(att_id)), None)
    shutil.rmtree(d, ignore_errors=True)
    return True


def sweep_expired(max_age_hours: float = RETENTION_HOURS) -> int:
    """Delete attachments older than the retention window. Returns the count."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for d in ATTACH_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            created = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("created", 0)
        except (OSError, ValueError):
            created = d.stat().st_mtime
        if created < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


# ------------------------------------------------------------------ OCR

def ocr_available() -> bool:
    return Path(TESSERACT).exists()


def _preprocess_for_ocr(img):
    """Grayscale, upscale small images, stretch contrast. Tesseract's LSTM
    wants ~30 px x-height; phone photos of documents are usually fine, but a
    small screenshot benefits from 2x."""
    from PIL import ImageOps

    img = ImageOps.exif_transpose(img)
    g = img.convert("L")
    w, h = g.size
    if max(w, h) < 1500:
        g = g.resize((w * 2, h * 2))
    g = ImageOps.autocontrast(g, cutoff=1)
    return g


def ocr_image(img, psm: int = 3) -> str:
    """Run Tesseract on a PIL image via a temp PNG. Returns the text ('' if
    Tesseract is missing or fails — OCR is best-effort, never fatal)."""
    if not ocr_available():
        return ""
    import tempfile

    g = _preprocess_for_ocr(img)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "page.png"
        g.save(p, format="PNG")
        try:
            out = subprocess.run(
                [TESSERACT, str(p), "stdout", "--psm", str(psm), "-l", "eng"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=OCR_TIMEOUT_S,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (subprocess.TimeoutExpired, OSError):
            return ""
    if out.returncode != 0:
        return ""
    return _clean_ocr(out.stdout)


_DIGIT_CONFUSIONS = str.maketrans({"@": "0", "O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])(?=[\d@OolI|.,:/\-]*\d)[\d@OolI|][\d@OolI|.,:/\-]*(?![A-Za-z])")


def _fix_digit_confusions(text: str) -> str:
    """Repair the classic OCR swaps (0/O/@, 1/l/I) but ONLY inside tokens that
    are otherwise numeric and not attached to letters, so "3@ days" -> "30 days"
    while "Hello" and "INV-2026" are untouched."""
    def fix(m: re.Match) -> str:
        tok = m.group(0)
        if not any(ch.isdigit() for ch in tok):
            return tok
        return tok.translate(_DIGIT_CONFUSIONS)
    return _NUMERIC_TOKEN.sub(fix, text)


_LIST_START = re.compile(r"^\s*(?:\d+[.)]\s|[-*•]\s)")
_HEADING_LINE = re.compile(r"^[A-Z][A-Z0-9 \-/&'()]{3,}$")


def _unwrap_lines(lines: list[str]) -> list[str]:
    """Re-join sentences that OCR split at the printed line width.

    A scanned policy read "…pattern require" / "30 days written notice…" as two
    lines, and the 1.7B answered "23 days". Joining a line that ends mid-sentence
    (no terminal punctuation) with a continuation that starts in lower case or
    with a digit puts the number back next to its verb. Headings, list items
    and blank lines are never joined."""
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        prev = out[-1].strip() if out else ""
        if (prev and s and not prev.endswith((".", ":", ";", "!", "?"))
                and (s[0].islower() or s[0].isdigit()) and not _LIST_START.match(s)
                and not _HEADING_LINE.match(prev)):
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(ln)
    return out


def _clean_ocr(text: str) -> str:
    text = text.replace("\f", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    # drop lines that are only OCR noise (1-2 punctuation chars)
    lines = [ln for ln in lines if len(ln.strip()) > 2 or ln.strip().isalnum()]
    lines = _unwrap_lines(lines)
    return _fix_digit_confusions(re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip())


# ------------------------------------------------------------------ extraction

def extract(meta: dict) -> dict:
    """Extract text for the stored file, write text.txt, update and return meta."""
    data = file_path(meta).read_bytes()
    t0 = time.time()
    ocr_ms = 0.0
    pages = 0
    ocr_pages = 0
    if meta["kind"] == "text":
        text = chunking.extract_text(meta["name"], data)
    elif meta["kind"] == "pdf":
        text, pages, ocr_pages, ocr_ms = _extract_pdf(data)
    else:  # image
        from PIL import Image

        t = time.time()
        text = ocr_image(Image.open(io.BytesIO(data)))
        ocr_ms = (time.time() - t) * 1000
        pages = 1
        ocr_pages = 1
    text = _normalise(text)
    text_path(meta["id"]).write_text(text, encoding="utf-8")
    meta.update({
        "text_chars": len(text), "pages": pages, "ocr_pages": ocr_pages,
        "ocr_used": ocr_pages > 0, "status": "extracted",
    })
    meta["timings_ms"]["extract"] = round((time.time() - t0) * 1000)
    meta["timings_ms"]["ocr"] = round(ocr_ms)
    save_meta(meta)
    return meta


def _extract_pdf(data: bytes) -> tuple[str, int, int, float]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — a damaged page must not lose the document
            page_texts.append("")
    n = len(page_texts)
    needs_ocr = [i for i, t in enumerate(page_texts) if len(t.strip()) < MIN_TEXT_LAYER_CHARS]
    ocr_ms = 0.0
    ocr_done = 0
    if needs_ocr and ocr_available():
        t = time.time()
        try:
            import pymupdf
            from PIL import Image

            doc = pymupdf.open(stream=data, filetype="pdf")
            for i in needs_ocr[:MAX_OCR_PAGES]:
                pix = doc[i].get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY)
                img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                got = ocr_image(img)
                if got:
                    page_texts[i] = got
                ocr_done += 1
            doc.close()
        except Exception:  # noqa: BLE001 — fall back to whatever text layer exists
            pass
        ocr_ms = (time.time() - t) * 1000
    text = "\n\n".join(f"[page {i + 1}]\n{t}" if n > 1 else t
                       for i, t in enumerate(page_texts) if t.strip())
    return text, n, ocr_done, ocr_ms


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


# ------------------------------------------------------------------ indexing

def index(meta: dict, embedder_id: str | None = None) -> dict:
    """Chunk + embed the extracted text into the attachment's own kb.sqlite."""
    text = text_path(meta["id"]).read_text(encoding="utf-8") if text_path(meta["id"]).exists() else ""
    t0 = time.time()
    chunks = chunking.split_text(text) if text else []
    meta["small"] = len(text) <= FULLTEXT_CHARS
    if chunks and not meta["small"]:
        # Same embedder as the agent's own KB, so one query vector serves both.
        vecs = embeddings.embed_passages(chunks, embedder_id)
        kb = kbstore.connect(kb_path(meta["id"]))
        kbstore.add_document(kb, meta["name"], chunks, vecs, meta={"bytes": meta["bytes"]},
                             embedder_id=embedder_id)
        kb.close()
    meta["chunks"] = len(chunks)
    meta["embedder"] = embedder_id
    meta["status"] = "ready" if text else "no-text"
    meta["timings_ms"]["index"] = round((time.time() - t0) * 1000)
    save_meta(meta)
    return meta


def full_text(att_id: str) -> str:
    p = text_path(att_id)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def context_for(meta: dict, query: str, top_k: int = ATTACH_TOP_K,
                min_score: float = ATTACH_MIN_SCORE) -> list[dict]:
    """On-demand retrieval over one attachment.

    Small document -> every chunk, no embedding (the whole file fits).
    Large document -> lexical gate, then vector search with a low floor.
    Result rows look like kbstore.search rows plus kind='attachment'.
    """
    text = full_text(meta["id"])
    if not text:
        return []
    if meta.get("small"):
        chunks = chunking.split_text(text) or [text]
        return [{"chunk_id": -1, "doc_name": meta["name"], "chunk_index": i, "text": c,
                 "score": 1.0, "kind": "attachment"} for i, c in enumerate(chunks)]
    ok, _ = gating.should_retrieve(query)
    if not ok:
        return []
    kbp = kb_path(meta["id"])
    if not kbp.exists():
        return []
    kb = kbstore.connect(kbp)
    try:
        hits = kbstore.search(kb, embeddings.embed_query(query, meta.get("embedder")), top_k=top_k, min_score=min_score)
    finally:
        kb.close()
    for h in hits:
        h["kind"] = "attachment"
    return hits


def public(meta: dict) -> dict:
    """The attachment as the UI sees it — never the raw text or findings' values."""
    out = {k: meta.get(k) for k in (
        "id", "name", "ext", "kind", "bytes", "status", "text_chars", "pages",
        "ocr_pages", "ocr_used", "chunks", "small", "timings_ms", "created")}
    g = meta.get("guard")
    out["guard"] = g if g is None else {k: g.get(k) for k in (
        "enabled", "blocked", "summary", "judged", "judge_note", "ms", "message")}
    return out
