// Per-conversation file attachments on the handset — the on-device port of
// runtime/attachments.py.
//
// One file per turn: jpg/jpeg/pdf/txt/md, 5 MB max, checked by extension AND
// by the file's first bytes. Text is read directly; images are OCR'd with
// ML Kit (on-device, no network); PDFs are rendered page by page with the
// platform's PdfRenderer and OCR'd the same way — one uniform path for text
// PDFs and scans alike, capped at MAX_PDF_PAGES to bound the wall-clock.
//
// "On-demand RAG": a small document (<= FULLTEXT_CHARS) is injected whole at
// question time; a larger one is chunked with the same section-aware splitter
// as the KB, embedded with the same bge-small model, and searched per question
// behind the same lexical gate. Attachment chunks live in their own SQLite
// file beside the copied file, never in the agent's inline KB — an attachment
// is conversation-scoped user data, not knowledge, and is swept after
// RETENTION_HOURS.
//
// The PII Guard verdict is stored on the attachment so every later question
// against a blocked file is refused without rescanning (see chat.ts).
import { open, DB } from '@op-engineering/op-sqlite';
import ReactNativeBlobUtil from 'react-native-blob-util';
import { pick, types } from '@react-native-documents/picker';
import TextRecognition from '@react-native-ml-kit/text-recognition';
import PdfThumbnail from 'react-native-pdf-thumbnail';

import { AGENTS_DIR } from './config';
import { DEFAULT_EMBEDDER, EmbedderSpec } from './embedder';
import { splitText } from './chunker';
import { shouldRetrieve } from './gating';
import { embedText } from './llm';
import { SearchHit } from './kb';
import { Verdict } from './guard';
import { log } from './logger';

const fs = ReactNativeBlobUtil.fs;

export const ALLOWED_EXT: Record<string, 'image' | 'pdf' | 'text'> = {
  '.jpg': 'image', '.jpeg': 'image', '.pdf': 'pdf', '.txt': 'text', '.md': 'text',
};
export const MAX_ATTACH_BYTES = 5 * 1024 * 1024;
export const RETENTION_HOURS = 24;
export const FULLTEXT_CHARS = 3500;
export const ATTACH_TOP_K = 4;
export const ATTACH_MIN_SCORE = 0.30;
export const MAX_PDF_PAGES = 20;
const PDF_RENDER_QUALITY = 85;

export type PickedFile = { name: string; size: number; uri: string };

export type Attachment = {
  id: string;
  agentId: string;
  name: string;
  ext: string;
  kind: 'image' | 'pdf' | 'text';
  bytes: number;
  dir: string;
  path: string;              // copied file, always a plain filesystem path
  text: string;              // extracted text (kept in memory for the session)
  textChars: number;
  pages: number;
  ocrPages: number;
  chunks: number;
  small: boolean;
  status: 'ready' | 'no-text';
  guardEnabled: boolean;
  guard: Verdict | null;
  judgedChunks: number[];
  timingsMs: Record<string, number>;
  embedder: EmbedderSpec;      // the agent's embedder, so KB and file share one query vector
};

export class AttachmentError extends Error {}

// ------------------------------------------------------------------ picking

/** Open the system picker limited to the supported types. null = cancelled. */
export async function pickAttachment(): Promise<PickedFile | null> {
  try {
    const [res] = await pick({
      mode: 'open',
      allowMultiSelection: false,
      type: [types.pdf, types.images, types.plainText, 'text/markdown', 'text/x-markdown'],
    });
    if (!res) return null;
    return { name: res.name || 'attachment', size: Number(res.size || 0), uri: res.uri };
  } catch (e: any) {
    if (/cancel/i.test(String(e?.message || e))) return null;
    throw e;
  }
}

export function extOf(name: string): string {
  const m = /\.[A-Za-z0-9]+$/.exec(name || '');
  return m ? m[0].toLowerCase() : '';
}

/** Extension + size check — the cheap half of validation, before any copy. */
export function validateMeta(file: PickedFile): { ext: string; kind: 'image' | 'pdf' | 'text' } {
  const ext = extOf(file.name);
  const kind = ALLOWED_EXT[ext];
  if (!kind) throw new AttachmentError(`'${ext || file.name}' is not supported — attach a jpg, jpeg, pdf, txt or md file`);
  if (file.size <= 0) throw new AttachmentError('the file is empty');
  if (file.size > MAX_ATTACH_BYTES) {
    throw new AttachmentError(`file is ${(file.size / 1048576).toFixed(1)} MB; the limit is 5 MB`);
  }
  return { ext, kind };
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/** Decode the first `max` bytes of a base64 string (React Native has no Buffer). */
export function base64Head(b64: string, max: number): number[] {
  const out: number[] = [];
  let bits = 0;
  let acc = 0;
  for (let i = 0; i < b64.length && out.length < max; i++) {
    const v = B64.indexOf(b64[i]);
    if (v < 0) continue;          // '=' padding or whitespace
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push((acc >> bits) & 0xff);
    }
  }
  return out;
}

/** Magic-byte check on the copied file — a renamed binary is refused, not guessed at. */
async function validateBytes(path: string, kind: 'image' | 'pdf' | 'text'): Promise<void> {
  const b64 = await fs.readFile(path, 'base64');
  const head = base64Head(String(b64).slice(0, 8192), 4096);
  if (kind === 'image' && !(head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff)) {
    throw new AttachmentError('this is not a JPEG image (the bytes do not match the extension)');
  }
  if (kind === 'pdf') {
    const ascii = String.fromCharCode(...head.slice(0, 1024)).trimStart();
    if (!ascii.startsWith('%PDF-')) {
      throw new AttachmentError('this is not a PDF (the bytes do not match the extension)');
    }
  }
  if (kind === 'text' && head.includes(0)) {
    throw new AttachmentError('this is not a text file (binary content)');
  }
}

// ------------------------------------------------------------------ storage

const attachRoot = (agentId: string) => `${AGENTS_DIR}/${agentId}/attachments`;

function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

async function copyIn(src: string, dest: string): Promise<void> {
  // content:// and file:// both read through RNBU; a 5 MB base64 round-trip is
  // the simplest path that works for every provider the picker can return.
  const from = src.startsWith('content://') ? src : src.replace('file://', '');
  const b64 = await fs.readFile(from, 'base64');
  await fs.writeFile(dest, b64, 'base64');
}

export async function removeAttachment(att: Attachment): Promise<void> {
  await fs.unlink(att.dir).catch(() => {});
}

/** Delete attachments older than the retention window for this agent. */
export async function sweepAttachments(agentId: string): Promise<number> {
  const root = attachRoot(agentId);
  if (!(await fs.exists(root))) return 0;
  let removed = 0;
  const cutoff = Date.now() - RETENTION_HOURS * 3600 * 1000;
  for (const id of await fs.ls(root)) {
    const dir = `${root}/${id}`;
    try {
      const st: any = await fs.stat(dir);
      if (Number(st.lastModified) < cutoff) { await fs.unlink(dir); removed++; }
    } catch {}
  }
  if (removed) await log(`attach: swept ${removed} expired attachment(s)`);
  return removed;
}

// ------------------------------------------------------------------ OCR text repair
// Ports of runtime/attachments._fix_digit_confusions and _unwrap_lines.

const DIGIT_CONFUSIONS: Record<string, string> = { '@': '0', O: '0', o: '0', l: '1', I: '1', '|': '1' };
const NUMERIC_TOKEN = /(?<![A-Za-z])(?=[\d@OolI|.,:/\-]*\d)[\d@OolI|][\d@OolI|.,:/\-]*(?![A-Za-z])/g;

export function fixDigitConfusions(text: string): string {
  return text.replace(NUMERIC_TOKEN, tok =>
    /\d/.test(tok) ? tok.replace(/[@OolI|]/g, c => DIGIT_CONFUSIONS[c]) : tok);
}

const LIST_START = /^\s*(?:\d+[.)]\s|[-*•]\s)/;
const HEADING_LINE = /^[A-Z][A-Z0-9 \-/&'()]{3,}$/;

export function unwrapLines(lines: string[]): string[] {
  const out: string[] = [];
  for (const ln of lines) {
    const s = ln.trim();
    const prev = out.length ? out[out.length - 1].trim() : '';
    if (prev && s && !/[.:;!?]$/.test(prev) && /^[a-z0-9]/.test(s)
        && !LIST_START.test(s) && !HEADING_LINE.test(prev)) {
      out[out.length - 1] = out[out.length - 1].trimEnd() + ' ' + s;
    } else {
      out.push(ln);
    }
  }
  return out;
}

export function cleanOcr(text: string): string {
  let lines = text.replace(/\f/g, '\n').split(/\r?\n/).map(l => l.trimEnd());
  lines = lines.filter(l => l.trim().length > 2 || /^[A-Za-z0-9]+$/.test(l.trim()));
  lines = unwrapLines(lines);
  return fixDigitConfusions(lines.join('\n').replace(/\n{3,}/g, '\n\n')).trim();
}

// ------------------------------------------------------------------ extraction

async function ocrImage(path: string): Promise<string> {
  const res = await TextRecognition.recognize(path.startsWith('file://') ? path : `file://${path}`);
  // Blocks preserve reading order better than the flat `text` on multi-column pages.
  const lines: string[] = [];
  for (const block of res.blocks || []) {
    for (const line of block.lines || []) lines.push(line.text);
    lines.push('');
  }
  return cleanOcr(lines.length ? lines.join('\n') : (res.text || ''));
}

async function extractPdf(path: string): Promise<{ text: string; pages: number }> {
  const parts: string[] = [];
  let pages = 0;
  for (let i = 0; i < MAX_PDF_PAGES; i++) {
    let thumb;
    try {
      thumb = await PdfThumbnail.generate(`file://${path}`, i, PDF_RENDER_QUALITY);
    } catch {
      break;   // past the last page (the renderer throws for an out-of-range index)
    }
    pages++;
    try {
      const t = await ocrImage(thumb.uri);
      if (t.trim()) parts.push(`[page ${i + 1}]\n${t}`);
    } finally {
      await fs.unlink(thumb.uri.replace('file://', '')).catch(() => {});
    }
  }
  return { text: parts.join('\n\n'), pages };
}

// ------------------------------------------------------------------ indexing

const SCHEMA = `CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL
);`;

function openDb(att: Attachment): DB {
  return open({ name: 'attach.sqlite', location: att.dir });
}

async function indexChunks(att: Attachment, chunks: string[], onStatus?: (s: string) => void): Promise<void> {
  const spec = att.embedder;
  const db = openDb(att);
  try {
    await db.execute(SCHEMA);
    for (let i = 0; i < chunks.length; i++) {
      onStatus?.(`Indexing ${att.name} — chunk ${i + 1}/${chunks.length}…`);
      const vec = await embedText(chunks[i], false, spec);
      await db.execute('INSERT INTO chunks (chunk_index, text, embedding) VALUES (?, ?, ?)',
        [i, chunks[i], vec.buffer as ArrayBuffer]);
    }
  } finally {
    db.close();
  }
}

function blobToVec(blob: any): Float32Array | null {
  if (blob instanceof ArrayBuffer) return new Float32Array(blob);
  if (blob?.buffer instanceof ArrayBuffer) {
    const bytes = Number(blob.byteLength ?? blob.buffer.byteLength - (blob.byteOffset ?? 0));
    return new Float32Array(blob.buffer, blob.byteOffset ?? 0, Math.floor(bytes / 4));
  }
  return null;
}

async function searchAttachment(att: Attachment, query: Float32Array, topK: number, minScore: number): Promise<SearchHit[]> {
  const db = openDb(att);
  try {
    const res = await db.execute('SELECT chunk_index, text, embedding FROM chunks ORDER BY chunk_index');
    const scored: Array<{ idx: number; text: string; score: number }> = [];
    for (const row of res.rows ?? []) {
      const vec = blobToVec((row as any).embedding);
      if (!vec || vec.length !== query.length) continue;
      let dot = 0;
      for (let i = 0; i < query.length; i++) dot += vec[i] * query[i];
      if (dot >= minScore) scored.push({ idx: Number((row as any).chunk_index), text: String((row as any).text), score: dot });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => ({
      doc_name: att.name, chunk_index: s.idx, text: s.text,
      score: Math.round(s.score * 1e4) / 1e4, source: 'attachment' as const,
    }));
  } finally {
    db.close();
  }
}

/** On-demand retrieval over one attachment (see runtime/attachments.context_for). */
export async function contextFor(att: Attachment, query: string): Promise<SearchHit[]> {
  if (!att.text) return [];
  if (att.small) {
    const chunks = splitText(att.text);
    return (chunks.length ? chunks : [att.text]).map((c, i) => ({
      doc_name: att.name, chunk_index: i, text: c, score: 1.0, source: 'attachment' as const,
    }));
  }
  const [ok] = shouldRetrieve(query);
  if (!ok) return [];
  const qvec = await embedText(query, true, att.embedder);
  return searchAttachment(att, qvec, ATTACH_TOP_K, ATTACH_MIN_SCORE);
}

// ------------------------------------------------------------------ ingest

/** validate -> copy -> extract (OCR) -> chunk (+embed when large). The guard
 *  runs afterwards in chat.ts, where the agent's model is available. */
export async function ingest(agentId: string, file: PickedFile,
                             onStatus?: (s: string) => void,
                             embedder: EmbedderSpec = DEFAULT_EMBEDDER): Promise<Attachment> {
  const { ext, kind } = validateMeta(file);
  const id = newId();
  const dir = `${attachRoot(agentId)}/${id}`;
  await fs.mkdir(attachRoot(agentId)).catch(() => {});
  await fs.mkdir(dir).catch(() => {});
  const path = `${dir}/file${ext}`;
  const timingsMs: Record<string, number> = {};

  try {
    onStatus?.(`Copying ${file.name}…`);
    let t = Date.now();
    await copyIn(file.uri, path);
    await validateBytes(path, kind);
    timingsMs.copy = Date.now() - t;

    t = Date.now();
    let text = '';
    let pages = 0;
    let ocrPages = 0;
    if (kind === 'text') {
      text = String(await fs.readFile(path, 'utf8'));
    } else if (kind === 'image') {
      onStatus?.('Reading the image (OCR)…');
      text = await ocrImage(path);
      pages = 1; ocrPages = 1;
    } else {
      onStatus?.('Rendering and reading PDF pages (OCR)…');
      const r = await extractPdf(path);
      text = r.text; pages = r.pages; ocrPages = r.pages;
    }
    text = text.replace(/\r\n?/g, '\n').replace(/[ \t]+\n/g, '\n').trim();
    timingsMs.extract = Date.now() - t;

    const att: Attachment = {
      id, agentId, name: file.name.slice(0, 120), ext, kind, bytes: file.size, dir, path,
      text, textChars: text.length, pages, ocrPages, chunks: 0,
      small: text.length <= FULLTEXT_CHARS, status: text ? 'ready' : 'no-text',
      guardEnabled: false, guard: null, judgedChunks: [], timingsMs, embedder,
    };

    t = Date.now();
    const chunks = text ? splitText(text) : [];
    att.chunks = chunks.length;
    if (chunks.length && !att.small) await indexChunks(att, chunks, onStatus);
    timingsMs.index = Date.now() - t;

    await log(`attach: ${att.name} ${kind} ${att.bytes}B -> ${att.textChars} chars, ${pages} pages, ` +
              `ocr=${ocrPages}, ${att.chunks} chunks, small=${att.small} ` +
              `(copy ${timingsMs.copy}ms, extract ${timingsMs.extract}ms, index ${timingsMs.index}ms)`);
    return att;
  } catch (e) {
    await fs.unlink(dir).catch(() => {});
    throw e;
  }
}
