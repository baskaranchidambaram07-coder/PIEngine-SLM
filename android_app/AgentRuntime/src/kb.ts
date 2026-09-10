// Knowledge-base access: reads the bundle's kb.sqlite (built by the Studio)
// plus an optional device-local inline.sqlite created from user uploads.
// Same math as core/kbstore.py: cosine over normalized float32 blobs.
//
// Two things keep the scan cheap, mirroring core/kbstore.py:
//   * vectors are decoded once into a flat Float32Array cached per KB file,
//     so a second question never re-reads the table or re-crosses the JSI
//     bridge for the blobs;
//   * chunk text is not selected while scoring — only the winning rows are
//     fetched by id.
// The cache is keyed on the file's size+mtime so a reinstalled bundle is
// picked up automatically; writes from this app clear it explicitly, since a
// WAL-mode write need not touch the main file's mtime.
import { open, DB } from '@op-engineering/op-sqlite';
import ReactNativeBlobUtil from 'react-native-blob-util';
import { AGENTS_DIR, EMBEDDING_DIM } from './config';
import { log } from './logger';

export type SearchHit = {
  doc_name: string;
  chunk_index: number;
  text: string;
  score: number;
  source: 'bundle' | 'inline';
};

type Source = 'bundle' | 'inline';
type Candidate = { id: number; score: number; source: Source };
type VecCache = { stamp: string; ids: number[]; mat: Float32Array };

const INLINE_SCHEMA = `CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_name TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL
);`;

// 384 dims x 4 bytes is ~1.5 KB per chunk; a handful of KB files stays small.
const CACHE_MAX_FILES = 6;
const vecCache = new Map<string, VecCache>();

function openDb(dir: string, name: string): DB {
  return open({ name, location: dir });
}

function blobToVec(blob: any): Float32Array | null {
  if (blob instanceof ArrayBuffer) return new Float32Array(blob);
  if (blob?.buffer instanceof ArrayBuffer) {
    return new Float32Array(blob.buffer, blob.byteOffset ?? 0, EMBEDDING_DIM);
  }
  return null;
}

/** Drop cached vectors — for one agent, or all of them when no id is given. */
export function clearVectorCache(agentId?: string): void {
  if (!agentId) {
    vecCache.clear();
    return;
  }
  const prefix = `${AGENTS_DIR}/${agentId}/`;
  for (const key of Array.from(vecCache.keys())) {
    if (key.startsWith(prefix)) vecCache.delete(key);
  }
}

async function fileStamp(path: string): Promise<string> {
  try {
    const st: any = await ReactNativeBlobUtil.fs.stat(path);
    return `${st.size}:${st.lastModified}`;
  } catch {
    return '';
  }
}

/** Chunk ids and their vectors as one flat matrix, cached per KB file. */
async function loadVectors(db: DB, path: string): Promise<VecCache> {
  const stamp = await fileStamp(path);
  const hit = vecCache.get(path);
  if (hit && stamp && hit.stamp === stamp) return hit;

  const t0 = Date.now();
  const res = await db.execute('SELECT id, embedding FROM chunks ORDER BY id');
  const rows = res.rows ?? [];
  const buf = new Float32Array(rows.length * EMBEDDING_DIM);
  const ids: number[] = [];
  for (const row of rows) {
    const vec = blobToVec((row as any).embedding);
    if (!vec || vec.length !== EMBEDDING_DIM) continue;
    buf.set(vec, ids.length * EMBEDDING_DIM);
    ids.push(Number((row as any).id));
  }

  const entry: VecCache = {
    stamp,
    ids,
    mat: buf.subarray(0, ids.length * EMBEDDING_DIM),
  };
  vecCache.set(path, entry);
  while (vecCache.size > CACHE_MAX_FILES) {
    const oldest = vecCache.keys().next().value;
    if (oldest === undefined) break;
    vecCache.delete(oldest);
  }
  await log(`kb: decoded ${ids.length} vectors in ${Date.now() - t0}ms (${path.split('/').pop()})`);
  return entry;
}

/** Bounded top-k cosine scan — allocates k entries, not one per chunk. */
function topCandidates(
  cache: VecCache, query: Float32Array, source: Source, k: number, minScore: number,
): Candidate[] {
  const best: Candidate[] = [];
  for (let r = 0; r < cache.ids.length; r++) {
    const off = r * EMBEDDING_DIM;
    let dot = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) dot += cache.mat[off + i] * query[i];
    if (dot < minScore) continue;
    if (best.length < k) {
      best.push({ id: cache.ids[r], score: dot, source });
      best.sort((a, b) => b.score - a.score);
    } else if (dot > best[best.length - 1].score) {
      best[best.length - 1] = { id: cache.ids[r], score: dot, source };
      best.sort((a, b) => b.score - a.score);
    }
  }
  return best;
}

async function fetchRows(db: DB, ids: number[]): Promise<Map<number, any>> {
  const placeholders = ids.map(() => '?').join(',');
  const res = await db.execute(
    `SELECT id, doc_name, chunk_index, text FROM chunks WHERE id IN (${placeholders})`,
    ids,
  );
  const byId = new Map<number, any>();
  for (const row of res.rows ?? []) byId.set(Number((row as any).id), row);
  return byId;
}

export async function search(
  agentId: string, query: Float32Array, topK: number, minScore: number,
): Promise<SearchHit[]> {
  if (topK <= 0) return [];
  const dir = `${AGENTS_DIR}/${agentId}`;
  const handles: Array<{ source: Source; db: DB }> = [];
  const candidates: Candidate[] = [];
  let scanned = 0;

  try {
    for (const [source, file] of [['bundle', 'kb.sqlite'], ['inline', 'inline.sqlite']] as const) {
      const path = `${dir}/${file}`;
      if (!(await ReactNativeBlobUtil.fs.exists(path))) {
        if (source === 'bundle') await log(`kb: no kb.sqlite at ${dir}`);
        continue;
      }
      const db = openDb(dir, file);
      handles.push({ source, db });
      const cache = await loadVectors(db, path);
      scanned += cache.ids.length;
      candidates.push(...topCandidates(cache, query, source, topK, minScore));
    }

    candidates.sort((a, b) => b.score - a.score);
    const winners = candidates.slice(0, topK);
    if (!winners.length) {
      await log(`kb: scored ${scanned} chunks, 0 hits >= ${minScore}`);
      return [];
    }

    const rowsBySource = new Map<Source, Map<number, any>>();
    for (const { source, db } of handles) {
      const ids = winners.filter(w => w.source === source).map(w => w.id);
      if (ids.length) rowsBySource.set(source, await fetchRows(db, ids));
    }

    const hits: SearchHit[] = [];
    for (const w of winners) {
      const row = rowsBySource.get(w.source)?.get(w.id);
      if (!row) continue;  // chunk vanished since the vectors were cached
      hits.push({
        doc_name: row.doc_name,
        chunk_index: row.chunk_index,
        text: row.text,
        score: Math.round(w.score * 1e4) / 1e4,
        source: w.source,
      });
    }
    await log(`kb: scored ${scanned} chunks, ${hits.length} hits >= ${minScore}`);
    return hits;
  } catch (e: any) {
    await log(`kb: search FAILED: ${String(e?.message || e).slice(0, 200)}`);
    throw new Error(`KB search failed: ${String(e?.message || e).slice(0, 200)}`);
  } finally {
    for (const { db } of handles) {
      try { db.close(); } catch {}
    }
  }
}

/** Store an uploaded document's chunks+embeddings as the agent's inline KB. */
export async function addInlineDoc(
  agentId: string, docName: string, chunks: string[], embeddings: Float32Array[],
): Promise<number> {
  const dir = `${AGENTS_DIR}/${agentId}`;
  const db = openDb(dir, 'inline.sqlite');
  try {
    await db.execute(INLINE_SCHEMA);
    for (let i = 0; i < chunks.length; i++) {
      await db.execute(
        'INSERT INTO chunks (doc_name, chunk_index, text, embedding) VALUES (?, ?, ?, ?)',
        [docName, i, chunks[i], embeddings[i].buffer as ArrayBuffer],
      );
    }
    const res = await db.execute('SELECT COUNT(*) AS n FROM chunks');
    return Number((res.rows?.[0] as any)?.n ?? chunks.length);
  } finally {
    db.close();
    clearVectorCache(agentId);
  }
}

export async function inlineStats(agentId: string): Promise<{ docs: number; chunks: number }> {
  const dir = `${AGENTS_DIR}/${agentId}`;
  if (!(await ReactNativeBlobUtil.fs.exists(`${dir}/inline.sqlite`))) {
    return { docs: 0, chunks: 0 };
  }
  const db = openDb(dir, 'inline.sqlite');
  try {
    await db.execute(INLINE_SCHEMA);
    const res = await db.execute(
      'SELECT COUNT(DISTINCT doc_name) AS d, COUNT(*) AS c FROM chunks');
    const row: any = res.rows?.[0] ?? {};
    return { docs: Number(row.d ?? 0), chunks: Number(row.c ?? 0) };
  } finally {
    db.close();
  }
}
