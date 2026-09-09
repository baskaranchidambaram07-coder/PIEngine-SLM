// Knowledge-base access: reads the bundle's kb.sqlite (built by the Studio)
// plus an optional device-local inline.sqlite created from user uploads.
// Same math as core/kbstore.py: cosine over normalized float32 blobs.
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

const INLINE_SCHEMA = `CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_name TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL
);`;

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

async function searchDb(
  db: DB, query: Float32Array, source: 'bundle' | 'inline',
): Promise<SearchHit[]> {
  const res = await db.execute('SELECT doc_name, chunk_index, text, embedding FROM chunks');
  const hits: SearchHit[] = [];
  for (const row of res.rows ?? []) {
    const vec = blobToVec((row as any).embedding);
    if (!vec || vec.length !== EMBEDDING_DIM) continue;
    let dot = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) dot += vec[i] * query[i];
    hits.push({
      doc_name: (row as any).doc_name,
      chunk_index: (row as any).chunk_index,
      text: (row as any).text,
      score: Math.round(dot * 1e4) / 1e4,
      source,
    });
  }
  return hits;
}

export async function search(
  agentId: string, query: Float32Array, topK: number, minScore: number,
): Promise<SearchHit[]> {
  const dir = `${AGENTS_DIR}/${agentId}`;
  const all: SearchHit[] = [];

  try {
    if (await ReactNativeBlobUtil.fs.exists(`${dir}/kb.sqlite`)) {
      const db = openDb(dir, 'kb.sqlite');
      try { all.push(...await searchDb(db, query, 'bundle')); } finally { db.close(); }
    } else {
      await log(`kb: no kb.sqlite at ${dir}`);
    }
    if (await ReactNativeBlobUtil.fs.exists(`${dir}/inline.sqlite`)) {
      const db = openDb(dir, 'inline.sqlite');
      try { all.push(...await searchDb(db, query, 'inline')); } finally { db.close(); }
    }
  } catch (e: any) {
    await log(`kb: search FAILED: ${String(e?.message || e).slice(0, 200)}`);
    throw new Error(`KB search failed: ${String(e?.message || e).slice(0, 200)}`);
  }

  all.sort((a, b) => b.score - a.score);
  const hits = all.filter(h => h.score >= minScore).slice(0, topK);
  await log(`kb: scored ${all.length} chunks, ${hits.length} hits >= ${minScore}`);
  return hits;
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
