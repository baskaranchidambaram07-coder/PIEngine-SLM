"""Portable knowledge-base store: a single SQLite file with embedded vectors.

The same file format ships inside the agent bundle and is readable on-device
(SQLite runs everywhere, embeddings are raw float32 blobs). Search is
brute-force cosine over normalized vectors — for on-device KBs of a few
thousand chunks this is single-digit milliseconds and needs no index.

Two things keep the scan cheap as KBs grow:
  * the vector matrix is cached in memory per KB file, so repeated queries
    skip the table scan and the blob decode entirely;
  * chunk text is never read while scoring — only the top_k winners are
    fetched by id.
The cache is keyed on the file's mtime+size, so a KB rebuilt by the Studio or
a kb.sqlite replaced by a bundle install is picked up without a restart.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .embeddings import EMBEDDING_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    doc_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# How many distinct KB files keep a cached matrix. 384 dims x 4 bytes is
# ~1.5 KB per chunk, so a few thousand chunks per agent stays in single-digit MB.
CACHE_MAX_FILES = 8

# path -> (stamp, chunk_ids, matrix)
_cache: OrderedDict[str, tuple[tuple[int, int], list[int], np.ndarray]] = OrderedDict()


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _db_path(conn: sqlite3.Connection) -> str:
    for _seq, name, file in conn.execute("PRAGMA database_list"):
        if name == "main":
            return file or ""
    return ""


def _stamp(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def invalidate(conn: sqlite3.Connection) -> None:
    """Drop the cached matrix for this KB (called after every write)."""
    _cache.pop(_db_path(conn), None)


def _vectors(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Chunk ids and their normalized vectors, cached per KB file."""
    path = _db_path(conn)
    stamp = _stamp(path) if path else None

    if stamp is not None:
        hit = _cache.get(path)
        if hit is not None and hit[0] == stamp:
            _cache.move_to_end(path)
            return hit[1], hit[2]

    rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    ids = [r[0] for r in rows]
    if rows:
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
        mat = mat.reshape(len(rows), EMBEDDING_DIM)
    else:
        mat = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if stamp is not None:
        _cache[path] = (stamp, ids, mat)
        _cache.move_to_end(path)
        while len(_cache) > CACHE_MAX_FILES:
            _cache.popitem(last=False)
    return ids, mat


def add_document(conn: sqlite3.Connection, name: str, chunks: list[str],
                 embeddings: np.ndarray, meta: dict | None = None) -> int:
    cur = conn.execute("INSERT INTO docs (name, meta) VALUES (?, ?)",
                       (name, json.dumps(meta or {})))
    doc_id = cur.lastrowid
    rows = [
        (doc_id, name, i, text, emb.astype(np.float32).tobytes())
        for i, (text, emb) in enumerate(zip(chunks, embeddings))
    ]
    conn.executemany(
        "INSERT INTO chunks (doc_id, doc_name, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    invalidate(conn)
    return doc_id


def delete_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
    conn.commit()
    invalidate(conn)


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT d.id, d.name, d.meta, d.created_at, COUNT(c.id) AS chunk_count
           FROM docs d LEFT JOIN chunks c ON c.doc_id = d.id
           GROUP BY d.id ORDER BY d.id"""
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "meta": json.loads(r[2]), "created_at": r[3], "chunks": r[4]}
        for r in rows
    ]


def stats(conn: sqlite3.Connection) -> dict:
    docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"docs": docs, "chunks": chunks}


def search(conn: sqlite3.Connection, query_vec: np.ndarray, top_k: int = 4,
           min_score: float = 0.0) -> list[dict]:
    if top_k <= 0:
        return []
    ids, mat = _vectors(conn)
    if not ids:
        return []

    scores = mat @ query_vec.astype(np.float32)
    k = min(top_k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    winners = [(ids[int(i)], float(scores[int(i)])) for i in top
               if float(scores[int(i)]) >= min_score]
    if not winners:
        return []

    placeholders = ",".join("?" * len(winners))
    rows = {
        r[0]: r for r in conn.execute(
            f"SELECT id, doc_name, chunk_index, text FROM chunks WHERE id IN ({placeholders})",
            [w[0] for w in winners],
        )
    }

    results = []
    for chunk_id, score in winners:
        row = rows.get(chunk_id)
        if row is None:
            continue  # chunk deleted by another process since the matrix was cached
        results.append({
            "chunk_id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "text": row[3], "score": round(score, 4),
        })
    return results
