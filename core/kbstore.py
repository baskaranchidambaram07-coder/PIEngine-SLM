"""Portable knowledge-base store: a single SQLite file with embedded vectors.

The same file format ships inside the agent bundle and is readable on-device
(SQLite runs everywhere, embeddings are raw float32 blobs). Search is
brute-force cosine over normalized vectors — for on-device KBs of a few
thousand chunks this is single-digit milliseconds and needs no index.
"""
from __future__ import annotations

import json
import sqlite3
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


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


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
    return doc_id


def delete_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
    conn.commit()


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
    rows = conn.execute("SELECT id, doc_name, chunk_index, text, embedding FROM chunks").fetchall()
    if not rows:
        return []
    mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), EMBEDDING_DIM)
    scores = mat @ query_vec.astype(np.float32)
    order = np.argsort(-scores)[:top_k]
    results = []
    for idx in order:
        score = float(scores[idx])
        if score < min_score:
            continue
        r = rows[int(idx)]
        results.append({
            "chunk_id": r[0], "doc_name": r[1], "chunk_index": r[2],
            "text": r[3], "score": round(score, 4),
        })
    return results
