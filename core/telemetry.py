"""Shared governance telemetry — usage metadata only, never chat content.

Design principle: the on-device privacy promise is that message text, KB
passages, and answers never leave the handset. This store therefore records
only *metadata* — which agent/model, token counts, tokens/sec, load times, KB
hit counts, install/download events, and error messages (system strings, not
user text). The runtime (Journey 2) and the Android app write here; the Studio
(Journey 1) reads it for the Governance dashboard.

Writers open a fresh connection per call (SQLite connections are not shareable
across threads); WAL + busy_timeout make concurrent runtime-writes / studio-
reads safe.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .paths import TELEMETRY_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    source        TEXT NOT NULL,        -- 'android' | 'web-runtime' | 'portal'
    device_id     TEXT NOT NULL DEFAULT 'unknown',
    device_model  TEXT,                 -- e.g. 'Pixel 7' (android, optional)
    event         TEXT NOT NULL,        -- install | chat | model_load | error
                                        -- | bundle_download | model_download
                                        -- | adapter_download | apk_download
    agent_id      TEXT,
    agent_version INTEGER,
    model_id      TEXT,
    adapter_id    TEXT,                 -- LoRA adapter applied, if any
    tokens        INTEGER,
    tok_per_sec   REAL,
    prefill_tokens INTEGER,
    load_ms       INTEGER,
    kb_hits       INTEGER,
    duration_ms   INTEGER,
    bytes         INTEGER,
    ok            INTEGER,              -- 1 success / 0 failure
    detail        TEXT,                 -- error text / note (no user content)
    meta          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
"""

# Columns the caller may set directly (everything except id/ts).
_FIELDS = (
    "source", "device_id", "device_model", "event", "agent_id", "agent_version",
    "model_id", "adapter_id", "tokens", "tok_per_sec", "prefill_tokens", "load_ms", "kb_hits",
    "duration_ms", "bytes", "ok", "detail", "meta",
)

# Never let these leak in even if a caller passes them by mistake.
_CONTENT_KEYS = {"query", "prompt", "message", "messages", "answer", "text", "content"}


_SCHEMA_READY = False


def connect(path: str | Path = TELEMETRY_DB, ensure_schema: bool = True) -> sqlite3.Connection:
    global _SCHEMA_READY
    conn = sqlite3.connect(str(path), timeout=8.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    # Only run the schema DDL once per process — repeated CREATE ... IF NOT
    # EXISTS on every insert is a needless write-lock contender.
    if ensure_schema and not _SCHEMA_READY:
        conn.executescript(SCHEMA)
        _migrate(conn)
        _SCHEMA_READY = True
    return conn


# Columns added after the first release. CREATE TABLE IF NOT EXISTS is a no-op
# on an existing table, so new columns have to be added explicitly or every
# insert fails on a database that predates them.
_ADDED_COLUMNS = {"adapter_id": "TEXT"}


def _migrate(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    for col, decl in _ADDED_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(**fields) -> None:
    """Best-effort insert. Strips any content-bearing keys defensively and
    never raises into the request path."""
    for k in list(fields):
        if k in _CONTENT_KEYS:
            fields.pop(k)
    meta = fields.get("meta")
    if isinstance(meta, dict):
        fields["meta"] = json.dumps(meta)
    row = {k: fields.get(k) for k in _FIELDS}
    # NOT NULL columns: an explicit NULL in the INSERT bypasses the SQL DEFAULT,
    # so fill them here (portal events carry no device_id).
    row["device_id"] = row.get("device_id") or "unknown"
    row["source"] = row.get("source") or "unknown"
    row["event"] = row.get("event") or "event"
    if not row.get("meta"):
        row["meta"] = "{}"
    cols = ["ts"] + list(_FIELDS)
    placeholders = ", ".join(["?"] * len(cols))
    values = [_now()] + [row[k] for k in _FIELDS]
    sql = f"INSERT INTO events ({', '.join(cols)}) VALUES ({placeholders})"
    # Retry through transient "database is locked" without ever raising into
    # the request path. Low volume, so a few short retries fully cover it.
    for attempt in range(4):
        try:
            conn = connect()
            try:
                conn.execute(sql, values)
                conn.commit()
                return
            finally:
                conn.close()
        except sqlite3.OperationalError:
            time.sleep(0.15 * (attempt + 1))
        except sqlite3.Error:
            return  # non-transient — drop the event, never break the feature


# ------------------------------------------------------------------ reads

def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def summary(days: int = 30) -> dict:
    """KPIs for the governance dashboard header."""
    conn = connect()
    try:
        since = f"-{int(days)} days"
        def scalar(sql, params=()):
            return conn.execute(sql, params).fetchone()[0]

        total_events = scalar("SELECT COUNT(*) FROM events WHERE ts >= datetime('now', ?)", (since,))
        installs = scalar("SELECT COUNT(*) FROM events WHERE event='install' AND ok=1 AND ts >= datetime('now', ?)", (since,))
        chats = scalar("SELECT COUNT(*) FROM events WHERE event='chat' AND ts >= datetime('now', ?)", (since,))
        tokens = scalar("SELECT COALESCE(SUM(tokens),0) FROM events WHERE event='chat' AND ts >= datetime('now', ?)", (since,)) or 0
        devices = scalar("SELECT COUNT(DISTINCT device_id) FROM events WHERE device_id!='unknown' AND ts >= datetime('now', ?)", (since,))
        errors = scalar("SELECT COUNT(*) FROM events WHERE (event='error' OR ok=0) AND ts >= datetime('now', ?)", (since,))
        avg_tps = scalar("SELECT ROUND(AVG(tok_per_sec),1) FROM events WHERE event='chat' AND tok_per_sec > 0 AND ts >= datetime('now', ?)", (since,))
        avg_load = scalar("SELECT ROUND(AVG(load_ms)) FROM events WHERE event='model_load' AND load_ms > 0 AND ts >= datetime('now', ?)", (since,))
        bytes_served = scalar("SELECT COALESCE(SUM(bytes),0) FROM events WHERE event IN ('bundle_download','model_download','apk_download') AND ts >= datetime('now', ?)", (since,)) or 0
        chat_ok = scalar("SELECT COUNT(*) FROM events WHERE event='chat' AND ts >= datetime('now', ?)", (since,))
        chat_fail = scalar("SELECT COUNT(*) FROM events WHERE event='error' AND ts >= datetime('now', ?)", (since,))

        return {
            "days": days,
            "total_events": total_events,
            "installs": installs,
            "chats": chats,
            "tokens": tokens,
            "devices": devices,
            "errors": errors,
            "avg_tok_per_sec": avg_tps,
            "avg_load_ms": avg_load,
            "bytes_served": bytes_served,
            "reliability_pct": (round(100 * chat_ok / (chat_ok + chat_fail)) if (chat_ok + chat_fail) else None),
        }
    finally:
        conn.close()


def per_agent(days: int = 30) -> list[dict]:
    conn = connect()
    try:
        since = f"-{int(days)} days"
        return _rows(conn, """
            SELECT agent_id,
                   COUNT(*) FILTER (WHERE event='install' AND ok=1)  AS installs,
                   COUNT(*) FILTER (WHERE event='chat')              AS chats,
                   COALESCE(SUM(tokens) FILTER (WHERE event='chat'),0) AS tokens,
                   ROUND(AVG(tok_per_sec) FILTER (WHERE event='chat' AND tok_per_sec>0),1) AS avg_tps,
                   ROUND(AVG(kb_hits) FILTER (WHERE event='chat'),1)  AS avg_kb_hits,
                   COUNT(*) FILTER (WHERE event='error')             AS errors,
                   MAX(ts)                                           AS last_seen
            FROM events
            WHERE agent_id IS NOT NULL AND ts >= datetime('now', ?)
            GROUP BY agent_id ORDER BY chats DESC, installs DESC
        """, (since,))
    finally:
        conn.close()


def per_model(days: int = 30) -> list[dict]:
    conn = connect()
    try:
        since = f"-{int(days)} days"
        return _rows(conn, """
            SELECT model_id,
                   COUNT(*) FILTER (WHERE event='chat') AS chats,
                   ROUND(AVG(tok_per_sec) FILTER (WHERE event='chat' AND tok_per_sec>0),1) AS avg_tps,
                   ROUND(AVG(load_ms) FILTER (WHERE event='model_load' AND load_ms>0)) AS avg_load_ms,
                   COUNT(*) FILTER (WHERE event='error') AS errors
            FROM events
            WHERE model_id IS NOT NULL AND ts >= datetime('now', ?)
            GROUP BY model_id ORDER BY chats DESC
        """, (since,))
    finally:
        conn.close()


def daily_activity(days: int = 14) -> list[dict]:
    conn = connect()
    try:
        since = f"-{int(days)} days"
        return _rows(conn, """
            SELECT substr(ts,1,10) AS day,
                   COUNT(*) FILTER (WHERE event='chat')    AS chats,
                   COUNT(*) FILTER (WHERE event='install') AS installs,
                   COUNT(*) FILTER (WHERE event='error')   AS errors
            FROM events
            WHERE ts >= datetime('now', ?)
            GROUP BY day ORDER BY day
        """, (since,))
    finally:
        conn.close()


def recent_events(limit: int = 100, event: str | None = None,
                  agent_id: str | None = None) -> list[dict]:
    conn = connect()
    try:
        where, params = [], []
        if event:
            where.append("event = ?"); params.append(event)
        if agent_id:
            where.append("agent_id = ?"); params.append(agent_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(int(limit))
        return _rows(conn, f"""
            SELECT ts, source, device_id, device_model, event, agent_id,
                   agent_version, model_id, tokens, tok_per_sec, load_ms,
                   kb_hits, ok, detail
            FROM events {clause} ORDER BY id DESC LIMIT ?
        """, tuple(params))
    finally:
        conn.close()


def devices(days: int = 30) -> list[dict]:
    conn = connect()
    try:
        since = f"-{int(days)} days"
        return _rows(conn, """
            SELECT device_id,
                   MAX(device_model) AS device_model,
                   MAX(source)       AS source,
                   COUNT(*) FILTER (WHERE event='chat')    AS chats,
                   COUNT(*) FILTER (WHERE event='install') AS installs,
                   MIN(ts) AS first_seen, MAX(ts) AS last_seen
            FROM events
            WHERE device_id != 'unknown' AND ts >= datetime('now', ?)
            GROUP BY device_id ORDER BY last_seen DESC
        """, (since,))
    finally:
        conn.close()
