"""Who holds an installed agent (core/installs.py) and the Studio's delete gate."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException

from core import installs, telemetry


@pytest.fixture
def tdb(tmp_path):
    p = tmp_path / "telemetry.db"
    conn = telemetry.connect(p)
    conn.close()
    return p


def add(db: Path, **row):
    conn = sqlite3.connect(db)
    cols = ", ".join(row)
    conn.execute(f"INSERT INTO events ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
    conn.commit()
    conn.close()


def web_install(agents_dir: Path, agent_id: str, version: int):
    d = agents_dir / agent_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"id": agent_id, "version": version}))


def test_web_runtime_copy_is_a_holder(tmp_path, tdb):
    agents = tmp_path / "agents"
    web_install(agents, "acme", 3)
    h = installs.holders("acme", agents_dir=agents, db_path=tdb)
    assert h["total"] == 1 and h["web"][0]["version"] == 3 and h["android"] == []
    assert installs.holders("other", agents_dir=agents, db_path=tdb)["total"] == 0


def test_android_install_without_uninstall_is_a_holder(tmp_path, tdb):
    add(tdb, ts="2026-09-01T10:00:00Z", source="android", device_id="dev-1", device_model="Pixel 7",
        event="install", agent_id="acme", agent_version=2)
    add(tdb, ts="2026-09-02T10:00:00Z", source="android", device_id="dev-1", event="chat", agent_id="acme")
    h = installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb)
    assert [d["device_id"] for d in h["android"]] == ["dev-1"]
    assert h["android"][0]["device_model"] == "Pixel 7" and h["android"][0]["version"] == 2


def test_android_uninstall_releases_the_hold_and_reinstall_restores_it(tmp_path, tdb):
    add(tdb, ts="2026-09-01T10:00:00Z", source="android", device_id="dev-1", event="install", agent_id="acme")
    add(tdb, ts="2026-09-03T10:00:00Z", source="android", device_id="dev-1", event="uninstall", agent_id="acme")
    assert installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb)["total"] == 0
    add(tdb, ts="2026-09-04T10:00:00Z", source="android", device_id="dev-1", event="install", agent_id="acme")
    assert installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb)["total"] == 1


def test_stale_devices_are_forgotten(tmp_path, tdb):
    add(tdb, ts="2025-01-01T10:00:00Z", source="android", device_id="dev-old", event="install", agent_id="acme")
    assert installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb)["total"] == 0
    assert installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb, stale_days=100000)["total"] == 1


def test_web_runtime_installs_do_not_count_as_android(tmp_path, tdb):
    add(tdb, ts="2026-09-01T10:00:00Z", source="web-runtime", device_id="web-sim", event="install", agent_id="acme")
    h = installs.holders("acme", agents_dir=tmp_path / "none", db_path=tdb)
    assert h["total"] == 0   # the web copy is read from disk, not telemetry


def test_describe_block_names_the_holders(tmp_path, tdb):
    agents = tmp_path / "agents"
    web_install(agents, "acme", 5)
    add(tdb, ts="2026-09-01T10:00:00Z", source="android", device_id="dev-1", device_model="POCO M2", event="install", agent_id="acme")
    msg = installs.describe_block(installs.holders("acme", agents_dir=agents, db_path=tdb))
    assert "web runtime (v5)" in msg and "1 Android device (POCO M2)" in msg and "delete" in msg


def test_studio_refuses_to_delete_a_held_agent(monkeypatch):
    from studio import app as studio

    monkeypatch.setattr(studio, "load_agent", lambda conn, aid: {"id": aid})
    monkeypatch.setattr(studio, "db", lambda: None)
    held = {"agent_id": "acme", "web": [{"version": 2}], "android": [], "total": 1}
    monkeypatch.setattr(studio.installs, "holders", lambda aid: held)
    with pytest.raises(HTTPException) as ei:
        studio.delete_agent("acme")
    assert ei.value.status_code == 409 and "web runtime (v2)" in ei.value.detail

    info = studio.agent_installs("acme")
    assert info["deletable"] is False and "Uninstall it there first" in info["reason"]


def test_android_only_hold_can_be_forced_but_web_hold_cannot(monkeypatch):
    from studio import app as studio

    monkeypatch.setattr(studio, "load_agent", lambda conn, aid: {"id": aid})
    monkeypatch.setattr(studio, "db", lambda: None)
    monkeypatch.setattr(studio.telemetry, "record", lambda **k: None)
    android_only = {"agent_id": "acme", "web": [], "total": 1,
                    "android": [{"device_id": "dev-1", "device_model": "POCO", "version": 2}]}
    monkeypatch.setattr(studio.installs, "holders", lambda aid: android_only)
    with pytest.raises(HTTPException) as ei:
        studio.delete_agent("acme")
    assert ei.value.status_code == 409 and "force=true" in ei.value.detail

    # with force the delete proceeds past the guard (the rest is stubbed out)
    monkeypatch.setattr(studio.versions, "bundle_versions", lambda aid: [])
    monkeypatch.setattr(studio, "read_registry", lambda: [])
    monkeypatch.setattr(studio, "write_registry", lambda entries: None)
    monkeypatch.setattr(studio.versions, "forget_agent", lambda aid: None)

    class Conn:
        def execute(self, *a): return self
        def commit(self): pass
    monkeypatch.setattr(studio, "db", lambda: Conn())
    monkeypatch.setattr(studio, "kb_path", lambda aid: Path("nonexistent-kb.sqlite"))
    assert studio.delete_agent("acme", force=True)["ok"] is True

    web_hold = {"agent_id": "acme", "web": [{"version": 3}], "android": [], "total": 1}
    monkeypatch.setattr(studio.installs, "holders", lambda aid: web_hold)
    with pytest.raises(HTTPException) as ei:
        studio.delete_agent("acme", force=True)
    assert ei.value.status_code == 409
