"""Who still holds an installed copy of an agent.

The Studio owns the agent definition; devices own installed copies of it.
Deleting the definition while copies are still running leaves devices with
an agent that no longer exists anywhere else — no store entry, no version
lifecycle, no way to push a fix. So the Studio refuses to delete an agent
until nothing holds it, and this module answers the question it needs to ask.

Two kinds of holder:

* The **web runtime** (Journey 2) runs on this same machine, so its installed
  agents are read straight from `runtime/device_storage/agents/`. No HTTP,
  no race with a runtime that is down.
* **Android handsets** are only known through the governance telemetry they
  post: an `install` event for (device, agent) with no later `uninstall`
  event from the same device means the device still has it. The app never
  reported uninstalls before this change, so a handset that removed an agent
  with an older APK keeps counting as a holder until it either reports again
  or its record is older than the `stale_days` window.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import telemetry
from .paths import ROOT

RUNTIME_AGENTS_DIR = ROOT / "runtime" / "device_storage" / "agents"

# An Android device that has not reported ANY event for this long is presumed
# gone (wiped, reinstalled, retired). Without a cut-off, one old test phone
# would make its agents undeletable forever.
STALE_DAYS = 90


def web_holders(agent_id: str, agents_dir: Path | None = None) -> list[dict]:
    d = (agents_dir or RUNTIME_AGENTS_DIR) / agent_id
    manifest = d / "manifest.json"
    if not manifest.exists():
        return []
    try:
        version = json.loads(manifest.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        version = None
    return [{"device_id": "web-sim", "kind": "web", "version": version,
             "label": "Web runtime (Journey 2)"}]


def android_holders(agent_id: str, stale_days: int = STALE_DAYS,
                    db_path: Path | None = None) -> list[dict]:
    conn = telemetry.connect(db_path) if db_path else telemetry.connect()
    try:
        rows = conn.execute(
            """
            WITH per_device AS (
                SELECT device_id,
                       MAX(CASE WHEN event = 'install'   THEN ts END) AS last_install,
                       MAX(CASE WHEN event = 'uninstall' THEN ts END) AS last_uninstall,
                       MAX(CASE WHEN event = 'install'   THEN agent_version END) AS version
                FROM events
                WHERE source = 'android' AND agent_id = ? AND device_id != 'unknown'
                GROUP BY device_id
            ),
            seen AS (
                SELECT device_id, MAX(ts) AS last_seen, MAX(device_model) AS device_model
                FROM events WHERE source = 'android' GROUP BY device_id
            )
            SELECT p.device_id, s.device_model, p.version, p.last_install, s.last_seen
            FROM per_device p JOIN seen s USING (device_id)
            WHERE p.last_install IS NOT NULL
              AND (p.last_uninstall IS NULL OR p.last_uninstall < p.last_install)
              AND s.last_seen >= datetime('now', ?)
            ORDER BY s.last_seen DESC
            """,
            (agent_id, f"-{int(stale_days)} days"),
        ).fetchall()
    finally:
        conn.close()
    return [{"device_id": r[0], "kind": "android", "device_model": r[1], "version": r[2],
             "installed_at": r[3], "last_seen": r[4],
             "label": f"{r[1] or 'Android device'} ({r[0]})"} for r in rows]


def holders(agent_id: str, agents_dir: Path | None = None, db_path: Path | None = None,
            stale_days: int = STALE_DAYS) -> dict:
    web = web_holders(agent_id, agents_dir)
    android = android_holders(agent_id, stale_days, db_path)
    return {"agent_id": agent_id, "web": web, "android": android,
            "total": len(web) + len(android), "stale_days": stale_days}


def describe_block(h: dict) -> str:
    """The sentence the Studio shows when it refuses a delete."""
    parts = []
    if h["web"]:
        v = h["web"][0].get("version")
        parts.append(f"the web runtime (v{v})" if v else "the web runtime")
    n = len(h["android"])
    if n:
        names = ", ".join(x["device_model"] or x["device_id"] for x in h["android"][:3])
        more = f" and {n - 3} more" if n > 3 else ""
        parts.append(f"{n} Android device{'s' if n != 1 else ''} ({names}{more})")
    where = " and ".join(parts) or "a device"
    return (f"'{h['agent_id']}' is still installed on {where}. Uninstall it there first "
            f"(web runtime: the agent's Uninstall button; handsets: remove it in the app, or "
            f"disable its versions here so they stop running), then delete.")
