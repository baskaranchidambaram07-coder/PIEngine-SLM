"""Per-version lifecycle state for published agent bundles.

The Studio's `agents` table holds one row per agent with a version counter, and
`bundles/registry.json` holds one entry per agent describing the version devices
should install. Neither can express "v1 is retired but v9 is fine", so this
module adds a small side-car — `bundles/versions.json` — recording only the
versions that are NOT active:

    {"meeting-intelligence": {"1": "deleted", "5": "disabled"}}

Absence means active. Keeping it sparse means an untouched deployment has no
file at all, and registry.json keeps its existing shape so already-installed
devices and the Android app are unaffected.

  disabled — the bundle stays on disk but is not servable and cannot be run.
             Reversible.
  deleted  — the bundle file is removed. Irreversible; the record is kept so
             the version cannot silently reappear and so history is auditable.

Lives in core/ because both journeys need it: the Studio writes it, and the
Runtime reads it to refuse serving or running a retired version. The Runtime
reads it from the shared bundles directory, so enforcement still works with the
Studio offline.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .paths import BUNDLES_DIR

ACTIVE, DISABLED, DELETED = "active", "disabled", "deleted"
_SETTABLE = {ACTIVE, DISABLED}

_BUNDLE_RE = re.compile(r"^(?P<id>.+)-v(?P<v>\d+)\.zip$")


def _path():
    return BUNDLES_DIR / "versions.json"


def read_states() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt side-car must not take retrieval down; treat as "all active"
        # rather than failing every publish and install.
        return {}


def _write(states: dict) -> None:
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(states, indent=2, sort_keys=True), encoding="utf-8")


def state_of(agent_id: str, version: int) -> str:
    entry = read_states().get(agent_id) or {}
    val = entry.get(str(version), ACTIVE)
    return val.get("state", ACTIVE) if isinstance(val, dict) else val


def set_state(agent_id: str, version: int, state: str) -> None:
    """Record a version's state. `active` clears the record entirely."""
    states = read_states()
    per = states.setdefault(agent_id, {})
    key = str(version)
    if state == ACTIVE:
        per.pop(key, None)
        if not per:
            states.pop(agent_id, None)
    else:
        per[key] = {"state": state,
                    "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")}
    _write(states)


def forget_agent(agent_id: str) -> None:
    """Drop all recorded state for an agent (used when the agent is deleted)."""
    states = read_states()
    if states.pop(agent_id, None) is not None:
        _write(states)


def bundle_name(agent_id: str, version: int) -> str:
    return f"{agent_id}-v{version}.zip"


def bundle_versions(agent_id: str) -> list[int]:
    """Versions with a bundle file still on disk, newest first."""
    out = []
    for p in BUNDLES_DIR.glob(f"{agent_id}-v*.zip"):
        m = _BUNDLE_RE.match(p.name)
        if m and m.group("id") == agent_id:
            out.append(int(m.group("v")))
    return sorted(out, reverse=True)


def known_versions(agent_id: str) -> list[int]:
    """Every version we know of — on disk plus recorded-but-deleted."""
    seen = set(bundle_versions(agent_id))
    seen.update(int(v) for v in (read_states().get(agent_id) or {}))
    return sorted(seen, reverse=True)


def is_servable(agent_id: str, version: int) -> bool:
    return (state_of(agent_id, version) == ACTIVE
            and (BUNDLES_DIR / bundle_name(agent_id, version)).exists())


def newest_active(agent_id: str) -> int | None:
    for v in bundle_versions(agent_id):
        if state_of(agent_id, v) == ACTIVE:
            return v
    return None


def parse_bundle(name: str) -> tuple[str, int] | None:
    """'meeting-intelligence-v9.zip' -> ('meeting-intelligence', 9)."""
    m = _BUNDLE_RE.match(name)
    return (m.group("id"), int(m.group("v"))) if m else None


def describe(agent_id: str, current_version: int | None = None) -> list[dict]:
    """Every known version with its state, newest first, for the UI."""
    out = []
    for v in known_versions(agent_id):
        st = state_of(agent_id, v)
        path = BUNDLES_DIR / bundle_name(agent_id, v)
        exists = path.exists()
        out.append({
            "version": v,
            "state": DELETED if (st == DELETED or not exists) else st,
            "bundle": bundle_name(agent_id, v),
            "bundle_bytes": path.stat().st_size if exists else 0,
            "on_disk": exists,
            "is_current": current_version == v,
        })
    return out
