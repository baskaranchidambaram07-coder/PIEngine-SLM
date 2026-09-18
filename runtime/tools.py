"""On-device tool dispatcher.

Mirrors what a mobile runtime would do natively:
- ``builtin`` tools touch device-local storage (calendar, action items, outbox,
  notes) and always work offline.
- ``http`` tools call enterprise APIs (Jira, Confluence, ...) when the device
  has connectivity and degrade gracefully — the agent gets a structured error
  it can explain to the user — when offline.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

DEVICE_DATA = Path(__file__).resolve().parent / "device_storage" / "device_data.json"


def _load() -> dict:
    """Read device_data.json, backfilling any list this file predates.

    A file already on disk from before "notes"/"events" existed would KeyError
    the first time either new tool ran, since json.loads returns exactly what
    was written — nothing here ever merged in new keys. setdefault leaves
    whatever is already on disk untouched and only fills in what's missing.
    """
    defaults = {"action_items": [], "outbox": [], "notes": [], "events": [], "calendar": None}
    if DEVICE_DATA.exists():
        data = json.loads(DEVICE_DATA.read_text(encoding="utf-8"))
        for key, value in defaults.items():
            data.setdefault(key, value)
        return data
    return dict(defaults)


def _save(data: dict) -> None:
    DEVICE_DATA.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_DATA.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mock_calendar() -> list[dict]:
    today = date.today().isoformat()
    return [
        {"time": "09:30", "date": today, "title": "Acme Corp — Phase 2 steering committee",
         "attendees": ["You", "Priya Sharma (Acme CIO)", "Daniel Weber (Acme PMO)"]},
        {"time": "13:00", "date": today, "title": "Internal delivery sync — Acme programme",
         "attendees": ["You", "Delivery team leads"]},
        {"time": "16:00", "date": today, "title": "Proposal review — retail analytics RFP",
         "attendees": ["You", "Presales team"]},
    ]


def get_device_data() -> dict:
    data = _load()
    if not data.get("calendar"):
        data["calendar"] = _mock_calendar()
        _save(data)
    return data


def execute_tool(tool_def: dict, args: dict) -> dict:
    kind = tool_def.get("kind", "builtin")
    name = tool_def.get("name", "?")

    if kind == "builtin":
        return _execute_builtin(name, args)
    if kind == "http":
        return _execute_http(tool_def, args)
    return {"error": f"unsupported tool kind '{kind}'"}


def _execute_builtin(name: str, args: dict) -> dict:
    data = get_device_data()

    if name == "get_todays_meetings":
        return {"meetings": data["calendar"]}

    if name == "create_action_item":
        item = {
            "id": len(data["action_items"]) + 1,
            "title": args.get("title", "Untitled action"),
            "owner": args.get("owner", "me"),
            "due_date": args.get("due_date") or (date.today() + timedelta(days=7)).isoformat(),
            "status": "open",
            "created": date.today().isoformat(),
        }
        data["action_items"].append(item)
        _save(data)
        return {"created": item, "note": "Stored on device; will sync to Jira when online."}

    if name == "draft_email":
        draft = {
            "id": len(data["outbox"]) + 1,
            "to": args.get("to", ""),
            "subject": args.get("subject", ""),
            "body": args.get("body", ""),
            "status": "draft — awaiting user review",
        }
        data["outbox"].append(draft)
        _save(data)
        return {"drafted": {k: draft[k] for k in ("id", "to", "subject", "status")},
                "note": "Draft saved to device outbox for user review. Never auto-sent."}

    if name == "schedule_event":
        start = args.get("start") or (datetime.now() + timedelta(days=1)).replace(
            hour=15, minute=0, second=0, microsecond=0).isoformat(timespec="minutes")
        event = {
            "id": len(data["events"]) + 1,
            "title": args.get("title", "Untitled event"),
            "start": start,
            "duration_minutes": int(args.get("duration_minutes") or 30),
            "description": args.get("description", ""),
            "attendees": args.get("attendees") or [],
            "status": "scheduled — awaiting user confirmation",
            "created": date.today().isoformat(),
        }
        data["events"].append(event)
        _save(data)
        # Echo back an ack, not the full record: the model just wrote
        # "description"/"attendees" itself (they're already in this round's
        # tool-call args, which the UI shows alongside this result), and
        # echoing prose fields back on every tool call measurably blew the
        # context budget on Android's small on-device context window by the
        # next round (2048 tokens; no such limit server-side).
        return {"scheduled": {"id": event["id"], "title": event["title"], "start": event["start"],
                               "duration_minutes": event["duration_minutes"], "status": event["status"]},
                "note": "Event created on the device calendar (simulated here). On a phone this "
                        "opens the Calendar app pre-filled for the user to review and tap Save."}


    return {"error": f"unknown builtin tool '{name}'"}


def _execute_http(tool_def: dict, args: dict) -> dict:
    import requests

    cfg = tool_def.get("config", {})
    url = cfg.get("url", "")
    try:
        for key, val in args.items():
            url = url.replace("{" + key + "}", requests.utils.quote(str(val)))
        resp = requests.request(
            cfg.get("method", "GET"), url,
            headers=cfg.get("headers", {}), timeout=6,
        )
        body = resp.text[:2000]
        try:
            body = resp.json()
        except ValueError:
            pass
        return {"status": resp.status_code, "body": body}
    except requests.RequestException as exc:
        return {"error": "endpoint unreachable (device may be offline)",
                "detail": str(exc)[:200],
                "hint": "Tell the user this tool needs connectivity and offer an offline alternative."}
