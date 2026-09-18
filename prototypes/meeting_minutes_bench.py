"""Live end-to-end check of the Meeting Minutes agent against the RUNNING
runtime (port 8200): upload the synthetic transcript, ask "analyse this",
stream the real digest + tool-calling turn, and print what actually landed in
device_data.json afterward.

    venv\\Scripts\\python prototypes\\meeting_minutes_bench.py

Requires scripts/make_meeting_transcript_sample.py to have been run and the
"meeting-minutes" agent to be installed (see the Studio / this session's setup).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = ROOT / "sample_docs" / "attachments" / "meeting-transcript-acme-phase2.pdf"
DEVICE_DATA = ROOT / "runtime" / "device_storage" / "device_data.json"
BASE = "http://127.0.0.1:8200"
AGENT = "meeting-minutes"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    before = json.loads(DEVICE_DATA.read_text(encoding="utf-8")) if DEVICE_DATA.exists() else {}
    before_counts = {k: len(before.get(k, []) or []) for k in ("notes", "events", "outbox", "action_items")}
    print("before:", before_counts)

    with TRANSCRIPT.open("rb") as fh:
        up = requests.post(f"{BASE}/api/attachments", files={"file": (TRANSCRIPT.name, fh)},
                           data={"agent_id": AGENT}, timeout=300)
    up.raise_for_status()
    meta = up.json()
    print(f"upload: kind={meta['kind']} pages={meta['pages']} text_chars={meta['text_chars']} "
          f"small={meta['small']} chunks={meta['chunks']} guard={meta['guard']}")
    if meta["small"]:
        print("WARNING: attachment was small enough for whole-inject; the digest path will not run.")

    t0 = time.time()
    statuses: list[str] = []
    tool_calls: list[dict] = []
    answer = ""
    last_stats = None
    with requests.post(f"{BASE}/api/chat", json={"agent_id": AGENT, "attachment_id": meta["id"],
                                                  "messages": [{"role": "user", "content": "analyse this"}]},
                       stream=True, timeout=1800) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "status":
                statuses.append(ev["text"])
                print(f"  [{time.time() - t0:6.1f}s] status: {ev['text']}")
            elif ev["type"] == "tool":
                tool_calls.append(ev)
                print(f"  [{time.time() - t0:6.1f}s] tool: {ev['name']}({json.dumps(ev['args'])[:160]})")
            elif ev["type"] == "sources":
                print(f"  [{time.time() - t0:6.1f}s] sources: {len(ev['items'])} item(s), "
                      f"kinds={sorted({i['kind'] for i in ev['items']})}")
            elif ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "stats":
                last_stats = ev
            elif ev["type"] == "guard":
                print(f"  [{time.time() - t0:6.1f}s] GUARD REFUSED: {ev['text']}")
            elif ev["type"] == "error":
                print(f"  [{time.time() - t0:6.1f}s] ERROR: {ev['text']}")
    total_s = time.time() - t0

    print(f"\ntotal: {total_s:.1f}s | digest sections: {sum(1 for s in statuses if 'Reading section' in s)} "
         f"| tool calls: {[t['name'] for t in tool_calls]}")
    print(f"final answer: {answer.strip()!r}")
    if last_stats:
        print(f"last-round stats: {last_stats}")

    requests.delete(f"{BASE}/api/attachments/{meta['id']}", timeout=10)

    after = json.loads(DEVICE_DATA.read_text(encoding="utf-8"))
    after_counts = {k: len(after.get(k, []) or []) for k in ("notes", "events", "outbox", "action_items")}
    print("after: ", after_counts)

    new_notes = (after.get("notes") or [])[before_counts["notes"]:]
    new_emails = (after.get("outbox") or [])[before_counts["outbox"]:]
    new_events = (after.get("events") or [])[before_counts["events"]:]
    new_actions = (after.get("action_items") or [])[before_counts["action_items"]:]

    print(f"\n--- new note(s): {len(new_notes)} ---")
    for n in new_notes:
        print(f"  {n['title']}\n  {n['body']}")
    print(f"\n--- new email(s): {len(new_emails)} ---")
    for e in new_emails:
        print(f"  to={e['to']!r} subject={e['subject']!r}\n  {e['body']}")
    print(f"\n--- new event(s): {len(new_events)} ---")
    for ev in new_events:
        print(f"  {ev['title']} @ {ev['start']} ({ev['duration_minutes']}min) — {ev['description']}")
    if new_actions:
        print(f"\n--- new action item(s): {len(new_actions)} ---")
        for a in new_actions:
            print(f"  {a['title']} — owner={a.get('owner')} due={a.get('due_date')}")

    ok = bool(new_notes) and bool(new_emails) and bool(new_events)
    print(f"\nRESULT: {'PASS' if ok else 'INCOMPLETE'} — "
         f"note={'yes' if new_notes else 'NO'}, email={'yes' if new_emails else 'NO'}, "
         f"event={'yes' if new_events else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
