"""The device-action tools (schedule_event, draft_email — draft_email's device
handoff is Android-only and untestable here) and the whole-document analysis
mode they were built to support.

save_note existed briefly alongside these but was removed (2026-09-18): it
proved unreliable on-device and was dropped from the builtin tool catalogue
entirely, not just unassigned from any one agent — see runtime/tools.py,
android_app/AgentRuntime/src/tools.ts, and studio's TOOL_MIN_APP_VERSION /
TOOL_TEMPLATES, none of which mention it any more.

Two things are being proven, deliberately kept in separate test classes:
  1. runtime/tools.py: the new builtins, and the _load() default-merge fix
     that makes adding them to an EXISTING device_data.json safe at all.
  2. runtime/app.py: analysis_mode_enabled() is opt-in and off by default —
     an agent with no "analysis" key (every agent installed before this
     change) must behave byte-for-byte as it did before.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime import app as rt
from runtime import attachments, guard, vision
from runtime import tools as rt_tools

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs" / "attachments"


# ------------------------------------------------------------------ runtime/tools.py

@pytest.fixture(autouse=True)
def isolated_device_data(tmp_path, monkeypatch):
    monkeypatch.setattr(rt_tools, "DEVICE_DATA", tmp_path / "device_data.json")
    yield


def test_load_defaults_when_no_file_exists():
    assert rt_tools._load() == {"action_items": [], "outbox": [], "notes": [], "events": [], "calendar": None}


def test_load_backfills_missing_keys_without_touching_what_is_there():
    # exactly the shape of a device_data.json written before notes/events existed
    rt_tools.DEVICE_DATA.write_text(json.dumps({"action_items": [{"id": 1, "title": "keep me"}],
                                                "outbox": [], "calendar": None}))
    data = rt_tools._load()
    assert data["action_items"] == [{"id": 1, "title": "keep me"}]
    assert data["notes"] == [] and data["events"] == []


def test_schedule_event_creates_and_persists():
    result = rt_tools.execute_tool({"kind": "builtin", "name": "schedule_event"},
                                   {"title": "Follow-up call", "start": "2026-09-20T15:00:00",
                                    "duration_minutes": 45, "attendees": ["Priya"],
                                    "description": "Review the revised numbers from Tuesday's session."})
    assert result["scheduled"]["title"] == "Follow-up call"
    assert result["scheduled"]["duration_minutes"] == 45
    assert "Calendar app" in result["note"]
    # The full record — including the prose "description" and "attendees"
    # the model just wrote in its own tool-call args — is persisted intact;
    # only the echo fed back to the model is trimmed, to keep a phone's small
    # on-device context from filling up on round-tripped duplicate text.
    stored = rt_tools._load()["events"][0]
    assert stored["description"] == "Review the revised numbers from Tuesday's session."
    assert stored["attendees"] == ["Priya"]
    assert "description" not in result["scheduled"]
    assert "attendees" not in result["scheduled"]
    assert result["scheduled"] == {k: stored[k] for k in ("id", "title", "start", "duration_minutes", "status")}


def test_schedule_event_fills_in_a_start_time_when_none_given():
    result = rt_tools.execute_tool({"kind": "builtin", "name": "schedule_event"}, {"title": "No time given"})
    assert result["scheduled"]["start"]


def test_save_note_no_longer_exists():
    result = rt_tools.execute_tool({"kind": "builtin", "name": "save_note"}, {"title": "x", "body": "y"})
    assert result == {"error": "unknown builtin tool 'save_note'"}


def test_existing_three_builtins_are_unaffected():
    a = rt_tools.execute_tool({"kind": "builtin", "name": "create_action_item"}, {"title": "x"})
    assert a["created"]["title"] == "x"
    e = rt_tools.execute_tool({"kind": "builtin", "name": "draft_email"},
                              {"to": "a@b.com", "subject": "s", "body": "b"})
    assert e["drafted"]["to"] == "a@b.com"
    m = rt_tools.execute_tool({"kind": "builtin", "name": "get_todays_meetings"}, {})
    assert "meetings" in m


# ------------------------------------------------------------------ analysis_mode_enabled

def test_analysis_mode_enabled_is_opt_in_and_default_safe():
    assert rt.analysis_mode_enabled({}) is False                          # no key at all
    assert rt.analysis_mode_enabled({"analysis": None}) is False           # explicit null
    assert rt.analysis_mode_enabled({"analysis": {}}) is False             # empty dict
    assert rt.analysis_mode_enabled({"analysis": {"enabled": False}}) is False
    assert rt.analysis_mode_enabled({"analysis": {"enabled": True}}) is True


def test_studio_default_analysis_and_schema():
    from studio import app as studio
    assert studio.DEFAULT_ANALYSIS == {"enabled": False}
    cfg = studio.AgentIn(name="x")
    assert cfg.analysis == {"enabled": False}
    assert cfg.model_dump()["analysis"] == {"enabled": False}


# ------------------------------------------------------------------ end-to-end digest

pytestmark = pytest.mark.skipif(not SAMPLES.exists(),
                                reason="run scripts/make_attachment_samples.py first")


@pytest.fixture(autouse=True)
def isolated_attachments(tmp_path, monkeypatch):
    att_dir = tmp_path / "attachments"
    att_dir.mkdir()
    monkeypatch.setattr(attachments, "ATTACH_DIR", att_dir)
    monkeypatch.setattr(guard, "POLICY_PATH", tmp_path / "guard_policy.json")
    yield
    shutil.rmtree(att_dir, ignore_errors=True)


class ScriptedModel:
    """Distinguishes the digest map-pass calls (system prompt ==
    DIGEST_SYSTEM_PROMPT) from the main turn, and scripts a fixed tool-call
    sequence for the latter. The goal is to prove the WIRING — digest feeds
    the context, the context reaches the model, tool calls execute in order —
    not model quality; that's prototypes/attach_bench.py's job."""

    def __init__(self, tool_calls=None, digest_reply="- a key point was raised", final_text="Done."):
        self.tool_calls = tool_calls or []
        self.digest_reply = digest_reply
        self.final_text = final_text
        self.calls: list[list[dict]] = []

    def stream_chat(self, messages, generation, url=None):
        # runtime/app.py's tool loop keeps appending to the SAME messages list
        # across rounds, so a stored reference would show every call as the
        # final state by the time the request finishes — snapshot a copy.
        self.calls.append(list(messages))
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if system == rt.DIGEST_SYSTEM_PROMPT:
            for piece in (self.digest_reply[i:i + 7] for i in range(0, len(self.digest_reply), 7)):
                yield {"type": "token", "text": piece}
            yield {"type": "stats", "tokens": 5, "seconds": 0.05, "tok_per_sec": 100.0}
            return
        # Main turn: each round of runtime/app.py's tool loop appends an
        # assistant tool_call + a user tool_response per executed tool, so
        # counting tool_response messages already in the transcript says
        # which scripted call (if any) is still owed.
        done = sum(1 for m in messages if m["role"] == "user" and "<tool_response>" in m.get("content", ""))
        if done < len(self.tool_calls):
            call = self.tool_calls[done]
            yield {"type": "tool_call", "raw": json.dumps(call), "call": call}
            yield {"type": "stats", "tokens": 8, "seconds": 0.05, "tok_per_sec": 100.0}
            return
        for piece in (self.final_text[i:i + 7] for i in range(0, len(self.final_text), 7)):
            yield {"type": "token", "text": piece}
        yield {"type": "stats", "tokens": 5, "seconds": 0.05, "tok_per_sec": 100.0}


TOOL_DEFS = [
    {"name": "draft_email", "kind": "builtin", "description": "", "parameters": {}},
    {"name": "schedule_event", "kind": "builtin", "description": "", "parameters": {}},
]

# Long enough to clear FULLTEXT_CHARS (3500) so the "small" whole-inject path
# never fires and the test actually exercises the digest branch.
TRANSCRIPT = "\n\n".join(f"Section {i}: participants discussed topic {i} at some length. " * 20
                        for i in range(6))


def _install(monkeypatch, model, manifest):
    monkeypatch.setattr(rt, "load_manifest", lambda aid: manifest)
    monkeypatch.setattr(rt, "model_state", lambda m: "ready")
    monkeypatch.setattr(rt, "adapter_path", lambda m: None)
    monkeypatch.setattr(rt.versions, "state_of", lambda a, v: rt.versions.ACTIVE)
    monkeypatch.setattr(rt.llm, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(rt.llm, "status", lambda: {"model": "m.gguf", "adapter": None})
    monkeypatch.setattr(rt.llm, "stream_chat", model.stream_chat)
    monkeypatch.setattr(rt, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(rt.telemetry, "record", lambda **k: None)
    monkeypatch.setattr(vision, "available", lambda: False)
    return TestClient(rt.app)


def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.split("\n") if line.startswith("data: ")]


def test_analysis_mode_digests_a_transcript_and_drives_both_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(rt_tools, "DEVICE_DATA", tmp_path / "device_data.json")
    model = ScriptedModel(tool_calls=[
        {"name": "draft_email", "arguments": {"to": "team@example.com", "subject": "Summary", "body": "..."}},
        {"name": "schedule_event", "arguments": {"title": "Follow-up", "start": "2026-09-20T15:00:00"}},
    ])
    manifest = {
        "id": "meeting-minutes-test", "name": "Meeting Minutes Test", "version": 1, "description": "",
        "system_prompt": "You are a meeting-minutes agent.", "generation": {}, "rag": {"top_k": 4},
        "guardrails": {"pii": False}, "analysis": {"enabled": True}, "tools": TOOL_DEFS,
        "model": {"id": "m", "name": "M", "file": "m.gguf", "context_length": 8192},
    }
    client = _install(monkeypatch, model, manifest)

    up = client.post("/api/attachments", files={"file": ("transcript.txt", TRANSCRIPT.encode())},
                     data={"agent_id": "meeting-minutes-test"})
    assert up.status_code == 200, up.text
    meta = up.json()
    assert meta["small"] is False, "fixture must exceed FULLTEXT_CHARS to exercise the digest path"

    r = client.post("/api/chat", json={"agent_id": "meeting-minutes-test", "attachment_id": meta["id"],
                                       "messages": [{"role": "user", "content": "analyse this"}]})
    assert r.status_code == 200, r.text
    events = _events(r)

    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert any("Reading section" in s for s in statuses), "digest progress must be visible, not silent"

    tool_events = [e for e in events if e["type"] == "tool"]
    assert [t["name"] for t in tool_events] == ["draft_email", "schedule_event"]

    digest_calls = [c for c in model.calls if c[0]["content"] == rt.DIGEST_SYSTEM_PROMPT]
    assert len(digest_calls) >= 1
    main_calls = [c for c in model.calls if c[0]["content"] != rt.DIGEST_SYSTEM_PROMPT]
    assert "condensed digest of the whole document" in main_calls[0][-1]["content"]

    data = rt_tools._load()
    assert data["outbox"] and data["outbox"][0]["subject"] == "Summary"
    assert data["events"] and data["events"][0]["title"] == "Follow-up"


def test_analysis_mode_off_keeps_the_existing_retrieval_path(monkeypatch, tmp_path):
    """The exact manifest shape of every agent installed before this change
    (msock-advisor, product-insights: no "analysis" key at all) must never
    take the digest branch."""
    monkeypatch.setattr(rt_tools, "DEVICE_DATA", tmp_path / "device_data.json")
    model = ScriptedModel(final_text="No decisions found.")
    manifest = {
        "id": "no-analysis-test", "name": "No Analysis", "version": 1, "description": "",
        "system_prompt": "You are a Q&A agent.", "generation": {}, "rag": {"top_k": 4},
        "tools": [],  # no "guardrails" and no "analysis" key at all, like msock-advisor
        "model": {"id": "m", "name": "M", "file": "m.gguf", "context_length": 8192},
    }
    client = _install(monkeypatch, model, manifest)
    up = client.post("/api/attachments", files={"file": ("transcript.txt", TRANSCRIPT.encode())},
                     data={"agent_id": "no-analysis-test"})
    meta = up.json()
    assert meta["small"] is False

    r = client.post("/api/chat", json={"agent_id": "no-analysis-test", "attachment_id": meta["id"],
                                       "messages": [{"role": "user", "content": "what topics were discussed"}]})
    events = _events(r)
    assert not any("Reading section" in e.get("text", "") for e in events if e["type"] == "status")
    assert not any(c[0]["content"] == rt.DIGEST_SYSTEM_PROMPT for c in model.calls)


def test_analysis_mode_on_still_whole_injects_a_small_file(monkeypatch, tmp_path):
    """A small attachment already gets full-inject correctly; analysis mode
    must not turn that into an unnecessary digest pass."""
    monkeypatch.setattr(rt_tools, "DEVICE_DATA", tmp_path / "device_data.json")
    model = ScriptedModel(final_text="Handled.")
    manifest = {
        "id": "small-analysis-test", "name": "Small Analysis", "version": 1, "description": "",
        "system_prompt": "You are a meeting-minutes agent.", "generation": {}, "rag": {"top_k": 4},
        "analysis": {"enabled": True}, "tools": [],
        "model": {"id": "m", "name": "M", "file": "m.gguf", "context_length": 8192},
    }
    client = _install(monkeypatch, model, manifest)
    short = "Short note: decided to proceed on Friday."
    up = client.post("/api/attachments", files={"file": ("short.txt", short.encode())},
                     data={"agent_id": "small-analysis-test"})
    meta = up.json()
    assert meta["small"] is True

    r = client.post("/api/chat", json={"agent_id": "small-analysis-test", "attachment_id": meta["id"],
                                       "messages": [{"role": "user", "content": "analyse this"}]})
    events = _events(r)
    assert not any("Reading section" in e.get("text", "") for e in events if e["type"] == "status")
    assert not any(c[0]["content"] == rt.DIGEST_SYSTEM_PROMPT for c in model.calls)
