"""Attachment pipeline + PII Guard integration tests against the runtime app.

Everything runs offline: the agent model is replaced by a scripted fake, so
these tests exercise validation, extraction (real Tesseract when installed),
indexing, the guard's decisions and the chat wiring — not model quality.
The benchmark for that is prototypes/attach_bench.py.
"""
from __future__ import annotations

import io
import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime import app as rt
from runtime import attachments, guard, vision

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs" / "attachments"

pytestmark = pytest.mark.skipif(not SAMPLES.exists(),
                                reason="run scripts/make_attachment_samples.py first")


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Attachments and the guard policy go to a temp dir; nothing touches the
    real device_storage."""
    att_dir = tmp_path / "attachments"
    att_dir.mkdir()
    monkeypatch.setattr(attachments, "ATTACH_DIR", att_dir)
    monkeypatch.setattr(guard, "POLICY_PATH", tmp_path / "guard_policy.json")
    yield
    shutil.rmtree(att_dir, ignore_errors=True)


class FakeModel:
    """Scripted stand-in for llama-server. Records every prompt it is given."""

    def __init__(self, reply: str = "Fake answer."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def stream_chat(self, messages, generation, url=None):
        self.calls.append(messages)
        for piece in (self.reply[i:i + 7] for i in range(0, len(self.reply), 7)):
            yield {"type": "token", "text": piece}
        yield {"type": "stats", "tokens": 5, "seconds": 0.1, "tok_per_sec": 50.0}


@pytest.fixture
def fake_agent(monkeypatch):
    model = FakeModel()
    manifest = {
        "id": "test-agent", "name": "Test Agent", "version": 1, "description": "",
        "system_prompt": "You are a test agent.", "generation": {}, "rag": {"top_k": 4},
        "guardrails": {"pii": True},
        "tools": [], "model": {"id": "m", "name": "M", "file": "m.gguf", "context_length": 4096},
    }
    model.manifest = manifest
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
    return model


@pytest.fixture
def client():
    return TestClient(rt.app)


def upload(client, path: Path, agent_id: str | None = None, name: str | None = None):
    files = {"file": (name or path.name, path.read_bytes())}
    data = {"agent_id": agent_id} if agent_id else {}
    return client.post("/api/attachments", files=files, data=data)


def chat(client, text: str, att_id: str | None = None) -> list[dict]:
    r = client.post("/api/chat", json={"agent_id": "test-agent", "attachment_id": att_id,
                                       "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200, r.text
    return [json.loads(line[6:]) for line in r.text.split("\n") if line.startswith("data: ")]


# ------------------------------------------------------------------ validation

def test_validate_accepts_only_the_four_types():
    ok = attachments.validate("a.TXT", b"hello")
    assert ok == (".txt", "text")
    with pytest.raises(attachments.AttachmentError, match="not supported"):
        attachments.validate("a.docx", b"PK\x03\x04")
    with pytest.raises(attachments.AttachmentError, match="not supported"):
        attachments.validate("a.png", b"\x89PNG")
    with pytest.raises(attachments.AttachmentError, match="not supported"):
        attachments.validate("noext", b"x")


def test_validate_size_limit_is_5mb():
    attachments.validate("big.txt", b"a" * attachments.MAX_BYTES)
    with pytest.raises(attachments.AttachmentError) as ei:
        attachments.validate("big.txt", b"a" * (attachments.MAX_BYTES + 1))
    assert ei.value.status == 413


def test_validate_checks_magic_bytes_not_just_extension():
    with pytest.raises(attachments.AttachmentError, match="not a JPEG"):
        attachments.validate("x.jpg", b"%PDF-1.4 ...")
    with pytest.raises(attachments.AttachmentError, match="not a PDF"):
        attachments.validate("x.pdf", b"\xff\xd8\xff\xe0")
    with pytest.raises(attachments.AttachmentError, match="binary"):
        attachments.validate("x.txt", b"MZ\x90\x00\x03\x00")
    with pytest.raises(attachments.AttachmentError, match="empty"):
        attachments.validate("x.txt", b"")


def test_upload_rejections_over_http(client):
    r = client.post("/api/attachments", files={"file": ("a.csv", b"a,b")})
    assert r.status_code == 400 and "not supported" in r.json()["detail"]
    r = client.post("/api/attachments", files={"file": ("a.txt", b"x" * (attachments.MAX_BYTES + 1))})
    assert r.status_code == 413
    r = client.post("/api/attachments", files={"file": ("a.jpg", b"not a jpeg")})
    assert r.status_code == 400


# ------------------------------------------------------------------ extraction

def test_txt_and_md_extract_and_are_small(client):
    r = upload(client, SAMPLES / "clean-invoice.txt")
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["kind"] == "text" and m["status"] == "ready" and m["small"] is True
    assert m["chunks"] >= 1 and m["guard"]["blocked"] is False
    assert "INV-2026-0912" in attachments.full_text(m["id"])


def test_text_pdf_extracts_without_ocr(client):
    m = upload(client, SAMPLES / "clean-brief.pdf").json()
    assert m["kind"] == "pdf" and m["ocr_pages"] == 0 and m["pages"] >= 1
    assert "Three pilot sites" in attachments.full_text(m["id"])


@pytest.mark.skipif(not attachments.ocr_available(), reason="tesseract not installed")
def test_scanned_pdf_falls_back_to_ocr(client):
    m = upload(client, SAMPLES / "scanned-policy.pdf").json()
    assert m["ocr_used"] is True and m["ocr_pages"] == 1
    text = attachments.full_text(m["id"])
    assert "30 days" in text and "REMOTE WORK POLICY" in text.upper()


@pytest.mark.skipif(not attachments.ocr_available(), reason="tesseract not installed")
def test_jpeg_is_ocrd(client):
    m = upload(client, SAMPLES / "clean-receipt.jpg").json()
    assert m["kind"] == "image" and m["ocr_used"] is True
    text = attachments.full_text(m["id"])
    assert "42.75" in text and "NANDINI" in text.upper()


def test_large_text_is_indexed_and_retrieved(client):
    body = "\n\n".join(f"## Section {i}\n\n" + f"Topic {i} paragraph. " * 40 for i in range(12))
    body += "\n\n## Budget\n\nThe approved budget for the Kochi pilot is 4.2 million rupees."
    r = client.post("/api/attachments", files={"file": ("big.md", body.encode())})
    m = r.json()
    assert m["small"] is False and m["chunks"] > 3
    hits = attachments.context_for(attachments.load_meta(m["id"]), "what is the budget for the Kochi pilot?")
    assert hits and any("4.2 million" in h["text"] for h in hits)
    assert all(h["kind"] == "attachment" for h in hits)
    # a greeting never retrieves
    assert attachments.context_for(attachments.load_meta(m["id"]), "thanks!") == []


def test_public_view_hides_text_and_values(client):
    m = upload(client, SAMPLES / "pii-secrets.md").json()
    dumped = json.dumps(m)
    assert "AKIAIOSFODNN7EXAMPLE" not in dumped and "Hunter2" not in dumped
    assert "text" not in m and "findings" not in m


# ------------------------------------------------------------------ guard on files

@pytest.mark.parametrize("name,expect_label", [
    ("pii-contacts.txt", "Email address"),
    ("pii-secrets.md", "AWS access key"),
    ("pii-card.pdf", "Payment card number"),
])
def test_pii_files_are_blocked_at_upload(client, name, expect_label):
    m = upload(client, SAMPLES / name).json()
    g = m["guard"]
    assert g["blocked"] is True
    assert expect_label in {s["label"] for s in g["summary"]}
    assert "not ready to proceed further" in g["message"]
    # masked: a long value shows its first three and last two characters only
    for s in g["summary"]:
        for ex in s["examples"]:
            assert len(ex) <= 5 or set(ex[3:-2]) == {"*"}


@pytest.mark.skipif(not attachments.ocr_available(), reason="tesseract not installed")
def test_pii_in_an_image_is_caught_through_ocr(client):
    m = upload(client, SAMPLES / "pii-badge.jpg").json()
    assert m["guard"]["blocked"] is True
    assert {"Password", "Email address"} & {s["label"] for s in m["guard"]["summary"]}


@pytest.mark.parametrize("name", ["clean-invoice.txt", "clean-notes.md", "clean-brief.pdf"])
def test_clean_files_pass(client, name):
    m = upload(client, SAMPLES / name).json()
    assert m["guard"]["blocked"] is False and m["guard"]["summary"] == []


# ------------------------------------------------------------------ guard in chat

def test_query_with_secret_is_refused_before_any_model_call(client, fake_agent):
    events = chat(client, "my password is Hunter2! please remember it")
    kinds = [e["type"] for e in events]
    assert "guard" in kinds and "token" not in kinds
    g = next(e for e in events if e["type"] == "guard")
    assert "not ready to proceed further" in g["text"]
    assert "Hun***2!" in g["text"] and "Hunter2!" not in g["text"]
    assert fake_agent.calls == []


def test_clean_query_reaches_the_model(client, fake_agent):
    events = chat(client, "hello there, what can you do?")
    assert [e["type"] for e in events if e["type"] in ("guard", "token")][0] == "token"
    assert len(fake_agent.calls) == 1


def test_blocked_attachment_refuses_every_question(client, fake_agent):
    att = upload(client, SAMPLES / "pii-secrets.md").json()
    events = chat(client, "what does this file say about staging?", att["id"])
    g = next(e for e in events if e["type"] == "guard")
    assert "pii-secrets.md" in g["text"] and "AKI" + "*" * 15 + "LE" in g["text"]
    assert fake_agent.calls == []


def test_clean_attachment_text_is_given_to_the_model(client, fake_agent):
    att = upload(client, SAMPLES / "clean-invoice.txt").json()
    events = chat(client, "what is the total due?", att["id"])
    srcs = next(e for e in events if e["type"] == "sources")
    assert srcs["items"] and srcs["items"][0]["kind"] == "attachment"
    assert any(e["type"] == "token" for e in events)
    # default policy: with a file in play the model judge reviews the message
    # (and the file's chunks) BEFORE the agent sees anything
    judge_calls = [c for c in fake_agent.calls if c[0]["content"].startswith("You are PII Guard")]
    assert judge_calls, "judge should have run on an attachment turn"
    assert any("what is the total due?" in c[-1]["content"] for c in judge_calls)
    answer_call = fake_agent.calls[-1]
    user_msg = answer_call[-1]["content"]
    assert 'Attached document "clean-invoice.txt"' in user_msg and "35,542" in user_msg
    assert "attached a file" in answer_call[0]["content"]


def test_judge_is_skipped_on_plain_turns_by_default(client, fake_agent):
    chat(client, "what is the total due?")
    assert len(fake_agent.calls) == 1
    assert not fake_agent.calls[0][0]["content"].startswith("You are PII Guard")


def test_judge_can_be_switched_off(client, fake_agent):
    client.put("/api/guard/policy", json={"judge": "off"})
    att = upload(client, SAMPLES / "clean-invoice.txt").json()
    chat(client, "what is the total due?", att["id"])
    assert len(fake_agent.calls) == 1
    assert att["guard"]["judged"] is False


def test_model_output_is_redacted(client, fake_agent):
    fake_agent.reply = "Sure — contact them at priya.sharma@example.com for details."
    events = chat(client, "who should I contact about the invoice?")
    red = next(e for e in events if e["type"] == "redact")
    assert "sharma@example" not in red["text"] and red["text"].endswith("pri" + "*" * 19 + "om for details.")
    assert red["summary"][0]["label"] == "Email address"


def test_unknown_attachment_id_is_a_404(client, fake_agent):
    r = client.post("/api/chat", json={"agent_id": "test-agent", "attachment_id": "0" * 32,
                                       "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


# ------------------------------------------------------------------ per-agent switch
# PII Guard is a guardrail the agent's author turns on in the Studio. Off (the
# default, and the state of every manifest published before the switch
# existed) means the plain flow: no scan of the message, the file or the answer.

def test_guardrail_off_lets_a_secret_through_to_the_model(client, fake_agent):
    fake_agent.manifest["guardrails"] = {"pii": False}
    events = chat(client, "my password is Hunter2! please remember it")
    assert not any(e["type"] == "guard" for e in events)
    assert any(e["type"] == "token" for e in events)
    assert len(fake_agent.calls) == 1 and "Hunter2!" in fake_agent.calls[0][-1]["content"]


def test_guardrail_off_skips_the_upload_scan(client, fake_agent):
    fake_agent.manifest["guardrails"] = {"pii": False}
    att = upload(client, SAMPLES / "pii-secrets.md", agent_id="test-agent").json()
    assert att["guard"]["enabled"] is False and att["guard"]["blocked"] is False
    events = chat(client, "what does this file say about staging?", att["id"])
    assert not any(e["type"] == "guard" for e in events)
    assert "AKIAIOSFODNN7EXAMPLE" in fake_agent.calls[-1][-1]["content"]   # normal flow


def test_guardrail_off_does_not_redact_the_answer(client, fake_agent):
    fake_agent.manifest["guardrails"] = {"pii": False}
    fake_agent.reply = "Contact priya.sharma@example.com."
    events = chat(client, "who do I contact?")
    assert not any(e["type"] == "redact" for e in events)


def test_manifest_without_guardrails_key_means_off(client, fake_agent):
    del fake_agent.manifest["guardrails"]
    events = chat(client, "key AKIAIOSFODNN7EXAMPLE")
    assert not any(e["type"] == "guard" for e in events)
    assert client.get("/api/installed").status_code == 200


def test_guardrail_on_is_reported_and_enforced(client, fake_agent):
    att = upload(client, SAMPLES / "pii-secrets.md", agent_id="test-agent").json()
    assert att["guard"]["enabled"] is True and att["guard"]["blocked"] is True


# ------------------------------------------------------------------ judge

def test_judge_keeps_only_spans_that_exist_verbatim():
    text = "Ship to 14 Rose Lane, Kochi 682001. Ref 4500012345."
    reply = json.dumps({"findings": [
        {"type": "address", "value": "14 Rose Lane, Kochi 682001"},
        {"type": "government_id", "value": "Aadhaar"},           # the word, not a number
        {"type": "financial", "value": "+91 99999 11111"},        # invented
        {"type": "name", "value": "Rose"},                        # names off by policy
        {"type": "contact", "value": "Ref 4500012345"},           # regex's job, not the judge's
    ]})
    found, note = guard.judge(text, "attachment:x", lambda m, g: reply)
    assert [f.label for f in found] == ["Home address"]
    assert found[0].detector == "judge" and "3 dropped" in note


def test_judge_tolerates_garbage_replies():
    assert guard.judge("some text here", "query", lambda m, g: "I cannot help")[0] == []
    assert guard.judge("some text here", "query", lambda m, g: "{bad json")[0] == []

    def boom(m, g):
        raise RuntimeError("server down")
    found, note = guard.judge("some text here", "query", boom)
    assert found == [] and note.startswith("judge unavailable")


def test_policy_round_trip_and_validation(client):
    assert client.get("/api/guard/policy").json()["judge"] == "attachments"
    r = client.put("/api/guard/policy", json={"judge": "off", "block_names": True})
    assert r.status_code == 200 and r.json()["judge"] == "off" and r.json()["block_names"] is True
    assert client.put("/api/guard/policy", json={"judge": "sometimes"}).status_code == 400
    assert guard.judge_enabled_for("attachment") is False


def test_judge_policy_modes():
    assert guard.judge_enabled_for("query", {"judge": "attachments"}) is False
    assert guard.judge_enabled_for("attachment", {"judge": "attachments"}) is True
    assert guard.judge_enabled_for("query", {"judge": "always"}) is True
    assert guard.judge_enabled_for("attachment", {"judge": "off"}) is False


# ------------------------------------------------------------------ housekeeping

def test_delete_and_expiry(client):
    m = upload(client, SAMPLES / "clean-notes.md").json()
    assert client.get(f"/api/attachments/{m['id']}").status_code == 200
    assert client.delete(f"/api/attachments/{m['id']}").json()["ok"] is True
    assert client.get(f"/api/attachments/{m['id']}").status_code == 404

    m2 = upload(client, SAMPLES / "clean-notes.md").json()
    meta = attachments.load_meta(m2["id"])
    meta["created"] = time.time() - 48 * 3600
    attachments.save_meta(meta)
    assert attachments.sweep_expired() == 1
    assert not attachments.meta_path(m2["id"]).exists()


def test_capabilities_endpoint(client):
    caps = client.get("/api/capabilities").json()
    assert caps["attachments"]["max_bytes"] == 5 * 1024 * 1024
    assert caps["attachments"]["max_files"] == 1
    assert set(caps["attachments"]["allowed"]) == {".jpg", ".jpeg", ".md", ".pdf", ".txt"}
    assert caps["guard"]["name"] == "PII Guard"


def test_prepare_image_downscales():
    from PIL import Image
    img = Image.new("RGB", (4000, 3000), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg, info = vision.prepare_image(buf.getvalue())
    assert info["original"] == [4000, 3000] and max(info["sent"]) == vision.MAX_IMAGE_SIDE
    assert jpeg.startswith(b"\xff\xd8\xff")
