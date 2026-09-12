"""PII Guard — the agent that decides whether a turn may proceed at all.

It sits in front of every chat turn and every attachment. Its job is narrow
and its answer is binary: does the user's message, or the document they
attached, contain personal data, a password, or a secret/API key? If so, the
turn is refused with a message that names each finding by category and shows
only the first three and last two characters of the value — enough to recognise it, not
enough to re-disclose it — and the sentence the product asked for:
"I am not ready to proceed further."

Two detectors, in order:

1. `core.pii` — deterministic regexes with checksums (Luhn, Verhoeff,
   IBAN mod-97). Runs on everything, always, in milliseconds. This is the
   detector that is *relied on*: a pasted AWS key or card number is caught
   even if the model is asleep.

2. The model judge — the agent's own on-device SLM, prompted as a strict
   data-protection reviewer, returning JSON. It exists for what regexes cannot
   see: a home address in prose, an ID number in a format we have no rule
   for, a credential described rather than labelled. Every span it reports
   must occur VERBATIM in the text or it is discarded — a 1.7B model will
   happily invent a plausible phone number, and a guard that hallucinates
   findings is a guard that blocks legitimate work. It is also slow (~10-40 s
   per call on this box), so policy decides where it runs: on the query and
   the retrieved chunks of a turn that carries an attachment (default), on
   every turn, or never.

Personal names are a policy switch (`block_names`, default off): a meeting
minute is made of names, and blocking on them would refuse the flagship
scenario outright. The switch exists because a stricter deployment may want
it on.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core import pii

GUARD_NAME = "PII Guard"

APP_DIR = Path(__file__).resolve().parent
POLICY_PATH = APP_DIR / "device_storage" / "guard_policy.json"

DEFAULT_POLICY = {
    # where the model judge runs: "attachments" | "always" | "off"
    "judge": "attachments",
    # chunks judged at upload time (the rest is regex-only until retrieved)
    "judge_upload_chunks": 2,
    # retrieved chunks judged at question time (cached per attachment)
    "judge_query_chunks": 4,
    "judge_max_tokens": 256,
    "judge_max_chars": 3000,        # per judge call; a chunk is ~1200
    "block_names": False,
    "block_kinds": ["secret", "identity", "contact"],
    # also scan the model's own answer and mask anything that slipped through
    "redact_output": True,
}

JUDGE_SYSTEM = """You are PII Guard, a strict data-protection reviewer for an enterprise assistant.
Read the TEXT and list every item that is personal data about a private individual, or a credential:
- a private individual's home address
- government identity numbers (SSN, Aadhaar, PAN, passport, driving licence, tax id)
- personal bank account, card or IBAN numbers
- passwords, PINs, API keys, access tokens, secret keys, private keys
- health, salary or biometric details tied to a person{names_clause}

NOT personal data — never report these: a shop, cafe, office, venue or company address; company names;
invoice, order, table, receipt or reference numbers; prices and totals; ordinary dates and times;
product names; job titles.

Reply with ONLY this JSON, nothing else:
{{"findings":[{{"type":"address|government_id|financial|credential|health|name","value":"<the exact text, copied verbatim from TEXT>"}}]}}
If there is nothing, reply {{"findings":[]}}. Never invent a value. Never explain. /no_think"""

NAMES_CLAUSE = "\n- full names of private individuals"

# Emails, phone numbers and labelled dates of birth are the regex stage's job:
# it is exact and explainable there, and a small model asked for them reports
# every date and every number it sees. The judge covers only what regex cannot.
_JUDGE_KIND = {
    "credential": "secret",
    "address": "identity", "government_id": "identity", "financial": "identity",
    "health": "identity", "name": "identity",
}
_JUDGE_LABEL = {
    "credential": "Credential (model judge)", "address": "Home address",
    "government_id": "Government ID number", "financial": "Financial account number",
    "health": "Health / salary detail", "name": "Personal name",
}

# chat_fn(messages, generation) -> full text of the model's reply
ChatFn = Callable[[list[dict], dict], str]


@dataclass
class Verdict:
    blocked: bool
    findings: list[pii.Finding] = field(default_factory=list)
    judged: bool = False
    judge_note: str = ""
    ms: int = 0
    source: str = ""

    @property
    def summary(self) -> list[dict]:
        return pii.summarize(self.findings)

    def to_public(self) -> dict:
        return {"blocked": self.blocked, "summary": self.summary, "judged": self.judged,
                "judge_note": self.judge_note, "ms": self.ms, "source": self.source}


# ------------------------------------------------------------------ policy

def policy() -> dict:
    pol = dict(DEFAULT_POLICY)
    if POLICY_PATH.exists():
        try:
            pol.update({k: v for k, v in json.loads(POLICY_PATH.read_text(encoding="utf-8")).items()
                        if k in DEFAULT_POLICY})
        except (OSError, ValueError):
            pass
    return pol


def save_policy(updates: dict) -> dict:
    pol = policy()
    for k, v in updates.items():
        if k in DEFAULT_POLICY:
            pol[k] = v
    if pol["judge"] not in ("attachments", "always", "off"):
        raise ValueError("judge must be one of: attachments, always, off")
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(json.dumps(pol, indent=2), encoding="utf-8")
    return pol


def judge_enabled_for(where: str, pol: dict | None = None) -> bool:
    """where: 'query' (no attachment in play) | 'attachment'"""
    pol = pol or policy()
    mode = pol.get("judge", "attachments")
    if mode == "off":
        return False
    if mode == "always":
        return True
    return where == "attachment"


# ------------------------------------------------------------------ detectors

def scan(text: str, source: str, pol: dict | None = None) -> list[pii.Finding]:
    pol = pol or policy()
    kinds = set(pol.get("block_kinds", DEFAULT_POLICY["block_kinds"]))
    return [f for f in pii.scan(text, source) if f.kind in kinds]


_JSON_RX = re.compile(r"\{[\s\S]*\}")


def judge(text: str, source: str, chat_fn: ChatFn, pol: dict | None = None,
          known: list[pii.Finding] | None = None) -> tuple[list[pii.Finding], str]:
    """Ask the model. Returns (validated findings, note). Never raises."""
    pol = pol or policy()
    text = text[: int(pol.get("judge_max_chars", 3000))]
    if not text.strip():
        return [], "empty"
    system = JUDGE_SYSTEM.format(names_clause=NAMES_CLAUSE if pol.get("block_names") else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"TEXT:\n\"\"\"\n{text}\n\"\"\""}]
    try:
        reply = chat_fn(messages, {"temperature": 0.0, "top_p": 1.0,
                                   "max_tokens": int(pol.get("judge_max_tokens", 256))})
    except Exception as exc:  # noqa: BLE001 — the judge is advisory; regex already ran
        return [], f"judge unavailable: {str(exc)[:120]}"
    m = _JSON_RX.search(reply or "")
    if not m:
        return [], "judge returned no JSON"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate a trailing comma / truncated list
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return [], "judge JSON unparsable"
    items = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], "judge JSON had no findings list"

    kinds = set(pol.get("block_kinds", DEFAULT_POLICY["block_kinds"]))
    known = known or []
    out: list[pii.Finding] = []
    dropped = 0
    for it in items[:50]:
        if not isinstance(it, dict):
            continue
        typ = str(it.get("type", "")).strip().lower()
        val = str(it.get("value", "")).strip()
        if typ == "name" and not pol.get("block_names"):
            continue
        kind = _JUDGE_KIND.get(typ)
        if not kind or kind not in kinds or len(val) < 3:
            dropped += 1
            continue
        if not _plausible(typ, val):
            dropped += 1          # "Aadhaar" the word is not an Aadhaar number
            continue
        pos = _locate(text, val)
        if pos < 0:
            dropped += 1          # the model invented it — the whole point of validation
            continue
        f = pii.Finding(kind, _JUDGE_LABEL[typ], text[pos:pos + len(val)], pos, pos + len(val),
                        source, detector="judge")
        if any(f.start < k.end and k.start < f.end for k in known):
            continue              # regex already has it
        out.append(f)
    note = f"judge: {len(out)} kept, {dropped} dropped (not in text / off-policy)"
    return pii.dedupe(out), note


def _plausible(typ: str, value: str) -> bool:
    """Type-specific sanity check on a judge span. The model sometimes names
    the *kind* of thing ("Aadhaar", "credit card") instead of the value; an
    identifier, account or address must contain a digit, and a credential
    must look like one rather than being a lone dictionary word."""
    has_digit = any(c.isdigit() for c in value)
    if typ in ("government_id", "financial", "address"):
        return has_digit
    if typ == "credential":
        v = value.strip("\"'`")
        return len(v) >= 8 and (has_digit or v.lower() != v or not v.isalpha())
    return True


def _locate(text: str, value: str) -> int:
    pos = text.find(value)
    if pos >= 0:
        return pos
    # tolerate whitespace differences ("+91 98765 43210" vs "+91 9876543210")
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 3:
        return -1
    rx = re.compile(r"\s*".join(re.escape(ch) for ch in compact))
    m = rx.search(text)
    return m.start() if m else -1


# ------------------------------------------------------------------ the agent

def inspect(text: str, source: str, use_judge: bool = False,
            chat_fn: ChatFn | None = None, pol: dict | None = None) -> Verdict:
    """Run the guard over `text`. `source` is 'query' or 'attachment:<name>'."""
    pol = pol or policy()
    t0 = time.time()
    findings = scan(text, source, pol)
    judged = False
    note = ""
    if use_judge and chat_fn is not None:
        extra, note = judge(text, source, chat_fn, pol, known=findings)
        judged = not note.startswith("judge unavailable") and "no JSON" not in note \
            and "unparsable" not in note
        findings = pii.dedupe(findings + extra)
    return Verdict(blocked=bool(findings), findings=findings, judged=judged,
                   judge_note=note, ms=round((time.time() - t0) * 1000), source=source)


def inspect_chunks(chunks: list[str], source: str, chat_fn: ChatFn | None,
                   pol: dict | None = None, judge_limit: int | None = None) -> Verdict:
    """Regex over every chunk; judge over the first `judge_limit` of them."""
    pol = pol or policy()
    t0 = time.time()
    findings: list[pii.Finding] = []
    for c in chunks:
        findings += scan(c, source, pol)
    judged = False
    notes = []
    if chat_fn is not None and judge_limit:
        for c in chunks[:judge_limit]:
            extra, note = judge(c, source, chat_fn, pol, known=[])
            notes.append(note)
            judged = judged or note.startswith("judge:")
            findings += extra
            if extra:
                break        # one confirmed finding is enough to refuse
    # No cross-chunk dedupe: offsets are per chunk, so overlap tests would be
    # meaningless. Display grouping happens in pii.summarize anyway.
    return Verdict(blocked=bool(findings), findings=findings,
                   judged=judged, judge_note="; ".join(notes)[:300],
                   ms=round((time.time() - t0) * 1000), source=source)


def refusal_message(where: str, summary: list[dict]) -> str:
    """The user-facing refusal. `where` is e.g. 'your message' or
    'the attached document "invoice.pdf"'. Only masked values appear."""
    lines = [f"🛡 {GUARD_NAME}: I found sensitive data in {where}, so I am not ready to proceed further."]
    for g in summary[:8]:
        ex = ", ".join(g["examples"])
        more = f" (+{g['count'] - len(g['examples'])} more)" if g["count"] > len(g["examples"]) else ""
        what = "a secret" if g["kind"] == "secret" else "PII"
        lines.append(f"• {g['label']} — {what}: {ex}{more}")
    if len(summary) > 8:
        lines.append(f"• …and {len(summary) - 8} more categories")
    lines.append("Remove or redact this information and try again.")
    return "\n".join(lines)


def redact_text(text: str, pol: dict | None = None) -> tuple[str, list[dict]]:
    """Mask any PII/secret in a model answer before it is shown. Returns
    (masked text, summary). Regex-only: this runs on every answer."""
    findings = scan(text, "model", pol)
    if not findings:
        return text, []
    return pii.redact(text, findings), pii.summarize(findings)
