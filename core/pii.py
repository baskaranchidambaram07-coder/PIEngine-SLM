"""Deterministic detector for PII, passwords and secret/API keys.

This is the first and load-bearing half of the PII Guard (runtime/guard.py).
It is regex + checksum based on purpose: a 1-2B on-device model is neither
fast nor reliable enough to be the *only* thing standing between a pasted
AWS key and the rest of the pipeline, and a regex verdict is explainable to
the user ("AWS access key ending in ...PLE") and reproducible in a test.

What counts, and why:
  * secrets  — passwords, API keys with vendor prefixes, bearer tokens, JWTs,
               private-key blocks, labelled generic secrets. Always blocking.
  * identity — government IDs (US SSN, Indian Aadhaar with Verhoeff check,
               Indian PAN, passport when labelled), payment cards (Luhn),
               IBAN (mod-97), date of birth when labelled.
  * contact  — email addresses, phone numbers (10-13 digits).

Personal NAMES are deliberately not detected here: an enterprise document
(meeting minutes, a risk register) is made of colleagues' names, and blocking
on them would refuse every legitimate question in the flagship scenario.
Names and postal addresses are the job of the model-based judge in
runtime/guard.py, and whether they block is a policy switch there.

Masking rule (the product requirement): show the FIRST THREE and LAST TWO
characters of a value longer than five characters and mask everything else,
so the user can recognise which value tripped the guard without the guard
itself re-disclosing it. Values of five characters or fewer show only their
last character.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

MASK_CHAR = "*"
VISIBLE_HEAD = 3
VISIBLE_TAIL = 2
MAX_FINDINGS = 200          # a 5 MB dump of a log file must not build a 50k-item list


@dataclass
class Finding:
    kind: str               # secret | identity | contact
    label: str              # human label: "AWS access key", "Email address", ...
    value: str              # the matched text, verbatim
    start: int
    end: int
    source: str = ""        # "query" | "attachment" | "model" — filled by the caller
    detector: str = "regex"  # regex | judge

    @property
    def masked(self) -> str:
        return mask(self.value)

    def to_public(self) -> dict:
        """What may leave the process: the label and the masked value. Never the value."""
        d = asdict(self)
        d.pop("value")
        d.pop("start")
        d.pop("end")
        d["masked"] = self.masked
        return d


def mask(value: str, head: int = VISIBLE_HEAD, tail: int = VISIBLE_TAIL) -> str:
    """'priya.sharma@example.com' -> 'pri*******************om'.

    Longer than five characters: the first three and last two stay visible,
    enough to recognise WHICH email or key tripped the guard without the guard
    re-disclosing it. Five or fewer: only the last character. Whitespace
    inside the value is masked too, so a spaced card number does not leak
    its grouping.
    """
    value = value or ""
    n = len(value)
    if n == 0:
        return ""
    if n <= 5:
        return MASK_CHAR * (n - 1) + value[-1:]
    return value[:head] + MASK_CHAR * (n - head - tail) + value[n - tail:]


# ------------------------------------------------------------------ checksums

def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][ord(ch) - 48]]
    return c == 0


def iban_ok(iban: str) -> bool:
    s = iban.upper()
    if not (15 <= len(s) <= 34):
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    return int(num) % 97 == 1


# ------------------------------------------------------------------ patterns
# Each rule: (kind, label, compiled regex, group index of the sensitive value,
# optional validator(value, full_match) -> bool).

_PLACEHOLDER_VALUES = {
    "true", "false", "null", "none", "nil", "undefined", "xxx", "xxxx", "xxxxx",
    "redacted", "<redacted>", "changeme", "example", "placeholder", "hidden",
    "required", "optional", "string", "value", "text", "password", "secret",
    "your_api_key", "your-api-key", "<your_api_key>", "<key>", "<token>",
    "*****", "********", "n/a", "na", "tbd", "todo",
}


def _looks_like_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'`").lower()
    if not v or v in _PLACEHOLDER_VALUES:
        return True
    if v.startswith(("<", "${", "{{", "[")) or v.endswith((">", "}", "]")):
        return True
    if set(v) <= set("*x#•.-_"):
        return True
    return False


def _password_value_ok(value: str, _m: re.Match) -> bool:
    if _looks_like_placeholder(value):
        return False
    v = value.strip().strip("\"'`,;.:!?)(")   # "password: required." is prose
    if len(v) < 3:
        return False
    # "password is required" / "password must ..." are prose, not a value.
    if v.isalpha() and v.islower() and len(v) < 12:
        return False
    return True


def _secret_value_ok(value: str, _m: re.Match) -> bool:
    if _looks_like_placeholder(value):
        return False
    v = value.strip().strip("\"'`,;")
    if len(v) < 8:
        return False
    has_digit = any(c.isdigit() for c in v)
    mixed = v.lower() != v and v.upper() != v
    return has_digit or mixed or len(v) >= 20


def _card_ok(value: str, _m: re.Match) -> bool:
    digits = re.sub(r"\D", "", value)
    if not (13 <= len(digits) <= 19):
        return False
    if len(set(digits)) == 1:
        return False
    return luhn_ok(digits)


def _aadhaar_ok(value: str, _m: re.Match) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) == 12 and verhoeff_ok(digits)


def _iban_ok(value: str, _m: re.Match) -> bool:
    return iban_ok(re.sub(r"\s", "", value))


def _emirates_ok(value: str, _m: re.Match) -> bool:
    """UAE Emirates ID: 784-YYYY-NNNNNNN-C, 15 digits, Luhn check digit."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 15 or not digits.startswith("784"):
        return False
    year = int(digits[3:7])
    return 1900 <= year <= 2100 and luhn_ok(digits)


def _uk_licence_ok(value: str, _m: re.Match) -> bool:
    """UK driving licence: 5 surname chars, 6 date digits (decade, month with
    +50 for women, day, year digit), 2 initials, 1 digit, 2 check letters."""
    digits = value[5:11]
    month = int(digits[1:3]) % 50
    day = int(digits[3:5])
    return 1 <= month <= 12 and 1 <= day <= 31


def _labelled_id_ok(value: str, _m: re.Match) -> bool:
    v = value.strip()
    return sum(c.isdigit() for c in v) >= 4 and not _looks_like_placeholder(v)


def _phone_ok(value: str, m: re.Match) -> bool:
    digits = re.sub(r"\D", "", value)
    if not (10 <= len(digits) <= 13):
        return False
    if len(set(digits)) <= 2:          # 0000000000, 1111111111
        return False
    # A bare run of digits with no phone punctuation and no leading '+' is
    # more often an order/reference number than a phone number.
    if digits == value and not m.group(0).startswith("+"):
        return len(digits) in (10, 11) and value[0] in "6789"  # Indian/US-style mobiles
    return True


_RULES: list[tuple[str, str, re.Pattern, int, object]] = [
    # ---- secrets: vendor-prefixed keys (high precision, no validator needed)
    ("secret", "Private key block",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,4000}?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), 0, None),
    ("secret", "AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0, None),
    ("secret", "GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"), 0, None),
    ("secret", "Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), 0, None),
    ("secret", "Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0, None),
    ("secret", "Stripe key", re.compile(r"\b[srp]k_(?:live|test)_[0-9a-zA-Z]{10,}\b"), 0, None),
    ("secret", "Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), 0, None),
    ("secret", "OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"), 0, None),
    ("secret", "Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0, None),
    ("secret", "SendGrid key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"), 0, None),
    ("secret", "JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"), 0, None),
    ("secret", "Bearer token", re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=\-]{20,})"), 1, None),
    ("secret", "Storage account key", re.compile(r"AccountKey=([A-Za-z0-9+/=]{40,})"), 1, None),
    ("secret", "AWS secret key",
     re.compile(r"(?i)aws[\w\s]{0,20}secret[\w\s]{0,20}[:=]\s*[\"']?([A-Za-z0-9/+=]{40})\b"), 1, None),
    # ---- secrets: labelled values
    ("secret", "Password",
     re.compile(r"(?i)\b(?:password|passwd|passphrase|passcode|pwd|pass|pin)\b\s*(?:is|was|[:=\-]|is\s*[:=])\s*[\"'`]?([^\s\"'`,;]{3,})"),
     1, _password_value_ok),
    # "pin 4821" / "OTP 493 201" — a PIN or one-time code is digits right after the word.
    ("secret", "PIN / OTP",
     re.compile(r"(?i)\b(?:pin|otp|one[- ]time (?:code|password)|passcode|mpin)\b\s*(?:is|was|[:=\-])?\s*(\d{4,8}|\d{3}[ \-]\d{3})\b"),
     1, None),
    # "pass Tr0ub4dor&3" — no separator, so only a value that LOOKS like a
    # strong password (digit + symbol/mixed case) counts; "Password Reset" does not.
    ("secret", "Password",
     re.compile(r"(?i)\b(?:password|passwd|passphrase|pass|pwd)\b\s+([^\s\"'`,;]{8,})"),
     1, lambda v, m: (not _looks_like_placeholder(v) and any(c.isdigit() for c in v)
                      and (not v.isalnum() or (v.lower() != v and v.upper() != v)))),
    ("secret", "API key / secret",
     re.compile(r"(?i)\b(?:api[_ \-]?key|apikey|secret[_ \-]?key|access[_ \-]?token|auth[_ \-]?token|"
                r"refresh[_ \-]?token|client[_ \-]?secret|private[_ \-]?key|secret|token|credentials?|key)\b"
                r"\s*(?:is|[:=]|is\s*[:=])\s*[\"'`]?([A-Za-z0-9_\-./+=]{8,})"),
     1, _secret_value_ok),
    # ---- identity
    ("identity", "Emirates ID (UAE)", re.compile(r"\b784[ \-]?\d{4}[ \-]?\d{7}[ \-]?\d\b"), 0, _emirates_ok),
    ("identity", "Driving licence (India)",
     re.compile(r"\b[A-Z]{2}[ \-]?\d{2}[ \-]?\d{4}[ \-]?\d{7}\b"), 0, None),
    ("identity", "Driving licence (UK)",
     re.compile(r"\b[A-Z9]{5}\d{6}[A-Z9]{2}\d[A-Z]{2}\b"), 0, _uk_licence_ok),
    ("identity", "Payment card number", re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"), 0, _card_ok),
    ("identity", "US Social Security number",
     re.compile(r"\b(?!000|666|9\d{2})\d{3}[\- ](?!00)\d{2}[\- ](?!0000)\d{4}\b"), 0, None),
    ("identity", "Aadhaar number", re.compile(r"\b[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}\b"), 0, _aadhaar_ok),
    ("identity", "PAN (India)", re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"), 0, None),
    ("identity", "IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"), 0, _iban_ok),
    # Any card or licence introduced by its name: "Emirates ID 784…", "citizen
    # card no. …", "driver's license: D1234567", "voter id ABC1234567".
    ("identity", "ID / licence card number",
     re.compile(r"(?i)\b(?:driving licen[cs]e|driver'?s? licen[cs]e|licen[cs]e (?:no\.?|number|#)|dl (?:no\.?|number)|"
                r"emirates id|national id|identity card|id card|citizen(?:ship)? (?:card|id|number)|"
                r"resident(?:cy)? (?:card|id|permit)|voter id|epic (?:no\.?|number)|"
                r"(?:health|insurance|medicare|social security|ration|pan|tax) card|card (?:no\.?|number|#))"
                r"\s*(?:is|was|[:#\-])?\s*([A-Z0-9][A-Z0-9 \-/]{5,24}[A-Z0-9])\b"),
     1, _labelled_id_ok),
    ("identity", "Passport number",
     re.compile(r"(?i)\bpassport\s*(?:no\.?|number|#)?\s*[:#\-]?\s*([A-Z][0-9]{7,8}|[A-Z]{1,2}[0-9]{6,7})\b"), 1, None),
    ("identity", "Date of birth",
     re.compile(r"(?i)\b(?:dob|date of birth|birth\s*date|born(?: on)?)\s*[:\-]?\s*"
                r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"), 1, None),
    # ---- contact
    ("contact", "Email address", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0, None),
    ("contact", "Phone number",
     re.compile(r"(?<![\w.\-])(?:\+\d{1,3}[ \-.]?)?(?:\(\d{2,4}\)[ \-.]?|\d{2,5}[ \-.])?\d{3,5}[ \-.]?\d{3,5}(?![\w.\-])"),
     0, _phone_ok),
]

# When two findings overlap, the one earlier in this order wins (a JWT should
# not also be reported as a "phone number" because of a digit run inside it).
_PRIORITY = {rule[1]: i for i, rule in enumerate(_RULES)}


def scan(text: str, source: str = "") -> list[Finding]:
    """All PII/secret findings in `text`, overlap-free, in document order."""
    if not text:
        return []
    found: list[Finding] = []
    for kind, label, rx, grp, validator in _RULES:
        for m in rx.finditer(text):
            value = m.group(grp)
            if not value:
                continue
            if validator and not validator(value, m):
                continue
            start = m.start(grp)
            found.append(Finding(kind, label, value, start, start + len(value), source))
            if len(found) > MAX_FINDINGS * 4:
                break
    return dedupe(found)[:MAX_FINDINGS]


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop findings that overlap a higher-priority one; keep document order."""
    ordered = sorted(findings, key=lambda f: (_PRIORITY.get(f.label, 999), f.start))
    kept: list[Finding] = []
    for f in ordered:
        if any(f.start < k.end and k.start < f.end for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def redact(text: str, findings: list[Finding]) -> str:
    """Return `text` with every finding replaced by its masked form."""
    out, pos = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if f.start < pos:
            continue
        out.append(text[pos:f.start])
        out.append(f.masked)
        pos = f.end
    out.append(text[pos:])
    return "".join(out)


def summarize(findings: list[Finding]) -> list[dict]:
    """Group by label for display: [{label, kind, count, examples:[masked,...]}].
    Only masked values leave here."""
    groups: dict[str, dict] = {}
    for f in findings:
        g = groups.setdefault(f.label, {"label": f.label, "kind": f.kind, "count": 0,
                                        "examples": [], "detector": f.detector})
        g["count"] += 1
        if f.masked not in g["examples"] and len(g["examples"]) < 3:
            g["examples"].append(f.masked)
    order = {"secret": 0, "identity": 1, "contact": 2}
    return sorted(groups.values(), key=lambda g: (order.get(g["kind"], 9), -g["count"]))
