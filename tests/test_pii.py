"""Unit tests for the deterministic PII / secret detector (core/pii.py)."""
from __future__ import annotations

import pytest

from core import pii


def labels(text: str) -> set[str]:
    return {f.label for f in pii.scan(text)}


# ------------------------------------------------------------------ masking

def test_mask_shows_first_three_and_last_two():
    assert pii.mask("AKIAIOSFODNN7EXAMPLE") == "AKI" + "*" * 15 + "LE"
    assert pii.mask("Hunter2!") == "Hun***2!"
    assert pii.mask("priya.sharma@example.com") == "pri" + "*" * 19 + "om"
    assert pii.mask("abcdef") == "abc*ef"          # six chars: one masked


def test_mask_short_values_keep_one_character():
    assert pii.mask("abc") == "**c"
    assert pii.mask("abcde") == "****e"
    assert pii.mask("") == ""


def test_mask_never_reveals_the_middle():
    for v in ("4111 1111 1111 1111", "+91 98765 43210", "Wel(ome2Acme!"):
        m = pii.mask(v)
        assert len(m) == len(v) and m[3:-2] == "*" * (len(v) - 5)


def test_public_view_never_carries_the_value():
    f = pii.scan("key AKIAIOSFODNN7EXAMPLE")[0]
    pub = f.to_public()
    assert "value" not in pub and pub["masked"] == "AKI" + "*" * 15 + "LE"
    assert "AKIAIOSFODNN7EXAMPLE" not in str(pub)


# ------------------------------------------------------------------ secrets

@pytest.mark.parametrize("text,label", [
    ("AKIAIOSFODNN7EXAMPLE", "AWS access key"),
    ("ghp_" + "a" * 36, "GitHub token"),
    ("xoxb-1234567890-abcdefghij", "Slack token"),
    ("AIza" + "A" * 35, "Google API key"),
    ("sk_live_" + "z" * 24, "Stripe key"),
    ("sk-ant-" + "q" * 30, "Anthropic API key"),
    ("sk-proj-" + "q" * 30, "OpenAI API key"),
    ("hf_" + "b" * 34, "Hugging Face token"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "JWT"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789", "Bearer token"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "Private key block"),
    ("password: Hunter2!", "Password"),
    ("Password = Wel(ome2Acme!", "Password"),
    ("the pin is 4821", "PIN / OTP"),
    ("pin 4821", "PIN / OTP"),
    ("your OTP is 493 201", "PIN / OTP"),
    ("api_key = 3f9a8b7c6d5e4f3a2b1c", "API key / secret"),
    ("the staging key is 9f8a7d6c5b4a39281706f5e4d3c2b1a0", "API key / secret"),
    ("root login is admin with pass Tr0ub4dor&3", "Password"),
    ("client_secret: Q2xpZW50U2VjcmV0MTIz", "API key / secret"),
    ("AccountKey=" + "A" * 44 + "==", "Storage account key"),
])
def test_secret_patterns(text, label):
    assert label in labels(text), text


def test_secret_prose_is_not_a_secret():
    clean = ("The password policy requires 12 characters. Password must be reset every 90 days. "
             "Set the api_key = <your_api_key> in config. token: null. secret = ******** "
             "password: required. The key is to focus on outcomes; pass is required to enter. "
             "Primary key: customer_id")
    assert labels(clean) == set()


# ------------------------------------------------------------------ identity

def _valid_aadhaar() -> str:
    base = "23456789012"
    for d in "0123456789":
        if pii.verhoeff_ok(base + d):
            return base + d
    raise AssertionError("no check digit found")


def test_card_requires_luhn():
    assert "Payment card number" in labels("card 4111 1111 1111 1111")
    assert "Payment card number" in labels("4242424242424242")
    assert "Payment card number" not in labels("card 4111 1111 1111 1112")   # bad checksum


def test_aadhaar_requires_verhoeff():
    good = _valid_aadhaar()
    assert "Aadhaar number" in labels(f"aadhaar {good[:4]} {good[4:8]} {good[8:]}")
    assert "Aadhaar number" not in labels("aadhaar 1234 5678 9012")           # starts with 1


def test_iban_requires_mod97():
    assert "IBAN" in labels("IBAN GB82 WEST 1234 5698 7654 32")
    assert "IBAN" not in labels("IBAN GB82 WEST 1234 5698 7654 33")


def _valid_emirates_id() -> str:
    base = "78419871234567"
    for d in "0123456789":
        if pii.luhn_ok(base + d):
            return base + d
    raise AssertionError("no check digit found")


def test_emirates_id_requires_luhn_and_784_prefix():
    e = _valid_emirates_id()
    assert "Emirates ID (UAE)" in labels(f"EID {e[:3]}-{e[3:7]}-{e[7:14]}-{e[14]}")
    assert "Emirates ID (UAE)" in labels(e)
    bad = e[:-1] + str((int(e[-1]) + 1) % 10)
    assert "Emirates ID (UAE)" not in labels(bad)


def test_driving_licences():
    assert labels("DL KA01 20150001234") == {"Driving licence (India)"}
    assert labels("MH12-2011-0012345") == {"Driving licence (India)"}
    assert labels("licence MORGA657054SM9IJ") == {"Driving licence (UK)"}
    assert labels("MORGA697054SM9IJ") == set()            # month 69 is impossible


def test_labelled_id_cards():
    assert labels("driver's license: D1234567") == {"ID / licence card number"}
    assert labels("citizen card no. AE-2019-778812") == {"ID / licence card number"}
    assert labels("voter id ABC1234567") == {"ID / licence card number"}
    assert labels("card number 4111 1111 1111 1111") == {"Payment card number"}   # the specific rule wins
    assert labels("Employee ID E-10442; the meeting card number was small") == set()


def test_other_identity_patterns():
    assert "US Social Security number" in labels("SSN 123-45-6789")
    assert "PAN (India)" in labels("PAN ABCPE1234F")
    assert "Passport number" in labels("passport no: N1234567")
    assert "Date of birth" in labels("DOB: 14/02/1990")
    assert "Date of birth" in labels("Date of birth 1990-02-14")


# ------------------------------------------------------------------ contact

def test_contact_patterns():
    assert "Email address" in labels("mail priya.sharma@example.com now")
    assert "Phone number" in labels("call +91 98765 43210")
    assert "Phone number" in labels("tel (415) 555-0134")
    assert "Phone number" in labels("mobile 9876543210")


# ------------------------------------------------------------------ negatives
# Business documents are full of numbers; none of these may trip the guard.

@pytest.mark.parametrize("text", [
    "INVOICE INV-2026-0912 Total due INR 35,542 ($1,999.00) reference 4500012345",
    "Meeting on 12 June 2026 with 8 attendees, budget 250000 INR, version 2.3.1, port 8302",
    "Windows 10.0.26100, pi 3.14159265358979, build b10883, 4 vCPU 16 GB",
    "Phase 2 rollout moves from 22 September to 6 October 2026",
    "GST 18% INR 5,422; subtotal 30,120; PO-77120; ticket JIRA-4521",
    "Qwen3-1.7B-Q4_K_M.gguf is 1107409472 bytes",
])
def test_business_text_is_clean(text):
    assert labels(text) == set(), text


# ------------------------------------------------------------------ helpers

def test_dedupe_prefers_higher_priority_overlap():
    # the JWT's digit runs must not also be reported as phone numbers
    fs = pii.scan("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")
    assert [f.label for f in fs] == ["JWT"]


def test_redact_and_summarize():
    text = "mail a@example.com or b@example.com, key AKIAIOSFODNN7EXAMPLE"
    fs = pii.scan(text)
    red = pii.redact(text, fs)
    assert "example.com" not in red and red.endswith("AKI" + "*" * 15 + "LE")
    summ = pii.summarize(fs)
    assert summ[0]["label"] == "AWS access key"          # secrets first
    email = next(s for s in summ if s["label"] == "Email address")
    assert email["count"] == 2 and all(e[3:-2] == "*" * (len(e) - 5) for e in email["examples"])


def test_findings_are_capped():
    text = "\n".join(f"user{i}@example.com" for i in range(1000))
    assert len(pii.scan(text)) == pii.MAX_FINDINGS
