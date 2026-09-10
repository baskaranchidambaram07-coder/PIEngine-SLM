"""Deterministic assertions over a model's answer.

The same library grades probes at eval time and screens synthesised training
examples before they are allowed into the dataset. That symmetry matters: an
example the grader would fail is an example that teaches the model to fail, so
synth.py runs every candidate answer through these checks and drops the ones
that do not pass.

Everything here is lexical and deterministic — no LLM judge. On a 4-vCPU box a
judge would cost more than the tune, and a 1.7B judge grading a 1.7B student
mostly measures their shared blind spots. The price is that checks are narrow;
each one only claims the specific thing it can actually verify.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ observation


@dataclass
class Observation:
    """What the model did, and what it had to work with."""
    text: str                                   # visible answer, tool tags stripped
    tool_calls: list[dict] = field(default_factory=list)   # [{"name":..,"arguments":{..}}]
    context: str = ""                           # retrieved passages shown to the model
    doc_names: list[str] = field(default_factory=list)     # sources in that context
    tool_defs: list[dict] = field(default_factory=list)    # agent's declared tools
    seconds: float = 0.0
    tokens: int = 0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


# ------------------------------------------------------------------ vocabulary

# A refusal is a *sentence shape*, not a keyword. These are the forms Qwen3 and
# the seeded behaviour actually produce; matched on a lowercased answer.
_REFUSAL = re.compile(
    r"\b("
    r"i (?:don'?t|do not) have (?:that|this|it|any)"
    r"|not (?:in|available in|covered (?:in|by)|present in|mentioned in|found in) the (?:knowledge base|context|provided|retrieved)"
    r"|(?:knowledge base|context) (?:does not|doesn'?t) (?:contain|mention|include|cover|have)"
    r"|no (?:information|mention|record|reference|details?) (?:about|on|of|regarding)"
    r"|(?:i|there) (?:can'?t|cannot|is no way to) (?:find|answer|confirm)"
    r"|outside (?:this agent'?s |the |my )?scope"
    r"|(?:isn'?t|is not) something i can answer"
    r")\b",
    re.IGNORECASE,
)

# Words that start a sentence or a bullet and would otherwise look like a proper
# noun to the grounding check.
_COMMON_CAPS = frozenset("""
the this that these those a an and or but if then so it its is are was were be
i you we they he she there here what when where who why how from for with about
of in on at to by as per no not none yes ok source sources note notes context
knowledge base document summary answer action owner due status risk decision
next steps meeting project team update
""".split())

_NUM = re.compile(r"\d[\d,.:/%-]*\d|\d")
_CAP = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]{2,}\b")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ------------------------------------------------------------------ checks

def _contains_any(obs: Observation, needles) -> CheckResult:
    hay = normalise(obs.text)
    hits = [n for n in needles if normalise(n) and normalise(n) in hay]
    return CheckResult("contains_any", bool(hits),
                       f"found {hits}" if hits else f"none of {list(needles)}")


def _contains_all(obs: Observation, needles) -> CheckResult:
    hay = normalise(obs.text)
    missing = [n for n in needles if normalise(n) not in hay]
    return CheckResult("contains_all", not missing, f"missing {missing}" if missing else "")


def _not_contains_any(obs: Observation, needles) -> CheckResult:
    hay = normalise(obs.text)
    hits = [n for n in needles if normalise(n) and normalise(n) in hay]
    return CheckResult("not_contains_any", not hits, f"leaked {hits}" if hits else "")


def _cites(obs: Observation, want: bool) -> CheckResult:
    """The answer names one of the documents it was actually shown.

    Matched on the stem (filename without extension) because models write
    'status-report' as often as 'status-report.md'.
    """
    hay = normalise(obs.text)
    stems = {re.sub(r"\.(md|txt|pdf|docx?)$", "", d, flags=re.I) for d in obs.doc_names}
    hit = next((s for s in stems if normalise(s) and normalise(s) in hay), None)
    ok = bool(hit) if want else not hit
    detail = f"cited {hit!r}" if hit else f"no source named (had {sorted(stems)})"
    return CheckResult("cites", ok, detail)


def _refuses(obs: Observation, want: bool) -> CheckResult:
    m = _REFUSAL.search(obs.text)
    ok = bool(m) if want else not m
    return CheckResult("refuses", ok, f"matched {m.group(0)!r}" if m else "no refusal phrase")


def _tool_call(obs: Observation, name: str) -> CheckResult:
    names = [c.get("name") for c in obs.tool_calls]
    ok = name in names
    if ok and len(obs.tool_calls) > 1:
        return CheckResult("tool_call", False, f"called {names} — expected exactly {name!r}")
    return CheckResult("tool_call", ok, f"called {names or 'nothing'}")


def _no_tool_call(obs: Observation, want: bool) -> CheckResult:
    called = [c.get("name") for c in obs.tool_calls]
    ok = (not called) if want else bool(called)
    return CheckResult("no_tool_call", ok, f"called {called}" if called else "")


def _tool_args_valid(obs: Observation, want: bool) -> CheckResult:
    """Arguments parse as JSON and satisfy the declared required properties."""
    if not obs.tool_calls:
        return CheckResult("tool_args_valid", not want, "no tool call to validate")
    defs = {t["name"]: t for t in obs.tool_defs}
    problems = []
    for call in obs.tool_calls:
        args = call.get("arguments")
        if not isinstance(args, dict):
            problems.append(f"{call.get('name')}: arguments is {type(args).__name__}, not an object")
            continue
        schema = (defs.get(call.get("name"), {}) or {}).get("parameters") or {}
        for req in schema.get("required", []):
            if req not in args or args[req] in (None, ""):
                problems.append(f"{call.get('name')}: missing required {req!r}")
        props = schema.get("properties") or {}
        for k in args:
            if props and k not in props:
                problems.append(f"{call.get('name')}: unknown argument {k!r}")
    ok = (not problems) if want else bool(problems)
    return CheckResult("tool_args_valid", ok, "; ".join(problems))


def _grounded(obs: Observation, want: bool) -> CheckResult:
    """Every number and proper noun in the answer appears in the context.

    Deliberately one-sided: it catches invented figures, dates and names — the
    failures that make an answer dangerous — and says nothing about whether the
    prose is faithful. If there is no context, an answer with any hard fact in
    it is by definition ungrounded.
    """
    hay = normalise(obs.context)
    body = _strip_source_markers(obs.text)
    facts = set(_NUM.findall(body))
    facts |= {c for c in _CAP.findall(body) if c.lower() not in _COMMON_CAPS}
    invented = sorted({f for f in facts if normalise(f) and normalise(f) not in hay})
    ok = (not invented) if want else bool(invented)
    return CheckResult("grounded", ok, f"not in context: {invented[:6]}" if invented else "")


def _strip_source_markers(text: str) -> str:
    """Drop bracketed citations so the doc name is not itself graded as a fact."""
    return re.sub(r"\[[^\]]*\]|\([^)]*\.(?:md|txt|pdf)[^)]*\)", " ", text)


def _max_words(obs: Observation, n: int) -> CheckResult:
    c = len(obs.text.split())
    return CheckResult("max_words", c <= int(n), f"{c} words (limit {n})")


def _min_words(obs: Observation, n: int) -> CheckResult:
    c = len(obs.text.split())
    return CheckResult("min_words", c >= int(n), f"{c} words (floor {n})")


def _max_bullets(obs: Observation, n: int) -> CheckResult:
    c = len(_BULLET.findall(obs.text))
    return CheckResult("max_bullets", c <= int(n), f"{c} bullets (limit {n})")


def _bullets(obs: Observation, want: bool) -> CheckResult:
    c = len(_BULLET.findall(obs.text))
    ok = (c > 0) if want else (c == 0)
    return CheckResult("bullets", ok, f"{c} bullets")


def _regex(obs: Observation, pattern: str) -> CheckResult:
    m = re.search(pattern, obs.text, re.IGNORECASE | re.MULTILINE)
    return CheckResult("regex", bool(m), f"/{pattern}/ {'matched' if m else 'did not match'}")


def _json_valid(obs: Observation, want: bool) -> CheckResult:
    blob = obs.text.strip()
    blob = re.sub(r"^```(?:json)?|```$", "", blob, flags=re.MULTILINE).strip()
    try:
        json.loads(blob)
        ok = want
        detail = ""
    except json.JSONDecodeError as exc:
        ok = not want
        detail = str(exc)[:80]
    return CheckResult("json_valid", ok, detail)


def _max_seconds(obs: Observation, n: float) -> CheckResult:
    return CheckResult("max_seconds", obs.seconds <= float(n), f"{obs.seconds:.1f}s (limit {n}s)")


CHECKS = {
    "contains_any": _contains_any,
    "contains_all": _contains_all,
    "not_contains_any": _not_contains_any,
    "cites": _cites,
    "refuses": _refuses,
    "tool_call": _tool_call,
    "no_tool_call": _no_tool_call,
    "tool_args_valid": _tool_args_valid,
    "grounded": _grounded,
    "max_words": _max_words,
    "min_words": _min_words,
    "max_bullets": _max_bullets,
    "bullets": _bullets,
    "regex": _regex,
    "json_valid": _json_valid,
    "max_seconds": _max_seconds,
}


def run_checks(expect: dict, obs: Observation) -> list[CheckResult]:
    out = []
    for name, arg in (expect or {}).items():
        fn = CHECKS.get(name)
        if fn is None:
            out.append(CheckResult(name, False, f"unknown check {name!r}"))
            continue
        try:
            out.append(fn(obs, arg))
        except Exception as exc:  # noqa: BLE001 — a broken check must fail loudly, not crash the run
            out.append(CheckResult(name, False, f"check raised {exc!r}"))
    return out


def passed(results: list[CheckResult]) -> bool:
    return bool(results) and all(r.ok for r in results)
