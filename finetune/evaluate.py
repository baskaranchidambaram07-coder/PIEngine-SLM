"""Score an agent against its requirement spec, and gate a candidate adapter.

Produces a scorecard: one pass rate per requirement, one for the regression
suite, and a weighted total. Two scorecards (baseline vs candidate) go through
`gate()`, which is what decides whether an adapter is allowed to ship.

The gate is deliberately harder to pass than "the total went up":

  1. every requirement must reach its own declared target — a tune that fixes
     citations and leaves refusals broken has not met the requirement;
  2. the regression suite must not drop at all beyond tolerance — this project
     has already been bitten by an improvement that silently broke name
     attribution (README / meeting-intelligence prompt tuning);
  3. latency must not blow out — an adapter is free at inference, but a tune
     that makes the model chattier costs real seconds on 8 tok/s hardware.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .checks import Observation, passed, run_checks
from .spec import Probe, Spec

REPORT_DIR = Path(__file__).resolve().parent / "reports"

LATENCY_TOLERANCE = 1.25   # candidate p50 may be at most 25% slower
REGRESSION_TOLERANCE = 0.0  # fraction of the regression suite allowed to drop


# ------------------------------------------------------------------ running

# Checks that are decided by the first tool call, so the runtime's tool loop
# has nothing left to contribute once it has fired.
_TOOL_CHECKS = {"tool_call", "tool_args_valid"}


def _run_probe(target, probe: Probe, repeat: int) -> dict:
    stop_after_tool = bool(_TOOL_CHECKS & set(probe.expect))
    runs = []
    for _ in range(repeat):
        obs: Observation = target.ask(probe.ask, history=probe.history, context=probe.context,
                                      stop_after_tool=stop_after_tool)
        results = run_checks(probe.expect, obs)
        runs.append({
            "ok": passed(results),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in results],
            "answer": obs.text[:600],
            "tool_calls": obs.tool_calls,
            "seconds": obs.seconds,
            "tokens": obs.tokens,
        })
    hits = sum(r["ok"] for r in runs)
    best = next((r for r in runs if not r["ok"]), runs[0])   # show a failure if there is one
    return {
        "ask": probe.ask, "note": probe.note,
        "pass_rate": hits / len(runs), "runs": len(runs),
        "ok": hits == len(runs),
        "checks": best["checks"], "answer": best["answer"],
        "tool_calls": best["tool_calls"],
        "seconds": round(statistics.mean(r["seconds"] for r in runs), 2),
        "tokens": int(statistics.mean(r["tokens"] for r in runs)),
    }


def score(spec: Spec, target, repeat: int = 1, label: str = "") -> dict:
    """Run every probe in the spec and return a scorecard dict."""
    reqs = []
    latencies: list[float] = []
    for req in spec.requirements:
        probes = [_run_probe(target, p, repeat) for p in req.probes]
        latencies += [p["seconds"] for p in probes]
        rate = sum(p["pass_rate"] for p in probes) / len(probes)
        reqs.append({
            "id": req.id, "kind": req.kind, "statement": req.statement,
            "target": req.target, "weight": req.weight,
            "pass_rate": round(rate, 4), "met": rate >= req.target,
            "n_probes": len(probes), "probes": probes,
        })

    reg_probes = [_run_probe(target, p, repeat) for p in spec.regression]
    latencies += [p["seconds"] for p in reg_probes]
    reg_rate = (sum(p["pass_rate"] for p in reg_probes) / len(reg_probes)) if reg_probes else 1.0

    total_w = sum(r["weight"] for r in reqs) or 1.0
    weighted = sum(r["pass_rate"] * r["weight"] for r in reqs) / total_w

    return {
        "label": label or getattr(target, "name", "run"),
        "spec": str(spec.path), "agent": spec.agent, "adapter_id": spec.adapter_id,
        "base_model": spec.base_model,
        "target": target.describe(),
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repeat": repeat,
        "score": round(weighted, 4),
        "requirements": reqs,
        "regression": {"pass_rate": round(reg_rate, 4), "n_probes": len(reg_probes),
                       "probes": reg_probes},
        "latency": {"p50_seconds": round(statistics.median(latencies), 2) if latencies else 0.0,
                    "mean_seconds": round(statistics.mean(latencies), 2) if latencies else 0.0},
    }


# ------------------------------------------------------------------ gating

def gate(candidate: dict, baseline: dict | None = None) -> dict:
    """Promote / reject a candidate scorecard. Returns a verdict dict."""
    failures, notes = [], []

    for r in candidate["requirements"]:
        if not r["met"]:
            failures.append(f"{r['id']}: {r['pass_rate']:.0%} < target {r['target']:.0%}")
        elif baseline:
            was = next((b["pass_rate"] for b in baseline["requirements"] if b["id"] == r["id"]), None)
            if was is not None and r["pass_rate"] < was:
                failures.append(f"{r['id']}: regressed {was:.0%} -> {r['pass_rate']:.0%}")

    if baseline:
        b_reg = baseline["regression"]["pass_rate"]
        c_reg = candidate["regression"]["pass_rate"]
        if c_reg < b_reg - REGRESSION_TOLERANCE:
            failures.append(f"regression suite: {b_reg:.0%} -> {c_reg:.0%}")
        b_p50 = baseline["latency"]["p50_seconds"] or 0
        c_p50 = candidate["latency"]["p50_seconds"] or 0
        if b_p50 and c_p50 > b_p50 * LATENCY_TOLERANCE:
            failures.append(f"latency p50: {b_p50:.1f}s -> {c_p50:.1f}s "
                            f"(> {LATENCY_TOLERANCE:g}x)")
        delta = candidate["score"] - baseline["score"]
        notes.append(f"weighted score {baseline['score']:.0%} -> {candidate['score']:.0%} "
                     f"({delta:+.0%})")
        if delta <= 0 and not failures:
            failures.append("no net improvement over baseline — not worth a 1 GB fleet update")

    return {"promote": not failures, "failures": failures, "notes": notes}


# ------------------------------------------------------------------ reporting

def to_markdown(card: dict, baseline: dict | None = None, verdict: dict | None = None) -> str:
    def cmp(rid: str, now: float) -> str:
        """Trailing baseline cell, or nothing when there is no baseline."""
        if not baseline:
            return ""
        was = next((b["pass_rate"] for b in baseline["requirements"] if b["id"] == rid), None)
        if was is None:
            return " — |"
        arrow = "up" if now > was else ("DOWN" if now < was else "=")
        return f" {was:.0%} {arrow} |"

    head = "| requirement | kind | pass | target | met |" + (" baseline |" if baseline else "")
    rule = "|---|---|---|---|---|" + ("---|" if baseline else "")
    lines = [
        f"# Scorecard — {card['agent']} ({card['label']})",
        "",
        f"- spec: `{card['spec']}`",
        f"- target: `{json.dumps(card['target'])}`",
        f"- run at {card['at']}, {card['repeat']} sample(s) per probe",
        f"- **weighted score {card['score']:.0%}**, regression suite "
        f"{card['regression']['pass_rate']:.0%}, p50 {card['latency']['p50_seconds']:.1f}s",
        "",
        head, rule,
    ]
    for r in card["requirements"]:
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['pass_rate']:.0%} | {r['target']:.0%} | "
            f"{'yes' if r['met'] else 'NO'} |" + cmp(r["id"], r["pass_rate"]))

    lines += ["", "## Failing probes", ""]
    any_fail = False
    for r in card["requirements"] + [{"id": "regression", **card["regression"]}]:
        for p in r.get("probes", []):
            if p["ok"]:
                continue
            any_fail = True
            why = "; ".join(f"{c['name']}: {c['detail']}" for c in p["checks"] if not c["ok"])
            lines.append(f"**{r['id']}** — {p['ask']!r}")
            lines.append(f"- failed: {why}")
            lines.append(f"- answer: {p['answer'][:300]!r}")
            lines.append("")
    if not any_fail:
        lines.append("_none_")

    if verdict:
        lines += ["", "## Verdict", "",
                  f"**{'PROMOTE' if verdict['promote'] else 'REJECT'}**", ""]
        lines += [f"- {n}" for n in verdict["notes"]]
        lines += [f"- blocker: {f}" for f in verdict["failures"]]
    return "\n".join(lines) + "\n"


def save(card: dict, name: str) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    p = REPORT_DIR / f"{name}.json"
    p.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = REPORT_DIR / f"{Path(name_or_path).stem}.json"
    if not p.exists():
        raise SystemExit(f"scorecard not found: {name_or_path}")
    return json.loads(p.read_text(encoding="utf-8"))
