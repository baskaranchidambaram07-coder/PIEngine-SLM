"""Render prototypes/attach_bench_results.json as the markdown tables used in
docs/attachments-and-pii-guard.md §8. Prints to stdout; pass --write to
replace the RESULTS_TABLE_PLACEHOLDER (or the previously rendered block) in the doc.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "prototypes" / "attach_bench_results.json"
DOC = ROOT / "docs" / "attachments-and-pii-guard.md"
BEGIN, END = "<!-- bench:begin -->", "<!-- bench:end -->"


def render(res: dict) -> str:
    out = [BEGIN, "", "**Per file** (upload = validate + extract/OCR + embed + guard, incl. judge calls):", "",
           "| file | upload s | guard verdict | expected | questions correct | answer s (avg) | first token s | model |",
           "|---|---|---|---|---|---|---|---|"]
    for r in res["files"]:
        qs = r["questions"]
        npass = sum(1 for q in qs if q["ok"])
        avg = f"{sum(q['total_s'] for q in qs) / len(qs):.1f}" if qs else "—"
        ft = f"{sum((q['first_token_s'] or 0) for q in qs) / len(qs):.1f}" if qs else "—"
        model = next((q["model"] for q in qs if q.get("model")), "agent SLM") if qs else "—"
        mark = "✓" if r["verdict_ok"] else "✗"
        labels = f" ({', '.join(r['labels'])})" if r["labels"] else ""
        out.append(f"| `{r['file']}` | {r['upload_s']} | {r['verdict']} {mark}{labels} | {r['expected']} | "
                   f"{npass}/{len(qs)} | {avg} | {ft} | {model} |")
    out += ["", "**Questions and answers:**", "", "| file | question | answer (trimmed) | ok |", "|---|---|---|---|"]
    for r in res["files"]:
        for q in r["questions"]:
            a = q["answer"].replace("|", "\\|").replace("\n", " ")
            out.append(f"| `{r['file']}` | {q['question']} | {a[:140]} | {'✓' if q['ok'] else '✗'} |")
    out += ["", "**Message-only guard checks** (no file attached; regex only under the default policy):", "",
            "| message | blocked | expected | s |", "|---|---|---|---|"]
    for q in res["queries"]:
        out.append(f"| {q['text']} | {q['blocked']} | {q['expected_block']} | {q['total_s']} |")
    vt = sum(1 for r in res["files"] if r["verdict_ok"])
    qp = sum(1 for r in res["files"] for q in r["questions"] if q["ok"])
    qt = sum(len(r["questions"]) for r in res["files"])
    gq = sum(1 for q in res["queries"] if q["ok"])
    out += ["", f"**Totals:** guard verdicts {vt}/{len(res['files'])} · answers {qp}/{qt} · "
                f"message guard {gq}/{len(res['queries'])} · judge policy `{res['policy']['judge']}` · "
                f"vision `{res['vision']['model']}` · agent `{res['agent']}`", "", END]
    return "\n".join(out)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    block = render(res)
    if "--write" in sys.argv:
        doc = DOC.read_text(encoding="utf-8")
        if BEGIN in doc:
            doc = re.sub(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END), lambda _m: block, doc)
        else:
            doc = doc.replace("RESULTS_TABLE_PLACEHOLDER", block)
        DOC.write_text(doc, encoding="utf-8")
        print("written to", DOC.relative_to(ROOT))
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
