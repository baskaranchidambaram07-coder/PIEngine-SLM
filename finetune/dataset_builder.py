"""Build a scenario-specific SFT dataset for an agent created in the Studio.

Produces chat-format JSONL (system/user/assistant) that teaches the SLM the
*behaviour* of the agent — grounded answering, citation discipline, refusing
when the context lacks the answer, and correct tool selection — using the
agent's own system prompt and knowledge base as raw material.

Three data sources are combined:
1. Curated behaviour seeds (hand-written, scenario-agnostic discipline).
2. KB-grounded pairs generated heuristically from the agent's chunks.
3. Optional teacher distillation: point --teacher at any OpenAI-compatible
   endpoint (a bigger local model via llama-server, or a cloud API) to
   synthesise higher-quality Q/A pairs from each chunk.

Usage (from repo root, venv active):
    python finetune/dataset_builder.py --agent meeting-intelligence
    python finetune/dataset_builder.py --agent meeting-intelligence \
        --teacher http://127.0.0.1:8303/v1 --teacher-model whatever
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUDIO_DB = ROOT / "studio" / "studio.db"
KB_DIR = ROOT / "studio" / "kb"
OUT_DIR = Path(__file__).resolve().parent / "datasets"

# ------------------------------------------------------------- behaviour seeds

BEHAVIOUR_SEEDS = [
    {
        "user": "Context from the knowledge base:\n\n[status-report.md #0]\nProject Phoenix is GREEN. Milestone M3 completed 2 days early. Budget consumed: 41% vs 45% planned.\n\n---\n\nHow is Project Phoenix doing?",
        "assistant": "Project Phoenix is **GREEN** (status-report.md):\n- M3 completed 2 days early\n- Budget: 41% consumed vs 45% planned — tracking under budget",
    },
    {
        "user": "Context from the knowledge base:\n\n[mom-2026-05-02.md #1]\nAction A-12: Prepare the migration runbook — Owner: Kavya N, due 9 May.\n\n---\n\nWho owns the security audit action?",
        "assistant": "I don't have that in the knowledge base. The only action in the retrieved context is A-12 (migration runbook, Kavya N, due 9 May) — nothing about a security audit.",
    },
    {
        "user": "Context from the knowledge base:\n\n[customer-profile.md #0]\nDavid Chen — VP Engineering. Prefers data-first discussions; skeptical of vendor claims. Decision maker for platform budget.\n\n---\n\nGive me a one-line brief on David Chen before my call.",
        "assistant": "David Chen (VP Engineering) is your platform-budget decision maker — lead with data, not claims; he's skeptical of vendor pitches (customer-profile.md).",
    },
    {
        "user": "What's the capital of France?",
        "assistant": "That's outside this agent's scope — I answer from your project knowledge base (meetings, actions, risks, stakeholders). For general questions, use a general-purpose assistant.",
    },
]

TOOL_SEEDS = [
    {
        "user": "Add an action item: review the DR failover test results, owner Anil, due next Wednesday.",
        "assistant": "<tool_call>\n{\"name\": \"create_action_item\", \"arguments\": {\"title\": \"Review the DR failover test results\", \"owner\": \"Anil\", \"due_date\": \"2026-07-15\"}}\n</tool_call>",
    },
    {
        "user": "What's on my calendar today?",
        "assistant": "<tool_call>\n{\"name\": \"get_todays_meetings\", \"arguments\": {}}\n</tool_call>",
    },
]

# ------------------------------------------------------- heuristic generation

_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)

QUESTION_FORMS = [
    "What does the knowledge base say about {topic}?",
    "Summarize {topic} for me — I'm walking into a meeting.",
    "Quick brief: {topic}?",
    "What do I need to know about {topic}?",
]


def kb_chunks(agent_id: str) -> list[dict]:
    kb = KB_DIR / f"{agent_id}.sqlite"
    if not kb.exists():
        raise SystemExit(f"no knowledge base at {kb}")
    conn = sqlite3.connect(kb)
    rows = conn.execute("SELECT doc_name, chunk_index, text FROM chunks").fetchall()
    return [{"doc": r[0], "idx": r[1], "text": r[2]} for r in rows]


def heuristic_pairs(chunks: list[dict], rng: random.Random) -> list[dict]:
    """Turn each chunk's headings into 'brief me on X' pairs whose answer is
    the chunk body — teaching extractive, cited answering over this domain."""
    pairs = []
    for ch in chunks:
        headings = _HEADING.findall(ch["text"])
        topic = headings[0] if headings else None
        if not topic:
            row = _TABLE_ROW.findall(ch["text"])
            topic = row[1][0].strip() if len(row) > 1 else None
        if not topic or len(topic) < 6:
            continue
        topic_clean = re.sub(r"\s*[—\-–]\s*.*$", "", topic).strip()
        question = rng.choice(QUESTION_FORMS).format(topic=topic_clean)
        body = re.sub(r"^#{1,3}\s+.+$", "", ch["text"], flags=re.MULTILINE).strip()
        answer_lines = [ln.strip("* ") for ln in body.splitlines() if ln.strip()][:6]
        answer = f"From {ch['doc']}:\n" + "\n".join(f"- {ln}" for ln in answer_lines)
        pairs.append({
            "user": f"Context from the knowledge base:\n\n[{ch['doc']} #{ch['idx']}]\n{ch['text']}\n\n---\n\n{question}",
            "assistant": answer,
        })
    return pairs


# ------------------------------------------------------- teacher distillation

def teacher_pairs(chunks: list[dict], endpoint: str, model: str) -> list[dict]:
    import requests

    pairs = []
    for ch in chunks:
        prompt = (
            "You create fine-tuning data for a mobile meeting-intelligence assistant.\n"
            "Given the document excerpt below, write ONE realistic question a delivery "
            "manager would ask on their phone between meetings, and the ideal concise "
            "answer (bullet points, cite the document name, only use facts present).\n"
            f"Document: {ch['doc']}\n---\n{ch['text']}\n---\n"
            'Reply as JSON: {"question": "...", "answer": "..."}'
        )
        try:
            r = requests.post(f"{endpoint}/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7, "max_tokens": 500,
            }, timeout=120)
            content = r.json()["choices"][0]["message"]["content"]
            qa = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0))
            pairs.append({
                "user": f"Context from the knowledge base:\n\n[{ch['doc']} #{ch['idx']}]\n{ch['text']}\n\n---\n\n{qa['question']}",
                "assistant": qa["answer"],
            })
        except Exception as exc:  # noqa: BLE001 — skip bad generations, keep going
            print(f"  teacher skip ({ch['doc']} #{ch['idx']}): {exc}")
    return pairs


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="agent id from the Studio")
    ap.add_argument("--teacher", help="OpenAI-compatible base URL for distillation")
    ap.add_argument("--teacher-model", default="teacher")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    conn = sqlite3.connect(STUDIO_DB)
    row = conn.execute("SELECT config FROM agents WHERE id = ?", (args.agent,)).fetchone()
    if not row:
        raise SystemExit(f"agent '{args.agent}' not found in studio.db")
    cfg = json.loads(row[0])
    system_prompt = cfg.get("system_prompt", "You are a helpful enterprise assistant.")

    rng = random.Random(args.seed)
    chunks = kb_chunks(args.agent)
    print(f"agent '{args.agent}': {len(chunks)} KB chunks")

    examples = []
    for seed in BEHAVIOUR_SEEDS + TOOL_SEEDS:
        examples.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed["user"]},
            {"role": "assistant", "content": seed["assistant"]},
        ]})

    hp = heuristic_pairs(chunks, rng)
    print(f"heuristic pairs: {len(hp)}")
    tp = teacher_pairs(chunks, args.teacher.rstrip("/"), args.teacher_model) if args.teacher else []
    if args.teacher:
        print(f"teacher pairs: {len(tp)}")

    for p in hp + tp:
        examples.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p["user"]},
            {"role": "assistant", "content": p["assistant"]},
        ]})

    rng.shuffle(examples)
    n_val = max(2, len(examples) // 10)
    OUT_DIR.mkdir(exist_ok=True)
    train_f = OUT_DIR / f"{args.agent}-train.jsonl"
    val_f = OUT_DIR / f"{args.agent}-val.jsonl"
    with open(train_f, "w", encoding="utf-8") as fh:
        for ex in examples[n_val:]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_f, "w", encoding="utf-8") as fh:
        for ex in examples[:n_val]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples) - n_val} train / {n_val} val examples")
    print(f"  {train_f}\n  {val_f}")
    print("NOTE: for production quality you want 300-2000 examples — add real "
          "MOM/Q&A transcripts and use --teacher distillation over a larger corpus.")


if __name__ == "__main__":
    main()
