"""Seed the Meeting Intelligence demo through the Studio HTTP API.

Creates the agent, uploads the sample knowledge base, attaches tools,
and publishes v1 — exercising every Journey-1 API on the way.

Usage:  venv\\Scripts\\python scripts\\seed_demo.py
"""
import sys
from pathlib import Path

import requests

STUDIO = "http://127.0.0.1:8100"
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "sample_docs"

SYSTEM_PROMPT = """You are the Meeting Intelligence Agent — a private, on-device copilot for delivery managers, project managers, and solution architects.

You have access to a knowledge base containing this user's meeting minutes, project status reports, customer profiles, risk registers, and decision logs. Context passages retrieved from it are provided with each question.

Your job:
- BEFORE meetings: brief the user — agenda, previous minutes, open action items, stakeholder context, risks.
- DURING meetings: answer factual questions instantly from the knowledge base; identify commitments being made.
- AFTER meetings: draft minutes of meeting (MOM), action item lists, and follow-up emails.

Rules:
- Answer ONLY from the provided context. If the context does not contain the answer, say "I don't have that in the knowledge base" — never invent facts, names, dates, or commitments.
- Cite the source document name when you state a fact.
- Be concise: the user is on a phone between meetings and needs answers in under 10 seconds.
- Format lists as short bullet points. Lead with the direct answer."""

TOOLS = [
    {
        "name": "create_action_item", "kind": "builtin",
        "description": "Create an action item / task on the device. Stored locally and synced when online.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Action item title"},
            "owner": {"type": "string", "description": "Person responsible"},
            "due_date": {"type": "string", "description": "Due date YYYY-MM-DD"}},
            "required": ["title"]},
        "config": {},
    },
    {
        "name": "get_todays_meetings", "kind": "builtin",
        "description": "List today's meetings from the device calendar.",
        "parameters": {"type": "object", "properties": {}},
        "config": {},
    },
    {
        "name": "draft_email", "kind": "builtin",
        "description": "Draft a follow-up email in the device outbox for user review.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"]},
        "config": {},
    },
]


def main() -> int:
    # idempotent: remove previous demo agent if present
    for a in requests.get(f"{STUDIO}/api/agents", timeout=10).json():
        if a["id"].startswith("meeting-intelligence"):
            requests.delete(f"{STUDIO}/api/agents/{a['id']}", timeout=10)
            print(f"removed existing agent {a['id']}")

    agent = requests.post(f"{STUDIO}/api/agents", json={
        "name": "Meeting Intelligence",
        "description": "Personal meeting copilot for delivery managers — briefings, instant answers, MOMs and follow-ups. Runs fully offline.",
        "scenario": ("Delivery managers, project managers and solution architects move between customer meetings all day. "
                     "They need sub-10-second answers about previous minutes, open actions, stakeholders, risks and decisions "
                     "without opening a laptop or searching Outlook/Confluence/Jira — even with no connectivity."),
        "system_prompt": SYSTEM_PROMPT,
        "model_id": "qwen3-1.7b-q4_k_m",
        "generation": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 768},
        "rag": {"top_k": 4, "min_score": 0.35},
        "tools": TOOLS,
    }, timeout=10).json()
    aid = agent["id"]
    print(f"created agent: {aid}")

    files = [("files", (p.name, p.read_bytes(), "text/markdown")) for p in sorted(DOCS.glob("*.md"))]
    r = requests.post(f"{STUDIO}/api/agents/{aid}/docs", files=files, timeout=300).json()
    for u in r["uploaded"]:
        print(f"  indexed {u['name']}: {u.get('chunks', u.get('error'))} chunks")
    print(f"KB total: {r['stats']}")

    probe = requests.post(f"{STUDIO}/api/agents/{aid}/search",
                          json={"query": "what are the open high risks?", "top_k": 3}, timeout=60).json()
    top = probe["results"][0]
    print(f"retrieval probe: top hit {top['doc_name']} (score {top['score']})")
    assert top["score"] > 0.5, "retrieval quality check failed"

    pub = requests.post(f"{STUDIO}/api/agents/{aid}/publish", timeout=30).json()
    print(f"published: {pub['bundle']} (v{pub['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
