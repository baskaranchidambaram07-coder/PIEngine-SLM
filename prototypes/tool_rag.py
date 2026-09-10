"""Prototype: retrieval as a TOOL the model calls, instead of unconditional
pre-injection. Measures the number that decides whether this is viable — how
reliably a 1.7B model calls the KB tool when it should, and leaves it alone
when it should not.

Runs against the real runtime.llm streaming path so the numbers are faithful.
Does not touch the production /api/chat endpoint.
"""
import json
import re
import sys
import time

sys.path.insert(0, r"C:\slm")

from core import embeddings, gating, kbstore   # noqa: E402
from runtime import llm                        # noqa: E402
from runtime.app import QWEN_TOOLS_HEADER, AGENTS_DIR  # noqa: E402

KB_TOOL = {
    "name": "search_knowledge_base",
    "description": ("Search the user's private knowledge base of documents. Call this "
                    "whenever the user asks about facts, people, dates, numbers, decisions "
                    "or documents. Do NOT call it for greetings, thanks, or small talk."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "What to look up. Rephrase into keywords that would "
                                     "appear in the documents."},
        },
        "required": ["query"],
    },
}


def load_manifest(agent_id):
    return json.loads((AGENTS_DIR / agent_id / "manifest.json").read_text(encoding="utf-8"))


def kb_search(agent_id, query, top_k=4, verify=True):
    """Search, and optionally tell the model which query terms are ABSENT.

    A vector search always returns its top_k, however irrelevant — so the model
    sees passages and infers the topic must be covered. That is where the
    off-topic hallucination comes from ("Apple did announce a MacBook Pro").

    Cosine cannot flag this: unrelated text still scores ~0.45-0.55 (see
    docs/rag-gating.md). Lexical overlap can — a term either appears in the
    retrieved text or it does not. So we hand the model that fact directly
    rather than hoping it notices the absence itself.
    """
    path = AGENTS_DIR / agent_id / "kb.sqlite"
    if not path.exists():
        return {"results": [], "note": "no knowledge base installed"}
    hits = kbstore.search(kbstore.connect(path), embeddings.embed_query(query),
                          top_k=top_k, min_score=0.0)
    out = {"results": [{"doc": h["doc_name"], "chunk": h["chunk_index"],
                        "score": h["score"], "text": h["text"]} for h in hits]}
    if not verify:
        return out

    haystack = " ".join(h["text"] for h in hits).lower()
    missing = absent_entities(query, haystack)
    out["subjects_missing"] = missing
    if missing:
        out["warning"] = (
            "The knowledge base contains NOTHING about: " + ", ".join(missing)
            + ". Tell the user that is not in the knowledge base. Do NOT answer about a "
              "different product, person or topic that happens to appear in the passages, "
              "and do NOT say yes."
        )
    return out


# Words that start a sentence or a question and therefore carry no capitalisation
# signal of their own.
_LEADING = frozenset("what which who whom whose when where why how did do does is are was "
                     "were can could will would should has have had tell show give list "
                     "the a an in on at for of and or".split())


def _present(term, haystack):
    """Whole-word match. Substring matching gives false negatives: 'CTO' is
    'inside' the word 'factor', which would hide a genuinely absent entity."""
    return re.search(r"\b" + re.escape(term.lower()) + r"\b", haystack) is not None


def absent_entities(query, haystack):
    """Entity-like terms from the query that appear nowhere in the passages.

    Checking EVERY content word does not work: a question says "how much does X
    cost" while the document says "starts at $1,999", so ordinary interrogative
    vocabulary ("cost", "cheapest", "runs") reads as evidence of absence and the
    model refuses questions it can actually answer. Measured: 11/12 -> 6/12.

    What actually distinguishes an unanswerable question is that its SUBJECT is
    missing — MacBook, iPad, HomePod, M5, Phase 4 — and those are capitalised or
    carry digits, unlike the phrasing words around them. So only entity-like
    tokens count as evidence.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    entities, span = [], []
    for tok in tokens:
        low = tok.lower()
        has_digit = any(c.isdigit() for c in tok)
        # any uppercase anywhere, so camelCase product names (iPad, iPhone, iOS)
        # count — their first letter is lowercase.
        has_upper = any(c.isupper() for c in tok)
        if (has_digit or has_upper) and low not in _LEADING:
            span.append(tok)
        else:
            if span:
                entities.append(span)
            span = []
    if span:
        entities.append(span)

    missing = []
    for span in entities:
        phrase = " ".join(span)
        # A multi-token entity ("Phase 4") can be absent even when each token is
        # present separately, so check the phrase as well as its parts.
        if len(span) > 1 and not _present(phrase, haystack):
            missing.append(phrase)
            continue
        for tok in span:
            # bare digits are too common to be evidence on their own
            if tok.isdigit():
                continue
            if not _present(tok, haystack):
                missing.append(tok)
    # de-duplicate, preserve order
    seen, out = set(), []
    for m in missing:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


TOOL_FIRST = (
    "You are a helpful assistant with a private knowledge base of the user's documents.\n"
    "You cannot see the knowledge base directly. The ONLY way to read it is to call "
    "search_knowledge_base.\n"
    "For ANY question about facts, people, dates, numbers, products, decisions or "
    "documents, your FIRST action must be a search_knowledge_base call. Never answer "
    "such a question from memory, and never say you lack the information before you "
    "have searched.\n"
    "For greetings and small talk, just reply in one short sentence without searching.\n"
    "After the search returns, answer only from its results, briefly and factually.\n"
    "The search always returns its closest matches, even when the knowledge base has "
    "nothing on the subject. Before answering, check that the specific thing the user "
    "asked about actually appears in the passages. If the result carries a 'warning' "
    "field, or the thing asked about is absent, reply that it is not in the knowledge "
    "base. Never answer 'yes' about something you did not find, and never substitute a "
    "different product, person or topic that happens to appear in the passages."
)


def tool_system_prompt(manifest, mode="agent"):
    """Agent's own prompt, or a tool-first prompt with no pre-tool decline rule.

    The agent prompts say "answer only from the provided context, else say you
    don't have it". With nothing pre-injected that rule fires BEFORE the model
    considers the tool, so it declines instead of searching. mode="toolfirst"
    removes that conflict.
    """
    if mode == "toolfirst":
        base = TOOL_FIRST
    else:
        base = manifest.get("system_prompt", "")
        base += ("\n\nThe knowledge base is NOT automatically provided. To answer any question "
                 "about the user's documents you MUST first call search_knowledge_base. "
                 "Answer only from what that tool returns.")
    lines = json.dumps({"type": "function", "function": {
        "name": KB_TOOL["name"], "description": KB_TOOL["description"],
        "parameters": KB_TOOL["parameters"]}})
    return base + QWEN_TOOLS_HEADER.format(tool_lines=lines) + " /no_think"


def run(agent_id, question, generation, max_rounds=3, mode="agent", verify=True):
    """Returns (called?, searched_queries, answer, seconds)."""
    manifest = load_manifest(agent_id)
    llm.ensure_model(manifest["model"]["file"],
                     context=min(int(manifest["model"].get("context_length", 4096)), 8192))

    messages = [{"role": "system", "content": tool_system_prompt(manifest, mode)},
                {"role": "user", "content": question}]
    searched, answer = [], ""
    t0 = time.time()

    for _round in range(max_rounds):
        called = False
        for ev in llm.stream_chat(messages, generation):
            if ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "tool_call":
                call = ev.get("call") or {}
                if call.get("name") == "search_knowledge_base":
                    q = (call.get("arguments") or {}).get("query", question)
                    searched.append(q)
                    result = kb_search(agent_id, q, verify=verify)
                    messages.append({"role": "assistant",
                                     "content": f"<tool_call>\n{ev['raw']}\n</tool_call>"})
                    messages.append({"role": "user",
                                     "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>"})
                    called = True
                    answer = ""      # the pre-tool text is not the final answer
        if not called:
            break

    return bool(searched), searched, answer.strip(), time.time() - t0


# ---- leaked / malformed tool syntax the parser does not catch ---------------
LEAK = re.compile(r'<tool>|"name"\s*:\s*"search_knowledge_base"|<function|```json', re.I)
