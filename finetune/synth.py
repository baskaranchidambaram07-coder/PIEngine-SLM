"""Turn requirements into training data.

Every example produced here is traceable to one requirement id, and every
generated answer is screened by checks.py before it is allowed into the set —
the same assertions that will later grade the model. Generation that fails its
own grader is discarded, not repaired. That is rejection sampling (STaR / RFT),
and it is the reason a 1.7B teacher is usable at all: we are not asking the
teacher to be better than the student, we are asking it to occasionally be
right and letting a deterministic filter keep only those attempts.

Contexts are rendered from finetune/surrogates/*.tmpl with a fresh entity cast
per example, so the adapter learns the behaviour and cannot memorise a fact.
Recipes that have an exactly-known correct answer — refusals, tool calls, small
talk, the MOM skeleton — are built deterministically and never touch a teacher.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

from core import chunking

from .checks import Observation, passed, run_checks
from .spec import Requirement, Spec
from .surrogates import entities
from .targets import AgentCtx, context_block

SURROGATE_DIR = Path(__file__).resolve().parent / "surrogates"
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

DEFAULT_TEACHER = "http://127.0.0.1:8302/v1"

CONTEXT_SEPARATOR = "---\n\n"


def probe_key(user_text: str) -> str:
    """Normalise a user turn down to the question, for probe/leak comparison."""
    return user_text.rsplit(CONTEXT_SEPARATOR, 1)[-1].strip().lower()


# ------------------------------------------------------------------ example

@dataclass
class Example:
    requirement: str
    recipe: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)

    def user_text(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "user"), "")

    def question(self) -> str:
        """The user's actual question, with any retrieved-context preamble cut."""
        return probe_key(self.user_text())

    def key(self) -> str:
        """Dedup key: the graded pair, whitespace-normalised."""
        u = re.sub(r"\s+", " ", self.user_text()).strip().lower()
        a = re.sub(r"\s+", " ", self.messages[-1]["content"]).strip().lower()
        return f"{u}||{a}"


# ------------------------------------------------------------------ surrogate

def surrogate_docs(name: str, rng: random.Random) -> list[tuple[str, str]]:
    """Render one surrogate set with a fresh cast -> [(doc_name, text)]."""
    d = SURROGATE_DIR / name
    if not d.is_dir():
        raise SystemExit(f"unknown surrogate {name!r} (have: "
                         f"{[p.name for p in SURROGATE_DIR.iterdir() if p.is_dir()]})")
    cast = entities.draw(rng)
    out = []
    for tmpl in sorted(d.glob("*.md.tmpl")):
        text = entities.render(tmpl.read_text(encoding="utf-8"), cast)
        out.append((tmpl.name.replace(".tmpl", ""), text))
    return out


def surrogate_chunks(name: str, rng: random.Random) -> list[dict]:
    out = []
    for doc, text in surrogate_docs(name, rng):
        for i, ch in enumerate(chunking.split_text(text)):
            out.append({"doc_name": doc, "chunk_index": i, "text": ch})
    return out


def pick_context(chunks: list[dict], rng: random.Random, k: int = 3) -> list[dict]:
    k = min(k, len(chunks))
    start = rng.randrange(0, max(1, len(chunks) - k + 1))
    return chunks[start:start + k]


# ------------------------------------------------------------------ teacher

class Teacher:
    """Any OpenAI-compatible endpoint. Local llama-server by default."""

    def __init__(self, url: str | None, model: str = "teacher", temperature: float = 0.8):
        self.url = (url or "").rstrip("/") or None
        self.model = model
        self.temperature = temperature
        self.calls = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def json_obj(self, prompt: str, keys: tuple[str, ...], max_tokens: int = 400) -> dict | None:
        if not self.enabled:
            return None
        self.calls += 1
        try:
            r = requests.post(f"{self.url}/chat/completions", timeout=300, json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt + " /no_think"}],
                "temperature": self.temperature, "top_p": 0.9, "max_tokens": max_tokens,
            })
            r.raise_for_status()
            r.encoding = "utf-8"
            content = r.json()["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                self.failures += 1
                return None
            obj = json.loads(m.group(0))
            if not all(k in obj and isinstance(obj[k], str) for k in keys):
                self.failures += 1
                return None
            return obj
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            self.failures += 1
            return None


# ------------------------------------------------------------------ recipes
#
# A recipe signature is (ctx, req, rng, teacher, n) -> list[Example].
# `ctx` is the agent (system prompt + declared tools); `req` carries the
# requirement's own `synth:` block and, importantly, its `probes` — whose
# `expect:` checks are reused verbatim to screen what the recipe produces.


def _screen(req: Requirement, obs: Observation, extra: dict | None = None) -> list:
    """Grade a candidate answer with the requirement's own checks."""
    expect = dict(req.probes[0].expect) if req.probes else {}
    expect.pop("contains_any", None)     # probe-specific facts; meaningless here
    expect.pop("contains_all", None)
    expect.update(extra or {})
    return run_checks(expect, obs)


def _chat(ctx: AgentCtx, user: str, assistant: str) -> list[dict]:
    from runtime.app import build_system_prompt
    return [
        {"role": "system", "content": build_system_prompt(
            {"system_prompt": ctx.manifest.get("system_prompt", ""), "tools": ctx.tools})},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


QUESTION_HINTS = [
    "a factual lookup about one person or date",
    "a status question a manager would ask walking into a meeting",
    "a question about ownership or accountability",
    "a question about money or percentages",
    "a question about a decision that was taken",
    "a question about what happens next and when",
]


def grounded_qa(ctx: AgentCtx, req: Requirement, rng: random.Random,
                teacher: Teacher, n: int) -> list[Example]:
    """Cited, grounded, short answers over rendered surrogate context."""
    sets = req.synth.get("surrogates") or ["project-delivery"]
    k = int(req.synth.get("k", 3))            # rejection-sampling attempts
    out: list[Example] = []
    attempts = 0
    while len(out) < n and attempts < n * k * 2:
        attempts += 1
        chunks = surrogate_chunks(rng.choice(sets), rng)
        picked = pick_context(chunks, rng, k=rng.choice([2, 3, 4]))
        ctx_block = context_block(picked)
        docs = [c["doc_name"] for c in picked]
        hint = rng.choice(QUESTION_HINTS)

        qa = teacher.json_obj(
            "You are writing evaluation data for an offline meeting assistant.\n"
            f"Below is context retrieved from a knowledge base. Write ONE question "
            f"({hint}) that IS answerable from this context, and the ideal answer.\n"
            "The answer must: use only facts present in the context, name the source "
            "document, be under 60 words, and use short bullets.\n\n"
            f"{ctx_block}"
            'Reply with JSON only: {"question": "...", "answer": "..."}',
            ("question", "answer"))
        if qa is None:
            ex = _extractive_pair(picked, rng)
            if ex is None:
                continue
            question, answer = ex
        else:
            question, answer = qa["question"].strip(), qa["answer"].strip()

        obs = Observation(text=answer, context=ctx_block, doc_names=docs,
                          tool_defs=ctx.tools)
        results = _screen(req, obs, {"grounded": True, "cites": True, "max_words": 90})
        if not passed(results):
            continue
        out.append(Example(req.id, "grounded_qa", _chat(ctx, ctx_block + question, answer),
                           {"docs": docs, "teacher": qa is not None}))
    return out


_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)


def _extractive_pair(picked: list[dict], rng: random.Random) -> tuple[str, str] | None:
    """Teacher-free fallback: lift a table row and cite it.

    Lower variety than a teacher, but it is guaranteed grounded, so it keeps
    the pipeline productive on a box with no teacher available.
    """
    for ch in rng.sample(picked, len(picked)):
        rows = [r for r in _ROW.findall(ch["text"])
                if "---" not in r[0] and len(r[0]) > 2 and not r[0].lower().startswith(("ref", "name", "milestone", "risk"))]
        if not rows:
            continue
        a, b, c = rng.choice(rows)
        q = rng.choice([f"What is the status of {a}?", f"Tell me about {a}.",
                        f"Who owns {a}?", f"Quick brief on {a}?"])
        ans = f"{a} — {b}, {c} ({ch['doc_name']})."
        return q, ans
    return None


REFUSAL_FORMS = [
    "I don't have that in the knowledge base.",
    "That isn't in the retrieved context — I can't confirm it.",
    "The knowledge base doesn't mention {subject}.",
    "No information about {subject} in the documents I have.",
]


def absent_entity(ctx: AgentCtx, req: Requirement, rng: random.Random,
                  teacher: Teacher, n: int) -> list[Example]:
    """Plausible in-domain questions the context cannot answer -> refusal.

    The absent subject is drawn from a DIFFERENT cast than the one used to
    render the context, so it reads like it belongs and provably is not there.
    This is the failure docs/rag-gating.md measured: vector search always
    returns its top_k, so the model sees relevant-looking passages and infers
    the subject must be covered.
    """
    sets = req.synth.get("surrogates") or ["project-delivery"]
    out: list[Example] = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        chunks = surrogate_chunks(rng.choice(sets), rng)
        picked = pick_context(chunks, rng, k=rng.choice([2, 3, 4]))
        ctx_block = context_block(picked)
        haystack = ctx_block.lower()

        other = entities.draw(rng)
        candidates = [
            (f"{other['sponsor']}", f"Who is {other['sponsor']}?"),
            (f"{other['vendor']}", f"What did we agree with {other['vendor']}?"),
            ("Phase 4", "What is the Phase 4 budget?"),
            (other["city"], f"Who is based in {other['city']}?"),
            (other["system"], f"When does {other['system']} migrate?"),
            ("the penalty clause", "What is the penalty clause for late delivery?"),
            ("the disaster recovery test", "When was the disaster recovery test signed off?"),
            (other["ref1"], f"What is the status of action {other['ref1']}?"),
        ]
        subject, question = rng.choice(candidates)
        if subject.lower() in haystack:
            continue

        answer = rng.choice(REFUSAL_FORMS).format(subject=subject)
        obs = Observation(text=answer, context=ctx_block,
                          doc_names=[c["doc_name"] for c in picked], tool_defs=ctx.tools)
        if not passed(run_checks({"refuses": True, "max_words": 40}, obs)):
            continue
        out.append(Example(req.id, "absent_entity", _chat(ctx, ctx_block + question, answer),
                           {"absent_subject": subject}))
    return out


def tool_router(ctx: AgentCtx, req: Requirement, rng: random.Random,
                teacher: Teacher, n: int) -> list[Example]:
    """Right tool, valid JSON — plus the negatives that stop over-calling.

    The assistant turn is constructed by us from the tool's own schema, so the
    gold call is correct by definition. Only the user's phrasing is teacher-
    written (and falls back to templates), because phrasing is the one part a
    small model actually needs variety in.
    """
    tools = ctx.tools
    if not tools:
        return []
    out: list[Example] = []
    positives = int(n * 0.7)

    while len(out) < positives:
        tool = rng.choice(tools)
        args = _sample_args(tool, rng)
        gold = "<tool_call>\n" + json.dumps({"name": tool["name"], "arguments": args}) + "\n</tool_call>"
        phrasing = teacher.json_obj(
            "Write how a busy delivery manager would ask, in one natural sentence, for this "
            "action to be performed. Do not mention JSON, tools or function names.\n"
            f"Action: {tool['name']} — {tool.get('description', '')}\n"
            f"Details: {json.dumps(args)}\n"
            'Reply with JSON only: {"request": "..."}',
            ("request",), max_tokens=120)
        request = (phrasing or {}).get("request", "").strip() or _template_request(tool, args)
        if len(request) < 8 or len(request) > 280:
            request = _template_request(tool, args)

        obs = Observation(text="", tool_calls=[{"name": tool["name"], "arguments": args}],
                          tool_defs=tools)
        if not passed(run_checks({"tool_call": tool["name"], "tool_args_valid": True}, obs)):
            continue
        out.append(Example(req.id, "tool_router", _chat(ctx, request, gold), {"tool": tool["name"]}))

    # Negatives: a KB question and small talk must NOT fire a tool. Without
    # these, a tool-heavy set teaches the model that tools are always the answer.
    # Negatives: a knowledge-base question must be ANSWERED, not routed to a
    # tool. Without them a tool-heavy set teaches the model that tools are
    # always the move. Small talk is a negative too, but it belongs to the
    # register requirement — generating it here as well only produces
    # duplicates, which the global dedup then charges to whichever recipe ran
    # second.
    for _ in range(n - len(out)):
        # a fresh cast per example: rendering once outside the loop makes every
        # negative a near-duplicate of the last
        picked = pick_context(surrogate_chunks("project-delivery", rng), rng, k=2)
        ex = _extractive_pair(picked, rng)
        if ex is None:
            continue
        q, a = ex
        out.append(Example(req.id, "tool_router_negative",
                           _chat(ctx, context_block(picked) + q, a), {"tool": None}))
    return out


def _sample_args(tool: dict, rng: random.Random) -> dict:
    cast = entities.draw(rng)
    props = (tool.get("parameters") or {}).get("properties") or {}
    required = set((tool.get("parameters") or {}).get("required") or [])
    args: dict = {}
    for name, schema in props.items():
        # optional properties appear about half the time, so the model learns
        # that "required" is the floor and not the whole shape
        if name not in required and rng.random() < 0.5:
            continue
        args[name] = _sample_value(name, schema, cast, rng)
    return args


def _sample_value(name: str, schema: dict, cast: dict, rng: random.Random):
    if schema.get("enum"):
        return rng.choice(schema["enum"])
    t = schema.get("type", "string")
    if t in ("integer", "number"):
        return rng.randint(1, 12)
    if t == "boolean":
        return rng.choice([True, False])
    low = name.lower()
    if "date" in low or "due" in low or "when" in low:
        return cast["d2"]
    if "owner" in low or "assignee" in low or "who" in low or "to" in low:
        return cast["ops"]
    if "title" in low or "subject" in low or "summary" in low:
        return rng.choice([f"Close out {cast['topic']}",
                           f"Confirm the {cast['system']} cutover slot",
                           f"Review {cast['risk']}",
                           f"Chase {cast['vendor']} on the support rota"])
    if "body" in low or "content" in low or "notes" in low or "description" in low:
        return (f"Following the {cast['programme']} steerco on {cast['d1']}, please confirm "
                f"the position on {cast['topic']} before {cast['d2']}.")
    return rng.choice([cast["programme"], cast["topic"], cast["system"]])


def _template_request(tool: dict, args: dict) -> str:
    bits = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in args.items())
    verb = tool["name"].replace("_", " ")
    return f"Please {verb}" + (f" — {bits}." if bits else ".")


# One opener can have several acceptable replies; the pairs are enumerated so
# the recipe can fill a larger n without repeating itself. Every reply is short
# and none of them mentions the knowledge base.
_OPENERS = {
    "hi": ["Hi — what do you need?", "Hi. What can I pull up?"],
    "hello": ["Hello. What can I pull up for you?", "Hello — what are you after?"],
    "hey there": ["Hey. What are you looking for?", "Hey — what do you need?"],
    "good morning": ["Morning. What's first?", "Good morning — where do you want to start?"],
    "good afternoon": ["Afternoon. What do you need?"],
    "thanks!": ["Any time.", "No problem."],
    "thank you": ["You're welcome.", "Any time."],
    "thanks, that helps": ["Glad it helped.", "Good."],
    "ok": ["Got it.", "Understood."],
    "okay, noted": ["Noted."],
    "cheers": ["No problem.", "Any time."],
    "perfect": ["Good.", "Great."],
    "nice one": ["Any time."],
    "great, thanks": ["Glad it helped."],
    "bye": ["Speak later.", "Bye."],
    "talk later": ["Speak later."],
    "sorry, ignore that": ["No problem — what did you want instead?"],
}
SMALLTALK = [(k, v) for k, vs in _OPENERS.items() for v in vs]


def smalltalk(ctx: AgentCtx, req: Requirement, rng: random.Random,
              teacher: Teacher, n: int) -> list[Example]:
    """No context in the prompt -> one short natural line.

    docs/rag-gating.md: once the lexical gate stops injecting context for a
    greeting, a system prompt that says "answer only from the context" turns
    "hi" into "I don't have that in the knowledge base". These examples are the
    other half of that fix, learned rather than prompted.
    """
    out = []
    pool = list(SMALLTALK)
    rng.shuffle(pool)
    if n > len(pool):
        # asking for more small talk than there are distinct exchanges just
        # duplicates rows; three epochs over a duplicate is six epochs on it
        print(f"  {req.id}: capping smalltalk at {len(pool)} distinct exchanges "
              f"(spec asked for {n})")
        n = len(pool)
    for greet, reply in pool[:n]:
        obs = Observation(text=reply, context="", doc_names=[], tool_defs=ctx.tools)
        if not passed(run_checks({"max_words": 15, "refuses": False, "no_tool_call": True}, obs)):
            continue
        out.append(Example(req.id, "smalltalk", _chat(ctx, greet, reply), {}))
    return out


MOM_ASK = [
    "Draft the minutes for this meeting.",
    "Write up the MOM from these notes.",
    "Turn this into minutes I can send.",
    "Give me the meeting minutes.",
]


def format_mom(ctx: AgentCtx, req: Requirement, rng: random.Random,
               teacher: Teacher, n: int) -> list[Example]:
    """House MOM skeleton, built deterministically from the surrogate cast.

    The correct output shape is known exactly, so there is nothing for a
    teacher to add and a great deal for it to get wrong.
    """
    out = []
    for _ in range(n):
        cast = entities.draw(rng)
        doc = "mom-steerco.md"
        body = entities.render(
            (SURROGATE_DIR / "project-delivery" / "mom-steerco.md.tmpl").read_text(encoding="utf-8"),
            cast)
        ctx_block = context_block([{"doc_name": doc, "chunk_index": 0, "text": body}])
        gold = (
            f"**{cast['programme']} — Steering Committee, {cast['d1']}** ({doc})\n\n"
            f"**Decisions**\n"
            f"- {cast['topic'].capitalize()} approach approved; implementation starts {cast['d2']}.\n"
            f"- Go-live confirmed for {cast['d3']}, scope frozen.\n"
            f"- {cast['vendor']} to provide {cast['n1']} weeks of hypercare at no extra cost.\n\n"
            f"**Actions**\n"
            f"- {cast['ref1']} — cutover runbook for {cast['system']} — {cast['ops']} — {cast['d2']}\n"
            f"- {cast['ref2']} — close security review findings — {cast['sec']} — {cast['d3']}\n\n"
            f"**Risks**\n"
            f"- {cast['risk'].capitalize()} — owner {cast['pm']}.\n"
        )
        obs = Observation(text=gold, context=ctx_block, doc_names=[doc], tool_defs=ctx.tools)
        if not passed(run_checks({"grounded": True, "cites": True, "bullets": True}, obs)):
            continue
        out.append(Example(req.id, "format_mom",
                           _chat(ctx, ctx_block + rng.choice(MOM_ASK), gold), {}))
    return out


RECIPES = {
    "grounded_qa": grounded_qa,
    "absent_entity": absent_entity,
    "tool_router": tool_router,
    "smalltalk": smalltalk,
    "format_mom": format_mom,
}


# ------------------------------------------------------------------ driver

def build(spec: Spec, teacher: Teacher, seed: int = 7,
          skip: set[str] | None = None) -> tuple[list[Example], dict]:
    """Generate the whole dataset for a spec. `skip` = requirement ids that
    already pass at baseline and must not consume adapter capacity."""
    rng = random.Random(seed)
    ctx = AgentCtx(spec.agent)
    skip = skip or set()
    examples: list[Example] = []
    report: dict = {"per_requirement": {}, "skipped": sorted(skip),
                    "teacher": {"url": teacher.url, "calls": 0, "failures": 0}}

    banned = spec.probe_texts()
    seen: set[str] = set()

    def keep(made: list[Example]) -> tuple[list[Example], int, int]:
        """Drop probe collisions and duplicates as they are produced.

        A probe question in the training set turns the scorecard into a
        memorisation test, so this filter runs at generation time and
        validate.py keeps the same check as a backstop.
        """
        kept, leaked, dup = [], 0, 0
        for ex in made:
            question = probe_key(ex.user_text())
            if question in banned:
                leaked += 1
                continue
            k = ex.key()
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            kept.append(ex)
        return kept, leaked, dup

    for req in spec.requirements:
        if req.id in skip:
            report["per_requirement"][req.id] = {"recipe": None, "produced": 0,
                                                 "reason": "already passing at baseline"}
            continue
        recipe = req.synth.get("recipe")
        if not recipe:
            report["per_requirement"][req.id] = {"recipe": None, "produced": 0,
                                                 "reason": "no synth recipe declared"}
            continue
        fn = RECIPES.get(recipe)
        if fn is None:
            raise SystemExit(f"{req.id}: unknown recipe {recipe!r} (have {sorted(RECIPES)})")
        want = int(req.synth.get("n", 40))
        made, leaked, dup = keep(fn(ctx, req, rng, teacher, want))
        examples += made
        report["per_requirement"][req.id] = {"recipe": recipe, "requested": want,
                                             "produced": len(made),
                                             "dropped_probe_collision": leaked,
                                             "dropped_duplicate": dup,
                                             "yield": round(len(made) / want, 2) if want else 0}

    report["teacher"]["calls"] = teacher.calls
    report["teacher"]["failures"] = teacher.failures
    report["agent_source"] = ctx.source
    return examples, report
