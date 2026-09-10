# RAG gating: why `min_score` cannot solve this

Found 2026-09-09, when a bare `hi` to the Product Insights agent returned a
six-bullet product summary costing 1,085 prompt tokens and roughly 141 seconds
on-device.

## The two defects

1. **Retrieval was unconditional.** `embed → search → inject` ran on every turn
   before the model saw the message. Nothing decided *whether* to retrieve.
2. **The default `min_score` of 0.35 could not filter anything.** It sat below
   the score floor the embedder produces for unrelated text, so every message
   passed.

A third trap sits behind them: raising the threshold alone converts a greeting
into `I don't know`, because the system prompt says to answer only from context.
Any threshold change must be paired with a prompt clause allowing a natural
reply when no context is retrieved.

## Why a threshold is the wrong instrument

Cosine similarity from a bi-encoder is a **ranking** function, not a **relevance
detector**. Sentence embeddings are anisotropic — they occupy a narrow cone, so
unrelated text scores ~0.45–0.55 rather than ~0. There is no "no match" value to
threshold against, and the usable band shifts per corpus: a homogeneous KB
compresses every score upward.

The `bge` query prefix (`Represent this sentence for searching relevant
passages:`) makes it worse. It is applied to *every* message, so `hi` is
rewritten into query-space and pulled toward the passages.

### Measured on this project's own two KBs

Four candidate signals, negatives = chitchat + off-topic questions,
positives = genuine questions:

| signal | product-insights (7 chunks) | meeting-intelligence (12 chunks) |
| --- | --- | --- |
| `top1` (what `min_score` uses) | separates (+0.107) | **overlaps (−0.004)** |
| `margin` (top1 − mean) | overlaps (−0.035) | overlaps (−0.021) |
| `z` ((top1 − mean) / sd) | overlaps (−1.162) | overlaps (−1.098) |
| `gap12` (top1 − top2) | overlaps (−0.031) | overlaps (−0.041) |

The `margin` hypothesis — that a real question yields a *peaked* score
distribution while noise yields a flat one — was the lead theory going in. It is
worse than raw `top1` on both KBs. Distribution shape tracks KB homogeneity, not
query relevance.

The decisive case, on meeting-intelligence:

```
0.5587  offtopic  "what time does the train to Bangalore leave?"
0.5553  chitchat  "thanks!"
0.5547  ontopic   "what did we decide about the architecture?"
```

A real question ranks **below** both a greeting and an unrelated question. No
threshold separates them. A value tuned on one KB will break another — which is
why `DEFAULT_RAG.min_score` is now 0.45 and is documented as a **floor**, not a
gate.

### Lexical evidence transfers better, but not far enough

Unlike cosine, term overlap genuinely reaches zero when a query shares no
content words with the corpus.

| gate | product-insights | meeting-intelligence |
| --- | --- | --- |
| IDF-weighted overlap alone | separates (+0.59) | overlaps (−0.87) |
| lexical AND cosine | 0/18 false positives, 8/8 recall | 3/18 false positives, 6/6 recall |

Full recall on both, most false positives removed — a good mitigation, not a
solution.

## What ships today

`core/gating.py`, mirrored line-for-line in
`android_app/AgentRuntime/src/gating.ts`, called before embedding in
`runtime/app.py:retrieve()` and `src/chat.ts`.

The gate is deterministic and lexical, and answers only the question it can
answer safely: *is there anything here to look up?* A message with no content
words, or one made entirely of greeting tokens, cannot be a KB query.

Validated at 20/20 recall on real questions (including `price?`,
`hi, when does the Duo ship?`, `thanks - and the price?`) and 16/16 on chitchat.
Result: `hi` went from 1,085 prompt tokens to 6.

**It deliberately does not catch off-topic questions.** `what is the weather in
Chennai?` has content words and passes. No lexical rule catches that, and
`min_score` is not a reliable backstop.

## The remaining gap, and the permanent fix

The gate patches the symptom. The architectural gap is that **retrieval is a
fixed preprocessing stage rather than a decision**, so the only component that
understands intent — the model — has no say in whether to retrieve, and no
ability to reformulate the query.

The fix is to expose retrieval as a **tool** the model calls. This project
already has the machinery: Qwen3 `<tool_call>` parsing, `MAX_TOOL_ROUNDS`,
`tools.py`, and per-agent tool declarations in the manifest. Wiring
`search_knowledge_base(query)` into it would:

- make `hi` cost nothing, with no lexicon to maintain;
- handle off-topic questions, which no rule-based gate can;
- allow query reformulation, which would fix the `cheapest product` failure
  (the model can search `price list` instead of echoing the user's words);
- allow multi-hop lookups;
- reduce `min_score` to what cosine is actually good at — ranking *within* an
  already-decided retrieval.

**The risk is tool-calling reliability at 1.7B, and worse at the 0.6B ceiling of
the target handset.** A model that fails to call the tool answers ungrounded —
a silent wrong answer, which is worse than a slow right one. Measure call rate
on a question set before adopting, and keep a fallback that retrieves anyway
when a question-shaped message produced no tool call.

## Separate axis: retrieval quality

Gating decides *whether*. These are *what*, and are not fixed by any of the
above:

- Chunks straddle section boundaries, so the AirPods tail and the Watch head
  share a chunk (`core/chunking.py` merges to 1200 chars ignoring headings).
- Aggregate questions (`cheapest`, `how many`, `compare all`) need the whole
  table, not the nearest chunks. Hybrid BM25 (SQLite FTS5, already available in
  both stock SQLite and op-sqlite) plus vector search is the standard remedy.

## Measured: retrieval-as-a-tool (2026-09-09)

Prototype in `prototypes/tool_rag.py`, measured by `prototypes/tool_measure.py`
against the real `runtime/llm.py` streaming path. 12 real questions across both
agents, 8 small-talk inputs, 4 off-topic questions. Qwen3-1.7B Q4_K_M, temp 0.3.

The decisive variable was **the prompt, not the model**.

| | agent's own prompt | tool-first prompt |
| --- | --- | --- |
| tool-call rate on real questions | **3/12 (25%)** | **12/12 (100%)** |
| answered correctly | 3/12 (25%) | 11/12 (92%) |
| false calls on small talk | 0/8 | 0/8 |
| query reformulated, not echoed | 3/12 | 11/12 |
| malformed / leaked tool syntax | 0/12 | 0/12 |

Run them with `venv\Scripts\python prototypes/tool_measure.py agent` and
`... toolfirst`.

### Why the agent's own prompt scores 25%

Both agent prompts contain a rule of the form *answer only from the provided
context; if it does not contain the answer, say you don't have it*. With nothing
pre-injected, that rule fires **before** the model considers the tool, so it
declines instead of searching. Meeting Intelligence, whose prompt states the
decline sentence verbatim, scored 0/6 — it answered "I don't have that in the
knowledge base" to six questions whose answers sit in its KB.

That is the dangerous failure mode: a confident, ungrounded refusal that looks
like a correct answer. Any migration to tool-based retrieval **must rewrite the
agent prompt at the same time**; the two are not independent.

### What is genuinely better

- Small talk costs nothing: 1–2s and no embedding, vs ~25s with pre-injection.
- Query reformulation works — `cheapest product announced`, `Pro Max battery
  life`, `Phase 2 budget consumption` — and it fixed the `cheapest product`
  question that pre-injection got wrong at `top_k=2`.
- No threshold anywhere in the path.

### What is worse, and must be fixed before adopting

- **Latency on KB questions rises**: ~27–44s vs ~25s, from the extra
  generation round to emit the call. Tool RAG trades KB-question latency for
  small-talk latency.
- **Off-topic hallucination regressed.** Asked *did Apple announce a MacBook
  Pro?*, pre-injection correctly answered no; the tool version searched and
  then asserted "Apple did announce a MacBook Pro in 2026, as outlined in the
  product briefing." Confident and false. Pre-injection showed the model the
  absence of evidence; the tool path lets it reason about a search result it
  half-read.
- The Anita/Meera name conflation persists identically in both architectures,
  confirming it as a chunk-content and generation problem, not a retrieval one.

### Recommendation

Viable, but not a drop-in. Adopt only with a rewritten tool-first prompt per
agent, and only after the off-topic hallucination is addressed — a wrong
confident answer is worse than a slow correct one. Keep the lexical gate either
way: it costs nothing and does not depend on model behaviour.

## Fixing the off-topic hallucination (2026-09-10)

Measured by `prototypes/final_measure.py`. Two suites that must BOTH hold:
**A** absent entities (subjects that sound like they belong but are not in the
KB — a MacBook Pro in an Apple-event KB, a Phase 4 budget in a Phase-2 project
KB) must be refused; **B** real questions must still be answered.

| approach | A. absent refused | B. real correct |
| --- | --- | --- |
| no verification (baseline) | 8/12 | 11/12 |
| verify **every content word** | **12/12** | **6/12** ✗ |
| verify **entity terms only** | **11/12** | **10/12** ✓ |

### Why checking every content word fails

The tool result told the model which query words were missing from the
passages. It stopped hallucination completely and destroyed ordinary answering:

```
BAD  warned  How much does the foldable iPhone cost?   missing: ['cost']
BAD  warned  What is the cheapest product announced?   missing: ['cheapest']
BAD  warned  Who runs the InfoSec review?              missing: ['runs']
```

A question says *how much does X cost*; the document says *starts at $1,999*.
Interrogative vocabulary is not evidence of absence, but the model treated the
warning as authoritative and refused. Worse than the bug it fixed.

### What works: entity-only verification

`absent_entities()` in `prototypes/tool_rag.py` counts a term as evidence only
if it is entity-like — carries an uppercase letter anywhere (so camelCase
product names like `iPad` and `iPhone` count) or contains a digit — and checks
it against the retrieved text on **whole-word** boundaries. Multi-token spans
(`Phase 4`) are checked as a phrase too, since each token can be present
separately while the entity is absent.

Two bugs found by an offline harness before spending model time:

- `iPad`/`iPhone`/`iOS` were missed — capitalisation tests that look only at the
  first character do not see camelCase.
- `CTO` was missed — substring matching finds `cto` inside `fa`**`cto`**`r`, so
  it needs `\b` boundaries.

Offline the final rule fires on 10/12 absent entities and **0/12 real
questions** — the zero is the property the all-words version lacked.

### The residual failures are all pre-existing

- **A, 1 miss**: *what did we decide about the mobile app?* — an all-lowercase
  subject carries no entity signal. Genuinely out of reach for this rule.
- **B, 2 misses**, both independently confirmed as unrelated to the gate:
  - *Who is the executive sponsor at Acme?* — fails 0/3 **with** the gate and
    0/3 **without** it, and the gate fires no warning on it. The chunk that
    contains "Executive sponsor" is not retrieved at all, so the model declines
    correctly on what it was given. A retrieval miss, not a grounding failure.
    (The earlier 11/12 baseline scored this as a pass; it is flaky.)
  - *Who runs the InfoSec review?* — the Anita/Meera conflation, identical in
    both architectures.

### Also fixed

`runtime/llm.py` now passes `min_p` through (SmolChat-Android exposes it and we
did not). It is sent **only when an agent sets it**: llama.cpp defaults to 0.05,
so unconditionally sending 0.0 would have silently disabled min-p sampling for
every agent.

### On SmolChat-Android / OnDevice-RAG-Android

Reviewed as a possible source for this fix. SmolChat itself has no RAG. Its
companion `OnDevice-RAG-Android` does, but has the same defect this document
describes: `getSimilarChunks(queryEmbedding, n = 5)` with no threshold, and a
default prompt that is bare context + query with no grounding instruction. Its
ObjectBox HNSW retrieval (over-fetch 25, trim to 5) is a useful reference for
scale, which is not a constraint at 12 chunks.
