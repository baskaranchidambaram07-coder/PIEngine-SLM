# Requirement-driven fine-tuning

How a scenario agent in this platform gets weights that behave the way the
scenario requires — and how we prove it did, before a fleet downloads it.

Code: `finetune/`, `core/adapters.py`. Entry point: `python -m finetune`.

---

## 1. The problem this solves, and the one it does not

The Studio already produces a working agent from a system prompt, a knowledge
base and a tool list. Everything factual is handled by RAG. So what is left
for training?

What is left is **behaviour the prompt cannot reliably buy on a 1.7B model**:

| Failure | What it looks like | Why the prompt does not fix it |
|---|---|---|
| No attribution | An answer with no source document named | The instruction is in the prompt already; the model drops it under context pressure |
| Confident invention | "The Phase 4 budget is £240,000" when Phase 4 is not in the KB | Retrieval always returns its top_k, so the model sees plausible passages and infers coverage (`rag-gating.md`) |
| Malformed tool calls | `create_action_item` with an owner and no title | The runtime's parser drops it silently, so the user sees the assistant ignore them |
| Length | 300 words at 8-14 tok/s | Length is a latency defect on this hardware, not a style preference |
| Wrong register with no context | "hi" answered with "I don't have that in the knowledge base" | The direct consequence of the "answer only from context" rule once the lexical gate correctly suppresses retrieval |

And the thing training must **not** be used for: teaching the model facts. The
knowledge base is a bundle artifact that changes weekly, per agent, per
customer. Weights are a shared 1 GB download that reaches every device. Any
fact baked into weights is stale the next week, duplicated by RAG, and
unrevocable once distributed. This is decision **D1** below, and the pipeline
enforces it mechanically rather than relying on discipline.

---

## 2. Decisions

**D1 — Train behaviour, never facts.**
Training contexts are rendered from templated surrogate documents
(`finetune/surrogates/*.md.tmpl`) with a fresh randomised entity cast per
example. The same document *shape* carries different people, dates and numbers
every time, so the only learnable signal is the behaviour. `validate.py`
additionally scans the finished dataset for capitalised terms that appear in
the agent's real knowledge base and nowhere in the synthesiser's own source,
and **blocks the dataset** if it finds any.

**D2 — Ship an adapter, not a merged model.**
A merged fine-tune means a second 1 GB Q4_K_M per tuned agent. A LoRA adapter
is 20-60 MB on top of the base model every agent already shares. Both runtimes
support this natively — verified in this repo:

```
llama-server --lora <adapter.gguf> --lora-init-without-apply
             POST /lora-adapters  [{"id":0,"scale":1.0}]
llama.rn     initLlama({ ..., lora_list: [{path, scaled}] })
             applyLoraAdapters() / removeLoraAdapters()
```

Consequences: a fleet of five tuned agents costs one base model plus five small
files; rollback is removing one line from a manifest and republishing; and
baseline-vs-candidate A/B runs in a single llama-server process by flipping the
adapter scale between 0 and 1, so nothing but the weights differs between the
two scorecards.

**D3 — The requirement spec is the contract.**
One YAML per tune (`finetune/specs/<agent>.yaml`) states each requirement, the
probes that test it, the pass rate it must reach, and the recipe that generates
data for it. Three consumers, one file:

```
requirements[].synth   ->  synth.py      what data to make
requirements[].probes  ->  evaluate.py   how the model is scored
requirements[].target  ->  the gate      whether it may ship
```

A requirement with no probes is rejected at load time. An untestable
requirement cannot be claimed.

**D4 — Cheapest instrument first.**
`baseline` runs before `synth`. Any requirement already at target is reported
and `--skip-passing` drops it from the dataset. Adapter capacity spent on
solved behaviour is how tunes regress. If the whole spec passes at baseline,
there is nothing to train and the pipeline says so.

**D5 — Gate on requirements *and* on blast radius.**
Promotion needs all three: every requirement at its own target, no drop on a
regression suite that is never trained on, and no latency blow-out. This
project has already shipped an "improvement" that silently broke name
attribution (README, prompt-tuning note) — the regression suite exists because
of that.

**D6 — The gradient step is the only off-box stage.**
This server is a 4-vCPU t3.xlarge with no GPU and ~4 GB free disk; it cannot
hold a PyTorch install, let alone train. Rather than pretend, the pipeline
isolates the gradient step behind a portable **job pack** that runs unchanged
on Colab, an Ubuntu box or a spot g4dn, and returns an **adapter pack**. That
boundary doubles as the compliance boundary: what leaves this machine is
surrogate text and hyperparameters — no customer document, no chat log, no KB.

---

## 3. Pipeline

```
   ON THIS BOX (CPU)                                      OFF-BOX (GPU)
   ─────────────────                                      ─────────────
   agent in Studio
        │
        │ scaffold
        ▼
   specs/<agent>.yaml ──────┐
        │                   │
        │ baseline          │ (probes)                    ┌──────────────┐
        ▼                   │                             │  Colab T4 /  │
   scorecard (stock) ───────┤                             │  Ubuntu GPU  │
        │  "what actually   │                             │  / spot g4dn │
        │   needs training" │                             └──────┬───────┘
        │ synth  ◄──────────┘ (recipes)                          │
        ▼                                                        │
   surrogates + teacher                                          │
        │  rejection sampling: every candidate answer graded     │
        │  by checks.py, failures discarded                      │
        ▼                                                        │
   validate  ── leakage · duplication · balance · length ·       │
        │       PRIVACY (real KB terms)                          │
        ▼                                                        │
   datasets/*.jsonl                                              │
        │ pack                                                   │
        ▼                                                        │
   <adapter>-jobpack.zip ─────── upload ──────────────────────►  │
                                                            train_qlora.py
                                                            convert_lora_to_gguf
   <adapter>-adapterpack.zip ◄── download ─────────────────────  ┘
        │ import
        ▼
   models/adapters/ + registry.json   (status: imported)
        │ evaluate   ── same probes, adapter scale 0 vs 1, one process
        ▼
   scorecard diff + gate ──► promoted | rejected
        │
        │ publish (Studio)
        ▼
   manifest.adapter ──► /adapters/<file> ──► device applies it at load
```

Each stage writes the artifact the next one reads, so a run can pause for a
week between `pack` and `import` without losing the thread.

---

## 4. The requirement spec

```yaml
spec: slm-finetune-spec/1
agent: meeting-intelligence
base_model: qwen3-1.7b-q4_k_m
adapter_id: meeting-intelligence-behaviour-v1

requirements:
  - id: R2-refuse
    kind: refusal
    statement: >-
      Declines when the retrieved context does not contain the answer.
    rationale: >-
      Vector search always returns its top_k, so an absent entity arrives
      surrounded by plausible passages.
    target: 0.9        # pass rate required to promote
    weight: 2          # contribution to the weighted score
    probes:            # the held-out test set
      - ask: "What is the Phase 4 budget?"
        expect: { refuses: true }
    synth:
      recipe: absent_entity
      n: 70
      surrogates: [project-delivery, customer]

regression:            # never trained on; the blast-radius check
  - ask: "ok"
    expect: { max_words: 25, no_tool_call: true }
```

Requirement kinds are behavioural by construction: `citation`, `grounding`,
`refusal`, `tool_call`, `format`, `length`, `register`. There is no kind that
expresses a fact, which is D1 encoded in the schema.

### Checks

`finetune/checks.py` is deterministic and lexical — no LLM judge. On a 4-vCPU
box a judge would cost more than the tune, and a 1.7B judge grading a 1.7B
student mostly measures their shared blind spots. Each check only claims what
it can actually verify:

| Check | What it asserts |
|---|---|
| `cites` | the answer names one of the documents it was shown |
| `grounded` | every number and proper noun in the answer appears in the context |
| `refuses` | the answer matches a refusal sentence shape |
| `tool_call` / `no_tool_call` | exactly the expected call, or none |
| `tool_args_valid` | arguments parse and satisfy the declared `required` set |
| `max_words`, `max_bullets`, `regex`, `json_valid`, `max_seconds` | shape and budget |

`grounded` is deliberately one-sided: it catches invented figures, dates and
names — what makes an answer dangerous — and says nothing about whether the
prose is faithful.

---

## 5. Data synthesis

Recipes turn a requirement into examples. Each is tagged with the requirement
id it serves, so the dataset is traceable line by line.

| Recipe | Produces | Teacher |
|---|---|---|
| `grounded_qa` | cited, grounded, short answers over surrogate context | yes, with extractive fallback |
| `absent_entity` | plausible in-domain question the context cannot answer → refusal | no |
| `tool_router` | correct tool + valid JSON, plus KB-question negatives | phrasing only |
| `smalltalk` | no context → one short natural line | no |
| `format_mom` | the house MOM skeleton | no |

Two things make this work without a big teacher:

**Rejection sampling.** Every generated answer is run through `checks.py`
before it is admitted, using the requirement's own checks. Failures are
discarded, never repaired. We are not asking the teacher to be better than the
student — we are asking it to occasionally be right, and letting a
deterministic filter keep only those attempts (STaR / RFT). The default teacher
is the local llama-server on 8302; any OpenAI-compatible endpoint works, and a
larger teacher simply raises the yield.

**Known-answer recipes skip the teacher entirely.** For refusals, tool calls,
small talk and the MOM skeleton the correct output is known exactly, so there
is nothing for a teacher to add and a great deal for it to get wrong. Only the
*user's phrasing* is ever teacher-written in those recipes.

---

## 6. The dataset gate

`validate.py` runs before anything is written, because every failure mode below
is cheap here and expensive after a GPU has been booked:

| Check | Why |
|---|---|
| probe leakage | a probe in the training set turns the scorecard into a memorisation test |
| duplication | 3 epochs over a duplicated example is 6 epochs on that example |
| balance | one requirement dominating the set is how tunes start over-refusing |
| refusal share | above ~35% refusals, models begin refusing answerable questions |
| length | examples past `max_len` are truncated mid-answer, teaching the model to stop mid-answer |
| **privacy** | real KB terms in the set means customer content is being written into distributed weights |

It found all three of its non-privacy failure modes on the first real run of
this pipeline (19 probe collisions, 15% duplicates) plus a noisy privacy check,
and blocked the dataset until the generator was fixed. That is the check
earning its place, not a formality.

---

## 7. Evaluation and promotion

Two targets, and the difference matters:

* **`runtime`** — `POST /api/chat` on the Device Runtime. Measures the shipped
  system: lexical gate, retrieval, prompt assembly, tool loop, streaming. This
  is what a user experiences.
* **`llama`** — llama-server directly, same prompt and same retrieval, nothing
  else. Isolates the weights, so an A/B attributes the delta to the tune rather
  than to a prompt edit that landed the same week.

Two practical notes on the `runtime` target. Tool probes run the real tool
loop, so grading `tool_call` requirements appends demo action items and outbox
entries to `runtime/device_storage/device_data.json` — harmless, but that file
is scratch after an eval run. And a tool probe costs several generation rounds,
so it dominates wall-clock: on this 4-vCPU box a 28-probe suite takes roughly
45 minutes, most of it in the tool requirement.

Adapter A/B runs in one llama-server process started with
`--lora <adapter> --lora-init-without-apply`, flipping scale 0 → 1 between the
two passes. Same weights in RAM, same cache behaviour, one variable.

The gate (`evaluate.gate`) promotes only when **all** hold:

1. every requirement at or above its own target;
2. no requirement lower than it was at baseline;
3. the regression suite no worse than baseline;
4. p50 latency within 1.25× of baseline;
5. a net improvement in the weighted score — an adapter that changes nothing is
   not worth a fleet update.

The verdict is written into `models/adapters/registry.json`, and
`core/adapters.manifest_entry()` refuses to attach anything but a `promoted`
adapter to a bundle. The gate is not advisory.

---

## 8. Deployment

Publishing an agent with `adapter_id` set produces a manifest block:

```json
"adapter": {
  "id": "meeting-intelligence-behaviour-v1",
  "file": "meeting-intelligence-behaviour-v1-f16.gguf",
  "size_bytes": 31457280,
  "sha256": "…",
  "base_model_id": "qwen3-1.7b-q4_k_m",
  "scale": 1.0
}
```

Publish refuses if the adapter's `base_model_id` differs from the agent's
model: applying a LoRA to the wrong base produces garbage rather than an error,
so it has to be caught at build time.

* **Web runtime** — `runtime/llm.py` passes `--lora` to llama-server; the
  adapter is part of the server's identity, so switching adapters restarts it
  exactly like switching models does.
* **Android** — `installAgent` downloads the adapter from `/adapters/<file>`
  (portal only; no CDN mirrors our adapters), and `ensureChatModel` passes
  `lora_list` to `initLlama`. A missing adapter is not fatal — the agent runs
  on stock weights — because failing the chat would make a 30 MB download a
  hard dependency of a 1 GB model that is already present.
* **Governance** — `telemetry.events.adapter_id` records which weights actually
  answered, so the dashboard can compare tuned and stock populations on real
  usage rather than on the eval suite alone.

---

## 9. Hardware and cost

| Stage | Where | Cost |
|---|---|---|
| spec, baseline, synth, validate, pack | this box (4 vCPU, no GPU) | minutes; baseline is the slow part at ~25 s/probe |
| train + convert | any NVIDIA ≥ 8 GB VRAM | QLoRA 4-bit, r=16, ~6-8 GB VRAM; a few hundred examples × 3 epochs does not fill an hour. Colab T4 free; g4dn.xlarge spot ≈ $0.16/h |
| evaluate, gate, publish | this box | minutes |

Qwen3 is Apache-2.0 and ungated, so no Hugging Face token is needed. Gated
bases (Llama, Gemma) need one on the GPU box only.

---

## 10. Status

### Measured baseline (2026-09-10, stock Qwen3-1.7B, `runtime` target)

Weighted score **78%**, regression suite 100%, p50 19.8 s (the box was
contended by another job, so treat the latency as an upper bound).

| requirement | kind | pass | target | met |
|---|---|---|---|---|
| R1-cite | citation | 83% | 90% | no |
| R2-refuse | refusal | 67% | 90% | no |
| R3-tools | tool_call | 80% | 85% | no |
| R4-brief | length | 100% | 90% | **yes** |
| R5-social | register | 67% | 90% | no |

The value of this run is not the numbers, it is that the harness reproduced,
automatically and by name, three defects this project had previously found by
hand:

* **R1** — "Who runs the InfoSec review?" → *"Meera Iyer"*. That is the exact
  Anita/Meera attribution confusion documented in the README's prompt-tuning
  note: both names sit in the same chunk, one performs the review and the
  other owns the risk. The answer was correctly cited, so `cites` passed and
  `contains_any` caught it — the citation discipline is there, the attribution
  is not.
* **R2** — "Who is the CTO at Acme?" → *"Priya Sharma"* (she is the sponsor,
  and Acme has no CTO in the KB). "What penalty applies if we miss go-live?" →
  an invented answer assembled out of an unrelated outage passage. Exactly the
  absent-entity failure measured in `rag-gating.md`.
* **R5** — "thanks, that's helpful" → 110 words of invented Q&A. The other
  half of the gating fix, still open.

R4-brief passing at baseline is the pipeline working as designed: `synth
--skip-passing` drops it, and the dataset went from 308 examples to 269 across
four requirements rather than five. Nothing is trained for behaviour the stock
model already has.

**Built and exercised on this box**

- Requirement spec schema, loader, validator, scaffolder (`spec.py`)
- Deterministic check library shared by grading and generation (`checks.py`)
- Runtime and llama targets, including adapter-scale A/B (`targets.py`)
- Scorecards, baseline/candidate diff, promotion gate, markdown reports (`evaluate.py`)
- Five synthesis recipes with rejection sampling over templated surrogates (`synth.py`)
- Dataset gate including the privacy check (`validate.py`) — **run: 269 examples, all gates green**
- Job pack builder with `run.sh` + Colab notebook (`pack.py`) — **run: built**
- Adapter registry, status lifecycle, manifest integration (`core/adapters.py`)
- Runtime `--lora`, `/adapters/<file>` serving, adapter-aware install and telemetry
- Android `lora_list` init, adapter download, adapter-aware context cache (typechecks; **APK not rebuilt**)
- A real spec for the flagship agent (`specs/meeting-intelligence.yaml`): 5 requirements, 23 probes, 5 regression probes — **scored end to end against the running system**
- `selftest.py`: 34 assertions over the check library, the gate arithmetic, the registry lifecycle and the import path

**Not done, and why**

- **No trained adapter exists yet.** The gradient step needs a GPU (D6). The
  job pack is built and waiting; everything downstream of it —
  `import`/`evaluate`/`gate` — has been written against the real registry but
  has not been run on a real adapter file.
- **No CPU smoke-train.** ~4 GB free disk on this box will not hold torch plus
  a base model, so even a 135M-parameter sanity run is not possible here.
- **Android path is typechecked, not device-tested.** The APK build takes ~17
  minutes via Task Scheduler and needs a handset to verify.
- The `format_mom` recipe is implemented but unused by the current spec.

---

## 11. Running it

```powershell
# 1. draft a spec from an existing agent, then edit the placeholders
venv\Scripts\python -m finetune scaffold --agent meeting-intelligence

# 2. what does the stock model already do? (this is the "do we even need a
#    tune" question, and it is asked first on purpose)
venv\Scripts\python -m finetune baseline --spec meeting-intelligence

# 3. generate + validate + write the dataset
venv\Scripts\python -m finetune synth --spec meeting-intelligence --skip-passing
venv\Scripts\python -m finetune synth --spec meeting-intelligence --no-teacher   # faster, lower variety

# 4. build the job pack, run it on a GPU (Colab / Ubuntu / spot)
venv\Scripts\python -m finetune pack --spec meeting-intelligence

# 5. bring the adapter back
venv\Scripts\python -m finetune import --pack meeting-intelligence-behaviour-v1-adapterpack.zip ^
                                       --spec meeting-intelligence

# 6. A/B it against stock weights and apply the gate
llama\llama-server.exe -m models\Qwen3-1.7B-Q4_K_M.gguf ^
    --lora models\adapters\meeting-intelligence-behaviour-v1-f16.gguf ^
    --lora-init-without-apply --port 8302 -c 4096 --jinja --no-webui
venv\Scripts\python -m finetune evaluate --spec meeting-intelligence

# 7. if promoted: set the agent's adapter_id in the Studio and republish
```

Reports land in `finetune/reports/` as JSON and markdown; adapters and their
verdicts in `models/adapters/registry.json` (`python -m finetune adapters`).

---

## 12. Related

- `docs/rag-gating.md` — the retrieval defects several of these requirements
  exist to close, and the measurements behind them.
- `core/adapters.py` — why adapters rather than merged models, in code.
- `README.md` §0 Pending #3 — the original one-line statement of this gap.
