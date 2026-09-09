# Business Requirements Document
## Offline Enterprise Agent Runtime (PIEngine-SLM)

| Field | Value |
|---|---|
| **Document** | Business Requirements Document — as-built |
| **Product** | Offline Enterprise Agent Runtime ("Private Intelligence Engine, SLM-powered") |
| **Version** | 1.0 |
| **Date** | 9 September 2026 |
| **Status** | Working prototype, verified end-to-end on a real handset |
| **Audience** | Delivery team — as the baseline for planning improvements |
| **Repository** | https://github.com/baskaranchidambaram07-coder/PIEngine-SLM |

> **How to read this document.** This is an *as-built* BRD: it states the business
> requirement, then records what was actually implemented against it, where the
> code lives, and whether it is verified, partial, or outstanding. Anything not
> built is listed explicitly in §12 and §13 rather than being implied by silence.

---

## 1. Executive summary

Enterprises want AI assistants over their internal documents, but the moment a
prompt leaves the device, a data-residency, confidentiality and compliance
conversation begins — and it is usually the reason a promising pilot stalls.

This product removes that conversation. A **small language model (SLM) runs on
the employee's own phone.** The prompt, the retrieved document passages, and the
answer never leave the handset. There is no inference API, no token bill, and no
network dependency once an agent is installed.

Three components were built and are working:

| Component | What it does | State |
|---|---|---|
| **Agent Studio** (Journey 1) | Business teams author an agent — persona, knowledge base, tools — and publish it as a versioned bundle | Working |
| **Device Runtime** (Journey 2) | Installs bundles and runs them fully offline; also the portal phones sync from | Working |
| **Android app** | The real thing: SLM inference, vector search and tool calls executing on the handset | Working on a real device |

**The core claim is proven.** A 0.6B-parameter agent installed from the portal
answered questions grounded in its knowledge base entirely on a physical Android
phone, with no network involvement during inference, at **13 tokens/second** on a
warm cache.

**The commercial framing.** Per-seat cost after distribution is zero — no
inference spend that scales with usage. The constraint is model capability at
1–4B parameters, not infrastructure.

---

## 2. Business context and problem statement

### 2.1 The problem

| # | Problem | Consequence |
|---|---|---|
| P-1 | Cloud LLMs require sending confidential material off-device | Legal/InfoSec review blocks or delays deployment |
| P-2 | Per-token pricing scales with adoption | Success makes the business case worse |
| P-3 | Field staff work in poor-connectivity settings — client sites, travel, secure floors | A cloud assistant is unavailable exactly when it is most useful |
| P-4 | Generic assistants do not know the organisation's projects, people or decisions | Answers are plausible but not useful |
| P-5 | Existing on-device model runners (PocketPal, ChatterUI) run a model, but have no enterprise knowledge base, no tools, no governance and no distribution | Not deployable as a company product |

### 2.2 The flagship scenario — "Meeting Intelligence"

Chosen because it is the sponsor's own working context (delivery management) and
because it is document-heavy, time-critical and confidential — the exact profile
that makes cloud AI awkward.

- **Before a meeting** — brief me: agenda, last minutes, open actions, stakeholder context, live risks.
- **During** — answer factual questions instantly from the knowledge base.
- **After** — draft the minutes, the action list, the follow-up email.

### 2.3 Why now

4-bit quantised 1–4B models (Qwen3, Llama 3.2, Gemma 3) reached the point where
a mid-range phone can run useful instruction-following and tool-calling locally.
This product is an assessment of whether that is *deployable*, not just possible.

---

## 3. Business objectives and success criteria

| # | Objective | Success criterion | Result |
|---|---|---|---|
| BO-1 | Prove useful enterprise AI with **zero data egress** | An agent answers from a private KB with no network call during inference | **Met** — verified on a physical handset |
| BO-2 | Make agent creation a **business-team activity**, not an engineering one | A non-engineer can create, ground, test and publish an agent through a web UI | **Met** — Studio covers the full lifecycle |
| BO-3 | Distribute agents to phones **without an app store** | Install from an internal portal URL, including multi-GB models | **Met** — portal serves bundles, models and the APK |
| BO-4 | Give the organisation **governance visibility without surveillance** | Usage, performance and reliability visible; message content provably never collected | **Met** — metadata-only telemetry, content keys stripped in code |
| BO-5 | Establish **feasibility limits** on real hardware | Documented tokens/sec and model-size ceiling per device class | **Met** — measured; a hard device ceiling was found and diagnosed |
| BO-6 | Provide a **specialisation path** beyond prompting | A fine-tuning pipeline that improves citation discipline and tool accuracy | **Partial** — pipeline written, not executed (no GPU) |

---

## 4. Stakeholders and personas

| Persona | Role in the product | Primary need |
|---|---|---|
| **Delivery Manager / PM / Solution Architect** | End user on the handset | Fast, grounded answers about their own programme, offline |
| **Agent Author** (business analyst, practice lead) | Studio user | Create and update agents without writing code |
| **IT / Platform Operations** | Runs the servers and tunnels | Predictable deployment, stable URLs, restartable services |
| **InfoSec / Compliance** | Approves the deployment | Evidence that content never leaves the device |
| **Engineering team** | Consumers of this document | A clear as-built baseline to plan improvements against |

---

## 5. Scope

### 5.1 In scope (built)

Agent authoring; knowledge-base ingestion and retrieval; tool declaration and
execution; bundle publishing and versioning; a device runtime simulator; a
native Android runtime; fully offline inference with RAG and tools; governance
telemetry and dashboard; portal distribution including model and APK delivery;
public access via Cloudflare tunnels; interoperability export for third-party
model runners; a fine-tuning pipeline (code only).

### 5.2 Out of scope for this phase

iOS; user authentication and RBAC; multi-tenancy; bundle signing and encryption
at rest; MDM/enterprise app distribution; production APK signing; automated
tests and CI; high-availability or horizontal scaling; connectors to live
enterprise systems beyond the generic HTTP tool.

---

## 6. Solution overview

### 6.1 The two journeys

```
JOURNEY 1 — AUTHOR (server)                JOURNEY 2 — CONSUME (device)
┌──────────────────────────────┐           ┌────────────────────────────────┐
│ Agent Studio  :8100          │           │ Device Runtime  :8200          │
│  • pick an SLM               │  bundle   │  • agent store                 │
│  • write the persona         │  ───────► │  • install bundle + model      │
│  • upload docs → chunk+embed │  (zip)    │  • OFFLINE chat: RAG + tools   │
│  • declare tools             │           │  • portal for phones           │
│  • publish (versioned)       │           └───────────────┬────────────────┘
│  • governance dashboard      │                           │ APK + bundle + model
└──────────────────────────────┘                           ▼
            ▲                                  ┌────────────────────────────┐
            │ metadata only (no content)       │ Android handset            │
            └──────────────────────────────────│  • llama.rn SLM inference  │
                                               │  • SQLite vector KB        │
                                               │  • on-device tools         │
                                               │  • 100% offline after      │
                                               │    install                 │
                                               └────────────────────────────┘
```

### 6.2 The central design decision

**The knowledge base is compiled at authoring time and shipped inside the
bundle.** Documents are chunked and embedded on the server; the result is a
single portable SQLite file containing text plus raw float32 vectors. The phone
never embeds a corpus and never calls a retrieval service — it opens the file
and does brute-force cosine similarity locally.

This is what makes true offline operation possible, and it is why the same
embedding model must exist on both sides. That equivalence was explicitly
verified: **0.9998 cosine agreement** between the server's fastembed ONNX
embeddings and the phone's llama.rn GGUF embeddings on identical text.

### 6.3 Technology choices

| Layer | Choice | Business rationale |
|---|---|---|
| Server apps | Python 3.12, FastAPI, uvicorn | Small, fast to change; no build step for the UI |
| Server inference | llama.cpp `llama-server`, managed subprocess | Same engine family as the phone → behaviour parity |
| Embeddings (server) | fastembed ONNX, `bge-small-en-v1.5`, 384-dim | CPU-only, no PyTorch in the runtime |
| Embeddings (device) | same model as GGUF via llama.rn | Vector compatibility with the shipped KB |
| Knowledge store | Plain SQLite + float32 blobs | Runs everywhere, ships in a zip, no vector DB to operate |
| Mobile | React Native 0.79 + llama.rn 0.12.x + op-sqlite | One codebase; direct llama.cpp binding |
| Default model | Qwen3 1.7B Q4_K_M (Apache-2.0, ungated) | Strong tool-calling for its size; no licence gate |
| Remote access | Cloudflare Tunnel | No transfer cap — mandatory for multi-GB model downloads |

---

## 7. Business capabilities delivered

Each capability below states the requirement, what was implemented, and the
evidence. **Status** is one of: **Verified** (exercised end-to-end), **Built**
(implemented, not formally exercised), **Partial**, **Not built**.

### BC-1 — Agent authoring

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-1.1 | Create, edit, delete agents | Full CRUD over a SQLite-backed agent store; IDs auto-slugified from the name with collision suffixing | Verified |
| FR-1.2 | Author the persona | Free-text system prompt per agent, carried into the bundle verbatim | Verified |
| FR-1.3 | Choose a model | Curated catalog of 5 quantised models with size, RAM floor, context length, licence and guidance notes; the UI flags which are already downloaded | Verified |
| FR-1.4 | Control generation behaviour | Per-agent temperature, top-p and max-tokens (defaults 0.7 / 0.8 / 768) | Built |
| FR-1.5 | Control retrieval behaviour | Per-agent `top_k` (default 4) and `min_score` similarity floor (default 0.35) | Verified |
| FR-1.6 | Describe the business scenario | Scenario and description fields, published in the bundle for the store listing | Verified |

**Model catalog as shipped** — `core/catalog.py`:

| Model | Size | Min device RAM | Context | Licence |
|---|---|---|---|---|
| Qwen3 1.7B Q4_K_M *(default)* | 1.03 GB | 4 GB | 8k | Apache-2.0, ungated |
| Qwen3 0.6B Q4_K_M | 0.37 GB | 3 GB | 8k | Apache-2.0, ungated |
| Qwen3 4B Instruct 2507 Q4_K_M | 2.33 GB | 8 GB | 16k | Apache-2.0, ungated |
| Llama 3.2 3B Instruct Q4_K_M | 1.88 GB | 6 GB | 8k | Llama 3.2 Community |
| Gemma 3 4B IT Q4_K_M | 2.32 GB | 8 GB | 8k | Gemma Terms (acceptance) |

### BC-2 — Knowledge base

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-2.1 | Upload business documents | Multi-file upload; PDF via pypdf, plus TXT/MD/CSV with UTF-8 → UTF-16 → Latin-1 decoding fallback | Verified |
| FR-2.2 | Split documents sensibly | Paragraph-first recursive splitter, sentence-boundary fallback, ~1,200-character target with 150-character overlap, small paragraphs merged — sized for a 1–4B context budget | Verified |
| FR-2.3 | Make documents searchable by meaning | Every chunk embedded to a 384-dim normalised vector at upload time | Verified |
| FR-2.4 | Portable knowledge format | One SQLite file per agent: `docs`, `chunks` (text + float32 blob), `kb_meta`; the identical file ships in the bundle and is read unchanged on the phone | Verified |
| FR-2.5 | Manage the corpus | List documents with chunk counts, delete a document (cascades to its chunks) | Built |
| FR-2.6 | Test retrieval before publishing | Search endpoint returns ranked chunks with similarity scores, so the author can tune `top_k`/`min_score` against real queries | Verified |
| FR-2.7 | Add knowledge on the handset | Users attach files (≤2 MB, txt/md/csv/log) in the app; chunked and embedded **on the device** into a separate `inline.sqlite`, searched alongside the shipped KB | Built |

> **Retrieval approach.** Brute-force cosine over all chunks — no index. For KBs
> of a few thousand chunks this is single-digit milliseconds and removes an
> entire class of operational complexity. It is a deliberate trade-off with a
> known ceiling (see §13, R-6).

### BC-3 — Tools (actions, not just answers)

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-3.1 | Declare tools per agent | JSON-schema function definitions attached to the agent and published in the bundle | Verified |
| FR-3.2 | Model-initiated tool calls | Qwen3 `<tool_call>` XML convention; the tool block is injected into the system prompt and parsed out of the token stream | Verified |
| FR-3.3 | Multi-step tool use | Tool result is fed back as `<tool_response>` and generation continues — up to 4 rounds server-side, 3 on device | Verified |
| FR-3.4 | Offline-capable device actions | Built-in tools: `get_todays_meetings`, `create_action_item`, `draft_email` — all backed by device-local storage, all functional with no network | Verified |
| FR-3.5 | Enterprise system access when online | Generic `http` tool kind: URL templating from arguments, configurable method and headers, 6-second timeout | Built |
| FR-3.6 | Degrade gracefully offline | A failed `http` tool returns a structured error *plus a hint instructing the model to explain the limitation and offer an offline alternative* — the user gets an explanation, not a stack trace | Built |

> **Deliberate safety property.** `draft_email` writes to a device outbox with
> status `"draft — awaiting user review"` and the note *"Never auto-sent."* The
> agent drafts; the human sends. Autonomous outbound communication was excluded
> by design.

### BC-4 — Publishing and distribution

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-4.1 | Publish an immutable, versioned artefact | Zip bundle `<agent>-v<n>.zip` containing `manifest.json` + `kb.sqlite`; version auto-increments on every publish | Verified |
| FR-4.2 | Self-describing bundles | Manifest schema `slm-agent-bundle/1` carries persona, generation settings, RAG settings, tools, full model descriptor (including exact byte size and download URL) and the embedder identity | Verified |
| FR-4.3 | Central catalog | `registry.json` holds one current entry per agent; served to both the runtime and phones | Verified |
| FR-4.4 | Distribute to devices | Portal endpoints: `/api/published` (catalog), `/bundles/<file>`, `/models/<file>`, `/apk` | Verified |
| FR-4.5 | Resilient model delivery | Phone tries the Hugging Face CDN first, then the portal's own cached copy — covers networks that block large CDN downloads | Built |
| FR-4.6 | Path-traversal protection | Bundle and model file serving resolve the path and verify the parent directory before responding | Built |

### BC-5 — Device Runtime (web simulator, Journey 2)

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-5.1 | Browse and install published agents | Agent store showing available / installed / update-available state per agent | Verified |
| FR-5.2 | Acquire models on demand | Background threaded download with live progress, written to a `.part` file and renamed on completion | Verified |
| FR-5.3 | Chat offline with an installed agent | Full pipeline: retrieve → build prompt → stream → tool rounds, streamed to the browser over SSE | Verified |
| FR-5.4 | Show the workings | Retrieved sources with document name, chunk index and score; tool calls with arguments and results; token/sec statistics per turn | Verified |
| FR-5.5 | Manage one model at a time | `llama-server` subprocess started per model and restarted when another agent needs a different GGUF; health-gated startup with a 120-second ceiling | Verified |
| FR-5.6 | Keep the UI honest | Chat is refused with a clear message until the model is actually present on device | Built |

### BC-6 — Native Android runtime

This is the deliverable that proves the concept, since everything executes on
the handset.

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-6.1 | Install agents from the portal | Bundle downloaded and unzipped into app storage; models cached and shared across agents | Verified |
| FR-6.2 | Run the SLM on the handset | llama.rn (llama.cpp binding) with CPU-only parameters proven on real Android hardware: `n_gpu_layers: 0`, `no_gpu_devices: true`, flash-attention off, unified KV, single slot | Verified |
| FR-6.3 | Retrieve on the handset | op-sqlite reads the bundle's `kb.sqlite`; cosine computed in JS over the float32 blobs | Verified |
| FR-6.4 | Embed on the handset | `bge-small-en-v1.5-q8_0.gguf` (~35 MB) via llama.rn, in a separate context | Verified |
| FR-6.5 | Execute tools on the handset | Full parity with the server dispatcher, backed by AsyncStorage | Built |
| FR-6.6 | Survive real-world networks | Large models fetched through Android's Download Manager (own network stack, notification progress, resumable); small files buffered in memory to avoid a known file-stream failure; two retries per source; `Accept-Encoding: identity` to defeat a false "Download interrupted" on gzipped responses | Verified |
| FR-6.7 | Never run a corrupt model | Manifest carries the model's exact byte size; a mismatched file is deleted with instructions to reinstall; the Download-Manager copy is also size-verified | Verified |
| FR-6.8 | Fail safely on under-powered devices | A model that fails to initialise is recorded in persistent storage and never re-attempted — repeated failed native initialisations were found to hard-crash the app uncatchably. The user is told which model failed and which to use instead | Verified |
| FR-6.9 | Be diagnosable in the field | In-app log with a Share action, including the native llama.cpp log tail — the only way to see the real error behind generic native rejections | Verified |
| FR-6.10 | Survive a changed portal address | The app re-resolves its portal URL from a stable discovery endpoint when its saved address stops answering | Built |

### BC-7 — Offline conversational experience

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-7.1 | Ground answers in the knowledge base | Retrieved passages prefixed to the user turn as a labelled context block with document names and chunk indices | Verified |
| FR-7.2 | Show provenance | Every answer can display which document and chunk it drew on, with similarity scores | Verified |
| FR-7.3 | Hide model scaffolding | Streaming tag filter strips Qwen3 `<think>` blocks and extracts `<tool_call>` payloads, with hold-back buffering so a tag split across token boundaries is never leaked to the UI | Verified |
| FR-7.4 | Keep responses fast | Thinking mode disabled by appending `/no_think` to the system prompt | Verified |
| FR-7.5 | Multi-turn conversation | Full history maintained; KV-cache reuse makes later turns substantially faster (measured 4.1 → 13 tok/s) | Verified |
| FR-7.6 | Report performance honestly | Per-turn statistics taken from the engine's own timings, separating prompt processing from generation | Verified |

### BC-8 — Governance and privacy

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-8.1 | Usage visibility | Dashboard KPIs: chats, tokens generated, installs, active devices, average tokens/sec, average model-load time, reliability %, total bytes served | Verified |
| FR-8.2 | Breakdowns for decision-making | Per-agent usage, per-model performance (tok/s, load time, errors), daily activity trend, device inventory, live event log | Verified |
| FR-8.3 | Fleet reach | Both the web runtime and every Android handset report to one shared store | Verified |
| FR-8.4 | **Content must never be collected** | Only metadata columns exist in the schema. The write path defensively strips any key named `query`, `prompt`, `message(s)`, `answer`, `text` or `content` even if a caller passes one by mistake | Verified |
| FR-8.5 | Telemetry must never break the product | Best-effort writes: WAL journaling, busy timeout, retry with backoff on lock contention, and silent drop on non-transient failure — the request path never raises | Built |

> **This is the InfoSec conversation, settled in code.** The privacy promise is
> not a policy statement; it is enforced by a schema that has nowhere to put
> content and a write path that discards it. `core/telemetry.py` is the single
> artefact to show a reviewer.

### BC-9 — Interoperability export

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-9.1 | Run an agent in third-party apps | Export to PocketPal AI (persona + knowledge digest as a paste-ready prompt) and ChatterUI/Maid (SillyTavern `chara_card_v2` JSON) | Built |
| FR-9.2 | Carry knowledge without RAG | The KB is compressed into a bounded digest — 8,000-character budget, 700 characters per chunk, grouped by document — and baked into the prompt | Built |
| FR-9.3 | Easy handset transfer | Export page with QR codes that honour proxy headers so the encoded URL is the public HTTPS one | Built |

### BC-10 — Remote access and operations

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-10.1 | Reach both journeys from outside | Cloudflare tunnels; no transfer cap, which is mandatory because ngrok's free 1 GB/month cap fails part-way through a single 1.03 GB model download | Verified |
| FR-10.2 | Permanent addresses | Portable named-tunnel setup: one tunnel serves both journeys at fixed hostnames; config re-renders itself for whatever machine it is copied to | Built, **not activated** — requires a domain |
| FR-10.3 | Survive reboots | All services run as SYSTEM scheduled tasks with auto-start | Verified |
| FR-10.4 | Resilience to address rotation | Studio publishes a discovery endpoint reporting the portal's current address; the app re-resolves automatically | Verified (server side) |
| FR-10.5 | Resumable large downloads | HTTP range requests pass through the tunnel — confirmed `206 Partial Content` | Verified |

### BC-11 — Model specialisation pipeline

| ID | Requirement | As implemented | Status |
|---|---|---|---|
| FR-11.1 | Build training data from a live agent | Dataset builder combines hand-written behaviour seeds (citation discipline, refusal, scope control), KB-grounded pairs generated from the agent's own chunks, and optional teacher distillation via any OpenAI-compatible endpoint | Built (runs without a GPU) |
| FR-11.2 | Train a scenario adapter | QLoRA training script | Built, **never executed** |
| FR-11.3 | Ship the result | Merge-and-export to GGUF for the catalog | Built, **never executed** |

---

## 8. Non-functional characteristics (measured)

### 8.1 Performance

| Environment | Model | Result | Conditions |
|---|---|---|---|
| Server (4 vCPU, 16 GB, **no GPU**) | Qwen3 1.7B Q4_K_M | **~8 tok/s** generation, **~49 tok/s** prompt processing | Through the public tunnel, 1,487-token prompt |
| Physical Android handset | Qwen3 0.6B Q4_K_M | **4.1 tok/s** first turn (cold), **13 tok/s** second turn (warm KV cache) | 140 and 390 tokens respectively |
| Physical Android handset | Model load | **1.7 seconds** | Qwen3 0.6B from app storage |
| Embedding agreement | bge-small-en-v1.5 | **0.9998 cosine** | fastembed ONNX vs llama.rn GGUF, identical text |

### 8.2 Capacity and footprint

| Item | Value |
|---|---|
| Agent bundle | ~25 KB for a 5-document, 12-chunk knowledge base |
| Chat model on device | 0.37–2.33 GB depending on catalog choice |
| Embedder on device | ~35 MB, shared by every agent |
| Android APK | ~88.8 MB |
| Model storage | Shared across agents — a second agent on the same model downloads nothing |
| Retrieval | Brute-force; comfortable to a few thousand chunks |

### 8.3 Privacy and security posture

| Property | State |
|---|---|
| Inference data egress | **None.** Prompts, passages and answers never leave the device |
| Telemetry content | **Metadata only**, enforced by schema and by a stripping write path |
| Outbound communication | Never autonomous — email is drafted for human review |
| KB integrity | Bundles carry exact byte sizes; corrupt models are deleted, not run |
| Portal authentication | **None** — see §13, R-1 |
| Bundle signing / encryption at rest | **Not implemented** — see §13, R-2 |
| APK signing | Stock React Native **debug** keystore — see §13, R-3 |

---

## 9. Interfaces

### 9.1 Agent Studio (port 8100)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/catalog` | Model catalog with download state |
| GET/POST | `/api/agents` | List / create agents |
| GET/PUT/DELETE | `/api/agents/{id}` | Read / update / delete an agent |
| POST/GET | `/api/agents/{id}/docs` | Upload documents / list them |
| DELETE | `/api/agents/{id}/docs/{doc_id}` | Remove a document |
| POST | `/api/agents/{id}/search` | Test retrieval |
| POST | `/api/agents/{id}/publish` | Publish a new version |
| GET | `/api/published` · `/bundles/{name}` | Catalog · bundle download |
| GET | `/api/governance/overview` · `/events` | Dashboard data · event log |
| GET | `/api/portal-url` | Portal address discovery |
| GET | `/welcome` | Platform landing page |

### 9.2 Device Runtime / portal (port 8200)

| Method | Endpoint | Purpose |
|---|---|---|
| GET/DELETE | `/api/installed[/{id}]` | Installed agents / uninstall |
| GET | `/api/store` | Store view with install state |
| POST | `/api/install` | Install a bundle and trigger the model download |
| POST | `/api/chat` | Streaming chat (SSE) |
| GET | `/api/published` · `/bundles/{n}` · `/models/{n}` · `/apk` | The full phone-facing surface |
| POST | `/api/telemetry` | Metadata ingest from handsets |
| GET | `/api/device` · `/api/llm/status` | Device data · engine state |
| GET | `/export/{id}` · `/api/export/{id}/prompt.txt` · `/card.json` | Third-party export |

---

## 10. Assumptions

| # | Assumption | Impact if wrong |
|---|---|---|
| A-1 | Target handsets have ≥4 GB RAM with ~2–3 GB usable | Model choice must drop to 0.6B, reducing answer quality |
| A-2 | Knowledge bases are department-sized (hundreds to a few thousand chunks) | Brute-force retrieval must be replaced with an index |
| A-3 | English-language content | The embedder is English-specific; multilingual needs a different model |
| A-4 | Devices are company-managed or trusted | There is no authentication on the portal today |
| A-5 | A 1–4B model is adequate for grounded Q&A and summarisation | Larger models will not fit on the handset; scope must narrow |

---

## 11. Constraints

| # | Constraint | Consequence |
|---|---|---|
| C-1 | Host server has **no GPU** | Fine-tuning cannot run here; it needs an external GPU box |
| C-2 | Handsets have a hard memory ceiling for model initialisation | The test device could not run 1.7B at all — see §13, R-4 |
| C-3 | ngrok free tier caps transfer at ~1 GB/month | Unusable for model distribution; Cloudflare is required |
| C-4 | Cloudflare *quick* tunnels mint a new hostname on every restart | Permanent URLs require a customer-owned domain |
| C-5 | The build server blocks Java NIO pipes in interactive shells | Gradle builds must run through Task Scheduler |
| C-6 | iOS builds require macOS hardware | No iOS deliverable in this phase |

---

## 12. Explicitly not built

State this plainly when presenting: the following were **never implemented**,
and any plan should treat them as new work rather than hardening.

1. User authentication, authorisation or role-based access — in either web app or the portal.
2. Multi-tenancy or per-department isolation.
3. Bundle signing, bundle encryption, or encryption of the KB at rest on the device.
4. MDM / enterprise app-store distribution.
5. Production APK signing.
6. Automated tests of any kind, and any CI pipeline.
7. Delta/incremental knowledge-base sync — updating an agent re-downloads the bundle.
8. An evaluation harness gating publication on answer quality.
9. iOS application.
10. Live connectors (Jira, Confluence, Outlook, SharePoint) beyond the generic HTTP tool.

---

## 13. Risks and known limitations

| # | Risk / limitation | Impact | Current state |
|---|---|---|---|
| R-1 | **The portal is unauthenticated.** Anyone with the URL can list agents and download bundles, models and the APK | Confidential KBs are exposed to anyone holding the link | Open — mitigated only by URL obscurity |
| R-2 | **Bundles are unsigned and unencrypted.** A KB on a lost device is readable | Data-at-rest exposure | Open |
| R-3 | **Release APK uses the public debug keystore** | Not distributable through any managed channel; no update-integrity guarantee | Open |
| R-4 | **Qwen3-1.7B fails to initialise on the test handset.** Root-caused: the compute buffer is driven by `n_ubatch`, not `n_ctx`, so shrinking context alone never reduced peak memory; the device OOMs after weights load | The default catalog model does not run on that class of device; 0.6B is its ceiling | Diagnosed, mitigated by the block-and-guide safety gate |
| R-5 | **No automated tests.** Every verification to date has been manual | Regressions will not be caught | Open |
| R-6 | **Retrieval is brute-force** | Degrades past a few thousand chunks | Known trade-off; acceptable at current scale |
| R-7 | **Answer quality is unmeasured.** No accuracy, groundedness or hallucination benchmark exists | Quality claims rest on demonstration, not measurement | Open |
| R-8 | **Public URLs rotate** without a customer domain | Handsets need reconfiguration on restart | Mitigated by auto-discovery; fixed permanently by the named tunnel |
| R-9 | **Single-node deployment.** One server, one tunnel, no redundancy | Any outage stops distribution (though installed agents keep working offline) | Accepted for a prototype |
| R-10 | **English-only embeddings** | Non-English documents retrieve poorly | Open |

---

## 14. Recommended improvement themes

Offered as input to the team's planning, in the order the evidence supports.

**Theme 1 — Make it deployable (security).** R-1, R-2 and R-3 together are what
stand between this prototype and a pilot with real documents: portal
authentication, bundle signing plus encryption at rest, and a production signing
key. Nothing else on this list matters if a security review stops the pilot.

**Theme 2 — Make quality measurable.** R-7 is the strategic gap. An evaluation
set with groundedness and refusal checks would turn "it works in the demo" into
a number that can be tracked across model and prompt changes — and it is the
prerequisite for judging whether fine-tuning (BC-11) actually earns its cost.

**Theme 3 — Close the device-capability gap.** R-4 has a clean root cause. Decide
deliberately between defaulting the fleet to 0.6B, tuning batch parameters
further, or setting a minimum device specification. Publishing a supported-device
matrix would make this a procurement decision rather than a support surprise.

**Theme 4 — Reduce operational friction.** Activating the named tunnel (FR-10.2)
retires the whole class of rotating-URL problems. Delta KB sync would cut
update cost for large knowledge bases.

**Theme 5 — Broaden reach.** iOS, then live enterprise connectors, then
multilingual embeddings — each is significant new work and should be sequenced
after Themes 1 and 2.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **SLM** | Small Language Model — 0.6B to 4B parameters, sized to run on a phone |
| **GGUF** | The llama.cpp model file format |
| **Q4_K_M** | 4-bit quantisation — roughly a quarter of the memory at a modest quality cost |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant passages, then answer from them |
| **Chunk** | A ~1,200-character passage; the unit of retrieval |
| **Embedding** | A 384-number vector representing a passage's meaning; similarity is a dot product |
| **Bundle** | The publishable agent artefact: manifest + knowledge base, versioned |
| **Manifest** | The JSON inside a bundle describing persona, model, retrieval settings and tools |
| **Tool call** | A model-requested function invocation, executed on the device |
| **Tunnel** | A secure outbound connection making a local server reachable on the public internet |
| **Journey 1 / 2** | The authoring experience / the device experience |

---

## 16. Appendix — where the capability lives

| Capability | Code |
|---|---|
| Agent authoring, publishing, governance API, landing page | `studio/app.py`, `studio/static/` |
| Device runtime, portal, chat orchestration, telemetry ingest | `runtime/app.py`, `runtime/static/` |
| Server inference, streaming, tag filtering | `runtime/llm.py` |
| Tool dispatch (server) | `runtime/tools.py` |
| Third-party export | `runtime/export.py` |
| Chunking · embeddings · vector store · catalog · telemetry | `core/` |
| Android app | `android_app/AgentRuntime/src/`, `App.tsx` |
| On-device inference and prompt building | `src/llm.ts` |
| On-device retrieval | `src/kb.ts` |
| On-device tools | `src/tools.ts` |
| Install, download resilience, portal discovery | `src/portal.ts` |
| Permanent public URLs | `cloudflare/` |
| Fine-tuning pipeline | `finetune/` |
| Demo seeding | `scripts/seed_demo.py`, `sample_docs/` |

**Operational detail** — startup, health checks, restart procedures, build
instructions and troubleshooting are in the root [`README.md`](../README.md).
Tunnel setup is in [`cloudflare/README.md`](../cloudflare/README.md). Android
build specifics are in [`android_app/BUILD.md`](../android_app/BUILD.md).
