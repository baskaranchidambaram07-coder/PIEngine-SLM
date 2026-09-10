# Offline Enterprise Agent Runtime — SLM Prototype

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-uvicorn-009688?logo=fastapi&logoColor=white)
![React Native 0.79](https://img.shields.io/badge/React%20Native-0.79-61DAFB?logo=react&logoColor=black)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Qwen3%20GGUF-FF6F00)
![Cloudflare Tunnel](https://img.shields.io/badge/Cloudflare-named%20tunnel-F38020?logo=cloudflare&logoColor=white)
![Inference on-device](https://img.shields.io/badge/inference-100%25%20on--device-2EA043)
![Status: prototype](https://img.shields.io/badge/status-working%20prototype-DAA520)

Task-specific AI agents that run **entirely on an employee's device** — no cloud
LLM, no data leaving the phone. Built around the "Meeting Intelligence" scenario
for delivery managers / PMs / solution architects.

Three deliverables in this repo:

| # | Deliverable | Where | Public URL |
|---|---|---|---|
| **Journey 1** | Agent Studio — create/tune agents, KB, tools, publish bundles | `studio/` on port **8100** | `https://<your-static-domain>.ngrok-free.dev` *(static)* — or `https://studio.<your-domain>` via §1c |
| **Journey 2** | Device Runtime — agent store, offline RAG chat (web simulator) + phone-facing portal | `runtime/` on port **8200** | `https://<random>.trycloudflare.com` *(rotates per restart)* — or `https://portal.<your-domain>` via §1c |
| **Android app** | Native runtime: installs agents from the portal, runs SLM + RAG + tools on the handset | `android_app/AgentRuntime/` | served at `<journey-2-url>/apk` |

---

## 0. Scope and status

### What this set out to prove

That a **small language model on the employee's own handset** can do useful,
task-specific enterprise work — with a maker journey to build the agents and a
device journey to run them — while no prompt, document or answer ever leaves
the device. Concretely:

1. Author an agent (persona + knowledge base + tools) in a web Studio and
   publish it as a portable bundle.
2. Install that bundle on a phone and run it **fully offline**: SLM inference,
   RAG retrieval and tool calls, all on the handset.
3. Give the organisation governance visibility without collecting content.
4. Distribute it over a stable public URL, including multi-GB model downloads.

### Done and verified

| Area | State | Evidence |
|---|---|---|
| Journey 1 — Agent Studio | working | authoring, PDF/MD/TXT chunk + embed to SQLite, tool declarations, versioned bundle publish |
| Journey 2 — Device Runtime | working | agent store, offline RAG chat over a managed `llama-server`, portal serving bundles/models/APK |
| Android app | working on-device | Qwen3-0.6B end-to-end on a real handset: install → byte-exact model verify → RAG (12 chunks, 4 hits) → **4.1 tok/s** first turn (cold), **13 tok/s** warm |
| Server-side inference | measured | Qwen3-1.7B Q4_K_M: **~8 tok/s** generation, **~49 tok/s** prefill (4 vCPU, no GPU) |
| Embedding parity | verified | fastembed ONNX (Studio) vs llama.rn GGUF (phone): **0.9998** cosine on identical text |
| Governance dashboard | working | KPIs, per-agent/per-model breakdowns, device list, live event log — metadata only, content stripped in code |
| Landing page + light theme | working | `/welcome`, AI-Native light theme across both SPAs |
| Portal URL auto-discovery | working | `GET /api/portal-url`; the app re-resolves and persists a new portal URL when its saved one stops answering |
| Cloudflare named tunnel | implemented, not activated | `cloudflare/` renders and passes `cloudflared tunnel ingress validate`; **needs a domain to go live** |
| Fine-tuning pipeline | built, GPU stage pending | requirement spec (5 requirements, 23 probes) scored against the running system — **stock model 78%**, and the harness reproduced three previously hand-found defects by name; 269 synthetic examples passing leakage/duplication/balance/length/**privacy** gates → job pack. Adapters ship as ~30 MB LoRA on the shared base, gated on a scorecard before publish |
| Public repo | done | source only (~2.7 MB); weights, toolchains and runtime state excluded |

### Pending

| # | Item | Blocked by |
|---|---|---|
| 1 | **Activate the named tunnel** — permanent `studio.` / `portal.` hostnames | a Cloudflare account + a domain; then `cloudflare\setup.ps1` (§1c) |
| 2 | **Qwen3-1.7B on the test handset** — init OOMs after weights load; 0.6B is that device's ceiling | root cause found (compute buffer driven by `n_ubatch`); needs a device with more headroom, or further batch tuning |
| 3 | **Train the first adapter** — the pipeline around it is built and exercised (spec → synthesis → dataset gate → job pack, and the import/A-B/promotion path behind it); only the gradient step is missing | no GPU and ~4 GB free disk on this server; the job pack is built and runs unchanged on a Colab T4 or any 8 GB NVIDIA box ([`docs/finetuning.md`](docs/finetuning.md)) |
| 4 | **iOS build** of the same RN app | needs a Mac or macOS CI runner |
| 5 | **Production APK signing** | release currently uses the stock RN *debug* keystore — must be replaced before real distribution |
| 6 | **Scale-out retrieval** — sqlite-vec + hybrid BM25/vector past ~10k chunks | not started |
| 7 | **Bundle signing + encryption at rest, MDM distribution** | not started |
| 8 | **Automated tests / CI** | none yet; verification to date is manual and end-to-end |

---

## 0a. Competitive landscape

### Luxand LLM SDK — https://www.luxand.com/llm-sdk/

The closest thing to us in market positioning, and the most useful reference
point we have. Same core thesis: a local model inside the app, nothing sent
anywhere, no telemetry, tool-calling included.

**It is a different kind of thing, though.** Luxand sells a *developer SDK* —
a library you embed to get local inference. We built a *product*: the authoring
console, the knowledge pipeline, the distribution channel and the governance
view that sit on top of inference. Their layer is roughly the layer we get from
llama.cpp / llama.rn.

| | **Luxand LLM SDK** | **This project** |
|---|---|---|
| Category | Commercial SDK/library | End-to-end internal product |
| Platforms | Android, iOS, Windows, Linux, macOS, ARM/Pi | Android + Windows server (no iOS) |
| Backends | CPU, Metal, CUDA | CPU only |
| Languages | C, C++, Kotlin, Swift, .NET, Python (one C ABI) | Python + TypeScript/RN |
| Streaming chat | Yes | Yes |
| Tool calling | Yes, OpenAI-style with typed accessors | Yes, Qwen3 `<tool_call>` convention |
| Schema-constrained JSON | Yes, grammar-constrained | **No** |
| Vision / multimodal | Yes | **No** |
| Concurrent sessions | Yes, sharing one pass over the weights | One model, one session |
| OpenAI-compatible local server | Yes, in-process | Only via `llama-server` on the host |
| **Knowledge base / RAG** | **Not offered** | **Yes** — chunk, embed, ship, retrieve on-device |
| **Agent authoring for non-engineers** | **Not offered** | **Yes** — the Studio |
| **Versioned distribution to a fleet** | **Not offered** | **Yes** — bundles, portal, install/update |
| **Governance dashboard** | **Not offered** | **Yes** — metadata-only telemetry |
| Model guidance | Curated list, benchmarked on BFCL V4 agentic subsets | Curated list, **unbenchmarked** |
| Licensing | Free for dev/non-commercial; $990/yr (startup) or $5,990/yr (business) per product; Enterprise custom. Device-side only — not for server deployment | Internal; no licence declared yet |

### What this means for us

**They are more plausibly a component than a rival.** Their SDK could replace
our llama.rn inference layer and would hand us three things we do not have:
**iOS**, **vision**, and **GPU/Metal backends** — for a per-product annual fee,
with no per-user or per-token cost. Worth a build-vs-buy evaluation before we
spend engineering time on an iOS runtime ourselves. The counter-argument is
lock-in on the layer where open alternatives are strongest, and their licence
explicitly excludes server-side use, which our Journey-2 web runtime is.

**Where we are genuinely differentiated:** everything above the model. An SDK
does not give a delivery manager a way to build an agent, does not compile a
knowledge base into a portable artefact, does not version and distribute it to
a fleet, and does not answer "who used what, and how well is it working". That
stack is our product, and nothing on their page competes with it.

**Where they expose a real gap in our work:** they publish agentic benchmark
scores (BFCL V4 subsets, ~40,000 tasks by their own account) and pick
recommended models per hardware class from measurement. We pick models from
reasoning and vendor notes — see §0 Pending #8 and the BRD's R-7. Their
methodology is a good template for the evaluation harness we are missing.

**Their model table also suggests our catalog is dated.** Their
best-score-per-gigabyte pick for mass-market phones is a ~2.7 GB 4-bit model
scoring 67.0% agentic, where our default is Qwen3 1.7B. A catalog refresh
against current small models is worth a spike.

> Figures above are as published by Luxand (read September 2026), including
> benchmark numbers they describe as their own internal runs on a BFCL V4
> subset. Nothing here has been independently verified by us.

### Open-source reference implementations — [shubham0204](https://github.com/shubham0204)

A different category again: Apache-2.0 projects by one developer (Shubham
Panchal, 46 repos, ~465 followers) that are the closest **technical** analogues
to our device layer. They are not commercial products and no organisation would
buy them instead of this platform — but they solve several of the same problems
in the open, and three of them speak directly to items on our pending list.

| Repo | Stars | What it is | Relevance to us |
|---|---|---|---|
| [SmolChat-Android](https://github.com/shubham0204/SmolChat-Android) | 891 | Consumer chat app for any GGUF model on Android — Kotlin + llama.cpp over JNI, on Google Play | Direct analogue of our Android runtime |
| [OnDevice-RAG-Android](https://github.com/shubham0204/OnDevice-RAG-Android) | 198 | On-device RAG over PDF/DOCX — splitter, embeddings and vector DB all local; LLM local **or** Gemini cloud | Direct analogue of our KB/RAG path |
| [OnDevice-Face-Recognition-Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android) | 183 | Face recognition using **ObjectBox's embedded vector database** on Android | A proven on-device vector index — our R-6 |
| [Sentence-Embeddings-Android](https://github.com/shubham0204/Sentence-Embeddings-Android) | 72 | sentence-transformers embeddings on Android via ONNX Runtime — **supports `bge-small-en`** | An alternative to how we embed today |

*(The remaining repos — CLIP, Segment-Anything, Depth-Anything, MiDaS, age/gender
estimation — are computer vision and not relevant here.)*

**SmolChat is a bring-your-own-model consumer app.** The user picks a GGUF,
writes their own system prompt, tunes temperature/min-p, and chats; "tasks"
save a reusable prompt. There is no organisation in the picture: no authored
agent, no knowledge base compiled by someone else, no versioned distribution,
no fleet governance. Its roadmap lists integrating on-device RAG and trying
Vulkan for GPU inference — i.e. it is heading toward capabilities we already
have, from the consumer end.

Where it is ahead of us: a native Kotlin/JNI binding (thinner than our React
Native bridge), Play Store distribution, and 891 stars' worth of hardening
across far more device models than we have tested on.

### What this group means for us

**It settles a positioning question.** A polished, free, open-source on-device
chat app already exists on the Play Store. Our Android app therefore has no
standalone consumer value — **our value is the enterprise stack around it**:
authoring, KB compilation, versioned distribution, governance. That is the
story to tell, and it is the part none of these repos attempt.

**Three concrete engineering leads**, all Apache-2.0 and therefore reusable
with attribution:

1. **`Sentence-Embeddings-Android` supports the exact embedding model we use
   (`bge-small-en`), via ONNX Runtime rather than a second llama.cpp context.**
   We currently load the embedder as its own llama.rn context and release it
   before the chat model precisely because memory is tight. An ONNX embedder
   could cut that peak — which is the direct cause of our Qwen3-1.7B failure
   (§0 Pending #2). Worth a spike before we write off 1.7B on mid-range devices.
2. **ObjectBox as an embedded vector database** is a ready answer to brute-force
   retrieval when knowledge bases outgrow a few thousand chunks (BRD R-6).
3. **Their Vulkan investigation** is the same GPU-offload question we parked
   after finding llama.rn's OpenCL/Adreno auto-selection unreliable.

**One design contrast worth keeping.** `OnDevice-RAG-Android` indexes documents
*on the phone* — the user adds a PDF and the device does the chunking and
embedding. We compile the knowledge base *on the server* and ship it inside the
bundle. Ours suits an organisation distributing curated, consistent knowledge to
many handsets; theirs suits personal documents. We already support the second
pattern as a secondary path (inline KB upload, BRD FR-2.7), so the two models
coexist rather than compete.

> Stars, licences and descriptions read from GitHub in September 2026.

---

## 1. Starting everything (day-to-day operations)

All services run as **Windows scheduled tasks under SYSTEM** — they auto-start
on server boot and survive terminal/session closes. Normally you start nothing.

| Task name | What it runs | Port / output |
|---|---|---|
| `slm-studio` | Journey 1 web app (`scripts/start_studio.cmd`) | 8100 |
| `slm-runtime` | Journey 2 web app (`scripts/start_runtime.cmd`) | 8200 |
| `slm-studio-tunnel` | ngrok → 8100 (`scripts/start_studio_tunnel.cmd`) | static URL above |
| `slm-tunnel` | cloudflared quick tunnel → 8200 (`scripts/start_tunnel.cmd`) | URL in `C:\slm\cloudflared.log` |
| `slm-cf-tunnel` | cloudflared **named** tunnel → 8100 + 8200 (`cloudflare/run.cmd`) | permanent URLs — see §1c |
| `slm-gradle-build` | Android APK build (`android_app/run_build.cmd`) | on demand only |

`slm-cf-tunnel` replaces both tunnel tasks above once set up. Until then the
quick tunnel + ngrok pair keeps working unchanged.

**Health check (run anytime):**

```powershell
curl.exe -s -o NUL -w "studio:%{http_code} "  http://127.0.0.1:8100/api/agents
curl.exe -s -o NUL -w "runtime:%{http_code} " http://127.0.0.1:8200/api/installed
```

**Restart a service** (e.g. after editing its Python code):

```powershell
# kill the listener, then re-run the task
Get-NetTCPConnection -LocalPort 8200 -State Listen | % { Stop-Process -Id $_.OwningProcess -Force }
schtasks /Run /TN "slm-runtime"
```

**Find the current Journey-2 public URL** (needed after a tunnel restart / server reboot):

```powershell
Select-String -Path C:\slm\cloudflared.log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1
```

> ⚠ Cloudflare *quick tunnels* mint a **new random URL on every start**. After a
> reboot or tunnel restart: get the new URL with the command above, then update
> it on each phone via **⚙ → Portal URL → Save & test** in the Android app.
> The ngrok URL for Journey 1 is a static domain and never changes.
>
> Two things soften this, in order of preference: set up the **named tunnel**
> (§1c) and the URL becomes permanent; failing that, app v2.6+ **re-resolves
> the portal URL by itself** from `GET /api/portal-url` on the Studio's stable
> domain, so phones recover from a rotation without anyone retyping anything.

**Logs:** `C:\slm\logs\studio.log`, `logs\runtime.log`, `logs\ngrok.log`,
`C:\slm\cloudflared.log`, `logs\cloudflare-tunnel.log` (named tunnel),
Android build → `C:\slm\android_app\build.log`.

---

## 1a. Landing page & theme

Both web apps ship a **premium light theme** by default (AI-Native design
language: neutral surfaces + AI purple `#7C3AED`, Plus Jakarta Sans, soft
depth shadows). A marketing **landing page** for the platform ("Enclave") is
served by the Studio at **`/welcome`** (also `/landing`): hero with a live
streaming-chat demo, AI-capability feature cards, integration band, two-journey
explainer, and a prominent *Try Now* CTA into the Studio. The landing page's
"live device demo" link is injected server-side with the current Device Runtime
tunnel URL, so it survives tunnel rotation. Reachable in the Studio via the
**✦ Platform home** sidebar link.

## 1b. Governance dashboard (in Journey 1)

The Studio has a **Governance** tab showing usage, performance and reliability
across Journey 2 and every installed Android device:

- **KPIs**: chats, tokens generated, installs, active devices, avg tokens/sec,
  model-load time, reliability %, and total data (bundles + models + APK) served.
- **Activity chart** (chats vs installs over time), **usage by agent**,
  **performance by model** (tok/s, load time, errors), **device list**, and a
  live **event log** streamed from Journey 2 and the phones.

**Privacy by design — metadata only.** The on-device promise is that message
text, KB passages, queries, and answers never leave the handset. Telemetry
therefore records *only* usage metadata (which agent/model, token counts,
tok/s, load times, KB hit counts, install/download events, and system error
strings). `core/telemetry.py` defensively strips any content-bearing keys.

Data flow: the runtime (Journey 2) records its own portal serves + web chats,
and the Android app POSTs usage events to `<portal>/api/telemetry`; both write
to the shared `telemetry.db`, which the Studio reads via
`GET /api/governance/overview` and `…/events`.

## 1c. Stable public URLs — Cloudflare named tunnel

Everything needed lives in **`cloudflare/`**, so the whole setup travels with
the project folder to any machine. Full detail: [`cloudflare/README.md`](cloudflare/README.md).

| | quick tunnel (today) | ngrok free | **named tunnel** |
|---|---|---|---|
| URL stability | new random URL per restart | static | **permanent** |
| Transfer cap | none | ~1 GB/month | **none** |
| Multi-GB GGUF downloads | ok | **fails mid-download** | **ok** |
| Needs a domain | no | no | **yes** |

Requires a Cloudflare account plus a domain added to it (both free plan; the
domain itself is the only cost). Then:

```powershell
cd C:\slm\cloudflare
.\setup.ps1 -Domain your-domain.com    # login, create tunnel, DNS, config
.\install_task.ps1                     # boot-start SYSTEM task (as Administrator)
```

Result — these never change again:

```
https://studio.your-domain.com   ->  127.0.0.1:8100   (Journey 1)
https://portal.your-domain.com   ->  127.0.0.1:8200   (Journey 2 — set this in the app)
```

**On another machine:** copy the project folder (including
`cloudflare\<tunnel-id>.json`, which is the tunnel's secret) and run
`cloudflare\run.cmd`. No second login, no new DNS, same URLs — so no handset
needs reconfiguring. `run.ps1` re-renders `config.yml` with the new machine's
paths at every start.

Once set up, `studio/app.py` picks the stable hostnames up automatically from
`cloudflare/tunnel.json` (landing page links and `/api/portal-url` both switch
over) — and add `https://studio.your-domain.com` to the front of
`DISCOVERY_URLS` in `android_app/AgentRuntime/src/config.local.ts`.

### Deployment-specific files (not in git)

Real hostnames of a running deployment are kept out of this public repo. After
cloning, copy the templates and fill in your own addresses:

```powershell
copy android_app\AgentRuntime\src\config.local.example.ts android_app\AgentRuntime\src\config.local.ts
copy scripts\start_studio_tunnel.cmd.example scripts\start_studio_tunnel.cmd
```

`config.local.ts` supplies `DEFAULT_PORTAL` and `DISCOVERY_URLS` — **the app
will not bundle without it.** `start_studio_tunnel.cmd` is only needed if you
use ngrok for Journey 1 instead of the named tunnel.

---

## 2. Journey 1 — Agent Studio (port 8100, ngrok)

**Purpose:** the maker experience. Pick an SLM from the curated catalog, author
the persona/system prompt, upload documents (txt/md/pdf) that get chunked and
embedded into a portable knowledge base, attach tools, and publish a versioned
**agent bundle** that devices download.

**Tech stack**

| Layer | Choice | Why |
|---|---|---|
| API server | Python 3.12 + FastAPI + uvicorn | small, async, zero boilerplate |
| UI | vanilla-JS SPA served by FastAPI (`studio/static/`) | no build step |
| Chunking | custom recursive splitter (~1,200 chars, 150 overlap) — `core/chunking.py` | tuned for 1–4B-model context budgets |
| Embeddings | **fastembed** (ONNX, CPU) with `BAAI/bge-small-en-v1.5`, 384-dim | no torch; same model family runs on phones |
| KB store | SQLite, embeddings as float32 blobs — `core/kbstore.py` | single portable file, readable on-device |
| Model catalog | `core/catalog.py` — Qwen3 0.6B/1.7B/4B, Llama-3.2-3B, Gemma-3-4B (4-bit GGUF, exact byte sizes) | licence + RAM guidance per phone class |
| Bundle format | zip: `manifest.json` (persona, model ref + `size_bytes`, gen/RAG params, tool declarations) + `kb.sqlite` | runtime-agnostic JSON; models referenced, not embedded |
| Registry | `bundles/registry.json` | what the store serves |
| Tunnel | **ngrok** with the account's static domain | URL never changes; free-tier 1 GB/month is fine for Studio traffic (bundles are KBs) |

**Start / restart manually** (if ever needed):

```powershell
schtasks /Run /TN "slm-studio"          # app
schtasks /Run /TN "slm-studio-tunnel"   # ngrok (static URL)
# or fully manual, from C:\slm:
venv\Scripts\python -m uvicorn studio.app:app --port 8100
```

**Use it:** open the URL → Agents → New Agent → fill Details (or *Insert Meeting
Intelligence template*) → Knowledge Base tab: upload docs, test retrieval →
Tools tab → Publish. Re-publishing bumps the version; devices see an *Update*.

**Demo seed** (recreates the Meeting Intelligence agent from `sample_docs/`):

```powershell
venv\Scripts\python scripts\seed_demo.py
```

---

## 3. Journey 2 — Device Runtime (port 8200, Cloudflare)

**Purpose:** two things in one app —
1. a **web device-simulator**: phone-styled chat UI with the agent store, used
   to demo/verify agents without a handset (inference via llama.cpp on this server);
2. the **phone-facing portal**: everything the Android app talks to.

**Portal surface (what the Android app calls):**

| Endpoint | Serves |
|---|---|
| `GET /api/published` | agent store catalog |
| `GET /bundles/<file>` | agent bundle zips |
| `GET /models/<file>` | GGUF models cached on this server (fallback source; phones try Hugging Face first) |
| `GET /apk` | the Android app itself |
| `GET /export/<agent>` | per-agent "get on phone" page (QRs: native app, PocketPal persona, ChatterUI card) |

**Tech stack**

| Layer | Choice | Why |
|---|---|---|
| API server | FastAPI + uvicorn, SSE streaming chat | server-sent events stream tokens to the web UI |
| Inference | **llama.cpp** `llama-server` (b9957, CPU) managed as a subprocess, one model at a time | same engine family as the phone (llama.rn) |
| Models | Qwen3 GGUF Q4_K_M in `models/` | shared across agents |
| RAG | fastembed query embedding + brute-force cosine over bundle `kb.sqlite` | identical scoring to the phone |
| Tool calls | Qwen3-native `<tool_call>` JSON parsed from the token stream; builtin tools write to `runtime/device_storage/` | no jinja plumbing → same logic ports to the phone |
| Thinking mode | `/no_think` soft switch + `<think>` stream filter | snappy responses from hybrid Qwen3 |
| Tunnel | **cloudflared quick tunnel** | https, no data cap (models are GBs — ngrok free's 1 GB/month died mid-download; that's why the two journeys use different tunnels) |

**Start / restart manually:**

```powershell
schtasks /Run /TN "slm-runtime"   # app
schtasks /Run /TN "slm-tunnel"    # cloudflared (NEW URL each start — re-configure phones!)
# or fully manual, from C:\slm:
venv\Scripts\python -m uvicorn runtime.app:app --port 8200
```

---

## 4. Android app (`android_app/AgentRuntime/`)

**Purpose:** the real Journey-2 client. Syncs with the portal, installs agents
(bundle + model + embedder), then runs **everything on the handset**: inference,
query embedding, vector search, tool execution, and user file-uploads (≤2 MB)
into an inline KB layered over the bundle KB. Airplane-mode capable after install.

**Tech stack**

| Layer | Choice | Notes |
|---|---|---|
| Framework | React Native **0.79.5**, Hermes, new architecture, arm64-only | built headless on this server (no Android Studio) |
| Inference + embeddings | **llama.rn 0.12.4** (llama.cpp binding) | one engine for chat *and* bge-small-q8 GGUF embeddings (cross-verified vs fastembed: cosine 0.9998). Pinned to PocketPal's proven version. CPU-only (`no_gpu_devices`), `flash_attn_type:'off'`, `kv_unified:true` |
| KB | **@op-engineering/op-sqlite** reads bundle `kb.sqlite` + device `inline.sqlite` | brute-force cosine in JS over float32 blobs |
| Downloads | bundles: in-memory fetch; models: **Android Download Manager** via react-native-blob-util | DM survives screen-off, resumes, shows shade progress; byte-exact copy verification |
| Files/zip | react-native-blob-util, react-native-zip-archive, @react-native-documents/picker | 📎 upload → chunk → embed → inline KB |
| State | AsyncStorage (portal URL, tool data, 📋 diagnostics ring buffer) | |
| Toolchain | Temurin JDK 17 + Android SDK 35 in `C:\slm\tools\` | ~5 GB, already installed |

**Build a new APK** (after any change under `android_app/AgentRuntime/`):

```powershell
# 1. bump the version or Android will refuse to update over the installed copy
#    android/app/build.gradle → versionCode +1, versionName as desired
# 2. typecheck
cd C:\slm\android_app\AgentRuntime; npx tsc --noEmit
# 3. build — MUST go through Task Scheduler on this server (see quirk below)
schtasks /Run /TN "slm-gradle-build"
# 4. wait for "EXITCODE 0" at the end of C:\slm\android_app\build.log (~3 min incremental)
# 5. verify + it's immediately live at <journey-2-url>/apk
C:\slm\tools\android-sdk\build-tools\35.0.0\aapt2.exe dump badging android\app\build\outputs\apk\release\app-release.apk | findstr versionName
```

> ⚠ **Server quirks (cost hours — don't rediscover them):**
> 1. Gradle/Java IPC is blocked in interactive shells on this EC2 box
>    ("Unable to establish loopback connection") — always build via the
>    `slm-gradle-build` scheduled task.
> 2. `npm install` of llama.rn must run in **PowerShell**, not Git-Bash
>    (GNU tar breaks on `C:\` paths in its postinstall).
> Full details: `android_app/BUILD.md`.

**Install & use on a phone:** open `<journey-2-url>/apk` in Chrome → allow
"install unknown apps" → in the app **⚙ → Portal URL** = the Journey-2 URL →
Save & test → Install an agent (Wi-Fi, keep app foregrounded) → chat offline.
Diagnostics: **📋** on the home screen → Share (every download, install step,
model load, tok/s, and error is recorded).

**Current device guidance:** Qwen3-0.6B is the proven tier on the test device
(13 tok/s generation, 1.7 s model load). Qwen3-1.7B loads its weights but fails
context init there — under investigation via the in-app native engine log; use
*Meeting Intelligence Lite* (0.6B) for demos meanwhile.

---

## 5. Feasibility on target phones (the original question — yes)

| Device class | RAM | 4-bit SLM that fits | Expected speed* |
|---|---|---|---|
| iPhone 14 / 14 Plus (A15, 6 GB) | 6 GB | 1–2B (Qwen3 1.7B ≈ 1.0 GB) | 15–25 tok/s |
| iPhone 15 Pro+ / 16 / 17 (8 GB+) | 8 GB | 3–4B (Qwen3 4B ≈ 2.3 GB) | 15–30 tok/s |
| Galaxy S23 / S24 / S25 (8–12 GB) | 8–12 GB | 3–4B | 15–30 tok/s |
| Measured: test Android device, Qwen3-0.6B, CPU | — | 0.37 GB | **13 tok/s gen, model load 1.7 s** |
| Measured: EC2 t3.xlarge (4 vCPU), Qwen3-1.7B | 16 GB | 1.0 GB | 8–14 tok/s gen |

*community benchmarks; NPU/GPU backends (Metal, OpenCL-Adreno) raise these,
especially prefill. On-device budget: SLM 0.4–2.3 GB + embedder 35 MB (GGUF q8)
+ KB SQLite (a few MB per agent).

## 6. Repo map

| Path | What |
|---|---|
| `core/` | shared: chunking, fastembed embeddings, SQLite vector store, SLM catalog (exact byte sizes) |
| `studio/` | Journey 1 web app |
| `runtime/` | Journey 2 web app + phone portal + export pages + telemetry ingest |
| `core/telemetry.py` | shared governance store (usage metadata only, no content) |
| `android_app/AgentRuntime/` | React Native app (see `android_app/BUILD.md`) |
| `finetune/` | requirement-driven fine-tuning: spec → synth → dataset gate → job pack → adapter gate — see [`docs/finetuning.md`](docs/finetuning.md) |
| `core/adapters.py` | LoRA adapter registry: import, promotion gate, manifest block |
| `models/` | GGUF files + embedder cache (served at `/models/`) |
| `bundles/` | published agent bundles + `registry.json` |
| `sample_docs/` | Acme demo corpus (MOMs, status, risks, ADRs, customer profile) |
| `scripts/` | service start scripts, `seed_demo.py` |
| `cloudflare/` | named-tunnel setup for permanent URLs, portable to any machine — see its README |
| `docs/` | [BRD](docs/BRD.md) (as-built capability reference), [rag-gating](docs/rag-gating.md), [finetuning](docs/finetuning.md) |
| `llama/` | llama.cpp b9957 Windows CPU binaries |
| `tools/` | JDK 17, Android SDK, cloudflared |

## 7. Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| Phone app: "Portal unreachable" | Tunnel restarted → new cloudflare URL → find it (§1) and update ⚙ in the app |
| Cloudflare URL returns 502/530 | runtime app down → `schtasks /Run /TN slm-runtime` |
| ngrok URL shows error page | `schtasks /Run /TN slm-studio-tunnel`; check `logs\ngrok.log` (auth = config path must be Administrator's) |
| Install fails on phone | 📋 → Share; log names the file, source, HTTP status, byte counts per attempt |
| "model file is incomplete" | device file ≠ manifest `size_bytes` → re-install (auto-cleaned) |
| Gradle "Unable to establish loopback connection" | you built in a shell — use `schtasks /Run /TN slm-gradle-build` |
| Studio/runtime code change not visible | restart that service (kill port listener + `schtasks /Run`, §1) |

## 8. Design choices that matter for mobile

- **Qwen3 family, Apache-2.0, ungated** — 0.6B (budget devices, proven 13 tok/s),
  1.7B (default), 4B (8 GB+ flagships); strong native tool-calling.
- **bge-small embeddings everywhere** — fastembed ONNX on servers, the *same
  weights* as GGUF under llama.rn on phones (verified interchangeable).
- **Brute-force cosine over SQLite blobs** — zero index dependencies; <10 ms at
  few-thousand-chunk scale; sqlite-vec is the upgrade path.
- **Qwen-native `<tool_call>` stream parsing** — identical logic in the web
  runtime (Python) and the app (TypeScript port, unit-tested).
- **Bundle manifest is runtime-agnostic JSON** — web simulator, Android app,
  and any future SwiftUI client consume the same artifact.

## 9. Roadmap

1. **Run the first fine-tune** — the requirement-driven pipeline is built
   (`finetune/`, [`docs/finetuning.md`](docs/finetuning.md)): a spec states the
   behaviour, generates its own training data over surrogate documents, and
   gates the resulting adapter on a scorecard before it can be published. What
   is left is the gradient step, which needs a GPU. Adapters ship as ~30 MB
   LoRA files applied on top of the shared base model, so a tuned agent costs a
   small download rather than another gigabyte.
2. ~~React Native device app~~ **done** — resolve Qwen3-1.7B init on the test
   device (native-log capture armed), then GPU/NPU offload as opt-in.
3. iOS build of the same RN app (needs a Mac or CI macOS runner).
4. sqlite-vec + hybrid search (BM25 + vector) past ~10k chunks.
5. Bundle signing + encryption at rest; MDM distribution; fixed portal domain
   (kills the rotating-tunnel-URL problem for good).
6. MCP tool bridge for desktop; `http` tool kind stays the mobile fallback.
7. Delta KB sync; eval harness gate before publish.
