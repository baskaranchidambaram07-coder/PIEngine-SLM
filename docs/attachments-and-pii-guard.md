# File attachments, on-demand RAG, vision, and the PII Guard

*Status: built and measured on the web runtime (Journey 2), September 2026.
The web runtime is the test and benchmarking surface; the capability ships to
users in the mobile app afterwards (see §9).*

## 1. What was asked

The customer-facing app must let a user attach **one file at a time** —
**jpg, jpeg, pdf, txt or md, at most 5 MB** — and ask questions about it. Text
must be recovered from scanned pages and photos (OCR), an image must also be
read by a real **vision model**, and retrieval over the file must happen **on
demand** for the question asked. In front of all of it sits an agent — the
**PII Guard** — that checks the document *and* the user's message for personal
data, passwords, secret keys or API keys and, if it finds any, refuses to
proceed, naming each finding with only its **first three and last two characters visible**.

## 2. Shape of the solution

```
 browser ──POST /api/attachments──▶ validate ─▶ extract ─▶ index ─▶ PII Guard ─▶ meta.json
   │          (one file, ≤5 MB)      ext+magic   text/OCR   chunk+embed  regex(+judge)  verdict stored
   │
   └──POST /api/chat {attachment_id}──▶ Guard(stage 0: stored verdict)
                                        Guard(stage 1: the message)      ── refuse ──▶ "🛡 … not ready to proceed"
                                        on-demand retrieval over the file
                                        Guard(stage 2: retrieved chunks)  ── refuse ──▶
                                        image? ─▶ vision model (Qwen3-VL-2B, own llama-server :8303)
                                        else   ─▶ agent's SLM with "Attached document" context (+ KB)
                                        Guard(stage 3: the answer, masked in place)
```

Everything runs on this machine with the models the product already uses;
nothing calls a cloud service. New code:

| File | Role |
|---|---|
| `core/pii.py` | Deterministic detector: regexes + checksums, masking, redaction, grouping |
| `runtime/guard.py` | The PII Guard agent: policy, regex stage, model judge with verbatim-span validation, refusal text |
| `runtime/attachments.py` | Validation (extension **and** magic bytes, size, single file), storage, extraction (pypdf → PyMuPDF raster → Tesseract), chunk+embed, on-demand retrieval, expiry |
| `runtime/vision.py` | Second managed `llama-server` with `--mmproj` for Qwen3-VL-2B; image downscaling; streaming answers |
| `runtime/app.py` | `/api/attachments`, `/api/capabilities`, `/api/guard/policy`, `/api/vision/status`; the guarded chat flow |
| `runtime/static/index.html` | 📎 button, client-side validation, attachment chip with the guard verdict, refusal bubble, masked-answer notice |
| `tests/` | 67 offline tests (`venv\Scripts\python -m pytest`) |
| `prototypes/attach_bench.py` | Live benchmark against the running runtime |
| `scripts/make_attachment_samples.py` | Synthetic fixtures under `sample_docs/attachments/` |

## 3. Attachments

**Limits are enforced twice.** The browser refuses the wrong extension, more
than one file, an empty file or anything over 5 MB before uploading; the
server re-checks all of it and additionally reads the first bytes — a `.jpg`
that is really a PDF, or a `.txt` containing NUL bytes, is rejected (400/413)
rather than guessed at.

**Extraction**

| Type | Path | Fallback |
|---|---|---|
| txt / md | UTF-8 → UTF-16 → Latin-1 decode (`core.chunking.extract_text`) | — |
| pdf | pypdf text layer, page by page | any page with < 40 chars of text is rasterised at 300 dpi with PyMuPDF and OCR'd with Tesseract 5.5 (cap 40 pages) |
| jpg | Tesseract OCR after grayscale + 2× upscale (if small) + autocontrast | none needed — the vision model reads the pixels regardless |

OCR output gets one conservative repair: inside tokens that are otherwise
numeric, `@`/`O`/`o`→`0` and `l`/`I`/`|`→`1`. This came from the benchmark:
at 200 dpi Tesseract read "30 days" as "3@ days"; 300 dpi fixed that case and
the repair covers the residual class without touching words or identifiers
like `INV-2026`.

**Storage and retention.** Each upload lives in
`runtime/device_storage/attachments/<id>/` (`file.<ext>`, `text.txt`,
`kb.sqlite`, `meta.json`). It is deleted when the user removes the chip and
swept after 24 h regardless. The directory is git-ignored. `meta.json` holds
the guard verdict so a blocked file is refused on every later question without
rescanning; nothing the API returns ever contains extracted text or an
unmasked value.

## 4. On-demand RAG

A small document (≤ 3,500 characters, ~3 chunks) is injected **whole** —
retrieving 3 of 3 chunks by similarity can only lose information. Anything
larger is chunked with the section-aware splitter the knowledge base uses,
embedded with the same bge-small model, and searched at question time: the
lexical gate (`core.gating`) runs first so "thanks!" costs nothing, then
top-4 by cosine with a floor of 0.30 (deliberately lower than the KB's 0.45 —
the user chose this file). The agent's own knowledge base is still searched
and appended after the attachment context, so a question can join the two.

The prompt gains one paragraph telling the model the user attached a file,
that its text appears under *Attached document*, and to say plainly when the
file does not contain the answer. Sources are streamed to the UI tagged
`attachment` (📎) or `kb` (📄).

## 5. Vision

Images are answered by **Qwen3-VL-2B-Instruct** (Apache-2.0; base Q4_K_M
1.1 GB from `unsloth`, projector Q8_0 0.45 GB from `ggml-org`) in a second
`llama-server` on port 8303 with `--mmproj`, started on first use and left
running. The agent's persona prompt is reused (tools dropped), and the OCR
transcript is passed alongside the image with the instruction to trust the
pixels where they disagree — the pairing is markedly more reliable for
amounts, codes and serial numbers than either alone.

Images are EXIF-oriented and downscaled to 1,024 px on the longest side
before encoding: Qwen3-VL spends one token per 32×32 patch, so a 12 MP photo
would be ~12,000 tokens and minutes of CPU prefill; the cap keeps it under
~800. `--image-max-tokens 1024` enforces the same bound server-side.

Why not the 7B the Studio had onboarded: `Qwen2.5-VL-7B Q2_K_L` (3.1 GB) has
no projector downloaded, a 2-bit quant hurts exactly the fine detail OCR-type
questions need, and CPU prefill cost scales with parameters. The 2B answers a
receipt question in ~15–25 s on this 4-vCPU box; the 7B would be several
times slower for no measured gain here. If the vision files are missing,
image turns degrade to OCR-only through the agent's text model and the UI
says so.

## 6. The PII Guard

### 6.0 Per-agent switch (Studio → Guardrails)

The guard is a **guardrail the agent's author turns on**, not a runtime
default. In the Studio, every agent has a *Guardrails* panel under *Model*
with a single checkbox, **🛡 PII Guard**, off by default. The setting is saved
in the agent config (`guardrails: {"pii": true|false}`), published in the
bundle's `manifest.json`, shown on the agent card and in the bundle preview,
and enforced by the runtime from the installed manifest:

| Guardrail | user message | attached file (upload + retrieved chunks) | model answer |
|---|---|---|---|
| **on** | scanned; refused on a hit | scanned; refused on a hit | masked in place |
| **off** (default) | passes through untouched | extracted, indexed and used as-is | shown as generated |

The runtime exposes the state in `GET /api/installed` (`guardrails.pii`), the
upload response (`guard.enabled`), the agent list ("🛡 PII Guard" tag), the
chat header and the welcome text, so a tester can always see which mode is in
force. Manifests published before the switch existed have no `guardrails` key
and are treated as **off**. An upload with no `agent_id` (direct API use) gets
the regex stage regardless, as the conservative default.

Why opt-in: the guard refuses entire documents and adds a model-judge call to
turns that carry a file. That is right for an assistant that handles user
uploads (the shipped `pii-guard` agent has it on) and wrong for a curated-KB
assistant such as Meeting Intelligence, whose knowledge base is *made of*
people, phone numbers and email addresses that the organisation put there
deliberately.

### 6.1 Taxonomy

| Kind | Detected (deterministic) | Blocks by default |
|---|---|---|
| **secret** | passwords/PINs/passphrases after a label; API keys with vendor prefixes (AWS `AKIA…`, GitHub `ghp_…`, Slack `xox…`, Google `AIza…`, Stripe, OpenAI, Anthropic, Hugging Face, SendGrid); bearer tokens; JWTs; `-----BEGIN … PRIVATE KEY-----`; storage account keys; labelled generic secrets (`api_key = …`, `client_secret: …`) | yes |
| **identity** | payment cards (Luhn), UAE Emirates ID (784-YYYY-NNNNNNN-C, Luhn), Indian driving licence (SS-RR-YYYY-NNNNNNN), UK driving licence (with date-field check), US SSN, Aadhaar (Verhoeff), Indian PAN, IBAN (mod-97), passport numbers when labelled, dates of birth when labelled, and **any card or licence introduced by its name** (driving/driver's licence, national/identity/citizen/resident card, voter ID, health/insurance/ration card, "card number") | yes |
| **contact** | email addresses, phone numbers (10–13 digits with phone punctuation, or bare mobile-shaped runs) | yes |
| names, postal addresses, health/salary details | **model judge only** (see 6.3) | addresses/health yes; **names off** (`block_names`) |

Personal names are a policy switch and default **off** because an enterprise
document — meeting minutes, a risk register — is made of colleagues' names,
and blocking on them would refuse the flagship Meeting Intelligence scenario
outright. The switch exists for deployments that want it.

Prose that merely *talks about* secrets is not a secret: "the password policy
requires 12 characters", `api_key = <your_api_key>`, `token: null` and
`password: ********` all pass. Placeholders, all-lowercase dictionary words
after `password is`, and values under 8 characters after a generic label are
ignored. The negative test set (invoice totals, PO numbers, dates, versions,
ports, file sizes, budgets) is in `tests/test_pii.py`.

### 6.2 Masking

Values longer than five characters show their **first three and last two**
characters and mask everything else, internal spaces included:
`AKIAIOSFODNN7EXAMPLE` → `AKI***************LE`, `priya.sharma@example.com` →
`pri*******************om`, `4111 1111 1111 1111` → `411**************11`.
Values of five characters or fewer show only their last character. The
unmasked value never leaves the process: the finding's public form drops it,
and the stored verdict keeps only the summary.

### 6.3 The model judge

The agent's own on-device SLM (Qwen3-1.7B for most agents) is prompted as a
strict reviewer and asked for JSON `{"findings":[{"type","value"}]}`. Every
value must occur **verbatim** in the text (whitespace-tolerant) or it is
dropped — a 1.7B model will happily invent a plausible phone number, and a
guard that hallucinates findings blocks legitimate work. Types map onto the
taxonomy; `name` is discarded unless `block_names` is on.

Because a judge call costs 10–40 s here, **policy** (`GET/PUT
/api/guard/policy`, stored in `device_storage/guard_policy.json`) decides
where it runs:

| `judge` | Behaviour |
|---|---|
| `attachments` *(default)* | on the user's message and the retrieved chunks of any turn that carries a file; at upload, on the first 2 chunks |
| `always` | additionally on every plain message |
| `off` | regex only — for latency benchmarks |

Chunk verdicts are cached in the attachment's `meta.json` (`judged_chunks`),
so a chunk is judged once however many questions retrieve it. Regex runs on
**every** chunk of **every** file at upload, always, in milliseconds.

### 6.4 Where it fires

| Stage | Input | Cost | On finding |
|---|---|---|---|
| upload | whole extracted text (regex); first 2 chunks (judge) | ms + 0–2 judge calls | chip turns red, refusal bubble, every later question refused |
| chat 0 | stored verdict | free | refuse |
| chat 1 | the user's message | ms (+ judge if a file is in play) | refuse, no model call |
| chat 2 | retrieved chunks not yet judged | judge, cached | refuse and mark the file blocked |
| chat 3 | the model's answer | ms | answer masked in place, notice under the bubble |

The refusal is the same sentence everywhere:

> 🛡 PII Guard: I found sensitive data in the attached document "pii-secrets.md", so I am not ready to proceed further.
> • AWS access key — a secret: `*****************PLE`
> • Password — a secret: `***************ing`
> Remove or redact this information and try again.

Refusals are recorded in telemetry as `guard_block` with the **labels only**
(never values) so the Governance dashboard can count them.

## 7. API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/attachments` (multipart `file`, optional `agent_id`) | validate → extract → index → guard; returns the public meta incl. `guard.blocked`, `guard.summary` (masked), `guard.message` |
| GET / DELETE | `/api/attachments/{id}` | meta / remove now |
| POST | `/api/chat` `{agent_id, messages, attachment_id?}` | SSE; new events `status`, `guard`, `redact`, `vision`; `sources[].kind` |
| GET | `/api/capabilities` | limits, allowed types, OCR/vision availability, guard policy |
| GET / PUT | `/api/guard/policy` | read / change the judge mode, `block_names`, etc. |
| GET | `/api/vision/status` | vision model files present / server running |

`agent_id` on upload lets the judge run on that agent's model; without it the
upload is regex-only and says so (`guard.judged=false`).

## 8. Measurements

See `prototypes/attach_bench_results.json` (produced by
`venv\Scripts\python prototypes\attach_bench.py --agent product-insights`).
Numbers below are from that run on the EC2 t3.xlarge (4 vCPU, no GPU), agent
model Qwen3-1.7B Q4_K_M, judge policy `attachments`.

<!-- bench:begin -->

**Per file** (upload = validate + extract/OCR + embed + guard, incl. judge calls):

| file | upload s | guard verdict | expected | questions correct | answer s (avg) | first token s | model |
|---|---|---|---|---|---|---|---|
| `clean-invoice.txt` | 8.9 | clear ✓ | clear | 2/2 | 20.2 | 17.6 | agent SLM |
| `clean-notes.md` | 5.0 | clear ✓ | clear | 2/2 | 20.6 | 18.8 | agent SLM |
| `clean-brief.pdf` | 5.3 | clear ✓ | clear | 2/2 | 16.6 | 14.1 | agent SLM |
| `scanned-policy.pdf` | 6.1 | clear ✓ | clear | 2/2 | 13.6 | 11.3 | agent SLM |
| `clean-receipt.jpg` | 5.4 | clear ✓ | clear | 2/2 | 45.1 | 43.9 | Qwen3-VL 2B (Q4_K_M) |
| `pii-contacts.txt` | 7.5 | blocked ✓ (Email address, Phone number) | blocked | 0/0 | — | — | — |
| `pii-secrets.md` | 12.6 | blocked ✓ (Credential (model judge), AWS access key, AWS secret key, Password, API key / secret) | blocked | 0/0 | — | — | — |
| `pii-card.pdf` | 11.7 | blocked ✓ (Credential (model judge), Payment card number, US Social Security number, Financial account number, Government ID number) | blocked | 0/0 | — | — | — |
| `pii-badge.jpg` | 2.5 | blocked ✓ (Password, Email address) | blocked | 0/0 | — | — | — |

**Questions and answers:**

| file | question | answer (trimmed) | ok |
|---|---|---|---|
| `clean-invoice.txt` | What is the invoice number and the total due in INR? | The invoice number is INV-2026-0912, and the total due in INR is INR 35,542. | ✓ |
| `clean-invoice.txt` | What are the payment terms? | The payment terms are 15 days net. | ✓ |
| `clean-notes.md` | What was decided about the Phase 2 rollout date? | The Phase 2 rollout date was decided to be 6 October 2026, allowing UAT to finish. | ✓ |
| `clean-notes.md` | Who owns the payment gateway risk? | The payment gateway risk is owned by the PMO. | ✓ |
| `clean-brief.pdf` | How many pilot sites are planned and where? | The attached document mentions three pilot sites planned for Q4 2026: Pune, Chennai, and Kochi. | ✓ |
| `clean-brief.pdf` | What does success mean for the pilot? | Success means a 20% reduction in report turnaround time and no inspection blocked by connectivity. | ✓ |
| `scanned-policy.pdf` | How many days of notice are needed to change a remote-work pattern? | 2. Notice. Changes to a standing remote-work pattern require 30 days written notice to the line manager and HR. | ✓ |
| `scanned-policy.pdf` | How many days a week can eligible employees work remotely? | According to the REMOTE WORK POLICY, eligible employees may work remotely up to three days a week. | ✓ |
| `clean-receipt.jpg` | What is the total on this receipt? | The total on this receipt is $42.75$. | ✓ |
| `clean-receipt.jpg` | Which cafe is this receipt from? | The receipt is from NANDINI CAFE - MG ROAD. | ✓ |

**Message-only guard checks** (no file attached; regex only under the default policy):

| message | blocked | expected | s |
|---|---|---|---|
| hi there | False | False | 2.6 |
| What did we decide about the architecture? | False | False | 5.8 |
| my password is Hunter2! can you save it for me | True | True | 0.0 |
| use key AKIAIOSFODNN7EXAMPLE for the bucket | True | True | 0.0 |
| call me on +91 98765 43210 tomorrow | True | True | 0.0 |
| the budget is 250000 INR and the PO is 4500012345 | False | False | 13.8 |

**Totals:** guard verdicts 9/9 · answers 10/10 · message guard 6/6 · judge policy `attachments` · vision `Qwen3-VL 2B (Q4_K_M)` · agent `product-insights`

<!-- bench:end -->

**Reading the numbers.**

- Text and PDF turns cost 5–24 s end to end; almost all of it is prefill of
  the ~500–900 prompt tokens (context + persona) at ~50 tok/s. The second
  question on a file is faster because llama-server reuses the cached prefix.
- Image turns cost ~45 s: ~690 image + text tokens through the vision model's
  CPU prefill at ~17 tok/s, then ~8.5 tok/s generation. That is the price of
  a real vision pass on 4 vCPUs; the OCR-only path would answer the same two
  receipt questions in ~20 s but cannot answer anything OCR misses.
- Upload cost is dominated by the judge (2 chunks × 3–6 s); regex-only
  uploads (`judge: off`) complete in well under a second for text and ~1.5 s
  for an OCR'd page.
- The message-only checks show the fast path: a pasted secret is refused in
  0.0 s with no model call, a greeting costs 2.6 s, and a message full of
  business numbers (budget, PO) is correctly **not** blocked.
- Round 1 of this benchmark (before the fixes described in §3 and §6.3) had
  two failures: the 1.7B answered "23 days" from OCR text where the number
  sat on a separate line from its verb, and the judge blocked the clean
  receipt as PII (shop address, receipt date, table number). Both are the
  reason those fixes exist; both pass in round 2 with no other regressions.
- The judge occasionally re-reports a value regex already found under its
  own label (e.g. `pii-card.pdf` shows both "Payment card number" and
  "Financial account number"). Harmless — the verdict is the same — and
  the display groups by label so the user sees two lines, not a leak.

## 9. The mobile app (ported in v2.7)

The web runtime is the reference implementation. The Android app
(`android_app/AgentRuntime`, v2.8 / versionCode 19) now carries the same
capability, built the same way — the list below is what was done and where:

1. **Validation and limits** — `src/attachments.ts`: same five extensions,
   one file, 5 MB, and the copied file's magic bytes (`FF D8 FF`, `%PDF-`, no
   NUL) — React Native has no `Buffer`, so a 20-line base64 head decoder does it.
2. **Detector** — `src/pii.ts` is a line-for-line port of `core/pii.py`
   (checked by compiling it and running the same sentences through both:
   15/15 identical). Both gained Emirates ID, Indian and UK driving licences,
   labelled ID/licence cards and PIN/OTP rules in the same change.
3. **OCR** — ML Kit Text Recognition (`@react-native-ml-kit/text-recognition`,
   on-device, no Play Services call). PDFs are rendered page by page with
   the platform `PdfRenderer` (`react-native-pdf-thumbnail`, capped at 20
   pages) and OCR'd the same way — one path for text PDFs and scans. The
   line-unwrapping and digit-confusion repairs are ported unchanged.
4. **Vision** — `src/vision.ts` + `llm.ensureVisionModel`: Qwen3-VL-2B via
   llama.rn `initMultimodal`, **off by default** behind a ⚙ switch that
   downloads the two files from the portal (`/models/…`, Hugging Face
   fallback). The blocked-model safety gate applies, so a phone that cannot
   hold it fails once, cleanly, and falls back to OCR-only for good.
5. **Guard** — `src/guard.ts`: regex stage on message, file and answer; the
   model judge (same JSON prompt, verbatim-span validation) runs only on
   turns with a file, on the message and one document chunk, using whichever
   big model is loaded so a turn never holds two models.
6. **Indexing** — attachments get their own `attach.sqlite` beside the copied
   file (not the agent's inline KB), with the `small` whole-file shortcut,
   top-4 and the 0.30 floor; swept after 24 h.
7. **Uninstall** — per-agent button on the home screen; removes bundle, KBs
   and attachments, clears the vector cache and posts `uninstall` telemetry,
   which releases the Studio's delete guard for that device.

Verification so far is build-level: `npx tsc --noEmit` clean, Gradle release
build green after one dependency patch (see `android_app/BUILD.md` §4), and
the APK's Hermes bundle contains the new strings while the old inline-KB
upload strings are gone. Verified on two handsets on 11–12 September: PDF and photo attachments
read on device, PII blocks in milliseconds, the model judge runs, the
vision-off image question is answered from OCR text, uninstall works, and
(from v2.9) Qwen3-1.7B agents load as well — see
[`android-1.7b-root-cause.md`](android-1.7b-root-cause.md).

## 10. Agent lifecycle: sync between Studio and devices

Deleting an agent in the Studio while a device still runs it leaves that copy
orphaned — no store entry, no version lifecycle, no way to push a fix. Two
changes close that gap (`core/installs.py`):

- **Delete is refused while anything holds the agent.** The Studio checks the
  web runtime's installed agents directly on disk and Android handsets through
  telemetry (an `install` event with no later `uninstall` from the same
  device, seen within 90 days). `GET /api/agents/{id}/installs` lists the
  holders; the editor shows them next to the agent id and disables *Delete
  agent* with the reason as a tooltip.
- **Devices show what the Studio no longer has.** The runtime marks an
  installed agent whose id is missing from the bundle registry as *removed in
  Studio* and offers an *uninstall* button on every installed agent;
  uninstalling records an `uninstall` telemetry event so the Studio's check
  sees the copy go. The Android app's telemetry type gains `uninstall` for the
  same purpose, but the app has no uninstall screen yet, so handsets currently
  release their hold only by going quiet for 90 days.

## 11. Known limits

- The judge is advisory and slow; regex is the guarantee. Unlabelled
  free-form secrets ("my key is 9f8a…") and postal addresses depend on it.
- Phone detection is deliberately inclusive: any 10–13 digit run with phone
  punctuation counts, so a spaced 12-digit reference number will be flagged.
- Personal names are not blocked by default (policy `block_names`).
- One image per turn; multi-page PDFs are OCR'd but not shown to the vision
  model page by page (CPU cost).
- Attachments are not part of the conversation history the model sees on
  later turns; the chip stays active so each turn re-attaches explicitly.
