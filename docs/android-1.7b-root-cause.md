# Post-mortem: "Qwen3-1.7B can't run on this device"

*Resolved 12 September 2026, app v2.9. Two months of a wrong conclusion,
one evening of a right method.*

## Symptom

On every Android handset we tried (POCO X3 Pro, iQOO, Pixel 7), the app
downloaded `Qwen3-1.7B-Q4_K_M.gguf`, loaded it to 100%, and then failed with:

```
llm: init FAILED (progress 100%): Unknown error
```

Qwen3-0.6B always worked on the same phones. The app then showed "it needs
more memory than is available… use Qwen3 0.6B" and blocked the model.

## What we believed, and why it was wrong

| Belief | Evidence that seemed to support it | Why it was wrong |
|---|---|---|
| **Out of memory** (July 2026) | 6 GB phone, 1.1 GB weights, 224 MB KV cache, 304 MB compute buffer; the native log ended right after those allocations | The allocations *succeeded*. The log ended there because the failing step emits no engine log lines. A v2.4 experiment already shrank buffers 4× to no effect — that should have killed the hypothesis then. |
| **The new v2.7 build broke it** (11 Sep) | It failed the evening the attachment features shipped | v2.6 built from the last commit failed identically on a fresh install. Telemetry showed no phone had *ever* loaded 1.7B. |
| **The model file** | unsloth GGUFs carry patched templates | The ggml-org build of the same model failed identically. |
| **The engine version** | llama.rn 0.12.4 was from May; nine releases since | 0.12.9 failed identically. |

Each hypothesis was reasonable and each was falsified by one controlled
experiment. The common mistake was reading "the log ends after the memory
lines" as "the failure is memory".

## Method that found it

1. **Compare against ground truth, not memory.** Portal telemetry and the
   phone's own ring-buffer log both showed zero successful 1.7B loads ever,
   including the day the user remembered it working (they had used the web
   portal on the phone's browser, where 1.7B runs on the server).
2. **Isolate one variable per build.** Same phone, same day: old code (v2.6),
   different model file (ggml-org), different engine (0.12.9). All failed the
   same way → the cause was in none of those.
3. **Get the hidden error text.** The engine wrapper (llama.rn) logs its own
   messages to Android logcat, not to the log callback the app captures. A
   20-line Kotlin module (`LogcatModule.kt`) reads the app's own logcat and
   appends it to the in-app 📋 log. That showed the wrapper's *post-load*
   steps completing ("Attached ggml threadpool") 22 ms before the failure.
4. **Name the exception.** The wrapper's catch-all reported "Unknown error"
   for anything it could not match. Building the engine from source
   (`rnllamaBuildFromSource=true`) with a two-line patch made it append the
   demangled exception type, and `diag:` markers were added between each
   post-load step. Result: `std::runtime_error`, thrown between
   `diag: metadata ok` and `diag: chat templates ok`.

## Root cause

After a model loads, llama.rn's `createModelDetails` **probes the chat
template embedded in the GGUF** to report whether Jinja/tool-use formatting
is supported. The engine's own Jinja implementation throws on the *older*
Qwen3 template (`{%- set content = message.content %}` with no `is defined`
guard, `'</think>' in message.content` on a possibly-null value) that both
1.7B files carry. The unsloth 0.6B file carries the newer, guarded template
and passes.

The `std::runtime_error` then escaped the wrapper's `catch (const
std::exception&)` and hit `catch (...)` — a type-info mismatch between the
engine library and the JNI library — so the app only ever saw "Unknown
error", which our own message translated into "needs more memory".

**Nothing about the phones' memory, the model bytes or the engine version
was involved.** The probe is informational: this app builds ChatML prompts
itself and never asks the engine to render a template.

## Fix

`android_app/AgentRuntime/scripts/apply-patches.js` (re-applied by
`npm install`) wraps the probe in try/catch and reports "no template
support" instead of failing the load. The engine is built from source so the
patch takes effect (`android/gradle.properties`, ~60 min cold, ~10 min
incremental). Shipped as v2.9 (versionCode 23).

Verified on the POCO: 1.7B `init OK · loaded in 5420ms`, 13 tok/s; 0.6B
unchanged at 20+ tok/s; attachments, OCR and PII Guard unchanged.

Kept as well, because they are useful regardless: the ⚙ "Retry blocked
models" control (a block should never be permanent) and the 8-bit K cache for
large models (a quarter less KV memory, no quality change observed).

## Lessons

- **A log that ends is not a log that explains.** Find the *next* expected
  line and ask why it is missing before naming a cause.
- **"Unknown error" is a bug in the messenger.** Fix the messenger first;
  every later hypothesis is cheaper with the real text in hand.
- **One variable per experiment, on the same device, the same day.** Five
  builds (v2.6 from git, ggml-org file, 0.12.9 engine, diag, diag2) took an
  evening and each eliminated exactly one explanation.
- **User memory and telemetry disagree sometimes; check which surface.** The
  web portal on a phone looks like the app and says "on-device" while running
  on the server.
- **Do not let a safety message assert a cause.** "Blocked after a failed
  load" was right; "needs more memory" was a guess dressed as a fact, and it
  steered two months of thinking. The message now says only what is known.
- **Keep the tools.** Own-process logcat capture, the exception-type
  reporter and the `diag:` markers stay in the build; the next mystery will
  be cheaper.

## Upstream

The wrapper's behaviour is worth reporting to llama.rn: a template-probe
failure should not fail `initLlama`, and the catch-all should surface the
exception's `what()`. Until then our from-source build carries the patch.
