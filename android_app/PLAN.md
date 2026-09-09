# Android Runtime App — blueprint

Goal: a small Android app that consumes the **exact agent bundles** the Studio
publishes (manifest.json + kb.sqlite) and runs them fully on-device — live RAG
with citations and on-device tools, not just the baked-prompt export that
PocketPal/ChatterUI give us.

## Stack (all pieces proven on-device)

| Concern | Choice | Why |
|---|---|---|
| App shell | React Native 0.76+ (Hermes) | one codebase, iOS later for free |
| SLM inference | `llama.rn` | llama.cpp with **prebuilt arm64 libs** — no NDK build; streams tokens; loads the same GGUF |
| KB store | `react-native-quick-sqlite` (or op-sqlite) | reads our kb.sqlite unchanged |
| Query embedding | `onnxruntime-react-native` + bge-small-en-v1.5 ONNX (~130 MB) | same embedder id as the bundle manifest |
| Retrieval | JS cosine over Float32Array blobs | identical math to `core/kbstore.py`; <10 ms for few-k chunks |
| Tool calls | port of `runtime/llm.py` stream filter | parse `<think>`/`<tool_call>` from the llama.rn token stream |
| Builtin tools | AsyncStorage (action items, outbox), Calendar via `react-native-calendar-events` | graceful offline behaviour identical to the web runtime |
| Store/install | fetch `/api/published` from the Studio URL (ngrok in the demo), download bundle zip + GGUF with `react-native-blob-util` | same protocol the web runtime uses |

## Screens (2)

1. **Store/Agents** — list from Studio URL, install/update, download progress,
   offline-ready badge (mirrors the web sidebar).
2. **Chat** — reuse the phone-frame design from `runtime/static/index.html`:
   streaming bubbles, source chips, tool activity, tok/s stats.

## Build path on this Windows server (no Android Studio needed)

1. Temurin JDK 17 + Android cmdline-tools (`sdkmanager` installs
   platform-tools, `platforms;android-35`, `build-tools;35.0.0`) — ~3 GB.
2. `npx @react-native-community/cli init AgentRuntime`, add llama.rn +
   the libs above.
3. `gradlew assembleRelease` → debug-signed APK (fine for prototype/sideload).
4. Serve the APK from the runtime web app (`/apk`) → user downloads it on the
   phone **through the existing ngrok tunnel** and sideloads ("install
   unknown apps" permission, no developer mode needed).

Estimated one-time cost: ~3 GB of tooling downloads, 30–60 min first build.
iOS phase 2: same codebase via llama.rn, but needs a Mac or macOS CI runner.

## Definition of done (parity checklist)

- [ ] installs `meeting-intelligence-v1.zip` from the Studio store URL
- [ ] airplane-mode chat answers with citations from kb.sqlite
- [ ] `create_action_item` / `get_todays_meetings` / `draft_email` work offline
- [ ] model swap when a bundle references a different GGUF
- [ ] bundle update flow (v2 shows "Update")
