# Building the Android app on this Windows server

Toolchain (installed under `C:\slm\tools\`, no Android Studio needed):
- Temurin JDK 17 (`jdk-17.0.19+10`)
- Android SDK cmdline-tools + platform-tools + platforms;android-35 + build-tools;35.0.0
- Node 24 + React Native 0.79.5, deps: llama.rn, @op-engineering/op-sqlite,
  react-native-blob-util, react-native-zip-archive, async-storage,
  @react-native-documents/picker, @react-native-ml-kit/text-recognition (on-device OCR),
  react-native-pdf-thumbnail (PdfRenderer -> JPEG per page; needs the Kotlin-2.0 patch in
  `patches/`, re-applied by `npm install` through `scripts/apply-patches.js`)

## Build

```
C:\slm\android_app\run_build.cmd        # or:
cd AgentRuntime\android && set JAVA_HOME=C:\slm\tools\jdk-17.0.19+10 && gradlew --no-daemon assembleRelease
```

APK output: `AgentRuntime\android\app\build\outputs\apk\release\app-release.apk`
(debug-signed by the RN template — fine for sideloading; use a real keystore
for store distribution). Served by the web runtime at `/apk`.

## ⚠ Quirks of THIS server (EC2 Windows 2025)

1. **Java NIO pipes/selectors fail** ("Unable to establish loopback
   connection" ← "Invalid argument: connect") in interactive/agent shells:
   AF_UNIX socket *connect* is blocked for this session's process tree, and
   JDK 16+ pipes/selectors require it. Gradle's daemon IPC dies instantly.
   **Workaround: run the build via Task Scheduler** (different session, no
   filter) — that's what `run_build.cmd` + `schtasks /Run /TN slm-gradle-build`
   does. Plain TCP loopback (uvicorn, llama-server) is unaffected.
2. **npm postinstall of llama.rn needs Windows bsdtar**: Git-Bash's GNU tar
   treats `C:\...` as a remote host. Run `npm install` from PowerShell (where
   System32 tar wins), not Git-Bash.
3. `gradle.properties` pins `kotlin.compiler.execution.strategy=in-process`
   and arm64-v8a only — keep both.
4. `react-native-pdf-thumbnail@1.3.1` does not compile under Kotlin 2.0.21 as
   published (nullable `bitmap.config`). The one-line fix lives in `patches/README.md`
   and is applied by the `postinstall` script; if a build fails in
   `:react-native-pdf-thumbnail:compileReleaseKotlin`, run `node scripts/apply-patches.js`.
5. Verify a build shipped your change by grepping `assets/index.android.bundle` inside the
   APK. It is Hermes bytecode: ASCII strings are searchable as UTF-8, but any string that
   contains a non-ASCII character (emoji, …) is stored as UTF-16LE — search both.

## Install on a phone

Download `<portal>/apk` (QR on every agent's `/export/<id>` page) → allow
"install unknown apps" → open app → ⚙ set portal URL → Install agent.
6. **The engine is built from source** (`rnllamaBuildFromSource=true` in `android/gradle.properties`)
   so the local llama.rn patches in `scripts/apply-patches.js` take effect — one of them is the
   fix that lets Qwen3-1.7B GGUFs load on handsets. A cold build compiles llama.cpp for arm64
   and takes ~60 min on this box; incremental rebuilds after JS-only changes are ~5-10 min.
   Needs NDK 27.1 and CMake 3.22 under `tools/android-sdk` (both present).
