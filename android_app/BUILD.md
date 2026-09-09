# Building the Android app on this Windows server

Toolchain (installed under `C:\slm\tools\`, no Android Studio needed):
- Temurin JDK 17 (`jdk-17.0.19+10`)
- Android SDK cmdline-tools + platform-tools + platforms;android-35 + build-tools;35.0.0
- Node 24 + React Native 0.79.5, deps: llama.rn, @op-engineering/op-sqlite,
  react-native-blob-util, react-native-zip-archive, async-storage,
  @react-native-documents/picker

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

## Install on a phone

Download `<portal>/apk` (QR on every agent's `/export/<id>` page) → allow
"install unknown apps" → open app → ⚙ set portal URL → Install agent.
