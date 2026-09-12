# Local patches to node_modules

Applied automatically by `npm install` (see `scripts.postinstall` in package.json)
via `scripts/apply-patches.js`. Re-run it by hand after a manual dependency change:

    node scripts/apply-patches.js

| package | why |
|---|---|
| `react-native-pdf-thumbnail@1.3.1` | `PdfThumbnailModule.kt:101` passes the nullable `bitmap.config` to `Bitmap.createBitmap`, which Kotlin 2.0 (this app pins 2.0.21) rejects. Patched to `bitmap.config ?: Bitmap.Config.ARGB_8888`. |
| `llama.rn@0.12.4` (diagnostics) | The JSI promise wrapper reports any non-`std::exception` as the bare string "Unknown error" (`cpp/jsi/JSIUtils.cpp`). Patched to append the demangled exception type. `cpp/jsi/RNLlamaJSI.cpp` gains `diag:` logcat markers through the post-load hand-off (thread pools → devices → system info → JS result → metadata → chat templates). These only take effect when the engine is **built from source**: `rnllamaBuildFromSource=true` in `android/gradle.properties` (set on 2026-09-12 to chase the Qwen3-1.7B "Unknown error" on handsets). Remove that property to go back to the prebuilt engine libraries; the patches are then inert. |
| `llama.rn@0.12.4` (fix) | `createModelDetails` probes the model's embedded Jinja chat template; the engine's Jinja implementation throws `std::runtime_error` on the older Qwen3 template inside the unsloth and ggml-org **Qwen3-1.7B** GGUFs (the 0.6B GGUF carries the newer, guarded template and passes). That exception escaped as "Unknown error" and failed the whole model load. Patched to wrap the probe in try/catch and report "no template support" instead — harmless here because the app builds ChatML prompts itself. Requires the from-source build. |
