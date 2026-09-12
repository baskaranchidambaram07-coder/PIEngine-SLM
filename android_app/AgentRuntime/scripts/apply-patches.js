// Re-apply the small source patches listed in patches/README.md after npm install.
// Idempotent: each patch is a literal search/replace that is a no-op once applied.
const fs = require('fs');
const path = require('path');

const JSI_UTILS = 'node_modules/llama.rn/cpp/jsi/JSIUtils.cpp';
const RN_JSI = 'node_modules/llama.rn/cpp/jsi/RNLlamaJSI.cpp';

// Diagnostic helper for llama.rn's promise wrapper: name the C++ exception
// type behind "Unknown error" instead of discarding it.
const UNKNOWN_HELPER = `#include "ThreadPool.h"
#include <cxxabi.h>
#include <cstdlib>
#include <typeinfo>

namespace rnllama_jsi {
    // Diagnostic (local patch): the type of the exception currently being
    // handled by a catch(...) block, demangled, so the JS side sees WHAT was
    // thrown rather than a bare "Unknown error".
    static std::string unknownErrorName() {
        std::string out = "Unknown error";
        const std::type_info* t = abi::__cxa_current_exception_type();
        if (t != nullptr) {
            int status = 0;
            char* d = abi::__cxa_demangle(t->name(), nullptr, nullptr, &status);
            out += std::string(" (exception type: ") + ((status == 0 && d) ? d : t->name()) + ")";
            std::free(d);
        }
        return out;
    }
}
`;

const patches = [
  {
    file: 'node_modules/react-native-pdf-thumbnail/android/src/main/java/org/songsterq/pdfthumbnail/PdfThumbnailModule.kt',
    from: 'Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config)',
    to: 'Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config ?: Bitmap.Config.ARGB_8888)',
  },
  // ---- llama.rn diagnostics (only take effect when built from source:
  //      rnllamaBuildFromSource=true in android/gradle.properties)
  { file: JSI_UTILS, from: '#include "ThreadPool.h"\n', to: UNKNOWN_HELPER, once: 'unknownErrorName' },
  { file: JSI_UTILS, from: 'createJsiError(rt, "Unknown error")', to: 'createJsiError(rt, unknownErrorName())', all: true },
  // step markers through the post-load hand-off, to logcat tag RNLlama
  { file: RN_JSI,
    from: '                         ctx->attachThreadpoolsIfAvailable();\n',
    to: '                         ctx->attachThreadpoolsIfAvailable();\n                         __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: threadpools attached");\n' },
  { file: RN_JSI,
    from: '                         addContext(contextId, (long)ctx);\n',
    to: '                         __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: devices=%zu gpu=%d", usedDevices.size(), (int)gpuEnabled);\n                         addContext(contextId, (long)ctx);\n' },
  { file: RN_JSI,
    from: '                         std::string system_info = common_params_get_system_info(ctx->params);\n',
    to: '                         std::string system_info = common_params_get_system_info(ctx->params);\n                         __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: system_info ok (%zu chars)", system_info.size());\n' },
  { file: RN_JSI,
    from: '                         return [gpuEnabled, reasonNoGPU, system_info, usedDevices, contextId](jsi::Runtime& rt) {\n                             jsi::Object result(rt);\n',
    to: '                         return [gpuEnabled, reasonNoGPU, system_info, usedDevices, contextId](jsi::Runtime& rt) {\n                             __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: building JS result");\n                             jsi::Object result(rt);\n' },
  { file: RN_JSI,
    from: '        model.setProperty(runtime, "metadata", metadata);\n',
    to: '        model.setProperty(runtime, "metadata", metadata);\n        __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: metadata ok (%d keys)", metaCount);\n' },
  { file: RN_JSI,
    from: '        model.setProperty(runtime, "chatTemplates", chatTemplates);\n',
    to: '        model.setProperty(runtime, "chatTemplates", chatTemplates);\n        __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: chat templates ok");\n' },
  { file: RN_JSI,
    from: '        char desc[1024];\n        llama_model_desc(ctx->model, desc, sizeof(desc));\n',
    to: '        __android_log_print(ANDROID_LOG_INFO, "RNLlama", "diag: model details start");\n        char desc[1024];\n        llama_model_desc(ctx->model, desc, sizeof(desc));\n' },
  // ---- THE FIX (2026-09-12): the chat-template capability probe threw
  //      std::runtime_error for Qwen3-1.7B GGUFs (older Qwen3 Jinja template
  //      that this engine's Jinja implementation cannot render in the probe),
  //      and the promise wrapper turned that into a failed model load. The
  //      probe is informational — this app builds ChatML prompts itself — so a
  //      template the engine cannot probe must never fail the load.
  { file: RN_JSI,
    from: '        // Chat template capabilities\n        jsi::Object chatTemplates(runtime);\n        bool llamaChat = ctx->validateModelChatTemplate(false, nullptr);\n',
    to: '        // Chat template capabilities\n        jsi::Object chatTemplates(runtime);\n        bool llamaChat = false;\n        try {\n        llamaChat = ctx->validateModelChatTemplate(false, nullptr);\n' },
  { file: RN_JSI,
    from: '        chatTemplates.setProperty(runtime, "jinja", jinja);\n        model.setProperty(runtime, "chatTemplates", chatTemplates);\n',
    to: '        chatTemplates.setProperty(runtime, "jinja", jinja);\n' +
        '        } catch (const std::exception& e) {\n' +
        '            __android_log_print(ANDROID_LOG_WARN, "RNLlama", "diag: chat template probe failed: %s — reporting no template support (local patch)", e.what());\n' +
        '            chatTemplates.setProperty(runtime, "llamaChat", false);\n' +
        '            jsi::Object jinjaFallback(runtime);\n' +
        '            jinjaFallback.setProperty(runtime, "default", false);\n' +
        '            jinjaFallback.setProperty(runtime, "toolUse", false);\n' +
        '            chatTemplates.setProperty(runtime, "jinja", jinjaFallback);\n' +
        '        } catch (...) {\n' +
        '            __android_log_print(ANDROID_LOG_WARN, "RNLlama", "diag: chat template probe failed (non-std exception) — reporting no template support (local patch)");\n' +
        '            chatTemplates.setProperty(runtime, "llamaChat", false);\n' +
        '            jsi::Object jinjaFallback(runtime);\n' +
        '            jinjaFallback.setProperty(runtime, "default", false);\n' +
        '            jinjaFallback.setProperty(runtime, "toolUse", false);\n' +
        '            chatTemplates.setProperty(runtime, "jinja", jinjaFallback);\n' +
        '        }\n' +
        '        model.setProperty(runtime, "chatTemplates", chatTemplates);\n' },
];

for (const p of patches) {
  const abs = path.join(__dirname, '..', p.file);
  if (!fs.existsSync(abs)) { console.log(`[patches] skip (missing) ${p.file}`); continue; }
  const src = fs.readFileSync(abs, 'utf8');
  const marker = p.once || p.to;
  if (src.includes(marker)) { console.log(`[patches] already applied ${p.file} :: ${(p.once || p.to).slice(0, 40).replace(/\n/g, ' ')}`); continue; }
  if (!src.includes(p.from)) { console.error(`[patches] ANCHOR NOT FOUND in ${p.file} :: ${p.from.slice(0, 60).replace(/\n/g, ' ')}`); process.exitCode = 1; continue; }
  const out = p.all ? src.split(p.from).join(p.to) : src.replace(p.from, p.to);
  fs.writeFileSync(abs, out, 'utf8');
  console.log(`[patches] applied ${p.file} :: ${(p.once || p.to).slice(0, 40).replace(/\n/g, ' ')}`);
}
