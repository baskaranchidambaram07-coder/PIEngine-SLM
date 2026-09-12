// llama.rn lifecycle + prompt building + streaming with Qwen3 tag filtering.
// Direct port of runtime/llm.py: strips <think> blocks, surfaces <tool_call>
// payloads as structured events, and builds ChatML prompts manually so
// behaviour is identical to the verified web runtime.
import {
  addNativeLogListener, initLlama, loadLlamaModelInfo, LlamaContext,
  toggleNativeLog,
} from 'llama.rn';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ADAPTERS_DIR, MODELS_DIR } from './config';
import { DEFAULT_EMBEDDER, EmbedderSpec } from './embedder';
import { log } from './logger';

// Models that failed to initialize on THIS device are remembered so we never
// re-attempt them — repeated heavy native inits can hard-crash the app. Keyed
// with a schema tag bumped when the init path changes, so a fixed build gets a
// clean slate to re-evaluate.
const BLOCK_KEY = 'blocked_models_v2_5';
async function blockedModels(): Promise<string[]> {
  try { return JSON.parse((await AsyncStorage.getItem(BLOCK_KEY)) || '[]'); }
  catch { return []; }
}
async function isModelBlocked(file: string): Promise<boolean> {
  return (await blockedModels()).includes(file);
}
async function markModelFailed(file: string): Promise<void> {
  const s = new Set(await blockedModels()); s.add(file);
  await AsyncStorage.setItem(BLOCK_KEY, JSON.stringify([...s]));
}
async function markModelOk(file: string): Promise<void> {
  const s = new Set(await blockedModels());
  if (s.delete(file)) await AsyncStorage.setItem(BLOCK_KEY, JSON.stringify([...s]));
}

/** Models this device has blocked after a failed load (for ⚙ settings). */
export async function listBlockedModels(): Promise<string[]> {
  return blockedModels();
}

/** Forget every block so each model gets ONE fresh attempt. A block is meant
 *  to stop repeat-crash loops, not to be permanent: memory conditions change
 *  (other apps closed, a reboot), and a user who saw the model run elsewhere
 *  deserves a way to try again without reinstalling the app. */
export async function unblockAllModels(): Promise<number> {
  const s = await blockedModels();
  await AsyncStorage.removeItem(BLOCK_KEY);
  await log(`llm: cleared ${s.length} blocked model(s) at user request`);
  return s.length;
}

let chatCtx: LlamaContext | null = null;
let chatModelFile: string | null = null;
let chatAdapterKey = '';
let embedCtx: LlamaContext | null = null;
let embedFile: string | null = null;   // which embedder GGUF embedCtx holds

// Capture the native engine's own log lines — the only way to see llama.cpp's
// real error text behind generic JSI "Unknown error" rejections.
const nativeTail: string[] = [];
let nativeLogStarted = false;
async function ensureNativeLogCapture() {
  if (nativeLogStarted) return;
  nativeLogStarted = true;
  try {
    await toggleNativeLog(true);
    addNativeLogListener((level: string, text: string) => {
      nativeTail.push(`[${level}] ${text}`.slice(0, 220));
      while (nativeTail.length > 80) nativeTail.shift();
    });
    await log('llm: native engine log capture enabled');
  } catch (e: any) {
    await log(`llm: native log capture unavailable: ${String(e?.message || e).slice(0, 100)}`);
  }
}

async function dumpNativeTail(context: string, lines = 30) {
  await log(`llm: ---- native engine log tail (${context}) ----`);
  for (const l of nativeTail.slice(-lines)) await log(`  ${l}`);
  await log('llm: ---- end native log ----');
}

export type LoraAdapter = { file: string; scale?: number };

export async function ensureChatModel(
  modelFile: string, onStatus?: (s: string) => void,
  expectedBytes?: number, adapter?: LoraAdapter | null,
): Promise<LlamaContext> {
  // The adapter is part of the context identity: same GGUF with a different
  // adapter is different weights, so a cached context cannot be reused.
  const adapterKey = adapter ? `${adapter.file}@${adapter.scale ?? 1}` : '';
  if (chatCtx && chatModelFile === modelFile && chatAdapterKey === adapterKey) return chatCtx;

  // SAFETY GATE (first, before any native work): if this model already failed
  // to init on THIS device, never re-attempt. Repeated heavy native inits on a
  // memory-tight device can hard-crash the app (uncatchable). Fail fast with
  // guidance. Exits before releasing the embedder or touching native code.
  if (await isModelBlocked(modelFile)) {
    await log(`llm: ${modelFile} is blocked (previously failed on this device) — not re-attempting`);
    // Say only what is known. An earlier version of this message asserted
    // "needs more memory" — a guess that turned out wrong (the real cause was a
    // chat-template probe in the engine wrapper) and that steered debugging
    // for weeks. See docs/android-1.7b-root-cause.md.
    throw new Error(
      `${modelFile} failed to load on this device earlier, so the app is not retrying it automatically ` +
      '(a repeated failed load can crash the app). Use ⚙ → "Retry blocked models" to try again — ' +
      'close other apps first — or pick an agent on a smaller model. The 📋 log has the engine\'s error text.');
  }

  if (chatCtx) { await chatCtx.release(); chatCtx = null; }
  await ensureNativeLogCapture();

  // Single-context pattern (like PocketPal): release the embedder before the
  // big model init — holding two contexts during init is a difference from
  // the proven configuration and a suspected failure trigger.
  if (embedCtx) {
    await embedCtx.release().catch(() => {});
    embedCtx = null; embedFile = null;
    await log('llm: released embedder before chat init (single-context mode)');
  }

  const ReactNativeBlobUtil = require('react-native-blob-util').default;
  const path = `${MODELS_DIR}/${modelFile}`;
  const stat = await ReactNativeBlobUtil.fs.stat(path).catch(() => null);
  const sizeBytes = Number(stat?.size ?? 0);
  await log(`llm: model file ${modelFile} = ${sizeBytes} bytes on disk` +
    (expectedBytes ? ` (expected ${expectedBytes})` : ''));
  // exact-byte integrity check (manifest carries the file's true size)
  if (expectedBytes && sizeBytes !== expectedBytes) {
    await ReactNativeBlobUtil.fs.unlink(path).catch(() => {});
    const msg = `model file is incomplete on device (${(sizeBytes / 1048576).toFixed(0)} MB of ` +
      `${(expectedBytes / 1048576).toFixed(0)} MB). It has been removed — go to the Agent Store ` +
      'and tap Install to re-download. Keep the app in the foreground until install completes.';
    await log(`llm: ${msg}`);
    throw new Error(msg);
  }

  // Diagnostic split: metadata-only read (no context creation). If THIS
  // fails, the problem is gguf metadata marshalling; if it succeeds, the
  // problem is context/memory setup.
  try {
    const mi: any = await loadLlamaModelInfo(path);
    await log(`llm: modelInfo OK (${String(mi?.['general.name'] ?? 'gguf')} · ${Object.keys(mi ?? {}).length} keys)`);
  } catch (e: any) {
    await log(`llm: modelInfo FAILED: ${String(e?.message || e).slice(0, 200)}`);
  }

  const big = /1\.7b|3b|4b/i.test(modelFile) || (expectedBytes ?? 0) > 750 * 1024 * 1024;

  onStatus?.('Loading model into RAM…');
  const t0 = Date.now();
  // ONE init attempt only. Earlier multi-attempt ladders hard-crashed the app
  // on memory-tight devices because repeated failed native inits exhaust
  // native memory. A single attempt is a catchable JS rejection. Settings
  // scale to model size: large models get a memory-conservative config that
  // still serves typical agent prompts; small models get full context.
  const cfg = big ? { n_ctx: 2048, n_batch: 256 } : { n_ctx: 2048, n_batch: 512 };
  // Large models only: an 8-bit K cache trims the 224 MiB the native log shows
  // for a 2048-token f16 KV cache by a quarter. Only K — llama.cpp needs flash
  // attention for a quantised V cache, and flash attention is off here on
  // purpose. The 0.6B path is untouched: it is the configuration proven on
  // every handset so far.
  const kvQuant = big ? { cache_type_k: 'q8_0' as const } : {};

  // A LoRA adapter is applied at init rather than merged into the GGUF, so the
  // base model stays shared across every agent on the device. If the adapter
  // file is missing the agent still runs — on stock weights — because failing
  // the chat would make a 30 MB download a hard dependency of a model that is
  // already present and working.
  let loraList: Array<{ path: string; scaled: number }> | undefined;
  if (adapter) {
    const lp = `${ADAPTERS_DIR}/${adapter.file}`;
    if (await ReactNativeBlobUtil.fs.exists(lp)) {
      loraList = [{ path: lp, scaled: adapter.scale ?? 1.0 }];
      await log(`llm: applying LoRA adapter ${adapter.file} (scale ${adapter.scale ?? 1.0})`);
    } else {
      await log(`llm: adapter ${adapter.file} not on device — running stock weights`);
    }
  }

  await log(`llm: loading chat model ${modelFile} (n_ctx=${cfg.n_ctx}, n_batch=${cfg.n_batch}, big=${big})`);
  let lastProgress = 0;
  try {
    // Param set mirrors PocketPal's Android config (proven on real devices):
    // CPU-only, FA off, unified KV, single slot, n_ubatch=n_batch.
    chatCtx = await initLlama({
      model: path,
      n_ctx: cfg.n_ctx,
      n_batch: cfg.n_batch,
      n_ubatch: cfg.n_batch,
      n_threads: 4,
      n_parallel: 1,
      kv_unified: true,
      flash_attn_type: 'off',
      use_mmap: true,
      use_mlock: false,
      n_gpu_layers: 0,
      no_gpu_devices: true,
      ...kvQuant,
      ...(loraList ? { lora_list: loraList } : {}),
    } as any, (progress: number) => {
      if (progress >= lastProgress + 25 || progress === 100) {
        lastProgress = progress;
        log(`llm: load progress ${progress}%`);
      }
    });
  } catch (e: any) {
    const lastErr = String(e?.message || e).slice(0, 200);
    await log(`llm: init FAILED (progress ${lastProgress}%): ${lastErr}`);
    await dumpNativeTail('init');
    // The engine wrapper's real error text only reaches logcat; pull it in.
    const { dumpLogcatToLog } = require('./logcat');
    await dumpLogcatToLog(`init ${modelFile}`);
    chatCtx = null;
    await markModelFailed(modelFile);  // never re-attempt → no repeat-crash
    throw new Error(
      `Couldn't start ${modelFile} on this device: ${lastErr}. ` +
      'It is blocked for now so a repeated failed load cannot crash the app; ⚙ → "Retry blocked models" ' +
      'tries again. The 📋 log holds the engine\'s error text — share it if this persists.');
  }
  await markModelOk(modelFile);  // clear any stale block; this model works here
  const info: any = chatCtx as any;
  await log(`llm: init OK · loaded in ${Date.now() - t0}ms (gpu=${info.gpu ?? '?'} ${info.reasonNoGPU ?? ''})`);
  chatModelFile = modelFile;
  chatAdapterKey = adapterKey;
  return chatCtx;
}

export async function ensureEmbedder(spec: EmbedderSpec = DEFAULT_EMBEDDER): Promise<LlamaContext> {
  if (embedCtx && embedFile === spec.file) return embedCtx;
  if (embedCtx) { await embedCtx.release().catch(() => {}); embedCtx = null; embedFile = null; }
  await ensureNativeLogCapture();
  const t0 = Date.now();
  await log(`llm: loading embedder ${spec.file} (${spec.dim}-d, ${spec.pooling} pooling)`);
  try {
    // Pooling comes from the catalogue entry so the device matches the Studio
    // exactly (bge = CLS, MiniLM/nomic = mean), not from whatever the GGUF
    // header happens to declare.
    embedCtx = await initLlama({
      model: `${MODELS_DIR}/${spec.file}`,
      n_ctx: 512,
      n_threads: 2,
      embedding: true,
      pooling_type: spec.pooling,
      flash_attn_type: 'off',
      n_gpu_layers: 0,
      no_gpu_devices: true,
    } as any);
    embedFile = spec.file;
  } catch (e: any) {
    await log(`llm: embedder init FAILED: ${String(e?.message || e).slice(0, 200)}`);
    throw new Error(`embedder init failed: ${String(e?.message || e).slice(0, 200)}`);
  }
  await log(`llm: embedder loaded in ${Date.now() - t0}ms`);
  return embedCtx;
}

export async function embedText(text: string, isQuery: boolean,
                                spec: EmbedderSpec = DEFAULT_EMBEDDER): Promise<Float32Array> {
  const ctx = await ensureEmbedder(spec);
  // Prefixes are the catalogue's, applied identically in core/embeddings.py.
  const input = (isQuery ? spec.query_prefix : spec.passage_prefix) + text;
  let res: any;
  try {
    res = await ctx.embedding(input);
  } catch (e: any) {
    await log(`llm: embedding() FAILED: ${String(e?.message || e).slice(0, 200)}`);
    throw new Error(`embedding failed: ${String(e?.message || e).slice(0, 200)}`);
  }
  const vec = Float32Array.from(res.embedding ?? res);
  if (!vec.length) {
    await log('llm: embedding returned empty vector');
    throw new Error('embedding returned empty vector');
  }
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

// ---------------------------------------------------------------- vision

/**
 * Load the vision model (base GGUF + multimodal projector) into the single
 * chat slot. Same safety pattern as ensureChatModel: blocked-model gate first,
 * one init attempt, block on failure so a memory-tight device never
 * repeat-crashes. The context replaces the chat model (single-context rule);
 * the next text turn reloads the agent's own model.
 */
export async function ensureVisionModel(
  modelFile: string, mmprojFile: string, onStatus?: (s: string) => void,
): Promise<LlamaContext> {
  const key = `${modelFile}+${mmprojFile}`;
  if (chatCtx && chatModelFile === key) return chatCtx;
  if (await isModelBlocked(key)) {
    throw new Error(
      `The vision model (${modelFile}) failed to load on this device earlier and is not retried automatically. ` +
      'Use ⚙ → "Retry blocked models" to try again, or turn Vision off; images are still read with on-device OCR.');
  }
  if (chatCtx) { await chatCtx.release().catch(() => {}); chatCtx = null; chatModelFile = null; chatAdapterKey = ''; }
  if (embedCtx) { await embedCtx.release().catch(() => {}); embedCtx = null; embedFile = null; }
  await ensureNativeLogCapture();

  onStatus?.('Loading vision model into RAM…');
  const t0 = Date.now();
  await log(`llm: loading vision model ${modelFile} + ${mmprojFile}`);
  let ctx: LlamaContext;
  try {
    ctx = await initLlama({
      model: `${MODELS_DIR}/${modelFile}`,
      n_ctx: 3072,
      n_batch: 256,
      n_ubatch: 256,
      n_threads: 4,
      n_parallel: 1,
      kv_unified: true,
      flash_attn_type: 'off',
      ctx_shift: false,          // required for media token positioning
      use_mmap: true,
      use_mlock: false,
      n_gpu_layers: 0,
      no_gpu_devices: true,
    } as any);
  } catch (e: any) {
    await log(`llm: vision init FAILED: ${String(e?.message || e).slice(0, 200)}`);
    await dumpNativeTail('vision init');
    const { dumpLogcatToLog } = require('./logcat');
    await dumpLogcatToLog(`vision init ${modelFile}`);
    await markModelFailed(key);
    throw new Error(`Couldn't start the vision model on this device: ${String(e?.message || e).slice(0, 160)}. ` +
      'It is blocked for now (⚙ → "Retry blocked models" tries again); turn Vision off to use OCR only.');
  }
  let ok = false;
  try {
    ok = await (ctx as any).initMultimodal({
      path: `${MODELS_DIR}/${mmprojFile}`, use_gpu: false, image_max_tokens: 768,
    });
  } catch (e: any) {
    await log(`llm: initMultimodal FAILED: ${String(e?.message || e).slice(0, 200)}`);
  }
  if (!ok) {
    await ctx.release().catch(() => {});
    await markModelFailed(key);
    throw new Error('The vision projector could not be loaded on this device. Vision is now blocked; images still work via OCR.');
  }
  await markModelOk(key);
  chatCtx = ctx;
  chatModelFile = key;
  chatAdapterKey = '';
  await log(`llm: vision model ready in ${Date.now() - t0}ms`);
  return ctx;
}

/** One image + question through the vision model, streaming filtered tokens. */
export async function completionWithImage(
  ctx: LlamaContext, system: string, question: string, imagePath: string,
  generation: any, onEvent: (ev: StreamEvent) => void,
): Promise<{ text: string; stats: CompletionStats | null }> {
  const filter = new TagFilter();
  let raw = '';
  const url = imagePath.startsWith('file://') ? imagePath : `file://${imagePath}`;
  const result: any = await ctx.completion(
    {
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: [
          { type: 'text', text: question },
          { type: 'image_url', image_url: { url } },
        ] },
      ],
      n_predict: generation?.max_tokens ?? 512,
      temperature: generation?.temperature ?? 0.3,
      top_p: generation?.top_p ?? 0.8,
      stop: ['<|im_end|>', '<|im_start|>'],
    } as any,
    (data: any) => {
      const piece = data?.token ?? '';
      if (!piece) return;
      raw += piece;
      for (const ev of filter.push(piece)) onEvent(ev);
    },
  );
  for (const ev of filter.push('', true)) onEvent(ev);
  const t = result?.timings;
  const stats: CompletionStats | null = t
    ? { tokens: t.predicted_n ?? 0, tok_per_sec: Math.round((t.predicted_per_second ?? 0) * 10) / 10,
        prefill_tokens: t.prompt_n ?? 0, prefill_per_sec: Math.round((t.prompt_per_second ?? 0) * 10) / 10 }
    : null;
  return { text: raw, stats };
}

export async function releaseAll() {
  if (chatCtx) { await chatCtx.release(); chatCtx = null; chatModelFile = null; chatAdapterKey = ''; }
  if (embedCtx) { await embedCtx.release(); embedCtx = null; embedFile = null; }
}

// ---------------------------------------------------------------- prompts

const TOOLS_HEADER = (toolLines: string) => `

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${toolLines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>`;

export function buildSystemPrompt(manifest: any): string {
  let prompt = manifest.system_prompt || 'You are a helpful enterprise assistant.';
  const tools = manifest.tools || [];
  if (tools.length) {
    const lines = tools.map((t: any) => JSON.stringify({
      type: 'function',
      function: {
        name: t.name,
        description: t.description || '',
        parameters: t.parameters || { type: 'object', properties: {} },
      },
    })).join('\n');
    prompt += TOOLS_HEADER(lines);
  }
  return prompt + ' /no_think';
}

export type ChatTurn = { role: 'user' | 'assistant'; content: string };

export function buildPrompt(system: string, turns: ChatTurn[]): string {
  let p = `<|im_start|>system\n${system}<|im_end|>\n`;
  for (const t of turns) {
    p += `<|im_start|>${t.role}\n${t.content}<|im_end|>\n`;
  }
  return p + '<|im_start|>assistant\n';
}

// ------------------------------------------------------------ tag filtering

import { StreamEvent, TagFilter } from './stream';
export type { StreamEvent };

export type CompletionStats = {
  tokens: number;
  tok_per_sec: number;
  prefill_tokens: number;
  prefill_per_sec: number;
};

export async function streamCompletion(
  ctx: LlamaContext,
  prompt: string,
  generation: any,
  onEvent: (ev: StreamEvent) => void,
): Promise<{ text: string; stats: CompletionStats | null }> {
  const filter = new TagFilter();
  let raw = '';
  const result: any = await ctx.completion(
    {
      prompt,
      n_predict: generation?.max_tokens ?? 768,
      temperature: generation?.temperature ?? 0.7,
      top_p: generation?.top_p ?? 0.8,
      stop: ['<|im_end|>', '<|im_start|>'],
    },
    (data: any) => {
      const piece = data?.token ?? '';
      if (!piece) return;
      raw += piece;
      for (const ev of filter.push(piece)) onEvent(ev);
    },
  );
  for (const ev of filter.push('', true)) onEvent(ev);

  const t = result?.timings;
  const stats: CompletionStats | null = t
    ? {
        tokens: t.predicted_n ?? 0,
        tok_per_sec: Math.round((t.predicted_per_second ?? 0) * 10) / 10,
        prefill_tokens: t.prompt_n ?? 0,
        prefill_per_sec: Math.round((t.prompt_per_second ?? 0) * 10) / 10,
      }
    : null;
  return { text: raw, stats };
}
