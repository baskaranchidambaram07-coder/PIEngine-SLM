// Own-process logcat access (android/.../LogcatModule.kt).
//
// The JS layer of the inference engine reports a failed load as the bare
// string "Unknown error" because a C++ exception thrown inside the engine
// library crosses a shared-object boundary and loses its type. The real
// message is still written to logcat by the engine wrapper, and an Android
// app may read its own logcat lines without any permission — so after a
// failed load we pull them into the in-app diagnostics log.
import { NativeModules } from 'react-native';
import { log } from './logger';

const Native: { dump?: (maxLines: number) => Promise<string> } | undefined = NativeModules.Logcat;

const INTERESTING = /rnllama|RNLlama|llama|ggml|libc\+\+|terminating|abort|SIGABRT|exception|Exception|error|Error|OutOfMemory|lowmemory|E\/|F\//;
const NOISE = /ReactNativeJS: \[AGENT\]|chatty|dumping logcat|\bViewRootImpl\b|InputMethodManager|Choreographer|\bOpenGLRenderer\b|hwui/i;

export function logcatAvailable(): boolean {
  return typeof Native?.dump === 'function';
}

/** Append the last relevant logcat lines to the in-app log. Best effort. */
export async function dumpLogcatToLog(context: string, maxLines = 600, keep = 45): Promise<void> {
  if (!logcatAvailable()) {
    await log(`logcat: native module not available (${context})`);
    return;
  }
  try {
    const raw = await Native!.dump!(maxLines);
    const lines = raw.split('\n').map(l => l.trimEnd())
      .filter(l => l && INTERESTING.test(l) && !NOISE.test(l));
    const tail = lines.slice(-keep);
    await log(`logcat: ---- last ${tail.length} relevant lines (${context}) ----`);
    for (const l of tail) await log(`  ${l.slice(0, 240)}`);
    await log('logcat: ---- end ----');
  } catch (e: any) {
    await log(`logcat: dump failed: ${String(e?.message || e).slice(0, 120)}`);
  }
}
