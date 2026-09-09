// Persistent in-app diagnostics log (ring buffer in AsyncStorage).
// Everything download/install/chat related is recorded so device-side
// failures can be shared and analysed without adb.
import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'app_logs_v1';
const MAX_LINES = 300;
let cache: string[] | null = null;
let listeners: Array<() => void> = [];

export async function getLogs(): Promise<string[]> {
  if (cache) return cache;
  try {
    cache = JSON.parse((await AsyncStorage.getItem(KEY)) || '[]');
  } catch {
    cache = [];
  }
  return cache!;
}

export async function log(msg: string): Promise<void> {
  const line = `${new Date().toISOString().slice(5, 19).replace('T', ' ')} ${msg}`;
  const arr = await getLogs();
  arr.push(line);
  while (arr.length > MAX_LINES) arr.shift();
  cache = arr;
  AsyncStorage.setItem(KEY, JSON.stringify(arr)).catch(() => {});
  console.log('[AGENT]', line);
  listeners.forEach(fn => fn());
}

export async function clearLogs(): Promise<void> {
  cache = [];
  await AsyncStorage.removeItem(KEY);
  listeners.forEach(fn => fn());
}

export function onLogsChanged(fn: () => void): () => void {
  listeners.push(fn);
  return () => { listeners = listeners.filter(f => f !== fn); };
}
