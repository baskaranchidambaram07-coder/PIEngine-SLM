// Governance telemetry — posts USAGE METADATA ONLY to the portal.
// Never sends chat text, queries, KB passages, or answers: only which
// agent/model, token counts, tokens/sec, load times, KB hit counts, and
// error strings. Fire-and-forget; failures never affect the chat.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { getPortalUrl } from './portal';
import { PORTAL_HEADERS } from './config';
import { log } from './logger';

const DEVICE_ID_KEY = 'device_id_v1';
let cachedId: string | null = null;

function randomId(): string {
  return 'dev-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

async function deviceId(): Promise<string> {
  if (cachedId) return cachedId;
  let id = await AsyncStorage.getItem(DEVICE_ID_KEY);
  if (!id) { id = randomId(); await AsyncStorage.setItem(DEVICE_ID_KEY, id); }
  cachedId = id;
  return id;
}

function deviceModel(): string {
  const c: any = Platform.constants || {};
  // Android exposes Brand/Model/Manufacturer on Platform.constants
  const brand = c.Brand || c.Manufacturer || '';
  const model = c.Model || '';
  const label = [brand, model].filter(Boolean).join(' ').trim();
  return label || `Android ${Platform.Version}`;
}

export type TelemetryEvent = {
  // 'uninstall' releases the Studio's delete guard for this device — report it
  // whenever an agent is removed on the handset (the Studio refuses to delete
  // an agent that a device still holds).
  event: 'install' | 'uninstall' | 'chat' | 'model_load' | 'error' | 'attachment' | 'guard_block';
  agent_id?: string;
  agent_version?: number;
  model_id?: string;
  tokens?: number;
  tok_per_sec?: number;
  prefill_tokens?: number;
  load_ms?: number;
  kb_hits?: number;
  duration_ms?: number;
  ok?: number;
  detail?: string;      // error/system string only — never user content
};

export async function reportTelemetry(ev: TelemetryEvent): Promise<void> {
  try {
    const base = await getPortalUrl();
    const body = { device_id: await deviceId(), device_model: deviceModel(), ...ev };
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    await fetch(`${base}/api/telemetry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...PORTAL_HEADERS },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timer);
  } catch (e: any) {
    // governance is best-effort; note it locally but never surface to the user
    log(`telemetry post skipped: ${String(e?.message || e).slice(0, 60)}`);
  }
}
