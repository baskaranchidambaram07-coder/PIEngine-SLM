// Portal sync: talks to the Studio web portal (published agents + bundles),
// installs bundles into app storage, and downloads GGUF models on demand.
// After install everything runs offline; the portal is only needed to sync.
import AsyncStorage from '@react-native-async-storage/async-storage';
import ReactNativeBlobUtil from 'react-native-blob-util';
import { unzip } from 'react-native-zip-archive';
import {
  AGENTS_DIR, DEFAULT_PORTAL, DISCOVERY_TIMEOUT_MS, DISCOVERY_URLS,
  EMBEDDER_FILE, EMBEDDER_URL, MODELS_DIR, PORTAL_HEADERS, PORTAL_PROBE_MS,
} from './config';
import { log } from './logger';

const PORTAL_KEY = 'portal_url_v1';
const fs = ReactNativeBlobUtil.fs;

export async function getPortalUrl(): Promise<string> {
  return (await AsyncStorage.getItem(PORTAL_KEY)) || DEFAULT_PORTAL;
}

export async function setPortalUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(PORTAL_KEY, url.trim().replace(/\/+$/, ''));
}

// --------------------------------------------------------------- discovery
// A cloudflare *quick* tunnel mints a new random hostname on every restart,
// which used to strand every installed handset until someone retyped the URL
// by hand. The Studio sits at a stable address and reports where the portal
// currently lives, so the app can re-resolve it on its own.
// With a *named* tunnel the portal URL is permanent and this never fires.

function fetchWithTimeout(url: string, ms: number): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  return fetch(url, { headers: PORTAL_HEADERS, signal: ctrl.signal })
    .finally(() => clearTimeout(timer));
}

async function portalAlive(base: string): Promise<boolean> {
  try {
    const r = await fetchWithTimeout(`${base}/api/published`, PORTAL_PROBE_MS);
    return r.ok;
  } catch {
    return false;
  }
}

/** The saved portal URL if it still answers, otherwise one re-resolved from a
 *  discovery host (and persisted). Falls back to the saved value, so an
 *  offline handset keeps whatever it had. */
export async function resolvePortalUrl(force = false): Promise<string> {
  const saved = await getPortalUrl();
  if (!force && (await portalAlive(saved))) return saved;

  log(`portal ${saved} not answering — asking ${DISCOVERY_URLS.length} discovery host(s)`);
  for (const host of DISCOVERY_URLS) {
    try {
      const r = await fetchWithTimeout(`${host}/api/portal-url`, DISCOVERY_TIMEOUT_MS);
      if (!r.ok) {
        log(`discovery ${host} → HTTP ${r.status}`);
        continue;
      }
      const body = await r.json();
      const url = String(body?.url || '').trim().replace(/\/+$/, '');
      if (!/^https?:\/\//.test(url)) {
        log(`discovery ${host} → unusable url ${JSON.stringify(body?.url)}`);
        continue;
      }
      if (url !== saved) {
        await setPortalUrl(url);
        log(`portal URL re-resolved → ${url} (${body?.source}, stable=${body?.stable})`);
      }
      return url;
    } catch (e: any) {
      log(`discovery ${host} failed: ${e?.message || e}`);
    }
  }
  log('discovery exhausted — keeping saved portal URL');
  return saved;
}

export type StoreEntry = {
  id: string;
  name: string;
  description: string;
  version: number;
  bundle: string;
  bundle_bytes: number;
  model: { id: string; name: string; file: string; download_url: string; size_gb: number };
  kb: { docs: number; chunks: number };
  tools: number;
  installed_version?: number;
};

export async function fetchStore(): Promise<StoreEntry[]> {
  const base = await resolvePortalUrl();
  const resp = await fetch(`${base}/api/published`, { headers: PORTAL_HEADERS });
  if (!resp.ok) throw new Error(`portal returned ${resp.status}`);
  const published: StoreEntry[] = await resp.json();
  const installed = await listInstalled();
  for (const p of published) {
    const local = installed.find(i => i.id === p.id);
    p.installed_version = local?.version;
  }
  return published;
}

export type InstalledAgent = {
  id: string;
  name: string;
  description: string;
  version: number;
  manifest: any;
  modelReady: boolean;
  embedderReady: boolean;
};

export async function listInstalled(): Promise<InstalledAgent[]> {
  if (!(await fs.exists(AGENTS_DIR))) return [];
  const ids = await fs.ls(AGENTS_DIR);
  const out: InstalledAgent[] = [];
  for (const id of ids) {
    const mPath = `${AGENTS_DIR}/${id}/manifest.json`;
    if (!(await fs.exists(mPath))) continue;
    try {
      const manifest = JSON.parse(await fs.readFile(mPath, 'utf8'));
      out.push({
        id,
        name: manifest.name,
        description: manifest.description || '',
        version: manifest.version,
        manifest,
        modelReady: await fs.exists(`${MODELS_DIR}/${manifest.model.file}`),
        embedderReady: await fs.exists(`${MODELS_DIR}/${EMBEDDER_FILE}`),
      });
    } catch {}
  }
  return out;
}

export type Progress = (label: string, done: number, total: number) => void;

/** Small files (bundles): download fully into memory, then write once.
 * Avoids ReactNativeBlobUtil's file-stream path, which proved unreliable on
 * real devices ("Download interrupted." despite the server sending all bytes). */
async function downloadSmall(url: string, dest: string, label: string, onProgress?: Progress) {
  await log(`GET(mem) ${url.split('/')[2]} ${url.split('/').pop()?.slice(0, 50)}`);
  onProgress?.(label, 0, 0);
  const res = await ReactNativeBlobUtil.fetch('GET', url, PORTAL_HEADERS);
  const status = res.info().status;
  if (status !== 200) {
    await log(`  -> HTTP ${status}`);
    throw new Error(`HTTP ${status}`);
  }
  await fs.writeFile(dest, res.base64(), 'base64');
  const stat = await fs.stat(dest);
  await log(`  -> OK ${stat.size} bytes (memory)`);
}

/** Large files (models): use Android's system Download Manager — its own
 * network stack, notification-shade progress, resume on flaky networks. DM
 * can only write to external storage, so download to public Downloads and
 * copy into app-internal storage afterwards. */
async function downloadLarge(url: string, dest: string, label: string, onProgress?: Progress) {
  const fileName = `agent_dl_${Date.now()}_${dest.split('/').pop()}`;
  const dmPath = `${fs.dirs.DownloadDir}/${fileName}`;
  await log(`GET(DM) ${url.split('/')[2]} ${url.split('/').pop()?.slice(0, 50)}`);
  onProgress?.(`${label} — via Android Download Manager, progress in notification shade`, 0, 0);
  const res = await ReactNativeBlobUtil.config({
    addAndroidDownloads: {
      useDownloadManager: true,
      notification: true,
      title: label,
      description: 'Enterprise Agents download',
      path: dmPath,
      mediaScannable: false,
    },
  }).fetch('GET', url, PORTAL_HEADERS)
    .progress({ interval: 2000 }, (received: number, total: number) =>
      onProgress?.(label, Number(received), Number(total)));
  const gotPath = (res as any).path ? (res as any).path() : dmPath;
  const stat = await fs.stat(gotPath);
  if (Number(stat.size) < 1024) {
    await fs.unlink(gotPath).catch(() => {});
    await log(`  -> suspiciously small: ${stat.size} bytes`);
    throw new Error(`download produced only ${stat.size} bytes`);
  }
  onProgress?.(`${label} — moving into app storage (keep app open)…`, 0, 0);
  if (await fs.exists(dest)) await fs.unlink(dest);
  await fs.cp(gotPath, dest);
  // verify the copy byte-for-byte in size — a backgrounded app can truncate it
  const copied = await fs.stat(dest);
  if (Number(copied.size) !== Number(stat.size)) {
    await fs.unlink(dest).catch(() => {});
    await log(`  -> copy truncated: ${copied.size} of ${stat.size} bytes`);
    throw new Error(`copy into app storage truncated (${copied.size}/${stat.size} bytes) — keep the app in the foreground`);
  }
  await fs.unlink(gotPath).catch(() => {});
  await log(`  -> OK ${stat.size} bytes (DM, copy verified)`);
}

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

/** Try each source URL in order, with retries per source. The thrown error
 * names the file and every failure reason. */
async function downloadWithFallback(
  urls: string[], dest: string, label: string,
  large: boolean, onProgress?: Progress,
) {
  const reasons: string[] = [];
  for (const url of urls) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        if (large) await downloadLarge(url, dest, label, onProgress);
        else await downloadSmall(url, dest, label, onProgress);
        return;
      } catch (e: any) {
        const reason = `${url.split('/')[2]} try${attempt}: ${String(e?.message || e).slice(0, 70)}`;
        reasons.push(reason);
        await log(`  -> FAIL ${reason}`);
        await sleep(1000 * attempt);
      }
    }
  }
  throw new Error(`${label} failed — ${reasons.join(' | ')}`);
}

/** Install/update an agent: bundle zip + chat model + embedder model. */
export async function installAgent(entry: StoreEntry, onProgress?: Progress): Promise<void> {
  const base = await getPortalUrl();
  await fs.mkdir(AGENTS_DIR).catch(() => {});
  await fs.mkdir(MODELS_DIR).catch(() => {});

  // 1. bundle (small: manifest + kb.sqlite)
  const zipPath = `${fs.dirs.CacheDir}/${entry.bundle}`;
  await downloadWithFallback([`${base}/bundles/${entry.bundle}`], zipPath, 'Agent bundle', false, onProgress);
  const target = `${AGENTS_DIR}/${entry.id}`;
  await fs.mkdir(target).catch(() => {});
  await unzip(zipPath, target);
  await fs.unlink(zipPath).catch(() => {});

  // 2. chat model (cached across agents) — Hugging Face CDN first (built for
  // multi-GB files), portal cache as fallback for HF-blocked networks
  const modelDest = `${MODELS_DIR}/${entry.model.file}`;
  if (!(await fs.exists(modelDest))) {
    await downloadWithFallback(
      [entry.model.download_url, `${base}/models/${encodeURIComponent(entry.model.file)}`],
      modelDest, `Model ${entry.model.name} (${entry.model.size_gb} GB)`, true, onProgress);
  }

  // 3. embedder (one per device, needed for RAG + inline KB)
  const embDest = `${MODELS_DIR}/${EMBEDDER_FILE}`;
  if (!(await fs.exists(embDest))) {
    await downloadWithFallback(
      [EMBEDDER_URL, `${base}/models/${EMBEDDER_FILE}`],
      embDest, 'Embedder (35 MB)', true, onProgress);
  }
  await log(`install complete: ${entry.id} v${entry.version}`);
}

export async function uninstallAgent(id: string): Promise<void> {
  await fs.unlink(`${AGENTS_DIR}/${id}`).catch(() => {});
}
