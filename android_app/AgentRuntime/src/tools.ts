// On-device tool dispatcher — parity with runtime/tools.py.
// builtin tools always work offline (AsyncStorage); http tools degrade
// gracefully with a structured error the model can explain.
import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'device_data_v1';

export type DeviceData = {
  action_items: any[];
  outbox: any[];
  calendar: any[];
};

function mockCalendar(): any[] {
  const today = new Date().toISOString().slice(0, 10);
  return [
    { time: '09:30', date: today, title: 'Acme Corp — Phase 2 steering committee' },
    { time: '13:00', date: today, title: 'Internal delivery sync — Acme programme' },
    { time: '16:00', date: today, title: 'Proposal review — retail analytics RFP' },
  ];
}

export async function getDeviceData(): Promise<DeviceData> {
  const raw = await AsyncStorage.getItem(KEY);
  if (raw) return JSON.parse(raw);
  const fresh = { action_items: [], outbox: [], calendar: mockCalendar() };
  await AsyncStorage.setItem(KEY, JSON.stringify(fresh));
  return fresh;
}

async function save(data: DeviceData) {
  await AsyncStorage.setItem(KEY, JSON.stringify(data));
}

export async function executeTool(toolDef: any, args: any): Promise<any> {
  if (!toolDef) return { error: 'tool is not attached to this agent' };
  const kind = toolDef.kind || 'builtin';
  if (kind === 'builtin') return executeBuiltin(toolDef.name, args || {});
  if (kind === 'http') return executeHttp(toolDef, args || {});
  return { error: `unsupported tool kind '${kind}'` };
}

async function executeBuiltin(name: string, args: any): Promise<any> {
  const data = await getDeviceData();

  if (name === 'get_todays_meetings') return { meetings: data.calendar };

  if (name === 'create_action_item') {
    const item = {
      id: data.action_items.length + 1,
      title: args.title || 'Untitled action',
      owner: args.owner || 'me',
      due_date: args.due_date || new Date(Date.now() + 7 * 864e5).toISOString().slice(0, 10),
      status: 'open',
      created: new Date().toISOString().slice(0, 10),
    };
    data.action_items.push(item);
    await save(data);
    return { created: item, note: 'Stored on device; will sync to Jira when online.' };
  }

  if (name === 'draft_email') {
    const draft = {
      id: data.outbox.length + 1,
      to: args.to || '',
      subject: args.subject || '',
      body: args.body || '',
      status: 'draft — awaiting user review',
    };
    data.outbox.push(draft);
    await save(data);
    return {
      drafted: { id: draft.id, to: draft.to, subject: draft.subject, status: draft.status },
      note: 'Draft saved to device outbox for user review. Never auto-sent.',
    };
  }

  return { error: `unknown builtin tool '${name}'` };
}

async function executeHttp(toolDef: any, args: any): Promise<any> {
  const cfg = toolDef.config || {};
  let url: string = cfg.url || '';
  for (const [k, v] of Object.entries(args)) {
    url = url.replace(`{${k}}`, encodeURIComponent(String(v)));
  }
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 6000);
    const resp = await fetch(url, {
      method: cfg.method || 'GET',
      headers: cfg.headers || {},
      signal: controller.signal,
    });
    clearTimeout(timer);
    const text = await resp.text();
    let body: any = text.slice(0, 2000);
    try { body = JSON.parse(text); } catch {}
    return { status: resp.status, body };
  } catch (e: any) {
    return {
      error: 'endpoint unreachable (device may be offline)',
      detail: String(e?.message || e).slice(0, 200),
      hint: 'Tell the user this tool needs connectivity and offer an offline alternative.',
    };
  }
}
