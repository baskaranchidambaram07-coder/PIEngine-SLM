// On-device tool dispatcher — parity with runtime/tools.py.
// builtin tools always work offline (AsyncStorage); http tools degrade
// gracefully with a structured error the model can explain.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Linking, NativeModules } from 'react-native';

const KEY = 'device_data_v1';

export type DeviceData = {
  action_items: any[];
  outbox: any[];
  notes: any[];
  events: any[];
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

// A device_data_v1 blob saved before "notes"/"events" existed has neither key
// — JSON.parse returns exactly what was written, nothing here ever merged in
// new keys — so an existing install would throw the first time either new
// tool ran. Same fix as runtime/tools.py's _load(): backfill, never drop.
export async function getDeviceData(): Promise<DeviceData> {
  const raw = await AsyncStorage.getItem(KEY);
  const data: DeviceData = raw
    ? JSON.parse(raw)
    : { action_items: [], outbox: [], notes: [], events: [], calendar: mockCalendar() };
  let changed = !raw;
  for (const key of ['notes', 'events'] as const) {
    if (!Array.isArray((data as any)[key])) { (data as any)[key] = []; changed = true; }
  }
  if (changed) await AsyncStorage.setItem(KEY, JSON.stringify(data));
  return data;
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
    // Real handoff: open the phone's own mail app with the draft pre-filled.
    // The local outbox record above is unchanged — this is additive, and if
    // no mail app answers the mailto: intent the local draft still stands.
    let opened = false;
    try {
      const url = `mailto:${encodeURIComponent(draft.to)}` +
        `?subject=${encodeURIComponent(draft.subject)}&body=${encodeURIComponent(draft.body)}`;
      opened = await Linking.canOpenURL(url);
      if (opened) await Linking.openURL(url);
    } catch { opened = false; }
    return {
      drafted: { id: draft.id, to: draft.to, subject: draft.subject, status: draft.status },
      note: opened
        ? 'Opened the mail app with this draft — review and tap Send. Never auto-sent.'
        : 'Draft saved to device outbox for user review. Never auto-sent. (No mail app answered on this device.)',
    };
  }

  if (name === 'schedule_event') {
    const start = args.start || new Date(Date.now() + 24 * 3600e3).toISOString();
    const startMs = Date.parse(start) || Date.now() + 24 * 3600e3;
    const durationMin = Number(args.duration_minutes) || 30;
    const event = {
      id: data.events.length + 1,
      title: args.title || 'Untitled event',
      start: new Date(startMs).toISOString(),
      duration_minutes: durationMin,
      description: args.description || '',
      attendees: args.attendees || [],
      status: 'scheduled — awaiting user confirmation',
      created: new Date().toISOString().slice(0, 10),
    };
    data.events.push(event);
    await save(data);
    // Real handoff: hand the event to the Calendar app for the user to save.
    // No permission is declared and nothing is inserted silently — the
    // Calendar app owns the actual write, same "draft, never auto-sent"
    // shape as draft_email above. Linking.openURL cannot express typed
    // Intent extras (only URL-scheme intents like mailto:), so this one
    // needs the small native module in android/.../CalendarModule.kt.
    let opened = false;
    try {
      const CalendarInsert = (NativeModules as any).CalendarInsert;
      if (CalendarInsert?.openInsert) {
        await CalendarInsert.openInsert(event.title, event.description, startMs, startMs + durationMin * 60e3);
        opened = true;
      }
    } catch { opened = false; }
    // Echo back an ack, not the full record: the model just wrote
    // "description"/"attendees" itself (still in this round's tool-call
    // args, which the UI shows alongside this result) — re-including them
    // here measurably blew the 2048-token on-device context budget by the
    // next tool-calling round (no such limit on the web runtime, where this
    // was fine). See src/config.ts's n_ctx note in llm.ts.
    return {
      scheduled: { id: event.id, title: event.title, start: event.start,
                   duration_minutes: event.duration_minutes, status: event.status },
      note: opened
        ? 'Opened the Calendar app with this event — review and tap Save.'
        : 'Saved on device; could not open the Calendar app on this build.',
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
