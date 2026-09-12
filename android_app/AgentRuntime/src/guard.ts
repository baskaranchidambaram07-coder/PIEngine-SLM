// PII Guard — on-device port of runtime/guard.py.
//
// The guard is a per-agent guardrail: it runs only when the installed
// manifest carries `guardrails.pii = true` (set by the checkbox in the Studio).
// Off means the plain flow — no scan of the message, the file or the answer.
//
// Two detectors, in order:
//   1. src/pii.ts — deterministic regex + checksums. Milliseconds, relied on.
//   2. the model judge — the agent's own on-device SLM prompted as a strict
//      reviewer returning JSON. Every span it reports must occur VERBATIM in
//      the text or it is dropped (a small model invents plausible numbers).
//      It is slow on a handset, so it runs only on turns that carry a file,
//      on the message and on at most JUDGE_CHUNKS chunks of the document.
//
// Masking: first three + last two characters (src/pii.ts).
import * as pii from './pii';
import { log } from './logger';

export const GUARD_NAME = 'PII Guard';

// Mirrors runtime/guard.py DEFAULT_POLICY, fixed on device (no policy file).
export const POLICY = {
  judgeOnAttachments: true,   // judge the message + document chunks when a file is attached
  judgeChunks: 1,             // document chunks judged (each is a 10-40 s model call here)
  judgeMaxChars: 2000,
  judgeMaxTokens: 200,
  blockNames: false,
  blockKinds: new Set<pii.Kind>(['secret', 'identity', 'contact']),
  redactOutput: true,
};

export type Verdict = {
  blocked: boolean;
  findings: pii.Finding[];
  summary: pii.Summary[];
  judged: boolean;
  judgeNote: string;
  ms: number;
  source: string;
};

/** chatFn(systemPrompt, userText) -> the model's full reply (tags already filtered). */
export type ChatFn = (system: string, user: string) => Promise<string>;

export function piiEnabled(manifest: any): boolean {
  return !!(manifest?.guardrails && manifest.guardrails.pii);
}

// ------------------------------------------------------------------ regex stage

export function scan(text: string, source: string): pii.Finding[] {
  return pii.scan(text, source).filter(f => POLICY.blockKinds.has(f.kind));
}

function verdict(findings: pii.Finding[], source: string, t0: number,
                 judged = false, judgeNote = ''): Verdict {
  return { blocked: findings.length > 0, findings, summary: pii.summarize(findings),
           judged, judgeNote, ms: Date.now() - t0, source };
}

export function inspect(text: string, source: string): Verdict {
  const t0 = Date.now();
  return verdict(scan(text, source), source, t0);
}

// ------------------------------------------------------------------ model judge

const JUDGE_SYSTEM = (names: boolean) => `You are PII Guard, a strict data-protection reviewer for an enterprise assistant.
Read the TEXT and list every item that is personal data about a private individual, or a credential:
- a private individual's home address
- government identity numbers (SSN, Aadhaar, PAN, passport, driving licence, Emirates ID, tax id)
- personal bank account, card or IBAN numbers
- passwords, PINs, API keys, access tokens, secret keys, private keys
- health, salary or biometric details tied to a person${names ? '\n- full names of private individuals' : ''}

NOT personal data — never report these: a shop, cafe, office, venue or company address; company names;
invoice, order, table, receipt or reference numbers; prices and totals; ordinary dates and times;
product names; job titles.

Reply with ONLY this JSON, nothing else:
{"findings":[{"type":"address|government_id|financial|credential|health|name","value":"<the exact text, copied verbatim from TEXT>"}]}
If there is nothing, reply {"findings":[]}. Never invent a value. Never explain. /no_think`;

const JUDGE_KIND: Record<string, pii.Kind> = {
  credential: 'secret', address: 'identity', government_id: 'identity',
  financial: 'identity', health: 'identity', name: 'identity',
};
const JUDGE_LABEL: Record<string, string> = {
  credential: 'Credential (model judge)', address: 'Home address',
  government_id: 'Government ID number', financial: 'Financial account number',
  health: 'Health / salary detail', name: 'Personal name',
};

function plausible(typ: string, value: string): boolean {
  const hasDigit = /\d/.test(value);
  if (typ === 'government_id' || typ === 'financial' || typ === 'address') return hasDigit;
  if (typ === 'credential') {
    const v = value.replace(/^["'`]+|["'`]+$/g, '');
    return v.length >= 8 && (hasDigit || v.toLowerCase() !== v || !/^[A-Za-z]+$/.test(v));
  }
  return true;
}

function escapeRx(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function locate(text: string, value: string): number {
  const pos = text.indexOf(value);
  if (pos >= 0) return pos;
  const compact = value.replace(/\s+/g, '');
  if (compact.length < 3) return -1;
  const rx = new RegExp(compact.split('').map(escapeRx).join('\\s*'));
  const m = rx.exec(text);
  return m ? m.index : -1;
}

/** Ask the model. Returns validated findings and a diagnostic note. Never throws. */
export async function judge(text: string, source: string, chatFn: ChatFn,
                            known: pii.Finding[] = []): Promise<{ findings: pii.Finding[]; note: string }> {
  const t = text.slice(0, POLICY.judgeMaxChars);
  if (!t.trim()) return { findings: [], note: 'empty' };
  let reply = '';
  try {
    reply = await chatFn(JUDGE_SYSTEM(POLICY.blockNames), `TEXT:\n"""\n${t}\n"""`);
  } catch (e: any) {
    return { findings: [], note: `judge unavailable: ${String(e?.message || e).slice(0, 120)}` };
  }
  const m = /\{[\s\S]*\}/.exec(reply || '');
  if (!m) return { findings: [], note: 'judge returned no JSON' };
  let data: any;
  try { data = JSON.parse(m[0]); }
  catch {
    try { data = JSON.parse(m[0].replace(/,\s*([}\]])/g, '$1')); }
    catch { return { findings: [], note: 'judge JSON unparsable' }; }
  }
  const items = Array.isArray(data?.findings) ? data.findings : null;
  if (!items) return { findings: [], note: 'judge JSON had no findings list' };

  const out: pii.Finding[] = [];
  let dropped = 0;
  for (const it of items.slice(0, 50)) {
    if (!it || typeof it !== 'object') continue;
    const typ = String(it.type ?? '').trim().toLowerCase();
    const val = String(it.value ?? '').trim();
    if (typ === 'name' && !POLICY.blockNames) continue;
    const kind = JUDGE_KIND[typ];
    if (!kind || !POLICY.blockKinds.has(kind) || val.length < 3) { dropped++; continue; }
    if (!plausible(typ, val)) { dropped++; continue; }
    const pos = locate(t, val);
    if (pos < 0) { dropped++; continue; }           // invented — the whole point of validation
    const f: pii.Finding = { kind, label: JUDGE_LABEL[typ], value: t.slice(pos, pos + val.length),
                             start: pos, end: pos + val.length, source, detector: 'judge' };
    if (known.some(k => f.start < k.end && k.start < f.end)) continue;   // regex already has it
    out.push(f);
  }
  return { findings: pii.dedupe(out), note: `judge: ${out.length} kept, ${dropped} dropped` };
}

/** Regex always; the judge when `chatFn` is given. */
export async function inspectText(text: string, source: string, chatFn?: ChatFn | null): Promise<Verdict> {
  const t0 = Date.now();
  let findings = scan(text, source);
  let judged = false;
  let note = '';
  if (chatFn) {
    const r = await judge(text, source, chatFn, findings);
    note = r.note;
    judged = note.startsWith('judge:');
    findings = pii.dedupe([...findings, ...r.findings]);
  }
  await log(`guard: ${source} -> ${findings.length ? 'BLOCK ' + findings.map(f => f.label).join(', ') : 'clear'}` +
            (note ? ` (${note})` : '') + ` in ${Date.now() - t0}ms`);
  return verdict(findings, source, t0, judged, note);
}

/** Regex over every chunk; judge over the first `judgeLimit` of them. */
export async function inspectChunks(chunks: string[], source: string, chatFn?: ChatFn | null,
                                    judgeLimit = POLICY.judgeChunks): Promise<Verdict> {
  const t0 = Date.now();
  let findings: pii.Finding[] = [];
  for (const c of chunks) findings = findings.concat(scan(c, source));
  let judged = false;
  const notes: string[] = [];
  if (chatFn && judgeLimit > 0 && findings.length === 0) {   // regex already decided if it found anything
    for (const c of chunks.slice(0, judgeLimit)) {
      const r = await judge(c, source, chatFn, []);
      notes.push(r.note);
      judged = judged || r.note.startsWith('judge:');
      findings = findings.concat(r.findings);
      if (r.findings.length) break;
    }
  }
  await log(`guard: ${source} (${chunks.length} chunks) -> ${findings.length ? 'BLOCK' : 'clear'} in ${Date.now() - t0}ms`);
  return verdict(findings, source, t0, judged, notes.join('; ').slice(0, 200));
}

// ------------------------------------------------------------------ output

export function refusalMessage(where: string, summary: pii.Summary[]): string {
  const lines = [`🛡 ${GUARD_NAME}: I found sensitive data in ${where}, so I am not ready to proceed further.`];
  for (const g of summary.slice(0, 8)) {
    const more = g.count > g.examples.length ? ` (+${g.count - g.examples.length} more)` : '';
    lines.push(`• ${g.label} — ${g.kind === 'secret' ? 'a secret' : 'PII'}: ${g.examples.join(', ')}${more}`);
  }
  if (summary.length > 8) lines.push(`• …and ${summary.length - 8} more categories`);
  lines.push('Remove or redact this information and try again.');
  return lines.join('\n');
}

/** Mask anything in a model answer that the regex stage recognises. */
export function redactText(text: string): { text: string; summary: pii.Summary[] } {
  const findings = scan(text, 'model');
  if (!findings.length) return { text, summary: [] };
  return { text: pii.redact(text, findings), summary: pii.summarize(findings) };
}
