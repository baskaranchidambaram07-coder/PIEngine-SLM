// Deterministic detector for PII, passwords and secret/API keys.
//
// Line-for-line port of core/pii.py — the same rules, checksums and masking
// run on the phone as on the web runtime, and tests/test_pii.py's vectors are
// the contract for both. Change the two files together.
//
// Kinds: secret (passwords, keys, tokens), identity (government IDs, licences,
// cards, IBAN, DOB), contact (email, phone). Personal names are deliberately
// NOT detected here — an enterprise document is made of colleagues' names.
//
// Masking: values longer than five characters show their first three and last
// two characters; shorter ones show only the last character.

export type Kind = 'secret' | 'identity' | 'contact';

export type Finding = {
  kind: Kind;
  label: string;
  value: string;
  start: number;
  end: number;
  source: string;
  detector: 'regex' | 'judge';
};

export type Summary = { label: string; kind: Kind; count: number; examples: string[]; detector: string };

export const MASK_CHAR = '*';
export const VISIBLE_HEAD = 3;
export const VISIBLE_TAIL = 2;
export const MAX_FINDINGS = 200;

export function mask(value: string, head = VISIBLE_HEAD, tail = VISIBLE_TAIL): string {
  const v = value || '';
  const n = v.length;
  if (n === 0) return '';
  if (n <= 5) return MASK_CHAR.repeat(n - 1) + v.slice(-1);
  return v.slice(0, head) + MASK_CHAR.repeat(n - head - tail) + v.slice(n - tail);
}

// ------------------------------------------------------------------ checksums

export function luhnOk(digits: string): boolean {
  let total = 0;
  let alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (alt) { d *= 2; if (d > 9) d -= 9; }
    total += d;
    alt = !alt;
  }
  return total % 10 === 0;
}

const VERHOEFF_D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
  [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const VERHOEFF_P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

export function verhoeffOk(digits: string): boolean {
  let c = 0;
  const rev = digits.split('').reverse();
  for (let i = 0; i < rev.length; i++) {
    c = VERHOEFF_D[c][VERHOEFF_P[i % 8][rev[i].charCodeAt(0) - 48]];
  }
  return c === 0;
}

export function ibanOk(iban: string): boolean {
  const s = iban.toUpperCase();
  if (s.length < 15 || s.length > 34) return false;
  const rearranged = s.slice(4) + s.slice(0, 4);
  let num = '';
  for (const ch of rearranged) num += /[A-Z]/.test(ch) ? String(ch.charCodeAt(0) - 55) : ch;
  // mod 97 on a long digit string, chunked to stay within Number precision
  let rem = 0;
  for (let i = 0; i < num.length; i += 7) {
    rem = Number(String(rem) + num.slice(i, i + 7)) % 97;
  }
  return rem === 1;
}

// ------------------------------------------------------------------ helpers

const PLACEHOLDER_VALUES = new Set([
  'true', 'false', 'null', 'none', 'nil', 'undefined', 'xxx', 'xxxx', 'xxxxx',
  'redacted', '<redacted>', 'changeme', 'example', 'placeholder', 'hidden',
  'required', 'optional', 'string', 'value', 'text', 'password', 'secret',
  'your_api_key', 'your-api-key', '<your_api_key>', '<key>', '<token>',
  '*****', '********', 'n/a', 'na', 'tbd', 'todo',
]);

const stripQuotes = (v: string, extra = '') => {
  const chars = '"\'`' + extra;
  let s = v.trim();
  while (s.length && chars.includes(s[0])) s = s.slice(1);
  while (s.length && chars.includes(s[s.length - 1])) s = s.slice(0, -1);
  return s;
};

function looksLikePlaceholder(value: string): boolean {
  const v = stripQuotes(value).toLowerCase();
  if (!v || PLACEHOLDER_VALUES.has(v)) return true;
  if (/^(<|\$\{|\{\{|\[)/.test(v) || /(>|\}|\])$/.test(v)) return true;
  if ([...v].every(c => '*x#•.-_'.includes(c))) return true;
  return false;
}

const hasDigit = (v: string) => /\d/.test(v);
const isAlpha = (v: string) => /^[A-Za-z]+$/.test(v);
const isAlnum = (v: string) => /^[A-Za-z0-9]+$/.test(v);
const mixedCase = (v: string) => v.toLowerCase() !== v && v.toUpperCase() !== v;

type Validator = (value: string, whole: string) => boolean;

const passwordValueOk: Validator = value => {
  if (looksLikePlaceholder(value)) return false;
  const v = stripQuotes(value, ',;.:!?)(');
  if (v.length < 3) return false;
  if (isAlpha(v) && v === v.toLowerCase() && v.length < 12) return false;
  return true;
};

const strongPasswordOk: Validator = v =>
  !looksLikePlaceholder(v) && hasDigit(v) && (!isAlnum(v) || mixedCase(v));

const secretValueOk: Validator = value => {
  if (looksLikePlaceholder(value)) return false;
  const v = stripQuotes(value, ',;');
  if (v.length < 8) return false;
  return hasDigit(v) || mixedCase(v) || v.length >= 20;
};

const cardOk: Validator = value => {
  const digits = value.replace(/\D/g, '');
  if (digits.length < 13 || digits.length > 19) return false;
  if (new Set(digits).size === 1) return false;
  return luhnOk(digits);
};

const aadhaarOk: Validator = value => {
  const digits = value.replace(/\D/g, '');
  return digits.length === 12 && verhoeffOk(digits);
};

const ibanValueOk: Validator = value => ibanOk(value.replace(/\s/g, ''));

const emiratesOk: Validator = value => {
  const digits = value.replace(/\D/g, '');
  if (digits.length !== 15 || !digits.startsWith('784')) return false;
  const year = Number(digits.slice(3, 7));
  return year >= 1900 && year <= 2100 && luhnOk(digits);
};

const ukLicenceOk: Validator = value => {
  const digits = value.slice(5, 11);
  const month = Number(digits.slice(1, 3)) % 50;
  const day = Number(digits.slice(3, 5));
  return month >= 1 && month <= 12 && day >= 1 && day <= 31;
};

const labelledIdOk: Validator = value => {
  const v = value.trim();
  return (v.match(/\d/g) || []).length >= 4 && !looksLikePlaceholder(v);
};

const phoneOk: Validator = (value, whole) => {
  const digits = value.replace(/\D/g, '');
  if (digits.length < 10 || digits.length > 13) return false;
  if (new Set(digits).size <= 2) return false;
  if (digits === value && !whole.startsWith('+')) {
    return (digits.length === 10 || digits.length === 11) && '6789'.includes(value[0]);
  }
  return true;
};

// ------------------------------------------------------------------ rules
// Same order as core/pii.py: when two findings overlap, the earlier rule wins.

type Rule = { kind: Kind; label: string; rx: RegExp; group: number; ok?: Validator };

const R = (kind: Kind, label: string, rx: RegExp, group = 0, ok?: Validator): Rule =>
  ({ kind, label, rx, group, ok });

const RULES: Rule[] = [
  // ---- secrets: vendor-prefixed keys
  R('secret', 'Private key block',
    /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,4000}?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/g),
  R('secret', 'AWS access key', /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g),
  R('secret', 'GitHub token', /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b/g),
  R('secret', 'Slack token', /\bxox[abprs]-[A-Za-z0-9-]{10,}\b/g),
  R('secret', 'Google API key', /\bAIza[0-9A-Za-z_\-]{35}\b/g),
  R('secret', 'Stripe key', /\b[srp]k_(?:live|test)_[0-9a-zA-Z]{10,}\b/g),
  R('secret', 'Anthropic API key', /\bsk-ant-[A-Za-z0-9_\-]{20,}\b/g),
  R('secret', 'OpenAI API key', /\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b/g),
  R('secret', 'Hugging Face token', /\bhf_[A-Za-z0-9]{30,}\b/g),
  R('secret', 'SendGrid key', /\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b/g),
  R('secret', 'JWT', /\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b/g),
  R('secret', 'Bearer token', /\bBearer\s+([A-Za-z0-9._~+/=\-]{20,})/g, 1),
  R('secret', 'Storage account key', /AccountKey=([A-Za-z0-9+/=]{40,})/g, 1),
  R('secret', 'AWS secret key', /aws[\w\s]{0,20}secret[\w\s]{0,20}[:=]\s*["']?([A-Za-z0-9/+=]{40})\b/gi, 1),
  // ---- secrets: labelled values
  R('secret', 'Password',
    /\b(?:password|passwd|passphrase|passcode|pwd|pass|pin)\b\s*(?:is|was|[:=\-]|is\s*[:=])\s*["'`]?([^\s"'`,;]{3,})/gi,
    1, passwordValueOk),
  R('secret', 'PIN / OTP',
    /\b(?:pin|otp|one[- ]time (?:code|password)|passcode|mpin)\b\s*(?:is|was|[:=\-])?\s*(\d{4,8}|\d{3}[ \-]\d{3})\b/gi, 1),
  R('secret', 'Password',
    /\b(?:password|passwd|passphrase|pass|pwd)\b\s+([^\s"'`,;]{8,})/gi, 1, strongPasswordOk),
  R('secret', 'API key / secret',
    /\b(?:api[_ \-]?key|apikey|secret[_ \-]?key|access[_ \-]?token|auth[_ \-]?token|refresh[_ \-]?token|client[_ \-]?secret|private[_ \-]?key|secret|token|credentials?|key)\b\s*(?:is|[:=]|is\s*[:=])\s*["'`]?([A-Za-z0-9_\-./+=]{8,})/gi,
    1, secretValueOk),
  // ---- identity: cards and licences with a fixed shape
  R('identity', 'Emirates ID (UAE)', /\b784[ \-]?\d{4}[ \-]?\d{7}[ \-]?\d\b/g, 0, emiratesOk),
  R('identity', 'Driving licence (India)', /\b[A-Z]{2}[ \-]?\d{2}[ \-]?\d{4}[ \-]?\d{7}\b/g),
  R('identity', 'Driving licence (UK)', /\b[A-Z9]{5}\d{6}[A-Z9]{2}\d[A-Z]{2}\b/g, 0, ukLicenceOk),
  R('identity', 'Payment card number', /\b(?:\d[ \-]?){12,18}\d\b/g, 0, cardOk),
  R('identity', 'US Social Security number', /\b(?!000|666|9\d{2})\d{3}[\- ](?!00)\d{2}[\- ](?!0000)\d{4}\b/g),
  R('identity', 'Aadhaar number', /\b[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}\b/g, 0, aadhaarOk),
  R('identity', 'PAN (India)', /\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b/g),
  R('identity', 'IBAN', /\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b/g, 0, ibanValueOk),
  // any card or licence introduced by its name
  R('identity', 'ID / licence card number',
    /\b(?:driving licen[cs]e|driver'?s? licen[cs]e|licen[cs]e (?:no\.?|number|#)|dl (?:no\.?|number)|emirates id|national id|identity card|id card|citizen(?:ship)? (?:card|id|number)|resident(?:cy)? (?:card|id|permit)|voter id|epic (?:no\.?|number)|(?:health|insurance|medicare|social security|ration|pan|tax) card|card (?:no\.?|number|#))\s*(?:is|was|[:#\-])?\s*([A-Z0-9][A-Z0-9 \-/]{5,24}[A-Z0-9])\b/gi,
    1, labelledIdOk),
  R('identity', 'Passport number',
    /\bpassport\s*(?:no\.?|number|#)?\s*[:#\-]?\s*([A-Z][0-9]{7,8}|[A-Z]{1,2}[0-9]{6,7})\b/gi, 1),
  R('identity', 'Date of birth',
    /\b(?:dob|date of birth|birth\s*date|born(?: on)?)\s*[:\-]?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})/gi, 1),
  // ---- contact
  R('contact', 'Email address', /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g),
  R('contact', 'Phone number',
    /(?<![\w.\-])(?:\+\d{1,3}[ \-.]?)?(?:\(\d{2,4}\)[ \-.]?|\d{2,5}[ \-.])?\d{3,5}[ \-.]?\d{3,5}(?![\w.\-])/g,
    0, phoneOk),
];

const PRIORITY = new Map<string, number>();
RULES.forEach((r, i) => { if (!PRIORITY.has(r.label)) PRIORITY.set(r.label, i); });

/** Offset of capture group `group` inside match `m` (JS has no m.start(g) without the d flag). */
function groupStart(m: RegExpExecArray, group: number): number {
  if (group === 0) return m.index;
  const idx = m[0].indexOf(m[group]);
  return m.index + (idx >= 0 ? idx : 0);
}

export function scan(text: string, source = ''): Finding[] {
  if (!text) return [];
  const found: Finding[] = [];
  outer: for (const rule of RULES) {
    rule.rx.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = rule.rx.exec(text)) !== null) {
      if (m[0].length === 0) { rule.rx.lastIndex++; continue; }
      const value = m[rule.group];
      if (!value) continue;
      if (rule.ok && !rule.ok(value, m[0])) continue;
      const start = groupStart(m, rule.group);
      found.push({ kind: rule.kind, label: rule.label, value, start, end: start + value.length,
                   source, detector: 'regex' });
      if (found.length > MAX_FINDINGS * 4) break outer;
    }
  }
  return dedupe(found).slice(0, MAX_FINDINGS);
}

/** Drop findings that overlap a higher-priority one; keep document order. */
export function dedupe(findings: Finding[]): Finding[] {
  const ordered = [...findings].sort((a, b) =>
    (PRIORITY.get(a.label) ?? 999) - (PRIORITY.get(b.label) ?? 999) || a.start - b.start);
  const kept: Finding[] = [];
  for (const f of ordered) {
    if (kept.some(k => f.start < k.end && k.start < f.end)) continue;
    kept.push(f);
  }
  return kept.sort((a, b) => a.start - b.start);
}

export function redact(text: string, findings: Finding[]): string {
  let out = '';
  let pos = 0;
  for (const f of [...findings].sort((a, b) => a.start - b.start)) {
    if (f.start < pos) continue;
    out += text.slice(pos, f.start) + mask(f.value);
    pos = f.end;
  }
  return out + text.slice(pos);
}

/** Group by label for display; only masked values leave here. */
export function summarize(findings: Finding[]): Summary[] {
  const groups = new Map<string, Summary>();
  for (const f of findings) {
    let g = groups.get(f.label);
    if (!g) {
      g = { label: f.label, kind: f.kind, count: 0, examples: [], detector: f.detector };
      groups.set(f.label, g);
    }
    g.count++;
    const m = mask(f.value);
    if (!g.examples.includes(m) && g.examples.length < 3) g.examples.push(m);
  }
  const order: Record<string, number> = { secret: 0, identity: 1, contact: 2 };
  return [...groups.values()].sort((a, b) =>
    (order[a.kind] ?? 9) - (order[b.kind] ?? 9) || b.count - a.count);
}
