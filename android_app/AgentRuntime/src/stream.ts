// Pure stream filter for Qwen3 output — no native deps, unit-testable.
// Direct port of runtime/llm.py's drain(): strips <think> blocks and
// surfaces <tool_call> payloads as structured events.
const OPEN_THINK = '<think>';
const CLOSE_THINK = '</think>';
const OPEN_TOOL = '<tool_call>';
const CLOSE_TOOL = '</tool_call>';
const HOLDBACK = CLOSE_TOOL.length - 1;

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'tool_call'; raw: string; call: any | null };

export class TagFilter {
  private pending = '';
  private mode: 'text' | 'think' | 'tool' = 'text';

  push(piece: string, final = false): StreamEvent[] {
    this.pending += piece;
    const out: StreamEvent[] = [];
    for (;;) {
      if (this.mode === 'think') {
        const idx = this.pending.indexOf(CLOSE_THINK);
        if (idx < 0) { if (final) this.pending = ''; return out; }
        this.pending = this.pending.slice(idx + CLOSE_THINK.length).replace(/^\n+/, '');
        this.mode = 'text';
        continue;
      }
      if (this.mode === 'tool') {
        const idx = this.pending.indexOf(CLOSE_TOOL);
        if (idx < 0) {
          if (final && this.pending.trim()) {
            out.push({ type: 'token', text: this.pending });
            this.pending = '';
          }
          return out;
        }
        const raw = this.pending.slice(0, idx).trim();
        this.pending = this.pending.slice(idx + CLOSE_TOOL.length);
        this.mode = 'text';
        let call: any = null;
        try { call = JSON.parse(raw); } catch {}
        out.push({ type: 'tool_call', raw, call });
        continue;
      }
      const iThink = this.pending.indexOf(OPEN_THINK);
      const iTool = this.pending.indexOf(OPEN_TOOL);
      const candidates: Array<[number, 'think' | 'tool']> = [];
      if (iThink >= 0) candidates.push([iThink, 'think']);
      if (iTool >= 0) candidates.push([iTool, 'tool']);
      if (candidates.length) {
        candidates.sort((a, b) => a[0] - b[0]);
        const [idx, newMode] = candidates[0];
        if (idx > 0) out.push({ type: 'token', text: this.pending.slice(0, idx) });
        this.pending = this.pending.slice(
          idx + (newMode === 'think' ? OPEN_THINK.length : OPEN_TOOL.length));
        this.mode = newMode;
        continue;
      }
      const safe = final ? this.pending.length : Math.max(0, this.pending.length - HOLDBACK);
      if (safe > 0) {
        out.push({ type: 'token', text: this.pending.slice(0, safe) });
        this.pending = this.pending.slice(safe);
      }
      return out;
    }
  }
}
