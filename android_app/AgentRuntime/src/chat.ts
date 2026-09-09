// Chat orchestrator — the on-device equivalent of runtime/app.py's /api/chat:
// retrieve → prompt → stream → tool rounds. Emits UI-ready events.
import { search, SearchHit } from './kb';
import {
  buildPrompt, buildSystemPrompt, ChatTurn, CompletionStats,
  embedText, ensureChatModel, streamCompletion,
} from './llm';
import { executeTool } from './tools';
import { MAX_TOOL_ROUNDS } from './config';
import { log } from './logger';
import { reportTelemetry } from './appTelemetry';

export type ChatEvent =
  | { type: 'status'; text: string }
  | { type: 'sources'; items: SearchHit[] }
  | { type: 'token'; text: string }
  | { type: 'tool'; name: string; args: any; result: any }
  | { type: 'stats'; stats: CompletionStats }
  | { type: 'error'; text: string };

export async function runChat(
  manifest: any,
  agentId: string,
  history: ChatTurn[],           // full conversation, last entry = current user msg
  onEvent: (ev: ChatEvent) => void,
): Promise<string> {
  try {
    const userQuery = history[history.length - 1]?.content ?? '';
    await log(`chat: query "${userQuery.slice(0, 60)}" agent=${agentId}`);

    onEvent({ type: 'status', text: 'Searching knowledge base…' });
    const rag = manifest.rag || {};
    const qvec = await embedText(userQuery, true);
    await log('chat: query embedded');
    const hits = await search(agentId, qvec, Number(rag.top_k ?? 4), Number(rag.min_score ?? 0.35));
    if (hits.length) onEvent({ type: 'sources', items: hits });

    let contextBlock = '';
    if (hits.length) {
      contextBlock =
        'Context from the knowledge base:\n\n' +
        hits.map(h => `[${h.doc_name} #${h.chunk_index}]\n${h.text}`).join('\n\n') +
        '\n\n---\n\n';
    }

    const modelId = manifest.model?.id;
    const tLoad = Date.now();
    const ctx = await ensureChatModel(manifest.model.file,
      s => onEvent({ type: 'status', text: s }),
      Number(manifest.model.size_bytes) || undefined);
    const loadMs = Date.now() - tLoad;
    if (loadMs > 400) {  // a real load, not a warm-cache reuse
      reportTelemetry({ event: 'model_load', agent_id: agentId, model_id: modelId,
                        load_ms: loadMs, ok: 1 });
    }

    const system = buildSystemPrompt(manifest);
    const turns: ChatTurn[] = [
      ...history.slice(0, -1),
      { role: 'user', content: contextBlock + userQuery },
    ];
    const toolDefs: Record<string, any> = {};
    for (const t of manifest.tools || []) toolDefs[t.name] = t;

    let answer = '';
    let lastStats: CompletionStats | null = null;
    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      onEvent({ type: 'status', text: round === 0 ? 'Thinking on-device…' : 'Continuing after tool…' });
      let toolCalled = false;
      const pendingTools: Array<{ raw: string; call: any }> = [];

      await log(`chat: completion round ${round} starting`);
      const { stats } = await streamCompletion(ctx, buildPrompt(system, turns),
        manifest.generation || {}, ev => {
          if (ev.type === 'token') {
            answer += ev.text;
            onEvent(ev);
          } else if (ev.type === 'tool_call') {
            pendingTools.push(ev);
            toolCalled = true;
          }
        });
      if (stats) { lastStats = stats; onEvent({ type: 'stats', stats }); }

      if (stats) await log(`chat: round done, ${stats.tokens} tok @ ${stats.tok_per_sec}/s`);
      if (!toolCalled) break;
      for (const t of pendingTools) {
        const name = t.call?.name ?? '?';
        const args = t.call?.arguments ?? {};
        const result = await executeTool(toolDefs[name], args);
        onEvent({ type: 'tool', name, args, result });
        turns.push({ role: 'assistant', content: `<tool_call>\n${t.raw}\n</tool_call>` });
        turns.push({ role: 'user', content: `<tool_response>\n${JSON.stringify(result)}\n</tool_response>` });
      }
    }
    reportTelemetry({
      event: 'chat', agent_id: agentId, agent_version: manifest.version,
      model_id: modelId, kb_hits: hits.length,
      tokens: lastStats?.tokens, tok_per_sec: lastStats?.tok_per_sec,
      prefill_tokens: lastStats?.prefill_tokens, ok: 1,
    });
    return answer;
  } catch (e: any) {
    const msg = String(e?.message || e).slice(0, 400);
    await log(`chat: ERROR ${msg}`);
    reportTelemetry({ event: 'error', agent_id: agentId,
                      model_id: manifest.model?.id, detail: msg, ok: 0 });
    onEvent({ type: 'error', text: msg });
    return '';
  }
}
