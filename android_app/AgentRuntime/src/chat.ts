// Chat orchestrator — the on-device equivalent of runtime/app.py's /api/chat:
// guard → retrieve (KB + attachment) → prompt → stream → tool rounds. Emits
// UI-ready events.
//
// Memory order matters on a handset (single-context rule in llm.ts): every
// embedding — attachment retrieval and KB search — happens BEFORE the chat
// model is loaded, and the PII Guard's model judge runs on whichever big
// model is loaded for the turn (the agent's model, or the vision model on an
// image turn) so no turn ever loads two large models.
import { search, SearchHit } from './kb';
import { shouldRetrieve } from './gating';
import {
  buildPrompt, buildSystemPrompt, ChatTurn, CompletionStats,
  embedText, ensureChatModel, streamCompletion,
} from './llm';
import { executeTool } from './tools';
import { MAX_TOOL_ROUNDS } from './config';
import { log } from './logger';
import { reportTelemetry } from './appTelemetry';
import { Attachment, contextFor } from './attachments';
import * as guard from './guard';
import { embedderOf } from './embedder';
import { answerWithImage, visionAvailable, VISION_MODEL } from './vision';
import type { LlamaContext } from 'llama.rn';

export type ChatEvent =
  | { type: 'status'; text: string }
  | { type: 'sources'; items: SearchHit[] }
  | { type: 'token'; text: string }
  | { type: 'tool'; name: string; args: any; result: any }
  | { type: 'stats'; stats: CompletionStats; model?: string }
  | { type: 'guard'; text: string; where: string }
  | { type: 'redact'; text: string; summary: guard.Verdict['summary'] }
  | { type: 'vision'; model: string }
  | { type: 'error'; text: string };

const ATTACHMENT_RULES =
  '\n\nThe user has attached a file. Text from it appears under \'Attached document\'. ' +
  'Answer questions about the file from that text, quote it where useful, and say plainly ' +
  'when the file does not contain the answer.';

/** Text-only completion on an already-loaded context — the judge's channel. */
function judgeFnFor(ctx: LlamaContext): guard.ChatFn {
  return async (system, user) => {
    const { text } = await streamCompletion(ctx, buildPrompt(system, [{ role: 'user', content: user }]),
      { temperature: 0.0, top_p: 1.0, max_tokens: guard.POLICY.judgeMaxTokens }, () => {});
    return text;
  };
}

export async function runChat(
  manifest: any,
  agentId: string,
  history: ChatTurn[],           // full conversation, last entry = current user msg
  onEvent: (ev: ChatEvent) => void,
  attachment?: Attachment | null,
): Promise<string> {
  try {
    const userQuery = history[history.length - 1]?.content ?? '';
    const att = attachment || null;
    await log(`chat: query "${userQuery.slice(0, 60)}" agent=${agentId}` + (att ? ` att=${att.name}` : ''));

    // The whole guard is a per-agent guardrail (Studio checkbox, shipped in the
    // manifest). Off = the normal flow: no scan of the message, file or answer.
    const pii = guard.piiEnabled(manifest);

    const refuse = (where: string, summary: guard.Verdict['summary'], src: string) => {
      const text = guard.refusalMessage(where, summary);
      reportTelemetry({ event: 'guard_block', agent_id: agentId, agent_version: manifest.version,
                        model_id: manifest.model?.id, ok: 1,
                        detail: `${src}: ${summary.map(s => s.label).join(', ')}`.slice(0, 250) });
      onEvent({ type: 'guard', text, where });
      return text;
    };

    // ---- PII Guard stage 0: a file already judged unsafe refuses every question.
    if (pii && att?.guard?.blocked) {
      return refuse(`the attached document "${att.name}"`, att.guard.summary, 'attachment');
    }
    // ---- stage 1 (regex): the user's own message — free, before any model work.
    if (pii) {
      const v = guard.inspect(userQuery, 'query');
      if (v.blocked) return refuse('your message', v.summary, 'query');
    }

    // ---- retrieval, all embedding work first (attachment, then KB)
    const rag = manifest.rag || {};
    let attSources: SearchHit[] = [];
    if (att) {
      onEvent({ type: 'status', text: att.small ? 'Reading the attachment…' : 'Searching the attachment…' });
      attSources = await contextFor(att, userQuery);
    }
    const [needsKb, gateReason] = shouldRetrieve(userQuery);
    let hits: SearchHit[] = [];
    if (needsKb) {
      onEvent({ type: 'status', text: 'Searching knowledge base…' });
      const qvec = await embedText(userQuery, true, embedderOf(manifest));
      hits = await search(agentId, qvec, Number(rag.top_k ?? 4), Number(rag.min_score ?? 0.45));
    } else {
      await log(`chat: KB skipped (${gateReason}) — no embedding, no search`);
    }
    if (attSources.length || hits.length) onEvent({ type: 'sources', items: [...attSources, ...hits] });

    // ---- load the big model for this turn (vision model on an image turn)
    const useVision = !!(att && att.kind === 'image' && (await visionAvailable()));
    const modelId = manifest.model?.id;
    const tLoad = Date.now();
    let ctx: LlamaContext | null = null;
    if (!useVision) {
      ctx = await ensureChatModel(manifest.model.file,
        s => onEvent({ type: 'status', text: s }),
        Number(manifest.model.size_bytes) || undefined,
        manifest.adapter ? { file: manifest.adapter.file, scale: manifest.adapter.scale } : null);
      const loadMs = Date.now() - tLoad;
      if (loadMs > 400) {
        reportTelemetry({ event: 'model_load', agent_id: agentId, model_id: modelId, load_ms: loadMs, ok: 1 });
      }
    }

    // ---- stage 2 (model judge): only when a file is in play, on the agent's model
    if (pii && att && guard.POLICY.judgeOnAttachments && !useVision && ctx) {
      const judgeFn = judgeFnFor(ctx);
      onEvent({ type: 'status', text: `${guard.GUARD_NAME} reviewing your message…` });
      const vq = await guard.inspectText(userQuery, 'query', judgeFn);
      if (vq.blocked) return refuse('your message', vq.summary, 'query');
      const fresh = attSources.filter(s => !att.judgedChunks.includes(s.chunk_index));
      if (fresh.length) {
        onEvent({ type: 'status', text: `${guard.GUARD_NAME} reviewing the document…` });
        const vd = await guard.inspectChunks(fresh.map(s => s.text), `attachment:${att.name}`, judgeFn);
        att.judgedChunks.push(...fresh.map(s => s.chunk_index));
        if (vd.blocked) {
          att.guard = vd;
          return refuse(`the attached document "${att.name}"`, vd.summary, 'attachment');
        }
      }
    }

    // ---- answer
    let answer = '';
    let lastStats: CompletionStats | null = null;
    let statsModel: string | undefined;

    if (useVision && att) {
      onEvent({ type: 'vision', model: VISION_MODEL.name });
      const system = buildSystemPrompt({ ...manifest, tools: [] });
      const { stats } = await answerWithImage(att.path, userQuery, system, att.text,
        manifest.generation || {}, ev => {
          if (ev.type === 'token') { answer += ev.text; onEvent(ev); }
        }, s => onEvent({ type: 'status', text: s }));
      lastStats = stats;
      statsModel = VISION_MODEL.name;
      if (stats) onEvent({ type: 'stats', stats, model: statsModel });
    } else if (ctx) {
      let contextBlock = '';
      if (att && attSources.length) {
        const how = att.small ? 'full text' : 'relevant passages';
        contextBlock += `Attached document "${att.name}" (${how}):\n\n` +
          attSources.map(s => `[${att.name} #${s.chunk_index}]\n${s.text}`).join('\n\n') + '\n\n---\n\n';
      } else if (att) {
        contextBlock += `Attached document "${att.name}": no readable text could be extracted from it.\n\n---\n\n`;
      }
      if (hits.length) {
        contextBlock += 'Context from the knowledge base:\n\n' +
          hits.map(h => `[${h.doc_name} #${h.chunk_index}]\n${h.text}`).join('\n\n') + '\n\n---\n\n';
      }

      let system = buildSystemPrompt(manifest);
      if (att) system = system.replace(' /no_think', ATTACHMENT_RULES + ' /no_think');
      const turns: ChatTurn[] = [
        ...history.slice(0, -1),
        { role: 'user', content: contextBlock + userQuery },
      ];
      const toolDefs: Record<string, any> = {};
      for (const t of manifest.tools || []) toolDefs[t.name] = t;

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
        answer = '';
      }
    }

    // ---- stage 3: the answer itself — mask anything that slipped through
    if (pii && guard.POLICY.redactOutput) {
      const r = guard.redactText(answer);
      if (r.summary.length) {
        answer = r.text;
        onEvent({ type: 'redact', text: r.text, summary: r.summary });
      }
    }

    reportTelemetry({
      event: 'chat', agent_id: agentId, agent_version: manifest.version,
      model_id: modelId, kb_hits: hits.length + attSources.length,
      tokens: lastStats?.tokens, tok_per_sec: lastStats?.tok_per_sec,
      prefill_tokens: lastStats?.prefill_tokens, ok: 1,
      detail: att ? `attachment ${att.kind}${useVision ? ' vision' : ''}` : undefined,
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
