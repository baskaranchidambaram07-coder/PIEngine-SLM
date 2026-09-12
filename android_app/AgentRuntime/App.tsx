/**
 * Offline Enterprise Agent Runtime — Android client.
 * Syncs with the Agent Studio portal, installs agent bundles, and runs them
 * fully on-device: llama.rn inference, SQLite vector KB, on-device tools,
 * per-turn file attachments (jpg/jpeg/pdf/txt/md ≤ 5 MB, OCR on device,
 * on-demand retrieval, optional vision model) and the per-agent PII Guard.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Alert, FlatList, KeyboardAvoidingView, Modal,
  SafeAreaView, ScrollView, Share, StatusBar, StyleSheet, Switch, Text, TextInput,
  TouchableOpacity, View,
} from 'react-native';

import { C } from './src/theme';
import { DEFAULT_PORTAL } from './src/config';
import {
  fetchStore, getPortalUrl, installAgent, InstalledAgent, listInstalled,
  Progress, setPortalUrl, StoreEntry, uninstallAgent,
} from './src/portal';
import { runChat } from './src/chat';
import { ChatTurn, listBlockedModels, unblockAllModels } from './src/llm';
import { clearVectorCache, inlineStats, SearchHit } from './src/kb';
import { clearLogs, getLogs, log } from './src/logger';
import { reportTelemetry } from './src/appTelemetry';
import {
  Attachment, AttachmentError, ingest, pickAttachment, removeAttachment,
  sweepAttachments, validateMeta,
} from './src/attachments';
import { splitText } from './src/chunker';
import * as guard from './src/guard';
import { embedderOf } from './src/embedder';
import {
  downloadVision, isVisionEnabled, removeVisionFiles, setVisionEnabled,
  VISION_MODEL, visionFilesReady,
} from './src/vision';

type Msg = {
  id: string;
  kind: 'user' | 'bot' | 'tool' | 'status' | 'guard';
  text: string;
  sources?: SearchHit[];
  statsLine?: string;
  redacted?: string;
  attachmentName?: string;
};

let msgSeq = 0;
const nextId = () => `m${++msgSeq}`;
const fmtBytes = (b: number) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

export default function App() {
  const [screen, setScreen] = useState<'home' | 'chat'>('home');
  const [installed, setInstalled] = useState<InstalledAgent[]>([]);
  const [store, setStore] = useState<StoreEntry[]>([]);
  const [storeError, setStoreError] = useState('');
  const [portal, setPortal] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [installing, setInstalling] = useState<Record<string, string>>({});

  const [agent, setAgent] = useState<InstalledAgent | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [inline, setInline] = useState({ docs: 0, chunks: 0 });
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attStatus, setAttStatus] = useState('');      // non-empty while a file is being processed
  const attachmentRef = useRef<Attachment | null>(null);
  const historyRef = useRef<ChatTurn[]>([]);
  const listRef = useRef<FlatList<Msg>>(null);

  // vision setting (⚙)
  const [visionOn, setVisionOn] = useState(false);
  const [visionReady, setVisionReady] = useState(false);
  const [visionDl, setVisionDl] = useState('');
  const [blockedModels, setBlockedModels] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    setInstalled(await listInstalled());
    setPortal(await getPortalUrl());
    setVisionOn(await isVisionEnabled());
    setVisionReady(await visionFilesReady());
    setBlockedModels(await listBlockedModels());
    try {
      setStore(await fetchStore());
      setStoreError('');
    } catch (e: any) {
      setStore([]);
      setStoreError(`Portal unreachable (${String(e?.message || e).slice(0, 80)}). Installed agents keep working offline.`);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const setAtt = (a: Attachment | null) => { attachmentRef.current = a; setAttachment(a); };

  // ------------------------------------------------------------- install

  const onInstall = async (entry: StoreEntry) => {
    const progress: Progress = (label, done, total) => {
      const pct = total > 0 ? ` ${Math.round((100 * done) / total)}%` : '';
      setInstalling(p => ({ ...p, [entry.id]: `${label}${pct}` }));
    };
    setInstalling(p => ({ ...p, [entry.id]: 'Starting…' }));
    try {
      await installAgent(entry, progress);
      setInstalling(p => { const q = { ...p }; delete q[entry.id]; return q; });
      reportTelemetry({ event: 'install', agent_id: entry.id, agent_version: entry.version,
                        model_id: entry.model?.id, ok: 1 });
      await refresh();
      Alert.alert('Installed', `${entry.name} v${entry.version} is ready — works fully offline now.`);
    } catch (e: any) {
      setInstalling(p => { const q = { ...p }; delete q[entry.id]; return q; });
      log(`install failed: ${entry.id}: ${String(e?.message || e).slice(0, 200)}`);
      Alert.alert('Install failed', `${String(e?.message || e).slice(0, 300)}\n\nSee 📋 Logs (home screen) for the full trace — use Share there to send it.`);
    }
  };

  // ----------------------------------------------------------- uninstall

  const onUninstall = (a: InstalledAgent) => {
    Alert.alert(
      `Uninstall ${a.name}?`,
      'The agent, its knowledge base and any attachments are removed from this device. ' +
      'The shared model file stays for other agents. You can reinstall it from the store while the Studio still publishes it.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Uninstall', style: 'destructive', onPress: async () => {
          try {
            await uninstallAgent(a.id, a.version, a.manifest?.model?.id);
            clearVectorCache(a.id);
            if (agent?.id === a.id) { setAgent(null); setScreen('home'); }
            await refresh();
          } catch (e: any) {
            Alert.alert('Uninstall failed', String(e?.message || e).slice(0, 200));
          }
        } },
      ],
    );
  };

  const openLogs = async () => {
    setLogLines([...(await getLogs())].reverse());
    setShowLogs(true);
  };

  // ---------------------------------------------------------------- chat

  const openChat = async (a: InstalledAgent) => {
    setAgent(a);
    historyRef.current = [];
    if (attachmentRef.current) { removeAttachment(attachmentRef.current).catch(() => {}); setAtt(null); }
    const pii = guard.piiEnabled(a.manifest);
    setMsgs([{
      id: nextId(), kind: 'status',
      text: `🔒 ${a.name} v${a.version} — ${a.manifest.model.name}\nKB: ${a.manifest.rag?.chunks ?? 0} chunks · Tools: ${(a.manifest.tools || []).map((t: any) => t.name).join(', ') || 'none'}\nFirst answer loads the model into RAM.` +
            (pii ? '\n🛡 PII Guard is active.' : ''),
    }]);
    setInline(await inlineStats(a.id));
    sweepAttachments(a.id).catch(() => {});
    setScreen('chat');
  };

  const appendBotText = (id: string, piece: string) => {
    setMsgs(m => m.map(x => (x.id === id ? { ...x, text: x.text + piece } : x)));
  };

  const send = async () => {
    if (!agent || busy) return;
    const text = input.trim();
    if (!text) return;
    // A version retired in the Studio must stop running here too — this device
    // may have installed it before it was disabled.
    if (agent.versionState && agent.versionState !== 'active') {
      setMsgs(m => [...m, { id: nextId(), kind: 'bot',
        text: `${agent.name} v${agent.version} has been ${agent.versionState} by the Studio and can no longer be run. Open the store to install a newer version.` }]);
      return;
    }
    const att = attachmentRef.current;
    if (attStatus) { Alert.alert('One moment', 'The attachment is still being processed.'); return; }
    if (att?.guard?.blocked) { Alert.alert('PII Guard', 'This file was blocked — remove it (✕) to continue.'); return; }
    setInput('');
    setBusy(true);

    historyRef.current.push({ role: 'user', content: text });
    const botId = nextId();
    setMsgs(m => [...m,
      { id: nextId(), kind: 'user', text, attachmentName: att?.name },
      { id: botId, kind: 'bot', text: '' },
    ]);

    let sources: SearchHit[] = [];
    let statsLine = '';
    let guarded = false;
    let redacted = '';
    const answer = await runChat(agent.manifest, agent.id, [...historyRef.current], ev => {
      if (ev.type === 'token') appendBotText(botId, ev.text);
      else if (ev.type === 'sources') sources = ev.items;
      else if (ev.type === 'status') {
        setMsgs(m => m.map(x => (x.id === botId && !x.text ? { ...x, statsLine: ev.text } : x)));
      } else if (ev.type === 'guard') {
        guarded = true;
        setMsgs(m => m.map(x => (x.id === botId ? { ...x, kind: 'guard', text: ev.text, statsLine: '' } : x)));
      } else if (ev.type === 'redact') {
        redacted = `🛡 PII Guard masked ${ev.summary.map(s => s.label).join(', ')} in this answer`;
        setMsgs(m => m.map(x => (x.id === botId ? { ...x, text: ev.text } : x)));
      } else if (ev.type === 'tool') {
        setMsgs(m => [...m, {
          id: nextId(), kind: 'tool',
          text: `⚙ ${ev.name}(${JSON.stringify(ev.args)})\n→ ${JSON.stringify(ev.result).slice(0, 280)}`,
        }]);
      } else if (ev.type === 'stats') {
        statsLine = `${ev.stats.tokens} tok @ ${ev.stats.tok_per_sec}/s · prompt ${ev.stats.prefill_tokens} tok @ ${ev.stats.prefill_per_sec}/s` +
                    (ev.model ? ` · ${ev.model}` : '');
      } else if (ev.type === 'error') {
        appendBotText(botId, `\n⚠ ${ev.text}`);
      }
    }, att);

    historyRef.current.push({ role: 'assistant', content: guarded ? '[PII Guard refused this request]' : answer });
    if (!guarded) {
      setMsgs(m => m.map(x => (x.id === botId ? { ...x, sources, statsLine, redacted } : x)));
    } else if (att && attachmentRef.current === att) {
      setAtt({ ...att });     // re-render the chip with the stored verdict
    }
    setBusy(false);
  };

  // ------------------------------------------------------------ attachment
  // One file per turn. Validated here (type, size) and again on the copied
  // bytes; OCR for images and PDFs; the PII Guard's regex stage runs on the
  // whole text right away when the agent has the guardrail on. The model
  // judge runs later, inside the turn, on the agent's loaded model.

  const attachFile = async () => {
    if (!agent || busy || attStatus) return;
    try {
      const file = await pickAttachment();
      if (!file) return;
      try { validateMeta(file); } catch (e: any) {
        if (e instanceof AttachmentError) { Alert.alert('Cannot attach', e.message); return; }
        throw e;
      }
      if (attachmentRef.current) { removeAttachment(attachmentRef.current).catch(() => {}); setAtt(null); }
      setAttStatus(`Reading ${file.name}…`);
      const att = await ingest(agent.id, file, s => setAttStatus(s), embedderOf(agent.manifest));

      att.guardEnabled = guard.piiEnabled(agent.manifest);
      if (att.guardEnabled && att.text) {
        setAttStatus('PII Guard scanning…');
        const chunks = splitText(att.text);
        att.guard = await guard.inspectChunks(chunks.length ? chunks : [att.text], `attachment:${att.name}`, null, 0);
        if (att.guard.blocked) {
          reportTelemetry({ event: 'guard_block', agent_id: agent.id, agent_version: agent.version,
                            detail: `attachment: ${att.guard.summary.map(s => s.label).join(', ')}`.slice(0, 250), ok: 1 });
          setMsgs(m => [...m, { id: nextId(), kind: 'guard',
            text: guard.refusalMessage(`the attached document "${att.name}"`, att.guard!.summary) }]);
        }
      }
      reportTelemetry({ event: 'attachment', agent_id: agent.id, kb_hits: att.chunks, ok: 1,
                        detail: `${att.kind} ${att.status} pages=${att.pages} ocr=${att.ocrPages} guard=${att.guard?.blocked ? 'blocked' : att.guardEnabled ? 'clear' : 'off'}` });
      setAtt(att);
      if (att.status === 'no-text') {
        Alert.alert('No text found', att.kind === 'image'
          ? 'OCR found no text in this image. With the vision model enabled (⚙) the agent can still look at it.'
          : 'No readable text could be extracted from this file.');
      }
    } catch (e: any) {
      const s = String(e?.message || e);
      if (!/cancel/i.test(s)) {
        log(`attach: FAILED ${s.slice(0, 200)}`);
        Alert.alert('Attachment failed', s.slice(0, 240));
      }
    } finally {
      setAttStatus('');
    }
  };

  const dropAttachment = () => {
    const att = attachmentRef.current;
    if (att) removeAttachment(att).catch(() => {});
    setAtt(null);
  };

  // --------------------------------------------------------------- vision

  const toggleVision = async (on: boolean) => {
    if (on && !visionReady) {
      Alert.alert(
        'Download the vision model?',
        `${VISION_MODEL.name} + projector, about ${VISION_MODEL.size_gb} GB, from the portal (Hugging Face fallback). ` +
        `Needs a phone with ${VISION_MODEL.min_device_ram_gb} GB RAM or more; on smaller devices keep this off — images are still read with on-device OCR.`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Download', onPress: async () => {
            try {
              await downloadVision((label, done, total) =>
                setVisionDl(`${label}${total > 0 ? ` ${Math.round((100 * done) / total)}%` : ''}`));
              await setVisionEnabled(true);
              setVisionOn(true); setVisionReady(true); setVisionDl('');
            } catch (e: any) {
              setVisionDl('');
              Alert.alert('Download failed', String(e?.message || e).slice(0, 240));
            }
          } },
        ]);
      return;
    }
    await setVisionEnabled(on);
    setVisionOn(on);
  };

  // ------------------------------------------------------------ rendering

  const renderMsg = ({ item }: { item: Msg }) => {
    if (item.kind === 'status') {
      return <Text style={st.statusMsg}>{item.text}</Text>;
    }
    if (item.kind === 'tool') {
      return <View style={[st.bubble, st.toolBubble]}><Text style={st.toolText}>{item.text}</Text></View>;
    }
    if (item.kind === 'guard') {
      return <View style={[st.bubble, st.guardBubble]}><Text style={st.guardText}>{item.text}</Text></View>;
    }
    const user = item.kind === 'user';
    return (
      <View style={[st.bubble, user ? st.userBubble : st.botBubble]}>
        {user && !!item.attachmentName && <Text style={st.attTag}>📎 {item.attachmentName}</Text>}
        {item.text ? <Text style={st.msgText}>{item.text}</Text>
          : (
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
              <ActivityIndicator size="small" color={C.accent} />
              {!!item.statsLine && <Text style={st.stats}>{item.statsLine}</Text>}
            </View>
          )}
        {!user && item.sources && item.sources.length > 0 && (
          <View style={st.srcRow}>
            {item.sources.map((s2, i) => (
              <Text key={i} style={[st.srcChip, s2.source === 'attachment' && st.srcChipAtt]}>
                {s2.source === 'attachment' ? '📎' : '📄'} {s2.doc_name} #{s2.chunk_index} · {s2.score}{s2.source === 'inline' ? ' · 📎' : ''}
              </Text>
            ))}
          </View>
        )}
        {!user && !!item.redacted && <Text style={[st.stats, { color: C.warn }]}>{item.redacted}</Text>}
        {!user && !!item.text && !!item.statsLine && <Text style={st.stats}>{item.statsLine}</Text>}
      </View>
    );
  };

  const renderAttachChip = () => {
    if (!attachment && !attStatus) return null;
    const a = attachment;
    const blocked = !!a?.guard?.blocked;
    let meta = '';
    if (attStatus) meta = attStatus;
    else if (a) {
      const parts = [fmtBytes(a.bytes)];
      if (a.pages > 1) parts.push(`${a.pages} pages`);
      if (a.ocrPages) parts.push(`OCR ×${a.ocrPages}`);
      parts.push(a.status === 'no-text' ? 'no text' : `${a.chunks} chunk${a.chunks === 1 ? '' : 's'}${a.small ? ' (whole file used)' : ''}`);
      parts.push(!a.guardEnabled ? '🛡 PII Guard: off for this agent' : blocked ? '🛡 PII Guard: blocked' : '🛡 PII Guard: clear');
      meta = parts.join(' · ');
    }
    const icon = a?.kind === 'image' ? '🖼️' : a?.kind === 'pdf' ? '📄' : '📝';
    return (
      <View style={[st.chip, blocked && st.chipBlocked, !!a && !blocked && !attStatus && st.chipClear]}>
        <Text style={{ fontSize: 18 }}>{attStatus && !a ? '📎' : icon}</Text>
        <View style={{ flex: 1, marginHorizontal: 8 }}>
          <Text style={st.chipName} numberOfLines={1}>{a?.name || 'Attachment'}</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            {!!attStatus && <ActivityIndicator size="small" color={C.accent} />}
            <Text style={st.chipMeta} numberOfLines={2}>{meta}</Text>
          </View>
        </View>
        {!attStatus && (
          <TouchableOpacity onPress={dropAttachment} disabled={busy} hitSlop={8}>
            <Text style={st.chipX}>✕</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  };

  if (screen === 'chat' && agent) {
    const pii = guard.piiEnabled(agent.manifest);
    return (
      <SafeAreaView style={st.root}>
        <StatusBar barStyle="light-content" backgroundColor={C.panel} />
        <View style={st.header}>
          <TouchableOpacity onPress={() => setScreen('home')}><Text style={st.back}>‹ Agents</Text></TouchableOpacity>
          <View style={{ flex: 1, marginLeft: 10 }}>
            <Text style={st.hTitle} numberOfLines={1}>{agent.name}</Text>
            <Text style={st.hSub} numberOfLines={1}>
              ● on-device · {agent.manifest.model.name}
              {pii ? ' · 🛡 PII Guard active' : ''}
              {inline.chunks ? ` · 📎 ${inline.chunks} inline chunks` : ''}
            </Text>
          </View>
          <TouchableOpacity onPress={attachFile} disabled={busy || !!attStatus}>
            <Text style={[st.clip, (busy || !!attStatus) && { opacity: 0.4 }]}>📎</Text>
          </TouchableOpacity>
        </View>
        <FlatList
          ref={listRef}
          data={msgs}
          keyExtractor={m => m.id}
          renderItem={renderMsg}
          contentContainerStyle={{ padding: 14, gap: 8 }}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        />
        <KeyboardAvoidingView behavior={undefined}>
          {renderAttachChip()}
          <View style={st.composer}>
            <TextInput
              style={st.input}
              value={input}
              onChangeText={setInput}
              placeholder={attachment ? 'Ask about the attached file…' : 'Ask your agent…'}
              placeholderTextColor={C.muted}
              editable={!busy}
              onSubmitEditing={send}
            />
            <TouchableOpacity style={[st.sendBtn, busy && { opacity: 0.5 }]} onPress={send} disabled={busy}>
              <Text style={{ color: '#fff', fontSize: 16 }}>➤</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={st.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.panel} />
      <View style={st.header}>
        <View style={{ flex: 1 }}>
          <Text style={st.hTitle}>📱 Offline Enterprise Agents</Text>
          <Text style={st.hSub}>Runs 100% on-device — no cloud LLM</Text>
        </View>
        <TouchableOpacity onPress={openLogs}><Text style={st.clip}>📋</Text></TouchableOpacity>
        <TouchableOpacity onPress={() => setShowSettings(true)}><Text style={st.clip}>⚙</Text></TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={{ padding: 14 }}>
        <Text style={st.section}>INSTALLED AGENTS</Text>
        {installed.length === 0 && <Text style={st.empty}>Nothing installed yet — sync with the portal below.</Text>}
        {installed.map(a => (
          <TouchableOpacity key={a.id} style={st.card} onPress={() => {
            if (!a.modelReady || !a.embedderReady) {
              Alert.alert('Model missing', 'Re-install from the store to fetch the model files.');
              return;
            }
            openChat(a);
          }}>
            <Text style={st.cardTitle}>{a.name} <Text style={st.mutedSm}>v{a.version}</Text></Text>
            <Text style={st.mutedSm}>
              {a.manifest.model.name} · {a.manifest.rag?.chunks ?? 0} KB chunks · {(a.manifest.tools || []).length} tools
              {guard.piiEnabled(a.manifest) ? ' · 🛡 PII Guard' : ''}
            </Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
              <Text style={a.modelReady ? st.ready : st.warn}>
                {a.modelReady ? '✓ offline-ready — tap to chat' : '⚠ model files missing'}
              </Text>
              <TouchableOpacity onPress={() => onUninstall(a)} hitSlop={8}>
                <Text style={st.uninstall}>Uninstall</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        ))}

        <View style={st.storeHead}>
          <Text style={st.section}>AGENT STORE</Text>
          <TouchableOpacity onPress={refresh}><Text style={{ color: C.accent, fontSize: 12 }}>refresh</Text></TouchableOpacity>
        </View>
        <TouchableOpacity onPress={() => setShowSettings(true)}>
          <Text style={st.portalLine} numberOfLines={2}>🌐 Portal: {portal || 'not set — tap to configure'}</Text>
        </TouchableOpacity>
        {!!storeError && <Text style={st.warnBox}>{storeError}</Text>}
        {!storeError && store.length > 0 &&
          <Text style={{ color: C.green, fontSize: 11.5, marginBottom: 8 }}>✓ connected — {store.length} agent{store.length > 1 ? 's' : ''} published</Text>}
        {store.map(p => {
          const stateLabel = installing[p.id]
            ? installing[p.id]
            : p.installed_version === p.version ? 'installed'
            : p.installed_version ? `Update v${p.installed_version} → v${p.version}` : 'Install';
          const actionable = !installing[p.id] && p.installed_version !== p.version;
          return (
            <View key={p.id} style={st.card}>
              <Text style={st.cardTitle}>{p.name} <Text style={st.mutedSm}>v{p.version}</Text></Text>
              <Text style={st.mutedSm}>{p.description}</Text>
              <Text style={st.mutedSm}>
                bundle {(p.bundle_bytes / 1024).toFixed(0)} KB · model {p.model.size_gb} GB · {p.kb.chunks} chunks · {p.tools} tools
              </Text>
              {actionable ? (
                <TouchableOpacity style={st.installBtn} onPress={() => onInstall(p)}>
                  <Text style={{ color: '#fff', fontWeight: '600', fontSize: 13 }}>{stateLabel}</Text>
                </TouchableOpacity>
              ) : (
                <Text style={installing[p.id] ? st.warn : st.ready}>{stateLabel}</Text>
              )}
            </View>
          );
        })}
      </ScrollView>

      <Modal visible={showLogs} transparent animationType="fade">
        <View style={st.modalWrap}>
          <View style={[st.modal, { maxHeight: '85%' }]}>
            <Text style={[st.cardTitle, { fontSize: 16 }]}>📋 Diagnostics log</Text>
            <Text style={st.mutedSm}>Newest first. Share this when reporting a problem.</Text>
            <ScrollView style={{ marginTop: 8, maxHeight: 420 }}>
              {logLines.length === 0 && <Text style={st.empty}>No log entries yet.</Text>}
              {logLines.map((l, i) => (
                <Text key={i} selectable style={st.logLine}>{l}</Text>
              ))}
            </ScrollView>
            <View style={{ flexDirection: 'row', gap: 8, marginTop: 12 }}>
              <TouchableOpacity style={[st.installBtn, { flex: 1, marginTop: 0 }]}
                onPress={() => Share.share({ message: logLines.slice(0, 120).join('\n') })}>
                <Text style={{ color: '#fff', fontWeight: '600', textAlign: 'center' }}>Share</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[st.installBtn, { backgroundColor: C.panel2, flex: 1, marginTop: 0 }]}
                onPress={async () => { await clearLogs(); setLogLines([]); }}>
                <Text style={{ color: C.text, textAlign: 'center' }}>Clear</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[st.installBtn, { backgroundColor: C.panel2, flex: 1, marginTop: 0 }]}
                onPress={() => setShowLogs(false)}>
                <Text style={{ color: C.text, textAlign: 'center' }}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Modal visible={showSettings} transparent animationType="fade">
        <View style={st.modalWrap}>
          <View style={st.modal}>
            <Text style={[st.cardTitle, { fontSize: 16 }]}>🌐 Portal URL</Text>
            <Text style={[st.mutedSm, { fontSize: 12.5, lineHeight: 18, marginTop: 6 }]}>
              The web address of your company's Agent Studio portal — the same
              URL you open in a browser. There is ONE portal URL for all
              agents: the store below lists every agent published on it.
              {'\n\n'}For this prototype it is the tunnel address, e.g.
              {'\n'}{DEFAULT_PORTAL}
            </Text>
            <TextInput
              style={st.portalInput}
              value={portal}
              onChangeText={setPortal}
              autoCapitalize="none"
              autoCorrect={false}
              multiline
              placeholder="https://your-portal-address"
              placeholderTextColor={C.muted}
            />
            <View style={{ flexDirection: 'row', gap: 10, marginTop: 14 }}>
              <TouchableOpacity style={[st.installBtn, { flex: 1, marginTop: 0 }]} onPress={async () => {
                await setPortalUrl(portal);
                try {
                  const agents = await fetchStore();
                  setStore(agents); setStoreError('');
                  Alert.alert('Connected ✓',
                    `Portal is reachable — ${agents.length} agent${agents.length === 1 ? '' : 's'} published:\n\n` +
                    agents.map(a => `• ${a.name} v${a.version}`).join('\n'));
                  setShowSettings(false);
                  refresh();
                } catch (e: any) {
                  Alert.alert('Cannot reach portal',
                    `${String(e?.message || e).slice(0, 200)}\n\nCheck the URL (it must start with https://) and that the tunnel/portal is running.`);
                }
              }}>
                <Text style={{ color: '#fff', fontWeight: '600', textAlign: 'center' }}>Save & test</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[st.installBtn, { backgroundColor: C.panel2, flex: 1, marginTop: 0 }]}
                onPress={() => setShowSettings(false)}>
                <Text style={{ color: C.text, textAlign: 'center' }}>Cancel</Text>
              </TouchableOpacity>
            </View>

            <View style={st.divider} />
            <Text style={[st.cardTitle, { fontSize: 15 }]}>🖼️ Vision model for attached images</Text>
            <View style={st.switchRow}>
              <Text style={[st.mutedSm, { flex: 1, fontSize: 12.5, lineHeight: 18 }]}>
                {visionReady
                  ? `${VISION_MODEL.name} is on this device (${VISION_MODEL.size_gb} GB). ${visionOn ? 'Attached images are read by the vision model and by OCR.' : 'Off — images are read with OCR only.'}`
                  : `Not downloaded. Off — attached images are read with on-device OCR only. Needs ~${VISION_MODEL.size_gb} GB and a ${VISION_MODEL.min_device_ram_gb} GB+ phone.`}
              </Text>
              <Switch value={visionOn} onValueChange={toggleVision} disabled={!!visionDl}
                      trackColor={{ true: C.accent, false: C.border }} />
            </View>
            {!!visionDl && <Text style={st.warn}>{visionDl}</Text>}
            {visionReady && (
              <TouchableOpacity onPress={async () => {
                await setVisionEnabled(false); await removeVisionFiles();
                setVisionOn(false); setVisionReady(false);
              }}>
                <Text style={[st.uninstall, { marginTop: 8 }]}>Remove vision model files</Text>
              </TouchableOpacity>
            )}

            <View style={st.divider} />
            <Text style={[st.cardTitle, { fontSize: 15 }]}>🧠 Blocked models</Text>
            <Text style={[st.mutedSm, { fontSize: 12.5, lineHeight: 18 }]}>
              {blockedModels.length
                ? `A model is blocked after one failed load so the app cannot crash in a loop. Blocked here: ${blockedModels.join(', ')}. Retry gives each one a fresh attempt — close other apps first.`
                : 'None. A model is blocked here only after it fails to load on this phone.'}
            </Text>
            {blockedModels.length > 0 && (
              <TouchableOpacity style={[st.installBtn, { marginTop: 10 }]} onPress={async () => {
                const n = await unblockAllModels();
                setBlockedModels([]);
                Alert.alert('Unblocked', `${n} model${n === 1 ? '' : 's'} will be tried again on the next chat.`);
              }}>
                <Text style={{ color: '#fff', fontWeight: '600', fontSize: 13 }}>Retry blocked models</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const st = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  header: {
    flexDirection: 'row', alignItems: 'center', padding: 14,
    backgroundColor: C.panel, borderBottomWidth: 1, borderBottomColor: C.border,
  },
  hTitle: { color: C.text, fontSize: 16, fontWeight: '700' },
  hSub: { color: C.muted, fontSize: 11.5, marginTop: 2 },
  back: { color: C.accent, fontSize: 15 },
  clip: { fontSize: 22, color: C.text, paddingHorizontal: 6 },
  section: { color: C.muted, fontSize: 11, letterSpacing: 0.5, marginTop: 14, marginBottom: 8 },
  storeHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' },
  empty: { color: C.muted, fontSize: 12.5, paddingVertical: 6 },
  card: {
    backgroundColor: C.panel, borderColor: C.border, borderWidth: 1,
    borderRadius: 12, padding: 14, marginBottom: 10,
  },
  cardTitle: { color: C.text, fontSize: 14.5, fontWeight: '600', marginBottom: 3 },
  mutedSm: { color: C.muted, fontSize: 11.5, marginBottom: 3 },
  ready: { color: C.green, fontSize: 12, marginTop: 5 },
  warn: { color: C.warn, fontSize: 12, marginTop: 5 },
  uninstall: { color: C.danger, fontSize: 12, marginTop: 5, paddingHorizontal: 4 },
  warnBox: {
    color: '#d8c9a3', fontSize: 12, backgroundColor: '#1a1712',
    borderLeftWidth: 3, borderLeftColor: C.warn, padding: 9, borderRadius: 6, marginBottom: 10,
  },
  installBtn: {
    backgroundColor: C.accent, borderRadius: 8, paddingVertical: 9,
    paddingHorizontal: 14, alignSelf: 'flex-start', marginTop: 8,
  },
  bubble: { maxWidth: '86%', borderRadius: 14, padding: 11, borderWidth: 1 },
  userBubble: { alignSelf: 'flex-end', backgroundColor: C.userBubble, borderColor: C.userBorder },
  botBubble: { alignSelf: 'flex-start', backgroundColor: C.panel2, borderColor: C.border },
  toolBubble: { alignSelf: 'flex-start', backgroundColor: C.toolBg, borderColor: C.toolBorder, maxWidth: '92%' },
  toolText: { color: C.toolText, fontSize: 11.5, fontFamily: 'monospace' },
  guardBubble: { alignSelf: 'flex-start', backgroundColor: '#1f1a0e', borderColor: '#6b5418', maxWidth: '92%' },
  guardText: { color: '#e8c76a', fontSize: 13, lineHeight: 19 },
  attTag: { color: '#cfe0ff', fontSize: 11.5, marginBottom: 4 },
  msgText: { color: C.text, fontSize: 13.5, lineHeight: 19 },
  statusMsg: { color: C.muted, fontSize: 12, textAlign: 'center', paddingVertical: 6, lineHeight: 17 },
  srcRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 4, marginTop: 8, paddingTop: 6, borderTopWidth: 1, borderTopColor: C.border },
  srcChip: {
    color: C.muted, fontSize: 10, backgroundColor: '#10151c', borderWidth: 1,
    borderColor: C.border, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 1,
  },
  srcChipAtt: { color: C.green, borderColor: C.toolBorder, backgroundColor: C.toolBg },
  stats: { color: C.muted, fontSize: 10, marginTop: 6 },
  chip: {
    flexDirection: 'row', alignItems: 'center', marginHorizontal: 10, marginTop: 8,
    padding: 9, borderRadius: 10, borderWidth: 1, borderColor: C.userBorder, backgroundColor: C.userBubble,
  },
  chipBlocked: { borderColor: '#7a2e2e', backgroundColor: '#2a1414' },
  chipClear: { borderColor: C.toolBorder, backgroundColor: C.toolBg },
  chipName: { color: C.text, fontSize: 12.5, fontWeight: '600' },
  chipMeta: { color: C.muted, fontSize: 11, flex: 1 },
  chipX: { color: C.muted, fontSize: 15, fontWeight: '700', paddingHorizontal: 4 },
  composer: {
    flexDirection: 'row', gap: 8, padding: 10, backgroundColor: C.panel,
    borderTopWidth: 1, borderTopColor: C.border,
  },
  input: {
    flex: 1, backgroundColor: C.bg, borderWidth: 1, borderColor: C.border,
    borderRadius: 20, color: C.text, paddingHorizontal: 15, paddingVertical: 8, fontSize: 13.5,
  },
  sendBtn: {
    backgroundColor: C.accent, borderRadius: 20, width: 42, alignItems: 'center', justifyContent: 'center',
  },
  portalLine: { color: C.muted, fontSize: 11.5, marginBottom: 8 },
  logLine: { color: C.muted, fontSize: 10.5, fontFamily: 'monospace', marginBottom: 2 },
  portalInput: {
    backgroundColor: C.bg, borderWidth: 1, borderColor: C.accent, borderRadius: 10,
    color: C.text, paddingHorizontal: 12, paddingVertical: 10, fontSize: 15,
    minHeight: 76, textAlignVertical: 'top', marginTop: 12, fontFamily: 'monospace',
  },
  divider: { height: 1, backgroundColor: C.border, marginVertical: 16 },
  switchRow: { flexDirection: 'row', alignItems: 'center', gap: 12, marginTop: 6 },
  modalWrap: { flex: 1, backgroundColor: '#000a', alignItems: 'center', justifyContent: 'center', padding: 24 },
  modal: {
    backgroundColor: C.panel, borderColor: C.border, borderWidth: 1,
    borderRadius: 14, padding: 18, width: '100%',
  },
});
