/**
 * Offline Enterprise Agent Runtime — Android client.
 * Syncs with the Agent Studio portal, installs agent bundles, and runs them
 * fully on-device: llama.rn inference, SQLite vector KB, on-device tools,
 * and user-uploaded inline KB (≤2 MB per file).
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Alert, FlatList, KeyboardAvoidingView, Modal,
  SafeAreaView, ScrollView, Share, StatusBar, StyleSheet, Text, TextInput,
  TouchableOpacity, View,
} from 'react-native';
import { pick } from '@react-native-documents/picker';
import ReactNativeBlobUtil from 'react-native-blob-util';

import { C } from './src/theme';
import { DEFAULT_PORTAL, MAX_UPLOAD_BYTES } from './src/config';
import {
  fetchStore, getPortalUrl, installAgent, InstalledAgent, listInstalled,
  Progress, setPortalUrl, StoreEntry,
} from './src/portal';
import { runChat } from './src/chat';
import { ChatTurn, embedText } from './src/llm';
import { splitText } from './src/chunker';
import { addInlineDoc, inlineStats, SearchHit } from './src/kb';
import { clearLogs, getLogs, log } from './src/logger';
import { reportTelemetry } from './src/appTelemetry';

type Msg = {
  id: string;
  kind: 'user' | 'bot' | 'tool' | 'status';
  text: string;
  sources?: SearchHit[];
  statsLine?: string;
};

let msgSeq = 0;
const nextId = () => `m${++msgSeq}`;

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
  const historyRef = useRef<ChatTurn[]>([]);
  const listRef = useRef<FlatList<Msg>>(null);

  const refresh = useCallback(async () => {
    setInstalled(await listInstalled());
    setPortal(await getPortalUrl());
    try {
      setStore(await fetchStore());
      setStoreError('');
    } catch (e: any) {
      setStore([]);
      setStoreError(`Portal unreachable (${String(e?.message || e).slice(0, 80)}). Installed agents keep working offline.`);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

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

  const openLogs = async () => {
    setLogLines([...(await getLogs())].reverse());
    setShowLogs(true);
  };

  // ---------------------------------------------------------------- chat

  const openChat = async (a: InstalledAgent) => {
    setAgent(a);
    historyRef.current = [];
    setMsgs([{
      id: nextId(), kind: 'status',
      text: `🔒 ${a.name} v${a.version} — ${a.manifest.model.name}\nKB: ${a.manifest.rag?.chunks ?? 0} chunks · Tools: ${(a.manifest.tools || []).map((t: any) => t.name).join(', ') || 'none'}\nFirst answer loads the model into RAM.`,
    }]);
    setInline(await inlineStats(a.id));
    setScreen('chat');
  };

  const appendBotText = (id: string, piece: string) => {
    setMsgs(m => m.map(x => (x.id === id ? { ...x, text: x.text + piece } : x)));
  };

  const send = async () => {
    if (!agent || busy) return;
    const text = input.trim();
    if (!text) return;
    setInput('');
    setBusy(true);

    historyRef.current.push({ role: 'user', content: text });
    const botId = nextId();
    setMsgs(m => [...m,
      { id: nextId(), kind: 'user', text },
      { id: botId, kind: 'bot', text: '' },
    ]);

    let sources: SearchHit[] = [];
    let statsLine = '';
    const answer = await runChat(agent.manifest, agent.id, [...historyRef.current], ev => {
      if (ev.type === 'token') appendBotText(botId, ev.text);
      else if (ev.type === 'sources') sources = ev.items;
      else if (ev.type === 'status') {
        setMsgs(m => m.map(x => (x.id === botId && !x.text
          ? { ...x, statsLine: ev.text } : x)));
      } else if (ev.type === 'tool') {
        setMsgs(m => [...m, {
          id: nextId(), kind: 'tool',
          text: `⚙ ${ev.name}(${JSON.stringify(ev.args)})\n→ ${JSON.stringify(ev.result).slice(0, 280)}`,
        }]);
      } else if (ev.type === 'stats') {
        statsLine = `${ev.stats.tokens} tok @ ${ev.stats.tok_per_sec}/s · prompt ${ev.stats.prefill_tokens} tok @ ${ev.stats.prefill_per_sec}/s`;
      } else if (ev.type === 'error') {
        appendBotText(botId, `\n⚠ ${ev.text}`);
      }
    });

    historyRef.current.push({ role: 'assistant', content: answer });
    setMsgs(m => m.map(x => (x.id === botId ? { ...x, sources, statsLine } : x)));
    setBusy(false);
  };

  // ------------------------------------------------------ inline KB upload

  const uploadFile = async () => {
    if (!agent || busy) return;
    try {
      const [res] = await pick({ mode: 'open' });
      if (!res) return;
      const name = res.name || 'upload.txt';
      const size = Number(res.size || 0);
      if (size > MAX_UPLOAD_BYTES) {
        Alert.alert('File too large', `Limit is 2 MB — "${name}" is ${(size / 1048576).toFixed(1)} MB.`);
        return;
      }
      if (!/\.(txt|md|markdown|csv|log)$/i.test(name)) {
        Alert.alert('Unsupported type', 'Upload .txt, .md, .csv or .log files. (PDFs: add them via the web Studio.)');
        return;
      }
      setBusy(true);
      setMsgs(m => [...m, { id: nextId(), kind: 'status', text: `📎 Indexing ${name} on-device…` }]);

      const path = res.uri.startsWith('content://') ? res.uri : res.uri.replace('file://', '');
      const content = await ReactNativeBlobUtil.fs.readFile(path, 'utf8');
      const chunks = splitText(String(content));
      if (!chunks.length) throw new Error('no extractable text');
      const embeddings: Float32Array[] = [];
      for (const chnk of chunks) embeddings.push(await embedText(chnk, false));
      await addInlineDoc(agent.id, name, chunks, embeddings);
      const st2 = await inlineStats(agent.id);
      setInline(st2);
      setMsgs(m => [...m, {
        id: nextId(), kind: 'status',
        text: `✅ ${name}: ${chunks.length} chunks added to inline KB (now ${st2.docs} docs / ${st2.chunks} chunks on top of the bundle KB). Ask away.`,
      }]);
    } catch (e: any) {
      const s = String(e?.message || e);
      if (!/cancel/i.test(s)) Alert.alert('Upload failed', s.slice(0, 200));
    } finally {
      setBusy(false);
    }
  };

  // ------------------------------------------------------------ rendering

  const renderMsg = ({ item }: { item: Msg }) => {
    if (item.kind === 'status') {
      return <Text style={st.statusMsg}>{item.text}</Text>;
    }
    if (item.kind === 'tool') {
      return <View style={[st.bubble, st.toolBubble]}><Text style={st.toolText}>{item.text}</Text></View>;
    }
    const user = item.kind === 'user';
    return (
      <View style={[st.bubble, user ? st.userBubble : st.botBubble]}>
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
              <Text key={i} style={st.srcChip}>
                📄 {s2.doc_name} #{s2.chunk_index} · {s2.score}{s2.source === 'inline' ? ' · 📎' : ''}
              </Text>
            ))}
          </View>
        )}
        {!user && !!item.text && !!item.statsLine && <Text style={st.stats}>{item.statsLine}</Text>}
      </View>
    );
  };

  if (screen === 'chat' && agent) {
    return (
      <SafeAreaView style={st.root}>
        <StatusBar barStyle="light-content" backgroundColor={C.panel} />
        <View style={st.header}>
          <TouchableOpacity onPress={() => setScreen('home')}><Text style={st.back}>‹ Agents</Text></TouchableOpacity>
          <View style={{ flex: 1, marginLeft: 10 }}>
            <Text style={st.hTitle} numberOfLines={1}>{agent.name}</Text>
            <Text style={st.hSub} numberOfLines={1}>
              ● on-device · {agent.manifest.model.name}
              {inline.chunks ? ` · 📎 ${inline.chunks} inline chunks` : ''}
            </Text>
          </View>
          <TouchableOpacity onPress={uploadFile} disabled={busy}>
            <Text style={st.clip}>📎</Text>
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
          <View style={st.composer}>
            <TextInput
              style={st.input}
              value={input}
              onChangeText={setInput}
              placeholder="Ask your agent…"
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
            </Text>
            <Text style={a.modelReady ? st.ready : st.warn}>
              {a.modelReady ? '✓ offline-ready — tap to chat' : '⚠ model files missing'}
            </Text>
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
  msgText: { color: C.text, fontSize: 13.5, lineHeight: 19 },
  statusMsg: { color: C.muted, fontSize: 12, textAlign: 'center', paddingVertical: 6, lineHeight: 17 },
  srcRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 4, marginTop: 8, paddingTop: 6, borderTopWidth: 1, borderTopColor: C.border },
  srcChip: {
    color: C.muted, fontSize: 10, backgroundColor: '#10151c', borderWidth: 1,
    borderColor: C.border, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 1,
  },
  stats: { color: C.muted, fontSize: 10, marginTop: 6 },
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
  modalWrap: { flex: 1, backgroundColor: '#000a', alignItems: 'center', justifyContent: 'center', padding: 24 },
  modal: {
    backgroundColor: C.panel, borderColor: C.border, borderWidth: 1,
    borderRadius: 14, padding: 18, width: '100%',
  },
});
