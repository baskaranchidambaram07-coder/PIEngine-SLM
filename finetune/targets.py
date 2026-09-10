"""Ways to ask an agent a question, so the same probe can grade either the
deployed system or the model in isolation.

Two targets, and the difference between them is the point:

  runtime  POST /api/chat on the Device Runtime. Measures the SHIPPED system —
           lexical gate, retrieval, prompt assembly, tool loop, streaming. This
           is the number that describes what a user experiences.

  llama    llama-server directly, with the same prompt and the same retrieval
           but nothing else. Isolates the WEIGHTS, so a baseline/adapter A/B
           attributes the delta to the tune rather than to a prompt edit that
           happened to land in the same week.

Both return the same Observation, so evaluate.py does not care which was used.
"""
from __future__ import annotations

import json
import re
import sqlite3

import requests

from core import embeddings, gating, kbstore
from core.paths import ROOT

from .checks import Observation

RUNTIME_URL = "http://127.0.0.1:8200"
LLAMA_URL = "http://127.0.0.1:8302"
STUDIO_DB = ROOT / "studio" / "studio.db"
STUDIO_KB = ROOT / "studio" / "kb"
DEVICE_AGENTS = ROOT / "runtime" / "device_storage" / "agents"

_TAG = re.compile(r"<tool_call>(.*?)</tool_call>|<think>.*?</think>", re.DOTALL)


# ------------------------------------------------------------------ agent view

class AgentCtx:
    """Everything a probe needs about one agent, from whichever store has it.

    Prefers the installed bundle (that is the artifact devices actually run);
    falls back to the Studio's working copy so a spec can be evaluated before
    the agent has ever been published.
    """

    def __init__(self, agent_id: str):
        self.id = agent_id
        manifest_path = DEVICE_AGENTS / agent_id / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.kb_path = DEVICE_AGENTS / agent_id / "kb.sqlite"
            self.source = "installed bundle"
        else:
            row = sqlite3.connect(STUDIO_DB).execute(
                "SELECT config FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if not row:
                raise SystemExit(f"agent '{agent_id}' is neither installed nor in studio.db")
            cfg = json.loads(row[0])
            self.manifest = {
                "id": agent_id, "system_prompt": cfg.get("system_prompt", ""),
                "generation": cfg.get("generation", {}), "rag": cfg.get("rag", {}),
                "tools": cfg.get("tools", []), "model": {"id": cfg.get("model_id")},
            }
            self.kb_path = STUDIO_KB / f"{agent_id}.sqlite"
            self.source = "studio draft"

    @property
    def tools(self) -> list[dict]:
        return self.manifest.get("tools", [])

    @property
    def generation(self) -> dict:
        return self.manifest.get("generation", {})

    def retrieve(self, query: str) -> list[dict]:
        """The runtime's own retrieval path, gate included."""
        if not self.kb_path.exists():
            return []
        ok, _reason = gating.should_retrieve(query)
        if not ok:
            return []
        rag = self.manifest.get("rag", {})
        qvec = embeddings.embed_query(query)
        return kbstore.search(kbstore.connect(self.kb_path), qvec,
                              top_k=int(rag.get("top_k", 4)),
                              min_score=float(rag.get("min_score", 0.45)))

    def doc_names(self) -> list[str]:
        if not self.kb_path.exists():
            return []
        conn = sqlite3.connect(self.kb_path)
        return [r[0] for r in conn.execute("SELECT DISTINCT doc_name FROM chunks")]


def context_block(sources: list[dict]) -> str:
    if not sources:
        return ""
    body = "\n\n".join(f"[{s['doc_name']} #{s['chunk_index']}]\n{s['text']}" for s in sources)
    return "Context from the knowledge base:\n\n" + body + "\n\n---\n\n"


# ------------------------------------------------------------------ targets

class RuntimeTarget:
    """The shipped system, over its own SSE chat endpoint."""

    name = "runtime"

    def __init__(self, agent: AgentCtx, url: str = RUNTIME_URL, timeout: int = 600):
        self.agent = agent
        self.url = url.rstrip("/")
        self.timeout = timeout

    def describe(self) -> dict:
        return {"target": self.name, "url": self.url, "agent": self.agent.id}

    def ask(self, question: str, history: list[dict] | None = None,
            context: str | None = None, stop_after_tool: bool = False) -> Observation:
        """Run one turn through the shipped chat endpoint.

        `stop_after_tool` closes the stream as soon as the first tool call is
        observed. A tool requirement is satisfied or failed by that first call;
        letting the runtime finish its loop costs several more generation
        rounds — minutes at 2-8 tok/s — and grades the follow-up answer, which
        is a different requirement.
        """
        if context is not None:
            raise SystemExit("probe `context:` overrides retrieval — use --target llama")
        messages = list(history or []) + [{"role": "user", "content": question}]
        resp = requests.post(f"{self.url}/api/chat", stream=True, timeout=self.timeout,
                             json={"agent_id": self.agent.id, "messages": messages})
        resp.raise_for_status()
        resp.encoding = "utf-8"

        text, calls, docs, stats = "", [], [], {}
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            kind = ev.get("type")
            if kind == "token":
                text += ev["text"]
            elif kind == "sources":
                docs = [i["doc"] for i in ev.get("items", [])]
            elif kind == "tool":
                calls.append({"name": ev.get("name"), "arguments": ev.get("args") or {}})
                if stop_after_tool:
                    resp.close()
                    break
            elif kind == "stats":
                stats = ev
            elif kind == "error":
                raise SystemExit(f"runtime error: {ev.get('text')}")
        sources = self.agent.retrieve(question)   # for the grounding check's haystack
        return Observation(
            text=text.strip(), tool_calls=calls,
            context=context_block(sources), doc_names=docs or [s["doc_name"] for s in sources],
            tool_defs=self.agent.tools,
            seconds=float(stats.get("seconds") or 0), tokens=int(stats.get("tokens") or 0),
        )


class LlamaTarget:
    """llama-server directly, with an optional LoRA adapter applied.

    Adapter A/B runs in ONE server process: start llama-server with
    `--lora <adapter.gguf> --lora-init-without-apply`, then flip the scale
    between 0 and 1 through POST /lora-adapters. Same weights in RAM, same KV
    cache behaviour, so the only variable is the adapter.
    """

    name = "llama"

    def __init__(self, agent: AgentCtx, url: str = LLAMA_URL, adapter_scale: float | None = None,
                 timeout: int = 600):
        self.agent = agent
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.adapter_scale = adapter_scale
        self.adapters = self._adapters()
        if adapter_scale is not None:
            self.set_adapter_scale(adapter_scale)

    # -------------------------------------------------------- adapter control
    def _adapters(self) -> list[dict]:
        try:
            r = requests.get(f"{self.url}/lora-adapters", timeout=5)
            return r.json() if r.ok else []
        except requests.RequestException:
            return []

    def set_adapter_scale(self, scale: float) -> None:
        if not self.adapters:
            if scale:
                raise SystemExit(
                    "llama-server has no LoRA adapter loaded. Restart it with:\n"
                    "  llama-server -m <base.gguf> --lora <adapter.gguf> "
                    "--lora-init-without-apply --port 8302")
            return
        payload = [{"id": a["id"], "scale": scale} for a in self.adapters]
        r = requests.post(f"{self.url}/lora-adapters", json=payload, timeout=10)
        r.raise_for_status()
        self.adapter_scale = scale

    def describe(self) -> dict:
        return {"target": self.name, "url": self.url, "agent": self.agent.id,
                "adapters": [a.get("path") for a in self.adapters],
                "adapter_scale": self.adapter_scale}

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, history: list[dict] | None = None,
            context: str | None = None, stop_after_tool: bool = False) -> Observation:
        # single-round by construction: no tool loop to cut short
        from runtime.app import build_system_prompt   # the exact deployed prompt

        if context is None:
            sources = self.agent.retrieve(question)
            ctx = context_block(sources)
            docs = [s["doc_name"] for s in sources]
        else:
            ctx = context
            docs = re.findall(r"\[([^\]\s#]+)", context)

        messages = [{"role": "system", "content": build_system_prompt(self.manifest_for_prompt())}]
        messages += list(history or [])
        messages.append({"role": "user", "content": ctx + question})

        gen = {**self.agent.generation}
        body = {"messages": messages, "stream": False,
                "temperature": gen.get("temperature", 0.7), "top_p": gen.get("top_p", 0.8),
                "max_tokens": gen.get("max_tokens", 768)}
        if gen.get("min_p"):
            body["min_p"] = gen["min_p"]
        r = requests.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        data = r.json()
        raw = data["choices"][0]["message"].get("content") or ""
        timings = data.get("timings", {}) or {}

        calls = []
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL):
            try:
                calls.append(json.loads(m.group(1).strip()))
            except json.JSONDecodeError:
                calls.append({"name": None, "arguments": m.group(1).strip()})
        text = _TAG.sub("", raw).strip()

        return Observation(
            text=text, tool_calls=calls, context=ctx, doc_names=docs,
            tool_defs=self.agent.tools,
            seconds=round((timings.get("predicted_ms", 0) + timings.get("prompt_ms", 0)) / 1000, 2),
            tokens=int(timings.get("predicted_n") or 0),
        )

    def manifest_for_prompt(self) -> dict:
        return {"system_prompt": self.agent.manifest.get("system_prompt", ""),
                "tools": self.agent.tools}


def build(agent_id: str, target: str, adapter_scale: float | None = None) -> RuntimeTarget | LlamaTarget:
    ctx = AgentCtx(agent_id)
    if target == "runtime":
        return RuntimeTarget(ctx)
    if target == "llama":
        return LlamaTarget(ctx, adapter_scale=adapter_scale)
    raise SystemExit(f"unknown target {target!r} (runtime | llama)")
