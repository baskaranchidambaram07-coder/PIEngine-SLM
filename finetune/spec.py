"""The requirement spec — the contract a fine-tune is held to.

One YAML file drives three consumers, which is the whole point of the design:

  synth.py     reads `requirements[].synth`  -> what training data to make
  evaluate.py  reads `requirements[].probes` -> how the model is scored
  gate         reads `requirements[].target` -> whether the adapter may ship

Because the same file states the requirement, generates the data for it and
tests it, a fine-tune cannot quietly drift into "we trained on something and it
feels better". A requirement with no probes cannot be claimed; a requirement
that already passes at baseline is not trained on at all.

A requirement is a BEHAVIOUR, never a fact. See docs/finetuning.md D1: the
knowledge base is a bundle artifact that changes weekly, so facts belong in
RAG. Weights get citation discipline, refusal discipline, tool-call shape and
house format — things that stay true across every KB the agent will ever see.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SPEC_SCHEMA = "slm-finetune-spec/1"

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent / "specs"
STUDIO_DB = ROOT / "studio" / "studio.db"

# Behavioural requirement kinds. Each maps to a synth recipe and a default
# check set; anything outside this list is a fact, and facts belong in the KB.
KINDS = {
    "citation",   # names the source document for every asserted fact
    "grounding",  # asserts only what the retrieved context supports
    "refusal",    # declines when the context lacks the answer
    "tool_call",  # emits the right tool with valid JSON arguments
    "format",     # house output shape (bullets, MOM skeleton, ...)
    "length",     # answers inside the phone-glance budget
    "register",   # domain phrasing and tone
}


@dataclass
class Probe:
    """One graded input. `expect` is a dict of checks from checks.py."""
    ask: str
    expect: dict = field(default_factory=dict)
    context: str | None = None   # override retrieval (isolated model A/B)
    history: list[dict] = field(default_factory=list)
    note: str = ""

    @classmethod
    def parse(cls, raw: dict | str) -> "Probe":
        if isinstance(raw, str):
            return cls(ask=raw)
        return cls(ask=raw["ask"], expect=raw.get("expect", {}) or {},
                   context=raw.get("context"), history=raw.get("history", []) or [],
                   note=raw.get("note", ""))


@dataclass
class Requirement:
    id: str
    kind: str
    statement: str
    target: float = 0.9          # pass rate needed to promote
    weight: float = 1.0
    probes: list[Probe] = field(default_factory=list)
    synth: dict = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def parse(cls, raw: dict) -> "Requirement":
        return cls(
            id=raw["id"], kind=raw["kind"], statement=raw["statement"],
            target=float(raw.get("target", 0.9)), weight=float(raw.get("weight", 1.0)),
            probes=[Probe.parse(p) for p in raw.get("probes", [])],
            synth=raw.get("synth", {}) or {}, rationale=raw.get("rationale", ""),
        )


@dataclass
class Spec:
    agent: str
    base_model: str
    adapter_id: str
    description: str = ""
    requirements: list[Requirement] = field(default_factory=list)
    regression: list[Probe] = field(default_factory=list)
    training: dict = field(default_factory=dict)
    path: Path | None = None

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, ref: str | Path) -> "Spec":
        """Load by path, or by bare name from finetune/specs/<name>.yaml."""
        p = Path(ref)
        if not p.exists():
            p = SPEC_DIR / f"{Path(ref).stem}.yaml"
        if not p.exists():
            raise SystemExit(f"spec not found: {ref} (looked in {SPEC_DIR})")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if raw.get("spec") != SPEC_SCHEMA:
            raise SystemExit(f"{p}: expected spec: {SPEC_SCHEMA}, got {raw.get('spec')!r}")
        spec = cls(
            agent=raw["agent"], base_model=raw["base_model"], adapter_id=raw["adapter_id"],
            description=raw.get("description", ""),
            requirements=[Requirement.parse(r) for r in raw.get("requirements", [])],
            regression=[Probe.parse(p) for p in raw.get("regression", [])],
            training=raw.get("training", {}) or {},
            path=p,
        )
        problems = spec.problems()
        if problems:
            raise SystemExit(f"{p} is invalid:\n  - " + "\n  - ".join(problems))
        return spec

    def problems(self) -> list[str]:
        out = []
        seen: set[str] = set()
        if not self.requirements:
            out.append("no requirements declared")
        for r in self.requirements:
            if r.id in seen:
                out.append(f"duplicate requirement id {r.id!r}")
            seen.add(r.id)
            if r.kind not in KINDS:
                out.append(f"{r.id}: kind {r.kind!r} is not one of {sorted(KINDS)}")
            if not r.probes:
                out.append(f"{r.id}: no probes — an untestable requirement cannot be claimed")
            if not 0 < r.target <= 1:
                out.append(f"{r.id}: target {r.target} must be in (0, 1]")
            for i, p in enumerate(r.probes):
                if not p.expect:
                    out.append(f"{r.id} probe[{i}]: no expect checks")
        return out

    # ------------------------------------------------------------- helpers
    def requirement(self, rid: str) -> Requirement | None:
        return next((r for r in self.requirements if r.id == rid), None)

    def probe_texts(self) -> set[str]:
        """Every graded input, for the train/eval leakage check."""
        return {p.ask.strip().lower() for r in self.requirements for p in r.probes} | \
               {p.ask.strip().lower() for p in self.regression}

    def train_cfg(self) -> dict:
        base = {
            "epochs": 3.0, "lr": 2e-4, "batch": 2, "grad_accum": 8,
            "max_len": 2048, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
            "seed": 7,
        }
        base.update(self.training)
        return base


# ------------------------------------------------------------------ scaffold

_HEADER = '''# Requirement spec for a scenario fine-tune.
# Generated by: python -m finetune scaffold --agent {agent}
#
# Before you edit:
#   * A requirement is a BEHAVIOUR, not a fact. "cites the source document" is a
#     requirement; "knows the Phase 2 budget" is a knowledge-base entry.
#   * Every requirement needs probes. Run `python -m finetune baseline` first —
#     anything already above target must be DELETED from this file, not trained
#     on. Spending adapter capacity on solved behaviour is how tunes regress.
#   * Probes are the held-out test set. synth.py refuses to emit any training
#     example whose question matches a probe.

spec: {schema}
agent: {agent}
base_model: {base_model}
adapter_id: {agent}-behaviour-v1
description: >-
  {description}

training:
  epochs: 3
  lr: 2.0e-4
  lora_r: 16
  max_len: 2048

requirements:
'''

_REGRESSION = '''
# Behaviour that must NOT regress. Not trained on — this is the blast-radius
# check. A tune that wins its requirements and loses these is rejected.
regression:
  - ask: "hi"
    expect:
      max_words: 30
      no_tool_call: true
  - ask: "thanks!"
    expect:
      max_words: 30
  - ask: "what is the weather in Chennai today?"
    expect:
      refuses: true
'''


def scaffold(agent_id: str) -> str:
    """Emit a starter spec from a Studio agent's own prompt, tools and KB."""
    conn = sqlite3.connect(STUDIO_DB)
    row = conn.execute("SELECT config FROM agents WHERE id = ?", (agent_id,)).fetchone()
    if not row:
        raise SystemExit(f"agent '{agent_id}' not found in {STUDIO_DB}")
    cfg = json.loads(row[0])

    out = _HEADER.format(
        schema=SPEC_SCHEMA, agent=agent_id,
        base_model=cfg.get("model_id", "qwen3-1.7b-q4_k_m"),
        description=(cfg.get("description") or "Scenario fine-tune.").replace("\n", " "),
    )

    blocks = [
        _req_block(
            "R1-cite", "citation",
            "Every asserted fact names the source document it came from.",
            "Without a source name the user cannot verify an answer, and an "
            "unverifiable answer from a 1.7B model is not usable evidence.",
            [("<a question your KB can answer>",
              ['cites: true', 'contains_any: ["<expected fact>"]'])],
            ["recipe: grounded_qa", "n: 60", "surrogates: [project-delivery, customer]"],
        ),
        _req_block(
            "R2-refuse", "refusal",
            "Declines when the retrieved context does not contain the answer.",
            "The dangerous failure is a confident wrong answer about an absent entity "
            "that sounds in-domain (see docs/rag-gating.md).",
            [("<a plausible question your KB CANNOT answer>", ['refuses: true'])],
            ["recipe: absent_entity", "n: 40", "surrogates: [project-delivery]"],
        ),
    ]
    tools = cfg.get("tools", [])
    if tools:
        name = tools[0]["name"]
        blocks.append(_req_block(
            "R3-tools", "tool_call",
            "Emits a single well-formed <tool_call> with the right name and required arguments.",
            "A malformed tool call is silently dropped by the runtime parser, so the "
            "user just sees the model ignore them.",
            [(f"<a request that should fire {name}>",
              [f'tool_call: {name}', 'tool_args_valid: true'])],
            ["recipe: tool_router", "n: 60"],
        ))
    blocks.append(_req_block(
        "R4-brief", "length",
        "Answers fit a phone glance: a lead line plus at most five bullets.",
        "The user is between meetings; a 400-word answer is a failed answer.",
        [("<any KB question>", ['max_words: 90'])],
        ["recipe: grounded_qa", "n: 20", "surrogates: [project-delivery]"],
    ))
    return out + "\n".join(blocks) + _REGRESSION


def _req_block(rid: str, kind: str, statement: str, rationale: str,
               probes: list[tuple[str, list[str]]], synth: list[str]) -> str:
    lines = [f"  - id: {rid}", f"    kind: {kind}",
             "    statement: >-", f"      {statement}",
             "    rationale: >-", f"      {rationale}",
             "    target: 0.9", "    weight: 1", "    probes:"]
    for ask, checks in probes:
        lines.append(f'      - ask: "{ask}"')
        lines.append("        expect:")
        lines += [f"          {c}" for c in checks]
    lines.append("    synth:")
    lines += [f"      {s}" for s in synth]
    return "\n".join(lines) + "\n"
