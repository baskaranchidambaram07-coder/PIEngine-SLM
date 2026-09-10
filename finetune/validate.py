"""Dataset gate — everything that must be true before a GPU is booked.

A bad dataset costs a training run, a merge, a quantise, a fleet download and
an evaluation before anyone finds out. All of the failure modes below are
cheap to detect here and expensive to detect later:

  leakage        a probe question in the training set turns the scorecard into
                 a memorisation test and the tune looks better than it is
  duplication    3 epochs over a duplicated example is 6 epochs on that example
  imbalance      one requirement dominating the set is how tunes over-refuse
  overlength     examples past max_len are silently truncated mid-answer, which
                 teaches the model to stop mid-answer
  PRIVACY        any real knowledge-base fact in the set means customer content
                 is being written into weights that get distributed to a fleet

The privacy check is the one that is not about model quality. Facts belong in
the bundle's kb.sqlite, which is per-agent, revocable and never leaves the
device; weights are global and permanent. See docs/finetuning.md D1.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

from .spec import Spec
from .synth import Example
from .targets import AgentCtx

# Qwen3's BPE averages ~3.6 chars/token on this kind of English + markdown.
# Only used for a length warning, so an estimate is fine.
CHARS_PER_TOKEN = 3.6

MAX_SHARE = 0.55        # no single requirement may own more than this
MAX_REFUSAL_SHARE = 0.35  # refusal-heavy sets produce models that refuse
MIN_EXAMPLES = 60

_CAP = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]{3,}\b")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict = {}

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["dataset validation"]
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for e in self.errors:
            lines.append(f"  FAIL  {e}")
        lines.append(f"  => {'PASS' if self.ok else 'BLOCKED'}")
        return "\n".join(lines)


def _surrogate_vocabulary() -> set[str]:
    """Every capitalised token the synthetic side is entitled to produce.

    Scanned from the package's own source: the templates, the entity pools and
    the literal strings the recipes build answers out of. A term the
    synthesiser can legitimately emit tells us nothing about leakage, so
    subtracting it turns this check from a wall of generic business words into
    a short list of things that could only have come from the customer's own
    documents.
    """
    pkg = Path(__file__).resolve().parent
    vocab: set[str] = set()
    for path in list(pkg.rglob("*.md.tmpl")) + list(pkg.rglob("*.py")):
        # specs/ is excluded on purpose: probe questions legitimately contain
        # real customer facts ("Acme", "Priya"), and treating those as allowed
        # would blind the check to exactly what it exists to catch.
        if "specs" in path.parts or "datasets" in path.parts:
            continue
        vocab |= set(_CAP.findall(path.read_text(encoding="utf-8")))
    return vocab


def _real_kb_terms(agent_id: str) -> set[str]:
    """Capitalised terms that appear in the agent's ACTUAL knowledge base and
    nowhere in the surrogate vocabulary — i.e. terms that could only have got
    into the dataset by copying customer content."""
    ctx = AgentCtx(agent_id)
    if not ctx.kb_path.exists():
        return set()
    conn = sqlite3.connect(ctx.kb_path)
    text = " ".join(r[0] for r in conn.execute("SELECT text FROM chunks"))
    terms = set(_CAP.findall(text))

    allowed = _surrogate_vocabulary()
    allowed |= set(_CAP.findall(ctx.manifest.get("system_prompt", "")))
    allowed |= {w for d in ctx.doc_names() for w in re.split(r"[^A-Za-z]+", d) if w}
    return {t for t in terms if t not in allowed and len(t) > 4}


def validate(spec: Spec, examples: list[Example], check_privacy: bool = True,
             skipped: set[str] | None = None) -> Report:
    rep = Report()
    rep.stats["examples"] = len(examples)
    if not examples:
        rep.errors.append("no examples produced")
        return rep

    # ---------------------------------------------------------- structure
    for i, ex in enumerate(examples):
        roles = [m["role"] for m in ex.messages]
        if roles[:1] != ["system"] or roles[-1] != "assistant" or len(roles) < 3:
            rep.errors.append(f"example[{i}] ({ex.requirement}) has roles {roles}")
        if not ex.messages[-1]["content"].strip():
            rep.errors.append(f"example[{i}] ({ex.requirement}) has an empty assistant turn")
        for m in ex.messages:
            if "{{" in m["content"]:
                rep.errors.append(f"example[{i}] has an unrendered placeholder")
                break

    # ---------------------------------------------------------- leakage
    probes = spec.probe_texts()
    leaked = [ex for ex in examples if ex.question() in probes]
    rep.stats["probe_leakage"] = len(leaked)
    if leaked:
        rep.errors.append(f"{len(leaked)} training example(s) reuse a probe question — "
                          f"e.g. {leaked[0].question()!r}")

    # ---------------------------------------------------------- duplication
    keys = Counter(ex.key() for ex in examples)
    dupes = sum(c - 1 for c in keys.values() if c > 1)
    rep.stats["exact_duplicates"] = dupes
    if dupes > len(examples) * 0.1:
        rep.errors.append(f"{dupes} duplicate pairs ({dupes / len(examples):.0%} of the set)")
    elif dupes:
        rep.warnings.append(f"{dupes} duplicate pairs — deduplicated on write")

    # ---------------------------------------------------------- balance
    per_req = Counter(ex.requirement for ex in examples)
    rep.stats["per_requirement"] = dict(per_req)
    top, n = per_req.most_common(1)[0]
    if n / len(examples) > MAX_SHARE:
        rep.warnings.append(f"{top} owns {n / len(examples):.0%} of the set "
                            f"(> {MAX_SHARE:.0%}) — the tune will bias toward it")
    # A requirement deliberately left out (--skip-passing, because the baseline
    # already meets it) is not the same as one whose recipe produced nothing.
    skipped = skipped or set()
    missing = [r.id for r in spec.requirements
               if r.synth.get("recipe") and r.id not in skipped and not per_req.get(r.id)]
    if missing:
        rep.errors.append(f"requirements with a recipe but zero examples: {missing}")
    if skipped:
        rep.stats["skipped_at_baseline"] = sorted(skipped)

    refusals = sum(1 for ex in examples if ex.recipe == "absent_entity")
    rep.stats["refusal_share"] = round(refusals / len(examples), 2)
    if refusals / len(examples) > MAX_REFUSAL_SHARE:
        rep.warnings.append(f"refusals are {refusals / len(examples):.0%} of the set — "
                            f"above {MAX_REFUSAL_SHARE:.0%} models start refusing answerable questions")

    if len(examples) < MIN_EXAMPLES:
        rep.warnings.append(f"only {len(examples)} examples; LoRA on a 1.7B base wants "
                            f"300-2000 for a stable behavioural shift")

    # ---------------------------------------------------------- length
    max_len = int(spec.train_cfg()["max_len"])
    lengths = [sum(len(m["content"]) for m in ex.messages) / CHARS_PER_TOKEN for ex in examples]
    over = [i for i, t in enumerate(lengths) if t > max_len]
    rep.stats["tokens_p50"] = int(sorted(lengths)[len(lengths) // 2])
    rep.stats["tokens_max"] = int(max(lengths))
    if over:
        rep.errors.append(f"{len(over)} example(s) exceed max_len={max_len} tokens and would be "
                          f"truncated mid-answer (longest ~{int(max(lengths))})")

    # ---------------------------------------------------------- privacy
    if check_privacy:
        terms = _real_kb_terms(spec.agent)
        blob = " ".join(m["content"] for ex in examples for m in ex.messages[1:])
        hits = sorted({t for t in terms if re.search(rf"\b{re.escape(t)}\b", blob)})
        rep.stats["real_kb_terms_checked"] = len(terms)
        if hits:
            rep.errors.append(
                f"real knowledge-base content reached the training set: {hits[:8]} — "
                f"facts belong in kb.sqlite, not in distributed weights")
    return rep


def _tail(user_text: str) -> str:
    """The question, with any retrieved-context preamble stripped."""
    return user_text.rsplit("---\n\n", 1)[-1].strip()


# ------------------------------------------------------------------ writing

def split_and_write(spec: Spec, examples: list[Example], out_dir: Path,
                    val_fraction: float = 0.12, seed: int = 7) -> dict:
    """Deduplicate, split stratified by requirement, write JSONL."""
    import json
    import random

    rng = random.Random(seed)
    seen: set[str] = set()
    unique: list[Example] = []
    for ex in examples:
        k = ex.key()
        if k in seen:
            continue
        seen.add(k)
        unique.append(ex)

    by_req: dict[str, list[Example]] = {}
    for ex in unique:
        by_req.setdefault(ex.requirement, []).append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for rid, group in by_req.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (("train", train), ("val", val)):
        p = out_dir / f"{spec.adapter_id}-{name}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps({"messages": ex.messages,
                                     "requirement": ex.requirement,
                                     "recipe": ex.recipe}, ensure_ascii=False) + "\n")
        paths[name] = p
    return {"train": paths["train"], "val": paths["val"],
            "n_train": len(train), "n_val": len(val), "n_dropped": len(examples) - len(unique)}
