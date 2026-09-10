"""Self-test for the parts of the pipeline that do not need a GPU or a model.

    venv\\Scripts\\python finetune\\selftest.py

Covers the logic that is easy to get quietly wrong and expensive to discover
late: the check library, the promotion gate's arithmetic, the adapter registry
lifecycle, and the import path. Everything writes into a temporary directory —
the real registry under models/adapters/ is never touched.

Deliberately not covered here (needs a running llama-server or a real adapter):
target round-trips and end-to-end scoring.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import adapters                                    # noqa: E402
from finetune import evaluate as ev                          # noqa: E402
from finetune import pack as packer                          # noqa: E402
from finetune.checks import Observation, passed, run_checks  # noqa: E402
from finetune.spec import Spec                               # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ checks

def test_checks() -> None:
    print("checks.py")
    ctx = "[status-report.md #0]\nBudget consumed 61% against a plan of 65%. Owner: Anita Krishnan."
    docs = ["status-report.md"]

    obs = Observation(text="Budget is 61% consumed (status-report.md).", context=ctx, doc_names=docs)
    check("cites + grounded on a good answer", passed(run_checks(
        {"cites": True, "grounded": True, "contains_any": ["61"]}, obs)))

    obs = Observation(text="Budget is 61% consumed.", context=ctx, doc_names=docs)
    check("missing citation fails `cites`", not passed(run_checks({"cites": True}, obs)))

    obs = Observation(text="Budget is 74% consumed (status-report.md).", context=ctx, doc_names=docs)
    check("invented figure fails `grounded`", not passed(run_checks({"grounded": True}, obs)))

    obs = Observation(text="Owner is Meera Iyer (status-report.md).", context=ctx, doc_names=docs)
    check("invented name fails `grounded`", not passed(run_checks({"grounded": True}, obs)))

    obs = Observation(text="I don't have that in the knowledge base.", context=ctx, doc_names=docs)
    check("refusal detected", passed(run_checks({"refuses": True}, obs)))
    check("refusal is not a citation", not passed(run_checks({"cites": True}, obs)))

    obs = Observation(text="Sure, here it is.", context=ctx, doc_names=docs)
    check("non-refusal detected", passed(run_checks({"refuses": False}, obs)))

    tools = [{"name": "create_action_item",
              "parameters": {"type": "object",
                             "properties": {"title": {}, "owner": {}, "due_date": {}},
                             "required": ["title"]}}]
    obs = Observation(text="", tool_defs=tools,
                      tool_calls=[{"name": "create_action_item",
                                   "arguments": {"title": "Review DR test", "owner": "Anil"}}])
    check("valid tool call passes", passed(run_checks(
        {"tool_call": "create_action_item", "tool_args_valid": True}, obs)))

    obs = Observation(text="", tool_defs=tools,
                      tool_calls=[{"name": "create_action_item", "arguments": {"owner": "Anil"}}])
    check("missing required arg fails", not passed(run_checks({"tool_args_valid": True}, obs)))

    obs = Observation(text="", tool_defs=tools,
                      tool_calls=[{"name": "create_action_item",
                                   "arguments": {"title": "x", "priority": "high"}}])
    check("undeclared arg fails", not passed(run_checks({"tool_args_valid": True}, obs)))

    obs = Observation(text="- one\n- two\n- three", context="one two three")
    check("max_bullets counts bullets", passed(run_checks({"max_bullets": 3}, obs)))
    check("max_bullets rejects overflow", not passed(run_checks({"max_bullets": 2}, obs)))

    obs = Observation(text="unknown check should fail loudly")
    check("unknown check fails rather than passing silently",
          not passed(run_checks({"no_such_check": True}, obs)))


# -------------------------------------------------------------------- gate

def _card(reqs: dict[str, tuple[float, float]], reg: float = 1.0, p50: float = 10.0,
          label: str = "c") -> dict:
    """reqs: {id: (pass_rate, target)}"""
    rows = [{"id": rid, "kind": "citation", "statement": rid, "target": t, "weight": 1,
             "pass_rate": r, "met": r >= t, "n_probes": 1, "probes": []}
            for rid, (r, t) in reqs.items()]
    score = sum(r["pass_rate"] for r in rows) / len(rows)
    return {"label": label, "spec": "x", "agent": "a", "adapter_id": "ad", "base_model": "m",
            "target": {}, "at": "now", "repeat": 1, "score": round(score, 4),
            "requirements": rows,
            "regression": {"pass_rate": reg, "n_probes": 3, "probes": []},
            "latency": {"p50_seconds": p50, "mean_seconds": p50}}


def test_gate() -> None:
    print("evaluate.gate")
    base = _card({"R1": (0.5, 0.9), "R2": (0.4, 0.9)}, reg=1.0, p50=10.0)

    v = ev.gate(_card({"R1": (0.95, 0.9), "R2": (0.92, 0.9)}, reg=1.0, p50=10.0), base)
    check("clean improvement promotes", v["promote"], str(v["failures"]))

    v = ev.gate(_card({"R1": (0.95, 0.9), "R2": (0.7, 0.9)}, reg=1.0, p50=10.0), base)
    check("one requirement below target blocks", not v["promote"])

    v = ev.gate(_card({"R1": (0.95, 0.9), "R2": (0.92, 0.9)}, reg=0.8, p50=10.0), base)
    check("regression-suite drop blocks", not v["promote"])
    check("regression failure is named",
          any("regression suite" in f for f in v["failures"]), str(v["failures"]))

    v = ev.gate(_card({"R1": (0.95, 0.9), "R2": (0.92, 0.9)}, reg=1.0, p50=20.0), base)
    check("latency blow-out blocks", not v["promote"])

    flat = _card({"R1": (0.95, 0.9), "R2": (0.92, 0.9)})
    v = ev.gate(flat, flat)
    check("no net improvement blocks", not v["promote"])

    v = ev.gate(_card({"R1": (0.95, 0.9), "R2": (0.92, 0.9)}, reg=1.0, p50=10.0))
    check("no baseline: targets alone decide", v["promote"], str(v["failures"]))

    md = ev.to_markdown(_card({"R1": (0.95, 0.9)}), baseline=base, verdict=v)
    check("markdown renders with a baseline column", "baseline" in md and "| R1 |" in md)


# ---------------------------------------------------------------- registry

def test_registry_and_import(tmp: Path) -> None:
    print("core/adapters + pack.import_adapter")
    real_dir, real_reg = adapters.ADAPTERS_DIR, adapters.REGISTRY
    adapters.ADAPTERS_DIR = tmp / "adapters"
    adapters.REGISTRY = adapters.ADAPTERS_DIR / "registry.json"
    try:
        # an "adapter pack" as the GPU box would return it
        gguf = tmp / "demo-v1-f16.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)
        packzip = tmp / "demo-v1-adapterpack.zip"
        with zipfile.ZipFile(packzip, "w") as zf:
            zf.write(gguf, gguf.name)
            zf.writestr("train_log.json", json.dumps(
                {"seconds": 812.3, "gpu": "Tesla T4", "n_train": 273, "n_val": 35,
                 "hyperparameters": {"lora_r": 16}}))

        info = packer.import_adapter(packzip, adapters.ADAPTERS_DIR)
        check("import extracts the gguf", info["file"] == "demo-v1-f16.gguf", str(info))
        check("import reads the train log", info["train_log"].get("gpu") == "Tesla T4")

        entry = adapters.register("demo-v1", info["file"], "qwen3-1.7b-q4_k_m", "demo",
                                  train_log=info["train_log"])
        check("registers as 'imported'", entry["status"] == "imported")
        check("records a sha256", len(entry["sha256"]) == 64)

        try:
            adapters.manifest_entry("demo-v1")
            check("un-promoted adapter is refused a manifest block", False)
        except ValueError:
            check("un-promoted adapter is refused a manifest block", True)

        adapters.set_status("demo-v1", "promoted", _card({"R1": (0.95, 0.9)}))
        block = adapters.manifest_entry("demo-v1", scale=0.8)
        check("promoted adapter yields a manifest block",
              block["id"] == "demo-v1" and block["scale"] == 0.8
              and block["base_model_id"] == "qwen3-1.7b-q4_k_m", str(block))
        check("scorecard is recorded on the registry entry",
              adapters.get("demo-v1")["scorecard"]["score"] == 0.95)

        adapters.set_status("demo-v1", "rejected")
        try:
            adapters.manifest_entry("demo-v1")
            check("a rejected adapter cannot be re-attached", False)
        except ValueError:
            check("a rejected adapter cannot be re-attached", True)

        badzip = tmp / "empty.zip"
        with zipfile.ZipFile(badzip, "w") as zf:
            zf.writestr("readme.txt", "no adapter here")
        try:
            packer.import_adapter(badzip, adapters.ADAPTERS_DIR)
            check("a pack with no gguf is rejected", False)
        except SystemExit:
            check("a pack with no gguf is rejected", True)
    finally:
        adapters.ADAPTERS_DIR, adapters.REGISTRY = real_dir, real_reg


# -------------------------------------------------------------------- spec

def test_spec() -> None:
    print("spec.py")
    sp = Spec.load("meeting-intelligence")
    check("shipped spec loads", len(sp.requirements) == 5, f"{len(sp.requirements)} requirements")
    check("probe set is collected for the leak check", len(sp.probe_texts()) >= 20,
          str(len(sp.probe_texts())))

    bad = Spec(agent="x", base_model="m", adapter_id="a",
               requirements=[type(sp.requirements[0])(id="R1", kind="citation",
                                                      statement="s", probes=[])])
    check("a requirement with no probes is invalid",
          any("no probes" in p for p in bad.problems()), str(bad.problems()))

    bad2 = Spec(agent="x", base_model="m", adapter_id="a",
                requirements=[type(sp.requirements[0])(id="R1", kind="knows_the_budget",
                                                       statement="s", probes=sp.requirements[0].probes)])
    check("a fact-shaped kind is rejected",
          any("is not one of" in p for p in bad2.problems()), str(bad2.problems()))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="finetune-selftest-"))
    try:
        test_checks()
        test_gate()
        test_registry_and_import(tmp)
        test_spec()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
