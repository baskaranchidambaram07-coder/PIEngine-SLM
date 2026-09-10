"""`python -m finetune <stage>` — the whole loop from one entry point.

    scaffold   draft a requirement spec from an existing Studio agent
    baseline   score the stock model against the spec  -> what needs training
    synth      generate + validate + write the dataset -> what to train on
    pack       build the job pack for a GPU box        -> the only off-box step
    import     register the adapter that came back
    evaluate   score the adapter, diff against baseline, apply the gate
    adapters   list what has been imported and what its verdict was
    report     re-render any saved scorecard as markdown

Stages are separate commands on purpose. Each one writes an artifact the next
one reads, so a run can be paused for a week between the pack and the import
without losing the thread.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core import adapters as adapter_registry

from . import evaluate as ev
from . import pack as packer
from . import spec as specmod
from . import synth as synthmod
from . import targets, validate

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "datasets"


# ------------------------------------------------------------------ stages

def cmd_scaffold(args) -> int:
    text = specmod.scaffold(args.agent)
    out = Path(args.out) if args.out else specmod.SPEC_DIR / f"{args.agent}.yaml"
    if out.exists() and not args.force:
        print(f"{out} exists — pass --force to overwrite")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}\n\nEdit the <placeholders>, then:\n"
          f"  python -m finetune baseline --spec {out.stem}")
    return 0


def _score(args, label: str, adapter_scale: float | None) -> dict:
    sp = specmod.Spec.load(args.spec)
    tgt = targets.build(sp.agent, args.target, adapter_scale=adapter_scale)
    print(f"scoring {sp.agent} via {args.target} ({tgt.describe()})")
    print(f"  {len(sp.requirements)} requirements, "
          f"{sum(len(r.probes) for r in sp.requirements)} probes, "
          f"{len(sp.regression)} regression probes, {args.repeat} sample(s) each")
    card = ev.score(sp, tgt, repeat=args.repeat, label=label)
    return card


def cmd_baseline(args) -> int:
    sp = specmod.Spec.load(args.spec)
    card = _score(args, "baseline", adapter_scale=0.0 if args.target == "llama" and args.has_adapter else None)
    name = f"{sp.adapter_id}-baseline"
    path = ev.save(card, name)
    md = ev.to_markdown(card)
    (ev.REPORT_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"saved {path}")

    met = [r["id"] for r in card["requirements"] if r["met"]]
    if met:
        print(f"\nAlready at target without training: {met}")
        print("Delete these from the spec, or accept that `synth --skip-passing` will "
              "leave them out of the dataset. Adapter capacity spent on solved behaviour "
              "is how tunes regress.")
    return 0


def cmd_synth(args) -> int:
    sp = specmod.Spec.load(args.spec)

    skip: set[str] = set()
    if args.skip_passing:
        try:
            base = ev.load(f"{sp.adapter_id}-baseline")
            skip = {r["id"] for r in base["requirements"] if r["met"]}
        except SystemExit:
            print("no baseline scorecard yet — run `baseline` first to skip solved "
                  "requirements. Generating everything.")

    teacher = synthmod.Teacher(None if args.no_teacher else args.teacher, args.teacher_model)
    if teacher.enabled:
        print(f"teacher: {teacher.url} ({args.teacher_model})")
    else:
        print("teacher: none — extractive fallbacks only, lower variety")

    examples, report = synthmod.build(sp, teacher, seed=args.seed, skip=skip)
    print("\ngeneration")
    for rid, info in report["per_requirement"].items():
        if info.get("reason"):
            print(f"  {rid}: skipped ({info['reason']})")
        else:
            dropped = []
            if info.get("dropped_probe_collision"):
                dropped.append(f"{info['dropped_probe_collision']} held-out")
            if info.get("dropped_duplicate"):
                dropped.append(f"{info['dropped_duplicate']} duplicate")
            tail = f", dropped {' + '.join(dropped)}" if dropped else ""
            print(f"  {rid}: {info['produced']}/{info['requested']} via {info['recipe']} "
                  f"(yield {info['yield']:.0%}{tail})")
    if teacher.enabled:
        print(f"  teacher calls {teacher.calls}, unusable {teacher.failures}")

    rep = validate.validate(sp, examples, check_privacy=not args.no_privacy_check,
                            skipped=skip)
    print()
    print(rep.render())
    if not rep.ok:
        print("\ndataset BLOCKED — nothing written.")
        return 1

    written = validate.split_and_write(sp, examples, DATASET_DIR, seed=args.seed)
    meta = {"spec": str(sp.path), "adapter_id": sp.adapter_id, "agent": sp.agent,
            "generation": report, "validation": rep.stats,
            "n_train": written["n_train"], "n_val": written["n_val"],
            "n_dropped_duplicates": written["n_dropped"]}
    (DATASET_DIR / f"{sp.adapter_id}-dataset.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {written['n_train']} train / {written['n_val']} val "
          f"({written['n_dropped']} duplicates dropped)")
    print(f"  {written['train']}\n  {written['val']}")
    print(f"\nnext: python -m finetune pack --spec {Path(str(sp.path)).stem}")
    return 0


def cmd_pack(args) -> int:
    sp = specmod.Spec.load(args.spec)
    train = DATASET_DIR / f"{sp.adapter_id}-train.jsonl"
    val = DATASET_DIR / f"{sp.adapter_id}-val.jsonl"
    meta_path = DATASET_DIR / f"{sp.adapter_id}-dataset.json"
    if not train.exists():
        raise SystemExit(f"no dataset for {sp.adapter_id} — run `synth` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    zip_path = packer.build(sp, train, val, meta)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"job pack: {zip_path} ({size_mb:.1f} MB)")
    print("\nThis box has no GPU. Run the pack on one:")
    print("  Colab T4 : upload the zip, open colab.ipynb, run all")
    print("  Linux GPU: unzip && bash run.sh")
    print("\nThen bring back <adapter_id>-adapterpack.zip and run:")
    print(f"  python -m finetune import --pack <zip> --spec {Path(str(sp.path)).stem}")
    return 0


def cmd_import(args) -> int:
    sp = specmod.Spec.load(args.spec)
    info = packer.import_adapter(Path(args.pack), adapter_registry.ADAPTERS_DIR)
    entry = adapter_registry.register(
        adapter_id=sp.adapter_id, file=info["file"], base_model_id=sp.base_model,
        agent=sp.agent, spec_sha256=packer._digest(sp.path) if sp.path else None,
        train_log=info.get("train_log"))
    print(f"imported {entry['id']} -> {adapter_registry.ADAPTERS_DIR / entry['file']} "
          f"({entry['size_bytes'] / 1e6:.1f} MB, sha256 {entry['sha256'][:16]}...)")
    print(f"status: {entry['status']} — not publishable until it passes the gate.")
    print("\nStart llama-server with the adapter loaded but unapplied, so baseline and")
    print("candidate run in one process:")
    print(f"  llama\\llama-server.exe -m models\\{_base_file(sp)} "
          f"--lora models\\adapters\\{entry['file']} --lora-init-without-apply "
          f"--port 8302 -c 4096 --jinja --no-webui")
    print(f"\nThen: python -m finetune evaluate --spec {Path(str(sp.path)).stem} "
          f"--adapter {entry['id']}")
    return 0


def _base_file(sp) -> str:
    from core.catalog import get_model
    m = get_model(sp.base_model)
    return m["file"] if m else "<base>.gguf"


def cmd_evaluate(args) -> int:
    sp = specmod.Spec.load(args.spec)
    adapter_id = args.adapter or sp.adapter_id
    entry = adapter_registry.get(adapter_id)
    if not entry:
        raise SystemExit(f"adapter {adapter_id!r} not imported — run `import` first")

    args.target = "llama"     # adapter A/B is only meaningful against the weights
    args.has_adapter = True
    base_card = _score(args, "baseline (adapter off)", adapter_scale=0.0)
    cand_card = _score(args, f"candidate {adapter_id}", adapter_scale=args.scale)

    verdict = ev.gate(cand_card, base_card)
    ev.save(base_card, f"{sp.adapter_id}-baseline-ab")
    path = ev.save(cand_card, f"{sp.adapter_id}-candidate")
    md = ev.to_markdown(cand_card, baseline=base_card, verdict=verdict)
    (ev.REPORT_DIR / f"{sp.adapter_id}-candidate.md").write_text(md, encoding="utf-8")
    print()
    print(md)

    adapter_registry.set_status(adapter_id, "promoted" if verdict["promote"] else "rejected",
                                cand_card)
    print(f"saved {path}")
    print(f"adapter {adapter_id} -> {'PROMOTED' if verdict['promote'] else 'REJECTED'}")
    if verdict["promote"]:
        print("\nAttach it to the agent and republish:")
        print(f"  PUT /api/agents/{sp.agent}  with  \"adapter_id\": \"{adapter_id}\"")
        print(f"  POST /api/agents/{sp.agent}/publish")
    return 0 if verdict["promote"] else 2


def cmd_adapters(args) -> int:
    rows = adapter_registry.load()
    if not rows:
        print("no adapters imported yet")
        return 0
    for a in rows:
        sc = a.get("scorecard") or {}
        score = f"{sc['score']:.0%}" if sc.get("score") is not None else "—"
        print(f"{a['id']:<34} {a['status']:<9} {a['size_bytes'] / 1e6:6.1f} MB  "
              f"base={a['base_model_id']:<28} score={score}")
    return 0


def cmd_report(args) -> int:
    card = ev.load(args.card)
    base = ev.load(args.baseline) if args.baseline else None
    print(ev.to_markdown(card, baseline=base,
                         verdict=ev.gate(card, base) if base else None))
    return 0


# ------------------------------------------------------------------ argparse

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m finetune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scaffold", help="draft a requirement spec from a Studio agent")
    p.add_argument("--agent", required=True)
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_scaffold)

    p = sub.add_parser("baseline", help="score the stock model against the spec")
    p.add_argument("--spec", required=True)
    p.add_argument("--target", choices=("runtime", "llama"), default="runtime")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--has-adapter", action="store_true",
                   help="llama-server already has an adapter loaded; score with scale 0")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("synth", help="generate, validate and write the dataset")
    p.add_argument("--spec", required=True)
    p.add_argument("--teacher", default=synthmod.DEFAULT_TEACHER)
    p.add_argument("--teacher-model", default="teacher")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--skip-passing", action="store_true",
                   help="omit requirements the baseline already meets")
    p.add_argument("--no-privacy-check", action="store_true",
                   help="skip the real-KB-content check (do not use for a shipping tune)")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("pack", help="build the GPU job pack")
    p.add_argument("--spec", required=True)
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("import", help="register an adapter pack from the GPU box")
    p.add_argument("--pack", required=True)
    p.add_argument("--spec", required=True)
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("evaluate", help="A/B the adapter against stock weights and gate it")
    p.add_argument("--spec", required=True)
    p.add_argument("--adapter")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--repeat", type=int, default=1)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("adapters", help="list imported adapters")
    p.set_defaults(fn=cmd_adapters)

    p = sub.add_parser("report", help="re-render a saved scorecard")
    p.add_argument("card")
    p.add_argument("--baseline")
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "has_adapter"):
        args.has_adapter = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
