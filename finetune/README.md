# finetune/ — requirement-driven scenario fine-tuning

Full design, decisions and status: **[`docs/finetuning.md`](../docs/finetuning.md)**.
This file is the map of the directory.

The short version: a fine-tune here starts from a written requirement, not from
a pile of documents. One YAML states what the agent must do, generates the
training data for it, grades the model on it, and decides whether the result is
allowed to ship. Facts stay in the knowledge base; only behaviour goes into
weights.

```
spec.py        requirement spec: schema, loader, scaffolder
checks.py      deterministic assertions — grade probes AND screen generated data
targets.py     ask an agent a question (deployed runtime, or llama-server alone)
evaluate.py    scorecards, baseline/candidate diff, the promotion gate
synth.py       recipes that turn requirements into examples (rejection sampled)
validate.py    the dataset gate: leakage, duplication, balance, length, privacy
pack.py        build a portable GPU job pack / import the adapter that returns
train_qlora.py the one file that imports torch — runs on the GPU box, not here
specs/         requirement specs, one per agent
surrogates/    templated stand-in documents + entity pools (never real content)
datasets/      generated JSONL (gitignored) + its provenance json
reports/       scorecards, json + markdown (gitignored: contains real answers)
jobs/          built job packs (gitignored)
```

## Commands

```powershell
venv\Scripts\python -m finetune scaffold --agent <id>     # draft a spec
venv\Scripts\python -m finetune baseline --spec <name>    # what needs training?
venv\Scripts\python -m finetune synth    --spec <name>    # make + validate data
venv\Scripts\python -m finetune pack     --spec <name>    # zip for a GPU box
venv\Scripts\python -m finetune import   --spec <name> --pack <zip>
venv\Scripts\python -m finetune evaluate --spec <name>    # A/B + gate
venv\Scripts\python -m finetune adapters                  # registry + verdicts
```

## Hardware

This server has no GPU and cannot train — `train_qlora.py` refuses to start
without CUDA and points at the job pack. Everything else runs here. The job
pack contains `run.sh` (Ubuntu + NVIDIA) and `colab.ipynb` (free T4); a
few-hundred-example QLoRA on a 1.7B base needs ~6-8 GB VRAM and well under an
hour.
