"""Build a self-contained training job pack for a GPU machine.

The prototype server has 4 vCPU, no GPU and 4 GB of free disk — it cannot take
a gradient step and cannot even hold a PyTorch install. Rather than pretend
otherwise, the pipeline treats the gradient step as the one stage that runs
elsewhere and makes "elsewhere" interchangeable: a Colab T4, the Ubuntu box,
or a spot g4dn. Everything the run needs travels in one zip, and the only
thing that comes back is an adapter pack.

That boundary is also the compliance boundary. What leaves this machine is
surrogate text and hyperparameters — no customer document, no chat log, no
knowledge base. validate.py enforces that before pack.py is allowed to run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from core.catalog import get_model

from .spec import Spec

HERE = Path(__file__).resolve().parent
PACK_DIR = HERE / "jobs"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def job_config(spec: Spec, train: Path, val: Path, dataset_report: dict) -> dict:
    model = get_model(spec.base_model)
    if not model:
        raise SystemExit(f"unknown base_model {spec.base_model!r} — add it to core/catalog.py")
    hf_repo = model.get("hf_repo")
    if not hf_repo:
        raise SystemExit(
            f"catalog entry {spec.base_model!r} has no hf_repo, so there is nothing to train. "
            f"Add the upstream Hugging Face repo id to core/catalog.py.")

    cfg = spec.train_cfg()
    return {
        "schema": "slm-finetune-job/1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapter_id": spec.adapter_id,
        "agent": spec.agent,
        "base_model_catalog_id": spec.base_model,
        "base_hf_repo": hf_repo,
        "base_gguf": model["file"],
        "spec_sha256": _digest(spec.path) if spec.path else None,
        "train_file": train.name,
        "val_file": val.name,
        "dataset": dataset_report,
        "hyperparameters": cfg,
        "requirements": [{"id": r.id, "kind": r.kind, "statement": r.statement,
                          "target": r.target} for r in spec.requirements],
    }


RUN_SH = """#!/usr/bin/env bash
# One-shot runner for a GPU box. Tested shape: Ubuntu + CUDA 12.x, >= 8 GB VRAM.
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-finetune.txt

python train_qlora.py --job job.json --out out/adapter

# LoRA -> GGUF, so the adapter can be applied by llama.cpp and llama.rn without
# merging or requantising the 1 GB base model.
if [ ! -d llama.cpp ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp; fi
pip install -r llama.cpp/requirements/requirements-convert_lora_to_gguf.txt
ADAPTER_ID=$(python -c "import json;print(json.load(open('job.json'))['adapter_id'])")
BASE=$(python -c "import json;print(json.load(open('job.json'))['base_hf_repo'])")
python llama.cpp/convert_lora_to_gguf.py out/adapter \\
    --base "$BASE" --outfile "out/${ADAPTER_ID}-f16.gguf"

cd out && zip -r "../${ADAPTER_ID}-adapterpack.zip" \\
    adapter "${ADAPTER_ID}-f16.gguf" train_log.json
echo
echo "Done. Copy ${ADAPTER_ID}-adapterpack.zip back and run:"
echo "  venv\\\\Scripts\\\\python -m finetune import --pack <path to the zip>"
"""

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Scenario fine-tune — Colab T4 runner\n",
            "\n",
            "Upload the job pack zip, then run every cell. Runtime > Change runtime type > T4 GPU.\n",
            "A few-hundred-example LoRA on a 1.7B base takes well under an hour.\n",
        ]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "from google.colab import files\n",
            "up = files.upload()          # select <adapter_id>-jobpack.zip\n",
            "import zipfile, pathlib\n",
            "name = next(iter(up))\n",
            "zipfile.ZipFile(name).extractall('job')\n",
            "%cd job\n",
        ]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "!nvidia-smi\n",
            "!pip -q install -r requirements-finetune.txt\n",
        ]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "!python train_qlora.py --job job.json --out out/adapter\n",
        ]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "import json\n",
            "job = json.load(open('job.json'))\n",
            "!git clone --depth 1 https://github.com/ggml-org/llama.cpp\n",
            "!pip -q install -r llama.cpp/requirements/requirements-convert_lora_to_gguf.txt\n",
            "!python llama.cpp/convert_lora_to_gguf.py out/adapter --base {job['base_hf_repo']} "
            "--outfile out/{job['adapter_id']}-f16.gguf\n",
        ]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "import json, shutil\n",
            "job = json.load(open('job.json'))\n",
            "shutil.make_archive(job['adapter_id'] + '-adapterpack', 'zip', 'out')\n",
            "from google.colab import files\n",
            "files.download(job['adapter_id'] + '-adapterpack.zip')\n",
        ]},
    ],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 5,
}

PACK_README = """# Training job pack — {adapter_id}

Self-contained. Nothing in here is customer data: contexts are rendered from
templated surrogate documents with a randomised entity cast, which is why this
zip is safe to upload to a third-party GPU.

## Run it

Linux + NVIDIA (>= 8 GB VRAM):

    bash run.sh

Colab T4: upload this zip, open `colab.ipynb`, run all.

Either way you end up with `{adapter_id}-adapterpack.zip`. Copy it back to the
Studio machine and run:

    venv\\Scripts\\python -m finetune import --pack {adapter_id}-adapterpack.zip
    venv\\Scripts\\python -m finetune evaluate --spec {agent} --adapter {adapter_id}

The second command is the gate. The adapter is not publishable until it passes.

## What is being trained

{requirements}

## Cost

QLoRA, 4-bit base, r={lora_r}, {epochs} epochs over {n_train} examples at
max_len {max_len}. Roughly 6-8 GB VRAM; a g4dn.xlarge spot instance is about
$0.16/h and this run does not fill an hour.
"""


def build(spec: Spec, train: Path, val: Path, dataset_report: dict,
          out_dir: Path = PACK_DIR) -> Path:
    cfg = job_config(spec, train, val, dataset_report)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{spec.adapter_id}-jobpack.zip"

    reqs = "\n".join(f"- **{r['id']}** ({r['kind']}) — {r['statement']} "
                     f"_target {r['target']:.0%}_" for r in cfg["requirements"])
    readme = PACK_README.format(
        adapter_id=spec.adapter_id, agent=spec.agent, requirements=reqs,
        lora_r=cfg["hyperparameters"]["lora_r"], epochs=cfg["hyperparameters"]["epochs"],
        max_len=cfg["hyperparameters"]["max_len"], n_train=dataset_report.get("n_train", "?"))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("job.json", json.dumps(cfg, indent=2))
        zf.writestr("README.md", readme)
        zf.writestr("run.sh", RUN_SH)
        zf.writestr("colab.ipynb", json.dumps(NOTEBOOK, indent=1))
        zf.write(HERE / "train_qlora.py", "train_qlora.py")
        zf.write(HERE / "requirements-finetune.txt", "requirements-finetune.txt")
        zf.write(train, train.name)
        zf.write(val, val.name)
        if spec.path:
            zf.write(spec.path, "spec.yaml")
    return zip_path


# ------------------------------------------------------------------ import

def import_adapter(pack: Path, adapters_dir: Path) -> dict:
    """Unpack what came back from the GPU box and return its descriptor."""
    adapters_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pack) as zf:
        names = zf.namelist()
        gguf = next((n for n in names if n.endswith(".gguf")), None)
        if not gguf:
            raise SystemExit(f"{pack} contains no .gguf adapter — did convert_lora_to_gguf run?")
        adapter_id = Path(gguf).name.replace("-f16.gguf", "").replace(".gguf", "")
        target = adapters_dir / Path(gguf).name
        with zf.open(gguf) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        log = {}
        if "train_log.json" in names:
            log = json.loads(zf.read("train_log.json"))
    return {"adapter_id": adapter_id, "file": target.name,
            "size_bytes": target.stat().st_size, "train_log": log, "path": target}
