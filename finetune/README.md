# Fine-tuning pipeline — specialize the SLM for a business scenario

The Studio prototype uses a stock Qwen3-1.7B with a system prompt + RAG. That
already works. Fine-tuning is the *enhancement* that fixes what small models
get wrong: citation discipline, refusing when context lacks the answer,
domain phrasing (MOM formats, RAID logs), and reliable tool selection.

**Hardware reality check:** this Windows server (t3.xlarge) has no GPU and
cannot train. QLoRA on Qwen3-1.7B needs ~8 GB VRAM (any T4/A10/RTX 3060+).
Options, in order of convenience:

1. The Ubuntu box over SSH — if `nvidia-smi` shows a GPU there, copy this
   `finetune/` folder over and run everything on it.
2. Google Colab free T4 — upload the dataset JSONL, run the same scripts.
3. Any cloud GPU instance (g4dn.xlarge spot ≈ $0.16/h; a 3-epoch run on a
   few hundred examples takes well under an hour).

A Hugging Face token is only needed if you pick a gated base model
(Llama/Gemma); Qwen3 is ungated Apache-2.0.

## Workflow

### 1. Build the dataset (runs on the Windows server — no GPU needed)

```bash
venv\Scripts\python finetune\dataset_builder.py --agent meeting-intelligence
```

Combines behaviour seeds + heuristic KB-grounded pairs. For production
quality add `--teacher <openai-compatible-url>` to distill Q/A pairs from a
larger model, and mix in real MOMs/transcripts. Target 300–2000 examples.

### 2. Train QLoRA (GPU machine)

```bash
pip install -r requirements-finetune.txt
python train_qlora.py \
  --base Qwen/Qwen3-1.7B \
  --train datasets/meeting-intelligence-train.jsonl \
  --val   datasets/meeting-intelligence-val.jsonl \
  --out   out/meeting-intelligence-lora
```

~30 MB of LoRA weights; 4-bit base keeps VRAM ≈ 6–8 GB at max_len 2048.

### 3. Merge and convert to a phone-ready GGUF (GPU machine or any Linux box)

```bash
python merge_and_export.py --base Qwen/Qwen3-1.7B \
  --adapter out/meeting-intelligence-lora \
  --out out/meeting-intelligence-merged

git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py out/meeting-intelligence-merged \
  --outfile meeting-intelligence-f16.gguf
llama.cpp/build/bin/llama-quantize meeting-intelligence-f16.gguf \
  meeting-intelligence-Q4_K_M.gguf Q4_K_M
```

### 4. Deploy the tuned model into the platform

1. Copy `meeting-intelligence-Q4_K_M.gguf` to `C:\slm\models\`.
2. Add an entry to `core/catalog.py` (id, file, size, notes) — it appears in
   the Studio model picker immediately.
3. In the Studio, switch the agent's model to the tuned entry and re-publish.
   Devices see the new version in the store and update.

## Evaluation before shipping

Keep a held-out set of 20–50 real questions. For each candidate model run
them through the Runtime chat API and check: answer grounded? source cited?
refusal when KB lacks the answer? correct tool JSON? A simple pass-rate
comparison against the stock model tells you if the tune earned its place.
