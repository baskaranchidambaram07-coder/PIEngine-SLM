"""QLoRA training step — the one stage that needs a GPU.

Runs from a job pack produced by `python -m finetune pack`, so nothing has to
be re-specified on the GPU box:

    python train_qlora.py --job job.json --out out/adapter

Everything else in the pipeline (spec, synthesis, dataset validation,
evaluation, gating, publishing) runs on the CPU Studio machine. This file is
deliberately the only one that imports torch.

Hardware: any NVIDIA card with >= 8 GB VRAM. A 1.7B base in 4-bit with r=16
LoRA at max_len 2048 sits around 6-8 GB. Colab's free T4 is enough.

Writes `train_log.json` next to the adapter — loss curve, sample counts and
the resolved config — so the eval scorecard can be traced back to the run that
produced it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="job.json", help="job config from the job pack")
    ap.add_argument("--out", required=True, help="output directory for the LoRA adapter")
    # Overrides, for sweeping without editing the pack.
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--max-len", type=int)
    args = ap.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    hp = dict(job["hyperparameters"])
    for key, val in (("epochs", args.epochs), ("lr", args.lr),
                     ("lora_r", args.lora_r), ("max_len", args.max_len)):
        if val is not None:
            hp[key] = val

    root = job_path.parent
    train_file = root / job["train_file"]
    val_file = root / job["val_file"]
    base = job["base_hf_repo"]

    print(f"adapter : {job['adapter_id']}")
    print(f"base    : {base}")
    print(f"data    : {train_file.name} / {val_file.name}")
    print(f"config  : {hp}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainerCallback)
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This script is the GPU stage — see the job pack README "
            "for the Colab and cloud-GPU paths.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base)

    lora = LoraConfig(
        r=hp["lora_r"], lora_alpha=hp.get("lora_alpha", hp["lora_r"] * 2),
        lora_dropout=hp.get("lora_dropout", 0.05),
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    data = load_dataset("json", data_files={"train": str(train_file), "val": str(val_file)})
    # `requirement` / `recipe` are provenance columns, not model inputs.
    keep = ["messages"]
    data = data.remove_columns([c for c in data["train"].column_names if c not in keep])

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_length=hp["max_len"],
        packing=False,
        # The prompt is mostly retrieved context that the model must not learn
        # to reproduce; grading only the assistant turn is what makes this a
        # behaviour tune rather than a document memoriser.
        completion_only_loss=True,
        gradient_checkpointing=True,
        seed=hp.get("seed", 7),
        report_to="none",
    )

    history: list[dict] = []

    class Collect(TrainerCallback):
        def on_log(self, cfg_, state, control, logs=None, **kw):
            if logs:
                history.append({"step": state.global_step, **logs})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        peft_config=lora,
        args=cfg,
        callbacks=[Collect()],
    )

    t0 = time.time()
    trainer.train()
    trainer.save_model(args.out)

    log = {
        "adapter_id": job["adapter_id"],
        "base_hf_repo": base,
        "spec_sha256": job.get("spec_sha256"),
        "hyperparameters": hp,
        "n_train": len(data["train"]), "n_val": len(data["val"]),
        "seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0),
        "history": history,
    }
    out = Path(args.out)
    (out.parent / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nadapter saved to {out}  ({log['seconds']}s on {log['gpu']})")
    print("next: convert_lora_to_gguf.py, then `python -m finetune import --pack ...` "
          "on the Studio machine")


if __name__ == "__main__":
    main()
