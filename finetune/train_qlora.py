"""QLoRA fine-tune of a Qwen3 SLM on an agent dataset.

RUN THIS ON A GPU MACHINE (the Ubuntu box if it has an NVIDIA GPU, or any
cloud GPU / Colab T4). The Windows prototype server (t3.xlarge, CPU-only)
cannot train — see finetune/README.md for the full workflow.

Setup on the GPU machine:
    pip install -r requirements-finetune.txt

Train:
    python train_qlora.py \
        --base Qwen/Qwen3-1.7B \
        --train datasets/meeting-intelligence-train.jsonl \
        --val   datasets/meeting-intelligence-val.jsonl \
        --out   out/meeting-intelligence-lora

Then merge + convert to GGUF (see README.md step 3).
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    data = load_dataset("json", data_files={"train": args.train, "val": args.val})

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        peft_config=lora,
        args=cfg,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"LoRA adapter saved to {args.out} — next: merge + convert to GGUF (README step 3)")


if __name__ == "__main__":
    main()
