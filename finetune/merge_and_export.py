"""Merge a LoRA adapter into the base model and save full HF weights.

ESCAPE HATCH, not the default path. The pipeline ships adapters as separate
GGUF files applied at load time (see core/adapters.py and docs/finetuning.md
D2): that keeps one shared base model on the device and makes a tuned agent a
~30 MB download instead of another gigabyte. Merging is for a runtime that
cannot apply a LoRA at all — llama.cpp and llama.rn both can.

If you do merge, the result is a full model: convert and quantise it, add it to
core/catalog.py as its own entry, and expect every device to download it.

Run on the GPU machine after train_qlora.py.

    python merge_and_export.py --base Qwen/Qwen3-1.7B \
        --adapter out/meeting-intelligence-lora \
        --out out/meeting-intelligence-merged
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load base in full precision (not 4-bit) so the merge is lossless
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model saved to {args.out}")
    print("next:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {args.out} --outfile agent-f16.gguf")
    print("  llama-quantize agent-f16.gguf agent-Q4_K_M.gguf Q4_K_M")


if __name__ == "__main__":
    main()
