"""Merge a trained LoRA adapter into the base model and save full HF weights,
ready for GGUF conversion. Run on the GPU machine after train_qlora.py.

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
