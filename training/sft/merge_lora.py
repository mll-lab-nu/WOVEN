"""Merge a LoRA adapter produced by train_sft.sh into the base model and save a full checkpoint.

Usage:
    python training/sft/merge_lora.py <adapter_dir> <out_dir> [--base Qwen/Qwen2.5-VL-3B-Instruct]

<adapter_dir> is a directory containing adapter_config.json; if it has none, the latest
checkpoint-* subdirectory is used. The base model defaults to $BASE_MODEL or
Qwen/Qwen2.5-VL-3B-Instruct. The processor/tokenizer is copied from the base model.
Runs on CPU.
"""
import argparse
import glob
import os
import re

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def resolve_adapter(path: str) -> str:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    ckpts = [p for p in glob.glob(os.path.join(path, "checkpoint-*"))
             if os.path.isfile(os.path.join(p, "adapter_config.json"))]
    if not ckpts:
        raise SystemExit(f"no adapter_config.json in {path} or its checkpoint-* subdirectories")
    return max(ckpts, key=lambda p: int(re.search(r"checkpoint-(\d+)$", p).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    args = ap.parse_args()

    adapter = resolve_adapter(args.adapter_dir)
    print(f"load base from {args.base}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu")
    print(f"attach LoRA from {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out_dir)
    print(f"saved merged model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
