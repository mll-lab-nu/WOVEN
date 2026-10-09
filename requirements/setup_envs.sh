#!/bin/bash
# Create the two conda environments used for training (their torch / vLLM pins conflict).
#   woven_sft  SFT with qwen-vl-finetune  (Python 3.11, torch 2.6 / CUDA 12.4, DeepSpeed, flash-attn 2.7.4)
#   woven_rl   GRPO with verl 0.7.1        (Python 3.12, torch 2.9 / CUDA 12.8, vLLM 0.12, flash-attn 2.8.3)
#
# Usage:
#   bash requirements/setup_envs.sh            # both
#   bash requirements/setup_envs.sh --sft-only
#   bash requirements/setup_envs.sh --rl-only
# Env names can be changed with SFT_ENV / RL_ENV. Idempotent.
#
# The SFT code itself comes from the Qwen2.5-VL repository (pinned commit), patched with
# training/sft/apply_patch.sh:
#   git clone https://github.com/QwenLM/Qwen2.5-VL.git
#   git -C Qwen2.5-VL checkout $(cat training/sft/qwenvl_patch/UPSTREAM_COMMIT)
#   bash training/sft/apply_patch.sh Qwen2.5-VL
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SFT_ENV=${SFT_ENV:-woven_sft}
RL_ENV=${RL_ENV:-woven_rl}
ONLY=${1:-}

have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

if [ "$ONLY" != "--rl-only" ]; then
  have_env "$SFT_ENV" || conda create -y -n "$SFT_ENV" python=3.11
  conda run -n "$SFT_ENV" pip install -r "$HERE/sft.txt"
fi

if [ "$ONLY" != "--sft-only" ]; then
  have_env "$RL_ENV" || conda create -y -n "$RL_ENV" python=3.12
  conda run -n "$RL_ENV" pip install -r "$HERE/rl.txt"
fi
echo "environments ready: ${SFT_ENV} (SFT), ${RL_ENV} (GRPO)"
