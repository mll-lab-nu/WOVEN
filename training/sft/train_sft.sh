#!/bin/bash
# LoRA SFT of Qwen2.5-VL on WOVEN with qwen-vl-finetune (single node, all visible GPUs).
#
# Usage:
#   QWENVL_REPO=/path/to/Qwen2.5-VL/qwen-vl-finetune \
#   TRAIN_JSON=data/sft/train.json RUN_NAME=woven_sft \
#   bash training/sft/train_sft.sh
#
# Prerequisites: a qwen-vl-finetune checkout patched with training/sft/apply_patch.sh, and
# TRAIN_JSON produced by training/prepare_data.py --format sft (use --permute 4).
#
# Environment variables:
#   QWENVL_REPO      qwen-vl-finetune directory (or the Qwen2.5-VL repo root)       [required]
#   TRAIN_JSON       SFT annotation file                                            [required]
#   RUN_NAME         run name; checkpoints go to $OUTPUT_ROOT/sft/$RUN_NAME          [required]
#   MODEL_PATH       base model, local dir or hub id; its name must contain "Qwen2.5"
#                    (qwen-vl-finetune picks the model class from it)  (default Qwen/Qwen2.5-VL-3B-Instruct)
#   OUTPUT_ROOT      (default ./outputs)
#   VAL_JSON         optional eval annotation file (only used if EVAL_STRATEGY != no)
#   NPROC_PER_NODE   GPUs to use (default: all visible)
#   REPORT_TO        wandb | none (default wandb; set WANDB_API_KEY or run `wandb login`)
#   EXTRA_ARGS       extra trainer flags, e.g. "--max_steps 10" for a smoke test
# Training resumes automatically from the latest checkpoint-* in the output directory.
#
# Recipe: LoRA r=16, alpha=32, dropout=0.05 on q/k/v/o projections; 3 epochs; lr 1e-4, cosine,
# warmup 0.05; global batch 64; max_pixels 50176, min_pixels 12544; bf16; DeepSpeed ZeRO-3.
set -euo pipefail
: "${QWENVL_REPO:?set QWENVL_REPO to the qwen-vl-finetune directory}"
: "${TRAIN_JSON:?set TRAIN_JSON (output of training/prepare_data.py)}"
: "${RUN_NAME:?set RUN_NAME}"

[ -d "$QWENVL_REPO/qwen-vl-finetune" ] && QWENVL_REPO="$QWENVL_REPO/qwen-vl-finetune"
[ -f "$QWENVL_REPO/qwenvl/train/train_qwen.py" ] || { echo "not a qwen-vl-finetune directory: $QWENVL_REPO" >&2; exit 1; }
TRAIN_JSON=$(realpath "$TRAIN_JSON")
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
[ -e "$MODEL_PATH" ] && MODEL_PATH=$(realpath "$MODEL_PATH")
OUTPUT_ROOT=$(realpath -m "${OUTPUT_ROOT:-./outputs}")
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/sft/$RUN_NAME}
mkdir -p "$OUTPUT_DIR"

# GPUs (respects CUDA_VISIBLE_DEVICES)
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
  fi
fi

# Global batch 64 = PER_DEVICE_BS x GRAD_ACCUM x NPROC_PER_NODE
PER_DEVICE_BS=${PER_DEVICE_BS:-2}
GRAD_ACCUM=${GRAD_ACCUM:-$(( 64 / (PER_DEVICE_BS * NPROC_PER_NODE) ))}

EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
MAX_PIXELS=${MAX_PIXELS:-50176}
MIN_PIXELS=${MIN_PIXELS:-12544}
MAX_LEN=${MAX_LEN:-8192}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
EVAL_STRATEGY=${EVAL_STRATEGY:-no}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LOGGING_STEPS=${LOGGING_STEPS:-10}
REPORT_TO=${REPORT_TO:-wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-woven}

EVAL_ARGS=()
if [ -n "${VAL_JSON:-}" ]; then
  EVAL_ARGS=(--eval_dataset_use "$(realpath "$VAL_JSON")")
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

cd "$QWENVL_REPO"
torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_use "$TRAIN_JSON" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --lora_enable True \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$((PER_DEVICE_BS * 2))" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model True \
    --load_best_model_at_end False \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --learning_rate "$LR" \
    --weight_decay 0.0 \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps "$LOGGING_STEPS" \
    --model_max_length "$MAX_LEN" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    ${EXTRA_ARGS:-}
