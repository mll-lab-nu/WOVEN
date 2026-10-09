#!/bin/bash
# GRPO on WOVEN with verl 0.7.1 (single node, all visible GPUs) and the anti-bias reward.
#
# Usage:
#   INIT_MODEL=outputs/merged/woven_sft PARQUET=data/rl/train.parquet RUN_NAME=woven_grpo \
#   VAL_PARQUET=data/rl/val.parquet bash training/grpo/train_grpo.sh
# or start from an SFT LoRA output (merged once into $OUTPUT_ROOT/rl_init/$RUN_NAME):
#   ADAPTER_DIR=outputs/sft/woven_sft PARQUET=... RUN_NAME=... bash training/grpo/train_grpo.sh
#
# Prepare parquet files with training/prepare_data.py --format verl.
#
# Environment variables:
#   INIT_MODEL     initial policy (full HF checkpoint)                    [INIT_MODEL or ADAPTER_DIR]
#   ADAPTER_DIR    SFT LoRA output to merge into BASE_MODEL as the initial policy
#   PARQUET        training parquet                                       [required]
#   RUN_NAME       experiment name; checkpoints in $OUTPUT_ROOT/rl/$RUN_NAME [required]
#   VAL_PARQUET    validation parquet (validated before training and at the last step);
#                  if unset, validation is disabled
#   REF_MODEL      KL reference model (default: $BASE_MODEL)
#   BASE_MODEL     base model (default Qwen/Qwen2.5-VL-3B-Instruct)
#   STEPS          total training steps (default 150)
#   NGPUS          GPUs on this node (default: all visible)
#   OUTPUT_ROOT    (default ./outputs)
#   LOGGER         verl trainer.logger (default '["console","wandb"]'; set WANDB_API_KEY)
#   EXTRA_ARGS     extra Hydra overrides
# Training resumes automatically from the latest checkpoint in the output directory (save_freq 50).
#
# Settings: GRPO, 16 prompts x 8 rollouts per step, actor lr 1e-6, PPO mini-batch 8,
# KL loss (low_var_kl, coef 0.005) against the reference model, no KL in reward, no entropy bonus,
# max response length 8 tokens, vLLM rollout, FSDP2 with parameter/optimizer offload.
set -euo pipefail
: "${RUN_NAME:?set RUN_NAME}"
: "${PARQUET:?set PARQUET}"

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
REF_MODEL=${REF_MODEL:-$BASE_MODEL}
STEPS=${STEPS:-150}
OUTPUT_ROOT=$(realpath -m "${OUTPUT_ROOT:-./outputs}")
OUT_DIR=${OUT_DIR:-$OUTPUT_ROOT/rl/$RUN_NAME}
REWARD_FN=${REWARD_FN:-$HERE/reward.py}
LOGGER=${LOGGER:-'["console","wandb"]'}
PARQUET=$(realpath "$PARQUET")
mkdir -p "$OUT_DIR"

if [ -z "${NGPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NGPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else NGPUS=$(nvidia-smi -L | wc -l); fi
fi

# Ray kills workers at 95% of node RAM by default; raise the threshold for headroom.
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.98}

# Initial policy: given directly, or merged once from an SFT LoRA adapter.
if [ -z "${INIT_MODEL:-}" ]; then
  : "${ADAPTER_DIR:?set INIT_MODEL or ADAPTER_DIR}"
  INIT_MODEL=$OUTPUT_ROOT/rl_init/$RUN_NAME
  if [ ! -e "$INIT_MODEL/config.json" ]; then
    python "$HERE/../sft/merge_lora.py" "$ADAPTER_DIR" "$INIT_MODEL" --base "$BASE_MODEL"
  fi
fi
[ -e "$INIT_MODEL" ] && INIT_MODEL=$(realpath "$INIT_MODEL")
[ -e "$REF_MODEL" ] && REF_MODEL=$(realpath "$REF_MODEL")

if [ -n "${VAL_PARQUET:-}" ]; then
  VAL_ARGS=(data.val_files="$(realpath "$VAL_PARQUET")" trainer.test_freq=1000)
else
  VAL_ARGS=(data.val_files="$PARQUET" trainer.val_before_train=False trainer.test_freq=-1)
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="$PARQUET" "${VAL_ARGS[@]}" \
    data.image_key=images data.train_batch_size=16 data.shuffle=${SHUFFLE:-False} data.max_prompt_length=${MAX_PROMPT:-4096} \
    data.max_response_length=8 data.filter_overlong_prompts=True data.truncation=error \
    actor_rollout_ref.model.path="$INIT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.ref.model.path="$REF_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.free_cache_engine=True \
    custom_reward_function.path="$REWARD_FN" custom_reward_function.name=compute_score \
    trainer.balance_batch=True trainer.logger="$LOGGER" \
    trainer.project_name=${WANDB_PROJECT:-woven_rl} trainer.experiment_name="$RUN_NAME" \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=$NGPUS trainer.nnodes=1 trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=${KEEP_CKPTS:-1} \
    trainer.total_epochs=100 trainer.total_training_steps=$STEPS trainer.default_local_dir="$OUT_DIR" \
    ${EXTRA_ARGS:-}
