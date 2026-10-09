#!/bin/bash
# Install the WOVEN dataset registry and eval-dataset support into a qwen-vl-finetune checkout.
#
# Usage:
#   git clone https://github.com/QwenLM/Qwen2.5-VL.git
#   git -C Qwen2.5-VL checkout $(cat training/sft/qwenvl_patch/UPSTREAM_COMMIT)
#   bash training/sft/apply_patch.sh Qwen2.5-VL/qwen-vl-finetune
#
# The argument may be either the repository root or its qwen-vl-finetune/ directory.
# Files copied (they replace the upstream versions at the pinned commit):
#   qwenvl/data/__init__.py        registry: accepts annotation-file paths and woven_train / woven_val
#   qwenvl/data/data_processor.py  builds an optional eval dataset from --eval_dataset_use
#   qwenvl/train/argument.py       adds the --eval_dataset_use argument
# Set FORCE=1 to copy even if the checkout is not at the pinned commit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$HERE/qwenvl_patch"
TARGET=${1:?usage: apply_patch.sh <Qwen2.5-VL repo or its qwen-vl-finetune dir>}
[ -d "$TARGET/qwen-vl-finetune" ] && TARGET="$TARGET/qwen-vl-finetune"
[ -f "$TARGET/qwenvl/train/train_qwen.py" ] || { echo "not a qwen-vl-finetune directory: $TARGET" >&2; exit 1; }

PINNED=$(cat "$PATCH/UPSTREAM_COMMIT")
HEAD=$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$HEAD" != "$PINNED" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "checkout is at $HEAD, expected $PINNED." >&2
  echo "run: git -C $TARGET checkout $PINNED   (or set FORCE=1)" >&2
  exit 1
fi

cp "$PATCH/__init__.py"       "$TARGET/qwenvl/data/__init__.py"
cp "$PATCH/data_processor.py" "$TARGET/qwenvl/data/data_processor.py"
cp "$PATCH/argument.py"       "$TARGET/qwenvl/train/argument.py"
echo "patched $TARGET"
