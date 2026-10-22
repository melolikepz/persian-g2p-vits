#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi
if [[ ! -f checkpoints/mms-tts-fas-train/config.json ]]; then
  echo "Missing training checkpoint. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi

source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  run_vits_finetuning.py \
  training_config_examples/finetune_mms_fa_local.json
