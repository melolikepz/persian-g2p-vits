#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi

source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

TEXT="${1:-سلام، این یک آزمایش تبدیل متن فارسی به گفتار است.}"
MODEL="outputs/persian-mms-tts"
if [[ ! -f "$MODEL/config.json" ]]; then
  MODEL="facebook/mms-tts-fas"
  echo "Fine-tuned model not found; using base model: $MODEL"
fi

python scripts/infer.py \
  --model "$MODEL" \
  --text "$TEXT" \
  --output outputs/sample.wav
