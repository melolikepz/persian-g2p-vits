#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Note: this setup is tuned for Apple Silicon; CPU/CUDA may need different settings."
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-local.txt

(
  cd monotonic_align
  python setup.py build_ext --inplace
)

mkdir -p checkpoints outputs
if [[ ! -f checkpoints/mms-tts-fas-train/config.json ]]; then
  python convert_original_discriminator_checkpoint.py \
    --language_code fas \
    --pytorch_dump_folder_path checkpoints/mms-tts-fas-train
else
  echo "Training checkpoint already exists; skipping conversion."
fi

python - <<'PY'
import platform
import torch

print(f"Python platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

echo "Setup complete. Run ./scripts/train_local.sh"
