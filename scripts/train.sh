#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python scripts/train_vits.py "$@"
