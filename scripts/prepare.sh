#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python scripts/prepare_dataset.py "$@"
