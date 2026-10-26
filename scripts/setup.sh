#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name persian-g2p-vits --display-name "Persian G2P VITS"

echo "Environment ready. Activate it with: source .venv/bin/activate"
