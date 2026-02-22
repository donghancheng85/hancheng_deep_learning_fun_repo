#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 1) Install CUDA PyTorch
pip install -r requirements.torch.txt

# 2) Install rest of deps from pinned lock
pip install -r requirements.lock

# Sanity checks
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip check
