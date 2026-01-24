#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/pipeline.py

echo "OK ✅ Pipeline executed."
echo "Check: outputs/, reports/, data/processed/"
