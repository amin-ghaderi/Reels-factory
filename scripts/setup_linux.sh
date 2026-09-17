#!/usr/bin/env bash
set -euo pipefail
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py setup
echo "Done. Put a video in data/inbox, then run: .venv/bin/python run_pipeline.py inbox"
