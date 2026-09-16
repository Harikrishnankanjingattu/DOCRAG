#!/usr/bin/env bash
set -e

echo "==================================================="
echo "  Starting Document RAG Interactive Studio"
echo "==================================================="
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
