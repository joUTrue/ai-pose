#!/usr/bin/env bash
set -e

# (선택) 로그 덜 시끄럽게
export PYTHONUNBUFFERED=1

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
