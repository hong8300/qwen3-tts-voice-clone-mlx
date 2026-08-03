#!/bin/bash
# Qwen3-TTS ボイスクローン UI を起動する
set -e
cd "$(dirname "$0")"
exec .venv/bin/python app.py "$@"
