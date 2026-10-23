#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate Taopy310 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc >/dev/null 2>&1 || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate Taopy310 2>/dev/null || true
fi

python scripts/tui.py "$@"
