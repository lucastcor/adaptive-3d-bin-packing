#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
# --reload-dir src: sem ele, o watcher observa o projeto inteiro e as escritas
# de artefatos/datasets durante uma campanha reiniciam o servidor no meio da
# execução (jobs em memória somem e o frontend recebe "Load failed").
"$PYTHON_BIN" -m uvicorn adaptive_3dbp.api.main:app --reload --reload-dir src --app-dir src --port 8000
