#!/usr/bin/env bash
# Sobe a aplicacao completa: core nativo (opcional), backend FastAPI e frontend.
# Uso: ./scripts/run_all.sh [--no-native] [--no-frontend]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_NATIVE=1
RUN_FRONTEND=1
for arg in "$@"; do
  case "$arg" in
    --no-native) BUILD_NATIVE=0 ;;
    --no-frontend) RUN_FRONTEND=0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p artifacts/pids artifacts/logs

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

PIDS=()
cleanup() {
  echo ""
  echo "Encerrando processos..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Core C++ (opcional, nao-fatal: a API faz fallback para o core Python).
if [[ "$BUILD_NATIVE" -eq 1 ]]; then
  if command -v g++ >/dev/null 2>&1; then
    ./scripts/build_native.sh || echo "WARN: build do core nativo falhou; seguindo com core Python"
  else
    echo "WARN: g++ nao encontrado; seguindo com core Python"
  fi
fi

# 2. Backend FastAPI.
"$PYTHON_BIN" -m uvicorn adaptive_3dbp.api.main:app --app-dir src --port 8000 \
  >artifacts/logs/backend.log 2>&1 &
PIDS+=($!)
echo $! > artifacts/pids/backend.pid
echo "Backend:  http://127.0.0.1:8000 (docs em /docs, log em artifacts/logs/backend.log)"

# 3. Frontend React/Vite.
if [[ "$RUN_FRONTEND" -eq 1 ]]; then
  if [[ ! -d frontend/node_modules ]]; then
    echo "Instalando dependencias do frontend..."
    (cd frontend && npm install --silent)
  fi
  (cd frontend && npm run dev >../artifacts/logs/frontend.log 2>&1) &
  PIDS+=($!)
  echo $! > artifacts/pids/frontend.pid
  echo "Frontend: http://127.0.0.1:5173 (log em artifacts/logs/frontend.log)"
fi

echo ""
echo "Aplicacao no ar. Ctrl+C encerra tudo."
wait
