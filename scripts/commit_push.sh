#!/usr/bin/env bash
# Explicit, user-invoked commit + push. Does not change repository visibility.
# Usage: bash scripts/commit_push.sh "commit message"
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 || -z "$1" ]]; then
  echo 'uso: bash scripts/commit_push.sh "mensagem do commit"' >&2
  exit 2
fi
# Never remove index.lock: another Git process may own it.
if [[ -e .git/index.lock ]]; then
  echo 'ABORTANDO: existe .git/index.lock; confira os processos Git ativos.' >&2
  exit 1
fi
BRANCH=$(git symbolic-ref --quiet --short HEAD)
git add -A
# Inspect the actual staged objects, including entries staged before this script.
python3 - <<'PY_GUARD'
import subprocess
from pathlib import PurePosixPath
names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]).decode().split("\0")
errors = []
for name in filter(None, names):
    path = PurePosixPath(name)
    forbidden = name.startswith((".private/", "artifacts/", "Vault Adaptive 3DBP/", "docs/articles/", "docs/artigo_latex/", "docs/guia_didatico/"))
    forbidden |= any(part.startswith(".env") and part != ".env.example" for part in path.parts)
    forbidden |= path.suffix.lower() in {".pdf", ".docx", ".pem", ".key", ".db", ".sqlite", ".npz"}
    forbidden |= name.startswith(".claude/") and not name.startswith(".claude/agents/")
    size = int(subprocess.check_output(["git", "cat-file", "-s", ":" + name]))
    if forbidden or size > 95 * 1024 * 1024:
        errors.append(name)
if errors:
    raise SystemExit("ABORTANDO: revise o stage; entradas privadas ou grandes: " + ", ".join(errors))
PY_GUARD
if git diff --cached --quiet; then
  echo 'Nada novo para commit; nenhum push realizado.'
  exit 0
fi
git commit -m "$1"
git push --set-upstream origin "$BRANCH"
