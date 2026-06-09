#!/usr/bin/env bash
# Push /home/adodas/CCAC and /home/adodas/zt to github.com/liuwenjing613-maker/abcde
# Usage: bash /home/adodas/sync.sh ["optional commit message"]

set -euo pipefail

export GIT_DIR="/home/adodas/.git-abcde"
export GIT_WORK_TREE="/home/adodas"

MSG="${1:-Update: CCAC and zt ($(date '+%Y-%m-%d %H:%M:%S'))}"

if [[ ! -d "$GIT_DIR" ]]; then
  echo "Error: git dir not found at $GIT_DIR" >&2
  exit 1
fi

cd "$GIT_WORK_TREE"

git add -f CCAC zt sync.sh

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$MSG"

echo
echo "Committed. Push when ready:"
echo "  export GIT_DIR=/home/adodas/.git-abcde GIT_WORK_TREE=/home/adodas && git push origin main"
echo "View: https://github.com/liuwenjing613-maker/abcde"
