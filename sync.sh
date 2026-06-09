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

git add CCAC zt sync.sh

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$MSG"
git push origin main

echo
echo "Done. View: https://github.com/liuwenjing613-maker/abcde"
