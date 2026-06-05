#!/usr/bin/env bash
# Sync /home/adodas/zt and /home/adodas/CCAC to the abcde GitHub repo.
# Usage: bash /home/adodas/abcde/sync.sh ["optional commit message"]

set -euo pipefail

REPO_DIR="/home/adodas/abcde"
SRC_ZT="/home/adodas/zt"
SRC_CCAC="/home/adodas/CCAC"
MSG="${1:-Update: sync zt and CCAC ($(date '+%Y-%m-%d %H:%M:%S'))}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Error: $REPO_DIR is not a git repo" >&2
  exit 1
fi

# Mirror source -> repo subdirs.
# --delete keeps repo in sync with source (removed files in source also removed here).
# Exclude nested .git so we don't accidentally embed submodules again.
rsync -a --delete --exclude='.git' "$SRC_ZT/"   "$REPO_DIR/zt/"
rsync -a --delete --exclude='.git' "$SRC_CCAC/" "$REPO_DIR/CCAC/"

cd "$REPO_DIR"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit."
  exit 0
fi

git add -A
git commit -m "$MSG"
git push origin main

echo
echo "Done. View: https://github.com/liuwenjing613-maker/abcde"
