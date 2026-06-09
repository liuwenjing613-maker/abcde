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

# git add -f overrides .gitignore; drop local-only binaries before commit
while IFS= read -r -d '' f; do
  git rm --cached -f -- "$f" >/dev/null
done < <(find CCAC zt \( -name '*.zip' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print0 2>/dev/null)

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit."
else
  git commit -m "$MSG"
fi

git push origin main

echo
echo "Done. View: https://github.com/liuwenjing613-maker/abcde"
