#!/usr/bin/env bash
# Rebuild the HF Spaces deploy branch from main and force-push to the
# Hugging Face Spaces git remote.
#
# Why this script exists:
#   1. HF Hub rejects pushes containing binary files anywhere in history.
#      The committed parquet (data/parquet_cache/iam_gw_pipeline.parquet)
#      is on main, so we can't just push main directly. Solution: orphan
#      branch with that file removed, single commit, no history to walk.
#   2. HF Spaces needs YAML frontmatter at the top of README.md (sdk,
#      app_port, etc.). We don't want that YAML on main because GitHub
#      renders it as raw text. Solution: prepend it on the deploy branch
#      only.
#   3. Both rules together mean every redeploy is the same 8-step dance.
#      This script is that dance.
#
# Usage:
#   scripts/deploy_to_hf.sh             # uses git remote 'hf'
#   scripts/deploy_to_hf.sh hf-staging  # use a different remote name
#
# Prerequisites:
#   - Clean working tree (no uncommitted changes anywhere)
#   - HF git remote configured. To add:
#       git remote add hf https://huggingface.co/spaces/<user>/<space>
#   - A write-scope HF token. Create at huggingface.co/settings/tokens
#     and use it as the git push password when prompted.

set -euo pipefail

REMOTE="${1:-hf}"

GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
RESET=$'\033[0m'

step() { echo "${GREEN}-->${RESET} $1"; }
warn() { echo "${YELLOW}warn:${RESET} $1" >&2; }
die()  { echo "${RED}error:${RESET} $1" >&2; exit 1; }

# YAML frontmatter prepended to README.md on the deploy branch.
# Keep short_description under 60 chars (HF metadata validator).
read -r -d '' YAML_BLOCK <<'YAML' || true
---
title: Historical Document Extractor
emoji: 📜
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: TrOCR + Claude OCR for historical handwriting
---

YAML

# Sanity checks before doing anything destructive.
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  die "git remote '$REMOTE' not configured. Add it with:
  git remote add $REMOTE https://huggingface.co/spaces/<user>/<space>"
fi

if ! git diff-index --quiet HEAD --; then
  die "working tree has uncommitted changes; commit or stash first"
fi

if [ -n "$(git ls-files --others --exclude-standard)" ]; then
  warn "untracked files present — they will be added to the deploy snapshot"
fi

ORIG_BRANCH=$(git symbolic-ref --short HEAD)

step "syncing main from origin"
git checkout main >/dev/null
git pull --ff-only origin main

if git show-ref --verify --quiet refs/heads/hf-deploy; then
  step "deleting existing local hf-deploy branch"
  git branch -D hf-deploy >/dev/null
fi

step "creating orphan deploy branch (no commit history → no binary files)"
git checkout --orphan hf-deploy >/dev/null
git rm --cached -r . >/dev/null

step "excluding binary parquet from deploy snapshot"
rm -rf data/parquet_cache

step "prepending HF Spaces YAML frontmatter to README.md"
{ printf '%s' "$YAML_BLOCK"; cat README.md; } > README.md.new
mv README.md.new README.md

step "staging deploy snapshot"
git add -A

step "committing"
git commit -m "HF Space deploy snapshot" >/dev/null

step "pushing to ${REMOTE}/main (--force)"
echo "${YELLOW}When prompted, paste your write-scope HF token as the password.${RESET}"
if ! git push "$REMOTE" hf-deploy:main --force; then
  echo "${RED}push failed.${RESET} You're on the hf-deploy branch. Switch back manually:" >&2
  echo "  git checkout $ORIG_BRANCH" >&2
  exit 1
fi

step "returning to $ORIG_BRANCH"
git checkout "$ORIG_BRANCH" >/dev/null

SPACE_PATH=$(git remote get-url "$REMOTE" | sed -E 's|.*/spaces/||')
echo ""
echo "${GREEN}Deploy snapshot pushed.${RESET} HF rebuild starts in ~30s. Watch:"
echo "  https://huggingface.co/spaces/${SPACE_PATH}"
