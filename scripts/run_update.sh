#!/usr/bin/env bash
# Local fallback for the daily update (NBA.com does not block home IPs).
# Usage: scripts/run_update.sh [--force]   --force skips the already-updated-today check.
set -euo pipefail
cd "$(dirname "$0")/.."

# Abort any rebase in progress so the repo is never left mid-rebase, then fail.
fail() {
  git rebase --abort >/dev/null 2>&1 || true
  echo "run_update: $*" >&2
  exit 1
}

branch=$(git branch --show-current)
if [[ "$branch" != "main" ]]; then
  echo "run_update: refusing to run on branch '$branch' (Pages deploys only from main)" >&2
  exit 1
fi

if ! git pull --rebase --quiet; then
  fail "git pull --rebase failed; resolve manually"
fi

# Check origin/main, not HEAD, so an unpushed local commit cannot mask a missing update.
if [[ "${1:-}" != "--force" ]]; then
  if [[ -n "$(git log origin/main --since="$(date +%F) 00:00" --format=%H -- site/data/history.json)" ]]; then
    echo "already updated today"
    exit 0
  fi
fi

uv run celtics update
git add site/data
if ! git diff --cached --quiet; then
  git commit -m "data: local update $(date +%F)"
  if ! git push --quiet; then
    echo "run_update: push rejected; rebasing onto origin and retrying once" >&2
    git pull --rebase --quiet || fail "git pull --rebase failed before retry push"
    git push --quiet || fail "git push failed after one rebase retry"
  fi
fi
