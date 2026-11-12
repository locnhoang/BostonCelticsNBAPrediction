#!/usr/bin/env bash
# Local fallback for the daily update (NBA.com does not block home IPs).
# Usage: scripts/run_update.sh [--force]   --force skips the already-updated-today check.
# Times are local (launchd runs this at 10:30 local; CI commits ~14:00 UTC).
set -euo pipefail
cd "$(dirname "$0")/.."

fail() {
  git rebase --abort >/dev/null 2>&1 || true
  echo "run_update: $*" >&2
  exit 1
}

# True if every commit in origin/main..HEAD touches only site/data/.
only_data_ahead() {
  local files
  files=$(git diff --name-only origin/main...HEAD)
  [[ -z "$files" ]] || ! grep -qv '^site/data/' <<<"$files"
}

# Generated data is disposable: drop local data edits and unpushed data-only commits
# so we always rebuild on top of origin/main. Anything else stops the script.
sync_to_origin() {
  git fetch --quiet origin main || fail "git fetch failed"
  git checkout -- site/data 2>/dev/null || true
  git clean -fdq -- site/data
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    fail "working tree has non-data changes; commit or stash them first"
  fi
  if [[ -n "$(git rev-list origin/main..HEAD)" ]]; then
    only_data_ahead || fail "local main has unpushed non-data commits; push them first"
    git reset --quiet --hard origin/main
  else
    git merge --quiet --ff-only origin/main || fail "cannot fast-forward to origin/main"
  fi
}

update_and_commit() {
  if ! uv run celtics update; then
    git checkout -- site/data 2>/dev/null || true
    fail "celtics update failed (see output above)"
  fi
  git add site/data
  if git diff --cached --quiet; then
    return 1
  fi
  git commit --quiet -m "data: local update $(date +%F)"
}

branch=$(git branch --show-current)
[[ "$branch" == "main" ]] || fail "refusing to run on branch '${branch:-detached}' (Pages deploys only from main)"

sync_to_origin

if [[ "${1:-}" != "--force" ]] && \
   [[ -n "$(git log origin/main --since="$(date +%F) 00:00" --format=%H -- site/data/history.json)" ]]; then
  echo "already updated today"
  exit 0
fi

update_and_commit || { echo "no data changes"; exit 0; }
if ! git push --quiet origin main; then
  echo "run_update: push rejected; rebuilding on latest origin/main and retrying once" >&2
  sync_to_origin
  update_and_commit || { echo "no data changes after resync"; exit 0; }
  git push --quiet origin main || fail "git push failed after one retry"
fi
echo "pushed data update"
