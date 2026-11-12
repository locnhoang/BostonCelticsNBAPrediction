#!/usr/bin/env bash
# Local fallback for the daily update (NBA.com does not block home IPs).
# Usage: scripts/run_update.sh [--force]   --force skips the already-updated-today check.
# Times are local (launchd runs this at 10:30 local; CI commits ~14:00 UTC).
# Generated site/data is disposable: every run rebuilds it on top of origin/main.
# Non-data work (uncommitted changes, unpushed commits, an in-progress rebase) is never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() {
  echo "run_update: $*" >&2
  exit 1
}

# Drop staged, unstaged and untracked changes under site/data only.
discard_data() {
  git reset --quiet HEAD -- site/data 2>/dev/null || true
  git checkout -- site/data 2>/dev/null || true
  git clean -fdq -- site/data
}

# True if the local side of origin/main...HEAD touches only site/data/.
only_data_ahead() {
  local files
  files=$(git diff --no-renames --name-only origin/main...HEAD)
  [[ -z "$files" ]] || ! grep -qv '^site/data/' <<<"$files"
}

sync_to_origin() {
  git fetch --quiet origin main || fail "git fetch failed"
  discard_data
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

# Sets committed=1 if a data commit was made, 0 if nothing changed. Any failure exits.
update_and_commit() {
  committed=0
  if ! uv run celtics update; then
    discard_data
    fail "celtics update failed (see output above)"
  fi
  git add site/data || fail "git add failed"
  if git diff --cached --quiet; then
    return 0
  fi
  if ! git commit --quiet -m "data: local update $(date +%F)"; then
    discard_data
    fail "git commit failed"
  fi
  committed=1
}

if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  fail "a rebase is in progress; finish or abort it first"
fi
branch=$(git branch --show-current)
[[ "$branch" == "main" ]] || fail "refusing to run on branch '${branch:-detached}' (Pages deploys only from main)"

sync_to_origin

if [[ "${1:-}" != "--force" ]] && \
   [[ -n "$(git log origin/main --since="$(date +%F) 00:00" --format=%H -- site/data/history.json)" ]]; then
  echo "already updated today"
  exit 0
fi

update_and_commit
if [[ $committed -eq 0 ]]; then
  echo "no data changes"
  exit 0
fi
if ! git push --quiet origin main; then
  echo "run_update: push rejected; rebuilding on latest origin/main and retrying once" >&2
  sync_to_origin
  update_and_commit
  if [[ $committed -eq 0 ]]; then
    echo "no data changes after resync"
    exit 0
  fi
  git push --quiet origin main || fail "git push failed after one retry"
fi
echo "pushed data update"
