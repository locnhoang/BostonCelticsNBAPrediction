#!/usr/bin/env bash
# Local fallback for the daily update (NBA.com does not block home IPs).
set -euo pipefail
cd "$(dirname "$0")/.."
git pull --rebase --quiet
uv run celtics update
git add site/data
if ! git diff --cached --quiet; then
  git commit -m "data: local update $(date +%F)"
  git push
fi
