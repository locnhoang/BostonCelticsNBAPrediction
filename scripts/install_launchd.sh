#!/usr/bin/env bash
# Install (or reinstall) the daily local update job. Usage: scripts/install_launchd.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.celtics.update.plist"
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" \
  "$REPO/scripts/com.celtics.update.plist.template" > "$TARGET"
launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"
echo "Installed $TARGET (runs daily at 10:30). Logs: ~/Library/Logs/celtics-update.log"
