#!/usr/bin/env bash
# Rsync CC JSONL files from desktop to MCPHub LXC, then run import.
#
# Install as cron on MCPHub LXC:
#   0 */6 * * * /opt/vex-brain/deploy/rsync-cron.sh >> /var/log/vex-brain-import.log 2>&1
#
# Prerequisites:
#   - SSH key from MCPHub LXC to desktop (tailscale hostname)
#   - Host directory exists: mkdir -p ${CC_SESSIONS_PATH:-/opt/vex-brain/data/cc-sessions}
#   - VEX_BRAIN_DATABASE_URL set in environment or .env file

set -euo pipefail

DESKTOP_HOST="${VEX_BRAIN_DESKTOP_HOST:-desktop}"
REMOTE_USER="${VEX_BRAIN_REMOTE_USER:-shane}"
REMOTE_PATH="/home/${REMOTE_USER}/.claude/projects/"
LOCAL_PATH="${CC_SESSIONS_PATH:-/opt/vex-brain/data/cc-sessions/}"
DEPLOY_DIR="${VEX_BRAIN_DEPLOY_DIR:-/opt/vex-brain}"

echo "$(date -Iseconds) Starting CC JSONL sync..."

# Ensure local path exists
mkdir -p "$LOCAL_PATH"

# Rsync from desktop — only .jsonl files, exclude subagents
# Note: exclude order matters — subagents must come before the catch-all
rsync -avz --include='*/' --exclude='*/subagents/' --include='*.jsonl' --exclude='*' \
    "$DESKTOP_HOST:$REMOTE_PATH" "$LOCAL_PATH"

echo "$(date -Iseconds) Sync complete. Running import..."

# Run import via the CLI
cd "$DEPLOY_DIR"
docker compose exec -T vex-brain python -m app.cli import cc --source /data/cc-sessions/

echo "$(date -Iseconds) Import complete."
