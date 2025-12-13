#!/bin/bash
# 🌙 Moon Dev's Auto-Sync Script
# Pushes trading data to GitHub every 8 hours
#
# SETUP:
# 1. Make executable: chmod +x src/scripts/sync_to_github.sh
# 2. Add to crontab: crontab -e
#    0 */8 * * * cd ~/moondev_follow && ./src/scripts/sync_to_github.sh >> ~/sync_log.txt 2>&1
#
# Or run manually: ./src/scripts/sync_to_github.sh

cd ~/moondev_follow

echo "=============================================="
echo "🌙 Moon Dev Sync - $(date)"
echo "=============================================="

# Check if there are changes
if git diff --quiet && git diff --cached --quiet; then
    # Check for untracked files in data directories
    UNTRACKED=$(git status --porcelain src/data/polymarket_websearch* 2>/dev/null | wc -l)
    if [ "$UNTRACKED" -eq "0" ]; then
        echo "✅ No changes to sync"
        exit 0
    fi
fi

echo "📊 Changes detected, syncing..."

# Add only the trading data files
git add src/data/polymarket_websearch/*.csv 2>/dev/null
git add src/data/polymarket_websearch_v2/*.csv 2>/dev/null

# Check if anything was staged
if git diff --cached --quiet; then
    echo "✅ No CSV changes to commit"
    exit 0
fi

# Create commit with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "📊 Trading data sync - $TIMESTAMP

Auto-synced from AWS EC2:
- V1 consensus picks & predictions
- V2 edge-based picks & predictions
- Portfolio & trade history

🤖 Generated with Moon Dev Auto-Sync"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push

if [ $? -eq 0 ]; then
    echo "✅ Sync complete!"
else
    echo "❌ Push failed - check your GitHub credentials"
    exit 1
fi

echo "=============================================="
