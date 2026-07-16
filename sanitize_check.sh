#!/bin/bash
# ============================================================
# Pre-publish sanitization check for mask2former_training repo
# Run this INSIDE the repo folder before pushing to GitHub
# Usage: bash sanitize_check.sh
# ============================================================

echo "=== Checking for personal file paths ==="
grep -rn "/Users/vivekyadav" . --exclude-dir=.git 2>/dev/null

echo ""
echo "=== Checking for dataset files that should NOT be uploaded ==="
find . -iname "*.tif" -o -iname "*.tiff" -o -iname "instances_*.json" -o -iname "panoptic_*.json" -o -iname "*coco*" 2>/dev/null | grep -v ".git"

echo ""
echo "=== Checking for large files (>50MB) - GitHub hard limit is 100MB ==="
find . -type f -size +50M 2>/dev/null | grep -v ".git"

echo ""
echo "=== Checking for possible API keys / tokens / secrets ==="
grep -rniE "(api[_-]?key|secret|token|password|wandb.*key|ssh-rsa|ssh-ed25519|AKIA[0-9A-Z]{16})" . --exclude-dir=.git 2>/dev/null

echo ""
echo "=== Checking for .env or credential files ==="
find . -iname ".env*" -o -iname "*credentials*" -o -iname "*.pem" -o -iname "id_rsa*" 2>/dev/null | grep -v ".git"

echo ""
echo "=== Checking for empty/junk dirs (.DS_Store, __pycache__) ==="
find . -iname ".DS_Store" -o -iname "__pycache__" -o -iname "*.pyc" 2>/dev/null | grep -v ".git"

echo ""
echo "=== Done. Review everything above before 'git add .' ==="
