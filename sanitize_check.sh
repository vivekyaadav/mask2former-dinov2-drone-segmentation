#!/bin/bash
# ============================================================
# Pre-publish sanitization check.
# Run INSIDE the repo folder before pushing to GitHub:
#   bash sanitize_check.sh
# ============================================================

fail=0

SELF="sanitize_check.sh"
echo "=== Personal file paths (/Users/...) ==="
if grep -rn "/Users/" . --exclude-dir=.git --exclude="$SELF" 2>/dev/null; then fail=1; fi

echo ""
echo "=== Dataset / annotation / imagery files that must NOT ship ==="
if find . -not -path './.git/*' \( -iname "*.tif" -o -iname "*.tiff" \
    -o -iname "instances_*.json" -o -iname "panoptic_*.json" \
    -o -iname "sem_seg_*.json" -o -iname "*coco*.json" \) 2>/dev/null | grep .; then fail=1; fi

echo ""
echo "=== Model weights (host on HF, not git) ==="
if find . -not -path './.git/*' \( -iname "*.pth" -o -iname "*.pt" -o -iname "*.ckpt" \) 2>/dev/null | grep .; then fail=1; fi

echo ""
echo "=== Large files (>50MB; GitHub hard limit 100MB) ==="
if find . -not -path './.git/*' -type f -size +50M 2>/dev/null | grep .; then fail=1; fi

echo ""
echo "=== Possible API keys / tokens / secrets ==="
if grep -rniE "(api[_-]?key|secret|password|bearer |wandb.{0,10}key|ssh-rsa|ssh-ed25519|AKIA[0-9A-Z]{16})" . --exclude-dir=.git --exclude="$SELF" --exclude=".gitignore" 2>/dev/null; then fail=1; fi

echo ""
echo "=== .env / credential files ==="
if find . -not -path './.git/*' \( -iname ".env*" -o -iname "*credentials*" \
    -o -iname "*.pem" -o -iname "id_rsa*" \) 2>/dev/null | grep .; then fail=1; fi

echo ""
echo "=== macOS / editor junk (.DS_Store, ._*, __pycache__) ==="
if find . -not -path './.git/*' \( -iname ".DS_Store" -o -iname "._*" \
    -o -iname "__pycache__" -o -iname "*.pyc" \) 2>/dev/null | grep .; then fail=1; fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "CLEAN — nothing flagged. Safe to review and push."
else
    echo "FLAGGED — review the items above before 'git add .'"
fi
exit $fail
