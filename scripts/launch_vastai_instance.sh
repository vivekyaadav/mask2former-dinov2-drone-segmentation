#!/bin/bash
# ============================================================
# launch_vastai_instance.sh
#
# Creates and waits for a Vast.ai GPU instance, then prints
# the SSH command and next steps.
#
# Requires the `vastai` CLI, authenticated (vastai set api-key ...).
#
# Usage:   bash launch_vastai_instance.sh <OFFER_ID>
# Example: bash launch_vastai_instance.sh 32898229
#
# Search for offers first:
#   vastai search offers 'gpu_name=RTX_5090 reliability>0.99 disk_space>80 dph_total<1.0' --order dph_total
#
# Optional environment overrides:
#   DOCKER_IMAGE   (default: pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel)
#   DISK_SPACE_GB  (default: 80)
#   CUDA_ARCH_LIST (default: 12.0, matches RTX 5090 / Blackwell)
# ============================================================

set -e

OFFER_ID=$1
DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel}"
DISK_SPACE_GB="${DISK_SPACE_GB:-80}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"

if [ -z "$OFFER_ID" ]; then
    echo "ERROR: No offer ID provided"
    echo "Usage: bash launch_vastai_instance.sh <OFFER_ID>"
    echo ""
    echo "Search for offers first:"
    echo "  vastai search offers 'gpu_name=RTX_5090 reliability>0.99 disk_space>80 dph_total<1.0' --order dph_total"
    exit 1
fi

echo "============================================"
echo "  Creating Vast.ai Instance"
echo "  Offer ID    : $OFFER_ID"
echo "  Image       : $DOCKER_IMAGE"
echo "  Disk        : ${DISK_SPACE_GB}GB"
echo "============================================"

vastai create instance "$OFFER_ID" \
    --image "$DOCKER_IMAGE" \
    --env "-p 8888:8888 -p 22:22 --shm-size=16g -e CUDA_HOME=/usr/local/cuda -e TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST" \
    --disk "$DISK_SPACE_GB" \
    --jupyter \
    --ssh \
    --direct

echo "Instance created. Polling for running status..."

while true; do
    STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
try:
    instances = json.load(sys.stdin)
    print(instances[-1].get('actual_status', 'unknown') if instances else 'unknown')
except Exception:
    print('unknown')
" 2>/dev/null)

    echo "  Status: $STATUS"

    if [ "$STATUS" = "running" ]; then
        break
    elif [[ "$STATUS" == *"error"* ]] || [[ "$STATUS" == *"Error"* ]]; then
        echo "ERROR: Instance failed. Try a different offer ID."
        vastai show instances
        exit 1
    fi
    sleep 15
done

INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
instances = json.load(sys.stdin)
print(instances[-1]['id'] if instances else '')
")

SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f2)

echo ""
echo "============================================"
echo "  Instance ready"
echo "  Instance ID : $INSTANCE_ID"
echo ""
echo "  ssh -i $SSH_IDENTITY -o ServerAliveInterval=30 -o TCPKeepAlive=yes root@$SSH_HOST -p $SSH_PORT"
echo ""
echo "  Next steps:"
echo "  1. SSH into the instance"
echo "  2. Upload this repo:"
echo "     rsync -avz --partial -e 'ssh -i $SSH_IDENTITY -p $SSH_PORT' ./ root@$SSH_HOST:/workspace/training/"
echo "  3. Run: bash /workspace/training/scripts/environment_setup.sh"
echo "  4. Upload your dataset, then run scripts/prepare_coco_dataset.py"
echo "  5. Run: bash /workspace/training/scripts/train.sh 512"
echo "============================================"

# Update local SSH config (only touches the 'vast' host entry)
if grep -q "^Host vast$" ~/.ssh/config 2>/dev/null; then
    sed -i.bak "/^Host vast$/,/^$/ s/HostName .*/HostName $SSH_HOST/" ~/.ssh/config
    sed -i.bak "/^Host vast$/,/^$/ s/Port .*/Port $SSH_PORT/" ~/.ssh/config
    rm -f ~/.ssh/config.bak
    echo "Updated existing 'vast' entry in ~/.ssh/config"
else
    cat >> ~/.ssh/config << EOF

Host vast
    HostName $SSH_HOST
    Port $SSH_PORT
    User root
    IdentityFile $SSH_IDENTITY
    ServerAliveInterval 20
    ServerAliveCountMax 999
    TCPKeepAlive yes
EOF
    echo "Added 'vast' entry to ~/.ssh/config"
fi

echo "Connect with: ssh vast"
