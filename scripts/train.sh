#!/bin/bash
# ============================================================
# train.sh
#
# Usage: bash train.sh [512|1024] [batch_size]
# Examples:
#   bash train.sh 512       -> 512px, default batch 16
#   bash train.sh 1024      -> 1024px, default batch 4
#   bash train.sh 512 8     -> 512px, batch 8
# ============================================================

set -e

RESOLUTION=${1:-512}
BATCH=${2:-"default"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M2F_DIR="${M2F_DIR:-/workspace/Mask2Former}"
LOG_DIR="/workspace/output/drone_${RESOLUTION}"

if [[ "$RESOLUTION" != "512" && "$RESOLUTION" != "1024" ]]; then
    echo "ERROR: Resolution must be 512 or 1024"
    exit 1
fi

CONFIG="$SCRIPT_DIR/../configs/drone/panoptic_${RESOLUTION}.yaml"
if [[ "$BATCH" == "default" ]]; then
    if [[ "$RESOLUTION" == "512" ]]; then
        BATCH=16
    else
        BATCH=4
    fi
fi

echo "============================================"
echo "  Launching Mask2Former Training"
echo "  Resolution : ${RESOLUTION}px"
echo "  Batch size : ${BATCH}"
echo "  Config     : $CONFIG"
echo "  Output     : $LOG_DIR"
echo "============================================"

# Step 1: Copy configs into the Mask2Former repo (so its _BASE_ paths resolve)
mkdir -p "$M2F_DIR/configs/drone"
cp "$SCRIPT_DIR"/../configs/drone/*.yaml "$M2F_DIR/configs/drone/"

# Step 2: Register the dataset in train_net.py
python3 "$SCRIPT_DIR/register_dataset.py"

# Step 3: Create output dir
mkdir -p "$LOG_DIR"

# Step 4: Launch training
cd "$M2F_DIR"
OMP_NUM_THREADS=4 python train_net.py \
    --config-file "configs/drone/panoptic_${RESOLUTION}.yaml" \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH "$BATCH" \
    OUTPUT_DIR "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "Training complete. Checkpoints saved to $LOG_DIR"
