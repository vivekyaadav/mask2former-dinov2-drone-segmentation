#!/bin/bash
# ============================================================
# train.sh — launch the ACTUAL v6 semantic run
#
# Reproduces model_best_v6.pth (74.45% val mIoU):
#   DINOv2 ViT-B/14 + Mask2Former, full fine-tune (no LoRA),
#   9-class semantic segmentation at 1024px, batch 6, 60k iters,
#   RareClassBalancedSampler + per-class matching/Dice weights.
#
# Prerequisites (see README "Setup"):
#   1. A clone of the Mask2Former fork with the DINOv2 backbone at $M2F_ROOT.
#   2. The mask2former_patches/ files copied into that fork (matcher.py,
#      criterion.py, config.py, backbone/dinov2_vitadapter.py) — WITHOUT them
#      the per-class costs/Dice are silently ignored and results won't match.
#   3. Data prepared by scripts/prepare_semantic_dataset.py at $DRONE_DATA_ROOT.
#   4. dinov2_vitb14_pretrain.pth at $DINOV2_WEIGHTS.
#
# Usage: bash scripts/train.sh
# ============================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M2F_ROOT="${M2F_ROOT:-/workspace/Mask2Former}"
DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/workspace/data}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-/workspace/checkpoints/dinov2_vitb14_pretrain.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/drone_dinov2_semantic_v6}"

echo "============================================"
echo "  Mask2Former + DINOv2 — semantic v6"
echo "  M2F_ROOT        : $M2F_ROOT"
echo "  DRONE_DATA_ROOT : $DRONE_DATA_ROOT"
echo "  DINOV2_WEIGHTS  : $DINOV2_WEIGHTS"
echo "  OUTPUT_DIR      : $OUTPUT_DIR"
echo "============================================"

# Make the v6 config resolvable from inside the fork.
mkdir -p "$M2F_ROOT/configs/drone"
cp "$REPO_DIR/configs/drone/drone_dinov2_semantic_v6.yaml" "$M2F_ROOT/configs/drone/"
mkdir -p "$OUTPUT_DIR"

cd "$M2F_ROOT"
export M2F_ROOT DRONE_DATA_ROOT
OMP_NUM_THREADS=4 python "$REPO_DIR/train_semantic_v6.py" \
    --config-file "configs/drone/drone_dinov2_semantic_v6.yaml" \
    --num-gpus 1 \
    MODEL.WEIGHTS "$DINOV2_WEIGHTS" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Done. Best checkpoint: $OUTPUT_DIR/model_best.pth"
echo "(The published model_best_v6.pth is a copy of model_best.pth at its peak.)"
