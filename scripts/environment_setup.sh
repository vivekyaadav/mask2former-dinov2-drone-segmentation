#!/bin/bash
# ============================================================
# environment_setup.sh
#
# Sets up a fresh GPU box to reproduce the v6 semantic run.
# Developed against RTX 5090 / CUDA 13.0 / PyTorch 2.10, but the
# steps are generic. Adjust the torch/detectron2 wheels for your CUDA.
#
# What it does:
#   1. Installs Python deps (requirements.txt) + detectron2.
#   2. Clones facebookresearch/Mask2Former into $M2F_ROOT and builds the
#      deformable-attention CUDA op.
#   3. Copies this repo's mask2former_patches/ INTO the fork — this adds the
#      DINOv2 backbone (config.py, backbone/dinov2_vitadapter.py) and the
#      per-class matcher.py / criterion.py patches. These are REQUIRED.
#   4. Downloads the DINOv2 ViT-B/14 pretrained checkpoint.
#
# Usage: bash scripts/environment_setup.sh
# ============================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M2F_ROOT="${M2F_ROOT:-/workspace/Mask2Former}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints}"

echo "[1/4] Python dependencies ..."
pip install -r "$REPO_DIR/requirements.txt"
# detectron2 is not on PyPI — build from source (match your torch/CUDA):
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "[2/4] Mask2Former fork -> $M2F_ROOT ..."
if [ ! -d "$M2F_ROOT" ]; then
    git clone https://github.com/facebookresearch/Mask2Former.git "$M2F_ROOT"
fi
( cd "$M2F_ROOT/mask2former/modeling/pixel_decoder/ops" && sh make.sh )

echo "[3/4] Installing DINOv2 backbone + per-class patches into the fork ..."
# These add build_dinov2_vitadapter_backbone + add_dinov2_config and the patched
# per-class matcher/criterion. Without them the run is a stock 9-class M2F run and
# will NOT reproduce 74.45%.
cp "$REPO_DIR/mask2former_patches/config.py"                 "$M2F_ROOT/mask2former/config.py"
cp "$REPO_DIR/mask2former_patches/matcher.py"               "$M2F_ROOT/mask2former/modeling/matcher.py"
cp "$REPO_DIR/mask2former_patches/criterion.py"             "$M2F_ROOT/mask2former/modeling/criterion.py"
cp "$REPO_DIR/mask2former_patches/backbone/dinov2_vitadapter.py" \
                                                             "$M2F_ROOT/mask2former/modeling/backbone/dinov2_vitadapter.py"
echo "  NOTE: ensure mask2former/modeling/backbone/__init__.py and mask2former/config.py"
echo "  export build_dinov2_vitadapter_backbone / add_dinov2_config (see README)."

echo "[4/4] DINOv2 ViT-B/14 pretrained checkpoint ..."
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/dinov2_vitb14_pretrain.pth" ]; then
    wget -O "$CKPT_DIR/dinov2_vitb14_pretrain.pth" \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
fi

echo "Done. Next: prepare data, then bash scripts/train.sh"
