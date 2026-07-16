#!/bin/bash
# ============================================================
# environment_setup.sh
#
# Mask2Former + DINOv2 — full environment setup for a fresh
# GPU instance (developed against Vast.ai, RTX 5090, CUDA 13.0,
# PyTorch 2.10 image — see launch_vastai_instance.sh).
#
# Usage: bash environment_setup.sh
# ============================================================

set -e

echo "============================================"
echo "  Mask2Former + DINOv2 Environment Setup"
echo "============================================"

# ── Step 1: System dependencies ──
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg

# ── Step 2: Python packages ──
echo "[2/7] Installing Python packages..."
pip install --break-system-packages --no-build-isolation \
  timm einops scipy opencv-python pycocotools \
  shapely wandb panopticapi

# ── Step 3: Detectron2 ──
echo "[3/7] Installing Detectron2 (compiles CUDA ops, ~5 min)..."
pip install --break-system-packages --no-build-isolation \
  git+https://github.com/facebookresearch/detectron2.git

# ── Step 4: Patch Detectron2 for PNG panoptic/semantic masks ──
# (Detectron2's COCO panoptic loader defaults to .jpg file extensions;
#  this pipeline exports masks as .png — see prepare_coco_dataset.py)
echo "[4/7] Patching Detectron2 for PNG image support..."
COCO_PAN=$(python3 -c "import detectron2.data.datasets.coco_panoptic as m; print(m.__file__)")

if [ -z "$COCO_PAN" ] || [ ! -f "$COCO_PAN" ]; then
    echo "ERROR: Could not locate detectron2's coco_panoptic.py. Aborting patch step."
    exit 1
fi

sed -i 's/os.path.splitext(ann\["file_name"\])\[0\] + ".jpg"/os.path.splitext(ann["file_name"])[0] + ".png"/g' "$COCO_PAN"
sed -i 's/load_sem_seg(sem_seg_root, image_root)/load_sem_seg(sem_seg_root, image_root, image_ext="png")/g' "$COCO_PAN"

echo "  Patched: $COCO_PAN"

# ── Step 5: Clone Mask2Former ──
M2F_DIR="${M2F_DIR:-/workspace/Mask2Former}"
echo "[5/7] Cloning Mask2Former to $M2F_DIR..."
git clone https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR"
cd "$M2F_DIR"
pip install --break-system-packages --no-build-isolation -r requirements.txt

# ── Step 6: Compile CUDA ops ──
echo "[6/7] Compiling MultiScaleDeformableAttention CUDA op..."
cd "$M2F_DIR/mask2former/modeling/pixel_decoder/ops"

# Patch deprecated PyTorch API (value.type() -> value.scalar_type())
sed -i 's/AT_DISPATCH_FLOATING_TYPES_AND_HALF(value\.type()/AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type()/g' \
  src/cuda/ms_deform_attn_cuda.cu
sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' \
  src/cuda/ms_deform_attn_cuda.cu

sh make.sh
echo "  CUDA op compiled successfully"

# ── Step 7: Download DINOv2 checkpoint ──
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
echo "[7/7] Downloading DINOv2 ViT-B/14 checkpoint (~331MB) to $CHECKPOINT_DIR..."
mkdir -p "$CHECKPOINT_DIR"
wget -q -O "$CHECKPOINT_DIR/dinov2_vitb14_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

echo ""
echo "============================================"
echo "  Setup complete."
echo "  Next: run scripts/prepare_coco_dataset.py, then scripts/train.sh"
echo "============================================"

# Verify environment
python3 -c "
import torch, detectron2, timm
print('PyTorch:', torch.__version__)
print('Detectron2:', detectron2.__version__)
print('Timm:', timm.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
else:
    print('WARNING: CUDA not available')
"
