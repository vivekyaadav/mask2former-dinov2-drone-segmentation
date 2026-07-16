---
license: mit
tags:
  - image-segmentation
  - semantic-segmentation
  - mask2former
  - dinov2
  - remote-sensing
  - drone-imagery
  - pytorch
pipeline_tag: image-segmentation
library_name: pytorch
---

# Mask2Former-DINOv2 — Drone Semantic Segmentation (9-class)

A **Mask2Former** semantic segmentation model with a **DINOv2 ViT-B/14** backbone,
full fine-tuned (no LoRA) for **9-class** land-cover segmentation of high-resolution
drone / orthomosaic imagery.

- **Best validation mIoU:** 74.45% (checkpoint `mask2former-dinov2-drone-seg-9cls.pth`, iter 24 000)
- **Input:** RGB image tiles at **1024px**
- **Output:** Per-pixel semantic class map (9 classes; head emits 9 + "no object")
- **Backbone:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) ViT-B/14 (`vit_base_patch14_dinov2`)
- **Architecture:** [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) + DINOv2 backbone
- **Training code:** https://github.com/vivekyaadav/mask2former-dinov2-drone-segmentation

> Trained on a private, custom-annotated dataset. The dataset is not released.

## Class taxonomy (0-indexed)

| ID | Class | | ID | Class |
|----|-------|--|----|-------|
| 0 | building_rcc | | 5 | overhead_tank |
| 1 | building_tin | | 6 | well |
| 2 | building_tiled | | 7 | solar_panel |
| 3 | building_others | | 8 | vehicle |
| 4 | waterbody | | | |

The predicted class-index map maps directly to this table; `255` = ignore.

## Intended use

Semantic segmentation of urban/rural land-cover — building types, water bodies, and small
infrastructure (tanks, wells, solar panels, vehicles) — from top-down drone or orthomosaic
imagery. Research / GIS prototyping only; not validated for safety-critical use.

## How to use

This is a Detectron2 + Mask2Former model with a **custom DINOv2 backbone and patched
matcher/criterion**. You need the training repo (for the config and the patched fork), not
just stock Mask2Former.

```bash
# 1. Set up the fork + patches (see the training repo's environment_setup.sh)
# 2. Download the checkpoint
python - <<'PY'
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="vivekyaadav/mask2former-dinov2-drone-segmentation",
    filename="mask2former-dinov2-drone-seg-9cls.pth")
print(ckpt)
PY
```

```python
import cv2, torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from mask2former.config import add_dinov2_config   # from the patched fork

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_dinov2_config(cfg)
cfg.merge_from_file("configs/drone/drone_dinov2_semantic_v6.yaml")  # from the training repo
cfg.MODEL.WEIGHTS = ckpt
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
image = cv2.imread("your_tile.png")          # BGR; 1024px recommended
sem = predictor(image)["sem_seg"].argmax(0).cpu().numpy()  # HxW class-index map (0..8)
```

## Training details

| | |
|---|---|
| Backbone | DINOv2 ViT-B/14, full fine-tune (no LoRA) |
| Head | Mask2Former, semantic mode, 200 queries, 10 decoder layers |
| Resolution | 1024px (train & test) |
| Batch / iters | 6 / 60 000 (AdamW, LR 5e-5 cosine → 1e-6, warmup 3000) |
| Rare-class handling | RareClassBalancedSampler 65% + per-class CE / match-cost / Dice weights |
| Framework | Detectron2, PyTorch 2.10 |
| Hardware | 1× RTX 5090 |
| Best val mIoU | 74.45% (iter 24 000) |

## Limitations

- Trained on a specific regional distribution; performance on other geographies, sensors,
  or resolutions is untested.
- Dataset size and class balance are not disclosed; treat performance as indicative.
- Heavy class imbalance (building_rcc ≈ 84% of pixels); rare classes such as overhead_tank
  remain comparatively weak (IoU ≈ 36).

## Citation

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{cheng2022mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Cheng, Bowen and others},
  booktitle={CVPR},
  year={2022}
}
```

## License

MIT. Underlying DINOv2 (Apache 2.0) and Mask2Former (MIT) licenses apply to those components.
