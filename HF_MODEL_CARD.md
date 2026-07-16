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

# Mask2Former-DINOv2 — Indian Urban Drone Semantic Segmentation

A **Mask2Former** semantic segmentation model with a **DINOv2 ViT-B/14** backbone, fine-tuned for 15-class land-cover segmentation of high-resolution Indian urban drone/orthomosaic imagery.

- **Best validation mIoU:** 74.45%
- **Input:** RGB image tiles (512px, orthomosaic-derived)
- **Output:** Per-pixel semantic class map (15 classes)
- **Backbone:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) ViT-B/14
- **Architecture:** [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)
- **Training code:** [link to your GitHub repo]

> This model was trained on a private, custom-annotated dataset. The dataset itself is not released with this model.

## Intended use

Semantic segmentation of urban land-cover categories — building types, road surfaces, water bodies, vehicles, and small infrastructure (tanks, wells, solar panels) — from top-down drone or orthomosaic imagery, primarily over Indian urban environments. Intended for research and GIS/urban-planning prototyping. Not validated for safety-critical or production geospatial decision-making.

## How to use

### 1. Install dependencies

```bash
pip install torch torchvision detectron2 huggingface_hub
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former && pip install -r requirements.txt
```

### 2. Download the checkpoint

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="YOUR_HF_USERNAME/mask2former-dinov2-drone-segmentation",
    filename="model_best_v6.pth"
)
```

Or via CLI:

```bash
huggingface-cli download YOUR_HF_USERNAME/mask2former-dinov2-drone-segmentation model_best_v6.pth --local-dir .
```

### 3. Run inference

```python
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/drone/panoptic_512.yaml")   # from the training repo
cfg.MODEL.WEIGHTS = ckpt_path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

import cv2
image = cv2.imread("your_tile.png")  # BGR, 512x512 recommended
outputs = predictor(image)
sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()  # HxW class-index map
```

## Class taxonomy

| ID | Class | Type |
|----|-------|------|
| 1 | building_rcc | thing |
| 2 | building_tin | thing |
| 3 | building_tiled | thing |
| 4 | building_others | thing |
| 5 | road_paved | stuff |
| 6 | road_mud | stuff |
| 7 | waterbody | stuff |
| 8 | overhead_tank | thing |
| 9 | well | thing |
| 10 | solar_panel | thing |
| 11 | vehicle_car | thing |
| 12 | vehicle_truck | thing |
| 13 | vehicle_two_wheeler | thing |
| 14 | vehicle_bus | thing |
| 15 | vehicle_others | thing |

## Training details

| | |
|---|---|
| Backbone | DINOv2 ViT-B/14 |
| Head | Mask2Former (semantic mode) |
| Resolution | 512px |
| Framework | Detectron2 |
| Hardware | 1x RTX 5090 |
| Best val mIoU | 74.45% (Run 6) |

## Limitations

- Trained on a specific regional distribution (Indian urban orthomosaic imagery); performance on other geographies, sensors, or resolutions is untested.
- Dataset size and class balance are not disclosed; treat performance claims as indicative, not benchmarked against a public standard.
- Not evaluated for adversarial robustness or edge-case terrain (e.g. informal settlements, mixed-use zones) beyond what's in the private validation set.

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
