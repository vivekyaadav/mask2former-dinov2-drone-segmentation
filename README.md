# Mask2Former + DINOv2 — Semantic Segmentation of Indian Urban Drone Imagery

Training pipeline for panoptic/semantic segmentation of high-resolution drone orthomosaic imagery, using a **DINOv2 ViT-B/14** backbone with a **Mask2Former** segmentation head over a 15-class urban land-cover taxonomy.

Best validation result: **74.45% mIoU** (Run 6, 512px).

> **Note on data:** This repository contains training code and configuration only. The annotated dataset (imagery and COCO annotations) is not included and is not publicly released. Trained weights are hosted on Hugging Face — see [Pretrained weights](#pretrained-weights).

---

## Overview

| | |
|---|---|
| Backbone | DINOv2 ViT-B/14 (self-supervised pretrained) |
| Head | Mask2Former (panoptic dataset registration, semantic eval) |
| Classes | 15 (4 building types, 3 stuff surfaces, 8 thing categories) |
| Domain | High-resolution orthomosaic / drone imagery, Indian urban scenes |
| Framework | Detectron2 + Mask2Former (Facebook Research) |
| Training hardware | 1x RTX 5090, CUDA 13.0, PyTorch 2.10 (Vast.ai) |

## Repository structure

```
.
├── configs/
│   └── drone/
│       ├── panoptic_512.yaml        # 512px training config
│       └── panoptic_1024.yaml       # 1024px training config
├── scripts/
│   ├── environment_setup.sh         # environment setup (fresh GPU instance)
│   ├── launch_vastai_instance.sh    # provision a Vast.ai GPU instance
│   ├── prepare_coco_dataset.py      # CVAT COCO -> COCO panoptic/semantic converter
│   ├── register_dataset.py          # registers the dataset with Mask2Former's train_net.py
│   └── train.sh                     # training launcher
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
bash scripts/environment_setup.sh
```

Installs Detectron2 and Mask2Former, compiles the deformable-attention CUDA op, patches Detectron2's panoptic loader for PNG masks, and downloads the DINOv2 ViT-B/14 pretrained checkpoint.

## Preparing your own dataset

This repo does not include data. To train on your own dataset, export annotations in **COCO instance format** (e.g. from CVAT) and convert to the panoptic/semantic format Mask2Former expects:

```bash
python scripts/prepare_coco_dataset.py \
    --input_json /path/to/instances_default.json \
    --image_dir  /path/to/images \
    --output_dir /workspace/data \
    --train_ratio 0.8 \
    --stuff_classes road_paved road_mud waterbody
```

`--stuff_classes` defaults to this project's taxonomy — override with your own category names for a different dataset.

## Dataset taxonomy (this project)

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

## Provisioning a GPU instance (Vast.ai)

```bash
vastai search offers 'gpu_name=RTX_5090 reliability>0.99 disk_space>80 dph_total<1.0' --order dph_total
bash scripts/launch_vastai_instance.sh <OFFER_ID>
```

Upload this repo and your prepared dataset to the instance, then run `environment_setup.sh` remotely.

## Training

```bash
bash scripts/train.sh 512        # 512px, default batch size 16
bash scripts/train.sh 1024 4     # 1024px, batch size 4
```

Checkpoints and logs are written to `/workspace/output/drone_<resolution>/`.

## Results

| Run | Resolution | Val mIoU |
|-----|-----------|----------|
| Run 6 (best) | 512px | **74.45%** |

## Pretrained weights

Trained weights for Run 6 are available on Hugging Face:
**[huggingface.co/YOUR_USERNAME/mask2former-dinov2-drone-segmentation](#)**

See the model card there for inference usage.

## License

MIT — see [LICENSE](LICENSE). The DINOv2 and Mask2Former upstream repositories carry their own licenses (Apache 2.0 / MIT respectively) — see [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) and [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former).

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
