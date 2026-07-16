# Mask2Former + DINOv2 — Semantic Segmentation of Drone Imagery (9-class)

Training pipeline for **semantic segmentation** of high-resolution drone / orthomosaic
imagery, using a **DINOv2 ViT-B/14** backbone with a **Mask2Former** head over a
**9-class** land-cover taxonomy.

**Best validation result: 74.45% mIoU** (checkpoint `model_best_v6.pth`, iteration 24 000).

> **Data:** This repository contains training code, configs, and the model patches only.
> The annotated dataset (imagery + masks) is private and not released. Trained weights are
> hosted separately on Hugging Face — see [Pretrained weights](#pretrained-weights).

---

## Overview

| | |
|---|---|
| Task | Semantic segmentation (`sem_seg` eval), **not** panoptic |
| Backbone | DINOv2 ViT-B/14 (`vit_base_patch14_dinov2`), full fine-tune — **no LoRA** |
| Head | Mask2Former (`MaskFormerHead`, MSDeformAttn pixel decoder, 200 queries, 10 decoder layers) |
| Classes | **9** (see taxonomy below) |
| Resolution | **1024px** (train & test), batch 6, 60 000 iters |
| Rare-class handling | `RareClassBalancedSampler` (65% rare/batch) + per-class CE weights + per-class Hungarian match costs + per-class Dice weights |
| Framework | Detectron2 + a Mask2Former fork with a DINOv2 backbone |
| Hardware | 1× RTX 5090, CUDA 13.0, PyTorch 2.10 |

## Class taxonomy (authoritative order — ids are 0-indexed)

| ID | Class | Approx. pixel share |
|----|-------|--------------------|
| 0 | building_rcc | 84.2% |
| 1 | building_tin | 10.8% |
| 2 | building_tiled | 1.7% |
| 3 | building_others | 1.2% |
| 4 | waterbody | 1.0% |
| 5 | overhead_tank | 0.6% |
| 6 | well | 0.04% |
| 7 | solar_panel | 0.14% |
| 8 | vehicle | 0.36% |

The class index in a prediction map corresponds directly to this table.
`255` is the ignore/unlabeled label. The model head outputs 10 logits (9 classes + a
"no object" slot, per Mask2Former).

> There are **no** `road_paved` / `road_mud` classes and **no** split vehicle subtypes
> (`vehicle_car`, `vehicle_truck`, …). The taxonomy is a single `vehicle` class and the
> 9 categories above. (Earlier documentation described a 15-class panoptic setup — that was
> never the trained model.)

## Repository structure

```
.
├── train_semantic_v6.py                  # the exact training entrypoint used
├── configs/
│   └── drone/
│       └── drone_dinov2_semantic_v6.yaml # the exact config used
├── mask2former_patches/                  # drop these into the Mask2Former fork
│   ├── config.py                         #   add_dinov2_config()
│   ├── matcher.py                        #   per-class Hungarian match cost patch
│   ├── criterion.py                      #   per-class Dice weight patch (+ NaN-safe dice)
│   └── backbone/
│       └── dinov2_vitadapter.py          #   build_dinov2_vitadapter_backbone
├── scripts/
│   ├── environment_setup.sh              # install deps + fork + patches + DINOv2 weights
│   ├── prepare_semantic_dataset.py       # COCO instances -> 9-class semantic PNG masks
│   ├── build_augmentation.py            # OPTIONAL offline aug used for the 74.45% result
│   ├── ortho_pipeline.py                 # end-to-end raw TIFF -> shapefile inference pipeline
│   └── train.sh                          # launcher (wraps train_semantic_v6.py)
├── requirements.txt
├── HF_MODEL_CARD.md
├── LICENSE
└── README.md
```

## What actually produces 74.45% (read this before reproducing)

Three custom pieces are **load-bearing** — remove any of them and the numbers change:

1. **DINOv2 backbone** (`mask2former_patches/config.py` + `backbone/dinov2_vitadapter.py`).
   Stock Mask2Former has no `build_dinov2_vitadapter_backbone` / `add_dinov2_config`.
2. **Per-class Hungarian matching cost** (`matcher.py`, patched). The matcher multiplies the
   class-matching cost per target class, pushing queries toward rare classes
   (well=10, overhead_tank=6, solar_panel=5, …).
3. **Per-class Dice weight** (`criterion.py`, patched, with a NaN-safe `dice_loss_per_sample`).
   Applies stronger mask-quality pressure on rare classes at the pixel level.

`train_semantic_v6.py` wires these up at runtime via `inject_weights_and_costs()`, which sets:

- **CE class weights** → `criterion.empty_weight`
  `[0.1, 0.8, 2.0, 0.84, 0.5, 5.0, 8.0, 7.237, 3.0]`
- **Per-class match costs** → `matcher.class_match_costs`
  `[1.0, 2.0, 1.5, 2.0, 2.0, 6.0, 10.0, 5.0, 3.0]`
- **Per-class Dice weights** → `criterion.per_class_dice_weights`
  `[1.0, 3.0, 2.0, 2.0, 1.0, 8.0, 10.0, 6.0, 4.0]`

`inject_weights_and_costs` only *sets attributes*; the patched `matcher.py`/`criterion.py`
are what actually *read* them. If you run against stock Mask2Former, these are silently ignored.

It also uses a **`RareClassBalancedSampler`** (defined in `train_semantic_v6.py`) that fills
65% of every batch with images containing a rare class (ids {1,5,6,7,8}).

## Setup

```bash
bash scripts/environment_setup.sh
```

Installs Detectron2 + Mask2Former, builds the deformable-attention CUDA op, copies the
`mask2former_patches/` into the fork, and downloads the DINOv2 ViT-B/14 checkpoint.

You must ensure the fork *exports* the patched symbols:
`mask2former/config.py` should expose `add_dinov2_config`, and
`mask2former/modeling/backbone/__init__.py` should import
`build_dinov2_vitadapter_backbone` from `dinov2_vitadapter`.

## Preparing your own dataset

This repo ships no data. Export annotations in **COCO instance format** (e.g. from CVAT)
and convert to the 9-class **semantic** PNG-mask layout the training script reads:

```bash
python scripts/prepare_semantic_dataset.py \
    --input_json /path/to/instances_default.json \
    --image_dir  /path/to/images \
    --output_dir ./data \
    --train_ratio 0.8
```

Produces:

```
data/
  images/{train,val}/
  semantic_masks_9cls/{train,val}/   # uint8 PNG, pixel value == class id (0..8), 255 = ignore
```

Category names are mapped onto the taxonomy by name; use `--name_map` to remap
(e.g. `{"car": "vehicle"}`).

### Optional: offline augmentation (used for the reported result)

The reported **74.45% mIoU** was obtained by training on an offline-augmented copy of the
train split. `scripts/build_augmentation.py` is the exact script that produced it — it reads
`images/train` + `semantic_masks_9cls/train` and writes
`images_augmented_v2/train` + `semantic_masks_9cls_aug_v2/train`, which is what
`train_semantic_v6.py` reads for the train split (validation is always the un-augmented split).

```bash
DRONE_DATA_ROOT=./data python scripts/build_augmentation.py
```

It rotates (90/180/270) **only** rare-class images, color-jitters all originals, and
CopyPastes rare-class instances onto background tiles.

> **Optional vs required — the distinction matters.** This augmentation is *optional*:
> training runs without it, and it is how the reported number was reached. The
> `mask2former_patches/` (DINOv2 backbone + per-class `matcher.py`/`criterion.py`) are
> *required* — without them the model architecture and per-class objectives differ and the
> result will not reproduce.

## Training

```bash
export DRONE_DATA_ROOT=./data
export M2F_ROOT=/path/to/Mask2Former
bash scripts/train.sh
```

Key hyperparameters (see `configs/drone/drone_dinov2_semantic_v6.yaml`):

| | |
|---|---|
| Optimizer | AdamW, base LR 5e-5 → 1e-6 (WarmupCosineLR), 3000 warmup iters |
| Backbone LR | ×0.1, with layer-wise decay 0.9 across ViT blocks |
| Batch / iters | `IMS_PER_BATCH: 6`, `MAX_ITER: 60000`, AMP on |
| Grad clip | full-model, `CLIP_VALUE: 0.3` |
| Loss weights | CLASS 4.0, MASK 5.0, DICE 5.0, NO_OBJECT 0.1 (× per-class factors above) |
| Eval | every 2000 iters; best-mIoU checkpoint saved as `model_best.pth` |

Best mIoU (74.45%) was reached at iteration 24 000 and not beaten through 60 000;
`model_best_v6.pth` is that checkpoint (a copy of `model_best.pth` at its peak — **not**
`model_final.pth`, which was 73.04%).

## Results

Per-class validation IoU at the best checkpoint (iter 24 000):

| Class | Val IoU |
|---|---|
| building_rcc | 96.74 |
| building_tin | 85.62 |
| building_tiled | 84.14 |
| building_others | 44.11 |
| waterbody | 99.10 |
| overhead_tank | 36.16 |
| well | 80.45 |
| solar_panel | 63.59 |
| vehicle | 80.09 |
| **mIoU** | **74.45** |

## Running inference

`scripts/ortho_pipeline.py` runs the full pipeline end-to-end on a raw drone orthomosaic:
GSD detection → resample to 5cm → tile → DINOv2 + Mask2Former inference → vectorize →
package as a shapefile/GeoPackage zip. This is the exact script used to produce production
outputs on real village orthomosaics.

**Requirements (beyond `requirements.txt`):** `rasterio`, `fiona`, `shapely`. The DINOv2
pretrained checkpoint (`dinov2_vitb14_pretrain.pth`) is not needed for inference — only for
training from scratch; the trained checkpoint is self-contained.

**Setup:**

1. Follow [Setup](#setup) to install Detectron2 + Mask2Former + `mask2former_patches/`.
2. Download `mask2former-dinov2-drone-seg-9cls.pth` from Hugging Face and place it (or point
   `WEIGHTS` at the top of `ortho_pipeline.py`) to match your setup.

**Usage:**

```bash
# Auto-detect source GSD from the GeoTIFF:
python3 scripts/ortho_pipeline.py --input village.tif

# Specify source GSD manually (metres), if auto-detection is unreliable:
python3 scripts/ortho_pipeline.py --input village.tif --gsd 0.035

# Full control:
python3 scripts/ortho_pipeline.py \
    --input   village.tif \
    --gsd     0.035 \
    --out     ./output/myvillage \
    --name    myvillage \
    --workers 48 \
    --min-px  500
```

- **Supported source GSD:** 2cm–10cm; always resampled to 5cm for inference (the model was
  trained on 3.5–5cm imagery).
- **Output:** `<out>/<name>_segmentation.zip` containing `<name>.gpkg` (all 9 classes,
  pre-styled for QGIS) plus per-class shapefiles (`.shp`/`.shx`/`.dbf`/`.cpg`/`.prj`) and a
  `README.txt`.
- **Resumable:** rerun the same command after an interruption — completed resampling and
  tiles are skipped automatically.

## Pretrained weights

The trained checkpoint (the training output `model_best_v6.pth`, iter 24 000) is hosted on
Hugging Face as **`mask2former-dinov2-drone-seg-9cls.pth`**:
**[huggingface.co/vivekyaadav/mask2former-dinov2-drone-segmentation](https://huggingface.co/vivekyaadav/mask2former-dinov2-drone-segmentation)** — see
[`HF_MODEL_CARD.md`](HF_MODEL_CARD.md) for inference usage.

## License

MIT — see [LICENSE](LICENSE). DINOv2 (Apache 2.0) and Mask2Former (MIT) upstream carry
their own licenses — see [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
and [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former).

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
