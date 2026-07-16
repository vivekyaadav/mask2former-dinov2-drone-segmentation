#!/usr/bin/env python3
"""
prepare_semantic_dataset.py

Convert a COCO instance-segmentation export (e.g. from CVAT COCO-1.0) into the
grayscale semantic-mask layout that `train_semantic_v6.py` expects:

    output_dir/
      images/{train,val}/                     RGB tiles (copied)
      semantic_masks_9cls/{train,val}/        uint8 PNG, pixel value == class id (0..8),
                                              255 = ignore / unlabeled

This matches the ACTUAL 9-class taxonomy that produced model_best_v6.pth
(74.45% val mIoU). It is a *semantic* pipeline — there is no panoptic JSON and no
segment-id RGB masks; Detectron2's `sem_seg` evaluator reads the PNG masks directly
via `sem_seg_file_name`.

Class ids are 0-indexed and MUST match CLASS_NAMES in train_semantic_v6.py:

    0 building_rcc      3 building_others   6 well
    1 building_tin      4 waterbody         7 solar_panel
    2 building_tiled    5 overhead_tank     8 vehicle

Your source COCO `category_id`s are mapped to these 0..8 ids by category *name*
(see --name_map to remap names). Any pixel not covered by an annotation stays 255
(ignored in the loss and in mIoU), which is what the training run used.

Augmentation (the real run trained on images_augmented_v2 / semantic_masks_9cls_aug_v2)
is intentionally NOT part of this script — offline augmentation is dataset-specific
and is left to you. Point DRONE_DATA_ROOT at whatever you produce; the val split here
is the un-augmented one the model was evaluated on.

Usage:
    python scripts/prepare_semantic_dataset.py \
        --input_json /path/to/instances_default.json \
        --image_dir  /path/to/images \
        --output_dir ./data \
        --train_ratio 0.8

No absolute/personal paths are baked in — everything comes from the arguments.
"""

import argparse
import json
import os
import random
import shutil

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

# 0-indexed class ids — order is authoritative and matches train_semantic_v6.py.
CLASS_NAMES = [
    "building_rcc",     # 0
    "building_tin",     # 1
    "building_tiled",   # 2
    "building_others",  # 3
    "waterbody",        # 4
    "overhead_tank",    # 5
    "well",             # 6
    "solar_panel",      # 7
    "vehicle",          # 8
]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
IGNORE_LABEL = 255

# Pixel-frequency ordering (rarest last) — larger classes are painted first so that
# small/rare classes drawn afterwards win overlaps and are not swallowed. This is a
# reasonable default for top-down imagery where e.g. a vehicle sits on a building roof.
PAINT_ORDER = [
    "building_rcc", "building_tin", "building_tiled", "building_others",
    "waterbody", "overhead_tank", "solar_panel", "vehicle", "well",
]


def polygons_to_mask(segmentation, height, width):
    """COCO polygon / RLE segmentation -> boolean HxW mask."""
    if isinstance(segmentation, list):  # polygons
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation["counts"], list):  # uncompressed RLE
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:  # compressed RLE
        rle = segmentation
    return coco_mask.decode(rle).astype(bool)


def build_id_lookup(categories, name_map):
    """Map source COCO category_id -> target 0..8 id, via (optionally remapped) name."""
    src_id_to_target = {}
    unknown = set()
    for c in categories:
        name = name_map.get(c["name"], c["name"])
        if name in NAME_TO_ID:
            src_id_to_target[c["id"]] = NAME_TO_ID[name]
        else:
            unknown.add(c["name"])
    if unknown:
        print(f"  Note: {sorted(unknown)} not in the 9-class taxonomy — "
              f"their pixels stay 255 (ignored). Use --name_map to remap.")
    return src_id_to_target


def process_split(split_imgs, img2anns, image_dir, output_dir, split_name,
                  src_id_to_target):
    img_out = os.path.join(output_dir, "images", split_name)
    mask_out = os.path.join(output_dir, "semantic_masks_9cls", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    paint_rank = {NAME_TO_ID[n]: r for r, n in enumerate(PAINT_ORDER)}
    n_written = n_skipped = 0

    for img_info in split_imgs:
        fname = img_info["file_name"]
        width, height = img_info["width"], img_info["height"]
        src = os.path.join(image_dir, fname)
        if not os.path.exists(src):
            print(f"  Warning: missing image {fname} — skipped")
            n_skipped += 1
            continue

        anns = img2anns.get(img_info["id"], [])
        anns = [a for a in anns if a["category_id"] in src_id_to_target]
        if not anns:
            n_skipped += 1
            continue

        # Paint common classes first, rare classes last, so rare wins overlaps.
        anns.sort(key=lambda a: paint_rank[src_id_to_target[a["category_id"]]])

        sem = np.full((height, width), IGNORE_LABEL, dtype=np.uint8)
        for ann in anns:
            cls_id = src_id_to_target[ann["category_id"]]
            m = polygons_to_mask(ann["segmentation"], height, width)
            sem[m] = cls_id

        stem = os.path.splitext(os.path.basename(fname))[0]
        shutil.copy(src, os.path.join(img_out, os.path.basename(fname)))
        Image.fromarray(sem).save(os.path.join(mask_out, stem + ".png"))
        n_written += 1

    print(f"  {split_name}: {n_written} written, {n_skipped} skipped")
    return n_written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_json", required=True,
                    help="COCO instance-segmentation JSON (e.g. instances_default.json)")
    ap.add_argument("--image_dir", required=True, help="Folder with the source images")
    ap.add_argument("--output_dir", default="./data",
                    help="Output root (default: ./data). Point DRONE_DATA_ROOT here for training.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name_map", default=None,
                    help='Optional JSON: {"src_name": "taxonomy_name"} to remap category '
                         'names onto the 9-class taxonomy (e.g. {"car":"vehicle"}).')
    args = ap.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        ap.error("--train_ratio must be between 0 and 1")

    name_map = {}
    if args.name_map:
        with open(args.name_map) as f:
            name_map = json.load(f)

    print(f"Loading {args.input_json} ...")
    with open(args.input_json) as f:
        data = json.load(f)

    src_id_to_target = build_id_lookup(data["categories"], name_map)
    if not src_id_to_target:
        ap.error("No source categories matched the 9-class taxonomy. "
                 "Check category names or provide --name_map.")

    img2anns = {}
    for ann in data["annotations"]:
        img2anns.setdefault(ann["image_id"], []).append(ann)

    images = list(data["images"])
    random.seed(args.seed)
    random.shuffle(images)
    k = int(len(images) * args.train_ratio)
    train_imgs, val_imgs = images[:k], images[k:]
    print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val")

    os.makedirs(args.output_dir, exist_ok=True)
    process_split(train_imgs, img2anns, args.image_dir, args.output_dir, "train", src_id_to_target)
    process_split(val_imgs, img2anns, args.image_dir, args.output_dir, "val", src_id_to_target)

    print(f"\nDone. Semantic dataset at: {args.output_dir}")
    print("  images/{train,val}/")
    print("  semantic_masks_9cls/{train,val}/   (uint8 PNG, value == class id, 255 = ignore)")
    print("\nNext: export DRONE_DATA_ROOT=<output_dir> and run scripts/train.sh")


if __name__ == "__main__":
    main()
