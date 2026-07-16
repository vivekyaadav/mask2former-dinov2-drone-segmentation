#!/usr/bin/env python3
"""
prepare_coco_dataset.py

Converts a CVAT COCO-1.0 instance-segmentation export into the
COCO panoptic + semantic format expected by Mask2Former:

    output_dir/
      images/{train,val}/
      annotations/
        panoptic_{train,val}.json
        panoptic_{train,val}/        (RGB segment-ID masks, PNG)
        sem_seg_{train,val}/         (grayscale class-ID masks, PNG)
        instances_{train,val}.json

Usage:
    python prepare_coco_dataset.py \\
        --input_json /path/to/instances_default.json \\
        --image_dir  /path/to/images \\
        --output_dir /workspace/data \\
        --train_ratio 0.8

Note: "stuff" classes (regions without individual instances, e.g. roads,
water bodies) are set via --stuff_classes. Defaults match this project's
taxonomy — override for your own dataset.
"""

import json
import os
import argparse
import random
import shutil
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def polygons_to_mask(segmentation, height, width):
    rles = coco_mask.frPyObjects(segmentation, height, width)
    rle = coco_mask.merge(rles)
    return coco_mask.decode(rle).astype(bool)


def process_split(split_imgs, data, image_dir, output_dir, split_name, cat_id2name, stuff_classes):
    img2anns = {}
    for ann in data["annotations"]:
        img2anns.setdefault(ann["image_id"], []).append(ann)

    panoptic_anns = []
    coco_images = []
    skipped = 0

    os.makedirs(f"{output_dir}/images/{split_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/annotations/panoptic_{split_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/annotations/sem_seg_{split_name}", exist_ok=True)

    for img_info in split_imgs:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        anns = img2anns.get(img_id, [])
        if not anns:
            skipped += 1
            continue

        src = os.path.join(image_dir, fname)
        if not os.path.exists(src):
            print(f"  Warning: missing image {fname}")
            skipped += 1
            continue

        shutil.copy(src, f"{output_dir}/images/{split_name}/{fname}")
        coco_images.append(img_info)

        panoptic_mask = np.zeros((height, width, 3), dtype=np.uint8)
        semantic_mask = np.zeros((height, width), dtype=np.uint8)

        segments = []
        for seg_id, ann in enumerate(anns, start=1):
            cat_id = ann["category_id"]
            cat_name = cat_id2name[cat_id]
            is_stuff = cat_name in stuff_classes

            # Encode segment ID as RGB (supports up to 2^24 segments/image)
            r = seg_id & 0xFF
            g = (seg_id >> 8) & 0xFF
            b = (seg_id >> 16) & 0xFF

            bin_mask = polygons_to_mask(ann["segmentation"], height, width)
            panoptic_mask[bin_mask] = [r, g, b]
            semantic_mask[bin_mask] = cat_id

            segments.append({
                "id": seg_id,
                "category_id": cat_id,
                "iscrowd": 0,
                "isthing": 0 if is_stuff else 1,
                "area": int(bin_mask.sum()),
                "bbox": ann["bbox"],
            })

        Image.fromarray(panoptic_mask).save(f"{output_dir}/annotations/panoptic_{split_name}/{fname}")
        Image.fromarray(semantic_mask).save(f"{output_dir}/annotations/sem_seg_{split_name}/{fname}")

        panoptic_anns.append({
            "image_id": img_id,
            "file_name": fname,
            "segments_info": segments,
        })

    with open(f"{output_dir}/annotations/panoptic_{split_name}.json", "w") as f:
        json.dump({
            "images": coco_images,
            "annotations": panoptic_anns,
            "categories": data["categories"],
        }, f)

    valid_ids = {i["id"] for i in coco_images}
    with open(f"{output_dir}/annotations/instances_{split_name}.json", "w") as f:
        json.dump({
            "images": coco_images,
            "annotations": [a for a in data["annotations"] if a["image_id"] in valid_ids],
            "categories": data["categories"],
        }, f)

    print(f"  {split_name}: {len(coco_images)} images, {len(panoptic_anns)} annotations (skipped {skipped})")
    return len(coco_images)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_json", required=True, help="Path to instances_default.json from CVAT")
    parser.add_argument("--image_dir", required=True, help="Path to the images folder")
    parser.add_argument("--output_dir", default="/workspace/data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stuff_classes",
        nargs="*",
        default=["road_paved", "road_mud", "waterbody"],
        help="Category names treated as 'stuff' (no individual instances). "
             "Default matches this project's 15-class taxonomy — override for your own dataset.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train_ratio must be between 0 and 1")

    print(f"Loading annotations from {args.input_json}...")
    with open(args.input_json) as f:
        data = json.load(f)

    categories = data["categories"]
    cat_id2name = {c["id"]: c["name"] for c in categories}
    stuff_classes = set(args.stuff_classes)

    unknown_stuff = stuff_classes - set(cat_id2name.values())
    if unknown_stuff:
        print(f"  Warning: --stuff_classes not found in dataset categories: {sorted(unknown_stuff)}")

    print(f"Dataset: {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"Categories: {[c['name'] for c in categories]}")

    images = data["images"]
    random.seed(args.seed)
    random.shuffle(images)
    split_idx = int(len(images) * args.train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val")

    os.makedirs(args.output_dir, exist_ok=True)
    process_split(train_imgs, data, args.image_dir, args.output_dir, "train", cat_id2name, stuff_classes)
    process_split(val_imgs, data, args.image_dir, args.output_dir, "val", cat_id2name, stuff_classes)

    print(f"\nDataset prepared at: {args.output_dir}")
    print("Structure:")
    print(f"  {args.output_dir}/images/{{train,val}}/")
    print(f"  {args.output_dir}/annotations/panoptic_{{train,val}}.json")
    print(f"  {args.output_dir}/annotations/instances_{{train,val}}.json")
    print(f"  {args.output_dir}/annotations/panoptic_{{train,val}}/  (RGB masks)")
    print(f"  {args.output_dir}/annotations/sem_seg_{{train,val}}/   (semantic masks)")


if __name__ == "__main__":
    main()
