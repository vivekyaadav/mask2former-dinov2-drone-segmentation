#!/usr/bin/env python3
"""
build_augmentation.py — OPTIONAL offline augmentation of the train split.

This is the exact augmentation that produced the train set used for the reported
**74.45% mIoU** result (originally `fix_augmentation.py`; it writes the
`images_augmented_v2` / `semantic_masks_9cls_aug_v2` dirs that
`train_semantic_v6.py` reads for the train split).

Distinct from `mask2former_patches/` (matcher.py / criterion.py / DINOv2 backbone),
which are REQUIRED to reproduce the numbers. This augmentation is OPTIONAL — training
runs without it, but the reported result was obtained WITH it.

What it does (rare classes = ids 4,5,6,7,8 = waterbody, overhead_tank, well,
solar_panel, vehicle):
  1. Copy all originals through unchanged.
  2. Rotation (90/180/270) applied ONLY to images containing a rare class — this is
     the "fix" over the earlier all-images version, which was inflating building_rcc.
  3. Color jitter (brightness/contrast/saturation) on all originals; masks unchanged.
  4. CopyPaste: paste rare-class instances onto background images
     (per-class copy counts in CP_COPIES).

Paths come from DRONE_DATA_ROOT (default: container path). It reads
  <root>/images/train  +  <root>/semantic_masks_9cls/train
and (re)creates
  <root>/images_augmented_v2/train  +  <root>/semantic_masks_9cls_aug_v2/train

Usage:
    DRONE_DATA_ROOT=./data python scripts/build_augmentation.py
"""
import os, shutil, random
import numpy as np
from PIL import Image, ImageEnhance
from collections import defaultdict
from multiprocessing import Pool

random.seed(42)
np.random.seed(42)

DATA_ROOT   = os.environ.get("DRONE_DATA_ROOT", "/workspace/data")
SRC_IMG_DIR = os.path.join(DATA_ROOT, "images", "train")
SRC_MSK_DIR = os.path.join(DATA_ROOT, "semantic_masks_9cls", "train")
OUT_IMG_DIR = os.path.join(DATA_ROOT, "images_augmented_v2", "train")
OUT_MSK_DIR = os.path.join(DATA_ROOT, "semantic_masks_9cls_aug_v2", "train")

IGNORE       = 255
NUM_CLASSES  = 9
RARE_CLASSES = [4, 5, 6, 7, 8]   # waterbody, overhead_tank, well, solar_panel, vehicle
CLASS_NAMES  = [
    "building_rcc", "building_tin", "building_tiled", "building_others",
    "waterbody", "overhead_tank", "well", "solar_panel", "vehicle"
]
CP_COPIES = {4: 2, 5: 3, 6: 8, 7: 6, 8: 2}
WORKERS   = int(os.environ.get("AUG_WORKERS", "64"))


def process_original(fname):
    shutil.copy2(os.path.join(SRC_IMG_DIR, fname), OUT_IMG_DIR)
    shutil.copy2(os.path.join(SRC_MSK_DIR, fname), OUT_MSK_DIR)


def process_rotation(fname):
    """Only applied to rare-class images."""
    img  = Image.open(os.path.join(SRC_IMG_DIR, fname))
    msk  = Image.open(os.path.join(SRC_MSK_DIR, fname))
    stem = os.path.splitext(fname)[0]
    for angle in [90, 180, 270]:
        img.rotate(angle).save(os.path.join(OUT_IMG_DIR, f"{stem}_rot{angle}.png"))
        msk.rotate(angle).save(os.path.join(OUT_MSK_DIR, f"{stem}_rot{angle}.png"))


def process_jitter(fname):
    """Applied to all images — color only, mask unchanged."""
    rng  = random.Random(hash(fname))
    img  = Image.open(os.path.join(SRC_IMG_DIR, fname))
    msk  = Image.open(os.path.join(SRC_MSK_DIR, fname))
    stem = os.path.splitext(fname)[0]
    img  = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img  = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img  = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    img.save(os.path.join(OUT_IMG_DIR, f"{stem}_jitter.png"))
    msk.save(os.path.join(OUT_MSK_DIR, f"{stem}_jitter.png"))


def process_copypaste(args):
    src_fname, dst_fname, cls_id, copy_idx = args
    rng = random.Random(hash(src_fname + str(copy_idx)))

    src_arr = np.array(Image.open(os.path.join(SRC_IMG_DIR, src_fname)).convert("RGB"))
    src_m   = np.array(Image.open(os.path.join(SRC_MSK_DIR, src_fname)))
    dst_arr = np.array(Image.open(os.path.join(SRC_IMG_DIR, dst_fname)).convert("RGB")).copy()
    dst_m   = np.array(Image.open(os.path.join(SRC_MSK_DIR, dst_fname))).copy()

    cls_mask = (src_m == cls_id)
    if cls_mask.sum() < 100:
        return

    rows = np.any(cls_mask, axis=1)
    cols = np.any(cls_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h_patch = rmax - rmin
    w_patch = cmax - cmin

    if h_patch >= dst_arr.shape[0] or w_patch >= dst_arr.shape[1]:
        return

    r_off = rng.randint(0, dst_arr.shape[0] - h_patch - 1)
    c_off = rng.randint(0, dst_arr.shape[1] - w_patch - 1)

    patch_mask = cls_mask[rmin:rmax, cmin:cmax]
    dst_arr[r_off:r_off+h_patch, c_off:c_off+w_patch][patch_mask] = \
        src_arr[rmin:rmax, cmin:cmax][patch_mask]
    dst_m[r_off:r_off+h_patch, c_off:c_off+w_patch][patch_mask] = cls_id

    stem     = os.path.splitext(src_fname)[0]
    new_stem = f"{stem}_cp{CLASS_NAMES[cls_id][:3]}{copy_idx}"
    Image.fromarray(dst_arr).save(os.path.join(OUT_IMG_DIR, new_stem + ".png"))
    Image.fromarray(dst_m).save(os.path.join(OUT_MSK_DIR,  new_stem + ".png"))


def main():
    print("Cleaning output dirs...")
    for d in [OUT_IMG_DIR, OUT_MSK_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    all_files = sorted(f for f in os.listdir(SRC_IMG_DIR) if f.endswith(".png"))

    print(f"Indexing {len(all_files)} images...")
    class_to_files   = defaultdict(list)
    background_files = []
    rare_any_files   = []

    for fname in all_files:
        m = np.array(Image.open(os.path.join(SRC_MSK_DIR, fname)))
        has_rare = False
        for cls_id in RARE_CLASSES:
            if (m == cls_id).sum() > 500:
                class_to_files[cls_id].append(fname)
                has_rare = True
        if has_rare:
            rare_any_files.append(fname)
        bg = int((m == 0).sum()) + int((m == 1).sum())
        if bg / m.size > 0.4:
            background_files.append(fname)

    print(f"  Images with any rare class: {len(rare_any_files)}")
    for cls_id in RARE_CLASSES:
        print(f"  {CLASS_NAMES[cls_id]:<20}: {len(class_to_files[cls_id])}")
    print(f"  Background images: {len(background_files)}")

    with Pool(WORKERS) as pool:
        print(f"\nStep 1: Copying {len(all_files)} originals...")
        pool.map(process_original, all_files)

        print(f"\nStep 2: Rotation (RARE-CLASS IMAGES ONLY: {len(rare_any_files)})...")
        pool.map(process_rotation, rare_any_files)

        print(f"\nStep 3: Color jitter (all {len(all_files)} originals)...")
        pool.map(process_jitter, all_files)

        print(f"\nStep 4: CopyPaste ({WORKERS} workers)...")
        cp_args = []
        for cls_id, n_copies in CP_COPIES.items():
            for copy_idx in range(n_copies):
                for src_fname in class_to_files[cls_id]:
                    dst_fname = random.choice(background_files)
                    cp_args.append((src_fname, dst_fname, cls_id, copy_idx))
        pool.map(process_copypaste, cp_args)

    final = len(os.listdir(OUT_IMG_DIR))
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Original:   {len(all_files):>6}")
    print(f"  Final:      {final:>6}  (+{final - len(all_files)})")
    print(f"  Multiplier: {final / max(len(all_files), 1):.2f}x")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
