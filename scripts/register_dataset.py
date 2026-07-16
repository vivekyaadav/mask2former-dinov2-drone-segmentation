#!/usr/bin/env python3
"""
register_dataset.py

Patches Mask2Former's train_net.py to:
1. Register the drone panoptic dataset (train/val splits)
2. Fix width/height metadata for evaluation
3. Insert the registration block at the correct location, after imports

This script is idempotent — running it twice on an already-patched
train_net.py is a no-op.

Environment variables (all optional, defaults match a fresh Vast.ai setup):
    M2F_TRAIN_NET_PATH   Path to Mask2Former's train_net.py
                          (default: /workspace/Mask2Former/train_net.py)
    DRONE_DATA_ROOT       Root directory containing images/ and annotations/
                          (default: /workspace/data)

Usage:
    python register_dataset.py
    DRONE_DATA_ROOT=/custom/path python register_dataset.py
"""

import os

TRAIN_NET = os.environ.get("M2F_TRAIN_NET_PATH", "/workspace/Mask2Former/train_net.py")
DATA_ROOT = os.environ.get("DRONE_DATA_ROOT", "/workspace/data")

# 15-class taxonomy: 4 "thing" building types, 3 "stuff" surface types,
# 1 thing infrastructure x3, 5 vehicle thing types.
DRONE_CATEGORIES = [
    {"id": 1,  "name": "building_rcc",        "isthing": 1, "color": [220, 20,  60]},
    {"id": 2,  "name": "building_tin",        "isthing": 1, "color": [119, 11,  32]},
    {"id": 3,  "name": "building_tiled",      "isthing": 1, "color": [0,   0,  142]},
    {"id": 4,  "name": "building_others",     "isthing": 1, "color": [0,   0,  230]},
    {"id": 5,  "name": "road_paved",          "isthing": 0, "color": [128, 64, 128]},
    {"id": 6,  "name": "road_mud",            "isthing": 0, "color": [244, 35, 232]},
    {"id": 7,  "name": "waterbody",           "isthing": 0, "color": [70,  70,  70]},
    {"id": 8,  "name": "overhead_tank",       "isthing": 1, "color": [102, 102, 156]},
    {"id": 9,  "name": "well",                "isthing": 1, "color": [190, 153, 153]},
    {"id": 10, "name": "solar_panel",         "isthing": 1, "color": [153, 153, 153]},
    {"id": 11, "name": "vehicle_car",         "isthing": 1, "color": [250, 170,  30]},
    {"id": 12, "name": "vehicle_truck",       "isthing": 1, "color": [220, 220,   0]},
    {"id": 13, "name": "vehicle_two_wheeler", "isthing": 1, "color": [107, 142,  35]},
    {"id": 14, "name": "vehicle_bus",         "isthing": 1, "color": [152, 251, 152]},
    {"id": 15, "name": "vehicle_others",      "isthing": 1, "color": [70,  130, 180]},
]

REGISTRATION_TEMPLATE = '''
# ── Drone Dataset Registration (auto-inserted by register_dataset.py) ──
import os as _os
import json as _json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco_panoptic import load_coco_panoptic_json
from detectron2.data.datasets import load_sem_seg

DATA_ROOT = "{data_root}"

DRONE_CATEGORIES = {categories!r}

_thing_ids    = [c["id"]    for c in DRONE_CATEGORIES if c["isthing"]]
_stuff_ids    = [c["id"]    for c in DRONE_CATEGORIES if not c["isthing"]]
_thing_names  = [c["name"]  for c in DRONE_CATEGORIES if c["isthing"]]
_stuff_names  = [c["name"]  for c in DRONE_CATEGORIES if not c["isthing"]]
_thing_colors = [c["color"] for c in DRONE_CATEGORIES if c["isthing"]]
_stuff_colors = [c["color"] for c in DRONE_CATEGORIES if not c["isthing"]]

DRONE_META = {{
    "thing_classes":  _thing_names,
    "thing_colors":   _thing_colors,
    "stuff_classes":  _stuff_names,
    "stuff_colors":   _stuff_colors,
    "thing_dataset_id_to_contiguous_id": {{k: i for i, k in enumerate(_thing_ids)}},
    "stuff_dataset_id_to_contiguous_id": {{k: i for i, k in enumerate(_stuff_ids)}},
}}

def _register_drone(split):
    name      = f"drone_panoptic_{{split}}"
    image_dir = _os.path.join(DATA_ROOT, f"images/{{split}}")
    pan_dir   = _os.path.join(DATA_ROOT, f"annotations/panoptic_{{split}}")
    pan_json  = _os.path.join(DATA_ROOT, f"annotations/panoptic_{{split}}.json")

    def _load(image_dir=image_dir, pan_dir=pan_dir, pan_json=pan_json):
        dicts = load_coco_panoptic_json(pan_json, image_dir, pan_dir, DRONE_META)
        with open(pan_json) as f:
            pan_data = _json.load(f)
        fname_to_size = {{
            img["file_name"]: (img["width"], img["height"])
            for img in pan_data["images"]
        }}
        for d in dicts:
            fname = _os.path.basename(d["file_name"])
            if fname in fname_to_size:
                d["width"], d["height"] = fname_to_size[fname]
        return dicts

    DatasetCatalog.register(name, _load)
    MetadataCatalog.get(name).set(
        panoptic_root=pan_dir,
        image_root=image_dir,
        panoptic_json=pan_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **DRONE_META,
    )

_register_drone("train")
_register_drone("val")
# ── End Drone Dataset Registration ──
'''


def patch() -> None:
    if not os.path.exists(TRAIN_NET):
        raise FileNotFoundError(
            f"train_net.py not found at {TRAIN_NET}. "
            "Set M2F_TRAIN_NET_PATH if Mask2Former is installed elsewhere."
        )

    with open(TRAIN_NET, "r") as f:
        content = f.read()

    if "Drone Dataset Registration" in content:
        print("train_net.py already patched — skipping")
        return

    marker = "    add_maskformer2_config,\n)"
    if marker not in content:
        raise RuntimeError(
            "Could not find the expected insertion point in train_net.py. "
            "This script targets facebookresearch/Mask2Former's default "
            "train_net.py layout — if you're on a fork, adjust `marker`."
        )

    registration_code = REGISTRATION_TEMPLATE.format(
        data_root=DATA_ROOT,
        categories=DRONE_CATEGORIES,
    )
    content = content.replace(marker, marker + registration_code)

    with open(TRAIN_NET, "w") as f:
        f.write(content)

    print(f"Successfully patched {TRAIN_NET}")
    print(f"Dataset root: {DATA_ROOT}")


if __name__ == "__main__":
    patch()
