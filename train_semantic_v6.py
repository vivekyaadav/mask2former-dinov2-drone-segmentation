#!/usr/bin/env python3
"""
DINOv2 + Mask2Former  |  Semantic Segmentation v6  |  Drone Dataset
====================================================================
Fresh start — no warm-start from any previous run checkpoint.
No LoRA — full fine-tune like run 2 (our best run).

Key additions over run 2:
  1. Per-class Hungarian matching cost — forces matcher to assign
     queries to rare classes (well, overhead_tank, solar_panel).
  2. Per-class Dice weights — stronger mask quality pressure on
     rare classes at pixel level.
  3. Better class weights — run 4 analysis showed tin=3.0 and
     overhead_tank=10.0 caused gradient conflicts.
  4. RareClassBalancedSampler — 65% rare class per batch.
  5. CLIP_VALUE=0.3 — tighter than run 2's 0.5.
  6. MAX_ITER=60000 — more convergence time than run 2's 30000.

Start weights: DINOv2 pretrained only (/workspace/checkpoints/dinov2_vitb14_pretrain.pth)
"""
import os as _os_bootstrap
import sys
# Make the Mask2Former fork importable. Override with M2F_ROOT if it lives elsewhere.
sys.path.insert(0, _os_bootstrap.environ.get("M2F_ROOT", "/workspace"))

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except Exception:
    pass

import itertools, os, csv, math
from typing import Any, Dict, List, Set

import torch
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from mask2former.config import add_dinov2_config
from mask2former import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
# Root containing images/ + semantic_masks_9cls/ (+ augmented train dirs).
# Override with the DRONE_DATA_ROOT env var; defaults to the container path.
DATA_ROOT = os.environ.get("DRONE_DATA_ROOT", "/workspace/data")

CLASS_NAMES = [
    "building_rcc",     # 0 — 84.2% pixels
    "building_tin",     # 1 — 10.8%
    "building_tiled",   # 2 —  1.7%
    "building_others",  # 3 —  1.2%
    "waterbody",        # 4 —  1.0%
    "overhead_tank",    # 5 —  0.6%
    "well",             # 6 —  0.04%
    "solar_panel",      # 7 —  0.14%
    "vehicle",          # 8 —  0.36%
]
CLASS_COLORS = [
    [220,  20,  60], [119,  11,  32], [  0,   0, 142], [  0,   0, 230],
    [ 70,  70,  70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170,  30],
]
NUM_CLASSES  = 9
RARE_CLASS_IDS = {1, 5, 6, 7, 8}

# ── Class weights (CE loss) ───────────────────────────────────────────────────
# Lessons from runs 3/4:
#   - building_tin 3.0 caused gradient war with rcc → 0.8
#   - overhead_tank 10.0 contributed to spikes → 5.0
#   - well 20.0 caused collapse after iter 2000 → 8.0
#   - waterbody already at 90%+ → reduce to 0.5
CLASS_WEIGHTS = torch.tensor(
    [0.1, 0.8, 2.0, 0.84, 0.5, 5.0, 8.0, 7.237, 3.0],
    dtype=torch.float32
)

# ── Per-class Hungarian matching costs ───────────────────────────────────────
# Higher = matcher tries harder to assign a query to this class.
# Injected into matcher.class_match_costs (matcher.py patched).
CLASS_MATCH_COSTS = torch.tensor(
    [1.0, 2.0, 1.5, 2.0, 2.0, 6.0, 10.0, 5.0, 3.0],
    dtype=torch.float32
)

# ── Per-class Dice weights ────────────────────────────────────────────────────
# Higher = stronger mask quality pressure on this class.
# Injected into criterion.per_class_dice_weights (criterion.py patched).
PER_CLASS_DICE_WEIGHTS = torch.tensor(
    [1.0, 3.0, 2.0, 2.0, 1.0, 8.0, 10.0, 6.0, 4.0],
    dtype=torch.float32
)


def inject_weights_and_costs(model, class_weights, class_match_costs,
                              per_class_dice_weights, no_obj_weight=0.1):
    """
    Inject into criterion:
      1. CE class weights → criterion.empty_weight
      2. Per-class Hungarian matching costs → matcher.class_match_costs
      3. Per-class Dice weights → criterion.per_class_dice_weights
    """
    criterion = model.criterion
    device    = next(model.parameters()).device

    # 1. CE weights
    weights     = class_weights.to(device)
    full_weight = torch.cat([weights,
                             torch.tensor([no_obj_weight], device=device)])
    criterion.register_buffer("empty_weight", full_weight)

    # 2. Matching costs
    criterion.matcher.class_match_costs = class_match_costs.to(device)

    # 3. Per-class Dice — already registered as buffer in criterion.py, just set value
    criterion.per_class_dice_weights = per_class_dice_weights.to(device)

    print(f"\n[Weights] Per-class weights injected:")
    print(f"  {'Class':<22} {'CE':>6}  {'Match':>6}  {'Dice':>6}")
    print(f"  {'-'*46}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22} {class_weights[i]:>6.3f}  "
              f"{class_match_costs[i]:>6.1f}  "
              f"{per_class_dice_weights[i]:>6.1f}")
    print(f"  {'no_object':<22} {no_obj_weight:>6.3f}\n")
    return model


# ── Dataset loader ────────────────────────────────────────────────────────────
def _load_semantic_dataset(split: str):
    if split == "train":
        image_dir = os.path.join(DATA_ROOT, "images_augmented_v2", "train")
        mask_dir  = os.path.join(DATA_ROOT, "semantic_masks_9cls_aug_v2", "train")
    else:
        image_dir = os.path.join(DATA_ROOT, "images", split)
        mask_dir  = os.path.join(DATA_ROOT, "semantic_masks_9cls", split)

    records = []
    n_rare = n_common = 0

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem      = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(mask_path):
            continue

        from PIL import Image as _PIL
        with _PIL.open(os.path.join(image_dir, fname)) as _im:
            _w, _h = _im.size

        if split == "train":
            mask_arr = np.array(_PIL.open(mask_path))
            has_rare = bool(any(np.any(mask_arr == rid) for rid in RARE_CLASS_IDS))
        else:
            has_rare = False

        if has_rare: n_rare   += 1
        else:        n_common += 1

        records.append({
            "file_name":         os.path.join(image_dir, fname),
            "sem_seg_file_name": mask_path,
            "image_id":          stem,
            "height": _h, "width": _w,
            "has_rare": has_rare,
        })

    total = len(records)
    if split == "train":
        print(f"[Dataset] train: {total} records "
              f"({n_rare} rare={100*n_rare/total:.1f}%, "
              f"{n_common} common={100*n_common/total:.1f}%)")
    else:
        print(f"[Dataset] {split}: {total} records")
    return records


def _register(split):
    name = f"drone_semantic_{split}"
    DatasetCatalog.register(name, lambda s=split: _load_semantic_dataset(s))
    MetadataCatalog.get(name).set(
        stuff_classes=CLASS_NAMES, stuff_colors=CLASS_COLORS,
        ignore_label=255, evaluator_type="sem_seg",
    )

_register("train"); _register("val"); _register("test")


# ── Rare Class Balanced Sampler ───────────────────────────────────────────────
class RareClassBalancedSampler(torch.utils.data.Sampler):
    """
    65% of each batch = images containing rare class pixels.
    Produces max_iter × batch_size total indices (no StopIteration).
    """
    def __init__(self, dataset_dicts, batch_size=8, rare_fraction=0.65,
                 max_iter=60000, seed=42):
        self.batch_size = batch_size
        self.seed       = seed
        self.target_len = max_iter * batch_size

        self.rare_indices   = [i for i, d in enumerate(dataset_dicts)
                               if d.get("has_rare", False)]
        self.common_indices = [i for i, d in enumerate(dataset_dicts)
                               if not d.get("has_rare", False)]

        self.n_rare   = round(batch_size * rare_fraction)
        self.n_common = batch_size - self.n_rare

        print(f"\n[Sampler] RareClassBalancedSampler")
        print(f"  Rare pool:   {len(self.rare_indices):>6} images")
        print(f"  Common pool: {len(self.common_indices):>6} images")
        print(f"  Per batch:   {self.n_rare} rare + {self.n_common} common")
        print(f"  Total indices: {self.target_len:,}\n")

    def __iter__(self):
        rng    = np.random.default_rng(self.seed)
        rare   = np.array(self.rare_indices,   dtype=np.int64)
        common = np.array(self.common_indices, dtype=np.int64)
        rng.shuffle(rare); rng.shuffle(common)

        indices = []
        ri = ci = 0
        while len(indices) < self.target_len:
            for _ in range(self.n_rare):
                if ri >= len(rare): rng.shuffle(rare); ri = 0
                indices.append(int(rare[ri])); ri += 1
            pool = common if len(common) > 0 else rare
            for _ in range(self.n_common):
                if ci >= len(pool): rng.shuffle(pool); ci = 0
                indices.append(int(pool[ci])); ci += 1
        return iter(indices[:self.target_len])

    def __len__(self):
        return self.target_len


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([
            SemSegEvaluator(dataset_name, distributed=True,
                            output_dir=output_folder)
        ])

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerSemanticDatasetMapper(cfg, True) \
                 if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic" \
                 else None
        from detectron2.data import get_detection_dataset_dicts
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        )
        sampler = RareClassBalancedSampler(
            dataset_dicts,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            rare_fraction=0.65,
            max_iter=cfg.SOLVER.MAX_ITER,
            seed=42,
        )
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm  = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        base_lr            = cfg.SOLVER.BASE_LR
        backbone_mul       = cfg.SOLVER.BACKBONE_MULTIPLIER
        layer_decay        = 0.9

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        num_blocks = 0
        for name, _ in model.named_modules():
            if "blocks." in name:
                try:
                    idx = int(name.split("blocks.")[1].split(".")[0])
                    num_blocks = max(num_blocks, idx + 1)
                except (IndexError, ValueError):
                    pass
        num_blocks = max(num_blocks, 12)
        print(f"[Optimizer] {num_blocks} ViT blocks for layer-wise LR decay")

        def get_lr_mul(module_name):
            if "backbone" not in module_name: return 1.0
            if "blocks." not in module_name:
                return backbone_mul * (layer_decay ** num_blocks)
            try:
                idx = int(module_name.split("blocks.")[1].split(".")[0])
                return backbone_mul * (layer_decay ** (num_blocks - idx - 1))
            except (IndexError, ValueError):
                return backbone_mul

        params: List[Dict[str, Any]] = []
        memo:   Set[torch.nn.parameter.Parameter] = set()

        for module_name, module in model.named_modules():
            for param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo: continue
                memo.add(value)
                hp = {"lr": base_lr * get_lr_mul(module_name),
                      "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
                if param_name in ("relative_position_bias_table",
                                  "absolute_pos_embed", "pos_embed", "cls_token"):
                    hp["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hp["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hp["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hp})

        clip_val    = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable_clip = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_val > 0.0
        )

        class GradClipAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                all_params = list(itertools.chain(
                    *[x["params"] for x in self.param_groups]
                ))
                grads = [p.grad for p in all_params if p.grad is not None]
                if grads:
                    total_norm = torch.norm(
                        torch.stack([torch.norm(g.detach(), 2)
                                     for g in grads]), 2
                    ).item()
                    try:
                        from detectron2.utils.events import get_event_storage
                        get_event_storage().put_scalar(
                            "train/grad_norm", total_norm,
                            smoothing_hint=False)
                    except Exception:
                        pass
                if enable_clip:
                    torch.nn.utils.clip_grad_norm_(all_params, clip_val)
                super().step(closure)

        optimizer = GradClipAdamW(params, base_lr)
        if not enable_clip:
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


# ── Monitor ───────────────────────────────────────────────────────────────────
class SemanticMonitor:
    def __init__(self, trainer, output_dir):
        self.trainer    = trainer
        self.output_dir = output_dir
        self.best_miou  = -1.0
        os.makedirs(output_dir, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
        except ImportError:
            self.tb = None

        self.step_csv = os.path.join(output_dir, "train_steps.csv")
        if not os.path.exists(self.step_csv):
            with open(self.step_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["iter","total_loss","grad_norm","lr_head","lr_backbone"])

        self.eval_csv = os.path.join(output_dir, "val_evals.csv")
        if not os.path.exists(self.eval_csv):
            with open(self.eval_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["iter","mIoU","fwIoU","pAcc"] +
                    [f"IoU-{n}" for n in CLASS_NAMES])

    def _tb(self, tag, val, step):
        if self.tb and not math.isnan(float(val)):
            self.tb.add_scalar(tag, val, step)

    def after_step(self):
        storage = self.trainer.storage
        it = storage.iter
        if it % 20 != 0: return

        def latest(key):
            try: return storage.history(key).latest()
            except: return float("nan")

        loss = latest("total_loss")
        gn   = latest("train/grad_norm")
        try:
            lr_head     = self.trainer.optimizer.param_groups[-1]["lr"]
            lr_backbone = self.trainer.optimizer.param_groups[0]["lr"]
        except:
            lr_head = lr_backbone = float("nan")

        self._tb("train/total_loss",  loss,        it)
        self._tb("train/grad_norm",   gn,          it)
        self._tb("train/lr_head",     lr_head,     it)
        self._tb("train/lr_backbone", lr_backbone, it)
        if torch.cuda.is_available():
            self._tb("train/gpu_mem_gb",
                     torch.cuda.max_memory_allocated()/1e9, it)

        if it % 100 == 0:
            with open(self.step_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    it,
                    f"{loss:.4f}"        if not math.isnan(loss)        else "",
                    f"{gn:.4f}"          if not math.isnan(gn)          else "",
                    f"{lr_head:.8f}"     if not math.isnan(lr_head)     else "",
                    f"{lr_backbone:.8f}" if not math.isnan(lr_backbone) else "",
                ])

    def after_eval(self):
        storage = self.trainer.storage
        it      = storage.iter
        results = self.trainer.test(self.trainer.cfg, self.trainer.model)
        sem     = results.get("sem_seg", {})

        miou  = sem.get("mIoU",  float("nan"))
        fwiou = sem.get("fwIoU", float("nan"))
        pacc  = sem.get("pAcc",  float("nan"))

        per_iou = {}
        for n in CLASS_NAMES:
            val = sem.get(f"IoU-{n}",
                  sem.get(f"iou_{n}", sem.get(n, float("nan"))))
            per_iou[n] = val

        for tag, val in [("val/mIoU",miou),("val/fwIoU",fwiou),
                         ("val/pAcc",pacc)]:
            self._tb(tag, val, it)
        for n in CLASS_NAMES:
            self._tb(f"val/IoU/{n}", per_iou[n], it)

        row = [it, f"{miou:.2f}",
               f"{fwiou:.2f}" if not math.isnan(fwiou) else "",
               f"{pacc:.2f}"  if not math.isnan(pacc)  else ""]
        row += [f"{per_iou[n]:.2f}" if not math.isnan(per_iou[n]) else ""
                for n in CLASS_NAMES]
        with open(self.eval_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"\n{'='*65}")
        print(f"  EVAL @ iter {it:,}")
        print(f"  mIoU={miou:.2f}%   fwIoU={fwiou:.2f}%   pAcc={pacc:.2f}%")
        print(f"\n  {'Class':<25} {'IoU':>7}  Bar")
        print(f"  {'-'*50}")
        for n in CLASS_NAMES:
            u   = per_iou[n]
            bar = "█" * int(u/5) if not math.isnan(u) else ""
            u_s = f"{u:.2f}" if not math.isnan(u) else "  N/A"
            rare= " ★" if n in [CLASS_NAMES[i] for i in [1,2,5,6,7,8]] else ""
            print(f"  {n:<25} {u_s:>7}  {bar}{rare}")
        print(f"{'='*65}\n")

        if not math.isnan(miou) and miou > self.best_miou:
            self.best_miou = miou
            self.trainer.checkpointer.save("model_best")
            print(f"  ★ New best mIoU={miou:.2f}% — saved model_best.pth\n")

        if self.tb: self.tb.flush()


# ── Setup & Main ─────────────────────────────────────────────────────────────
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_dinov2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR,
                 distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        model = inject_weights_and_costs(
            model, CLASS_WEIGHTS, CLASS_MATCH_COSTS, PER_CLASS_DICE_WEIGHTS,
            no_obj_weight=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        )
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        if cfg.TEST.AUG.ENABLED:
            model = SemanticSegmentorWithTTA(cfg, model)
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)

    # No LoRA — full fine-tune of all 90M params
    total     = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters()
                    if p.requires_grad)
    print(f"\n[Model] Total: {total/1e6:.1f}M  "
          f"Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
    print(f"[Model] No LoRA — full fine-tune from DINOv2 pretrained\n")

    # Inject CE weights + matching costs + per-class Dice
    trainer.model = inject_weights_and_costs(
        trainer.model, CLASS_WEIGHTS, CLASS_MATCH_COSTS, PER_CLASS_DICE_WEIGHTS,
        no_obj_weight=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
    )

    # Monitor
    monitor = SemanticMonitor(trainer, cfg.OUTPUT_DIR)

    original_after_step = trainer.after_step
    def patched_after_step():
        original_after_step()
        monitor.after_step()
        it = trainer.storage.iter
        ep = cfg.TEST.EVAL_PERIOD
        if ep > 0 and it % ep == 0 and it > 0:
            monitor.after_eval()
    trainer.after_step = patched_after_step

    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    # Final test evaluation
    if comm.is_main_process():
        best_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        if os.path.exists(best_path):
            print("\n[main] Loading model_best.pth for final test evaluation...")
            DetectionCheckpointer(trainer.model).load(best_path)

        from detectron2.data import build_detection_test_loader
        from detectron2.evaluation import inference_on_dataset
        evaluator = SemSegEvaluator(
            "drone_semantic_test", distributed=False,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "inference_test"))
        loader  = build_detection_test_loader(cfg, "drone_semantic_test")
        results = inference_on_dataset(trainer.model, loader, evaluator)
        sem     = results.get("sem_seg", {})

        print(f"\n  Test mIoU={sem.get('mIoU',0):.2f}%  "
              f"fwIoU={sem.get('fwIoU',0):.2f}%  "
              f"pAcc={sem.get('pAcc',0):.2f}%")
        print(f"\n  {'Class':<25} {'IoU':>7}")
        print(f"  {'-'*35}")
        for n in CLASS_NAMES:
            iou = sem.get(f"IoU-{n}", float("nan"))
            s   = f"{iou:.2f}" if not math.isnan(iou) else "N/A"
            print(f"  {n:<25} {s:>7}")

    return {}


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("="*65)
    print("  DINOv2 + Mask2Former  |  Semantic Seg v6  |  Fresh + Per-Class")
    print("="*65)
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url,
           args=(args,))