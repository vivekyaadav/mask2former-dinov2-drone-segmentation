#!/usr/bin/env python3
"""
Orthomosaic Semantic Segmentation Pipeline
==========================================
Generic pipeline for any drone orthomosaic.
Supported GSD: 2cm - 10cm (always resampled to 5cm for inference)

Usage:
    # Auto-detect GSD from file:
    python3 ortho_pipeline.py --input village.tif

    # Specify GSD manually:
    python3 ortho_pipeline.py --input village.tif --gsd 0.035

    # Full options:
    python3 ortho_pipeline.py --input village.tif --gsd 0.035 \
        --out /workspace/output/myvillage --name myvillage --workers 48

Output:
    <out_dir>/<name>_segmentation.zip
    Contains 9 class-wise shapefiles ready for QGIS.

Resumable: rerun same command if it crashes — completed tiles are skipped.
"""
import sys
sys.path.insert(0, "/workspace/Mask2Former")
sys.path.insert(0, "/workspace")

import os, json, shutil, zipfile, warnings, gc, argparse, math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import rasterio.features
import rasterio.windows
import rasterio.transform
from rasterio.enums import Resampling
from PIL import Image
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import fiona
from fiona.crs import from_epsg

warnings.filterwarnings("ignore")

# ── Fixed paths ───────────────────────────────────────────────────────────────
CONFIG_FILE = "/workspace/Mask2Former/configs/drone/drone_dinov2_semantic_v6.yaml"
WEIGHTS     = "/workspace/output/drone_dinov2_semantic_v6/model_best.pth"

TILE_SIZE        = 1024
CHUNK_ROWS       = 4096
TGT_GSD          = 0.05    # always resample to 5cm
MIN_GSD_SUPPORT  = 0.02    # 2cm minimum
MAX_GSD_SUPPORT  = 0.10    # 10cm maximum

CLASS_NAMES = [
    "building_rcc", "building_tin", "building_tiled", "building_others",
    "waterbody", "overhead_tank", "well", "solar_panel", "vehicle",
]

CONF_THRESH = {
    "building_rcc":    0.15,
    "building_tin":    0.14,
    "building_tiled":  0.14,
    "building_others": 0.14,
    "waterbody":       0.18,
    "overhead_tank":   0.13,
    "well":            0.13,
    "solar_panel":     0.13,
    "vehicle":         0.13,
}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Drone orthomosaic semantic segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ortho_pipeline.py --input /workspace/data/village.tif
  python3 ortho_pipeline.py --input /workspace/data/village.tif --gsd 0.035
  python3 ortho_pipeline.py --input /workspace/data/village.tif --name bagga --workers 48

Supported GSD: 2cm to 10cm. Auto-resampled to 5cm for inference.
        """
    )
    p.add_argument("--input",   required=True,
                   help="Path to input GeoTIFF orthomosaic")
    p.add_argument("--gsd",     default=None, type=float,
                   help="Source GSD in metres (e.g. 0.035). Auto-detected if omitted.")
    p.add_argument("--out",     default=None,
                   help="Output directory. Default: /workspace/output/<name>_inference")
    p.add_argument("--name",    default=None,
                   help="Name prefix for outputs. Default: input filename stem")
    p.add_argument("--workers", default=48, type=int,
                   help="CPU cores to use (default: 48)")
    p.add_argument("--min-px",  default=500, type=int,
                   help="Minimum polygon size in pixels (default: 500)")
    return p.parse_args()


# ── GSD detection ─────────────────────────────────────────────────────────────
def detect_gsd(tif_path):
    with rasterio.open(tif_path) as src:
        gsd_x = abs(src.transform.a)
        gsd_y = abs(src.transform.e)
        epsg  = src.crs.to_epsg() if src.crs else None
        # Geographic CRS (degrees) — convert to metres
        if epsg == 4326:
            lat   = (src.bounds.top + src.bounds.bottom) / 2
            gsd_x = gsd_x * math.cos(math.radians(lat)) * 111320
            gsd_y = gsd_y * 111320
        gsd = (gsd_x + gsd_y) / 2
        bands = src.count
        w, h  = src.width, src.height
        crs   = src.crs
    return gsd, w, h, bands, crs


# ── Step 1 — Resample (48 parallel processes) ─────────────────────────────────
def _resample_chunk(args):
    src_path, row_off, rows_this, src_row, src_rows, new_w, src_w, n_out = args
    with rasterio.open(src_path) as s:
        win   = rasterio.windows.Window(0, src_row, src_w, src_rows)
        bands = [s.read(i, window=win, out_shape=(rows_this, new_w),
                        resampling=Resampling.lanczos)
                 for i in range(1, n_out + 1)]
    return row_off, rows_this, bands


def resample(cfg):
    if os.path.exists(cfg["resamp_tif"]):
        print("[1/6] Resample already done")
        return

    src_gsd = cfg["src_gsd"]

    # If already at target GSD, just copy
    if abs(src_gsd - TGT_GSD) < 0.002:
        print("[1/6] GSD already at 5cm — copying without resampling...")
        shutil.copy2(cfg["input_tif"], cfg["resamp_tif"])
        print(f"[1/6] Done -> {cfg['resamp_tif']}")
        return

    print(f"[1/6] Resampling {src_gsd*100:.2f}cm -> {TGT_GSD*100:.1f}cm "
          f"({cfg['n_cores']} cores)...")
    scale = src_gsd / TGT_GSD

    with rasterio.open(cfg["input_tif"]) as src:
        new_w   = int(src.width  * scale)
        new_h   = int(src.height * scale)
        new_tf  = src.transform * src.transform.scale(
            src.width / new_w, src.height / new_h)
        n_out   = min(src.count, 3)
        src_h   = src.height
        src_w   = src.width
        profile = src.profile.copy()
        profile.update(count=n_out, width=new_w, height=new_h,
                       transform=new_tf, compress="lzw",
                       bigtiff="YES", dtype="uint8")

    chunk_args = []
    for row_off in range(0, new_h, CHUNK_ROWS):
        rows_this = min(CHUNK_ROWS, new_h - row_off)
        src_row   = int(row_off / scale)
        src_rows  = min(int(rows_this / scale) + 4, src_h - src_row)
        chunk_args.append((cfg["input_tif"], row_off, rows_this,
                           src_row, src_rows, new_w, src_w, n_out))

    print(f"  {src_w}x{src_h} -> {new_w}x{new_h} px | "
          f"{len(chunk_args)} chunks")

    with rasterio.open(cfg["resamp_tif"], "w", **profile) as dst:
        with mp.Pool(processes=cfg["n_cores"]) as pool:
            for done, (row_off, rows_this, bands) in enumerate(
                    pool.imap(_resample_chunk, chunk_args, chunksize=2)):
                dst_win = rasterio.windows.Window(0, row_off, new_w, rows_this)
                for i, band in enumerate(bands, 1):
                    dst.write(band, i, window=dst_win)
                print(f"  {done+1}/{len(chunk_args)} "
                      f"({100*(done+1)//len(chunk_args)}%)",
                      end="\r", flush=True)
    print(f"\n[1/6] Done -> {cfg['resamp_tif']}")


# ── Step 2 — Tile (48 parallel row reads + 16 thread save) ───────────────────
def _read_tile_row(args):
    src_path, row_i, cols_n, W, H, tf_params = args
    tf  = rasterio.transform.Affine(*tf_params)
    row = row_i * TILE_SIZE
    results = []
    with rasterio.open(src_path) as src:
        for col_i in range(cols_n):
            col = col_i * TILE_SIZE
            aw  = min(TILE_SIZE, W - col)
            ah  = min(TILE_SIZE, H - row)
            win = rasterio.windows.Window(col, row, aw, ah)
            data   = src.read(window=win)
            canvas = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            for bi in range(min(3, data.shape[0])):
                canvas[:ah, :aw, bi] = data[bi]
            if (canvas.sum(axis=2) > 0).mean() < 0.05:
                continue
            tile_tf = rasterio.transform.from_bounds(
                tf.c + col * tf.a,
                tf.f + (row + TILE_SIZE) * tf.e,
                tf.c + (col + TILE_SIZE) * tf.a,
                tf.f + row * tf.e,
                TILE_SIZE, TILE_SIZE
            )
            results.append({
                "tile_id":   f"r{row:07d}_c{col:07d}",
                "row": row, "col": col,
                "width": aw, "height": ah,
                "transform": list(tile_tf)[:6],
                "canvas": canvas,
            })
    return results


def tile_raster(cfg):
    if os.path.exists(cfg["index_path"]):
        with open(cfg["index_path"]) as f:
            records = json.load(f)
        print(f"[2/6] Already tiled: {len(records)} tiles")
        return records

    print(f"[2/6] Tiling {TILE_SIZE}x{TILE_SIZE}px "
          f"({cfg['n_cores']} cores)...")

    with rasterio.open(cfg["resamp_tif"]) as src:
        W, H      = src.width, src.height
        tf_params = list(src.transform)[:6]

    cols_n = (W + TILE_SIZE - 1) // TILE_SIZE
    rows_n = (H + TILE_SIZE - 1) // TILE_SIZE
    print(f"  {W}x{H}px -> {cols_n}x{rows_n} = {cols_n*rows_n} tiles max")

    row_args  = [(cfg["resamp_tif"], ri, cols_n, W, H, tf_params)
                 for ri in range(rows_n)]
    records   = []
    save_pool = ThreadPoolExecutor(max_workers=16)
    futs      = []

    def save_png(path, canvas):
        Image.fromarray(canvas).save(path, compress_level=1)

    with mp.Pool(processes=cfg["n_cores"]) as pool:
        for done, row_results in enumerate(
                pool.imap(_read_tile_row, row_args, chunksize=4)):
            for r in row_results:
                canvas   = r.pop("canvas")
                png_path = os.path.join(cfg["tiles_dir"], f"{r['tile_id']}.png")
                r["png_path"] = png_path
                futs.append(save_pool.submit(save_png, png_path, canvas))
                records.append(r)
            if (done + 1) % 20 == 0:
                print(f"  Row {done+1}/{rows_n} | tiles={len(records)}")

    for f in futs:
        f.result()
    save_pool.shutdown()

    with open(cfg["index_path"], "w") as f:
        json.dump(records, f)
    print(f"[2/6] Done: {len(records)} tiles")
    return records


# ── Step 3 — Load Model ───────────────────────────────────────────────────────
def load_model():
    print("[3/6] Loading model...")
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former.config import add_dinov2_config
    from mask2former import add_maskformer2_config

    for split in ["train", "val", "test"]:
        name = f"drone_semantic_{split}"
        if name not in DatasetCatalog:
            DatasetCatalog.register(name, lambda: [])
            MetadataCatalog.get(name).set(
                stuff_classes=CLASS_NAMES, ignore_label=255,
                evaluator_type="sem_seg")

    c = get_cfg()
    add_deeplab_config(c)
    add_maskformer2_config(c)
    add_dinov2_config(c)
    c.merge_from_file(CONFIG_FILE)
    c.MODEL.WEIGHTS = WEIGHTS
    c.freeze()
    model = DefaultTrainer.build_model(c)
    DetectionCheckpointer(model).load(WEIGHTS)
    model.eval().cuda()
    print("  Loaded")
    return model


# ── Step 4 — Inference (16 prefetch workers + 16 async save threads) ──────────
def run_inference(model, records, cfg):
    done_ids = set()
    if os.path.exists(cfg["done_path"]):
        with open(cfg["done_path"]) as f:
            done_ids = set(f.read().splitlines())

    todo = [r for r in records if r["tile_id"] not in done_ids]
    print(f"[4/6] Inference: {len(todo)} tiles ({len(done_ids)} done)...")

    class TileDS(torch.utils.data.Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, idx):
            r   = self.recs[idx]
            img = np.array(Image.open(r["png_path"]).convert("RGB"))
            bgr = img[:, :, ::-1].astype("float32")
            return idx, torch.as_tensor(bgr.transpose(2, 0, 1)), \
                   r["height"], r["width"]

    loader = torch.utils.data.DataLoader(
        TileDS(todo), batch_size=1,
        num_workers=16, pin_memory=True, prefetch_factor=8)

    save_pool = ThreadPoolExecutor(max_workers=16)
    done_fh   = open(cfg["done_path"], "a")

    def save_npy(path, arr):
        np.save(path, arr)

    for i, (idx_b, tensor_b, h_b, w_b) in enumerate(loader):
        rec        = todo[idx_b[0].item()]
        img_tensor = tensor_b[0].cuda()
        H, W       = h_b[0].item(), w_b[0].item()

        with torch.no_grad():
            out = model([{"image": img_tensor, "height": H, "width": W}])

        sem   = out[0]["sem_seg"]
        probs = sem.softmax(dim=0)
        probs = F.avg_pool2d(probs.unsqueeze(0), 5, 1, 2).squeeze(0)
        conf  = probs.max(dim=0).values.cpu().numpy()[:H, :W]
        pred  = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)[:H, :W]

        combined  = np.stack([pred, (conf * 255).astype(np.uint8)], axis=0)
        pred_path = os.path.join(cfg["preds_dir"], f"{rec['tile_id']}.npy")
        save_pool.submit(save_npy, pred_path, combined)

        done_fh.write(rec["tile_id"] + "\n")
        done_fh.flush()

        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            total_done = len(done_ids) + i + 1
            print(f"  [{total_done:>6}/{len(records)}] "
                  f"{100*total_done/len(records):.1f}%")

    save_pool.shutdown(wait=True)
    done_fh.close()
    print("[4/6] Inference complete")


# ── Step 5 — Vectorize (48 processes) + Merge ────────────────────────────────
def _vectorize_tile(args):
    rec, preds_dir, tgt_gsd, min_pixels = args
    pred_path = os.path.join(preds_dir, f"{rec['tile_id']}.npy")
    if not os.path.exists(pred_path):
        return {}
    combined = np.load(pred_path)
    if combined.ndim == 3 and combined.shape[0] == 2:
        pred = combined[0]
        conf = combined[1].astype(np.float32) / 255.0
    else:
        pred = combined
        conf = np.ones_like(pred, dtype=np.float32)

    tf       = rasterio.transform.Affine(*rec["transform"])
    min_area = min_pixels * tgt_gsd * tgt_gsd
    result   = {cls: [] for cls in CLASS_NAMES}

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        thresh = CONF_THRESH.get(cls_name, 0.13)
        mask   = ((pred == cls_id) & (conf >= thresh)).astype(np.uint8)
        if mask.sum() < min_pixels:
            continue
        for geom, _ in rasterio.features.shapes(
                mask, mask=mask, transform=tf):
            p = shape(geom)
            if p.area >= min_area:
                result[cls_name].append(
                    p.simplify(0.05, preserve_topology=True))
    return result


def vectorize_and_merge(records, cfg):
    print(f"[5/6] Vectorizing {len(records)} tiles "
          f"({cfg['n_cores']} workers)...")
    class_polys = {cls: [] for cls in CLASS_NAMES}

    args = [(r, cfg["preds_dir"], TGT_GSD, cfg["min_pixels"])
            for r in records]

    with ProcessPoolExecutor(max_workers=cfg["n_cores"]) as pool:
        for i, result in enumerate(
                pool.map(_vectorize_tile, args, chunksize=20)):
            for cls_name, polys in result.items():
                class_polys[cls_name].extend(polys)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(records)} vectorized")

    print("[5/6] Writing GeoPackage...")

    # Standardised colors per class (RGBA)
    CLASS_COLORS_RGBA = {
        "building_rcc":    (230, 0,   0,   200),   # pure red
        "building_tin":    (255, 165, 0,   200),   # bright orange
        "building_tiled":  (0,   200, 0,   200),   # bright green
        "building_others": (180, 0,   255, 200),   # vivid purple
        "waterbody":       (0,   80,  255, 210),   # bright blue
        "overhead_tank":   (255, 0,   200, 210),   # hot magenta
        "well":            (40,  40,  40,  220),   # near black
        "solar_panel":     (255, 255, 0,   210),   # pure yellow
        "vehicle":         (0,   255, 180, 210),   # cyan-green
    }

    with rasterio.open(cfg["resamp_tif"]) as src:
        epsg = src.crs.to_epsg()
    crs    = from_epsg(epsg)
    schema = {"geometry": "Polygon",
              "properties": {"class_id":   "int",
                             "class_name": "str",
                             "area_m2":    "float"}}
    min_a    = cfg["min_pixels"] * TGT_GSD ** 2
    gpkg_path = cfg["gpkg_out"]

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        polys = class_polys[cls_name]
        if not polys:
            print(f"  {cls_name}: no polygons")
            continue
        print(f"  {cls_name}: {len(polys)} raw -> merging...",
              end=" ", flush=True)
        try:
            merged = unary_union(polys)
            merged = merged.buffer(0.3).buffer(-0.3)
            geoms  = ([merged] if merged.geom_type == "Polygon"
                      else list(merged.geoms))
            geoms  = [g for g in geoms if g.area >= min_a]
            geoms  = [g.simplify(0.05, preserve_topology=True) for g in geoms]
        except Exception:
            geoms = polys
        print(f"{len(geoms)} final")

        # Write shapefile
        shp_path = os.path.join(cfg["shp_dir"], f"{cls_name}.shp")
        with fiona.open(shp_path, "w", driver="ESRI Shapefile",
                        crs=crs, schema=schema) as shp:
            for g in geoms:
                if g.is_empty or not g.is_valid:
                    continue
                shp.write({"geometry": mapping(g),
                            "properties": {"class_id":   cls_id,
                                           "class_name": cls_name,
                                           "area_m2":    round(g.area, 2)}})
        # Write GeoPackage layer (always fresh)
        with fiona.open(gpkg_path, "w", driver="GPKG",
                        layer=cls_name, crs=crs, schema=schema) as gpkg:
            for g in geoms:
                if g.is_empty or not g.is_valid:
                    continue
                gpkg.write({"geometry": mapping(g),
                             "properties": {"class_id":   cls_id,
                                            "class_name": cls_name,
                                            "area_m2":    round(g.area, 2)}})

    # Embed layer styles into GeoPackage sqlite metadata
    import sqlite3
    conn = sqlite3.connect(gpkg_path)
    cur  = conn.cursor()
    # Create layer_styles table (QGIS standard)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS layer_styles (
            id INTEGER PRIMARY KEY,
            f_table_catalog TEXT, f_table_schema TEXT,
            f_table_name TEXT, f_geometry_column TEXT,
            styleName TEXT, styleQML TEXT,
            styleSLD TEXT, useAsDefault INTEGER,
            description TEXT, owner TEXT,
            ui TEXT, update_time TEXT
        )
    """)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        if cls_name not in CLASS_COLORS_RGBA:
            continue
        r, g, b, a = CLASS_COLORS_RGBA[cls_name]
        br = max(0, r-40); bg2 = max(0, g-40); bb = max(0, b-40)
        qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0">
    <symbols>
      <symbol name="0" type="fill" alpha="0.8">
        <layer class="SimpleFill" enabled="1">
          <Option type="Map">
            <Option name="color" value="{r},{g},{b},{a}" type="QString"/>
            <Option name="outline_color" value="{br},{bg2},{bb},255" type="QString"/>
            <Option name="outline_width" value="0.26" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>2</layerGeometryType>
</qgis>"""
        cur.execute("""
            INSERT OR REPLACE INTO layer_styles
            (f_table_name, f_geometry_column, styleName, styleQML, useAsDefault)
            VALUES (?, 'geometry', 'default', ?, 1)
        """, (cls_name, qml))
    conn.commit()
    conn.close()
    print(f"  -> {gpkg_path}")


# ── Step 6 — Package ZIP ──────────────────────────────────────────────────────
def package_zip(cfg):
    print("[6/6] Packaging ZIP...")
    readme = (
        f"Segmentation GeoPackage — {cfg['name']}\n"
        f"{'='*45}\n"
        f"Model:  DINOv2 + Mask2Former v6\n"
        f"Source GSD: {cfg['src_gsd']*100:.2f}cm  "
        f"Inference GSD: {TGT_GSD*100:.1f}cm\n\n"
        "File:\n"
        f"  {cfg['name']}.gpkg  — All 9 classes in one file\n\n"
        "Layers inside GeoPackage:\n"
        "  building_rcc    Reinforced concrete rooftops (crimson)\n"
        "  building_tin    Tin/metal sheet rooftops (orange)\n"
        "  building_tiled  Tiled/terracotta rooftops (green)\n"
        "  building_others Other building types (purple)\n"
        "  waterbody       Ponds, lakes, rivers (blue)\n"
        "  overhead_tank   Water storage tanks (pink)\n"
        "  well            Water wells (dark gray)\n"
        "  solar_panel     Solar panels (yellow)\n"
        "  vehicle         Vehicles (lime green)\n\n"
        "Attributes per polygon:\n"
        "  class_id   Numeric class ID (0-8)\n"
        "  class_name Class name string\n"
        "  area_m2    Area in square metres\n\n"
        "QGIS Usage:\n"
        "  1. Drag the .gpkg file onto QGIS canvas\n"
        "  2. Select all layers when prompted\n"
        "  3. Colors load automatically — no setup needed\n"
        "  4. Load original orthomosaic as background\n"
    )
    with zipfile.ZipFile(cfg["zip_out"], "w", zipfile.ZIP_DEFLATED) as zf:
        # GeoPackage (single file, styled)
        zf.write(cfg["gpkg_out"], os.path.basename(cfg["gpkg_out"]))
        # Shapefiles (class-wise, for GIS compatibility)
        for fn in os.listdir(cfg["shp_dir"]):
            zf.write(os.path.join(cfg["shp_dir"], fn),
                     os.path.join("shapefiles", fn))
        zf.writestr("README.txt", readme)
    mb = os.path.getsize(cfg["zip_out"]) / 1e6
    print(f"  -> {cfg['zip_out']} ({mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    # Auto-detect GSD and file info
    detected_gsd, w, h, bands, crs = detect_gsd(str(input_path))
    if args.gsd is not None:
        src_gsd = args.gsd
        print(f"  GSD: {src_gsd*100:.2f}cm (user-specified)")
    else:
        src_gsd = detected_gsd
        print(f"  GSD: {src_gsd*100:.2f}cm (auto-detected)")

    # Validate GSD range
    if not (MIN_GSD_SUPPORT <= src_gsd <= MAX_GSD_SUPPORT):
        print(f"\n[WARN] GSD {src_gsd*100:.2f}cm is outside supported range "
              f"({MIN_GSD_SUPPORT*100:.0f}cm - {MAX_GSD_SUPPORT*100:.0f}cm)")
        print(f"  Model was trained on 3.5-5cm GSD imagery.")
        resp = input("  Continue anyway? [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(0)

    name    = args.name or input_path.stem
    out_dir = args.out  or f"/workspace/output/{name}_inference"

    cfg = {
        "input_tif":  str(input_path),
        "src_gsd":    src_gsd,
        "n_cores":    args.workers,
        "min_pixels": args.min_px,
        "name":       name,
        "out_dir":    out_dir,
        "resamp_tif": os.path.join(out_dir, "resampled.tif"),
        "tiles_dir":  os.path.join(out_dir, "tiles"),
        "preds_dir":  os.path.join(out_dir, "predictions"),
        "shp_dir":    os.path.join(out_dir, "shapefiles"),
        "zip_out":    os.path.join(out_dir, f"{name}_segmentation.zip"),
        "index_path": os.path.join(out_dir, "tile_index.json"),
        "done_path":  os.path.join(out_dir, "inference_done.txt"),
        "gpkg_out":   os.path.join(out_dir, f"{name}.gpkg"),
    }

    for d in [out_dir, cfg["tiles_dir"], cfg["preds_dir"], cfg["shp_dir"]]:
        os.makedirs(d, exist_ok=True)

    free = shutil.disk_usage("/workspace").free / 1e9
    print("=" * 60)
    print(f"  Orthomosaic Inference Pipeline")
    print(f"  Input:    {input_path.name}")
    print(f"  Size:     {w}x{h}px  |  Bands: {bands}")
    print(f"  GSD:      {src_gsd*100:.2f}cm -> {TGT_GSD*100:.1f}cm")
    print(f"  CRS:      {crs}")
    print(f"  Output:   {cfg['zip_out']}")
    print(f"  Cores:    {cfg['n_cores']}  |  Disk free: {free:.1f} GB")
    print("=" * 60 + "\n")

    if free < 15:
        print(f"[WARN] Only {free:.1f}GB free — may run out during pipeline")

    resample(cfg)
    records = tile_raster(cfg)
    model   = load_model()
    run_inference(model, records, cfg)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    vectorize_and_merge(records, cfg)
    package_zip(cfg)

    free2 = shutil.disk_usage("/workspace").free / 1e9
    print(f"\n  Done. Disk free: {free2:.1f} GB")
    print(f"\n  Output: {cfg['zip_out']}")
    # scp the output zip from your remote instance to your local machine


if __name__ == "__main__":
    main()