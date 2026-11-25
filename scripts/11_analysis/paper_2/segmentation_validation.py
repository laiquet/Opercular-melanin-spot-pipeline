#!/usr/bin/env python3
"""
Segmentation quality validation (referee comment 3).

1. Per-spot IoU and Boundary IoU on the held-out test split, stratified by spot
   size, for the raw YOLO masks and for the YOLO -> SAM 2.1 refined masks that
   the phenotyping pipeline actually measures.
2. Erosion sensitivity: spot intensity recomputed at 0, 1, 2 and 3 px erosion of
   every mask in the phenotyping set, to quantify edge-pixel contamination and
   decide whether erosion should be adopted.
3. Qualitative panel of successful, challenging and failed boundaries with
   ground truth overlaid.

Run:
    python segmentation_validation.py --output-dir results/paper_2

Datasets
--------
Reads   spots/test/images/, spots/test/labels/   27 held-out annotated images
        model_weights/yolov8_spots.pt, yolov11_spots.pt,
                      yolov8_operculum.pt, sam_checkpoint.pt
        images/, labels/    phenotyping set, for the erosion experiment
        results/paper_2/data/image_level_measures.csv        (script 1)
Writes  results/paper_2/data/test_split_spot_iou.csv
        results/paper_2/data/test_split_image_counts.csv
        results/paper_2/data/iou_by_spot_size.csv
        results/paper_2/data/test_split_gt_context.csv
        results/paper_2/data/detection_vs_contrast.csv
        results/paper_2/data/erosion_spotwise.csv
        results/paper_2/data/erosion_overall.csv
        results/paper_2/data/erosion_by_size.csv
        results/paper_2/data/erosion_downstream.csv
        results/paper_2/data/segmentation_validation_meta.json
        results/paper_2/figures/fig_iou_by_size.png
        results/paper_2/figures/fig_qualitative_segmentation.png
        results/paper_2/figures/fig_erosion_sensitivity.png

Requires ultralytics, torch and sam2, and a CUDA GPU for practical runtimes.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P


import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from photometric_analysis import (
    IMAGE_EXTENSIONS, MAX_LEVEL, AnnulusConfig, icc_2_1, load_paths,
    load_polygon_detections, occasion_matrix, polygon_mask, polygon_to_pixels,
    relative_luminance, sample_sort_key, tissue_mask_from_rgb,
)

BOUNDARY_BAND_PX = 2

CONF_THRESHOLD = 0.25
INFER_IMGSZ = 1024

AREA_BIN_EDGES = [0, 250, 500, 1000, 2000, np.inf]
AREA_BIN_LABELS = ["<250", "250-500", "500-1000", "1000-2000", ">2000"]

EROSION_LEVELS = [0, 1, 2, 3]

BOX_GREY = "0.72"
DETECTED_GREY = "0.32"
IOU50_GREY = "0.78"


def boundary_band(mask: np.ndarray, d: int = BOUNDARY_BAND_PX) -> np.ndarray:
    """Pixels of `mask` lying within d px of its own boundary."""
    if not mask.any():
        return mask
    return mask & (ndimage.distance_transform_edt(mask) <= d)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def boundary_iou(gt: np.ndarray, pred: np.ndarray, d: int = BOUNDARY_BAND_PX) -> float:
    return iou(boundary_band(gt, d), boundary_band(pred, d))


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode by a Euclidean distance, so the operation is isotropic."""
    if px <= 0:
        return mask
    return mask & (ndimage.distance_transform_edt(mask) > px)


def match_masks(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """
    One-to-one greedy-optimal matching of predictions to ground truth by IoU.

    Uses the Hungarian algorithm on the negated IoU matrix so that the total
    matched IoU is maximised; pairs with zero overlap are then discarded.
    """
    if not gt_masks or not pred_masks:
        return []
    def bbox(m):
        ys, xs = np.nonzero(m)
        return (ys.min(), ys.max(), xs.min(), xs.max()) if ys.size else (0, -1, 0, -1)

    gb = [bbox(m) for m in gt_masks]
    pb = [bbox(m) for m in pred_masks]
    cost = np.zeros((len(gt_masks), len(pred_masks)))
    for i, (y0, y1, x0, x1) in enumerate(gb):
        for j, (q0, q1, r0, r1) in enumerate(pb):
            if y1 < q0 or q1 < y0 or x1 < r0 or r1 < x0:
                continue
            cost[i, j] = iou(gt_masks[i], pred_masks[j])
    rows, cols = linear_sum_assignment(-cost)
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] > 0]


def load_yolo(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def yolo_masks(model, image_path: Path, shape: tuple[int, int]) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Predicted instance masks and box centres, resampled to the image grid."""
    res = model.predict(str(image_path), conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False)[0]
    if res.masks is None:
        return [], []
    h, w = shape
    data = res.masks.data.cpu().numpy()
    masks = []
    for m in data:
        if m.shape != (h, w):
            m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 127
        else:
            m = m > 0.5
        masks.append(m)
    boxes = res.boxes.xyxy.cpu().numpy()
    centres = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    return masks, centres


def build_sam_predictor(checkpoint: Path, config: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def sam_refine(predictor, rgb: np.ndarray, centres: list[tuple[float, float]]) -> list[np.ndarray]:
    """Refine each YOLO box centre into a mask, as the phenotyping pipeline does."""
    if not centres:
        return []
    predictor.set_image(rgb)
    out = []
    for cx, cy in centres:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            masks, _, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
            )
        out.append(masks[0].astype(bool))
    return out


def operculum_mask(model, image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Largest predicted operculum region, or the whole frame if none is found."""
    h, w = shape
    res = model.predict(str(image_path), conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return np.ones((h, w), bool)
    data = res.masks.data.cpu().numpy()
    m = data[int(np.argmax([(x > 0.5).sum() for x in data]))]
    if m.shape != (h, w):
        m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 127
    else:
        m = m > 0.5
    return m


def inside(mask: np.ndarray, region: np.ndarray, frac: float = 0.5) -> bool:
    a = mask.sum()
    return bool(a and (mask & region).sum() / a >= frac)


def evaluate_test_split(
    test_images: dict[str, Path],
    test_labels: dict[str, list[list[float]]],
    configs: dict[str, dict],
    operculum_model=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-spot IoU for each pipeline configuration, plus per-image counts.

    The phenotyping pipeline segments the operculum first and only then looks
    for spots, so the evaluation is restricted to the operculum region as well.
    Without this restriction, spots the model correctly finds elsewhere on the
    head are counted as false positives, which the deployed pipeline would
    never see.
    """
    spot_rows: list[dict] = []
    image_rows: list[dict] = []

    keys = sorted(set(test_images) & set(test_labels))
    for n, key in enumerate(keys, 1):
        rgb = np.array(Image.open(test_images[key]).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        region = (operculum_mask(operculum_model, test_images[key], (h, w))
                  if operculum_model is not None else np.ones((h, w), bool))
        gt_masks = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in test_labels[key]]
        gt_masks = [m for m in gt_masks if m.any() and inside(m, region)]

        for cfg_name, cfg in configs.items():
            masks, centres = yolo_masks(cfg["yolo"], test_images[key], (h, w))
            if cfg.get("sam") is not None:
                masks = sam_refine(cfg["sam"], rgb, centres)
            masks = [m for m in masks if m.any() and inside(m, region)]

            matches = match_masks(gt_masks, masks)
            matched_gt = {g for g, _, _ in matches}
            matched_pred = {p for _, p, _ in matches}

            for g, p, i in matches:
                spot_rows.append({
                    "config": cfg_name, "image": key, "gt_index": g,
                    "gt_area_px": int(gt_masks[g].sum()),
                    "pred_area_px": int(masks[p].sum()),
                    "iou": i,
                    "boundary_iou": boundary_iou(gt_masks[g], masks[p]),
                    "area_ratio": float(masks[p].sum() / gt_masks[g].sum()),
                    "detected": True,
                })
            for g in range(len(gt_masks)):
                if g not in matched_gt:
                    spot_rows.append({
                        "config": cfg_name, "image": key, "gt_index": g,
                        "gt_area_px": int(gt_masks[g].sum()), "pred_area_px": 0,
                        "iou": 0.0, "boundary_iou": 0.0, "area_ratio": 0.0,
                        "detected": False,
                    })
            image_rows.append({
                "config": cfg_name, "image": key,
                "n_gt": len(gt_masks), "n_pred": len(masks),
                "n_matched": len(matches),
                "n_false_positive": len(masks) - len(matched_pred),
            })
        print(f"  [{n:>2}/{len(keys)}] {key[:44]}: {len(gt_masks)} GT spots")

    return pd.DataFrame(spot_rows), pd.DataFrame(image_rows)


def stratify_by_size(spots: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of IoU per ground-truth size bin, per configuration."""
    df = spots.copy()
    df["size_bin"] = pd.cut(df["gt_area_px"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS,
                            right=False)
    rows: list[dict] = []
    for (cfg, b), g in df.groupby(["config", "size_bin"], observed=True):
        det = g[g["detected"]]
        rows.append({
            "config": cfg,
            "size_bin_px": str(b),
            "n_gt": len(g),
            "pct_of_gt": 100.0 * len(g) / len(df[df["config"] == cfg]),
            "detection_rate": 100.0 * len(det) / len(g) if len(g) else np.nan,
            "recall_at_iou50": 100.0 * (g["iou"] >= 0.5).sum() / len(g) if len(g) else np.nan,
            "iou_median": det["iou"].median() if len(det) else np.nan,
            "iou_q1": det["iou"].quantile(0.25) if len(det) else np.nan,
            "iou_q3": det["iou"].quantile(0.75) if len(det) else np.nan,
            "boundary_iou_median": det["boundary_iou"].median() if len(det) else np.nan,
            "boundary_iou_q1": det["boundary_iou"].quantile(0.25) if len(det) else np.nan,
            "boundary_iou_q3": det["boundary_iou"].quantile(0.75) if len(det) else np.nan,
            "area_ratio_median": det["area_ratio"].median() if len(det) else np.nan,
        })
    return pd.DataFrame(rows)


def gt_context_on_test_split(
    test_images: dict[str, Path],
    test_labels: dict[str, list[list[float]]],
    operculum_model,
) -> pd.DataFrame:
    """
    Per ground-truth spot: size relative to the operculum, and local contrast.

    Two things the raw pixel areas cannot tell us. First, the test images are
    whole-head frames whereas the phenotyping images are operculum crops resized
    to the same 1080 x 1080 grid, so absolute pixel areas are not comparable
    between the two; expressing spot area as a fraction of operculum area puts
    them on a common scale. Second, the qualitative review suggests misses are
    driven by faintness rather than by size, which needs a contrast measure to
    test.
    """
    rows: list[dict] = []
    for key in sorted(set(test_images) & set(test_labels)):
        rgb = np.array(Image.open(test_images[key]).convert("RGB"), dtype=np.uint8)
        lum = relative_luminance(rgb)
        h, w = rgb.shape[:2]

        res = operculum_model.predict(str(test_images[key]), conf=CONF_THRESHOLD,
                                      imgsz=INFER_IMGSZ, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            operculum = np.ones((h, w), bool)
            found = False
        else:
            data = res.masks.data.cpu().numpy()
            areas = [(m > 0.5).sum() for m in data]
            m = data[int(np.argmax(areas))]
            if m.shape != (h, w):
                m = np.array(Image.fromarray((m * 255).astype(np.uint8))
                             .resize((w, h), Image.NEAREST)) > 127
            else:
                m = m > 0.5
            operculum = m
            found = True
        operculum_area = int(operculum.sum())

        gts = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in test_labels[key]]
        gts = [m for m in gts if m.any()]
        all_spots = np.zeros((h, w), bool)
        for m in gts:
            all_spots |= m
        skin_pool = operculum & (ndimage.distance_transform_edt(~all_spots) > 4)

        for gi, gt in enumerate(gts):
            ys, xs = np.nonzero(gt)
            pad = 24
            y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
            sub_d = ndimage.distance_transform_edt(~gt[y0:y1, x0:x1])
            ann = skin_pool[y0:y1, x0:x1] & (sub_d > 4) & (sub_d <= 18)
            skin = float(np.median(lum[y0:y1, x0:x1][ann])) if ann.any() else np.nan
            spot = float(lum[gt].mean())
            rows.append({
                "image": key, "gt_index": gi,
                "gt_area_px": int(gt.sum()),
                "operculum_area_px": operculum_area,
                "operculum_found": found,
                "gt_area_pct_of_operculum": 100.0 * gt.sum() / operculum_area if operculum_area else np.nan,
                "gt_log_ratio_linear": float(np.log(spot / skin))
                if np.isfinite(skin) and skin > 0 and spot > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def detection_vs_contrast(spots: pd.DataFrame, context: pd.DataFrame, headline: str) -> pd.DataFrame:
    """Are missed spots fainter than detected ones, at matched size?"""
    df = spots[spots["config"] == headline].merge(context, on=["image", "gt_index"], how="left",
                                                  suffixes=("", "_ctx"))
    df["size_bin"] = pd.cut(df["gt_area_px"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS, right=False)
    rows: list[dict] = []
    for b, g in df.groupby("size_bin", observed=True):
        det = g[g["detected"]]["gt_log_ratio_linear"].dropna()
        mis = g[~g["detected"]]["gt_log_ratio_linear"].dropna()
        row = {"size_bin_px": str(b), "n_detected": len(det), "n_missed": len(mis),
               "contrast_detected": det.median() if len(det) else np.nan,
               "contrast_missed": mis.median() if len(mis) else np.nan}
        if len(det) >= 3 and len(mis) >= 3:
            from scipy import stats as sps
            u, p = sps.mannwhitneyu(mis, det, alternative="greater")
            row["p_missed_fainter"] = float(p)
        else:
            row["p_missed_fainter"] = np.nan
        rows.append(row)
    overall = {"size_bin_px": "all"}
    det = df[df["detected"]]["gt_log_ratio_linear"].dropna()
    mis = df[~df["detected"]]["gt_log_ratio_linear"].dropna()
    overall.update({"n_detected": len(det), "n_missed": len(mis),
                    "contrast_detected": det.median(), "contrast_missed": mis.median()})
    if len(det) >= 3 and len(mis) >= 3:
        from scipy import stats as sps
        overall["p_missed_fainter"] = float(sps.mannwhitneyu(mis, det, alternative="greater")[1])
    rows.append(overall)
    return pd.DataFrame(rows)


def erosion_sensitivity(
    image_paths: dict[str, Path],
    detections: dict[str, list[list[float]]],
    cfg: AnnulusConfig,
) -> pd.DataFrame:
    """Spot intensity and skin-referenced log-ratio at each erosion level."""
    rows: list[dict] = []
    keys = sorted(set(image_paths) & set(detections), key=sample_sort_key)
    for n, key in enumerate(keys, 1):
        rgb = np.array(Image.open(image_paths[key]).convert("RGB"), dtype=np.uint8)
        gray = np.array(Image.open(image_paths[key]).convert("L"), dtype=np.uint8).astype(float) / MAX_LEVEL
        lum = relative_luminance(rgb)
        h, w = gray.shape
        tissue = tissue_mask_from_rgb(rgb)

        masks = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in detections[key]]
        all_spots = np.zeros((h, w), bool)
        for m in masks:
            all_spots |= m
        skin_pool = tissue & (ndimage.distance_transform_edt(~all_spots) > cfg.gap)

        for idx, mask in enumerate(masks):
            core0 = mask & tissue
            if not core0.any():
                continue
            ys, xs = np.nonzero(mask)
            pad = int(cfg.gap + cfg.width) + 6
            y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
            sub_d = ndimage.distance_transform_edt(~mask[y0:y1, x0:x1])
            ann = skin_pool[y0:y1, x0:x1] & (sub_d > cfg.gap) & (sub_d <= cfg.gap + cfg.width)
            skin_lum = float(np.median(lum[y0:y1, x0:x1][ann])) if ann.any() else np.nan

            base_area = int(core0.sum())
            for e in EROSION_LEVELS:
                core = erode(mask, e) & tissue
                n_px = int(core.sum())
                if n_px == 0:
                    rows.append({
                        "sample": key, "spot": idx, "erosion_px": e,
                        "base_area_px": base_area, "area_px": 0,
                        "spot_mean": np.nan, "spot_median": np.nan,
                        "log_ratio_linear": np.nan, "log_ratio_median": np.nan, "lost": True,
                    })
                    continue
                valid_skin = np.isfinite(skin_lum) and skin_lum > 0
                lr = np.log(lum[core].mean() / skin_lum) if valid_skin else np.nan
                lr_med = np.log(np.median(lum[core]) / skin_lum) if valid_skin else np.nan
                rows.append({
                    "sample": key, "spot": idx, "erosion_px": e,
                    "base_area_px": base_area, "area_px": n_px,
                    "spot_mean": float(gray[core].mean()),
                    "spot_median": float(np.median(gray[core])),
                    "log_ratio_linear": float(lr) if np.isfinite(lr) else np.nan,
                    "log_ratio_median": float(lr_med) if np.isfinite(lr_med) else np.nan,
                    "lost": False,
                })
        if n % 20 == 0:
            print(f"  {n}/{len(keys)} images")
    return pd.DataFrame(rows)


def summarise_erosion(ero: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change relative to no erosion, overall and by spot size."""
    base = ero[ero["erosion_px"] == 0].set_index(["sample", "spot"])
    merged = ero.join(
        base[["spot_mean", "log_ratio_linear"]].rename(
            columns={"spot_mean": "base_mean", "log_ratio_linear": "base_lr"}),
        on=["sample", "spot"])
    merged["d_intensity"] = merged["spot_mean"] - merged["base_mean"]
    merged["d_log_ratio"] = merged["log_ratio_linear"] - merged["base_lr"]
    merged["size_bin"] = pd.cut(merged["base_area_px"], bins=AREA_BIN_EDGES,
                                labels=AREA_BIN_LABELS, right=False)

    overall = (merged.groupby("erosion_px")
               .agg(n_spots=("spot", "size"),
                    n_lost=("lost", "sum"),
                    pct_lost=("lost", lambda s: 100.0 * s.mean()),
                    mean_intensity=("spot_mean", "mean"),
                    d_intensity=("d_intensity", "mean"),
                    d_log_ratio=("d_log_ratio", "mean"),
                    median_area_retained=("area_px", "median"))
               .reset_index())

    by_size = (merged.groupby(["size_bin", "erosion_px"], observed=True)
               .agg(n=("spot", "size"),
                    pct_lost=("lost", lambda s: 100.0 * s.mean()),
                    d_intensity=("d_intensity", "mean"),
                    d_intensity_sd=("d_intensity", "std"),
                    d_log_ratio=("d_log_ratio", "mean"))
               .reset_index())
    return overall, by_size


def erosion_downstream(ero: pd.DataFrame, per_image: pd.DataFrame) -> pd.DataFrame:
    """Does erosion change the headline repeatability and pre/post results?"""
    from scipy import stats as sps

    meta = per_image[["sample", "FISH_ID", "DAY", "SIDE"]].astype(str)
    rows: list[dict] = []
    for e in EROSION_LEVELS:
        sub = ero[(ero["erosion_px"] == e) & (~ero["lost"])]
        agg = (sub.groupby("sample", as_index=False)
               .agg(raw=("spot_mean", "mean"), lr=("log_ratio_linear", "mean")))
        img = meta.merge(agg, on="sample", how="inner")
        for col, label in [("raw", "Raw mean spot intensity"),
                           ("lr", "Skin-referenced log-ratio (linearised)")]:
            mat, _ = occasion_matrix(img, col)
            res = icc_2_1(mat)
            t, p = sps.ttest_rel(mat[:, 1], mat[:, 0])
            rows.append({
                "erosion_px": e, "measure": label, "n_units": res["n"],
                "ICC(2,1)": res["icc"],
                "prepost_change": float(np.mean(mat[:, 1] - mat[:, 0])),
                "prepost_p": float(p),
                "prepost_dz": float(np.mean(mat[:, 1] - mat[:, 0]) / np.std(mat[:, 1] - mat[:, 0], ddof=1)),
            })
    return pd.DataFrame(rows)


def figure_iou_by_size(strat: pd.DataFrame, spots: pd.DataFrame, out_path: Path, headline: str) -> None:
    df = spots[spots["config"] == headline].copy()
    df["size_bin"] = pd.cut(df["gt_area_px"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS, right=False)
    det = df[df["detected"]]
    labels = [b for b in AREA_BIN_LABELS if (df["size_bin"] == b).any()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, label) in zip(axes[:2], [("iou", "Mask IoU"),
                                           ("boundary_iou", f"Boundary IoU (d = {BOUNDARY_BAND_PX} px)")]):
        data = [det[det["size_bin"] == b][col].dropna().to_numpy() for b in labels]
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_GREY)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2.0)
        ax.axhline(0.5, ls="--", lw=1.0, color="black")
        ax.set_xlabel("Ground-truth spot area (px)")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)

    s = strat[strat["config"] == headline].set_index("size_bin_px").reindex(labels)
    ax = axes[2]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, s["detection_rate"], width=0.4, label="detected (any match)",
           color=DETECTED_GREY, edgecolor="black", linewidth=0.7)
    ax.bar(x + 0.2, s["recall_at_iou50"], width=0.4, label="detected with IoU ≥ 0.5",
           color=IOU50_GREY, edgecolor="black", linewidth=0.7, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{int(n)} GT ({pct:.0f}%)"
                        for b, n, pct in zip(labels, s["n_gt"], s["pct_of_gt"])],
                       fontsize=8)
    ax.set_xlabel("Ground-truth spot area (px)")
    ax.set_ylabel("% of ground-truth spots")
    ax.set_ylim(0, 105)
    ax.set_title("Detection rate by ground-truth size", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_erosion(overall: pd.DataFrame, by_size: pd.DataFrame, out_path: Path) -> None:
    """
    Erosion sensitivity, in grayscale.

    The first two panels show the same five spot-size bins, which are ordered,
    so they take a dark-to-light grey ramp that encodes that ordering, with a
    distinct marker and line style per bin so they stay separable where the
    curves converge. The third panel compares two different measures rather than
    an ordered series, so it uses tone, style and marker contrast instead.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    labels = [b for b in AREA_BIN_LABELS if (by_size["size_bin"] == b).any()]

    greys = [str(round(v, 3)) for v in np.linspace(0.10, 0.68, max(len(labels), 1))]
    markers = ["o", "s", "^", "D", "v"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    series = list(zip(labels, greys, markers, styles))

    for ax, col, ylabel, title in [
        (axes[0], "d_intensity", "Δ mean spot intensity vs 0 px",
         "Edge contamination by spot size"),
        (axes[1], "pct_lost", "% of spots lost entirely",
         "Cost of erosion: small spots vanish"),
    ]:
        for b, grey, marker, style in series:
            g = by_size[by_size["size_bin"] == b]
            if g.empty:
                continue
            ax.plot(g["erosion_px"], g[col], color=grey, ls=style, marker=marker,
                    lw=1.8, markersize=6, markerfacecolor="white",
                    markeredgecolor=grey, markeredgewidth=1.5, label=b)
        ax.set_xlabel("Erosion (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(EROSION_LEVELS)
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.legend(title="GT area (px)", fontsize=8, title_fontsize=8)
    axes[0].axhline(0, color="black", lw=1.0, ls="--")

    ax = axes[2]
    ax.plot(overall["erosion_px"], overall["d_intensity"], marker="o", ls="-",
            color="0.20", lw=2.0, markersize=6, markerfacecolor="white",
            markeredgecolor="0.20", markeredgewidth=1.6, label="Δ raw intensity")
    ax.plot(overall["erosion_px"], overall["d_log_ratio"], marker="^", ls="--",
            color="0.55", lw=2.0, markersize=7, markerfacecolor="white",
            markeredgecolor="0.55", markeredgewidth=1.6,
            label="Δ skin-ref log-ratio")
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Erosion (px)")
    ax.set_ylabel("Mean change vs 0 px")
    ax.set_title("Pooled effect of erosion", fontsize=10)
    ax.set_xticks(EROSION_LEVELS)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_whole_frame_detection(
    test_images: dict[str, Path],
    test_labels: dict[str, list[list[float]]],
    configs: dict[str, dict],
    operculum_model,
    headline: str,
    out_path: Path,
) -> None:
    """
    Show why the evaluation is restricted to the operculum.

    The test images are whole-head frames, but the deployed pipeline segments
    the operculum first and only looks for spots inside it. Detections elsewhere
    on the head are correct findings the pipeline never sees; counted as false
    positives they inflate the apparent error rate several-fold. A single frame
    makes that concrete in a way the numbers alone do not.

    The frame shown is the one whose share of out-of-operculum detections is
    closest to the median, so it is representative rather than selected to
    flatter or to alarm.
    """
    cfg = configs[headline]

    shares = []
    for key in sorted(set(test_images) & set(test_labels)):
        rgb_shape = Image.open(test_images[key]).size[::-1]
        region = operculum_mask(operculum_model, test_images[key], rgb_shape)
        masks, _ = yolo_masks(cfg["yolo"], test_images[key], rgb_shape)
        masks = [m for m in masks if m.any()]
        if not masks:
            continue
        outside = sum(1 for m in masks if not inside(m, region))
        shares.append((outside / len(masks), key))
    if not shares:
        return
    shares.sort()
    _, key = shares[len(shares) // 2]

    rgb = np.array(Image.open(test_images[key]).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    region = operculum_mask(operculum_model, test_images[key], (h, w))
    masks, centres = yolo_masks(cfg["yolo"], test_images[key], (h, w))
    if cfg.get("sam") is not None:
        masks = sam_refine(cfg["sam"], rgb, centres)
    masks = [m for m in masks if m.any()]

    gts_all = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in test_labels[key]]
    gts_all = [m for m in gts_all if m.any()]
    gts = [m for m in gts_all if inside(m, region)]
    preds_in = [m for m in masks if inside(m, region)]
    preds_out = [m for m in masks if not inside(m, region)]

    matches = match_masks(gts, preds_in)
    matched_gt = {g for g, _, _ in matches}
    matched_pred = {p for _, p, _ in matches}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))

    ax = axes[0]
    ax.imshow(rgb)
    ax.contour(region.astype(float), levels=[0.5], colors="#ffd400", linewidths=2.4)
    for m in preds_in:
        ax.contour(m.astype(float), levels=[0.5], colors="#1b9e77", linewidths=1.1)
    for m in preds_out:
        ax.contour(m.astype(float), levels=[0.5], colors="#e7298a", linewidths=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"All {len(masks)} detections on the whole head frame\n"
        f"{len(preds_in)} inside the operculum (green), "
        f"{len(preds_out)} elsewhere on the head (pink)",
        fontsize=10)
    handles = [
        mpatches.Patch(color="#ffd400", label="operculum boundary"),
        mpatches.Patch(color="#1b9e77", label="detection inside operculum (evaluated)"),
        mpatches.Patch(color="#e7298a", label="detection elsewhere (not evaluated)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    ys, xs = np.nonzero(region)
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    ax = axes[1]
    ax.imshow(rgb[y0:y1, x0:x1])
    ax.contour(region[y0:y1, x0:x1].astype(float), levels=[0.5],
               colors="#ffd400", linewidths=2.0)
    for gi, g in enumerate(gts):
        colour = "#00e5ff" if gi in matched_gt else "#d95f02"
        ax.contour(g[y0:y1, x0:x1].astype(float), levels=[0.5],
                   colors=colour, linewidths=1.5)
    for pi, m in enumerate(preds_in):
        if pi in matched_pred:
            ax.contour(m[y0:y1, x0:x1].astype(float), levels=[0.5],
                       colors="#1b9e77", linewidths=1.1, linestyles="--")
        else:
            ax.contour(m[y0:y1, x0:x1].astype(float), levels=[0.5],
                       colors="#e7298a", linewidths=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    n_missed = len(gts) - len(matched_gt)
    n_fp = len(preds_in) - len(matched_pred)
    ax.set_title(
        f"Evaluated region: {len(gts)} ground-truth spots\n"
        f"{len(matches)} matched, {n_missed} missed, {n_fp} false positive",
        fontsize=10)
    handles = [
        mpatches.Patch(color="#00e5ff", label="ground truth, matched"),
        mpatches.Patch(color="#d95f02", label="ground truth, missed"),
        mpatches.Patch(color="#1b9e77", label="prediction, matched"),
        mpatches.Patch(color="#e7298a", label="prediction, unmatched"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_qualitative(
    spots: pd.DataFrame, test_images: dict[str, Path],
    test_labels: dict[str, list[list[float]]], configs: dict[str, dict],
    headline: str, out_path: Path,
) -> None:
    """Three successes, three challenging cases, three failures."""
    df = spots[spots["config"] == headline].copy()
    det = df[df["detected"]]
    successes = det.nlargest(60, "iou").sample(3, random_state=7)
    challenging = det[(det["iou"] > 0.15) & (det["iou"] < 0.55)].nsmallest(3, "gt_area_px")
    failures = df[~df["detected"]].nlargest(3, "gt_area_px")
    groups = [("Successful", successes, "#1b9e77"),
              ("Challenging", challenging, "#e6ab02"),
              ("Failure (missed)", failures, "#d95f02")]

    cache: dict[str, tuple] = {}

    def predictions_for(key: str):
        if key not in cache:
            rgb = np.array(Image.open(test_images[key]).convert("RGB"), dtype=np.uint8)
            h, w = rgb.shape[:2]
            cfg = configs[headline]
            masks, centres = yolo_masks(cfg["yolo"], test_images[key], (h, w))
            if cfg.get("sam") is not None:
                masks = sam_refine(cfg["sam"], rgb, centres)
            gts = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in test_labels[key]]
            gts = [m for m in gts if m.any()]
            cache[key] = (rgb, gts, [m for m in masks if m.any()])
        return cache[key]

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for r, (title, sel, colour) in enumerate(groups):
        for c in range(3):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(sel):
                ax.axis("off")
                continue
            row = sel.iloc[c]
            rgb, gts, preds = predictions_for(row["image"])
            gt = gts[int(row["gt_index"])]
            ys, xs = np.nonzero(gt)
            pad = 45
            y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
            crop = rgb[y0:y1, x0:x1]
            ax.imshow(crop)
            ax.contour(gt[y0:y1, x0:x1].astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.9)
            best, best_i = None, 0.0
            for m in preds:
                v = iou(gt, m)
                if v > best_i:
                    best, best_i = m, v
            if best is not None and best_i > 0:
                ax.contour(best[y0:y1, x0:x1].astype(float), levels=[0.5], colors="#ff1744",
                           linewidths=1.6, linestyles="--")
            ax.set_title(f"{title}\nGT area {int(row['gt_area_px'])} px, IoU {row['iou']:.2f}",
                         fontsize=9, color=colour)
    handles = [mpatches.Patch(color="#00e5ff", label="Ground truth"),
               mpatches.Patch(color="#ff1744", label="Predicted (SAM 2.1 refined)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spots-dir", type=Path, default=P.SPOTS_DATASET_DIR)
    p.add_argument("--images-dir", type=Path, default=P.IMAGES_DIR)
    p.add_argument("--labels-dir", type=Path, default=P.LABELS_DIR)
    p.add_argument("--weights-dir", type=Path, default=P.WEIGHTS_DIR)
    p.add_argument("--output-dir", type=Path, default=P.RESULTS_DIR)
    p.add_argument("--sam-config", default=P.SAM_CONFIG)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-erosion", action="store_true")
    p.add_argument("--reuse-iou", action="store_true",
                   help="Reload the cached test-split IoU table instead of re-running inference.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"

    test_images = load_paths(args.spots_dir / "test" / "images", IMAGE_EXTENSIONS)
    test_labels = load_polygon_detections(args.spots_dir / "test" / "labels")
    print(f"Test split: {len(test_images)} images, "
          f"{sum(len(v) for v in test_labels.values())} ground-truth spots")

    print("\n[1/3] Per-spot IoU on the test split")
    sam = build_sam_predictor(args.weights_dir / "sam_checkpoint.pt", args.sam_config, device)
    configs = {
        "YOLOv8-seg (raw)": {"yolo": load_yolo(args.weights_dir / "yolov8_spots.pt"), "sam": None},
        "YOLOv8-seg + SAM 2.1": {"yolo": load_yolo(args.weights_dir / "yolov8_spots.pt"), "sam": sam},
        "YOLOv11-seg + SAM 2.1": {"yolo": load_yolo(args.weights_dir / "yolov11_spots.pt"), "sam": sam},
    }
    operculum = load_yolo(args.weights_dir / "yolov8_operculum.pt")
    iou_cache = out / "data" / "test_split_spot_iou.csv"
    if args.reuse_iou and iou_cache.exists():
        print(f"  reusing cached {iou_cache.name}")
        spots = pd.read_csv(iou_cache)
        per_image_counts = pd.read_csv(out / "data" / "test_split_image_counts.csv")
    else:
        spots, per_image_counts = evaluate_test_split(test_images, test_labels, configs, operculum)
        spots.to_csv(iou_cache, index=False)
        per_image_counts.to_csv(out / "data" / "test_split_image_counts.csv", index=False)

    strat = stratify_by_size(spots)
    strat.to_csv(out / "data" / "iou_by_spot_size.csv", index=False)
    print()
    print(strat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    headline = "YOLOv8-seg + SAM 2.1"

    print("\n  Ground-truth context: operculum-relative size and local contrast")
    context = gt_context_on_test_split(test_images, test_labels, operculum)
    context.to_csv(out / "data" / "test_split_gt_context.csv", index=False)
    contrast = detection_vs_contrast(spots, context, headline)
    contrast.to_csv(out / "data" / "detection_vs_contrast.csv", index=False)
    print(contrast.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  median GT spot = {context['gt_area_pct_of_operculum'].median():.3f}% of operculum area")

    figure_whole_frame_detection(
        test_images, test_labels, configs, operculum, headline,
        out / "figures" / "fig_whole_frame_detection.png")
    figure_iou_by_size(strat, spots, out / "figures" / "fig_iou_by_size.png", headline)
    figure_qualitative(spots, test_images, test_labels, configs, headline,
                       out / "figures" / "fig_qualitative_segmentation.png")

    if not args.skip_erosion:
        print("\n[2/3] Erosion sensitivity on the phenotyping set")
        image_paths = load_paths(args.images_dir, IMAGE_EXTENSIONS)
        detections = load_polygon_detections(args.labels_dir)
        ero = erosion_sensitivity(image_paths, detections, AnnulusConfig())
        ero.to_csv(out / "data" / "erosion_spotwise.csv", index=False)
        overall, by_size = summarise_erosion(ero)
        overall.to_csv(out / "data" / "erosion_overall.csv", index=False)
        by_size.to_csv(out / "data" / "erosion_by_size.csv", index=False)
        print()
        print(overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        per_image = pd.read_csv(out / "data" / "image_level_measures.csv",
                                dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
        downstream = erosion_downstream(ero, per_image)
        downstream.to_csv(out / "data" / "erosion_downstream.csv", index=False)
        print()
        print(downstream.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        figure_erosion(overall, by_size, out / "figures" / "fig_erosion_sensitivity.png")

    print("\n[3/3] Done")
    summary = {
        "n_test_images": len(test_images),
        "n_test_gt_spots": int(sum(len(v) for v in test_labels.values())),
        "boundary_band_px": BOUNDARY_BAND_PX,
        "conf_threshold": CONF_THRESHOLD,
        "infer_imgsz": INFER_IMGSZ,
        "headline_config": headline,
        "device": device,
    }
    (out / "data" / "segmentation_validation_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
