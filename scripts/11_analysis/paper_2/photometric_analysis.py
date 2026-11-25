#!/usr/bin/env python3
"""
Photometric revision analysis for the opercular spot dataset.

Addresses the referee's concern that raw spot grayscale intensity is confounded
with per-image exposure and illumination. Implements:

  1. Skin-referenced spot intensity (annulus of surrounding non-spot opercular
     skin; spot intensity expressed as a ratio and as a difference).
  2. Background-referenced exposure check (mean channel values of the blue
     background board in the original photographs, plus EXIF exposure settings,
     used as independent exposure proxies).
  3. Session repeatability, ICC(2,1), across the two imaging occasions.

Inputs
------
  images/   1080x1080 background-masked opercular crops (measurement images)
  labels/   matching YOLO-style polygon label files
  head_images/  the uncropped camera frames, which still contain the blue board

Run:
  python photometric_analysis.py --output-dir results/paper_2

Datasets
--------
Reads   images/            84 opercular crops (P.IMAGES_DIR)
        labels/            84 polygon label files, 2,615 spots (P.LABELS_DIR)
        head_images/       84 uncropped camera frames retaining the blue board
                           and EXIF, the source of the independent exposure
                           proxy (P.ORIGINALS_DIR; the board stage is skipped
                           if the directory is absent)
Writes  results/paper_2/data/spot_level_skin_referenced.csv
        results/paper_2/data/image_level_measures.csv
        results/paper_2/data/background_board_exposure.csv
        results/paper_2/data/exposure_tracking.csv
        results/paper_2/data/exposure_tracking_within_fish.csv
        results/paper_2/data/icc_session_repeatability.csv
        results/paper_2/data/icc_sensitivity_by_side.csv
        results/paper_2/data/provenance.json
        results/paper_2/figures/fig_annulus_method.png
        results/paper_2/figures/fig_exposure_tracking.png
        results/paper_2/figures/fig_icc_occasions.png
        results/paper_2/figures/fig_dispersion.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P


import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageDraw
from scipy import ndimage, stats


GRAYSCALE_WEIGHTS = (0.299, 0.587, 0.114)
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
BIT_DEPTH = 8
MAX_LEVEL = 2 ** BIT_DEPTH - 1

BACKGROUND_LEVEL = 10

DEFAULT_ANNULUS_GAP = 4.0
DEFAULT_ANNULUS_WIDTH = 14.0
MIN_ANNULUS_PIXELS = 250
MAX_ANNULUS_GROWTH_STEPS = 4
ANNULUS_GROWTH_STEP = 8.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

RAW_GREY = "0.30"
REFERENCED_GREY = "0.78"
POINT_GREY = "0.55"

BOARD_HUE_RANGE = (150.0, 250.0)
BOARD_MIN_SATURATION = 0.25
BOARD_VALUE_RANGE = (0.05, 0.97)


def sample_key_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("_jpg")[0] if "_jpg" in stem else stem


def parse_sample_name(sample_name: str) -> tuple[str, str, str]:
    """fish_<id>_S_day<d>_id-<n>_<side>_h  ->  (fish_id, day, side)."""
    parts = sample_name.split("_")
    if len(parts) < 5:
        raise ValueError(f"Cannot parse sample name '{sample_name}'.")
    fish_id = parts[1]
    day_match = re.search(r"(\d+)$", parts[3])
    day = day_match.group(1) if day_match else parts[3][-1]
    side = parts[-2]
    return str(fish_id), str(day), str(side)


def sample_sort_key(sample_name: str) -> tuple[int, str, int, str]:
    try:
        fish_id, day, side = parse_sample_name(sample_name)
        return int(fish_id), side, int(day), sample_name
    except Exception:
        return 10 ** 9, "", 0, sample_name


def load_paths(directory: Path, allowed_extensions: set[str]) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    paths = {
        sample_key_from_path(p): p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in allowed_extensions
    }
    return dict(sorted(paths.items(), key=lambda kv: sample_sort_key(kv[0])))


def load_polygon_detections(label_dir: Path) -> dict[str, list[list[float]]]:
    detections: dict[str, list[list[float]]] = {}
    for path in sorted(label_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".csv"}:
            continue
        polygons: list[list[float]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            tokens = line.split()
            if len(tokens) <= 1:
                continue
            coords = [float(t) for t in tokens[1:]]
            if len(coords) % 2 != 0:
                raise ValueError(f"Odd coordinate count in {path} line {line_number}")
            polygons.append(coords)
        detections[sample_key_from_path(path)] = polygons
    return detections


def polygon_to_pixels(vertices, width: int, height: int) -> list[tuple[int, int]]:
    v = list(vertices)
    return [
        (
            int(np.clip(v[i] * width, 0, width - 1)),
            int(np.clip(v[i + 1] * height, 0, height - 1)),
        )
        for i in range(0, len(v), 2)
    ]


def polygon_mask(width: int, height: int, vertices) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(vertices, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def format_p(value: float, label: str = "p", threshold: float = 0.001) -> str:
    """
    Journal-style significance annotation for figures.

    Very small p-values carry no interpretable information beyond "below the
    threshold", and printing them in scientific notation (p = 5.3e-209) invites
    a reader to treat the exponent as an effect size. Reported as "p < 0.001"
    instead; exact values are retained in the CSV outputs and in the report
    tables, where they belong.
    """
    if value is None or not np.isfinite(value):
        return f"{label} = n/a"
    if value < threshold:
        return f"{label} < {threshold:g}"
    return f"{label} = {value:.3f}"


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """
    Inverse sRGB electro-optical transfer function, for v in [0, 1].

    A spot-to-skin ratio cancels a multiplicative exposure gain exactly only if
    the encoding is a power law. sRGB is a power law with a linear toe near
    black, so ratios taken on gamma-encoded values retain a residual exposure
    dependence. Undoing the transfer function first removes it.
    """
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Linear-domain relative luminance Y in [0, 1] from 8-bit sRGB."""
    linear = srgb_to_linear(rgb.astype(np.float64) / MAX_LEVEL)
    return linear @ np.array(LUMINANCE_WEIGHTS)


def tissue_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Opercular tissue mask for a background-masked crop.

    The upstream segmentation sets non-operculum pixels to zero, so tissue is
    simply "not black". Holes are filled first so that genuinely dark melanin
    spots inside the operculum are retained as tissue, then only the largest
    connected component is kept so that stray specks are dropped.
    """
    lit = rgb.max(axis=2) > BACKGROUND_LEVEL
    filled = ndimage.binary_fill_holes(lit)
    labelled, n = ndimage.label(filled)
    if n <= 1:
        return filled
    sizes = ndimage.sum(filled, labelled, index=np.arange(1, n + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


@dataclass
class AnnulusConfig:
    gap: float = DEFAULT_ANNULUS_GAP
    width: float = DEFAULT_ANNULUS_WIDTH


def spot_rows_for_image(
    sample_key: str,
    image_path: Path,
    polygons: list[list[float]],
    cfg: AnnulusConfig,
) -> tuple[list[dict], dict]:
    fish_id, day, side = parse_sample_name(sample_key)

    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.uint8).astype(float) / MAX_LEVEL
    lum = relative_luminance(rgb)
    height, width = gray.shape

    tissue = tissue_mask_from_rgb(rgb)

    spot_masks = [polygon_mask(width, height, polygon_to_pixels(p, width, height)) for p in polygons]
    all_spots = np.zeros((height, width), dtype=bool)
    for m in spot_masks:
        all_spots |= m

    dist_all = ndimage.distance_transform_edt(~all_spots)
    skin_pool = tissue & (dist_all > cfg.gap)

    image_skin_median = float(np.median(gray[skin_pool])) if skin_pool.any() else np.nan

    rows: list[dict] = []
    for spot_index, mask in enumerate(spot_masks):
        core = mask & tissue
        if not core.any():
            continue

        ys, xs = np.nonzero(mask)
        pad = int(np.ceil(cfg.gap + cfg.width + ANNULUS_GROWTH_STEP * MAX_ANNULUS_GROWTH_STEPS)) + 2
        y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)

        sub_mask = mask[y0:y1, x0:x1]
        sub_pool = skin_pool[y0:y1, x0:x1]
        sub_gray = gray[y0:y1, x0:x1]
        sub_lum = lum[y0:y1, x0:x1]
        sub_dist = ndimage.distance_transform_edt(~sub_mask)

        width_used = cfg.width
        annulus = sub_pool & (sub_dist > cfg.gap) & (sub_dist <= cfg.gap + width_used)
        grow_steps = 0
        while annulus.sum() < MIN_ANNULUS_PIXELS and grow_steps < MAX_ANNULUS_GROWTH_STEPS:
            width_used += ANNULUS_GROWTH_STEP
            annulus = sub_pool & (sub_dist > cfg.gap) & (sub_dist <= cfg.gap + width_used)
            grow_steps += 1

        n_annulus = int(annulus.sum())
        skin = float(np.median(sub_gray[annulus])) if n_annulus > 0 else np.nan
        skin_lum = float(np.median(sub_lum[annulus])) if n_annulus > 0 else np.nan

        spot_values = gray[core]
        spot_mean = float(spot_values.mean())
        spot_median = float(np.median(spot_values))
        spot_p10 = float(np.percentile(spot_values, 10))
        spot_lum = float(lum[core].mean())

        valid_skin = np.isfinite(skin) and skin > 0
        valid_lum = np.isfinite(skin_lum) and skin_lum > 0
        ratio = spot_mean / skin if valid_skin else np.nan
        ratio_lin = spot_lum / skin_lum if valid_lum else np.nan
        rows.append(
            {
                "sample": sample_key,
                "FISH_ID": fish_id,
                "DAY": day,
                "SIDE": side,
                "spot": spot_index,
                "area_px": int(core.sum()),
                "spot_mean": spot_mean,
                "spot_median": spot_median,
                "spot_p10": spot_p10,
                "skin_median": skin,
                "n_annulus_px": n_annulus,
                "annulus_width_used": width_used,
                "spot_luminance": spot_lum,
                "skin_luminance": skin_lum,
                "ratio_linear": ratio_lin,
                "log_ratio_linear": float(np.log(ratio_lin))
                if np.isfinite(ratio_lin) and ratio_lin > 0
                else np.nan,
                "ratio": ratio,
                "log_ratio": float(np.log(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan,
                "difference": spot_mean - skin if valid_skin else np.nan,
                "weber_contrast": (skin - spot_mean) / skin if valid_skin else np.nan,
                "darkening_integral": float(np.sum(skin - spot_values)) if valid_skin else np.nan,
            }
        )

    image_row = {
        "sample": sample_key,
        "FISH_ID": fish_id,
        "DAY": day,
        "SIDE": side,
        "tissue_px": int(tissue.sum()),
        "spot_px": int((all_spots & tissue).sum()),
        "image_skin_median": image_skin_median,
        "n_spots": len(rows),
        "labelled_spot_px": int(all_spots.sum()),
        "spot_px_outside_tissue": int((all_spots & ~tissue).sum()),
        "legacy_zero_px": int((gray[all_spots] == 0.0).sum()),
    }
    return rows, image_row


def compute_spot_table(
    image_paths: dict[str, Path],
    detections: dict[str, list[list[float]]],
    cfg: AnnulusConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot_rows: list[dict] = []
    image_rows: list[dict] = []
    keys = sorted(set(image_paths) & set(detections), key=sample_sort_key)
    for i, key in enumerate(keys, 1):
        rows, image_row = spot_rows_for_image(key, image_paths[key], detections[key], cfg)
        spot_rows.extend(rows)
        image_rows.append(image_row)
        print(f"  [{i:>2}/{len(keys)}] {key}: {len(rows)} spots")
    return pd.DataFrame(spot_rows), pd.DataFrame(image_rows)


def aggregate_per_image(spots: pd.DataFrame, image_meta: pd.DataFrame) -> pd.DataFrame:
    agg = (
        spots.groupby("sample", sort=False)
        .agg(
            raw_mean_intensity=("spot_mean", "mean"),
            raw_median_intensity=("spot_median", "mean"),
            skin_ref_ratio=("ratio", "mean"),
            skin_ref_log_ratio=("log_ratio", "mean"),
            skin_ref_log_ratio_linear=("log_ratio_linear", "mean"),
            skin_ref_ratio_linear=("ratio_linear", "mean"),
            skin_ref_difference=("difference", "mean"),
            weber_contrast=("weber_contrast", "mean"),
            mean_spot_area_px=("area_px", "mean"),
            darkening_integral=("darkening_integral", "sum"),
        )
        .reset_index()
    )
    merged = image_meta.merge(agg, on="sample", how="left")
    merged["spot_count"] = merged["n_spots"]
    merged["relative_spot_area_pct"] = 100.0 * merged["spot_px"] / merged["tissue_px"]
    merged["total_melanization"] = merged["darkening_integral"] / merged["tissue_px"]
    return merged


def rgb_to_hsv_arrays(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / MAX_LEVEL
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = arr.max(axis=2)
    c = v - arr.min(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(v > 0, c / v, 0.0)
        h = np.zeros_like(v)
        nz = c > 1e-6
        rmax = (v == r) & nz
        gmax = (v == g) & nz & ~rmax
        bmax = (v == b) & nz & ~rmax & ~gmax
        h[rmax] = ((g - b)[rmax] / c[rmax]) % 6.0
        h[gmax] = ((b - r)[gmax] / c[gmax]) + 2.0
        h[bmax] = ((r - g)[bmax] / c[bmax]) + 4.0
        h *= 60.0
    return h, s, v


EXIF_FIELDS = {
    "ExposureTime": "exposure_time_s",
    "FNumber": "f_number",
    "ISOSpeedRatings": "iso",
    "ExposureBiasValue": "exposure_bias_ev",
    "FocalLength": "focal_length_mm",
    "ExposureProgram": "exposure_program",
    "ExposureMode": "exposure_mode",
    "MeteringMode": "metering_mode",
    "WhiteBalance": "white_balance",
    "Flash": "flash",
}


def read_exif(path: Path) -> dict:
    out: dict[str, object] = {}
    try:
        exif = Image.open(path).getexif()
        base = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        ifd = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}
        for tag, column in EXIF_FIELDS.items():
            if tag in ifd:
                try:
                    out[column] = float(ifd[tag])
                except (TypeError, ValueError):
                    out[column] = ifd[tag]
        if base.get("DateTime"):
            out["capture_datetime"] = str(base["DateTime"])
        make = str(base.get("Make", "")).strip()
        model = str(base.get("Model", "")).strip()
        if make or model:
            out["camera"] = f"{make} {model}".strip()
    except Exception as exc:  # pragma: no cover - metadata is best effort
        warnings.warn(f"EXIF read failed for {path.name}: {exc}")
    return out


def board_measurements(path: Path) -> dict:
    """Mean/median channel values of the blue background board in one original."""
    image = Image.open(path)
    image.draft("RGB", (1000, 750))
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)

    hue, sat, val = rgb_to_hsv_arrays(rgb)
    board = (
        (hue >= BOARD_HUE_RANGE[0])
        & (hue <= BOARD_HUE_RANGE[1])
        & (sat >= BOARD_MIN_SATURATION)
        & (val >= BOARD_VALUE_RANGE[0])
        & (val <= BOARD_VALUE_RANGE[1])
    )
    out: dict[str, object] = {
        "board_pixel_fraction": float(board.mean()),
        "analysed_width": int(rgb.shape[1]),
        "analysed_height": int(rgb.shape[0]),
    }
    if board.sum() < 1000:
        for name in ("board_R", "board_G", "board_B", "board_luma", "board_luma_sd"):
            out[name] = np.nan
        return out

    px = rgb[board].astype(float)
    out["board_R"] = float(px[:, 0].mean())
    out["board_G"] = float(px[:, 1].mean())
    out["board_B"] = float(px[:, 2].mean())
    luma = px @ np.array(GRAYSCALE_WEIGHTS)
    out["board_luma"] = float(luma.mean())
    out["board_luma_sd"] = float(luma.std())

    labelled, n_components = ndimage.label(board)
    if n_components > 1:
        sizes = ndimage.sum(board, labelled, index=np.arange(1, n_components + 1))
        largest = labelled == (int(np.argmax(sizes)) + 1)
        full_luma = (rgb.astype(float) @ np.array(GRAYSCALE_WEIGHTS))
        out["board_luma_largest_cc"] = float(full_luma[largest].mean())
        out["board_stray_pct"] = float(100.0 * (board.sum() - largest.sum()) / board.sum())
    else:
        out["board_luma_largest_cc"] = out["board_luma"]
        out["board_stray_pct"] = 0.0
    out["board_luma_stray_bias"] = out["board_luma_largest_cc"] - out["board_luma"]
    return out


def compute_background_table(originals_dir: Path, wanted_keys: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    paths = {
        p.stem: p
        for p in sorted(originals_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    keys = sorted(set(paths) & wanted_keys, key=sample_sort_key)
    for i, key in enumerate(keys, 1):
        fish_id, day, side = parse_sample_name(key)
        row: dict[str, object] = {"sample": key, "FISH_ID": fish_id, "DAY": day, "SIDE": side}
        row.update(board_measurements(paths[key]))
        row.update(read_exif(paths[key]))
        rows.append(row)
        print(f"  [{i:>2}/{len(keys)}] {key}: board luma = {row.get('board_luma', float('nan')):.1f}")

    df = pd.DataFrame(rows)
    if not df.empty and {"exposure_time_s", "f_number", "iso"}.issubset(df.columns):
        df["exposure_value_ev"] = (
            np.log2(df["f_number"] ** 2 / df["exposure_time_s"]) - np.log2(df["iso"] / 100.0)
        )
    return df


def exposure_tracking(
    per_image: pd.DataFrame,
    background: pd.DataFrame,
    measures: dict[str, str],
    proxies: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = per_image.merge(background, on=["sample", "FISH_ID", "DAY", "SIDE"], how="inner")
    rows: list[dict] = []
    for proxy_col, proxy_label in proxies.items():
        if proxy_col not in merged.columns:
            continue
        for col, label in measures.items():
            data = merged[[proxy_col, col]].dropna()
            if len(data) < 4:
                continue
            x = data[proxy_col].to_numpy(float)
            y = data[col].to_numpy(float)
            r, p = stats.pearsonr(x, y)
            fit = stats.linregress(x, y)
            slope_per_sd = fit.slope * x.std(ddof=1)
            rows.append(
                {
                    "exposure_proxy": proxy_label,
                    "measure": label,
                    "n": len(data),
                    "pearson_r": r,
                    "r_squared": r ** 2,
                    "p_value": p,
                    "slope_per_sd_proxy": slope_per_sd,
                    "pct_of_mean_per_sd": 100.0 * slope_per_sd / abs(y.mean()) if y.mean() else np.nan,
                }
            )
    return pd.DataFrame(rows), merged


def exposure_tracking_within_fish(
    merged: pd.DataFrame, measures: dict[str, str], proxy_col: str = "board_luma"
) -> pd.DataFrame:
    """
    Repeat the exposure-tracking regression on fish-mean-centred values.

    Centring within fish removes all between-fish biological variation, so any
    residual association with the background board is attributable to
    acquisition rather than to the animal.

    Significance is assessed against n - k - 1 degrees of freedom, where k is the
    number of fish, because subtracting each fish's mean from both variables
    consumes one degree of freedom per fish. The default in
    `scipy.stats.pearsonr` is n - 2, which treats the centred values as though
    they were independent observations and yields anti-conservative p-values.
    """
    rows: list[dict] = []
    for col, label in measures.items():
        data = merged[["FISH_ID", proxy_col, col]].dropna()
        n_fish = int(data["FISH_ID"].nunique())
        if n_fish < 3:
            continue
        centred = data.copy()
        for c in (proxy_col, col):
            centred[c] = centred[c] - centred.groupby("FISH_ID")[c].transform("mean")
        x = centred[proxy_col].to_numpy(float)
        y = centred[col].to_numpy(float)
        if x.std() == 0 or y.std() == 0:
            continue
        r, p_naive = stats.pearsonr(x, y)

        df = len(centred) - n_fish - 1
        if df > 0 and abs(r) < 1:
            t_stat = r * np.sqrt(df / (1 - r ** 2))
            p = float(2 * stats.t.sf(abs(t_stat), df))
        else:
            t_stat, p = np.nan, np.nan

        rows.append(
            {
                "measure": label,
                "n": len(centred),
                "n_fish": n_fish,
                "pearson_r_within_fish": r,
                "r_squared": r ** 2,
                "df": df,
                "t_stat": float(t_stat),
                "p_value": p,
                "p_value_uncorrected_df": float(p_naive),
            }
        )
    return pd.DataFrame(rows)


def icc_2_1(matrix: np.ndarray, alpha: float = 0.05) -> dict:
    """
    ICC(2,1): two-way random effects, single measurement, absolute agreement.

    `matrix` is n subjects x k raters, complete cases only. Here the subjects
    are fish-side units and the raters are the two imaging occasions.
    Confidence limits follow McGraw & Wong (1996), Table 7, ICC(A,1).
    """
    x = np.asarray(matrix, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 3 or k < 2:
        return {
            "icc": np.nan, "lower": np.nan, "upper": np.nan, "n": n, "k": k,
            "f_stat": np.nan, "p_value": np.nan, "ms_rows": np.nan,
            "ms_cols": np.nan, "ms_error": np.nan, "occasion_mean_diff": np.nan,
        }

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows, df_cols = n - 1, k - 1
    df_error = df_rows * df_cols
    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    icc = (ms_rows - ms_error) / denom if denom != 0 else np.nan

    f_stat = ms_rows / ms_error if ms_error > 0 else np.inf
    p_value = float(stats.f.sf(f_stat, df_rows, df_error))

    lower = upper = np.nan
    if np.isfinite(icc) and 0 < icc < 1:
        a = (k * icc) / (n * (1 - icc))
        b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
        v_den = (a * ms_cols) ** 2 / df_cols + (b * ms_error) ** 2 / df_error
        v = ((a * ms_cols + b * ms_error) ** 2 / v_den) if v_den > 0 else np.nan
        if np.isfinite(v) and v > 0:
            f_lower = stats.f.ppf(1 - alpha / 2, df_rows, v)
            f_upper = stats.f.ppf(1 - alpha / 2, v, df_rows)
            lower = n * (ms_rows - f_lower * ms_error) / (
                f_lower * (k * ms_cols + (k * n - k - n) * ms_error) + n * ms_rows
            )
            upper = n * (f_upper * ms_rows - ms_error) / (
                k * ms_cols + (k * n - k - n) * ms_error + n * f_upper * ms_rows
            )

    return {
        "icc": float(icc),
        "lower": float(lower) if np.isfinite(lower) else np.nan,
        "upper": float(upper) if np.isfinite(upper) else np.nan,
        "n": int(n),
        "k": int(k),
        "f_stat": float(f_stat),
        "p_value": p_value,
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_error": float(ms_error),
        "occasion_mean_diff": float(col_means[1] - col_means[0]),
    }


def occasion_matrix(per_image: pd.DataFrame, column: str) -> tuple[np.ndarray, list[str]]:
    """Rows = fish x side units, columns = imaging occasion (day 1, day 2)."""
    wide = per_image.pivot_table(
        index=["FISH_ID", "SIDE"], columns="DAY", values=column, aggfunc="mean"
    )
    days = sorted(wide.columns, key=lambda d: int(d))
    wide = wide[days].dropna()
    units = [f"{f}-{s}" for f, s in wide.index]
    return wide.to_numpy(dtype=float), units


def build_icc_table(per_image: pd.DataFrame, measures: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for column, label in measures.items():
        if column not in per_image.columns:
            continue
        matrix, _ = occasion_matrix(per_image, column)
        result = icc_2_1(matrix)
        r_between = stats.pearsonr(matrix[:, 0], matrix[:, 1])[0] if len(matrix) >= 3 else np.nan
        within_sd = (
            float(np.sqrt(result["ms_error"])) if np.isfinite(result["ms_error"]) else np.nan
        )
        rows.append(
            {
                "measure": label,
                "column": column,
                "n_units": result["n"],
                "n_occasions": result["k"],
                "ICC(2,1)": result["icc"],
                "CI_lower": result["lower"],
                "CI_upper": result["upper"],
                "p_value": result["p_value"],
                "pearson_r_occasions": r_between,
                "mean": float(np.nanmean(matrix)),
                "SD_total": float(np.nanstd(matrix, ddof=1)),
                "SD_within(SEM)": within_sd,
                "occasion2_minus_occasion1": result["occasion_mean_diff"],
            }
        )
    return pd.DataFrame(rows)


def figure_annulus_example(
    image_path: Path, polygons: list[list[float]], cfg: AnnulusConfig, out_path: Path
) -> None:
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.uint8).astype(float) / MAX_LEVEL
    h, w = gray.shape
    tissue = tissue_mask_from_rgb(rgb)

    masks = [polygon_mask(w, h, polygon_to_pixels(p, w, h)) for p in polygons]
    all_spots = np.zeros((h, w), bool)
    for m in masks:
        all_spots |= m
    dist_all = ndimage.distance_transform_edt(~all_spots)
    skin_pool = tissue & (dist_all > cfg.gap)

    areas = [int(m.sum()) for m in masks]
    target = int(np.argsort(areas)[len(areas) // 2])
    mask = masks[target]
    dist = ndimage.distance_transform_edt(~mask)
    annulus = skin_pool & (dist > cfg.gap) & (dist <= cfg.gap + cfg.width)

    ys, xs = np.nonzero(mask)
    pad = 90
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].imshow(rgb)
    axes[0].add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#d95f02", lw=1.8)
    )
    axes[0].set_title("Measurement image\n(background already masked)", fontsize=10)

    crop = rgb[y0:y1, x0:x1]
    axes[1].imshow(crop)
    axes[1].set_title("Detail", fontsize=10)

    overlay = crop.astype(float).copy()
    sub_spot = mask[y0:y1, x0:x1]
    sub_ann = annulus[y0:y1, x0:x1]
    overlay[sub_spot] = 0.45 * overlay[sub_spot] + 0.55 * np.array([217, 95, 2])
    overlay[sub_ann] = 0.45 * overlay[sub_ann] + 0.55 * np.array([27, 158, 119])
    axes[2].imshow(overlay.astype(np.uint8))
    spot_val = gray[mask & tissue].mean()
    skin_val = float(np.median(gray[annulus]))
    axes[2].set_title(
        f"spot (orange) I = {spot_val:.3f}\nskin annulus (green) I = {skin_val:.3f}\n"
        f"ratio = {spot_val / skin_val:.3f}",
        fontsize=10,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_board_segmentation(
    background: pd.DataFrame, originals_dir: Path, out_path: Path
) -> None:
    """
    Show what the exposure proxy actually measures.

    A referee is entitled to ask how the background board was identified and
    whether the fish could have contaminated it. The two example frames are the
    dimmest and brightest in the set, so the panel doubles as a demonstration
    that the proxy responds to real exposure differences. The hue-saturation
    panel shows why the selection is unambiguous: the board is strongly
    saturated and blue, while salmon tissue and the white identification card
    are near-neutral and fall outside the acceptance window.
    """
    usable = background.dropna(subset=["board_luma"]).sort_values("board_luma")
    if usable.empty:
        return
    dim, bright = usable.iloc[0], usable.iloc[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4))

    hue_all, sat_all, board_all = None, None, None
    for ax, row, label in [(axes[0, 0], dim, "Dimmest frame"),
                           (axes[0, 1], bright, "Brightest frame")]:
        path = originals_dir / f"{row['sample']}.jpg"
        if not path.exists():
            candidates = list(originals_dir.glob(f"{row['sample']}.*"))
            if not candidates:
                ax.axis("off")
                continue
            path = candidates[0]
        image = Image.open(path)
        image.draft("RGB", (1000, 750))
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        hue, sat, val = rgb_to_hsv_arrays(rgb)
        board = (
            (hue >= BOARD_HUE_RANGE[0]) & (hue <= BOARD_HUE_RANGE[1])
            & (sat >= BOARD_MIN_SATURATION)
            & (val >= BOARD_VALUE_RANGE[0]) & (val <= BOARD_VALUE_RANGE[1])
        )
        if hue_all is None:
            hue_all, sat_all, board_all = hue, sat, board

        overlay = rgb.astype(float)
        grey = overlay.mean(axis=2, keepdims=True)
        dimmed = 0.35 * overlay + 0.65 * grey * 0.55
        shown = np.where(board[..., None], overlay, dimmed)
        ax.imshow(shown.astype(np.uint8))
        ax.contour(board.astype(float), levels=[0.5], colors="#ffd400", linewidths=1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}: board luma = {row['board_luma']:.1f}\n"
                     f"{row['board_pixel_fraction'] * 100:.0f}% of frame accepted "
                     f"(shown in colour; excluded area dimmed)", fontsize=10)

    ax = axes[1, 0]
    if hue_all is not None:
        step = max(1, hue_all.size // 120000)
        h = hue_all.ravel()[::step]
        s = sat_all.ravel()[::step]
        ax.hist2d(h, s, bins=[90, 60], range=[[0, 360], [0, 1]],
                  cmap="Greys", norm=matplotlib.colors.LogNorm())
        ax.add_patch(plt.Rectangle(
            (BOARD_HUE_RANGE[0], BOARD_MIN_SATURATION),
            BOARD_HUE_RANGE[1] - BOARD_HUE_RANGE[0], 1 - BOARD_MIN_SATURATION,
            fill=False, edgecolor="#d95f02", lw=2.4))
        ax.text(BOARD_HUE_RANGE[0] + 5, 0.95, "accepted as board", fontsize=9,
                color="#d95f02", va="top")
        ax.annotate("tissue and ID card\n(near-neutral, low saturation)",
                    xy=(30, 0.12), fontsize=8.5, color="black")
        ax.set_xlabel("Hue (degrees)")
        ax.set_ylabel("Saturation")
        ax.set_title("Pixel distribution in HSV, dimmest frame\n"
                     "board separates on saturation and hue", fontsize=10)

    ax = axes[1, 1]
    ax.hist(usable["board_luma"], bins=22, color="#7570b3", edgecolor="black",
            linewidth=0.5)
    for row, colour, label in [(dim, "#1b9e77", "dimmest"),
                               (bright, "#d95f02", "brightest")]:
        ax.axvline(row["board_luma"], color=colour, lw=2.2, label=f"{label} frame")
    ax.set_xlabel("Blue board luma (0-255)")
    ax.set_ylabel("Frames")
    ax.set_title(f"Realised exposure across the {len(usable)} frames\n"
                 f"mean {usable['board_luma'].mean():.1f}, "
                 f"SD {usable['board_luma'].std():.1f}, "
                 f"range {usable['board_luma'].min():.1f}-{usable['board_luma'].max():.1f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_exposure_tracking(merged: pd.DataFrame, out_path: Path) -> None:
    """
    Each descriptor against the independent exposure proxy.

    Grayscale, with the contaminated raw measure drawn dark with filled markers
    and a solid fit, and the exposure-robust referenced measure drawn lighter
    with open markers and a dashed fit. The point of the figure is the contrast
    between a sloping and a flat relationship, so the fit lines are the elements
    that must remain distinguishable.
    """
    panels = [
        ("raw_mean_intensity", "Raw mean spot intensity",
         RAW_GREY, "o", "full", "-"),
        ("skin_ref_log_ratio_linear", "Skin-referenced log-ratio (linearised)",
         "0.50", "s", "none", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (col, label, grey, marker, fillstyle, linestyle) in zip(axes, panels):
        data = merged[["board_luma", col]].dropna()
        x, y = data["board_luma"].to_numpy(), data[col].to_numpy()
        r, p = stats.pearsonr(x, y)
        fit = stats.linregress(x, y)
        ax.scatter(x, y, s=44, marker=marker, linewidths=0.9,
                   facecolors=grey if fillstyle == "full" else "none",
                   edgecolors="black" if fillstyle == "full" else grey,
                   alpha=0.85)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=2.2, ls=linestyle)
        ax.set_xlabel("Blue board luma (exposure proxy, 0-255)")
        ax.set_ylabel(label)
        ax.set_title(
            f"{label}\nr = {r:.3f}, R² = {r ** 2:.3f}, {format_p(p)}, n = {len(x)}", fontsize=10
        )
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_icc_panels(per_image: pd.DataFrame, measures: dict[str, str], out_path: Path) -> None:
    items = [(c, l) for c, l in measures.items() if c in per_image.columns]
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, label) in zip(axes, items):
        matrix, _ = occasion_matrix(per_image, col)
        res = icc_2_1(matrix)
        ax.scatter(
            matrix[:, 0], matrix[:, 1], s=42, alpha=0.85, color=POINT_GREY,
            edgecolors="black", linewidths=0.5,
        )
        lo, hi = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        span = (hi - lo) or 1.0
        lims = (lo - 0.06 * span, hi + 0.06 * span)
        ax.plot(lims, lims, ls="--", color="black", lw=1.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Session 1 (pre-stress)")
        ax.set_ylabel("Session 2 (post-stress)")
        ax.set_title(
            f"{label}\nICC(2,1) = {res['icc']:.3f} [{res['lower']:.3f}, {res['upper']:.3f}]",
            fontsize=10,
        )
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_dispersion(spots: pd.DataFrame, per_image: pd.DataFrame,
                      out_path: Path) -> None:
    """
    Spread of the raw versus the skin-referenced measure, at both levels.

    Rows are the two units of analysis (individual spots, then whole images);
    columns are the raw measure and its skin-referenced counterpart. Both
    columns are the gamma-encoded 8-bit quantities, so the coefficients of
    variation are compared within one domain. An earlier version paired the
    gamma-encoded raw measure against the linearised ratio, which mixed domains
    and understated the reduction.

    The coefficient of variation is used only on ratio-scale quantities with a
    meaningful zero. It is deliberately not reported for the log-ratio, whose
    mean is negative and whose zero point is arbitrary, so SD/|mean| would not
    be interpretable there.

    Drawn in grayscale, matching the other descriptor figures. Raw and
    referenced are separated by tone and by hatching rather than by hue, so the
    contrast survives monochrome printing.
    """
    panels = [
        (spots, "spot_mean", "Raw mean spot intensity", "Spot level"),
        (spots, "ratio", "Skin-referenced ratio", "Spot level"),
        (per_image, "raw_mean_intensity", "Raw mean spot intensity", "Image level"),
        (per_image, "skin_ref_ratio", "Skin-referenced ratio", "Image level"),
    ]
    counted = {"Spot level": "Spots", "Image level": "Images"}
    styles = [(RAW_GREY, None), (REFERENCED_GREY, "///")]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, (frame, col, label, row_label) in zip(axes.ravel(), panels):
        grey, hatch = styles[["Raw mean spot intensity",
                              "Skin-referenced ratio"].index(label)]
        v = frame[col].dropna().to_numpy()
        ax.hist(v, bins=30 if row_label == "Spot level" else 18, color=grey,
                edgecolor="black", linewidth=0.7, hatch=hatch)
        cv = 100 * v.std(ddof=1) / v.mean()
        ax.set_xlabel(label)
        ax.set_ylabel(counted[row_label])
        ax.set_title(f"{row_label} (n = {len(v):,})\n"
                     f"mean = {v.mean():.3f}, CV = {cv:.1f}%", fontsize=10)
        ax.grid(True, ls="--", lw=0.5, alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


ICC_MEASURES = {
    "raw_mean_intensity": "Raw mean spot intensity",
    "skin_ref_log_ratio_linear": "Skin-referenced log-ratio (linearised)",
    "skin_ref_ratio": "Skin-referenced intensity (ratio)",
    "skin_ref_difference": "Skin-referenced intensity (difference)",
    "relative_spot_area_pct": "Relative spot area (%)",
    "spot_count": "Spot count",
    "total_melanization": "Integrated melanization",
}

EXPOSURE_MEASURES = {
    "raw_mean_intensity": "Raw mean spot intensity",
    "skin_ref_log_ratio_linear": "Skin-referenced log-ratio (linearised)",
    "skin_ref_ratio": "Skin-referenced intensity (ratio)",
    "skin_ref_difference": "Skin-referenced intensity (difference)",
    "image_skin_median": "Median opercular skin intensity",
    "relative_spot_area_pct": "Relative spot area (%)",
}

EXPOSURE_PROXIES = {
    "board_luma": "Blue board luma",
    "board_B": "Blue board B channel",
    "exposure_value_ev": "EXIF exposure value (EV at ISO 100)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--images-dir", type=Path, default=P.IMAGES_DIR)
    p.add_argument("--labels-dir", type=Path, default=P.LABELS_DIR)
    p.add_argument(
        "--originals-dir",
        type=Path,
        default=P.ORIGINALS_DIR,
        help=("Directory of uncropped camera frames retaining the blue board and "
              "EXIF. Defaults to head_images/ (all 84 measurement images)."),
    )
    p.add_argument("--output-dir", type=Path, default=P.RESULTS_DIR)
    p.add_argument("--annulus-gap", type=float, default=DEFAULT_ANNULUS_GAP)
    p.add_argument("--annulus-width", type=float, default=DEFAULT_ANNULUS_WIDTH)
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def autodetect_originals(start: Path) -> Path | None:
    """Deprecated: resolution now lives in scripts/_paths.autodetect_originals."""
    return P.autodetect_originals(start)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg = AnnulusConfig(gap=args.annulus_gap, width=args.annulus_width)

    image_paths = load_paths(args.images_dir, IMAGE_EXTENSIONS)
    detections = load_polygon_detections(args.labels_dir)
    print(f"Loaded {len(image_paths)} images and {len(detections)} label files.")

    print("\n[1/4] Skin-referenced spot intensity")
    spots, image_meta = compute_spot_table(image_paths, detections, cfg)
    per_image = aggregate_per_image(spots, image_meta)
    spots.to_csv(out / "data" / "spot_level_skin_referenced.csv", index=False)
    per_image.to_csv(out / "data" / "image_level_measures.csv", index=False)

    print("\n[2/4] Background-referenced exposure proxies")
    originals = args.originals_dir or autodetect_originals(Path.cwd())
    background = pd.DataFrame()
    tracking = pd.DataFrame()
    merged = pd.DataFrame()
    if originals and Path(originals).is_dir():
        print(f"  Using originals from: {originals}")
        background = compute_background_table(Path(originals), set(image_paths))
        background.to_csv(out / "data" / "background_board_exposure.csv", index=False)
        tracking, merged = exposure_tracking(
            per_image, background, EXPOSURE_MEASURES, EXPOSURE_PROXIES
        )
        tracking.to_csv(out / "data" / "exposure_tracking.csv", index=False)
        merged.to_csv(out / "data" / "image_level_with_exposure.csv", index=False)
        print(
            tracking[tracking["exposure_proxy"] == "Blue board luma"][
                ["measure", "n", "pearson_r", "r_squared", "p_value"]
            ].to_string(index=False)
        )
        within = exposure_tracking_within_fish(merged, EXPOSURE_MEASURES)
        within.to_csv(out / "data" / "exposure_tracking_within_fish.csv", index=False)
        print("\n  Fish-mean-centred (between-fish biology removed):")
        print(within.to_string(index=False))
    else:
        print("  WARNING: originals directory not found; skipping the background check.")

    print("\n[3/4] Session repeatability, ICC(2,1)")
    icc_table = build_icc_table(per_image, ICC_MEASURES)
    icc_table.to_csv(out / "data" / "icc_session_repeatability.csv", index=False)
    print(icc_table[["measure", "n_units", "ICC(2,1)", "CI_lower", "CI_upper"]].to_string(index=False))

    sens_rows = []
    for side in sorted(per_image["SIDE"].unique()):
        t = build_icc_table(per_image[per_image["SIDE"] == side], ICC_MEASURES)
        t.insert(0, "subset", f"side {side} only")
        sens_rows.append(t)
    if sens_rows:
        pd.concat(sens_rows, ignore_index=True).to_csv(
            out / "data" / "icc_sensitivity_by_side.csv", index=False
        )

    print("\n[4/4] Figures")
    if not args.skip_figures:
        first_key = sorted(set(image_paths) & set(detections), key=sample_sort_key)[0]
        figure_annulus_example(
            image_paths[first_key], detections[first_key], cfg,
            out / "figures" / "fig_annulus_method.png",
        )
        figure_icc_panels(per_image, ICC_MEASURES, out / "figures" / "fig_icc_occasions.png")
        figure_dispersion(spots, per_image, out / "figures" / "fig_dispersion.png")
        if not merged.empty:
            figure_exposure_tracking(merged, out / "figures" / "fig_exposure_tracking.png")
            if originals and Path(originals).is_dir():
                figure_board_segmentation(
                    background, Path(originals),
                    out / "figures" / "fig_board_segmentation.png")
        print("  figures written")
    else:
        print("  skipped")

    provenance = {
        "n_images": int(len(per_image)),
        "n_spots": int(len(spots)),
        "n_fish": int(per_image["FISH_ID"].nunique()),
        "annulus_gap_px": cfg.gap,
        "annulus_width_px": cfg.width,
        "min_annulus_px": MIN_ANNULUS_PIXELS,
        "grayscale_weights": list(GRAYSCALE_WEIGHTS),
        "bit_depth": BIT_DEPTH,
        "background_level": BACKGROUND_LEVEL,
        "originals_dir": str(originals) if originals else None,
        "n_originals_matched": int(len(background)),
        "labelled_spot_px": int(per_image["labelled_spot_px"].sum()),
        "spot_px_outside_tissue": int(per_image["spot_px_outside_tissue"].sum()),
        "legacy_zero_px": int(per_image["legacy_zero_px"].sum()),
        "mean_tissue_fraction_of_frame": float(
            (per_image["tissue_px"] / (1080 * 1080)).mean()
        ),
    }
    (out / "data" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
