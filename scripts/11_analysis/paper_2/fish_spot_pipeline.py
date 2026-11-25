#!/usr/bin/env python3
"""
Single-file fish spot analysis pipeline.

This script consolidates the uploaded notebooks into one runnable module:
1. Load image and YOLO-style polygon label files.
2. Compute spot-wise grayscale signatures, spot areas, and per-image summaries.
3. Compute spot distances from the right edge of each resized image.
4. Compute background/canvas melanization metrics.
5. Generate summary plots.

Expected input layout by default:
    ./images/   image files
    ./labels/   matching label files

Run:
    python fish_spot_pipeline.py

Useful options:
    python fish_spot_pipeline.py --images-dir images --labels-dir labels --output-dir results
    python fish_spot_pipeline.py --skip-distance-plots --skip-summary-plots
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageDraw

SPOTWISE_COLUMNS = [
    "FISH_ID",
    "DAY",
    "SIDE",
    "spot",
    "level_1_m",
    "level_2_m",
    "level_3_m",
    "level_4_m",
    "level_5_m",
    "level_6_m",
    "level_7_m",
    "level_8_m",
    "level_9_m",
    "level_10_m",
    "level_1_c",
    "level_2_c",
    "level_3_c",
    "level_4_c",
    "level_5_c",
    "level_6_c",
    "level_7_c",
    "level_8_c",
    "level_9_c",
    "level_10_c",
    "level_1_p",
    "level_2_p",
    "level_3_p",
    "level_4_p",
    "level_5_p",
    "level_6_p",
    "level_7_p",
    "level_8_p",
    "level_9_p",
    "level_10_p",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTENSIONS = {".txt", ".csv"}
INTENSITY_BIN_EDGES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
INTENSITY_LABELS = [
    "[0.0,0.1]",
    "(0.1,0.2]",
    "(0.2,0.3]",
    "(0.3,0.4]",
    "(0.4,0.5]",
    "(0.5,0.6]",
    "(0.6,0.7]",
    "(0.7,0.8]",
    "(0.8,0.9]",
    "(0.9,1.0]",
]


def sample_key_from_path(path: Path) -> str:
    """Return the shared image/label sample key used by the original notebooks."""
    stem = path.stem
    return stem.split("_jpg")[0] if "_jpg" in stem else stem


def parse_sample_name(sample_name: str) -> tuple[str, str, str]:
    """
    Extract fish id, day, and side from the project filename convention.

    The notebooks used:
        fish_id = sample_name.split('_')[1]
        day     = sample_name.split('_')[3][-1]
        side    = sample_name.split('_')[-2]

    This function preserves that behavior while adding clear errors for invalid names.
    """
    parts = sample_name.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Cannot parse sample name '{sample_name}'. Expected at least 5 underscore-separated parts."
        )

    fish_id = parts[1]
    day_match = re.search(r"(\d+)$", parts[3])
    day = day_match.group(1) if day_match else parts[3][-1]
    side = parts[-2]
    return str(fish_id), str(day), str(side)


def sample_sort_key(sample_name: str) -> tuple[int, str, int, str]:
    """Sort by fish id, side, day, then full sample name."""
    try:
        fish_id, day, side = parse_sample_name(sample_name)
        fish_num = int(fish_id)
        day_num = int(day) if str(day).isdigit() else 0
        return fish_num, side, day_num, sample_name
    except Exception:
        return 10**9, "", 0, sample_name


def spot_sort_key(row: pd.Series) -> tuple[int, str, int, int]:
    fish_id = row.get("FISH_ID", "")
    day = row.get("DAY", "")
    try:
        fish_num = int(fish_id)
    except Exception:
        fish_num = 10**9
    try:
        day_num = int(day)
    except Exception:
        day_num = 0
    return fish_num, str(row.get("SIDE", "")), day_num, int(row.get("spot", 0))


def load_paths(directory: Path, allowed_extensions: set[str]) -> dict[str, Path]:
    """Load files in a directory keyed by their shared sample name."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    paths: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            paths[sample_key_from_path(path)] = path
    return dict(sorted(paths.items(), key=lambda item: sample_sort_key(item[0])))


def load_polygon_detections(label_paths: dict[str, Path]) -> dict[str, list[list[float]]]:
    """Load normalized polygon coordinates from YOLO-style label files."""
    detections: dict[str, list[list[float]]] = {}
    for key, path in label_paths.items():
        polygons: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                tokens = line.strip().split()
                if len(tokens) <= 1:
                    continue
                try:
                    coords = [float(token) for token in tokens[1:]]
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value in {path} line {line_number}") from exc
                if len(coords) % 2 != 0:
                    raise ValueError(f"Odd number of polygon coordinates in {path} line {line_number}")
                polygons.append(coords)
        detections[key] = polygons
    return detections


def matching_sample_keys(
    image_paths: dict[str, Path], polygon_detections: dict[str, list[list[float]]]
) -> list[str]:
    """Return image keys that also have labels, printing useful warnings."""
    image_keys = set(image_paths)
    label_keys = set(polygon_detections)
    missing_labels = sorted(image_keys - label_keys, key=sample_sort_key)
    missing_images = sorted(label_keys - image_keys, key=sample_sort_key)

    if missing_labels:
        print(f"Warning: {len(missing_labels)} image(s) have no matching label file and will be skipped.")
    if missing_images:
        print(f"Warning: {len(missing_images)} label file(s) have no matching image and will be skipped.")

    return sorted(image_keys & label_keys, key=sample_sort_key)


def normalized_polygon_to_pixels(
    vertices: Iterable[float], width: int, height: int
) -> list[tuple[int, int]]:
    """Convert normalized [x1, y1, x2, y2, ...] vertices to pixel coordinate tuples."""
    values = list(vertices)
    return [
        (
            int(np.clip(values[i] * width, 0, width - 1)),
            int(np.clip(values[i + 1] * height, 0, height - 1)),
        )
        for i in range(0, len(values), 2)
    ]


def polygon_mask(width: int, height: int, vertices: list[tuple[int, int]]) -> np.ndarray:
    """Create a boolean mask for one polygon."""
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(vertices, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def split_intensity_levels(values: np.ndarray) -> tuple[list[float], list[int], list[np.ndarray]]:
    """Split normalized pixel values into the 10 grayscale bins used by the notebooks."""
    if values.size == 0:
        empty_arrays = [np.array([], dtype=float) for _ in range(10)]
        return [np.nan] * 10, [0] * 10, empty_arrays

    bin_ids = np.digitize(values, INTENSITY_BIN_EDGES, right=True)
    arrays = [values[bin_ids == idx] for idx in range(10)]
    means = [float(arr.mean()) if arr.size else np.nan for arr in arrays]
    counts = [int(arr.size) for arr in arrays]
    return means, counts, arrays


def pixel_array_mean(arrays: list[np.ndarray]) -> float:
    non_empty = [arr for arr in arrays if arr.size]
    if not non_empty:
        return np.nan
    return float(np.concatenate(non_empty).mean())


def compute_spotwise_features(
    image_paths: dict[str, Path],
    polygon_detections: dict[str, list[list[float]]],
    sample_keys: list[str],
    image_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute spot-level signatures and per-image/global summaries."""
    spot_rows: list[dict[str, object]] = []
    global_spot_rows: list[dict[str, object]] = []

    for sample_key in sample_keys:
        fish_id, day, side = parse_sample_name(sample_key)
        image = Image.open(image_paths[sample_key]).convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        image_array = np.array(image, dtype=np.uint8)
        height, width = image_array.shape

        for spot_index, polygon in enumerate(polygon_detections[sample_key]):
            vertices = normalized_polygon_to_pixels(polygon, width, height)
            mask = polygon_mask(width, height, vertices)

            values = image_array[mask].astype(float) / 255.0
            values = values[values != 0.0]

            means, counts, arrays = split_intensity_levels(values)
            row: dict[str, object] = {
                "FISH_ID": fish_id,
                "DAY": day,
                "SIDE": side,
                "spot": spot_index,
            }
            row.update({f"level_{idx}_m": means[idx - 1] for idx in range(1, 11)})
            row.update({f"level_{idx}_c": counts[idx - 1] for idx in range(1, 11)})
            row.update({f"level_{idx}_p": arrays[idx - 1] for idx in range(1, 11)})
            spot_rows.append(row)

            global_spot_rows.append(
                {
                    "FISH_ID": fish_id,
                    "DAY": day,
                    "SIDE": side,
                    "spot": spot_index,
                    "Pixel Intensity": pixel_array_mean(arrays),
                    "Areas": int(sum(counts)),
                }
            )

        print(f"{sample_key}: processed {len(polygon_detections[sample_key])} spot(s)")

    spotwise_df = pd.DataFrame(spot_rows, columns=SPOTWISE_COLUMNS)
    global_spot_df = pd.DataFrame(
        global_spot_rows,
        columns=["FISH_ID", "DAY", "SIDE", "spot", "Pixel Intensity", "Areas"],
    )

    if global_spot_df.empty:
        global_df = pd.DataFrame(
            columns=["FISH_ID", "DAY", "SIDE", "Spots", "Pixel Intensity", "Areas"]
        )
    else:
        global_df = (
            global_spot_df.groupby(["FISH_ID", "DAY", "SIDE"], sort=False)
            .agg(
                Spots=("spot", "count"),
                **{"Pixel Intensity": ("Pixel Intensity", "mean"), "Areas": ("Areas", "sum")},
            )
            .reset_index()
        )
        sort_values = global_df.apply(spot_sort_key, axis=1)
        global_df = global_df.iloc[np.argsort(sort_values.to_numpy())].reset_index(drop=True)

    return spotwise_df, global_spot_df, global_df


def polygon_bounding_box(vertices: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_distances(
    image_paths: dict[str, Path],
    polygon_detections: dict[str, list[list[float]]],
    sample_keys: list[str],
    image_size: int,
    plots_dir: Path,
    save_plots: bool,
) -> pd.DataFrame:
    """Compute each spot center's horizontal distance from the right edge."""
    rows: list[dict[str, object]] = []
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for sample_key in sample_keys:
        fish_id, day, side = parse_sample_name(sample_key)
        image = Image.open(image_paths[sample_key]).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        width, height = image.size

        fig = ax = None
        if save_plots:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

        for spot_index, polygon in enumerate(polygon_detections[sample_key]):
            vertices = normalized_polygon_to_pixels(polygon, width, height)
            x_min, y_min, x_max, y_max = polygon_bounding_box(vertices)
            center_x = int((x_max - x_min) / 2) + x_min
            center_y = int((y_max - y_min) / 2) + y_min
            distance = round(euclidean_distance(width, center_y, center_x, center_y), 4)

            rows.append(
                {
                    "FISH_ID": fish_id,
                    "DAY": day,
                    "SIDE": side,
                    "spot": spot_index,
                    "distance": distance,
                }
            )

            if save_plots and ax is not None:
                patch = patches.Polygon(
                    xy=np.array(vertices),
                    closed=True,
                    edgecolor="green",
                    facecolor="none",
                    linewidth=1,
                )
                ax.add_patch(patch)
                ax.text(center_x, center_y + 5, str(spot_index), color="blue", fontsize=6)
                ax.add_line(Line2D([center_x, width], [center_y, center_y], color="green", linewidth=1))

        if save_plots and fig is not None and ax is not None:
            ax.set_title(sample_key)
            legend_handles = [patches.Patch(color="green", label=f"Spots ({len(polygon_detections[sample_key])})")]
            ax.legend(handles=legend_handles, loc="upper right")
            fig.savefig(plots_dir / f"{sample_key}.jpg", dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"{sample_key}: computed {len(polygon_detections[sample_key])} distance(s)")

    return pd.DataFrame(rows, columns=["FISH_ID", "DAY", "SIDE", "spot", "distance"])


def add_spot_areas_to_distances(distances_df: pd.DataFrame, global_spot_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the computed spot area to the corresponding distance row."""
    if distances_df.empty:
        distances_df["areas"] = pd.Series(dtype=float)
        return distances_df

    area_df = global_spot_df[["FISH_ID", "DAY", "SIDE", "spot", "Areas"]].copy()
    area_df = area_df.rename(columns={"Areas": "areas"})

    merged = distances_df.copy()
    for column in ["FISH_ID", "DAY", "SIDE"]:
        merged[column] = merged[column].astype(str)
        area_df[column] = area_df[column].astype(str)
    merged["spot"] = merged["spot"].astype(int)
    area_df["spot"] = area_df["spot"].astype(int)

    return merged.merge(area_df, how="left", on=["FISH_ID", "DAY", "SIDE", "spot"])


def draw_spot_canvases(
    image_paths: dict[str, Path],
    polygon_detections: dict[str, list[list[float]]],
    sample_keys: list[str],
    canvas_dir: Path,
) -> dict[str, Path]:
    """Save white canvases with black spot polygons."""
    canvas_dir.mkdir(parents=True, exist_ok=True)
    canvas_paths: dict[str, Path] = {}

    for sample_key in sample_keys:
        image = Image.open(image_paths[sample_key]).convert("L")
        width, height = image.size
        canvas = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(canvas)

        for polygon in polygon_detections[sample_key]:
            vertices = normalized_polygon_to_pixels(polygon, width, height)
            draw.polygon(vertices, fill=0)

        canvas_path = canvas_dir / f"{sample_key}.jpg"
        canvas.save(canvas_path)
        canvas_paths[sample_key] = canvas_path
        print(f"{sample_key}: saved spot canvas")

    return canvas_paths


def compute_background_counts(
    image_paths: dict[str, Path],
    canvas_paths: dict[str, Path],
    sample_keys: list[str],
    roi_canvas_dir: Path,
) -> dict[str, int]:
    """Compute non-zero background pixels after removing detected spots."""
    roi_canvas_dir.mkdir(parents=True, exist_ok=True)
    background_counts: dict[str, int] = {}

    for sample_key in sample_keys:
        image = Image.open(image_paths[sample_key]).convert("L")
        canvas = Image.open(canvas_paths[sample_key]).convert("L")
        result = ImageChops.multiply(image, canvas)
        values = np.array(result, dtype=np.uint8) / 255.0
        background_counts[sample_key] = int(np.count_nonzero(values != 0.0))
        result.save(roi_canvas_dir / f"{sample_key}.jpg")
        print(f"{sample_key}: background pixels = {background_counts[sample_key]}")

    return background_counts


def compute_canvas_data(background_counts: dict[str, int], distance_area_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-image background, total area, and melanization summary."""
    rows: list[dict[str, object]] = []
    working = distance_area_df.copy()
    for column in ["FISH_ID", "DAY", "SIDE"]:
        if column in working.columns:
            working[column] = working[column].astype(str)

    for sample_key in sorted(background_counts, key=sample_sort_key):
        fish_id, day, side = parse_sample_name(sample_key)
        background = background_counts[sample_key]
        subset = working[
            (working["FISH_ID"] == fish_id) & (working["DAY"] == day) & (working["SIDE"] == side)
        ]
        area_sum = float(subset["areas"].sum()) if "areas" in subset else 0.0
        melanization = (area_sum / background * 100.0) if background else np.nan
        rows.append(
            {
                "FISH_ID": fish_id,
                "DAY": day,
                "SIDE": side,
                "Background": background,
                "Areas": area_sum,
                "Melanization(%)": melanization,
            }
        )

    return pd.DataFrame(rows, columns=["FISH_ID", "DAY", "SIDE", "Background", "Areas", "Melanization(%)"])


def count_intensity_bins(values: pd.Series) -> dict[str, int]:
    clean_values = values.dropna().to_numpy(dtype=float)
    bin_ids = np.digitize(clean_values, INTENSITY_BIN_EDGES, right=True)
    return {label: int(np.count_nonzero(bin_ids == idx)) for idx, label in enumerate(INTENSITY_LABELS)}


def save_intensity_distribution_plot(global_spot_df: pd.DataFrame, figures_dir: Path) -> None:
    counts_by_bin = count_intensity_bins(global_spot_df["Pixel Intensity"])
    labels = list(counts_by_bin.keys())
    counts = list(counts_by_bin.values())
    colors = [plt.cm.gray(value) for value in np.linspace(0.1, 1.0, 10)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Grayscale ranges")
    ax.set_ylabel("Spot counts")
    fig.tight_layout()
    fig.savefig(figures_dir / "spot_distr_sc.jpg", dpi=800)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    positive_counts = [count for count in counts if count > 0]
    positive_labels = [label for label, count in zip(labels, counts) if count > 0]
    positive_colors = [color for color, count in zip(colors, counts) if count > 0]
    if positive_counts:
        ax.pie(
            positive_counts,
            labels=positive_labels,
            autopct="%1.1f%%",
            colors=positive_colors,
            startangle=140,
            wedgeprops={"edgecolor": "black"},
        )
    ax.set_title("Pixel intensity distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "pixels_level_pie_chart.jpg", dpi=1200)
    plt.close(fig)


def save_area_category_plot(global_spot_df: pd.DataFrame, figures_dir: Path) -> None:
    areas = global_spot_df["Areas"].dropna().to_numpy(dtype=float)
    if areas.size == 0:
        return

    max_area = areas.max()
    if max_area <= 0:
        return

    edges = np.linspace(0, max_area, 11)
    bin_ids = np.digitize(areas, edges[1:-1], right=True)
    counts = [int(np.count_nonzero(bin_ids == idx)) for idx in range(10)]
    categories = [f"c{idx}" for idx in range(1, 11)]
    colors = [plt.cm.gray(value) for value in np.linspace(0.1, 1.0, 10)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Spot size category")
    ax.set_ylabel("Spot counts")
    fig.tight_layout()
    fig.savefig(figures_dir / "spot_sizes_cat.jpg", dpi=1200)
    plt.close(fig)


def save_distance_area_plot(distance_area_df: pd.DataFrame, figures_dir: Path) -> None:
    if distance_area_df.empty or not {"distance", "areas"}.issubset(distance_area_df.columns):
        return

    data = distance_area_df[["distance", "areas"]].dropna()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.scatter(data["distance"], data["areas"], s=45, alpha=0.75, edgecolors="black", linewidths=0.4)

    if len(data) >= 2 and data["distance"].nunique() > 1:
        slope, intercept = np.polyfit(data["distance"], data["areas"], 1)
        x_values = np.linspace(data["distance"].min(), data["distance"].max(), 100)
        ax.plot(x_values, slope * x_values + intercept, linewidth=2.2)

    ax.set_xlabel("Distance, pixels")
    ax.set_ylabel("Areas, pixels")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(figures_dir / "distance_vs_areas_regplot_paper_ready.png", dpi=1200, bbox_inches="tight")
    fig.savefig(figures_dir / "distance_vs_areas_regplot_paper_ready.pdf", bbox_inches="tight")
    plt.close(fig)


def save_melanization_plot(canvas_df: pd.DataFrame, figures_dir: Path) -> None:
    if canvas_df.empty or "Melanization(%)" not in canvas_df.columns:
        return

    values = canvas_df["Melanization(%)"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values, vert=False, widths=0.3)
    ax.set_xlabel("Melanization (%)")
    for value, label, linestyle in [(q1, "Q1", "--"), (q3, "Q3", "--"), (mean, "Mean", ":")]:
        ax.axvline(value, linestyle=linestyle, linewidth=1.2)
        ax.text(value, 1.15, f"{label}: {value:.2f}", fontsize=10, fontweight="bold", ha="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "melanin_dist.jpg", dpi=1200)
    plt.close(fig)


def save_summary_plots(
    global_spot_df: pd.DataFrame,
    distance_area_df: pd.DataFrame,
    canvas_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_intensity_distribution_plot(global_spot_df, figures_dir)
    save_area_category_plot(global_spot_df, figures_dir)
    save_distance_area_plot(distance_area_df, figures_dir)
    save_melanization_plot(canvas_df, figures_dir)


def save_outputs(
    output_dir: Path,
    spotwise_df: pd.DataFrame,
    global_spot_df: pd.DataFrame,
    global_df: pd.DataFrame,
    distance_area_df: pd.DataFrame,
    canvas_df: pd.DataFrame,
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    spotwise_df.to_pickle(output_dir / "spotwise_data.pkl")
    spotwise_df.to_csv(output_dir / "spotwise_data.csv", index=False)
    global_spot_df.to_csv(output_dir / "spotwise_global_data.csv", index=False)
    global_df.to_csv(output_dir / "global_data.csv", index=False)
    distance_area_df.to_csv(data_dir / "distances_start_end_spots.csv", index=False)
    canvas_df.to_csv(output_dir / "canvas_data.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the consolidated fish spot analysis pipeline.")
    parser.add_argument("--images-dir", default=P.IMAGES_DIR, type=Path, help="Directory containing images.")
    parser.add_argument("--labels-dir", default=P.LABELS_DIR, type=Path, help="Directory containing label files.")
    parser.add_argument("--output-dir", default=P.RESULTS_DIR / "legacy_fish_spot_pipeline", type=Path, help="Directory where outputs will be written.")
    parser.add_argument("--image-size", default=1080, type=int, help="Square resize size used for spot and distance metrics.")
    parser.add_argument(
        "--skip-distance-plots",
        action="store_true",
        help="Compute distances without saving per-image annotated distance plots.",
    )
    parser.add_argument(
        "--skip-summary-plots",
        action="store_true",
        help="Skip generation of summary figures in new_figures/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_paths(args.images_dir, IMAGE_EXTENSIONS)
    label_paths = load_paths(args.labels_dir, LABEL_EXTENSIONS)
    polygon_detections = load_polygon_detections(label_paths)
    sample_keys = matching_sample_keys(image_paths, polygon_detections)

    if not sample_keys:
        raise RuntimeError("No matching image/label pairs were found.")

    print(f"Found {len(sample_keys)} matching image/label pair(s).")

    spotwise_df, global_spot_df, global_df = compute_spotwise_features(
        image_paths=image_paths,
        polygon_detections=polygon_detections,
        sample_keys=sample_keys,
        image_size=args.image_size,
    )

    distance_df = compute_distances(
        image_paths=image_paths,
        polygon_detections=polygon_detections,
        sample_keys=sample_keys,
        image_size=args.image_size,
        plots_dir=output_dir / "distance_plots",
        save_plots=not args.skip_distance_plots,
    )
    distance_area_df = add_spot_areas_to_distances(distance_df, global_spot_df)

    canvas_paths = draw_spot_canvases(
        image_paths=image_paths,
        polygon_detections=polygon_detections,
        sample_keys=sample_keys,
        canvas_dir=output_dir / "canvas",
    )
    background_counts = compute_background_counts(
        image_paths=image_paths,
        canvas_paths=canvas_paths,
        sample_keys=sample_keys,
        roi_canvas_dir=output_dir / "rois_canvased",
    )
    canvas_df = compute_canvas_data(background_counts, distance_area_df)

    save_outputs(output_dir, spotwise_df, global_spot_df, global_df, distance_area_df, canvas_df)

    if not args.skip_summary_plots:
        save_summary_plots(
            global_spot_df=global_spot_df,
            distance_area_df=distance_area_df,
            canvas_df=canvas_df,
            figures_dir=output_dir / "new_figures",
        )

    print("Pipeline complete. Outputs written to:", output_dir.resolve())


if __name__ == "__main__":
    main()
