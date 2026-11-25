#!/usr/bin/env python3
"""
Spatial normalisation of spot measures (referee comment 5).

Converts every spatial measure to dimensionless, operculum-relative units and
re-fits the spot-area/position model:

  * spot area          -> percentage of operculum area
  * centroid distance  -> fraction of operculum major-axis length
  * model              -> log(relative area) ~ relative distance + (1 | FishID)

It also documents what the acquisition geometry can and cannot be shown to be
from the data held in this repository, and records the geometric consequence of
the upstream cropping convention.

Inputs, all from this repository:
  images/                     1080 x 1080 opercular crops
  labels/                     spot polygons
  operculum_masks/            operculum mask per image, same frame as images/
  reference_tabular_data.csv  biometrics for the 21 fish

Run:
    python spatial_normalisation.py --output-dir results/paper_2

Datasets
--------
Reads   images/                     84 opercular crops (P.IMAGES_DIR)
        labels/                     spot polygons (P.LABELS_DIR)
        operculum_masks/            84 operculum masks (P.OPERCULUM_MASKS_DIR)
        reference_tabular_data.csv  biometrics, n = 21 (P.BIOMETRICS_CSV)
Writes  results/paper_2/data/spatial_relative_spotwise.csv
        results/paper_2/data/spatial_operculum_geometry.csv
        results/paper_2/data/spatial_crop_convention.json
        results/paper_2/data/spatial_models.csv
        results/paper_2/data/spatial_geometry_check.csv
        results/paper_2/data/spatial_geometry_merged.csv
        results/paper_2/data/spatial_biometrics_long.csv
        results/paper_2/data/spatial_biology_vs_size.csv
        results/paper_2/data/spatial_by_fish.csv
        results/paper_2/data/spatial_size_categories.csv
        results/paper_2/figures/fig_size_categories_relative.png
        results/paper_2/figures/fig_distance_area_relative.png
        results/paper_2/figures/fig_geometry_check.png

Note: 10 of the 2,615 labelled spots fall outside the supplied operculum masks
and are excluded here, so this and later scripts report 2,605. See report
Section 10.2 for the reconciliation.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image
from scipy import ndimage, stats

from photometric_analysis import (
    format_p, IMAGE_EXTENSIONS, load_paths, load_polygon_detections, parse_sample_name,
    polygon_to_pixels, polygon_mask, sample_sort_key, tissue_mask_from_rgb,
)

FRAME = 1080
MASK_THRESHOLD = 127

PIXEL_GREY = "0.30"
RELATIVE_GREY = "0.78"
POINT_GREY = "0.45"

PUBLISHED = {"estimate": -0.00151, "ci_lower": -0.00161, "ci_upper": -0.00141}

COL_DAY1 = "fiskenr dag 1"
COL_DAY2 = "fiskenr dag 2, post-stress"
COL_LENGTH = "lengde"
COL_WEIGHT = "vekt"
COL_K = "K"
COL_SEX = "Kjønn"


def image_id_from_sample(sample: str) -> int | None:
    """fish_10_S_day1_id-51_L_h -> 51."""
    m = re.search(r"id-(\d+)", sample)
    return int(m.group(1)) if m else None


def largest_component(mask: np.ndarray) -> np.ndarray:
    labelled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labelled, index=np.arange(1, n + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


def operculum_geometry(mask: np.ndarray) -> dict:
    """
    Area and principal axes of the operculum.

    The major axis is the first principal component of the mask's pixel
    coordinates, oriented so that increasing projection points in the +x
    direction. The original pipeline measured distance from the right-hand edge
    of the frame, so +x is the posterior reference direction; left- and
    right-side images were mirrored to a common orientation upstream, which is
    verified in `orientation_check`.
    """
    ys, xs = np.nonzero(mask)
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    centre = coords.mean(axis=0)
    centred = coords - centre
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, int(np.argmax(eigvals))]
    minor_vec = eigvecs[:, int(np.argmin(eigvals))]
    if major_vec[0] < 0:
        major_vec = -major_vec
    proj_major = centred @ major_vec
    proj_minor = centred @ minor_vec
    return {
        "operculum_area_px": int(mask.sum()),
        "major_axis_px": float(proj_major.max() - proj_major.min()),
        "minor_axis_px": float(proj_minor.max() - proj_minor.min()),
        "major_vec_x": float(major_vec[0]), "major_vec_y": float(major_vec[1]),
        "centre_x": float(centre[0]), "centre_y": float(centre[1]),
        "proj_major_min": float(proj_major.min()), "proj_major_max": float(proj_major.max()),
        "bbox_w": int(xs.max() - xs.min() + 1), "bbox_h": int(ys.max() - ys.min() + 1),
        "equiv_diameter_px": float(2.0 * np.sqrt(mask.sum() / np.pi)),
    }


def build_spot_table(
    image_paths: dict[str, Path],
    detections: dict[str, list[list[float]]],
    masks_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot_rows: list[dict] = []
    image_rows: list[dict] = []

    keys = sorted(set(image_paths) & set(detections), key=sample_sort_key)
    for n, key in enumerate(keys, 1):
        fish_id, day, side = parse_sample_name(key)
        mask_path = masks_dir / f"{key}.jpg"
        if not mask_path.exists():
            warnings.warn(f"No operculum mask for {key}; skipping")
            continue
        op = largest_component(np.array(Image.open(mask_path).convert("L")) > MASK_THRESHOLD)
        geom = operculum_geometry(op)

        tissue = tissue_mask_from_rgb(
            np.array(Image.open(image_paths[key]).convert("RGB"), dtype=np.uint8))
        inter = int((op & tissue).sum())
        union = int((op | tissue).sum())
        n_labelled = 0
        n_outside = 0
        centre = np.array([geom["centre_x"], geom["centre_y"]])
        major = np.array([geom["major_vec_x"], geom["major_vec_y"]])
        span = geom["proj_major_max"] - geom["proj_major_min"]

        spot_px_total = 0
        for idx, poly in enumerate(detections[key]):
            verts = polygon_to_pixels(poly, FRAME, FRAME)
            poly_mask = polygon_mask(FRAME, FRAME, verts)
            n_labelled += 1
            m = poly_mask & op
            area = int(m.sum())
            if area == 0:
                n_outside += 1
                continue
            spot_px_total += area
            ys, xs = np.nonzero(m)
            cx, cy = float(xs.mean()), float(ys.mean())

            legacy_distance = float(FRAME - cx)
            proj = float((np.array([cx, cy]) - centre) @ major)
            rel_distance = (geom["proj_major_max"] - proj) / span if span > 0 else np.nan

            spot_rows.append({
                "sample": key, "FISH_ID": fish_id, "DAY": day, "SIDE": side, "spot": idx,
                "area_px": area,
                "rel_area_pct": 100.0 * area / geom["operculum_area_px"],
                "centroid_x": cx, "centroid_y": cy,
                "distance_px": legacy_distance,
                "rel_distance_major": rel_distance,
                "rel_distance_simple": legacy_distance / geom["major_axis_px"],
                "operculum_area_px": geom["operculum_area_px"],
                "major_axis_px": geom["major_axis_px"],
            })

        row = {"sample": key, "FISH_ID": fish_id, "DAY": day, "SIDE": side,
               "spot_px_total": spot_px_total,
               "rel_spot_area_pct": 100.0 * spot_px_total / geom["operculum_area_px"],
               "n_labelled_spots": n_labelled,
               "n_spots_outside_mask": n_outside,
               "mask_tissue_iou": inter / union if union else float("nan")}
        row.update(geom)
        image_rows.append(row)
        if n % 20 == 0:
            print(f"  {n}/{len(keys)} images")

    return pd.DataFrame(spot_rows), pd.DataFrame(image_rows)


def orientation_check(spots: pd.DataFrame) -> dict:
    """
    Were left- and right-side images brought to a common orientation upstream?

    If they were, the normalised anterior-posterior position of spots should
    have the same distribution on both sides. If they were not, applying one
    directional convention to both would conflate two opposite axes.
    """
    left = spots[spots["SIDE"] == "L"]["rel_distance_major"].dropna()
    right = spots[spots["SIDE"] == "R"]["rel_distance_major"].dropna()
    u, p = stats.mannwhitneyu(left, right)
    return {
        "mean_left": float(left.mean()), "mean_right": float(right.mean()),
        "sd_left": float(left.std()), "sd_right": float(right.std()),
        "n_left": int(len(left)), "n_right": int(len(right)),
        "mannwhitney_p": float(p),
        "difference": float(left.mean() - right.mean()),
    }


def crop_convention(images: pd.DataFrame) -> dict:
    """
    Geometric consequences of the upstream crop, measured from the masks.

    Every image was cropped to the operculum bounding box and resized to a fixed
    square frame. That has a direct bearing on comment 5: pixel measures in such
    a frame are already implicitly expressed relative to operculum size.
    """
    ar = images["bbox_w"] / images["bbox_h"]
    return {
        "n": int(len(images)),
        "operculum_area_mean_px": float(images["operculum_area_px"].mean()),
        "operculum_area_cv_pct": float(100 * images["operculum_area_px"].std()
                                       / images["operculum_area_px"].mean()),
        "operculum_area_max_over_min": float(images["operculum_area_px"].max()
                                             / images["operculum_area_px"].min()),
        "bbox_w_mean": float(images["bbox_w"].mean()), "bbox_w_sd": float(images["bbox_w"].std()),
        "bbox_h_mean": float(images["bbox_h"].mean()), "bbox_h_sd": float(images["bbox_h"].std()),
        "bbox_aspect_mean": float(ar.mean()), "bbox_aspect_sd": float(ar.std()),
        "frame_px": FRAME,
        "fill_fraction_mean": float(images["operculum_area_px"].mean() / FRAME ** 2),
        "major_axis_mean_px": float(images["major_axis_px"].mean()),
        "major_axis_cv_pct": float(100 * images["major_axis_px"].std()
                                   / images["major_axis_px"].mean()),
        "n_labelled_spots": int(images["n_labelled_spots"].sum()),
        "n_spots_outside_mask": int(images["n_spots_outside_mask"].sum()),
        "n_spots_retained": int(images["n_labelled_spots"].sum()
                                - images["n_spots_outside_mask"].sum()),
        "pct_spots_outside_mask": float(100 * images["n_spots_outside_mask"].sum()
                                        / images["n_labelled_spots"].sum()),
        "mask_tissue_iou_mean": float(images["mask_tissue_iou"].mean()),
        "n_images_with_excluded_spots": int((images["n_spots_outside_mask"] > 0).sum()),
    }


def load_biometrics(path: Path) -> pd.DataFrame:
    """Long-form biometrics keyed by the image id that appears in filenames."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]
    rows: list[dict] = []
    for _, r in raw.iterrows():
        for col, day in [(COL_DAY1, "1"), (COL_DAY2, "2")]:
            if pd.isna(r.get(col)):
                continue
            rows.append({
                "image_id": int(r[col]), "DAY": day,
                "fork_length_cm": float(r[COL_LENGTH]) if pd.notna(r.get(COL_LENGTH)) else np.nan,
                "body_mass_g": float(r[COL_WEIGHT]) if pd.notna(r.get(COL_WEIGHT)) else np.nan,
                "condition_factor": float(r[COL_K]) if pd.notna(r.get(COL_K)) else np.nan,
                "sex": str(r[COL_SEX]).strip() if pd.notna(r.get(COL_SEX)) else None,
            })
    df = pd.DataFrame(rows)
    df["mass_two_thirds"] = df["body_mass_g"] ** (2.0 / 3.0)
    df["length_squared"] = df["fork_length_cm"] ** 2
    return df


def fit_models(spots: pd.DataFrame) -> pd.DataFrame:
    """The spot-size/position model, in legacy pixel units and in relative units."""
    specs = [
        ("Legacy: log(area px) ~ distance px", "np.log(area_px)", "distance_px"),
        ("Relative: log(area %) ~ relative distance", "np.log(rel_area_pct)", "rel_distance_major"),
        ("Relative (simple ratio): log(area %) ~ distance / major axis",
         "np.log(rel_area_pct)", "rel_distance_simple"),
    ]
    rows: list[dict] = []
    for label, response, predictor in specs:
        data = spots[["FISH_ID", "area_px", "rel_area_pct", "distance_px",
                      "rel_distance_major", "rel_distance_simple"]].dropna()
        fit = smf.mixedlm(f"{response} ~ {predictor}", data, groups=data["FISH_ID"]).fit(reml=True)
        beta = float(fit.params[predictor])
        lo, hi = (float(v) for v in fit.conf_int().loc[predictor])
        span = float(data[predictor].max() - data[predictor].min())
        rows.append({
            "model": label, "n_spots": int(len(data)),
            "n_fish": int(data["FISH_ID"].nunique()),
            "predictor": predictor, "estimate": beta,
            "ci_lower": lo, "ci_upper": hi,
            "p_value": float(fit.pvalues[predictor]),
            "predictor_span": span,
            "pct_change_over_span": 100.0 * (np.exp(beta * span) - 1.0),
            "group_var": float(fit.cov_re.iloc[0, 0]),
            "resid_var": float(fit.scale),
        })
    return pd.DataFrame(rows)


def geometry_check(images: pd.DataFrame, bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrospective geometry check against body size.

    Note the important limitation handled in the report: the operculum masks in
    this repository are in the cropped frame, in which operculum size is fixed
    by construction. The regression is therefore run and reported as a negative
    control that demonstrates the information is absent, not as evidence about
    working distance.
    """
    df = images.copy()
    df["image_id"] = df["sample"].map(image_id_from_sample)
    merged = df.merge(bio, on=["image_id", "DAY"], how="left")

    rows: list[dict] = []
    for response in ["operculum_area_px", "major_axis_px"]:
        for predictor in ["mass_two_thirds", "length_squared"]:
            data = merged[[response, predictor]].dropna()
            if len(data) < 5:
                continue
            fit = stats.linregress(data[predictor], data[response])
            resid = data[response] - (fit.intercept + fit.slope * data[predictor])
            rows.append({
                "response": response, "predictor": predictor, "n": int(len(data)),
                "slope": float(fit.slope), "r": float(fit.rvalue),
                "r_squared": float(fit.rvalue ** 2), "p_value": float(fit.pvalue),
                "residual_sd": float(resid.std(ddof=2)),
                "residual_cv_pct": float(100 * resid.std(ddof=2) / data[response].mean()),
                "response_cv_pct": float(100 * data[response].std() / data[response].mean()),
            })
    return pd.DataFrame(rows), merged


def biology_vs_size(images: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """
    Do the dimensionless pigmentation measures scale with body size?

    Body size is a property of the fish, not of the image, and each fish
    contributes four images (two sides x two occasions). The test is therefore
    run on fish-level means (n = 21 independent units); running it per image
    would quadruple the apparent sample size without adding information.
    """
    df = images.copy()
    df["image_id"] = df["sample"].map(image_id_from_sample)
    merged = df.merge(bio, on=["image_id", "DAY"], how="left")

    by_fish = (merged.groupby("FISH_ID", as_index=False)
               .agg(rel_spot_area_pct=("rel_spot_area_pct", "mean"),
                    fork_length_cm=("fork_length_cm", "mean"),
                    body_mass_g=("body_mass_g", "mean"),
                    condition_factor=("condition_factor", "mean"),
                    n_images=("sample", "size")))

    rows: list[dict] = []
    for predictor in ["fork_length_cm", "body_mass_g", "condition_factor"]:
        for label, data, unit in [
            ("per image (pseudoreplicated)", merged, "image"),
            ("per fish", by_fish, "fish"),
        ]:
            sub = data[["rel_spot_area_pct", predictor]].dropna()
            if len(sub) < 5:
                continue
            fit = stats.linregress(sub[predictor], sub["rel_spot_area_pct"])
            rows.append({
                "predictor": predictor, "unit_of_analysis": label, "n": int(len(sub)),
                "r": float(fit.rvalue), "r_squared": float(fit.rvalue ** 2),
                "p_value": float(fit.pvalue), "slope": float(fit.slope),
            })
    return pd.DataFrame(rows), by_fish


def figure_size_categories(spots: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Manuscript Figure 12, regenerated in operculum-relative units.

    The two panels are near-identical, and that is the result rather than a
    duplication: the crop convention fixes apparent operculum size to within a
    few percent (Section 10.1), so dividing pixel area by operculum area is
    close to dividing by a constant and barely reshuffles the categories.
    Percentages are therefore printed to one decimal, because rounding to whole
    percent collapses the real differences to identical labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    specs = [
        ("area_px", "Spot area (pixels)", PIXEL_GREY, None, "{:,.0f} px"),
        ("rel_area_pct", "Spot area (% of operculum area)", RELATIVE_GREY, "///",
         "{:.3f}%"),
    ]

    computed = []
    for col, label, grey, hatch, wfmt in specs:
        v = spots[col].dropna().to_numpy(float)
        edges = np.linspace(0, v.max(), 11)
        ids = np.clip(np.digitize(v, edges[1:-1], right=True), 0, 9)
        counts = np.array([(ids == i).sum() for i in range(10)])
        computed.append((label, grey, hatch, counts, 100.0 * counts / counts.sum(),
                         wfmt.format(edges[1])))

    ymax = max(p.max() for *_, p, _ in computed) * 1.20
    spread = np.abs(computed[0][4] - computed[1][4]).max()

    for ax, (label, grey, hatch, counts, pct, width) in zip(axes, computed):
        ax.bar(np.arange(10) + 1, pct, color=grey, edgecolor="black",
               linewidth=0.7, hatch=hatch)
        for i, p in enumerate(pct):
            if p >= 0.4:
                ax.text(i + 1, p + ymax * 0.015, f"{p:.1f}%", ha="center",
                        fontsize=8)
        ax.set_xticks(np.arange(10) + 1)
        ax.set_xticklabels([f"c{i}" for i in range(1, 11)])
        ax.set_xlabel("Size category", fontsize=9)
        ax.set_ylabel("% of spots")
        ax.set_ylim(0, ymax)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    v = spots["rel_area_pct"].dropna().to_numpy(float)
    edges = np.linspace(0, v.max(), 11)
    ids = np.clip(np.digitize(v, edges[1:-1], right=True), 0, 9)
    counts = np.array([(ids == i).sum() for i in range(10)])
    vp = spots["area_px"].dropna().to_numpy(float)
    ep = np.linspace(0, vp.max(), 11)
    ip_ = np.clip(np.digitize(vp, ep[1:-1], right=True), 0, 9)
    cp = np.array([(ip_ == i).sum() for i in range(10)])
    return pd.DataFrame({
        "category": [f"c{i}" for i in range(1, 11)],
        "n_pixel_units": cp, "pct_pixel_units": 100.0 * cp / cp.sum(),
        "upper_edge_rel_pct": edges[1:],
        "n_relative_units": counts, "pct_relative_units": 100.0 * counts / counts.sum(),
    })


def figure_distance_area(spots: pd.DataFrame, models: pd.DataFrame, out_path: Path) -> None:
    """Manuscript Figure 13, regenerated in dimensionless units."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    panels = [
        ("distance_px", "area_px", "Distance from posterior edge (pixels)",
         "Spot area (pixels)", PIXEL_GREY, "Pixel units",
         "Legacy: log(area px) ~ distance px"),
        ("rel_distance_major", "rel_area_pct",
         "Relative position along operculum major axis\n(0 = posterior, 1 = anterior)",
         "Spot area (% of operculum area)", POINT_GREY, "Dimensionless units",
         "Relative: log(area %) ~ relative distance"),
    ]
    for ax, (xc, yc, xl, yl, grey, unit_label, model_label) in zip(axes, panels):
        data = spots[[xc, yc]].dropna()
        ax.scatter(data[xc], data[yc], s=11, alpha=0.30, color=grey, edgecolors="none")
        m = models[models["model"] == model_label].iloc[0]
        xs = np.linspace(data[xc].min(), data[xc].max(), 100)
        intercept = np.log(data[yc]).mean() - m["estimate"] * data[xc].mean()
        ax.plot(xs, np.exp(intercept + m["estimate"] * xs), color="black", lw=2.2)
        ax.set_yscale("log")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f"{unit_label}\n"
                     f"β = {m['estimate']:.4g} [{m['ci_lower']:.4g}, {m['ci_upper']:.4g}], "
                     f"{format_p(m['p_value'])}", fontsize=10)
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_geometry(merged: pd.DataFrame, geo: pd.DataFrame, crop: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.hist(merged["operculum_area_px"] / 1000.0, bins=20, color="#7570b3",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Operculum area in the cropped frame (10³ px)")
    ax.set_ylabel("Images")
    ax.set_title(f"Size is fixed by the crop\nCV = {crop['operculum_area_cv_pct']:.1f}%, "
                 f"max/min = {crop['operculum_area_max_over_min']:.2f}x", fontsize=10)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)

    for ax, predictor, xl in zip(
        axes[1:],
        ["mass_two_thirds", "length_squared"],
        ["Body mass$^{2/3}$ (g$^{2/3}$)", "Fork length² (cm²)"],
    ):
        data = merged[["operculum_area_px", predictor]].dropna()
        ax.scatter(data[predictor], data["operculum_area_px"] / 1000.0, s=42, alpha=0.8,
                   color="#7570b3", edgecolors="black", linewidths=0.4)
        row = geo[(geo["response"] == "operculum_area_px") & (geo["predictor"] == predictor)]
        if len(row):
            r = row.iloc[0]
            fit = stats.linregress(data[predictor], data["operculum_area_px"])
            xs = np.linspace(data[predictor].min(), data[predictor].max(), 100)
            ax.plot(xs, (fit.intercept + fit.slope * xs) / 1000.0, color="black", lw=1.8)
            ax.set_title(f"r = {r['r']:.3f}, R² = {r['r_squared']:.3f}, {format_p(r['p_value'])}",
                         fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("Operculum area in crop (10³ px)")
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=P.IMAGES_DIR)
    p.add_argument("--labels-dir", type=Path, default=P.LABELS_DIR)
    p.add_argument("--masks-dir", type=Path, default=P.OPERCULUM_MASKS_DIR)
    p.add_argument("--biometrics", type=Path, default=P.BIOMETRICS_CSV)
    p.add_argument("--output-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()

    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    image_paths = load_paths(args.images_dir, IMAGE_EXTENSIONS)
    detections = load_polygon_detections(args.labels_dir)

    print("[1/4] Operculum geometry and relative spot measures")
    spots, images = build_spot_table(image_paths, detections, args.masks_dir)
    spots.to_csv(out / "data" / "spatial_relative_spotwise.csv", index=False)
    images.to_csv(out / "data" / "spatial_operculum_geometry.csv", index=False)
    print(f"  {len(spots)} spots across {len(images)} images")

    crop = crop_convention(images)
    orient = orientation_check(spots)
    (out / "data" / "spatial_crop_convention.json").write_text(
        json.dumps({"crop": crop, "orientation": orient}, indent=2), encoding="utf-8")
    print(f"  operculum area in crop frame: CV {crop['operculum_area_cv_pct']:.1f}%, "
          f"max/min {crop['operculum_area_max_over_min']:.2f}x")
    print(f"  bbox {crop['bbox_w_mean']:.0f} x {crop['bbox_h_mean']:.0f} px in a "
          f"{FRAME} x {FRAME} frame")
    print(f"  L/R orientation: mean rel. position {orient['mean_left']:.3f} vs "
          f"{orient['mean_right']:.3f}, p = {orient['mannwhitney_p']:.3f}")

    print("\n[2/4] Mixed models")
    models = fit_models(spots)
    models.to_csv(out / "data" / "spatial_models.csv", index=False)
    print(models[["model", "n_spots", "estimate", "ci_lower", "ci_upper",
                  "p_value", "pct_change_over_span"]].to_string(index=False))

    print("\n[3/4] Geometry check against biometrics")
    bio = load_biometrics(args.biometrics)
    bio.to_csv(out / "data" / "spatial_biometrics_long.csv", index=False)
    geo, merged = geometry_check(images, bio)
    geo.to_csv(out / "data" / "spatial_geometry_check.csv", index=False)
    merged.to_csv(out / "data" / "spatial_geometry_merged.csv", index=False)
    print(geo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bio_size, by_fish = biology_vs_size(images, bio)
    bio_size.to_csv(out / "data" / "spatial_biology_vs_size.csv", index=False)
    by_fish.to_csv(out / "data" / "spatial_by_fish.csv", index=False)
    print()
    print(bio_size.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[4/4] Figures")
    cats = figure_size_categories(spots, out / "figures" / "fig_size_categories_relative.png")
    cats.to_csv(out / "data" / "spatial_size_categories.csv", index=False)
    figure_distance_area(spots, models, out / "figures" / "fig_distance_area_relative.png")
    figure_geometry(merged, geo, crop, out / "figures" / "fig_geometry_check.png")
    print("  figures written")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
