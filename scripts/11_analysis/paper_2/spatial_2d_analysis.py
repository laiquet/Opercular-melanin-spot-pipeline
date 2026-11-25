#!/usr/bin/env python3
"""
Standardized operculum coordinate system and two-dimensional spatial analysis
(referee comment 6).

Builds a canonical, dimensionless frame for the operculum:

  1. PCA on the operculum mask's pixel coordinates gives its principal axes.
  2. The frame is rotated so the major axis is horizontal and the minor axis
     vertical, which removes rotational variation between images.
  3. Left- and right-side images are brought onto a common frame; whether a
     mirror is required is decided from the data rather than assumed.
  4. Coordinates are scaled to the operculum's own bounding box, so every spot
     centroid becomes a normalised (x, y) pair in [0,1]^2.

It then produces pooled spatial maps and re-analyses spot size in two
dimensions: log(relative area) ~ x + y + (1 | FishID).

Inputs, all from this repository:
  images/, labels/, operculum_masks/

Run:
    python spatial_2d_analysis.py --output-dir results/paper_2

Datasets
--------
Reads   images/, labels/, operculum_masks/
Writes  results/paper_2/data/spatial2d_spotwise.csv
        results/paper_2/data/spatial2d_image_geometry.csv
        results/paper_2/data/spatial2d_alignment.json
        results/paper_2/data/spatial2d_models.csv
        results/paper_2/data/spatial2d_occupancy.npy
        results/paper_2/figures/fig_spatial_maps.png
        results/paper_2/figures/fig_spatial_gradients.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P


import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import Normalize
from PIL import Image
from scipy import ndimage, stats

from photometric_analysis import (
    format_p, IMAGE_EXTENSIONS, load_paths, load_polygon_detections, parse_sample_name,
    polygon_mask, polygon_to_pixels, sample_sort_key,
)

FRAME = 1080
MASK_THRESHOLD = 127
GRID = 120
MAP_BINS = 14
MIN_BIN_COUNT = 8


SIDE_TRANSFORMS = {
    "none": (False, False),
    "flip_x": (True, False),
    "flip_y": (False, True),
    "flip_xy": (True, True),
}


MAJOR_GREY = "0.20"
MINOR_GREY = "0.55"

TRANSFORM_NAME = {
    "none": "no transform",
    "flip_x": "anterior-posterior flip",
    "flip_y": "dorsal-ventral flip",
    "flip_xy": "180 degree rotation",
}


def largest_component(mask: np.ndarray) -> np.ndarray:
    labelled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labelled, index=np.arange(1, n + 1))
    return labelled == (int(np.argmax(sizes)) + 1)


def canonical_basis(mask: np.ndarray) -> dict:
    """
    Principal axes of the operculum, in a fixed orientation convention.

    The major axis is oriented so that its image-x component is positive, and
    the minor axis so that its image-y component is positive. Image y increases
    downward, so with fish photographed dorsal-side up the minor axis points
    ventrally. Both conventions are applied identically to every image, which is
    what removes rotational variation; the anatomical labels attached to them
    are a separate claim, checked in `orientation_consistency`.
    """
    ys, xs = np.nonzero(mask)
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    centre = coords.mean(axis=0)
    centred = coords - centre
    eigvals, eigvecs = np.linalg.eigh(np.cov(centred, rowvar=False))
    major = eigvecs[:, int(np.argmax(eigvals))]
    minor = eigvecs[:, int(np.argmin(eigvals))]
    if major[0] < 0:
        major = -major
    if minor[1] < 0:
        minor = -minor
    u = centred @ major
    v = centred @ minor
    return {
        "centre": centre, "major": major, "minor": minor,
        "u_min": float(u.min()), "u_max": float(u.max()),
        "v_min": float(v.min()), "v_max": float(v.max()),
        "major_len": float(u.max() - u.min()),
        "minor_len": float(v.max() - v.min()),
        "area_px": int(mask.sum()),
        "elongation": float((u.max() - u.min()) / (v.max() - v.min())),
        "rotation_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
    }


def to_canonical(points: np.ndarray, basis: dict,
                 flip_x: bool = False, flip_y: bool = False) -> np.ndarray:
    """
    Map image-space points to the canonical unit frame.

    x runs 0 at the posterior extreme to 1 at the anterior extreme; y runs 0 to
    1 across the minor axis. The flips bring one side onto the other's frame.
    """
    centred = points - basis["centre"]
    u = centred @ basis["major"]
    v = centred @ basis["minor"]
    x = (basis["u_max"] - u) / (basis["u_max"] - basis["u_min"])
    y = (v - basis["v_min"]) / (basis["v_max"] - basis["v_min"])
    if flip_x:
        x = 1.0 - x
    if flip_y:
        y = 1.0 - y
    return np.column_stack([x, y])


def build_tables(
    image_paths: dict[str, Path],
    detections: dict[str, list[list[float]]],
    masks_dir: Path,
    side_transform: str = "none",
    transform_side: str = "R",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    spot_rows: list[dict] = []
    image_rows: list[dict] = []
    occupancy = np.zeros((GRID, GRID), dtype=float)
    n_masks = 0
    tx, ty = SIDE_TRANSFORMS[side_transform]

    keys = sorted(set(image_paths) & set(detections), key=sample_sort_key)
    for n, key in enumerate(keys, 1):
        fish_id, day, side = parse_sample_name(key)
        mask_path = masks_dir / f"{key}.jpg"
        if not mask_path.exists():
            continue
        op = largest_component(np.array(Image.open(mask_path).convert("L")) > MASK_THRESHOLD)
        basis = canonical_basis(op)
        fx, fy = (tx, ty) if side == transform_side else (False, False)

        ys, xs = np.nonzero(op)
        pts = to_canonical(np.column_stack([xs.astype(float), ys.astype(float)]), basis, fx, fy)
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=GRID, range=[[0, 1], [0, 1]])
        occupancy += (h > 0).astype(float)
        n_masks += 1

        for idx, poly in enumerate(detections[key]):
            m = polygon_mask(FRAME, FRAME, polygon_to_pixels(poly, FRAME, FRAME)) & op
            area = int(m.sum())
            if area == 0:
                continue
            sy, sx = np.nonzero(m)
            centroid = np.array([[sx.mean(), sy.mean()]])
            x, y = to_canonical(centroid, basis, fx, fy)[0]
            spot_rows.append({
                "sample": key, "FISH_ID": fish_id, "DAY": day, "SIDE": side, "spot": idx,
                "x": float(x), "y": float(y),
                "area_px": area,
                "rel_area_pct": 100.0 * area / basis["area_px"],
                "operculum_area_px": basis["area_px"],
            })

        image_rows.append({
            "sample": key, "FISH_ID": fish_id, "DAY": day, "SIDE": side,
            "rotation_deg": basis["rotation_deg"], "elongation": basis["elongation"],
            "major_len_px": basis["major_len"], "minor_len_px": basis["minor_len"],
            "operculum_area_px": basis["area_px"],
            "flip_x": fx, "flip_y": fy,
        })
        if n % 20 == 0:
            print(f"  {n}/{len(keys)} images")

    return pd.DataFrame(spot_rows), pd.DataFrame(image_rows), occupancy / max(n_masks, 1)


def side_alignment_test(spots: pd.DataFrame, bins: int = 12) -> dict:
    """
    How should right-side images be brought onto the left-side frame?

    Decided from the data rather than assumed. The pooled 2-D spot density of
    each side is compared under each candidate transform of the right side, and
    the transform that maximises agreement is adopted. The PCA gives axes but
    not their anatomical polarity, so this step is what fixes the polarity; it
    also reveals whether the upstream pipeline already aligned the sides on one
    axis but not the other.
    """
    left = spots[spots["SIDE"] == "L"]
    right = spots[spots["SIDE"] == "R"]
    hl, _, _ = np.histogram2d(left["x"], left["y"], bins=bins, range=[[0, 1], [0, 1]])
    hl = hl / hl.sum()

    out: dict = {}
    for name, (fx, fy) in SIDE_TRANSFORMS.items():
        rx = 1.0 - right["x"] if fx else right["x"]
        ry = 1.0 - right["y"] if fy else right["y"]
        hr, _, _ = np.histogram2d(rx, ry, bins=bins, range=[[0, 1], [0, 1]])
        hr = hr / hr.sum()
        out[name] = {
            "flip_x": fx, "flip_y": fy,
            "density_correlation": float(np.corrcoef(hl.ravel(), hr.ravel())[0, 1]),
            "total_variation_distance": float(0.5 * np.abs(hl - hr).sum()),
            "mean_x_left": float(left["x"].mean()), "mean_x_right": float(rx.mean()),
            "mean_y_left": float(left["y"].mean()), "mean_y_right": float(ry.mean()),
        }
    best = max(out, key=lambda k: out[k]["density_correlation"])
    out["decision"] = best
    return out


def apply_side_transform(spots: pd.DataFrame, transform: str, side: str = "R") -> pd.DataFrame:
    """Apply the chosen flip to one side, in place on a copy."""
    fx, fy = SIDE_TRANSFORMS[transform]
    df = spots.copy()
    sel = df["SIDE"] == side
    if fx:
        df.loc[sel, "x"] = 1.0 - df.loc[sel, "x"]
    if fy:
        df.loc[sel, "y"] = 1.0 - df.loc[sel, "y"]
    df["side_transform"] = np.where(sel, transform, "none")
    return df


def fit_2d_models(spots: pd.DataFrame) -> pd.DataFrame:
    """log(relative area) against the two canonical coordinates."""
    data = spots[["FISH_ID", "rel_area_pct", "x", "y"]].dropna()
    specs = [
        ("2D: log(rel area) ~ x + y", "np.log(rel_area_pct) ~ x + y", ["x", "y"]),
        ("1D anterior-posterior only: log(rel area) ~ x", "np.log(rel_area_pct) ~ x", ["x"]),
        ("1D dorsal-ventral only: log(rel area) ~ y", "np.log(rel_area_pct) ~ y", ["y"]),
    ]
    rows: list[dict] = []
    for label, formula, terms in specs:
        fit = smf.mixedlm(formula, data, groups=data["FISH_ID"]).fit(reml=False)
        for term in terms:
            beta = float(fit.params[term])
            lo, hi = (float(v) for v in fit.conf_int().loc[term])
            rows.append({
                "model": label, "term": term, "n_spots": int(len(data)),
                "n_fish": int(data["FISH_ID"].nunique()),
                "estimate": beta, "ci_lower": lo, "ci_upper": hi,
                "p_value": float(fit.pvalues[term]),
                "pct_change_across_frame": 100.0 * (np.exp(beta) - 1.0),
                "aic": float(fit.aic),
            })
    return pd.DataFrame(rows)


def orientation_consistency(images: pd.DataFrame, spots: pd.DataFrame) -> dict:
    """How much rotational variation the canonical frame removed, and side agreement."""
    left = spots[spots["SIDE"] == "L"]
    right = spots[spots["SIDE"] == "R"]
    return {
        "rotation_deg_mean": float(images["rotation_deg"].mean()),
        "rotation_deg_sd": float(images["rotation_deg"].std()),
        "rotation_deg_min": float(images["rotation_deg"].min()),
        "rotation_deg_max": float(images["rotation_deg"].max()),
        "rotation_range_deg": float(images["rotation_deg"].max() - images["rotation_deg"].min()),
        "elongation_mean": float(images["elongation"].mean()),
        "elongation_sd": float(images["elongation"].std()),
        "mean_x_left": float(left["x"].mean()), "mean_x_right": float(right["x"].mean()),
        "mean_y_left": float(left["y"].mean()), "mean_y_right": float(right["y"].mean()),
        "x_side_p": float(stats.mannwhitneyu(left["x"], right["x"])[1]),
        "y_side_p": float(stats.mannwhitneyu(left["y"], right["y"])[1]),
    }


def outline_from_occupancy(occ: np.ndarray, level: float = 0.5):
    return occ.T, level


def figure_coordinate_system(
    image_paths: dict[str, Path],
    detections: dict[str, list[list[float]]],
    masks_dir: Path,
    spots: pd.DataFrame,
    spots_raw: pd.DataFrame,
    occ: np.ndarray,
    align: dict,
    chosen: str,
    out_path: Path,
) -> None:
    """
    Illustrate how the standardized opercular frame is constructed.

    The Methods description is four abstract steps; a reader has no way to check
    them against the data without seeing an operculum before and after. The top
    row follows one image through the construction, and the bottom row shows the
    evidence for the side-alignment decision, which is the only step chosen
    empirically rather than by definition.

    The example is the image whose opercular rotation in the source frame is
    closest to the median absolute rotation, so the rotation step is visible
    without being extreme.
    """
    geom = {}
    for key in sorted(set(image_paths) & set(detections), key=sample_sort_key):
        mask_path = masks_dir / f"{key}.jpg"
        if mask_path.exists():
            op = largest_component(
                np.array(Image.open(mask_path).convert("L")) > MASK_THRESHOLD)
            geom[key] = (op, canonical_basis(op))
    if not geom:
        return
    median_rot = np.median([abs(b["rotation_deg"]) for _, b in geom.values()])
    candidates = [k for k in geom if parse_sample_name(k)[2] == "L"] or list(geom)
    key = min(candidates,
              key=lambda k: abs(abs(geom[k][1]["rotation_deg"]) - median_rot))
    op, basis = geom[key]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.4))

    ax = axes[0, 0]
    rgb = np.array(Image.open(image_paths[key]).convert("RGB"), dtype=np.uint8)
    ax.imshow(rgb)
    ax.contour(op.astype(float), levels=[0.5], colors="#ffd400", linewidths=2.0)
    cx, cy = basis["centre"]
    for vec, half, colour, label in [
        (basis["major"], basis["major_len"] / 2, "#d95f02", "major axis (A–P)"),
        (basis["minor"], basis["minor_len"] / 2, "#1b9e77", "minor axis (D–V)"),
    ]:
        ax.annotate("", xy=(cx + vec[0] * half, cy + vec[1] * half),
                    xytext=(cx - vec[0] * half, cy - vec[1] * half),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=2.6))
        ax.plot([], [], color=colour, lw=2.6, label=label)
    ax.plot(cx, cy, "o", color="white", markersize=8, markeredgecolor="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"1-2. Principal axes of the operculum mask\n"
                 f"major axis at {basis['rotation_deg']:+.0f}° in the source frame",
                 fontsize=10)

    ax = axes[0, 1]
    ys, xs = np.nonzero(op)
    pts = to_canonical(np.column_stack([xs.astype(float), ys.astype(float)]), basis)
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=GRID, range=[[0, 1], [0, 1]])
    ax.imshow((hist.T > 0).astype(float), origin="lower", extent=(0, 1, 0, 1),
              cmap="Greys", alpha=0.30, aspect="equal")
    ax.contour((hist.T > 0).astype(float), levels=[0.5], extent=(0, 1, 0, 1),
               colors="#ffd400", linewidths=2.0)
    this = spots[spots["sample"] == key]
    ax.scatter(this["x"], this["y"], s=34, color="#7570b3", edgecolors="black",
               linewidths=0.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x  (0 = posterior, 1 = anterior)", fontsize=9)
    ax.set_ylabel("y  (minor axis)", fontsize=9)
    ax.set_title(f"3-4. Rotated and scaled to the unit box\n"
                 f"{len(this)} spot centroids of this image", fontsize=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)

    ax = axes[0, 2]
    ax.imshow(occ.T, origin="lower", extent=(0, 1, 0, 1), cmap="Greys",
              alpha=0.22, aspect="equal", vmin=0, vmax=1.6)
    ax.contour(occ.T, levels=[0.5], extent=(0, 1, 0, 1), colors="#ffd400", linewidths=2.0)
    ax.scatter(spots["x"], spots["y"], s=2.6, color="#4a3f8f", alpha=0.55,
               edgecolors="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x  (0 = posterior, 1 = anterior)", fontsize=9)
    ax.set_ylabel("y  (minor axis)", fontsize=9)
    ax.set_title(f"All {len(spots):,} spots from {len(geom)} images\n"
                 f"shading: mean opercular outline", fontsize=10)

    left = spots_raw[spots_raw["SIDE"] == "L"]
    right = spots_raw[spots_raw["SIDE"] == "R"]
    bins = 12
    panels = [
        (axes[1, 0], left["x"], left["y"], "Left side (reference)", None),
        (axes[1, 1], right["x"], right["y"], "Right side, untransformed",
         align["none"]["density_correlation"]),
        (axes[1, 2], 1.0 - right["x"] if SIDE_TRANSFORMS[chosen][0] else right["x"],
         1.0 - right["y"] if SIDE_TRANSFORMS[chosen][1] else right["y"],
         f"Right side after {TRANSFORM_NAME.get(chosen, chosen)}",
         align[chosen]["density_correlation"]),
    ]
    for ax, xv, yv, title, corr in panels:
        ax.hist2d(xv, yv, bins=bins, range=[[0, 1], [0, 1]], cmap="magma")
        ax.contour(occ.T, levels=[0.5], extent=(0, 1, 0, 1), colors="white",
                   linewidths=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        sub = "" if corr is None else f"\ndensity correlation with left = {corr:+.3f}"
        ax.set_title(title + sub, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_maps(spots: pd.DataFrame, occ: np.ndarray, models: pd.DataFrame, out_path: Path) -> None:
    x = spots["x"].to_numpy()
    y = spots["y"].to_numpy()
    area = spots["rel_area_pct"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    occT, lvl = outline_from_occupancy(occ)
    extent = (0, 1, 0, 1)

    ax = axes[0]
    kde = stats.gaussian_kde(np.vstack([x, y]))
    gx, gy = np.mgrid[0:1:complex(0, GRID), 0:1:complex(0, GRID)]
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape).T
    dens = np.where(occT >= lvl, dens, np.nan)
    im = ax.imshow(dens, origin="lower", extent=extent, cmap="magma", aspect="equal")
    ax.contour(occT, levels=[lvl], extent=extent, colors="white", linewidths=1.8)
    ax.scatter(x, y, s=2.0, c="white", alpha=0.18, edgecolors="none")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Centroid density")
    ax.set_title(f"Pooled centroid density, {len(spots):,} spots", fontsize=11)

    ax = axes[1]
    edges = np.linspace(0, 1, MAP_BINS + 1)
    sums, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=area)
    counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts >= MIN_BIN_COUNT, sums / counts, np.nan)
    occ_map = np.array(Image.fromarray(occT).resize((MAP_BINS, MAP_BINS), Image.BILINEAR))
    means = np.where(occ_map.T >= lvl, means, np.nan)
    im = ax.imshow(means.T, origin="lower", extent=extent, cmap="viridis", aspect="equal",
                   norm=Normalize(vmin=np.nanpercentile(means, 3),
                                  vmax=np.nanpercentile(means, 97)))
    ax.contour(occT, levels=[lvl], extent=extent, colors="white", linewidths=1.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                 label="Mean spot area (% of operculum)")
    m2 = models[models["model"].str.startswith("2D")]
    bx = m2[m2["term"] == "x"].iloc[0]
    by = m2[m2["term"] == "y"].iloc[0]
    ax.set_title(f"Mean relative spot area by location\n"
                 f"x: β = {bx['estimate']:.2f} ({format_p(bx['p_value'])});  "
                 f"y: β = {by['estimate']:.2f} ({format_p(by['p_value'])})", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Normalised anterior–posterior position\n"
                      "(0 = posterior, 1 = anterior)", fontsize=9)
        ax.set_ylabel("Normalised dorsal–ventral position\n(0 = dorsal, 1 = ventral)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_gradients(spots: pd.DataFrame, out_path: Path) -> None:
    """
    Marginal size gradients along each canonical axis.

    Grayscale, with the two axes separated by tone, line style, marker and fill
    texture rather than by hue. The panels share a y-axis so the steep
    anterior-posterior gradient and the shallow dorsal-ventral one can be
    compared directly instead of each being rescaled to fill its own panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, coord, axis_name, xlabel, grey, style, marker, hatch in [
        (axes[0], "x", "Major axis (anterior–posterior)",
         "Relative position  (0 = posterior, 1 = anterior)",
         MAJOR_GREY, "-", "o", None),
        (axes[1], "y", "Minor axis (dorsal–ventral)",
         "Relative position  (0 = dorsal, 1 = ventral)",
         MINOR_GREY, "--", "s", "///"),
    ]:
        edges = np.linspace(0, 1, 11)
        mids = 0.5 * (edges[:-1] + edges[1:])
        med, q1, q3, ns = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = spots[(spots[coord] >= lo) & (spots[coord] < hi)]["rel_area_pct"]
            med.append(sel.median() if len(sel) else np.nan)
            q1.append(sel.quantile(0.25) if len(sel) else np.nan)
            q3.append(sel.quantile(0.75) if len(sel) else np.nan)
            ns.append(len(sel))
        ax.fill_between(mids, q1, q3, facecolor=grey, alpha=0.30, hatch=hatch,
                        edgecolor="black", linewidth=0.0, label="interquartile range")
        ax.plot(mids, med, color=grey, marker=marker, ls=style, lw=2.2,
                markersize=6, markerfacecolor="white", markeredgecolor=grey,
                markeredgewidth=1.6, label="median")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
        ax.set_title(f"{axis_name}\nn = {sum(ns):,} spots", fontsize=10)
    axes[0].set_ylabel("Spot area (% of operculum)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=P.IMAGES_DIR)
    p.add_argument("--labels-dir", type=Path, default=P.LABELS_DIR)
    p.add_argument("--masks-dir", type=Path, default=P.OPERCULUM_MASKS_DIR)
    p.add_argument("--output-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()

    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    image_paths = load_paths(args.images_dir, IMAGE_EXTENSIONS)
    detections = load_polygon_detections(args.labels_dir)

    print("[1/4] Deciding the left/right alignment convention from the data")
    spots0, _, _ = build_tables(image_paths, detections, args.masks_dir, "none")
    align = side_alignment_test(spots0)
    for name in SIDE_TRANSFORMS:
        r = align[name]
        print(f"  {name:<8}: density r = {r['density_correlation']:+.4f}, "
              f"TV = {r['total_variation_distance']:.4f}, "
              f"mean x {r['mean_x_left']:.3f} vs {r['mean_x_right']:.3f}, "
              f"mean y {r['mean_y_left']:.3f} vs {r['mean_y_right']:.3f}")
    chosen = align["decision"]
    print(f"  -> chosen transform for the right side: {chosen}")

    print("\n[2/4] Building the canonical coordinate system")
    spots, images, occ = build_tables(image_paths, detections, args.masks_dir, chosen)
    spots.to_csv(out / "data" / "spatial2d_spotwise.csv", index=False)
    images.to_csv(out / "data" / "spatial2d_image_geometry.csv", index=False)
    np.save(out / "data" / "spatial2d_occupancy.npy", occ)
    print(f"  {len(spots):,} spots in {len(images)} images mapped to [0,1]^2")

    consist = orientation_consistency(images, spots)
    align["orientation"] = consist
    align["transform_applied_to_right_side"] = chosen
    (out / "data" / "spatial2d_alignment.json").write_text(json.dumps(align, indent=2),
                                                           encoding="utf-8")
    print(f"  operculum rotation in the source frame: "
          f"{consist['rotation_deg_mean']:.1f} deg (SD {consist['rotation_deg_sd']:.1f}, "
          f"range {consist['rotation_range_deg']:.1f})")
    print(f"  side agreement after alignment: x {consist['mean_x_left']:.3f} vs "
          f"{consist['mean_x_right']:.3f}; y {consist['mean_y_left']:.3f} vs "
          f"{consist['mean_y_right']:.3f}")

    print("\n[3/4] Two-dimensional models")
    models = fit_2d_models(spots)
    models.to_csv(out / "data" / "spatial2d_models.csv", index=False)
    print(models[["model", "term", "estimate", "ci_lower", "ci_upper", "p_value", "aic"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[4/4] Figures")
    figure_coordinate_system(
        image_paths, detections, args.masks_dir, spots, spots0, occ, align, chosen,
        out / "figures" / "fig_coordinate_system.png")
    figure_maps(spots, occ, models, out / "figures" / "fig_spatial_maps.png")
    figure_gradients(spots, out / "figures" / "fig_spatial_gradients.png")
    print("  figures written")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
