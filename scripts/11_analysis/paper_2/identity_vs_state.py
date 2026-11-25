#!/usr/bin/env python3
"""
Stable versus dynamic components of the opercular phenotype (referee comment 9).

The referee raises a conceptual tension: opercular spot patterns have been used
for individual re-identification, which requires stability, while this framework
proposes them as candidate welfare indicators, which requires responsiveness.
The two imaging occasions let the question be answered with data rather than
argued.

  1. A stability table: ICC(2,1) and within-fish variability for every
     descriptor, including the spatial-configuration measures that
     re-identification actually relies on.
  2. A re-identification experiment: can a fish-side be matched across the two
     occasions from its spot configuration alone, and can it be matched from
     intensity alone? If configuration identifies and intensity does not, the
     tension dissolves: they are separable properties of one phenotype.

Inputs, all from this repository (via earlier stages):
  results/paper_2/data/image_level_measures.csv
  results/paper_2/data/spatial2d_spotwise.csv
  results/paper_2/data/spatial_operculum_geometry.csv

Run:
    python identity_vs_state.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/image_level_measures.csv        (script 1)
        results/paper_2/data/spatial2d_spotwise.csv          (script 6)
        results/paper_2/data/spatial_operculum_geometry.csv  (script 5)
Writes  results/paper_2/data/identity_image_level.csv
        results/paper_2/data/identity_stability_table.csv
        results/paper_2/data/identity_reidentification.csv
        results/paper_2/data/identity_reidentification_perquery.csv
        results/paper_2/figures/fig_identity_stability.png
        results/paper_2/figures/fig_reidentification.png
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from photometric_analysis import icc_2_1, occasion_matrix


PRE, POST = "1", "2"

CUE_FAMILIES = [
    ("Spot configuration (spatial)", "0.25", None),
    ("Number and area", "0.55", "///"),
    ("Intensity and melanization", "0.85", "\\\\"),
]

COMPONENT_STYLE = {
    "architecture": ("0.28", None),
    "composite": ("0.58", "///"),
    "state": ("0.85", "\\\\"),
}

DESCRIPTOR_CLASS = {
    "spot_count": ("Spot count", "architecture", True),
    "relative_spot_area_pct": ("Relative spot area (%)", "architecture", True),
    "pattern_centroid_x": ("Pattern centroid, A–P", "architecture", False),
    "pattern_centroid_y": ("Pattern centroid, D–V", "architecture", False),
    "pattern_dispersion": ("Pattern dispersion", "architecture", True),
    "mean_nn_distance": ("Mean nearest-neighbour distance", "architecture", True),
    "mean_rel_spot_area_pct": ("Mean relative spot area (%)", "architecture", True),
    "total_melanization": ("Integrated melanization", "composite", True),
    "raw_mean_intensity": ("Raw mean spot intensity", "state", True),
    "skin_ref_log_ratio_linear": ("Skin-referenced intensity", "state", False),
    "image_skin_median": ("Opercular skin intensity", "state", True),
}


def configuration_features(spots: pd.DataFrame) -> pd.DataFrame:
    """Per-image summaries of the spatial arrangement of spot centroids."""
    rows = []
    for sample, g in spots.groupby("sample"):
        pts = g[["x", "y"]].to_numpy()
        if len(pts) < 2:
            continue
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        rows.append({
            "sample": sample,
            "pattern_centroid_x": float(pts[:, 0].mean()),
            "pattern_centroid_y": float(pts[:, 1].mean()),
            "pattern_dispersion": float(np.sqrt(pts[:, 0].var() + pts[:, 1].var())),
            "mean_nn_distance": float(d[:, 1].mean()),
        })
    return pd.DataFrame(rows)


def stability_table(images: pd.DataFrame) -> pd.DataFrame:
    """ICC(2,1) across occasions plus within-fish variability, per descriptor."""
    rows = []
    for col, (label, kind, cv_ok) in DESCRIPTOR_CLASS.items():
        if col not in images.columns:
            continue
        matrix, _ = occasion_matrix(images, col)
        if len(matrix) < 3:
            continue
        res = icc_2_1(matrix)
        diffs = matrix[:, 1] - matrix[:, 0]
        means = matrix.mean(axis=1)
        within_sd = float(np.abs(diffs).mean() / np.sqrt(2) * np.sqrt(np.pi / 2))
        within_sd = float(np.sqrt((diffs ** 2).mean() / 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = float(np.mean(np.abs(diffs) / np.sqrt(2) / np.abs(means)) * 100) if cv_ok else np.nan
        rows.append({
            "descriptor": label, "column": col, "component": kind,
            "n_units": res["n"],
            "ICC(2,1)": res["icc"], "CI_lower": res["lower"], "CI_upper": res["upper"],
            "within_unit_sd": within_sd,
            "within_unit_cv_pct": cv,
            "between_unit_sd": float(matrix.mean(axis=1).std(ddof=1)),
            "occasion_shift": float(diffs.mean()),
        })
    out = pd.DataFrame(rows)
    out["variance_between_pct"] = 100 * out["ICC(2,1)"].clip(lower=0)
    return out.sort_values("ICC(2,1)", ascending=False)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return np.inf
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, _ = tb.query(a, k=1)
    d_ba, _ = ta.query(b, k=1)
    return float(d_ab.mean() + d_ba.mean())


def assignment_cost(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean matched distance under an optimal one-to-one assignment.

    Stricter than Chamfer: it forbids many-to-one matches, so a dense pattern
    cannot cheaply cover a sparse one.
    """
    if len(a) == 0 or len(b) == 0:
        return np.inf
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    r, c = linear_sum_assignment(d)
    return float(d[r, c].mean())


def reidentify(spots: pd.DataFrame, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match each occasion-2 fish-side to an occasion-1 fish-side.

    Every occasion-2 unit is scored against all occasion-1 units, using each cue
    in turn. Top-1 accuracy is compared with the chance rate of 1/n. A cue that
    identifies individuals carries stable information; a cue that does not is
    free to carry state.
    """
    spots = spots.copy()
    spots["unit"] = spots["FISH_ID"].astype(str) + "-" + spots["SIDE"].astype(str)
    images = images.copy()
    images["unit"] = images["FISH_ID"].astype(str) + "-" + images["SIDE"].astype(str)

    day = {}
    for (unit, d), g in spots.groupby(["unit", "DAY"]):
        day[(unit, str(d))] = g[["x", "y"]].to_numpy()
    meta = {(r["unit"], str(r["DAY"])): r for _, r in images.iterrows()}

    units = sorted({u for (u, _) in day} & {u for (u, d) in day if d == POST})
    units = [u for u in units if (u, PRE) in day and (u, POST) in day]
    n = len(units)

    cues = {
        "Spot configuration (assignment)": lambda q, r: assignment_cost(day[(q, POST)], day[(r, PRE)]),
        "Spot configuration (Chamfer)": lambda q, r: chamfer(day[(q, POST)], day[(r, PRE)]),
        "Spot count": lambda q, r: abs(meta[(q, POST)]["spot_count"] - meta[(r, PRE)]["spot_count"]),
        "Relative spot area": lambda q, r: abs(meta[(q, POST)]["rel_spot_area_pct"]
                                               - meta[(r, PRE)]["rel_spot_area_pct"]),
        "Skin-referenced intensity": lambda q, r: abs(
            meta[(q, POST)]["skin_ref_log_ratio_linear"] - meta[(r, PRE)]["skin_ref_log_ratio_linear"]),
        "Raw mean intensity": lambda q, r: abs(meta[(q, POST)]["raw_mean_intensity"]
                                               - meta[(r, PRE)]["raw_mean_intensity"]),
        "Integrated melanization": lambda q, r: abs(meta[(q, POST)]["total_melanization"]
                                               - meta[(r, PRE)]["total_melanization"]),
    }

    rows, per_query = [], []
    for cue, fn in cues.items():
        ranks = []
        for q in units:
            scores = np.array([fn(q, r) for r in units], dtype=float)
            order = np.argsort(scores, kind="stable")
            rank = int(np.where(np.array(units)[order] == q)[0][0]) + 1
            ranks.append(rank)
            per_query.append({"cue": cue, "unit": q, "rank": rank})
        ranks = np.array(ranks)
        rows.append({
            "cue": cue, "n_units": n,
            "top1_accuracy_pct": float(100 * (ranks == 1).mean()),
            "top5_accuracy_pct": float(100 * (ranks <= 5).mean()),
            "median_rank": float(np.median(ranks)),
            "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
            "chance_top1_pct": 100.0 / n,
            "chance_mrr": float(np.mean(1.0 / np.arange(1, n + 1))),
        })
    return pd.DataFrame(rows), pd.DataFrame(per_query)


def figure_stability(table: pd.DataFrame, out_path: Path) -> None:
    """
    ICC(2,1) per descriptor, in grayscale.

    The three components are the point of the figure, so they are separated by
    tone and by hatching together rather than by tone alone: architecture dark
    and solid, composite mid and hatched one way, state light and hatched the
    other. Tone runs dark to light down the stability gradient, so the visual
    weight matches the result.
    """
    df = table.sort_values("ICC(2,1)")
    fig, ax = plt.subplots(figsize=(9.5, 0.44 * len(df) + 1.8))
    y = np.arange(len(df))
    bars = ax.barh(y, df["ICC(2,1)"],
                   color=[COMPONENT_STYLE[c][0] for c in df["component"]],
                   edgecolor="black", linewidth=0.6)
    for bar, comp in zip(bars, df["component"]):
        bar.set_hatch(COMPONENT_STYLE[comp][1])
    ax.errorbar(df["ICC(2,1)"], y,
                xerr=[np.clip(df["ICC(2,1)"] - df["CI_lower"], 0, None),
                      np.clip(df["CI_upper"] - df["ICC(2,1)"], 0, None)],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["descriptor"], fontsize=9)
    ax.set_xlabel("ICC(2,1) across imaging sessions")
    ax.set_xlim(-0.15, 1.02)
    ax.axvline(0, color="0.55", lw=1.0)
    ax.axvline(0.75, color="black", ls="--", lw=1.0)
    ax.text(0.755, -0.6, "0.75", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=grey, hatch=hatch,
                             edgecolor="black", linewidth=0.6)
               for grey, hatch in COMPONENT_STYLE.values()]
    ax.legend(handles, [k.capitalize() for k in COMPONENT_STYLE],
              loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def cue_family(cue: str) -> int:
    """Index into CUE_FAMILIES for a given re-identification cue."""
    if "configuration" in cue:
        return 0
    if cue in ("Spot count", "Relative spot area"):
        return 1
    return 2


def figure_reid(reid: pd.DataFrame, out_path: Path) -> None:
    """
    Re-identification accuracy by cue, in grayscale.

    The three cue families are ordered by how identity-bearing they turn out to
    be, so tone is used ordinally: spatial configuration darkest, intensity
    lightest. Hatching repeats the same grouping, so the families stay separable
    in monochrome print where adjacent grey levels can merge.
    """
    df = reid.sort_values("top1_accuracy_pct", ascending=True)
    fams = [cue_family(c) for c in df["cue"]]
    colours = [CUE_FAMILIES[f][1] for f in fams]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    def draw(ax, values, xlabel, title, chance, chance_label):
        bars = ax.barh(np.arange(len(df)), values, color=colours,
                       edgecolor="black", linewidth=0.6)
        for bar, f in zip(bars, fams):
            bar.set_hatch(CUE_FAMILIES[f][2])
        ax.axvline(chance, color="black", ls="--", lw=1.4, label=chance_label)
        ax.set_yticks(np.arange(len(df)))
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=grey, hatch=hatch,
                                 edgecolor="black", linewidth=0.6)
                   for _, grey, hatch in CUE_FAMILIES]
        labels = [name for name, _, _ in CUE_FAMILIES]
        chance_handle, chance_lab = ax.get_legend_handles_labels()
        ax.legend(handles + chance_handle, labels + chance_lab,
                  fontsize=8, loc="lower right", framealpha=0.95)

    draw(axes[0], df["top1_accuracy_pct"], "Top-1 re-identification accuracy (%)",
         f"Matching session 2 (post-stress) to session 1 (pre-stress)\n"
         f"{int(df['n_units'].iloc[0])} fish-side units",
         df["chance_top1_pct"].iloc[0],
         f"chance ({df['chance_top1_pct'].iloc[0]:.1f}%)")
    axes[0].set_yticklabels(df["cue"], fontsize=9)

    draw(axes[1], df["mean_reciprocal_rank"], "Mean reciprocal rank",
         "Rank of the correct match", df["chance_mrr"].iloc[0], "chance")
    axes[1].set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()
    d = args.results_dir / "data"

    images = pd.read_csv(d / "image_level_measures.csv",
                         dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    geom = pd.read_csv(d / "spatial_operculum_geometry.csv", dtype={"DAY": str})
    images = images.merge(geom[["sample", "rel_spot_area_pct"]], on="sample", how="left")
    images["mean_rel_spot_area_pct"] = (images["relative_spot_area_pct"]
                                        / images["spot_count"])

    spots = pd.read_csv(d / "spatial2d_spotwise.csv",
                        dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    config = configuration_features(spots)
    images = images.merge(config, on="sample", how="left")
    images.to_csv(d / "identity_image_level.csv", index=False)

    print("=" * 78)
    print("[1/2] Stability of each descriptor across the two imaging occasions")
    print("=" * 78)
    table = stability_table(images)
    table.to_csv(d / "identity_stability_table.csv", index=False)
    print(table[["descriptor", "component", "n_units", "ICC(2,1)", "CI_lower", "CI_upper",
                 "within_unit_cv_pct"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("[2/2] Re-identification across occasions")
    print("=" * 78)
    reid, per_query = reidentify(spots, images)
    reid.to_csv(d / "identity_reidentification.csv", index=False)
    per_query.to_csv(d / "identity_reidentification_perquery.csv", index=False)
    print(reid[["cue", "n_units", "top1_accuracy_pct", "top5_accuracy_pct",
                "median_rank", "mean_reciprocal_rank", "chance_top1_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    figure_stability(table, args.results_dir / "figures" / "fig_identity_stability.png")
    figure_reid(reid, args.results_dir / "figures" / "fig_reidentification.png")
    print("\nFigures written.")


if __name__ == "__main__":
    main()
