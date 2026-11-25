#!/usr/bin/env python3
"""
Grayscale feature-extraction sensitivity (referee comment 4).

The manuscript reports "grayscale intensity" without stating which statistic of
the within-mask pixel distribution was used. It was the arithmetic mean. This
script quantifies what that choice costs:

  1. Mean versus median agreement across all phenotyped spots.
  2. The manuscript's intensity-distribution figure reproduced with the median,
     as a supplementary sensitivity analysis.
  3. Whether the mean-median gap is an edge-pixel effect, cross-checked against
     the erosion experiment from comment 3.
  4. Whether any downstream conclusion (repeatability, pre/post confinement)
     changes if the median is used instead.

Run after photometric_analysis.py (and, for part 3, segmentation_validation.py):
    python grayscale_estimator_check.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/spot_level_skin_referenced.csv  (script 1)
        results/paper_2/data/image_level_measures.csv        (script 1)
        results/paper_2/data/erosion_spotwise.csv            (script 3,
                          optional; enables the edge-pixel cross-check)
Writes  results/paper_2/data/grayscale_mean_vs_median.json
        results/paper_2/data/grayscale_mean_vs_median_by_size.csv
        results/paper_2/data/grayscale_distribution_mean_vs_median.csv
        results/paper_2/data/grayscale_erosion_crosscheck.json
        results/paper_2/data/grayscale_statistic_downstream.csv
        results/paper_2/figures/fig_mean_vs_median.png
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
from scipy import stats

from photometric_analysis import icc_2_1, occasion_matrix


BIN_EDGES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
BIN_LABELS = ["[0.0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]",
              "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]", "(0.8,0.9]", "(0.9,1.0]"]

AREA_BIN_EDGES = [0, 250, 500, 1000, 2000, np.inf]
AREA_BIN_LABELS = ["<250", "250-500", "500-1000", "1000-2000", ">2000"]

MEAN_GREY = "0.30"
MEDIAN_GREY = "0.78"
POINT_GREY = "0.45"
LIMIT_GREY = "0.25"


def bin_counts(values: np.ndarray) -> np.ndarray:
    ids = np.digitize(values, BIN_EDGES, right=True)
    return np.array([int((ids == i).sum()) for i in range(10)])


def agreement(spots: pd.DataFrame) -> dict:
    """Mean-versus-median agreement, spot by spot."""
    m, md = spots["spot_mean"].to_numpy(float), spots["spot_median"].to_numpy(float)
    diff = m - md
    r, p = stats.pearsonr(m, md)
    rho, p_rho = stats.spearmanr(m, md)
    return {
        "n": int(len(spots)),
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "mean_of_mean": float(m.mean()), "mean_of_median": float(md.mean()),
        "bias_mean_minus_median": float(diff.mean()),
        "sd_of_difference": float(diff.std(ddof=1)),
        "loa_lower": float(diff.mean() - 1.96 * diff.std(ddof=1)),
        "loa_upper": float(diff.mean() + 1.96 * diff.std(ddof=1)),
        "pct_spots_mean_exceeds_median": float(100.0 * (diff > 0).mean()),
        "median_abs_difference": float(np.median(np.abs(diff))),
    }


def agreement_by_size(spots: pd.DataFrame) -> pd.DataFrame:
    df = spots.copy()
    df["size_bin"] = pd.cut(df["area_px"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS, right=False)
    df["diff"] = df["spot_mean"] - df["spot_median"]
    return (df.groupby("size_bin", observed=True)
            .agg(n=("diff", "size"),
                 mean_minus_median=("diff", "mean"),
                 sd=("diff", "std"),
                 pct_of_mean=("diff", lambda s: 100.0 * s.mean() / df.loc[s.index, "spot_mean"].mean()))
            .reset_index())


def distribution_table(spots: pd.DataFrame) -> pd.DataFrame:
    """The manuscript's Figure 11 counts, computed both ways."""
    cm = bin_counts(spots["spot_mean"].to_numpy(float))
    cd = bin_counts(spots["spot_median"].to_numpy(float))
    n = len(spots)
    return pd.DataFrame({
        "bin": BIN_LABELS,
        "n_mean": cm, "pct_mean": 100.0 * cm / n,
        "n_median": cd, "pct_median": 100.0 * cd / n,
        "pct_difference": 100.0 * cd / n - 100.0 * cm / n,
    })


def erosion_cross_check(results_dir: Path, spots: pd.DataFrame) -> dict | None:
    """
    Is the mean-median gap an edge effect?

    If the outermost mask pixels are the ones inflating the mean, then eroding
    the mask should move the mean toward the un-eroded median, and should move
    the median far less. Both predictions are tested here.
    """
    path = results_dir / "data" / "erosion_spotwise.csv"
    if not path.exists():
        return None
    ero = pd.read_csv(path)
    if "spot_median" not in ero.columns:
        return None

    wide = ero[~ero["lost"]].pivot_table(
        index=["sample", "spot"], columns="erosion_px",
        values=["spot_mean", "spot_median"])
    out: dict = {}
    for px in [1, 2, 3]:
        if ("spot_mean", px) not in wide.columns:
            continue
        out[f"mean_shift_{px}px"] = float((wide[("spot_mean", px)] - wide[("spot_mean", 0)]).mean())
        out[f"median_shift_{px}px"] = float((wide[("spot_median", px)] - wide[("spot_median", 0)]).mean())
    out["mean_minus_median_at_0px"] = float((wide[("spot_mean", 0)] - wide[("spot_median", 0)]).mean())
    if ("spot_mean", 1) in wide.columns:
        gap0 = wide[("spot_mean", 0)] - wide[("spot_median", 0)]
        gap1 = wide[("spot_mean", 1)] - wide[("spot_median", 1)]
        out["gap_at_0px"] = float(gap0.mean())
        out["gap_at_1px"] = float(gap1.mean())
        out["gap_closed_pct"] = float(100.0 * (gap0.mean() - gap1.mean()) / gap0.mean())
        out["stability_ratio"] = float(
            abs((wide[("spot_mean", 1)] - wide[("spot_mean", 0)]).mean())
            / abs((wide[("spot_median", 1)] - wide[("spot_median", 0)]).mean()))
    return out


def downstream(spots: pd.DataFrame, per_image: pd.DataFrame) -> pd.DataFrame:
    """Repeatability and the pre/post effect, computed with each statistic."""
    agg = (spots.groupby("sample", as_index=False)
           .agg(mean_stat=("spot_mean", "mean"), median_stat=("spot_median", "mean")))
    img = per_image[["sample", "FISH_ID", "DAY", "SIDE"]].astype(str).merge(agg, on="sample")
    rows: list[dict] = []
    for col, label in [("mean_stat", "Arithmetic mean (as published)"),
                       ("median_stat", "Median (sensitivity analysis)")]:
        mat, _ = occasion_matrix(img, col)
        res = icc_2_1(mat)
        d = mat[:, 1] - mat[:, 0]
        t, p = stats.ttest_rel(mat[:, 1], mat[:, 0])
        rows.append({
            "statistic": label, "n_units": res["n"],
            "ICC(2,1)": res["icc"], "CI_lower": res["lower"], "CI_upper": res["upper"],
            "pre_mean": float(mat[:, 0].mean()), "post_mean": float(mat[:, 1].mean()),
            "prepost_change": float(d.mean()), "prepost_p": float(p),
            "prepost_dz": float(d.mean() / d.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def figure_distribution(table: pd.DataFrame, agree: dict, spots: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    x = np.arange(len(BIN_LABELS))

    ax = axes[0]
    ax.bar(x - 0.2, table["pct_mean"], width=0.4, label="mean",
           color=MEAN_GREY, edgecolor="black", linewidth=0.7)
    ax.bar(x + 0.2, table["pct_median"], width=0.4, label="median",
           color=MEDIAN_GREY, edgecolor="black", linewidth=0.7, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Grayscale intensity bin")
    ax.set_ylabel("% of spots")
    ax.set_title("Intensity distribution, both statistics", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)

    ax = axes[1]
    ax.scatter(spots["spot_mean"], spots["spot_median"], s=7, alpha=0.25,
               color=POINT_GREY, edgecolors="none")
    lim = (0, max(spots["spot_mean"].max(), spots["spot_median"].max()) * 1.05)
    ax.plot(lim, lim, ls="--", color="black", lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Mean intensity")
    ax.set_ylabel("Median intensity")
    ax.set_title(f"Mean vs median, all {agree['n']:,} spots\n"
                 f"r = {agree['pearson_r']:.4f}", fontsize=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)

    ax = axes[2]
    avg = (spots["spot_mean"] + spots["spot_median"]) / 2
    diff = spots["spot_mean"] - spots["spot_median"]
    ax.scatter(avg, diff, s=7, alpha=0.25, color=POINT_GREY, edgecolors="none")
    ax.axhline(agree["bias_mean_minus_median"], color="black", lw=1.8,
               label=f"bias {agree['bias_mean_minus_median']:+.4f}")
    ax.axhline(agree["loa_upper"], color=LIMIT_GREY, ls="--", lw=1.4, label="95% limits")
    ax.axhline(agree["loa_lower"], color=LIMIT_GREY, ls="--", lw=1.4)
    ax.axhline(0, color=POINT_GREY, lw=0.9, ls=":")
    ax.set_xlabel("(mean + median) / 2")
    ax.set_ylabel("mean − median")
    ax.set_title("Bland–Altman agreement", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()
    d = args.results_dir / "data"

    spots = pd.read_csv(d / "spot_level_skin_referenced.csv")
    per_image = pd.read_csv(d / "image_level_measures.csv",
                            dtype={"FISH_ID": str, "DAY": str, "SIDE": str})

    print("=" * 74)
    print("Grayscale statistic sensitivity (comment 4)")
    print("=" * 74)

    agree = agreement(spots)
    (d / "grayscale_mean_vs_median.json").write_text(json.dumps(agree, indent=2), encoding="utf-8")
    print(f"n = {agree['n']:,} spots")
    print(f"  Pearson r  = {agree['pearson_r']:.4f}")
    print(f"  Spearman   = {agree['spearman_rho']:.4f}")
    print(f"  mean(mean) = {agree['mean_of_mean']:.4f},  mean(median) = {agree['mean_of_median']:.4f}")
    print(f"  bias (mean - median) = {agree['bias_mean_minus_median']:+.4f} "
          f"[95% LoA {agree['loa_lower']:+.4f}, {agree['loa_upper']:+.4f}]")
    print(f"  mean exceeds median in {agree['pct_spots_mean_exceeds_median']:.1f}% of spots")

    by_size = agreement_by_size(spots)
    by_size.to_csv(d / "grayscale_mean_vs_median_by_size.csv", index=False)
    print()
    print(by_size.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    table = distribution_table(spots)
    table.to_csv(d / "grayscale_distribution_mean_vs_median.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    cross = erosion_cross_check(args.results_dir, spots)
    if cross:
        (d / "grayscale_erosion_crosscheck.json").write_text(json.dumps(cross, indent=2),
                                                             encoding="utf-8")
        print()
        print("Erosion cross-check:")
        for k, v in cross.items():
            print(f"  {k:<28} {v:+.5f}" if isinstance(v, float) else f"  {k:<28} {v}")
    else:
        print("\n  (erosion cross-check skipped: run segmentation_validation.py first)")

    down = downstream(spots, per_image)
    down.to_csv(d / "grayscale_statistic_downstream.csv", index=False)
    print()
    print(down.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    figure_distribution(table, agree, spots,
                        args.results_dir / "figures" / "fig_mean_vs_median.png")
    print(f"\nWrote outputs to {d}")


if __name__ == "__main__":
    main()
