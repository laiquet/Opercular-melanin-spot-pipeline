#!/usr/bin/env python3
"""
Dedicated figure for integrated melanization (Results section).

Integrated melanization weights each spot pixel by how much darker it is than
its own local skin, so it is a different quantity from total opercular
melanization, which counts each spot pixel equally. This figure establishes
what integrated melanization is, how it relates to coverage, and why the two
are not interchangeable.

  A  Distribution across the 84 phenotyped images.
  B  Integrated melanization against coverage. The two are related but not
     identical; the scatter about the proportional line is the darkness
     weighting doing work.
  C  The weighting factor itself, the area-weighted mean skin-referenced
     darkening per image. Its narrow spread is why the two measures correlate
     strongly, and its non-zero spread is why they differ at all.

Run:
    python integrated_melanization_figure.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/image_level_measures.csv   (script 1)
Writes  results/paper_2/figures/fig_integrated_melanization.png
        results/paper_2/data/integrated_melanization_summary.json
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

INTEGRATED_GREY = "0.30"
COVERAGE_GREY = "0.78"
POINT_GREY = "0.42"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = ap.parse_args()

    d = pd.read_csv(args.results_dir / "data" / "image_level_measures.csv")
    d["coverage"] = 100.0 * d["spot_px"] / d["tissue_px"]
    d["darkening"] = d["total_melanization"] / (d["coverage"] / 100.0)

    M, C, B = d["total_melanization"], d["coverage"], d["darkening"]
    r = float(stats.pearsonr(C, M)[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.hist(M, bins=18, color=INTEGRATED_GREY, edgecolor="black", linewidth=0.7)
    ax.axvline(M.median(), color="black", ls="--", lw=1.3,
               label=f"median {M.median():.4f}")
    ax.set_xlabel("Integrated melanization")
    ax.set_ylabel("Images")
    ax.set_title(f"Distribution across images (n = {len(d)})\n"
                 f"mean {M.mean():.4f}, range {M.min():.4f}–{M.max():.4f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.scatter(C, M, s=50, color=POINT_GREY, edgecolors="black",
               linewidths=0.6, alpha=0.9, zorder=3)
    xs = np.linspace(C.min(), C.max(), 100)
    fit = stats.linregress(C, M)
    ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=1.8, zorder=4,
            label="least-squares fit")
    ax.plot(xs, B.mean() * xs / 100.0, color="black", ls=":", lw=1.6, zorder=4,
            label=f"constant darkening ({B.mean():.3f})")
    ax.set_xlabel("Total opercular melanization (% coverage)")
    ax.set_ylabel("Integrated melanization")
    ax.set_title(f"Related but not interchangeable\n"
                 f"r = {r:.3f}, R² = {r ** 2:.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.hist(B, bins=18, color=COVERAGE_GREY, edgecolor="black", linewidth=0.7,
            hatch="///")
    ax.axvline(B.mean(), color="black", ls="--", lw=1.3,
               label=f"mean {B.mean():.3f}")
    ax.set_xlabel("Mean skin-referenced darkening")
    ax.set_ylabel("Images")
    ax.set_title(f"The weighting factor\n"
                 f"CV {100 * B.std() / B.mean():.1f}%, "
                 f"range {B.min():.3f}–{B.max():.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = args.results_dir / "figures" / "fig_integrated_melanization.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_images": int(len(d)),
        "integrated_mean": float(M.mean()), "integrated_sd": float(M.std()),
        "integrated_median": float(M.median()),
        "integrated_min": float(M.min()), "integrated_max": float(M.max()),
        "integrated_q1": float(M.quantile(0.25)), "integrated_q3": float(M.quantile(0.75)),
        "integrated_cv_pct": float(100 * M.std() / M.mean()),
        "integrated_fold_range": float(M.max() / M.min()),
        "coverage_mean_pct": float(C.mean()),
        "coverage_min_pct": float(C.min()), "coverage_max_pct": float(C.max()),
        "darkening_mean": float(B.mean()), "darkening_sd": float(B.std()),
        "darkening_cv_pct": float(100 * B.std() / B.mean()),
        "darkening_min": float(B.min()), "darkening_max": float(B.max()),
        "pearson_r_coverage": r, "r_squared_coverage": r ** 2,
    }
    (args.results_dir / "data" / "integrated_melanization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
