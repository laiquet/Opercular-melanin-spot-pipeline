#!/usr/bin/env python3
"""
Is the pre/post change measurable in an INDIVIDUAL fish?

The published pre/post comparison is a population-level result: averaged over
42 fish-side units, spots lightened. That does not establish that the change
can be measured in one animal, which is what welfare monitoring would require.

Each fish was imaged on both sides on both occasions, so every fish yields two
independent estimates of its own change. If a change is a real property of the
animal, the two sides must agree. If it is measurement noise, they will not.
That is the test this script runs, plus two follow-ups:

  1. Side agreement, raw and after removing the shared exposure component,
     since the two sides of a fish were photographed under similar conditions.
  2. Whether baseline pigmentation predicts the size of the response, using
     Oldham's method (change against the mean of pre and post) rather than
     change against baseline, which is biased negative by regression to the
     mean. Then whether any residual effect is simple proportionality.
  3. Whether any biometric predicts the change.

Run:
    python prepost_change_analysis.py

Datasets
--------
Reads   results/paper_2/data/oyvind_sugg/prepost_change_with_biometrics.csv
Writes  results/paper_2/data/oyvind_sugg/change_side_agreement.csv
        results/paper_2/data/oyvind_sugg/change_baseline_dependence.csv
        results/paper_2/figures/fig_change_reproducibility.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

from photometric_analysis import icc_2_1

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DESCRIPTORS = {
    "spot_count": "Spot count",
    "relative_spot_area_pct": "Relative spot area",
    "integrated_melanization": "Integrated melanization",
    "mean_rel_spot_area_pct": "Mean relative spot area",
    "skin_ref_log_ratio": "Skin-referenced intensity",
    "raw_mean_intensity": "Raw mean intensity",
}

FAMILY = {
    "spot_count": "geometric", "relative_spot_area_pct": "geometric",
    "mean_rel_spot_area_pct": "geometric",
    "integrated_melanization": "composite",
    "skin_ref_log_ratio": "photometric", "raw_mean_intensity": "photometric",
}

STYLE = {"geometric": ("0.28", None),
         "composite": ("0.58", "///"),
         "photometric": ("0.85", "\\\\")}


def residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residuals of y after removing the linear part explained by X."""
    X = np.column_stack([np.ones(len(y)), X])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=P.RESULTS_DIR / "data" / "oyvind_sugg")
    ap.add_argument("--fig-dir", type=Path, default=P.RESULTS_DIR / "figures")
    args = ap.parse_args()

    d = pd.read_csv(args.data_dir / "prepost_change_with_biometrics.csv")
    wide = d.pivot(index="FISH_ID", columns="SIDE",
                   values=[f"{c}_change" for c in DESCRIPTORS] + ["board_luma_change"])
    bL = wide["board_luma_change"]["L"].to_numpy()
    bR = wide["board_luma_change"]["R"].to_numpy()

    rows = []
    for col, label in DESCRIPTORS.items():
        s = wide[f"{col}_change"].dropna()
        L, R = s["L"].to_numpy(), s["R"].to_numpy()
        r_raw, p_raw = stats.pearsonr(L, R)
        rl = residualise(L, np.column_stack([bL, bR]))
        rr = residualise(R, np.column_stack([bL, bR]))
        r_par = float(np.corrcoef(rl, rr)[0, 1])
        df = len(L) - 4
        t = r_par * np.sqrt(df / (1 - r_par ** 2))
        p_par = float(2 * stats.t.sf(abs(t), df))
        icc = icc_2_1(np.column_stack([L, R]))
        rows.append({
            "descriptor": label, "column": col, "family": FAMILY[col],
            "n_fish": len(L),
            "r_left_right": r_raw, "p_left_right": p_raw,
            "r_partial_exposure": r_par, "p_partial_exposure": p_par,
            "icc_of_change": icc["icc"],
            "icc_lower": icc["lower"], "icc_upper": icc["upper"],
        })
    side = pd.DataFrame(rows).sort_values("icc_of_change", ascending=False)
    side.to_csv(args.data_dir / "change_side_agreement.csv", index=False)

    f = d.groupby("FISH_ID").mean(numeric_only=True)
    rows = []
    for col, label in DESCRIPTORS.items():
        pre, post = f[f"{col}_pre"], f[f"{col}_post"]
        mid, ch = (pre + post) / 2, f[f"{col}_change"]
        r_naive, p_naive = stats.pearsonr(pre, ch)
        r_old, p_old = stats.pearsonr(mid, ch)
        pct = f.get(f"{col}_pct_change")
        if pct is not None and pct.notna().sum() > 3:
            r_pct, p_pct = stats.pearsonr(mid[pct.notna()], pct.dropna())
            mean_pct = float(pct.mean())
        else:
            r_pct = p_pct = mean_pct = np.nan
        rows.append({
            "descriptor": label, "column": col,
            "r_change_vs_pre": r_naive, "p_change_vs_pre": p_naive,
            "r_oldham": r_old, "p_oldham": p_old,
            "r_pct_change_vs_baseline": r_pct, "p_pct_change_vs_baseline": p_pct,
            "mean_pct_change": mean_pct,
        })
    base = pd.DataFrame(rows)
    base.to_csv(args.data_dir / "change_baseline_dependence.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    y = np.arange(len(side))[::-1]
    bars = ax.barh(y, side["icc_of_change"],
                   color=[STYLE[f][0] for f in side["family"]],
                   edgecolor="black", linewidth=0.7)
    for bar, fam in zip(bars, side["family"]):
        bar.set_hatch(STYLE[fam][1])
    ax.errorbar(side["icc_of_change"], y,
                xerr=[np.clip(side["icc_of_change"] - side["icc_lower"], 0, None),
                      np.clip(side["icc_upper"] - side["icc_of_change"], 0, None)],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3)
    ax.axvline(0, color="0.55", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(side["descriptor"], fontsize=9)
    ax.set_xlabel("ICC(2,1) of the change, left side vs right side")
    ax.set_title("Is the change a property of the fish?\n21 fish, two sides each",
                 fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=g, hatch=h,
                             edgecolor="black", linewidth=0.6)
               for g, h in STYLE.values()]
    ax.legend(handles, [k.capitalize() for k in STYLE], loc="lower right", fontsize=8)
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    for ax, col in zip(axes[1:], ["relative_spot_area_pct", "skin_ref_log_ratio"]):
        s = wide[f"{col}_change"].dropna()
        L, R = s["L"], s["R"]
        r, p = stats.pearsonr(L, R)
        lim = [min(L.min(), R.min()), max(L.max(), R.max())]
        pad = 0.08 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, color="0.6", ls=":", lw=1.4, zorder=1,
                label="perfect agreement")
        ax.axhline(0, color="0.8", lw=0.8, zorder=1)
        ax.axvline(0, color="0.8", lw=0.8, zorder=1)
        ax.scatter(L, R, s=54, color=STYLE[FAMILY[col]][0],
                   edgecolors="black", linewidths=0.7, zorder=3)
        fit = stats.linregress(L, R)
        xs = np.linspace(*lim, 50)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=1.8, zorder=4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Left side, change in {DESCRIPTORS[col].lower()}")
        ax.set_ylabel(f"Right side, change in {DESCRIPTORS[col].lower()}")
        verdict = "sides agree" if p < 0.05 else "sides do not agree"
        ax.set_title(f"{DESCRIPTORS[col]}: {verdict}\nr = {r:.3f}, "
                     f"p = {p:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls="--", lw=0.5, alpha=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = args.fig_dir / "fig_change_reproducibility.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("SIDE AGREEMENT IN THE CHANGE")
    print(side[["descriptor", "r_left_right", "p_left_right",
                "r_partial_exposure", "p_partial_exposure",
                "icc_of_change"]].round(3).to_string(index=False))
    print("\nBASELINE DEPENDENCE")
    print(base[["descriptor", "r_change_vs_pre", "r_oldham", "p_oldham",
                "r_pct_change_vs_baseline", "p_pct_change_vs_baseline"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
