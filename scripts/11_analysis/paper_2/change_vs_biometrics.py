#!/usr/bin/env python3
"""
Does any biological covariate predict how much a fish changed?

The pre/post comparison establishes that the group changed. This asks the
next question: did some fish change more than others, and if so, can that be
predicted from body size, condition, sex, maturation or the visual fading
score?

Analysis is at fish level (n = 21), averaging the two sides, because the
covariates are recorded once per fish and the two sides of one animal are not
independent observations.

Two features distinguish this from a naive correlation sweep:

  1. Each response carries the reliability of the change itself, taken from
     the left-versus-right agreement (change_side_agreement.csv). A null
     result on a response whose change is not reliably measurable means
     "cannot be detected", not "no effect exists". Raw intensity change has an
     ICC of -0.04, so no covariate could correlate with it however strong the
     true relationship.

  2. The exposure proxy is offered as a covariate, because board luma change
     is itself weakly related to condition factor in this cohort (r = -0.378),
     so an association could in principle run through acquisition.

Run:
    python change_vs_biometrics.py

Datasets
--------
Reads   results/paper_2/data/oyvind_sugg/prepost_change_with_biometrics.csv
        results/paper_2/data/oyvind_sugg/change_side_agreement.csv
Writes  results/paper_2/data/oyvind_sugg/change_vs_biometrics.csv
        results/paper_2/figures/fig_change_vs_biometrics.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESPONSES = {
    "relative_spot_area_pct": "Relative spot area",
    "spot_count": "Spot count",
    "integrated_melanization": "Integrated melanization",
    "skin_ref_log_ratio": "Skin-referenced intensity",
}

PREDICTORS = {
    "condition_factor": ("Condition factor", "continuous"),
    "fork_length_cm": ("Fork length", "continuous"),
    "body_mass_g": ("Body mass", "continuous"),
    "sex": ("Sex", "categorical"),
    "maturation": ("Maturation", "categorical"),
}

VALIDATION = {"fading_score": ("Visual fading score", "continuous")}

SIG_GREY, NULL_GREY = "0.25", "0.72"


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for a correlation, via the Fisher z transform."""
    if n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return tuple(np.tanh([z - crit * se, z + crit * se]))


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Standardised mean difference (b - a) with its confidence interval."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (b.mean() - a.mean()) / sp
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=P.RESULTS_DIR / "data" / "oyvind_sugg")
    ap.add_argument("--fig-dir", type=Path, default=P.RESULTS_DIR / "figures")
    args = ap.parse_args()

    d = pd.read_csv(args.data_dir / "prepost_change_with_biometrics.csv")
    num = d.select_dtypes(include=[np.number]).columns.tolist()
    f = d.groupby("FISH_ID")[num].mean()
    f["sex"] = d.groupby("FISH_ID")["sex"].first()

    rel = pd.read_csv(args.data_dir / "change_side_agreement.csv")
    reliability = dict(zip(rel["column"], rel["icc_of_change"]))

    rows = []
    for family, preds in (("primary", PREDICTORS), ("validation", VALIDATION)):
        for pcol, (plabel, kind) in preds.items():
            for rcol, rlabel in RESPONSES.items():
                y = f[f"{rcol}_change"]
                sub = pd.concat([f[pcol], y], axis=1).dropna()
                if len(sub) < 6:
                    continue
                x, yy = sub[pcol], sub[f"{rcol}_change"]

                if kind == "continuous":
                    r, p = stats.pearsonr(x, yy)
                    lo, hi = fisher_ci(r, len(sub))
                    est, stat = r, "Pearson r"
                else:
                    groups = [g[f"{rcol}_change"].to_numpy() for _, g in sub.groupby(pcol)]
                    if len(groups) != 2 or min(len(g) for g in groups) < 3:
                        continue
                    _, p = stats.ttest_ind(*groups, equal_var=False)
                    est, lo, hi = hedges_g(*groups)
                    stat = "Hedges g"

                board = f.loc[sub.index, "board_luma_change"]
                if kind == "continuous":
                    res_y = yy - np.polyval(np.polyfit(board, yy, 1), board)
                    res_x = x - np.polyval(np.polyfit(board, x, 1), board)
                    r_adj = float(np.corrcoef(res_x, res_y)[0, 1])
                else:
                    r_adj = np.nan

                rows.append({
                    "family": family, "predictor": plabel, "response": rlabel,
                    "response_col": rcol, "kind": kind, "statistic": stat,
                    "n": len(sub), "estimate": est, "ci_lower": lo, "ci_upper": hi,
                    "p_value": p, "estimate_exposure_adjusted": r_adj,
                    "reliability_of_change": reliability.get(rcol, np.nan),
                })

    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    for fam, g in out.groupby("family"):
        out.loc[g.index, "q_value_bh"] = multipletests(g["p_value"], method="fdr_bh")[1]
    out["significant_q05"] = out["q_value_bh"] < 0.05
    out = out.sort_values(["family", "p_value"]).reset_index(drop=True)
    out.to_csv(args.data_dir / "change_vs_biometrics.csv", index=False)

    prim = out[out.family == "primary"].copy()
    prim["label"] = prim["predictor"] + "  →  " + prim["response"]
    prim = prim.sort_values("estimate").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.34 * len(prim) + 2.6),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    y = np.arange(len(prim))
    sig = prim["significant_q05"].to_numpy()
    for mask, face, ecol, lw in ((~sig, "white", NULL_GREY, 1.1),
                                 (sig, "black", "0.15", 1.6)):
        if not mask.any():
            continue
        ax.errorbar(prim["estimate"][mask], y[mask],
                    xerr=[np.clip(prim["estimate"] - prim["ci_lower"], 0, None)[mask],
                          np.clip(prim["ci_upper"] - prim["estimate"], 0, None)[mask]],
                    fmt="none", ecolor=ecol, elinewidth=lw, capsize=3, zorder=2)
        ax.scatter(prim["estimate"][mask], y[mask], s=58, facecolors=face,
                   edgecolors="black", linewidths=1.0, zorder=3,
                   label="Survives FDR (q < 0.05)" if mask is sig else "Not significant")
    ax.axvline(0, color="black", ls="--", lw=1.1)
    for s in (-0.58, 0.58):
        ax.axvline(s, color="0.6", ls=":", lw=1.3)
    ax.text(0.58, len(prim) - 0.4, " detection\n floor r = 0.58", fontsize=7.5,
            color="0.35", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(prim["label"], fontsize=8)
    ax.set_xlabel("Association with the change  (Pearson r, or Hedges g for sex and maturation)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    r_order = (prim.drop_duplicates("response")
               .sort_values("reliability_of_change", ascending=True))
    yy = np.arange(len(r_order))
    bars = ax.barh(yy, r_order["reliability_of_change"],
                   color=[SIG_GREY if v > 0.4 else NULL_GREY
                          for v in r_order["reliability_of_change"]],
                   edgecolor="black", linewidth=0.7)
    ax.axvline(0, color="0.55", lw=1.0)
    ax.axvline(0.4, color="black", ls=":", lw=1.2)
    ax.set_yticks(yy)
    ax.set_yticklabels(r_order["response"], fontsize=8)
    ax.set_xlabel("Reliability of the change\n(ICC, left side vs right side)")
    ax.set_title("Can the change even be measured\nin one fish?", fontsize=9.5)
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = args.fig_dir / "fig_change_vs_biometrics.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    cols = ["predictor", "response", "n", "estimate", "ci_lower", "ci_upper",
            "p_value", "q_value_bh", "estimate_exposure_adjusted",
            "reliability_of_change"]
    print("PRIMARY FAMILY  (%d tests, FDR corrected)" % (out.family == "primary").sum())
    print(out[out.family == "primary"][cols].round(3).to_string(index=False))
    print("\nVALIDATION FAMILY (visual fading score)")
    print(out[out.family == "validation"][cols].round(3).to_string(index=False))
    n_sig = int(out["significant_q05"].sum())
    print(f"\nsurviving FDR at q < 0.05: {n_sig} of {len(out)}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
