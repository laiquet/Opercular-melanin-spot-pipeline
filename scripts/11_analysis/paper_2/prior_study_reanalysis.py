#!/usr/bin/env python3
"""
Re-analysis of the pre/post-confinement grayscale comparison of Laique et al.
(2026), using the exposure-robust skin-referenced descriptor.

The 84 measurement images are the same images the earlier study analysed:
21 fish, each photographed on both sides before (day 1) and after (day 2) an
18-hour confinement stress test. The earlier study reported a significant
INCREASE in spot grayscale intensity after confinement, i.e. a shift toward
lighter pigmentation.

Referee comment 2 asks whether that effect could be an artefact of inconsistent
exposure between the two imaging sessions. This script tests that directly:

  1. Does the independent exposure proxy (blue background board) differ
     systematically between the pre- and post-confinement sessions? If it does
     not, a session-level exposure shift cannot explain a session-level
     intensity shift.
  2. Does the reported effect replicate on the raw descriptor?
  3. Does it survive when the exposure-robust skin-referenced descriptor is
     used instead?
  4. Does it survive when the exposure proxy is entered as a covariate?

Run after photometric_analysis.py:
    python prior_study_reanalysis.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/image_level_measures.csv        (script 1)
        results/paper_2/data/image_level_with_exposure.csv   (script 1)
        results/paper_2/data/background_board_exposure.csv   (script 1)
Writes  results/paper_2/data/prepost_confinement_reanalysis.csv
        results/paper_2/data/prepost_exposure_proxy.json
        results/paper_2/data/prepost_exposure_adjusted.csv
        results/paper_2/data/session_constancy_audit.csv
        results/paper_2/figures/fig_prepost_confinement.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

from photometric_analysis import format_p


import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


PRE, POST = "1", "2"

PRE_GREY = "0.35"
LINK_GREY = "0.65"

MEASURES = {
    "raw_mean_intensity": "Raw mean spot intensity",
    "skin_ref_log_ratio_linear": "Skin-referenced log-ratio (linearised)",
    "skin_ref_ratio": "Skin-referenced ratio (gamma-encoded)",
    "image_skin_median": "Median opercular skin intensity",
    "relative_spot_area_pct": "Relative spot area (%)",
    "spot_count": "Spot count",
    "total_melanization": "Integrated melanization",
}

PRIOR_REPORTED = {
    "pre_left": 0.203, "pre_right": 0.195,
    "post_left": 0.262, "post_right": 0.259,
    "p": 0.0003,
}


def paired_units(per_image: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per fish x side unit, with the pre and post values side by side."""
    wide = per_image.pivot_table(
        index=["FISH_ID", "SIDE"], columns="DAY", values=column, aggfunc="mean"
    )
    wide.columns = [str(c) for c in wide.columns]
    wide = wide[[PRE, POST]].dropna().reset_index()
    wide = wide.rename(columns={PRE: "pre", POST: "post"})
    wide["diff"] = wide["post"] - wide["pre"]
    return wide


def paired_test(per_image: pd.DataFrame, column: str, n_boot: int = 20000) -> dict:
    """
    Paired comparison of pre- versus post-confinement, on fish x side units.

    Reported alongside a mixed model that respects the nesting of the two sides
    within fish; the two agree closely because the design is balanced.
    """
    w = paired_units(per_image, column)
    d = w["diff"].to_numpy(float)
    n = len(d)
    t, p = stats.ttest_rel(w["post"], w["pre"])
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)

    rng = np.random.default_rng(20260821)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)

    long = per_image[["FISH_ID", "SIDE", "DAY", column]].dropna().copy()
    long["post"] = (long["DAY"].astype(str) == POST).astype(int)
    long = long.rename(columns={column: "y"})
    try:
        fit = smf.mixedlm("y ~ post + C(SIDE)", long, groups=long["FISH_ID"]).fit(reml=True)
        mm_beta = float(fit.params["post"])
        mm_p = float(fit.pvalues["post"])
        mm_lo, mm_hi = (float(v) for v in fit.conf_int().loc["post"])
    except Exception:
        mm_beta = mm_p = mm_lo = mm_hi = np.nan

    return {
        "measure": MEASURES.get(column, column),
        "column": column,
        "n_units": n,
        "pre_mean": float(w["pre"].mean()),
        "post_mean": float(w["post"].mean()),
        "mean_change": float(d.mean()),
        "pct_change": 100.0 * d.mean() / abs(w["pre"].mean()) if w["pre"].mean() else np.nan,
        "ci_lower": float(d.mean() - crit * se),
        "ci_upper": float(d.mean() + crit * se),
        "boot_ci_lower": float(np.percentile(boot, 2.5)),
        "boot_ci_upper": float(np.percentile(boot, 97.5)),
        "t": float(t),
        "p_value": float(p),
        "cohens_dz": float(dz),
        "mixed_beta": mm_beta,
        "mixed_ci_lower": mm_lo,
        "mixed_ci_upper": mm_hi,
        "mixed_p": mm_p,
    }


def exposure_covariate_models(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Does the pre/post effect survive adjustment for per-image exposure?

    Fits y ~ post + side + board_luma with fish as a random intercept, on the
    subset of images for which the original frame (and therefore the board) is
    available.
    """
    rows: list[dict] = []
    for column, label in MEASURES.items():
        long = merged[["FISH_ID", "SIDE", "DAY", "board_luma", column]].dropna().copy()
        if long.empty or long["DAY"].nunique() < 2:
            continue
        long["post"] = (long["DAY"].astype(str) == POST).astype(int)
        long = long.rename(columns={column: "y"})
        try:
            base = smf.mixedlm("y ~ post + C(SIDE)", long, groups=long["FISH_ID"]).fit(reml=True)
            adj = smf.mixedlm(
                "y ~ post + C(SIDE) + board_luma", long, groups=long["FISH_ID"]
            ).fit(reml=True)
        except Exception:
            continue
        rows.append({
            "measure": label,
            "n": len(long),
            "beta_unadjusted": float(base.params["post"]),
            "p_unadjusted": float(base.pvalues["post"]),
            "beta_adjusted": float(adj.params["post"]),
            "p_adjusted": float(adj.pvalues["post"]),
            "beta_shift_pct": 100.0 * (adj.params["post"] - base.params["post"]) / abs(base.params["post"])
            if base.params["post"] != 0 else np.nan,
            "board_luma_beta": float(adj.params["board_luma"]),
            "board_luma_p": float(adj.pvalues["board_luma"]),
        })
    return pd.DataFrame(rows)


def session_constancy_audit(background: pd.DataFrame) -> pd.DataFrame:
    """
    What was actually held constant between the two imaging sessions?

    Answers referee action 3 from the recorded EXIF rather than from memory, so
    the Methods statement can be specific about the boundary of what is known.
    """
    background = background.copy()
    background["date"] = background["capture_datetime"].str[:10].str.replace(":", "-")
    background["time"] = background["capture_datetime"].str[11:]
    rows: list[dict] = []

    def add(field: str, series_by_day: dict[str, str], constant: bool | None) -> None:
        rows.append({
            "field": field,
            "session_1_pre": series_by_day.get(PRE, ""),
            "session_2_post": series_by_day.get(POST, ""),
            "held_constant": {True: "yes", False: "no", None: "not recorded"}[constant],
        })

    g = background.groupby("DAY")
    add("Camera body",
        {d: ", ".join(sorted(set(v["camera"].dropna()))) for d, v in g},
        background["camera"].nunique() == 1)
    add("Date", {d: ", ".join(sorted(set(v["date"]))) for d, v in g}, False)
    add("Time of day (first-last)",
        {d: f"{v['time'].min()}-{v['time'].max()}" for d, v in g}, False)
    add("ISO", {d: ", ".join(str(int(x)) for x in sorted(set(v["iso"].dropna()))) for d, v in g},
        background["iso"].nunique() == 1)
    add("Focal length (mm)",
        {d: ", ".join(f"{x:g}" for x in sorted(set(v["focal_length_mm"].dropna()))) for d, v in g},
        background["focal_length_mm"].nunique() == 1)
    add("Aperture",
        {d: ", ".join(f"f/{x:g}" for x in sorted(set(v["f_number"].dropna()))) for d, v in g},
        background["f_number"].nunique() == 1)
    add("Shutter speed (s)",
        {d: f"{v['exposure_time_s'].min():g}-{v['exposure_time_s'].max():g}" for d, v in g}, False)
    add("Exposure compensation (EV)",
        {d: ", ".join(f"{x:+g} (n={int((v['exposure_bias_ev'] == x).sum())})"
                      for x in sorted(set(v["exposure_bias_ev"].dropna()))) for d, v in g},
        False)
    add("Flash", {d: "suppressed" if set(v["flash"].dropna()) == {16.0} else "varied" for d, v in g},
        background["flash"].nunique() == 1)
    add("Background board", {PRE: "blue board", POST: "blue board"}, True)
    add("Realised exposure (board luma)",
        {d: f"{v['board_luma'].mean():.1f} (SD {v['board_luma'].std():.1f})" for d, v in g}, None)
    add("Ambient illumination", {PRE: "not logged", POST: "not logged"}, None)
    add("Working distance", {PRE: "not recorded", POST: "not recorded"}, None)
    add("Operator", {PRE: "not recorded in metadata", POST: "not recorded in metadata"}, None)
    return pd.DataFrame(rows)


def figure_prepost(per_image: pd.DataFrame, out_path: Path) -> None:
    """
    Paired pre/post trajectories, in grayscale.

    Session is distinguished by marker fill rather than by hue: pre-stress is a
    filled dark circle, post-stress an open square. The connecting lines and the
    bold group-mean line carry the result, so they are the darkest elements.
    """
    panels = [
        ("raw_mean_intensity", "Raw mean spot intensity"),
        ("skin_ref_log_ratio_linear", "Skin-referenced log-ratio\n(linearised)"),
        ("image_skin_median", "Median opercular skin\nintensity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (col, label) in zip(axes, panels):
        w = paired_units(per_image, col)
        res = paired_test(per_image, col)
        for _, r in w.iterrows():
            ax.plot([0, 1], [r["pre"], r["post"]], color=LINK_GREY, lw=0.8,
                    alpha=0.75, zorder=1)
        ax.scatter(np.zeros(len(w)), w["pre"], s=38, marker="o",
                   facecolors=PRE_GREY, edgecolors="black", linewidths=0.5,
                   zorder=3, label="pre-stress")
        ax.scatter(np.ones(len(w)), w["post"], s=38, marker="s",
                   facecolors="white", edgecolors="black", linewidths=1.0,
                   zorder=3, label="post-stress")
        ax.plot([0, 1], [w["pre"].mean(), w["post"].mean()], color="black", lw=3.0,
                zorder=4, solid_capstyle="round")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Session 1\n(pre-stress)", "Session 2\n(post-stress)"])
        ax.set_ylabel(label)
        star = "n.s." if res["p_value"] >= 0.05 else format_p(res["p_value"])
        ax.set_title(f"{label}\nchange = {res['mean_change']:+.4f}, {star}", fontsize=10)
        ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlim(-0.35, 1.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()

    d = args.results_dir / "data"
    per_image = pd.read_csv(d / "image_level_measures.csv", dtype={"FISH_ID": str, "DAY": str, "SIDE": str})

    print("=" * 78)
    print("Pre/post-confinement re-analysis  (day 1 = pre, day 2 = post)")
    print("=" * 78)
    print(f"{len(per_image)} images, {per_image['FISH_ID'].nunique()} fish, "
          f"{len(paired_units(per_image, 'raw_mean_intensity'))} paired fish x side units\n")

    rows = [paired_test(per_image, c) for c in MEASURES if c in per_image.columns]
    table = pd.DataFrame(rows)
    table.to_csv(d / "prepost_confinement_reanalysis.csv", index=False)

    show = table[["measure", "n_units", "pre_mean", "post_mean", "mean_change",
                  "ci_lower", "ci_upper", "p_value", "cohens_dz"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    board_path = d / "image_level_with_exposure.csv"
    covariate = pd.DataFrame()
    if board_path.exists():
        merged = pd.read_csv(board_path, dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
        print("\n" + "-" * 78)
        print("Exposure proxy across sessions (the key confound test)")
        print("-" * 78)
        bres = paired_test(merged, "board_luma")
        print(f"  Blue board luma, pre  = {bres['pre_mean']:.2f}")
        print(f"  Blue board luma, post = {bres['post_mean']:.2f}")
        print(f"  paired change = {bres['mean_change']:+.2f} "
              f"[{bres['ci_lower']:+.2f}, {bres['ci_upper']:+.2f}], "
              f"p = {bres['p_value']:.3f}, dz = {bres['cohens_dz']:+.3f}, n = {bres['n_units']}")
        json.dump(bres, open(d / "prepost_exposure_proxy.json", "w"), indent=2)

        background = pd.read_csv(d / "background_board_exposure.csv", dtype={"DAY": str})
        audit = session_constancy_audit(background)
        audit.to_csv(d / "session_constancy_audit.csv", index=False)
        print("\n" + "-" * 78)
        print("What was held constant between sessions")
        print("-" * 78)
        print(audit.to_string(index=False))

        covariate = exposure_covariate_models(merged)
        covariate.to_csv(d / "prepost_exposure_adjusted.csv", index=False)
        print("\n" + "-" * 78)
        print("Pre/post effect before and after adjusting for board luma")
        print("-" * 78)
        print(covariate[["measure", "n", "beta_unadjusted", "p_unadjusted",
                         "beta_adjusted", "p_adjusted", "beta_shift_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    figure_prepost(per_image, args.results_dir / "figures" / "fig_prepost_confinement.png")
    print(f"\nWrote {d / 'prepost_confinement_reanalysis.csv'} and figure.")


if __name__ == "__main__":
    main()
