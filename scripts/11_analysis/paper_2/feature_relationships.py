#!/usr/bin/env python3
"""
Relationships among extracted descriptors (referee comment 7).

Asks how the descriptors relate to one another, and to the fish:

  1. Spot level: are larger spots darker? Skin-referenced intensity against
     log relative area, with fish as a random intercept.
  2. Image level: total melanization decomposed into its number and size
     components, and against mean spot intensity.
  3. Correlation matrices at both levels, with collinearity diagnostics.
  4. Descriptors against biometrics (fork length, body mass, condition factor,
     sex), at fish level so that body size is not pseudoreplicated.

Run after photometric_analysis.py, spatial_normalisation.py and
spatial_2d_analysis.py:

    python feature_relationships.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/spot_level_skin_referenced.csv  (script 1)
        results/paper_2/data/image_level_measures.csv        (script 1)
        results/paper_2/data/spatial2d_spotwise.csv          (script 6)
        results/paper_2/data/spatial_operculum_geometry.csv  (script 5)
        results/paper_2/data/spatial_biometrics_long.csv     (script 5)
Writes  results/paper_2/data/features_spot_models.csv
        results/paper_2/data/features_image_models.csv
        results/paper_2/data/features_melanization_decomposition.json
        results/paper_2/data/features_corr_spot.csv
        results/paper_2/data/features_corr_image.csv
        results/paper_2/data/features_vif_image.csv
        results/paper_2/data/features_biometric_associations.csv
        results/paper_2/data/features_sex_comparison.csv
        results/paper_2/data/features_by_fish.csv
        results/paper_2/figures/fig_feature_correlations.png
        results/paper_2/figures/fig_feature_relationships.png
        results/paper_2/figures/fig_biometric_associations.png
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


SPOT_DESCRIPTORS = {
    "rel_area_pct": "Relative area (%)",
    "log_rel_area": "log relative area",
    "log_ratio_linear": "Skin-ref. log-ratio",
    "spot_mean": "Raw mean intensity",
    "skin_median": "Local skin intensity",
    "weber_contrast": "Weber contrast",
    "x": "A–P position",
    "y": "D–V position",
}

IMAGE_DESCRIPTORS = {
    "total_melanization": "Integrated melanization",
    "spot_count": "Spot count",
    "relative_spot_area_pct": "Relative spot area (%)",
    "mean_rel_spot_area_pct": "Mean relative spot area (%)",
    "skin_ref_log_ratio_linear": "Skin-ref. log-ratio",
    "raw_mean_intensity": "Raw mean intensity",
    "image_skin_median": "Opercular skin intensity",
}

SCATTER_GREY = "0.45"
POINT_GREY = "0.35"
COMPONENT_GREY = "0.78"
JOINT_GREY = "0.30"

SEX_STYLE = [("0.80", None), ("0.55", "///")]

BIOMETRICS = {
    "fork_length_cm": "Fork length (cm)",
    "body_mass_g": "Body mass (g)",
    "condition_factor": "Condition factor K",
}


def image_id_from_sample(sample: str) -> int | None:
    import re

    m = re.search(r"id-(\d+)", sample)
    return int(m.group(1)) if m else None


def load_spot_level(d: Path) -> pd.DataFrame:
    photo = pd.read_csv(d / "spot_level_skin_referenced.csv")
    spatial = pd.read_csv(d / "spatial2d_spotwise.csv")
    keys = ["sample", "FISH_ID", "DAY", "SIDE", "spot"]
    df = photo.merge(spatial[keys + ["x", "y", "rel_area_pct", "operculum_area_px"]],
                     on=keys, how="inner", suffixes=("", "_sp"))
    df["log_rel_area"] = np.log(df["rel_area_pct"])
    return df


def load_image_level(d: Path) -> pd.DataFrame:
    img = pd.read_csv(d / "image_level_measures.csv",
                      dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    geom = pd.read_csv(d / "spatial_operculum_geometry.csv",
                       dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    df = img.merge(geom[["sample", "operculum_area_px", "rel_spot_area_pct"]],
                   on="sample", how="left")
    df["mean_rel_spot_area_pct"] = df["rel_spot_area_pct"] / df["spot_count"]
    return df


def add_biometrics(df: pd.DataFrame, d: Path) -> pd.DataFrame:
    bio = pd.read_csv(d / "spatial_biometrics_long.csv", dtype={"DAY": str})
    out = df.copy()
    out["image_id"] = out["sample"].map(image_id_from_sample)
    return out.merge(bio, on=["image_id", "DAY"], how="left")


def mixed(df: pd.DataFrame, formula: str, term: str, label: str) -> dict:
    fit = smf.mixedlm(formula, df, groups=df["FISH_ID"]).fit(reml=False)
    beta = float(fit.params[term])
    lo, hi = (float(v) for v in fit.conf_int().loc[term])
    fixed = fit.fittedvalues - fit.random_effects_var if False else None
    return {
        "model": label, "formula": formula, "term": term,
        "n": int(len(df)), "n_fish": int(df["FISH_ID"].nunique()),
        "estimate": beta, "ci_lower": lo, "ci_upper": hi,
        "p_value": float(fit.pvalues[term]),
        "group_var": float(fit.cov_re.iloc[0, 0]), "resid_var": float(fit.scale),
        "aic": float(fit.aic),
    }


def spot_level_models(spots: pd.DataFrame) -> pd.DataFrame:
    """Are larger spots darker, and is that independent of where they sit?"""
    df = spots[["FISH_ID", "log_ratio_linear", "log_rel_area", "x", "y",
                "spot_mean", "skin_median"]].dropna()
    rows = [
        mixed(df, "log_ratio_linear ~ log_rel_area", "log_rel_area",
              "Skin-ref. intensity ~ log(relative area)"),
        mixed(df, "log_ratio_linear ~ log_rel_area + x", "log_rel_area",
              "Skin-ref. intensity ~ log(relative area) + A–P position"),
        mixed(df, "log_ratio_linear ~ log_rel_area + x", "x",
              "Skin-ref. intensity ~ log(relative area) + A–P position"),
        mixed(df, "spot_mean ~ log_rel_area", "log_rel_area",
              "Raw intensity ~ log(relative area)"),
    ]
    return pd.DataFrame(rows)


def image_level_models(images: pd.DataFrame) -> pd.DataFrame:
    """Decompose total melanization into number and size components."""
    df = images[["FISH_ID", "total_melanization", "spot_count",
                 "mean_rel_spot_area_pct", "relative_spot_area_pct",
                 "skin_ref_log_ratio_linear", "raw_mean_intensity"]].dropna().copy()
    df["log_melanization"] = np.log(df["total_melanization"])
    rows = [
        mixed(df, "total_melanization ~ skin_ref_log_ratio_linear",
              "skin_ref_log_ratio_linear", "Integrated melanization ~ skin-ref. intensity"),
        mixed(df, "total_melanization ~ raw_mean_intensity", "raw_mean_intensity",
              "Integrated melanization ~ raw mean intensity"),
        mixed(df, "total_melanization ~ spot_count", "spot_count",
              "Integrated melanization ~ spot count"),
        mixed(df, "total_melanization ~ mean_rel_spot_area_pct", "mean_rel_spot_area_pct",
              "Integrated melanization ~ mean relative spot area"),
        mixed(df, "spot_count ~ mean_rel_spot_area_pct", "mean_rel_spot_area_pct",
              "Spot count ~ mean relative spot area"),
        mixed(df, "total_melanization ~ spot_count + mean_rel_spot_area_pct",
              "spot_count", "Integrated melanization ~ count + mean size (joint)"),
        mixed(df, "total_melanization ~ spot_count + mean_rel_spot_area_pct",
              "mean_rel_spot_area_pct", "Integrated melanization ~ count + mean size (joint)"),
    ]
    return pd.DataFrame(rows)


def melanization_decomposition(images: pd.DataFrame) -> dict:
    """
    How much of total melanization is number, and how much is size?

    Total melanization is, by construction, close to (spot count) x (mean spot
    area) x (mean darkening). Fitting the multiplicative form on the log scale
    partitions its variance between the components.
    """
    df = images[["total_melanization", "spot_count", "mean_rel_spot_area_pct",
                 "skin_ref_log_ratio_linear", "relative_spot_area_pct"]].dropna().copy()
    for c in ["total_melanization", "spot_count", "mean_rel_spot_area_pct"]:
        df[f"log_{c}"] = np.log(df[c])
    ols = smf.ols("log_total_melanization ~ log_spot_count + log_mean_rel_spot_area_pct",
                  df).fit()
    simple_count = smf.ols("log_total_melanization ~ log_spot_count", df).fit()
    simple_size = smf.ols("log_total_melanization ~ log_mean_rel_spot_area_pct", df).fit()
    area_only = smf.ols("total_melanization ~ relative_spot_area_pct", df).fit()
    return {
        "n": int(len(df)),
        "r2_count_only": float(simple_count.rsquared),
        "r2_size_only": float(simple_size.rsquared),
        "r2_joint": float(ols.rsquared),
        "coef_log_count": float(ols.params["log_spot_count"]),
        "coef_log_size": float(ols.params["log_mean_rel_spot_area_pct"]),
        "p_log_count": float(ols.pvalues["log_spot_count"]),
        "p_log_size": float(ols.pvalues["log_mean_rel_spot_area_pct"]),
        "r2_relative_area_only": float(area_only.rsquared),
        "corr_count_size": float(df["log_spot_count"].corr(df["log_mean_rel_spot_area_pct"])),
    }


def correlation_matrix(df: pd.DataFrame, descriptors: dict) -> pd.DataFrame:
    cols = [c for c in descriptors if c in df.columns]
    sub = df[cols].dropna()
    corr = sub.corr(method="spearman")
    corr.index = [descriptors[c] for c in corr.index]
    corr.columns = [descriptors[c] for c in corr.columns]
    return corr


def vif_table(df: pd.DataFrame, descriptors: dict) -> pd.DataFrame:
    """Variance inflation factors, as the collinearity diagnostic."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = [c for c in descriptors if c in df.columns]
    sub = df[cols].dropna()
    X = ((sub - sub.mean()) / sub.std()).to_numpy()
    rows = []
    for i, c in enumerate(cols):
        try:
            v = float(variance_inflation_factor(X, i))
        except Exception:
            v = np.nan
        rows.append({"descriptor": descriptors[c], "vif": v})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def biometric_associations(images: pd.DataFrame, d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descriptors against fish biometrics, at fish level.

    Body size and sex are properties of the fish, and each fish contributes four
    images, so the analysis is run on fish-level means (n = 21). Testing per
    image would quadruple the apparent sample size without adding information.
    """
    merged = add_biometrics(images, d)
    descriptors = ["total_melanization", "spot_count", "relative_spot_area_pct",
                   "mean_rel_spot_area_pct", "skin_ref_log_ratio_linear"]
    agg = {c: (c, "mean") for c in descriptors}
    agg.update({c: (c, "mean") for c in BIOMETRICS})
    by_fish = merged.groupby("FISH_ID", as_index=False).agg(
        **{k: v for k, v in agg.items()},
        sex=("sex", lambda s: s.dropna().iloc[0] if s.dropna().size else None),
        n_images=("sample", "size"))

    rows: list[dict] = []
    for desc in descriptors:
        for pred in BIOMETRICS:
            sub = by_fish[[desc, pred]].dropna()
            if len(sub) < 5:
                continue
            fit = stats.linregress(sub[pred], sub[desc])
            rows.append({"descriptor": desc, "predictor": pred, "n": int(len(sub)),
                         "r": float(fit.rvalue), "r_squared": float(fit.rvalue ** 2),
                         "p_value": float(fit.pvalue), "slope": float(fit.slope)})
    assoc = pd.DataFrame(rows)
    if len(assoc):
        from statsmodels.stats.multitest import multipletests

        assoc["q_value_bh"] = multipletests(assoc["p_value"], method="fdr_bh")[1]
        assoc["significant_at_q05"] = assoc["q_value_bh"] < 0.05

    sex_rows: list[dict] = []
    for desc in descriptors:
        sub = by_fish[[desc, "sex"]].dropna()
        groups = [g[desc].to_numpy() for _, g in sub.groupby("sex")]
        names = [str(n) for n, _ in sub.groupby("sex")]
        if len(groups) != 2 or min(len(g) for g in groups) < 3:
            continue
        u, p = stats.mannwhitneyu(groups[0], groups[1])
        t, pt = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        pooled = np.sqrt((groups[0].var(ddof=1) + groups[1].var(ddof=1)) / 2)
        sex_rows.append({
            "descriptor": desc, "group_a": names[0], "n_a": len(groups[0]),
            "group_b": names[1], "n_b": len(groups[1]),
            "mean_a": float(groups[0].mean()), "mean_b": float(groups[1].mean()),
            "cohens_d": float((groups[0].mean() - groups[1].mean()) / pooled) if pooled else np.nan,
            "mannwhitney_p": float(p), "welch_p": float(pt),
        })
    return assoc, pd.DataFrame(sex_rows), by_fish


def figure_corr(spot_corr: pd.DataFrame, image_corr: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, corr, title in [
        (axes[0], spot_corr, f"Spot level (n = {spot_corr.attrs.get('n', '')})"),
        (axes[1], image_corr, f"Image level (n = {image_corr.attrs.get('n', '')})"),
    ]:
        im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.to_numpy()[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_key_relationships(spots: pd.DataFrame, images: pd.DataFrame,
                             spot_models: pd.DataFrame, decomp: dict,
                             out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    ax = axes[0]
    d = spots[["log_rel_area", "log_ratio_linear"]].dropna()
    ax.scatter(np.exp(d["log_rel_area"]), d["log_ratio_linear"], s=8, alpha=0.22,
               color=SCATTER_GREY, edgecolors="none")
    m = spot_models.iloc[0]
    xs = np.linspace(d["log_rel_area"].min(), d["log_rel_area"].max(), 100)
    intercept = d["log_ratio_linear"].mean() - m["estimate"] * d["log_rel_area"].mean()
    ax.plot(np.exp(xs), intercept + m["estimate"] * xs, color="black", lw=2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Spot area (% of operculum)")
    ax.set_ylabel("Skin-referenced log-ratio\n(more negative = darker)")
    ax.set_title(f"Larger spots are darker\nβ = {m['estimate']:.3f} "
                 f"[{m['ci_lower']:.3f}, {m['ci_upper']:.3f}], {format_p(m['p_value'])}",
                 fontsize=10)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)

    ax = axes[1]
    log_count = np.log(images["spot_count"])
    log_melan = np.log(images["total_melanization"])
    ax.scatter(log_count, log_melan, s=44, alpha=0.85,
               color=POINT_GREY, edgecolors="black", linewidths=0.5)
    fit = stats.linregress(log_count, log_melan)
    xs = np.linspace(log_count.min(), log_count.max(), 100)
    ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=2.0)
    ax.set_xlabel("log spot count")
    ax.set_ylabel("log integrated melanization")
    ax.set_title(f"Spot count alone explains only {fit.rvalue ** 2 * 100:.0f}%\n"
                 f"r = {fit.rvalue:.3f}, R² = {fit.rvalue ** 2:.3f}", fontsize=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)

    ax = axes[2]
    parts = ["Spot count\nalone", "Mean spot size\nalone", "Both\njointly"]
    vals = [decomp["r2_count_only"], decomp["r2_size_only"], decomp["r2_joint"]]
    bars = ax.bar(parts, vals, color=[COMPONENT_GREY, COMPONENT_GREY, JOINT_GREY],
                  edgecolor="black", linewidth=0.8,
                  hatch=["///", "\\\\\\", None])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("R² for log(integrated melanization)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Decomposition of integrated melanization", fontsize=10)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_biometrics(by_fish: pd.DataFrame, assoc: pd.DataFrame, out_path: Path) -> None:
    panels = [("condition_factor", "total_melanization", "Condition factor K", "Integrated melanization"),
              ("body_mass_g", "relative_spot_area_pct", "Body mass (g)", "Relative spot area (%)"),
              ("fork_length_cm", "spot_count", "Fork length (cm)", "Spot count")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (pred, desc, xl, yl) in zip(axes[:3], panels):
        sub = by_fish[[pred, desc]].dropna()
        ax.scatter(sub[pred], sub[desc], s=54, alpha=0.9, color=POINT_GREY,
                   edgecolors="black", linewidths=0.5)
        fit = stats.linregress(sub[pred], sub[desc])
        xs = np.linspace(sub[pred].min(), sub[pred].max(), 100)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=1.8)
        row = assoc[(assoc["predictor"] == pred) & (assoc["descriptor"] == desc)]
        rr = row.iloc[0] if len(row) else None
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"r = {rr['r']:.3f}, {format_p(rr['p_value'])}, n = {int(rr['n'])}"
                     if rr is not None else "", fontsize=10)
        ax.grid(True, ls="--", lw=0.5, alpha=0.4)

    ax = axes[3]
    sub = by_fish[["sex", "total_melanization"]].dropna()
    groups = [(n, g["total_melanization"].to_numpy()) for n, g in sub.groupby("sex")]
    bp = ax.boxplot([g for _, g in groups], tick_labels=[n for n, _ in groups],
                    widths=0.5, patch_artist=True,
                    medianprops={"color": "black"})
    for box, (grey, hatch) in zip(bp["boxes"], SEX_STYLE):
        box.set_facecolor(grey)
        box.set_edgecolor("black")
        box.set_hatch(hatch)
    for i, (_, g) in enumerate(groups, start=1):
        ax.scatter(np.random.default_rng(3).normal(i, 0.05, len(g)), g, s=34,
                   facecolors="white", edgecolors="black", linewidths=0.7,
                   zorder=3)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Integrated melanization")
    ax.set_title("By sex (fish level)", fontsize=10)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = p.parse_args()
    d = args.results_dir / "data"

    spots = load_spot_level(d)
    images = load_image_level(d)
    print(f"Spot level: {len(spots):,} spots   Image level: {len(images)} images")

    print("\n[1/4] Spot-level model: are larger spots darker?")
    sm = spot_level_models(spots)
    sm.to_csv(d / "features_spot_models.csv", index=False)
    print(sm[["model", "term", "estimate", "ci_lower", "ci_upper", "p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[2/4] Image-level models")
    im = image_level_models(images)
    im.to_csv(d / "features_image_models.csv", index=False)
    print(im[["model", "term", "estimate", "ci_lower", "ci_upper", "p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    decomp = melanization_decomposition(images)
    (d / "features_melanization_decomposition.json").write_text(
        json.dumps(decomp, indent=2), encoding="utf-8")
    print("\n  Melanization decomposition:")
    for k, v in decomp.items():
        print(f"    {k:<26} {v:.4f}" if isinstance(v, float) else f"    {k:<26} {v}")

    print("\n[3/4] Correlation structure")
    spot_corr = correlation_matrix(spots, SPOT_DESCRIPTORS)
    image_corr = correlation_matrix(images, IMAGE_DESCRIPTORS)
    spot_corr.attrs["n"] = len(spots)
    image_corr.attrs["n"] = len(images)
    spot_corr.to_csv(d / "features_corr_spot.csv")
    image_corr.to_csv(d / "features_corr_image.csv")
    vif_i = vif_table(images, IMAGE_DESCRIPTORS)
    vif_i.to_csv(d / "features_vif_image.csv", index=False)
    print(image_corr.round(2).to_string())
    print("\n  VIF (image level):")
    print(vif_i.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n[4/4] Biometrics")
    assoc, sex, by_fish = biometric_associations(images, d)
    assoc.to_csv(d / "features_biometric_associations.csv", index=False)
    sex.to_csv(d / "features_sex_comparison.csv", index=False)
    by_fish.to_csv(d / "features_by_fish.csv", index=False)
    print(assoc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(sex.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    figure_corr(spot_corr, image_corr, args.results_dir / "figures" / "fig_feature_correlations.png")
    figure_key_relationships(spots, images, sm, decomp,
                             args.results_dir / "figures" / "fig_feature_relationships.png")
    figure_biometrics(by_fish, assoc, args.results_dir / "figures" / "fig_biometric_associations.png")
    print("\nFigures written.")


if __name__ == "__main__":
    main()
