#!/usr/bin/env python3
"""
Biological covariates and association analyses (referee comment 8).

Follows the order the revision requires:

  1. Audit the biometric table first. What exists, with what n and what
     missingness, determines which analyses are possible at all.
  2. Pre-specify responses, predictors and models, and write that
     specification to disk BEFORE any model is fitted.
  3. Fit image-level linear mixed models with fish as a random intercept, and
     a negative-binomial GEE for the count response, with FDR correction
     across the pre-specified family.
  4. Report effect sizes with 95% confidence intervals throughout.

Inputs, all from this repository:
  reference_tabular_data.csv   biometrics and welfare scores for the 21 fish
  results/paper_2/data/    descriptors from the earlier stages

Run:
    python biological_validation.py --results-dir results/paper_2

Datasets
--------
Reads   reference_tabular_data.csv  biometrics and welfare scores, n = 21
        results/paper_2/data/image_level_measures.csv        (script 1)
        results/paper_2/data/spatial_operculum_geometry.csv  (script 5)
Writes  results/paper_2/data/bio_availability_audit.csv
        results/paper_2/data/bio_prespecification.json
        results/paper_2/data/bio_primary_associations.csv
        results/paper_2/data/bio_validation_associations.csv
        results/paper_2/data/bio_welfare_associations.csv
        results/paper_2/data/bio_descriptive_sex.csv
        results/paper_2/data/bio_descriptive_maturation.csv
        results/paper_2/data/bio_predictor_collinearity.csv
        results/paper_2/data/bio_power.json
        results/paper_2/figures/fig_bio_forest.png
        results/paper_2/figures/fig_significant_associations.png
        results/paper_2/figures/fig_bio_validation.png

The biometric table contains no HSI, GSI or growth record; the audit stage
reports this explicitly rather than silently omitting those analyses.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

from photometric_analysis import format_p


import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


COLUMN_GLOSS = {
    "fiskenr dag 1": ("Fish number, day 1 (pre-stress)", "identifier"),
    "fiskenr dag 2, post-stress": ("Fish number, day 2 (post-stress)", "identifier"),
    "Pit": ("PIT tag", "identifier"),
    "Short PIT": ("Short PIT tag", "identifier"),
    "Linje": ("Line/strain", "constant"),
    "lengde": ("Fork length (cm)", "morphometric"),
    "vekt": ("Body mass (g)", "morphometric"),
    "K": ("Condition factor", "morphometric"),
    "Comment": ("Comment", "constant"),
    "Kjønn": ("Sex", "primary"),
    "Kjønnsmodning": ("Sexual maturation (0/1)", "primary"),
    "fading": ("Pigmentation fading score (0-3, visual)", "validation"),
    "Førsteinntrykk": ("First impression score", "welfare"),
    "Ryggradsdeform": ("Spinal deformity", "welfare"),
    "Avmagring": ("Emaciation", "welfare"),
    "Skjelltap": ("Scale loss", "welfare"),
    "Hudblødning": ("Skin haemorrhage", "welfare"),
    "Kroppssår": ("Body wounds", "welfare"),
    "Snutesår": ("Snout wounds", "welfare"),
    "Kjevedeform": ("Jaw deformity", "welfare"),
    "Øyeblakking": ("Eye opacity", "welfare"),
    "Øyeskade": ("Eye damage", "welfare"),
    "Gjellelokk": ("Operculum (gill cover) damage", "welfare"),
    "Gjellestatus": ("Gill status", "welfare"),
    "Finnestatus": ("Fin status", "welfare"),
    "Finnestatus.1": ("Fin status (duplicate column)", "welfare"),
}

ABSENT_REQUESTED = ["HSI (hepatosomatic index)", "GSI (gonadosomatic index)",
                    "Growth rate / specific growth rate"]

MIN_GROUP = 4

SUPPRESSED_GREY = "0.58"

POINT_GREY = "0.45"

GROUP_GREY = ("0.80", "0.55")

RESPONSES = {
    "total_melanization": ("Integrated melanization", "lmm"),
    "spot_count": ("Spot count", "nb"),
    "mean_rel_spot_area_pct": ("Mean relative spot area (%)", "lmm"),
    "skin_ref_log_ratio_linear": ("Mean skin-referenced intensity", "lmm"),
}

COL_DAY1 = "fiskenr dag 1"
COL_DAY2 = "fiskenr dag 2, post-stress"


def image_id_from_sample(sample: str) -> int | None:
    m = re.search(r"id-(\d+)", sample)
    return int(m.group(1)) if m else None


def load_biometrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide per-fish table plus a long per-image key, and the availability audit."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]

    audit_rows = []
    for col in raw.columns:
        gloss, role = COLUMN_GLOSS.get(col, (col, "unclassified"))
        series = raw[col]
        n = int(series.notna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        vc = series.value_counts(dropna=True)
        audit_rows.append({
            "column": col, "variable": gloss, "role": role,
            "n_present": n, "n_missing": int(len(raw) - n),
            "pct_missing": 100.0 * (len(raw) - n) / len(raw),
            "n_levels": int(series.nunique(dropna=True)),
            "min_group_n": int(vc.min()) if len(vc) and series.nunique() <= 6 else np.nan,
            "min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "max": float(numeric.max()) if numeric.notna().any() else np.nan,
            "distribution": (", ".join(f"{k}:{v}" for k, v in vc.sort_index().items())
                             if 0 < series.nunique(dropna=True) <= 6 else ""),
        })
    audit = pd.DataFrame(audit_rows)

    rows = []
    for _, r in raw.iterrows():
        for col, day in [(COL_DAY1, "1"), (COL_DAY2, "2")]:
            if pd.isna(r.get(col)):
                continue
            rows.append({
                "image_id": int(r[col]), "DAY": day,
                "fork_length_cm": pd.to_numeric(r.get("lengde"), errors="coerce"),
                "body_mass_g": pd.to_numeric(r.get("vekt"), errors="coerce"),
                "condition_factor": pd.to_numeric(r.get("K"), errors="coerce"),
                "sex": str(r.get("Kjønn")).strip() if pd.notna(r.get("Kjønn")) else None,
                "maturation": pd.to_numeric(r.get("Kjønnsmodning"), errors="coerce"),
                "fading_score": pd.to_numeric(r.get("fading"), errors="coerce"),
                "operculum_damage": pd.to_numeric(r.get("Gjellelokk"), errors="coerce"),
                "scale_loss": pd.to_numeric(r.get("Skjelltap"), errors="coerce"),
                "body_wounds": pd.to_numeric(r.get("Kroppssår"), errors="coerce"),
                "fin_status": pd.to_numeric(r.get("Finnestatus"), errors="coerce"),
                "first_impression": pd.to_numeric(r.get("Førsteinntrykk"), errors="coerce"),
            })
    return pd.DataFrame(rows), audit


def prespecify(bio_long: pd.DataFrame) -> dict:
    """
    Decide which predictors the design can support, before fitting anything.

    Categorical predictors whose smallest group has fewer than MIN_GROUP fish
    are excluded with a reason rather than tested and then explained away.
    """
    by_fish = bio_long.drop_duplicates("image_id")
    candidates = {
        "sex": ("Sex", "categorical", "primary"),
        "body_mass_g": ("Body mass (g)", "continuous", "primary"),
        "fork_length_cm": ("Fork length (cm)", "continuous", "primary"),
        "condition_factor": ("Condition factor", "continuous", "primary"),
        "maturation": ("Sexual maturation", "categorical", "primary"),
        "fading_score": ("Visual fading score", "continuous", "validation"),
        "operculum_damage": ("Operculum damage", "categorical", "welfare"),
        "scale_loss": ("Scale loss", "categorical", "welfare"),
        "body_wounds": ("Body wounds", "categorical", "welfare"),
        "fin_status": ("Fin status", "categorical", "welfare"),
        "first_impression": ("First impression", "categorical", "welfare"),
    }
    included, excluded = {}, {}
    for col, (label, kind, family) in candidates.items():
        s = by_fish[col].dropna()
        n_fish = len(s)
        if n_fish < 10:
            excluded[col] = {"label": label, "reason": f"only {n_fish} fish with data"}
            continue
        if kind == "categorical":
            vc = s.value_counts()
            if vc.min() < MIN_GROUP:
                excluded[col] = {
                    "label": label,
                    "reason": (f"smallest group has {int(vc.min())} fish "
                               f"(minimum {MIN_GROUP} required); distribution "
                               + ", ".join(f"{k}:{v}" for k, v in vc.sort_index().items()))}
                continue
        included[col] = {"label": label, "kind": kind, "family": family, "n_fish": n_fish}

    return {
        "responses": {k: {"label": v[0], "model": v[1]} for k, v in RESPONSES.items()},
        "included_predictors": included,
        "excluded_predictors": excluded,
        "absent_requested_biomarkers": ABSENT_REQUESTED,
        "model_spec": {
            "lmm": "descriptor ~ predictor + C(DAY) + C(SIDE), random intercept for fish",
            "nb": ("spot_count ~ predictor + C(DAY) + C(SIDE), negative-binomial GEE "
                   "with exchangeable working correlation clustered by fish"),
            "rationale": ("Imaging occasion and side are included as fixed covariates "
                          "because both were shown to shift the descriptors "
                          "(Sections 3 and 7); fish is a random intercept because each "
                          "fish contributes four images."),
        },
        "multiplicity": ("Benjamini-Hochberg FDR applied across the primary family "
                         "(responses x primary predictors). The validation and welfare "
                         "families are corrected separately."),
        "min_group_n": MIN_GROUP,
    }


def fit_one(df: pd.DataFrame, response: str, predictor: str, kind: str,
            model: str) -> dict | None:
    """One pre-specified association, with an effect size and 95% CI."""
    cols = [response, predictor, "FISH_ID", "DAY", "SIDE"]
    data = df[cols].dropna().copy()
    if len(data) < 20:
        return None

    if kind == "continuous":
        sd = data[predictor].std()
        if sd == 0:
            return None
        data["_pred"] = (data[predictor] - data[predictor].mean()) / sd
        term = "_pred"
        unit = "per 1 SD"
    else:
        data["_pred"] = data[predictor].astype(str)
        levels = sorted(data["_pred"].unique())
        if len(levels) != 2:
            return None
        term = f"C(_pred)[T.{levels[1]}]"
        unit = f"{levels[1]} vs {levels[0]}"

    formula = f"{response} ~ {'_pred' if kind == 'continuous' else 'C(_pred)'} + C(DAY) + C(SIDE)"
    try:
        if model == "nb":
            fit = smf.gee(formula, groups=data["FISH_ID"], data=data,
                          family=sm.families.NegativeBinomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        else:
            fit = smf.mixedlm(formula, data, groups=data["FISH_ID"]).fit(reml=False)
        beta = float(fit.params[term])
        lo, hi = (float(v) for v in fit.conf_int().loc[term])
        p = float(fit.pvalues[term])
    except Exception as exc:  # pragma: no cover
        return {"response": response, "predictor": predictor, "error": str(exc)[:80]}

    out = {
        "response": response, "predictor": predictor, "kind": kind, "model": model,
        "unit": unit, "n_images": int(len(data)),
        "n_fish": int(data["FISH_ID"].nunique()),
        "estimate": beta, "ci_lower": lo, "ci_upper": hi, "p_value": p,
        "response_sd": float(data[response].std()),
    }
    if model == "nb":
        out["rate_ratio"] = float(np.exp(beta))
        out["rr_lower"] = float(np.exp(lo))
        out["rr_upper"] = float(np.exp(hi))
        out["std_effect"] = float(np.exp(beta) - 1.0)
        out["std_lower"] = float(np.exp(lo) - 1.0)
        out["std_upper"] = float(np.exp(hi) - 1.0)
    else:
        s = out["response_sd"]
        out["std_effect"] = beta / s if s else np.nan
        out["std_lower"] = lo / s if s else np.nan
        out["std_upper"] = hi / s if s else np.nan
    return out


def run_family(df: pd.DataFrame, spec: dict, family: str) -> pd.DataFrame:
    rows = []
    for pred, info in spec["included_predictors"].items():
        if info["family"] != family:
            continue
        for resp, rinfo in RESPONSES.items():
            r = fit_one(df, resp, pred, info["kind"], rinfo[1])
            if r and "error" not in r:
                r["response_label"] = rinfo[0]
                r["predictor_label"] = info["label"]
                rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out["significant_q05"] = out["q_value_bh"] < 0.05
    return out


def predictor_collinearity(bio_long: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    How far are the predictors themselves confounded with one another?

    With 21 fish the predictors are not independent: body size, condition and
    the welfare scores co-vary, so an association with one may be an association
    with another. Reported so that no single association is read causally.
    """
    by_fish = bio_long.drop_duplicates("image_id").copy()
    cols = [c for c in spec["included_predictors"] if c in by_fish.columns]
    num = by_fish[cols].copy()
    if "sex" in num.columns:
        num["sex"] = (num["sex"] == "M").astype(float)
    num = num.apply(pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman")
    labels = {c: spec["included_predictors"][c]["label"] for c in cols}
    corr.index = [labels[c] for c in corr.index]
    corr.columns = [labels[c] for c in corr.columns]
    return corr


def detectable_effect(n_fish: int, alpha: float = 0.05, power: float = 0.80) -> dict:
    """
    Smallest correlation this sample could detect, and the power it had.

    Reported instead of a bare post-hoc power calculation on the observed
    effect, which would be circular.
    """
    from scipy.optimize import brentq

    def power_at_r(r: float) -> float:
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n_fish - 3)
        crit = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(crit - abs(z) / se) + stats.norm.cdf(-crit - abs(z) / se))

    try:
        r_min = brentq(lambda r: power_at_r(r) - power, 0.01, 0.99)
    except Exception:
        r_min = np.nan
    return {
        "n_fish": n_fish, "alpha": alpha, "target_power": power,
        "min_detectable_r": float(r_min),
        "min_detectable_r_squared": float(r_min ** 2),
        "power_at_r_0.3": power_at_r(0.3),
        "power_at_r_0.5": power_at_r(0.5),
    }


def descriptive_split(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Descriptive summary by a grouping variable, at fish level."""
    by_fish = (df.groupby(["FISH_ID", by], as_index=False)
               .agg(**{r: (r, "mean") for r in RESPONSES}))
    rows = []
    for resp, (label, _) in RESPONSES.items():
        g = by_fish.groupby(by)[resp]
        entry = {"descriptor": label}
        for name, vals in g:
            entry[f"{name}_n"] = int(vals.notna().sum())
            entry[f"{name}_mean"] = float(vals.mean())
            entry[f"{name}_sd"] = float(vals.std())
        rows.append(entry)
    return pd.DataFrame(rows)


def figure_forest(primary: pd.DataFrame, out_path: Path) -> None:
    """
    Forest plot of the pre-specified associations, in grayscale.

    The one distinction that matters is whether an association survives FDR
    correction. That is carried by marker fill rather than by tone: filled
    black markers on dark intervals survive at q < 0.05, open markers on light
    intervals do not. Fill is categorical and unambiguous in monochrome, where
    two grey levels would be easy to confuse against the gridlines.
    """
    df = primary.sort_values(["predictor_label", "response_label"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(df) + 2.2))
    ypos = np.arange(len(df))[::-1]
    sig = df["significant_q05"].to_numpy(dtype=bool)
    xerr = np.vstack([df["std_effect"] - df["std_lower"],
                      df["std_upper"] - df["std_effect"]])

    for mask, ecol, face, edge, lw, label in [
        (~sig, SUPPRESSED_GREY, "white", "0.30", 1.1,
         "Not significant after FDR"),
        (sig, "0.15", "black", "black", 1.5,
         "Survives FDR correction (q < 0.05)"),
    ]:
        if not mask.any():
            continue
        ax.errorbar(df["std_effect"][mask], ypos[mask], xerr=xerr[:, mask],
                    fmt="none", ecolor=ecol, elinewidth=lw, capsize=3,
                    zorder=2)
        ax.scatter(df["std_effect"][mask], ypos[mask], s=62, facecolors=face,
                   edgecolors=edge, linewidths=1.0, zorder=3, label=label)

    ax.axvline(0, color="black", lw=1.1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.predictor_label}  →  {r.response_label}"
                        for r in df.itertuples()], fontsize=8)
    ax.set_xlabel("Standardised effect  (response SDs per 1 SD of predictor;\n"
                  "proportional change in rate for spot count)", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8, loc="upper left",
              framealpha=0.95)
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_significant(df: pd.DataFrame, primary: pd.DataFrame, out_path: Path) -> None:
    """
    Every descriptor-biometric association that survives FDR correction, one
    panel each (referee comment 8).

    Points are fish means (n = 21), because the biometrics are per-fish and
    plotting all 84 images would show each fish four times and overstate the
    sample. The effect size, interval and q printed on each panel come from the
    image-level model that the analysis actually used, which keeps fish as a
    random intercept and adjusts for imaging session and side; the drawn line is
    an ordinary least-squares fit through the fish means and is a visual aid
    only. Panels are grouped by predictor so the shared body-size axis is
    obvious at a glance.
    """
    sig = primary[primary["significant_q05"]].copy()
    if sig.empty:
        return

    order = {"body_mass_g": 0, "fork_length_cm": 1, "condition_factor": 2,
             "maturation": 3, "sex": 4}
    sig["_p"] = sig["predictor"].map(order).fillna(9)
    sig = sig.sort_values(["_p", "q_value_bh"]).reset_index(drop=True)

    by_fish = (df.groupby("FISH_ID", as_index=False)
               .agg({**{p: "mean" for p in order if p in df.columns
                        and pd.api.types.is_numeric_dtype(df[p])},
                     **{r: "mean" for r in RESPONSES}}))

    n = len(sig)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, sig.itertuples()):
        pred, resp = row.predictor, row.response
        if pred not in by_fish.columns:
            ax.axis("off")
            continue
        sub = by_fish[[pred, resp]].dropna()

        if pd.notna(getattr(row, "rate_ratio", np.nan)):
            eff = f"rate ratio {row.rate_ratio:.2f} [{row.rr_lower:.2f}, {row.rr_upper:.2f}]"
        else:
            eff = f"{row.std_effect:+.2f} SD [{row.std_lower:+.2f}, {row.std_upper:+.2f}]"
        qtxt = format_p(row.q_value_bh, label="q")

        if sub[pred].nunique() <= 2:
            groups = sorted(sub[pred].unique())
            labels = {0.0: "Not maturing", 1.0: "Maturing"}
            rng = np.random.default_rng(7)
            for i, g in enumerate(groups, start=1):
                vals = sub.loc[sub[pred] == g, resp].to_numpy()
                bp = ax.boxplot([vals], positions=[i], widths=0.45,
                                patch_artist=True, medianprops={"color": "black"},
                                flierprops={"marker": ""})
                face = GROUP_GREY[0] if i == 1 else GROUP_GREY[1]
                for box in bp["boxes"]:
                    box.set_facecolor(face)
                    box.set_edgecolor("black")
                    box.set_hatch(None if i == 1 else "///")
                ax.scatter(rng.normal(i, 0.05, len(vals)), vals, s=42,
                           facecolors="white", edgecolors="black",
                           linewidths=0.8, zorder=3)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([f"{labels.get(g, g)}\n(n = {int((sub[pred] == g).sum())})"
                                for g in groups], fontsize=8)
            ax.set_xlabel("")
        else:
            ax.scatter(sub[pred], sub[resp], s=60, color=POINT_GREY,
                       edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3)
            fit = stats.linregress(sub[pred], sub[resp])
            xs = np.linspace(sub[pred].min(), sub[pred].max(), 100)
            ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=1.8, zorder=4)
            r, _ = stats.pearsonr(sub[pred], sub[resp])
            eff = f"{eff}\nfish-level r = {r:+.2f}"
            ax.set_xlabel(row.predictor_label, fontsize=9)

        ax.set_ylabel(row.response_label, fontsize=9)
        ax.set_title(f"{row.predictor_label}  \u2192  {row.response_label}\n{eff}, {qtxt}",
                     fontsize=8.5)
        ax.grid(True, ls="--", lw=0.5, alpha=0.35)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_validation(df: pd.DataFrame, valid: pd.DataFrame, out_path: Path) -> None:
    """Image-derived descriptors against the independent visual fading score."""
    by_fish = (df.groupby("FISH_ID", as_index=False)
               .agg(fading_score=("fading_score", "mean"),
                    **{r: (r, "mean") for r in RESPONSES}))
    items = list(RESPONSES.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.1 * len(items), 4.2))
    for ax, (resp, (label, _)) in zip(axes, items):
        sub = by_fish[["fading_score", resp]].dropna()
        ax.scatter(sub["fading_score"], sub[resp], s=54, alpha=0.9,
                   color=POINT_GREY, edgecolors="black", linewidths=0.5)
        if sub["fading_score"].nunique() > 1:
            fit = stats.linregress(sub["fading_score"], sub[resp])
            xs = np.linspace(sub["fading_score"].min(), sub["fading_score"].max(), 50)
            ax.plot(xs, fit.intercept + fit.slope * xs, color="black", lw=1.8)
            r = valid[(valid["response"] == resp)]
            sub_t = ""
            if len(r):
                row = r.iloc[0]
                if pd.notna(row.get("rate_ratio")):
                    eff = (f"rate ratio {row['rate_ratio']:.3f} "
                           f"[{row['rr_lower']:.3f}, {row['rr_upper']:.3f}]")
                else:
                    eff = (f"β = {row['std_effect']:+.2f} SD "
                           f"[{row['std_lower']:+.2f}, {row['std_upper']:+.2f}]")
                sub_t = f"{eff}\n{format_p(row['q_value_bh'], label='q')}"
            ax.set_title(f"{label}\n{sub_t}", fontsize=9)
        ax.set_xlabel("Visual fading score (0–3)")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(True, ls="--", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    p.add_argument("--biometrics", type=Path, default=P.BIOMETRICS_CSV)
    args = p.parse_args()
    d = args.results_dir / "data"

    print("=" * 78)
    print("[1/4] Biometric availability audit")
    print("=" * 78)
    bio_long, audit = load_biometrics(args.biometrics)
    audit.to_csv(d / "bio_availability_audit.csv", index=False)
    show = audit[["variable", "role", "n_present", "n_missing", "min_group_n", "distribution"]]
    print(show.to_string(index=False))
    print(f"\n  Requested but absent from the dataset: {', '.join(ABSENT_REQUESTED)}")

    images = pd.read_csv(d / "image_level_measures.csv",
                         dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    geom = pd.read_csv(d / "spatial_operculum_geometry.csv", dtype={"DAY": str})
    images = images.merge(geom[["sample", "rel_spot_area_pct"]], on="sample", how="left")
    images["mean_rel_spot_area_pct"] = images["rel_spot_area_pct"] / images["spot_count"]
    images["image_id"] = images["sample"].map(image_id_from_sample)
    df = images.merge(bio_long, on=["image_id", "DAY"], how="left")
    print(f"\n  Linked {df['image_id'].notna().sum()} of {len(df)} images to biometrics, "
          f"{df['FISH_ID'].nunique()} fish")

    print("\n" + "=" * 78)
    print("[2/4] Pre-specification (written before any model is fitted)")
    print("=" * 78)
    spec = prespecify(bio_long)
    (d / "bio_prespecification.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print("  Included predictors:")
    for k, v in spec["included_predictors"].items():
        print(f"    {v['label']:<26} {v['kind']:<12} family={v['family']}")
    print("  Excluded predictors:")
    for k, v in spec["excluded_predictors"].items():
        print(f"    {v['label']:<26} {v['reason']}")

    print("\n" + "=" * 78)
    print("[3/4] Fitting the pre-specified families")
    print("=" * 78)
    primary = run_family(df, spec, "primary")
    primary.to_csv(d / "bio_primary_associations.csv", index=False)
    cols = ["predictor_label", "response_label", "n_images", "n_fish", "std_effect",
            "std_lower", "std_upper", "p_value", "q_value_bh", "significant_q05"]
    print("\nPRIMARY FAMILY")
    print(primary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    valid = run_family(df, spec, "validation")
    valid.to_csv(d / "bio_validation_associations.csv", index=False)
    if len(valid):
        print("\nVALIDATION FAMILY (visual fading score)")
        print(valid[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    welfare = run_family(df, spec, "welfare")
    welfare.to_csv(d / "bio_welfare_associations.csv", index=False)
    if len(welfare):
        print("\nWELFARE FAMILY")
        print(welfare[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for by, name in [("sex", "sex"), ("maturation", "maturation")]:
        sub = df[df[by].notna()]
        if len(sub):
            desc = descriptive_split(sub, by)
            desc.to_csv(d / f"bio_descriptive_{name}.csv", index=False)
            print(f"\nDescriptive split by {name}:")
            print(desc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    coll = predictor_collinearity(bio_long, spec)
    coll.to_csv(d / "bio_predictor_collinearity.csv")
    print("\nPredictor collinearity (Spearman, fish level):")
    print(coll.round(2).to_string())

    power = detectable_effect(int(df["FISH_ID"].nunique()))
    (d / "bio_power.json").write_text(json.dumps(power, indent=2), encoding="utf-8")
    print(f"\nWith {power['n_fish']} fish, the smallest correlation detectable at "
          f"80% power is r = {power['min_detectable_r']:.3f} "
          f"(R² = {power['min_detectable_r_squared']:.3f}).")
    print(f"  Power to detect r = 0.30: {power['power_at_r_0.3']:.2f};  "
          f"r = 0.50: {power['power_at_r_0.5']:.2f}")

    print("\n" + "=" * 78)
    print("[4/4] Figures")
    figure_forest(primary, args.results_dir / "figures" / "fig_bio_forest.png")
    figure_significant(df, primary,
                       args.results_dir / "figures" / "fig_significant_associations.png")
    if len(valid):
        figure_validation(df, valid, args.results_dir / "figures" / "fig_bio_validation.png")
    print("  figures written")
    print(f"\nDone. Outputs in {d}")


if __name__ == "__main__":
    main()
