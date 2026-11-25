"""
Opercular melanin spot fading: complete statistical analysis and figures
========================================================================

Two-point photometrically normalised data (eye/ID-tag anchors, linear light).

Pipeline that produced the input CSVs
-------------------------------------
    linearise (inverse sRGB)  ->  two-point affine correction (L - Lb)/(Lw - Lb)
    ->  re-encode (forward sRGB)  ->  detect / register / segment
    ->  re-linearise the PIXELS  ->  average within mask  ->  per-spot descriptors
    ->  average per-spot descriptors within image


Primary outcome : skin-referenced log-ratio, ln(spot / background)
Secondary       : spot luminance, background luminance, spot area  (BH-adjusted, m = 3)
Derived         : Weber contrast. Reported for comparability with the pigmentation
                  literature; related to the log-ratio per spot as exp(R) - 1, so NOT an
                  independent test and no separate adjustment.

Expected input columns (per image)
----------------------------------
    Name, Spots, spot_lin, background_lin, contrast, log_ratio, Area, frac_clipped,
    contrast_glcm, Coorelation, Energy, Dissimilarity, Homogeneity

Inputs (in DATA_DIR, default all_data/data)
-------------------------------------------
    stressed/quantification_two_point/global_data.csv
    control/quantification_two_point/global_data.csv

Usage
-----
    python scripts/11_analysis/paper_3/opercular_analysis_complete_updated.py
        (defaults: --data all_data/data  --out all_data/results/paper_3)
    python opercular_analysis_complete_updated.py --data ./data --out ./results

Requires: numpy pandas scipy matplotlib   (statsmodels optional, for mixed models)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260803
N_BOOT = 10_000
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[3]

F_STRESS_tp_only = "stressed/quantification_two_point/global_data.csv"
F_CONTROL_tp_only = "control/quantification_two_point/global_data.csv"

RE_STRESS = r"fish_(?P<fish>\d+)_S_day(?P<day>\d)_id-\d+_(?P<side>[LR])"
RE_CONTROL = r"fish_(?P<fish>\d+)_C_(?P<side>[LR])"

FEATURES = [
    ("logratio", "Skin-referenced log-ratio", "primary",   "ln(spot/bg)"),
    ("spot_lin", "Spot luminance",            "secondary", "Luminance (rel.)"),
    ("bg_lin",   "Background luminance",      "secondary", "Luminance (rel.)"),
    ("Area",     "Spot area",                 "secondary", "Area (px)"),
]
WEBER = ("weber", "Weber contrast", "derived", "Contrast")

EXTRA = ["frac_clipped"]

G_REF, G_PRE, G_POST = "#3d3d3d", "#767676", "#cfcfcf"
BOXKW = dict(patch_artist=True, showfliers=False,
             medianprops=dict(color="k", lw=1.3),
             whiskerprops=dict(color="k", lw=.9),
             capprops=dict(color="k", lw=.9),
             boxprops=dict(edgecolor="k", lw=.9))


RENAME = {"background_lin": "bg_lin", "log_ratio": "logratio", "contrast": "weber"}


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the extraction-stage column names onto the names used here.

    No photometric transformation is applied. The values are already linear-light
    means of per-spot quantities:

      * applying the inverse sRGB function again would linearise twice;
      * recomputing logratio as ln(spot_lin / bg_lin) at image level would give the
        ratio of the means rather than the mean of the per-spot ratios. On this
        dataset the two differ by about 0.32 in the log-ratio, which is comparable
        to the entire confinement effect.
    """
    missing = [c for c in ("spot_lin", "background_lin", "log_ratio", "contrast", "Area")
               if c not in df.columns]
    if missing:
        raise KeyError("input is missing expected columns: %s. This script expects the "
                       "linear-light output of the extraction stage, not the older "
                       "*_grayscale columns." % missing)
    return df.rename(columns=RENAME)


def load_group(path: Path, pattern: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    p = d.Name.str.extract(pattern)
    d = pd.concat([d, p], axis=1)
    d["fish"] = d.fish.astype(int)
    if "day" in d.columns:
        d["day"] = d.day.astype(int)
    return add_descriptors(d)


def cluster_boot_ci(per_fish_values, nb=N_BOOT):
    """Bootstrap resampling FISH, not opercula (R1.2)."""
    n = len(per_fish_values)
    means = np.array([RNG.choice(per_fish_values, n, replace=True).mean()
                      for _ in range(nb)])
    return np.percentile(means, [2.5, 97.5])


def paired_summary(pre, post):
    d = post - pre
    n = len(d)
    t, p_t = stats.ttest_rel(post, pre)
    w, p_w = stats.wilcoxon(post, pre)
    dz = d.mean() / d.std(ddof=1)
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    lo, hi = cluster_boot_ci(d)
    se_dz = np.sqrt(1 / n + dz ** 2 / (2 * n))
    return dict(n=n,
                pre=pre.mean(), pre_sem=pre.std(ddof=1) / np.sqrt(n),
                post=post.mean(), post_sem=post.std(ddof=1) / np.sqrt(n),
                delta=d.mean(), ci_lo=d.mean() - q * se, ci_hi=d.mean() + q * se,
                t=t, p_t=p_t, W=w, p_wilcoxon=p_w, dz=dz,
                dz_lo=dz - q * se_dz, dz_hi=dz + q * se_dz,
                boot_lo=lo, boot_hi=hi)


def rm_anova_2x2(df, feat):
    """
    Fully within-subject 2x2 (Day x Side). With two levels per factor sphericity
    holds by construction, so no correction is applied.
    """
    t = df.pivot_table(index="fish", columns=["day", "side"], values=feat).dropna()
    n = len(t)
    Y = t.values.reshape(n, 2, 2)
    gm = Y.mean()
    res = {}
    for name, ax in (("Day", 1), ("Side", 2)):
        m = Y.mean(axis=2) if ax == 1 else Y.mean(axis=1)
        SS = n * 2 * ((m.mean(0) - gm) ** 2).sum()
        inter = m - m.mean(0, keepdims=True) - m.mean(1, keepdims=True) + gm
        SSe = 2 * (inter ** 2).sum()
        F = SS / (SSe / (n - 1))
        res[name] = (F, 1, n - 1, 1 - stats.f.cdf(F, 1, n - 1), SS / (SS + SSe))
    cell = Y.mean(0)
    SSi = n * ((cell - cell.mean(1, keepdims=True)
                - cell.mean(0, keepdims=True) + gm) ** 2).sum()
    resid = (Y - Y.mean(1, keepdims=True) - Y.mean(2, keepdims=True)
             + Y.mean((1, 2), keepdims=True) - cell
             + cell.mean(1, keepdims=True) + cell.mean(0, keepdims=True) - gm)
    F = SSi / ((resid ** 2).sum() / (n - 1))
    res["Day x Side"] = (F, 1, n - 1, 1 - stats.f.cdf(F, 1, n - 1), np.nan)
    return res, n


def bh(pvals):
    """Benjamini-Hochberg step-up adjustment."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        adj[i] = prev
    return adj


def tost(a, b, bound):
    """Two one-sided tests for equivalence within +/- bound (R3.4)."""
    d = a.mean() - b.mean()
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p1 = 1 - stats.t.cdf((d + bound) / se, df)
    p2 = stats.t.cdf((d - bound) / se, df)
    q90 = stats.t.ppf(0.95, df)
    return dict(diff=d, se=se, df=df, bound=bound,
                ci90_lo=d - q90 * se, ci90_hi=d + q90 * se,
                p_tost=max(p1, p2), equivalent=bool(max(p1, p2) < 0.05))


def mixed_model(S, feat):
    """feature ~ day * side + (1|fish)  [R2.5]. Requires statsmodels."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = S[["fish", "day", "side", feat]].dropna().copy()
    d.columns = ["fish", "day", "side", "y"]
    return smf.mixedlm("y ~ C(day) * C(side)", d, groups=d.fish).fit(reml=False)


def clipping_diagnostics(W):
    """
    Saturation is a quality flag, not an outcome. Clipped pixels carry no
    information about how much brighter the surface actually was, so an image with
    a large saturated fraction underestimates background luminance. If the
    saturated fraction changes between sessions, that underestimation does not
    cancel in the paired comparison.
    """
    if ("frac_clipped", 1) not in W.columns:
        return None
    pre = W[("frac_clipped", 1)].groupby(level=0).mean()
    post = W[("frac_clipped", 2)].groupby(level=0).mean()
    t, p = stats.ttest_rel(post.values, pre.values)
    rows = [dict(quantity="frac_clipped", pre=pre.mean(), post=post.mean(),
                 t=t, p=p, r=np.nan, p_r=np.nan)]
    dC = (post - pre).values
    for feat, label, _, _ in FEATURES:
        d = (W[(feat, 2)].groupby(level=0).mean()
             - W[(feat, 1)].groupby(level=0).mean()).values
        r, pr = stats.pearsonr(dC, d)
        rows.append(dict(quantity="delta_%s vs delta_clipping" % feat,
                         pre=np.nan, post=np.nan, t=np.nan, p=np.nan, r=r, p_r=pr))
    return pd.DataFrame(rows)


def pts(ax, x, y, s=20):
    """Open circles with a white halo, legible on both dark and light fills."""
    ax.scatter(x, y, s=s, facecolors="none", edgecolors="white", linewidths=1.8, zorder=3)
    ax.scatter(x, y, s=s, facecolors="none", edgecolors="0.15", linewidths=.7, zorder=4)


def stars(p):
    return "n.s." if p >= .05 else ("*" if p >= .01 else ("**" if p >= .001 else "***"))


def format_p_value(p, decimals=3):
    """Figure-friendly p-value formatting."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.{decimals}f}"


def paired_significance(row):
    """Fish-level paired t-test; BH correction for the 3 secondary endpoints."""
    p = float(row["p_bh"] if row["tier"] == "secondary" else row["p_t"])
    p_text = format_p_value(p)
    if row["tier"] == "secondary":
        p_text += " (BH)"
    return "%s; %s" % (stars(p), p_text)


def significance_bracket(ax, x1, x2, label, top_fraction=0.08):
    """Put a comparison bracket above plotted data, with room for its label."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    base = hi + span * top_fraction
    arm = span * 0.045
    ax.plot([x1, x1, x2, x2], [base, base + arm, base + arm, base],
            color="k", lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, base + arm + span * 0.025, label,
            ha="center", va="bottom", fontsize=7.5, clip_on=False)
    ax.set_ylim(lo, hi + span * (top_fraction + 0.23))


def side_brackets(ax, pos, labels=("Left", "Right")):
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - span * .14, hi)
    yl = lo - span * .10
    for (xa, xb), tx in zip([(pos[0], pos[1]), (pos[2], pos[3])], labels):
        ax.plot([xa - .3, xb + .3], [yl, yl], color="0.4", lw=1, clip_on=False)
        ax.text((xa + xb) / 2, yl + span * .015, tx, ha="center", va="bottom",
                fontsize=8.5, color="0.25")


def fig_trajectories(W, R, figs):
    """Fish-level paired trajectories, one panel per descriptor."""
    plot = [(f, l, y) for f, l, _, y in FEATURES]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, (feat, label, ylab) in zip(axes, plot):
        pre = W[(feat, 1)].groupby(level=0).mean().values
        post = W[(feat, 2)].groupby(level=0).mean().values
        for a, b in zip(pre, post):
            ax.plot([0, 1], [a, b], color="0.72", lw=.8, zorder=1)
        pts(ax, np.zeros(len(pre)), pre, 17)
        pts(ax, np.ones(len(post)), post, 17)
        for x, v, c in ((0, pre, G_PRE), (1, post, G_POST)):
            ax.errorbar(x + (0.2 if x else -0.2), v.mean(),
                        yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=8,
                        mfc=c, mec="k", mew=1.0, ecolor="k", capsize=4, lw=1.4, zorder=5)
        ax.set_xlim(-.45, 1.45)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre", "Post"])
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(label, fontsize=9)
        r = R.loc[R.key == feat].iloc[0]
        ax.text(.5, .02, "$d_z$=%.2f" % r.dz, transform=ax.transAxes,
                ha="center", fontsize=7.5, color="0.25")
        significance_bracket(ax, 0, 1, paired_significance(r))
    fig.suptitle("Within-fish change, pre- vs post-confinement (n = 21 fish)",
                 fontsize=10.5, y=1.04)
    fig.tight_layout(); fig.savefig(figs / "fig1_trajectories.png"); plt.close(fig)


def fig_forest(R, figs):
    """Standardised effects with 95% CI, all descriptors."""
    sub = R[R.tier.isin(("primary", "secondary", "derived"))]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ys = np.arange(len(sub))[::-1]
    for y, (_, r) in zip(ys, sub.iterrows()):
        ax.plot([r.dz_lo, r.dz_hi], [y, y], color="k", lw=1.5)
        ax.plot(r.dz, y, "o", ms=9, mfc=G_POST if r.dz > 0 else G_PRE,
                mec="k", mew=1.0, zorder=3)
        ax.text(r.dz_hi + .10, y, "%+.2f (%s)" % (r.dz, paired_significance(r)),
                va="center", ha="left", fontsize=7.5, color="0.25")
    ax.axvline(0, color="0.5", ls="--", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(sub.label, fontsize=8.5)
    ax.set_xlabel("Standardised within-fish effect ($d_z$, 95% CI), n = 21", fontsize=9)
    lo = min(sub.dz_lo.min(), 0) - 0.4
    hi = max(sub.dz_hi.max(), 0) + 2.6
    ax.set_xlim(lo, hi)
    ax.set_title("Within-fish change, pre- vs post-confinement", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "fig2_forest.png"); plt.close(fig)


def fig_boxplots(W, R, figs):
    """Distributions by session and side, operculum level.

    Significance is shown side-wise (Left pre vs Left post; Right pre vs Right post),
    because that is the meaningful repeated-measures comparison in this panel.
    """
    plot = [(f, l, y) for f, l, _, y in FEATURES]
    POS = [0, .75, 2.0, 2.75]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))

    sidewise_p = {}
    for side in ("L", "R"):
        raw = {}
        for feat, _, _, _ in FEATURES:
            side_df = W.xs(side, level=1)[feat].dropna()
            raw[feat] = float(stats.ttest_rel(side_df[2], side_df[1]).pvalue)
        corrected = bh([raw[f] for f in ("spot_lin", "bg_lin", "Area")])
        for feature, p_adj in zip(("spot_lin", "bg_lin", "Area"), corrected):
            raw[feature + "_bh"] = float(p_adj)
        sidewise_p[side] = raw

    for ax, (feat, label, ylab) in zip(axes, plot):
        data = [W.xs(s, level=1)[(feat, d)].values
                for s, d in (("L", 1), ("L", 2), ("R", 1), ("R", 2))]
        bp = ax.boxplot(data, positions=POS, widths=.6, **BOXKW)
        for patch, c in zip(bp["boxes"], (G_PRE, G_POST, G_PRE, G_POST)):
            patch.set_facecolor(c)
        for dd, pp in zip(data, POS):
            pts(ax, np.full(len(dd), pp) + RNG.normal(0, .06, len(dd)), dd, 19)
        ax.set_xticks(POS); ax.set_xticklabels(["Pre", "Post"] * 2, fontsize=8)
        ax.set_xlim(-.55, 3.3)
        ax.set_ylabel(ylab, fontsize=8.5); ax.set_title(label, fontsize=9)
        side_brackets(ax, POS)

        lo, hi = ax.get_ylim()
        sp = max(hi - lo, 1e-9)
        ax.set_ylim(lo, hi + sp * .30)
        hi = ax.get_ylim()[1]
        y = hi - sp * .18
        arm = sp * .04

        for (x1, x2, side, x_text) in ((POS[0], POS[1], "L", (POS[0] + POS[1]) / 2),
                                       (POS[2], POS[3], "R", (POS[2] + POS[3]) / 2)):
            is_secondary = feat in ("spot_lin", "bg_lin", "Area")
            p = sidewise_p[side][feat + "_bh"] if is_secondary else sidewise_p[side][feat]
            suffix = " (BH)" if is_secondary else ""
            label_txt = f"{stars(p)}; {format_p_value(p)}{suffix}"
            col = "k" if p < .05 else "0.55"
            ax.plot([x1, x1, x2, x2], [y, y + arm, y + arm, y], color=col, lw=0.9, clip_on=False)
            ax.text(x_text, y + arm + sp * .018, label_txt,
                    ha="center", va="bottom", fontsize=7.2, color=col, clip_on=False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=G_PRE, ec="k", lw=.9),
         plt.Rectangle((0, 0), 1, 1, fc=G_POST, ec="k", lw=.9)]
    fig.legend(h, ["Pre-confinement", "Post-confinement"], loc="upper right",
               ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(.99, 1.06))
    fig.suptitle("Distribution by session and side (opercula, n = 42)\n"
                 "Side-wise pre vs post paired tests shown separately for Left and Right",
                 fontsize=10.5, x=.42, y=1.08)
    fig.tight_layout(); fig.savefig(figs / "fig3_boxplots.png"); plt.close(fig)


def fig_fold_change(W, figs):
    """Spots lightened proportionally more than adjacent skin."""
    sp = (W[("spot_lin", 2)].groupby(level=0).mean()
          / W[("spot_lin", 1)].groupby(level=0).mean()).values
    bg = (W[("bg_lin", 2)].groupby(level=0).mean()
          / W[("bg_lin", 1)].groupby(level=0).mean()).values
    t_fc, p_fc = stats.ttest_rel(np.log(sp), np.log(bg))
    w_fc, pw_fc = stats.wilcoxon(np.log(sp), np.log(bg))
    ratio = sp / bg

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for a, b in zip(sp, bg):
        ax.plot([0, 1], [a, b], color="0.72", lw=.8, zorder=1)
    pts(ax, np.zeros(len(sp)), sp, 20)
    pts(ax, np.ones(len(bg)), bg, 20)
    for x, v, c in ((0, sp, G_REF), (1, bg, G_POST)):
        ax.errorbar(x + (0.2 if x else -0.2), v.mean(),
                    yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=8,
                    mfc=c, mec="k", mew=1.0, ecolor="k", capsize=4, lw=1.4, zorder=5)
    ax.axhline(1, color="0.7", ls=":", lw=1)
    ax.set_xlim(-.45, 1.45); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Spot", "Background"])
    ax.set_ylabel("Fold change in luminance (post / pre)", fontsize=9)
    ax.set_title("Proportional change\nt(%d)=%.2f, %s" % (len(sp) - 1, t_fc, format_p_value(p_fc)),
                 fontsize=9)

    ax = axes[1]
    ax.hist(ratio, bins=10, color=G_POST, edgecolor="k", lw=.8)
    ax.axvline(1, color="k", ls="--", lw=1.2)
    ax.axvline(ratio.mean(), color=G_REF, lw=2)
    ax.text(ratio.mean(), ax.get_ylim()[1] * .92, " mean %.2f" % ratio.mean(),
            fontsize=8.5, color=G_REF)
    ax.set_xlabel("Spot fold-change / background fold-change", fontsize=9)
    ax.set_ylabel("Fish", fontsize=9)
    ax.set_title("%d / %d fish above 1.0" % ((ratio > 1).sum(), len(ratio)), fontsize=9)

    fig.suptitle("Spots lightened proportionally more than adjacent skin (n = 21 fish)",
                 fontsize=10.5, y=1.04)
    fig.tight_layout(); fig.savefig(figs / "fig4_fold_change.png"); plt.close(fig)

    return dict(spot_fc=sp.mean(), spot_sem=sp.std(ddof=1) / np.sqrt(len(sp)),
                bg_fc=bg.mean(), bg_sem=bg.std(ddof=1) / np.sqrt(len(bg)),
                ratio=sp.mean() / bg.mean(), n_above_1=int((ratio > 1).sum()),
                n=len(sp), t=t_fc, p_t=p_fc, W=w_fc, p_wilcoxon=pw_fc)


def fig_two_point_reference_grid(W, C, R, figs):
    """
    Reference/pre/post summaries for every descriptor.

    Reference fish were imaged once, in a separate session and at a different
    working distance, so they are shown for context only and are not annotated.
    The pre/post comparison is paired by fish and is the basis for inference.
    """
    descriptors = FEATURES + [WEBER]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4))
    rows = []
    for ax, (feat, label, tier, _) in zip(axes.flat, descriptors):
        ref = C.groupby("fish")[feat].mean().dropna().values
        pre = W[(feat, 1)].groupby(level=0).mean()
        post = W[(feat, 2)].groupby(level=0).mean()
        paired = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna()
        groups = [ref, paired.pre.values, paired.post.values]
        bp = ax.boxplot(groups, positions=[0, 1, 2], widths=.55, **BOXKW)
        for patch, color in zip(bp["boxes"], (G_REF, G_PRE, G_POST)):
            patch.set_facecolor(color)
        for i, values in enumerate(groups):
            pts(ax, np.full(len(values), i) + RNG.normal(0, .055, len(values)), values, 18)

        p_ref_pre = float(stats.ttest_ind(ref, paired.pre.values, equal_var=False).pvalue)
        p_ref_post = float(stats.ttest_ind(ref, paired.post.values, equal_var=False).pvalue)
        r = R.loc[R.key == feat].iloc[0]
        p_paired = float(r.p_bh if tier == "secondary" else r.p_t)
        if p_ref_pre < .05 and p_ref_post >= .05:
            verdict = "resembles post"
        elif p_ref_post < .05 and p_ref_pre >= .05:
            verdict = "resembles pre"
        elif p_ref_pre < .05 and p_ref_post < .05:
            verdict = "differs from both"
        else:
            verdict = "differs from neither"
        rows.append(dict(descriptor=feat, p_ref_vs_pre=p_ref_pre,
                         p_ref_vs_post=p_ref_post, verdict=verdict,
                         p_pre_vs_post=r.p_t,
                         p_pre_vs_post_bh=(r.p_bh if tier == "secondary" else np.nan)))

        low = min(float(np.min(g)) for g in groups)
        high = max(float(np.max(g)) for g in groups)
        span = max(high - low, 1e-9)
        ax.set_ylim(low - .12 * span, high + .32 * span)

        y = high + .10 * span
        suffix = " (BH)" if tier == "secondary" else ""
        color = "k" if p_paired < .05 else "0.55"
        ax.plot([1, 1, 2, 2], [y, y + .035 * span, y + .035 * span, y],
                lw=.9, color=color)
        ax.text(1.5, y + .05 * span,
                "%s; %s%s" % (stars(p_paired), format_p_value(p_paired), suffix),
                ha="center", fontsize=7, color=color)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Reference", "Pre", "Post"], fontsize=8)
        ax.set_title(label, fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Two-point normalization: reference and paired pre/post measurements\n"
                 "Reference shown for context; annotations are paired fish-level pre/post tests",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(figs / "fig5_two_point_reference_grid.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"), help="directory containing the input CSVs")
    ap.add_argument("--out", default=str(ROOT / "results" / "paper_3"), help="output directory")
    a = ap.parse_args()

    data_dir = Path(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    figs = out / "figs"; figs.mkdir(exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300,
                         "savefig.bbox": "tight"})

    S = load_group(data_dir / F_STRESS_tp_only, RE_STRESS)
    C = load_group(data_dir / F_CONTROL_tp_only, RE_CONTROL)

    pivot_cols = ["logratio", "weber", "spot_lin", "bg_lin", "Area"]
    pivot_cols += [c for c in EXTRA if c in S.columns]
    W = S.pivot_table(index=["fish", "side"], columns="day", values=pivot_cols)

    print("=" * 94)
    print("PRIMARY ANALYSIS - FISH LEVEL (n = %d)" % S.fish.nunique())
    print("=" * 94)

    rows, sec_p = [], []
    for feat, label, tier, _ in FEATURES + [WEBER]:
        pre_f = W[(feat, 1)].groupby(level=0).mean().values
        post_f = W[(feat, 2)].groupby(level=0).mean().values
        r = paired_summary(pre_f, post_f)
        r.update(key=feat, label=label, tier=tier, level="fish")

        ro = paired_summary(W[(feat, 1)].values, W[(feat, 2)].values)
        r["dz_operculum"] = ro["dz"]; r["p_operculum"] = ro["p_t"]; r["n_operculum"] = ro["n"]

        aov, _ = rm_anova_2x2(S, feat)
        r.update(F_day=aov["Day"][0], p_day=aov["Day"][3], eta_day=aov["Day"][4],
                 F_side=aov["Side"][0], p_side=aov["Side"][3],
                 F_int=aov["Day x Side"][0], p_int=aov["Day x Side"][3], df2=aov["Day"][2])

        rows.append(r)
        if tier == "secondary":
            sec_p.append(r["p_t"])

        fm = "%.4f" if feat != "Area" else "%.0f"
        print("\n--- %s  [%s] ---" % (label, tier))
        print(("  pre  " + fm + " +/- " + fm + "    post " + fm + " +/- " + fm)
              % (r["pre"], r["pre_sem"], r["post"], r["post_sem"]))
        print(("  change " + fm + "  95%% CI [" + fm + ", " + fm + "]   d_z = %.2f [%.2f, %.2f]")
              % (r["delta"], r["ci_lo"], r["ci_hi"], r["dz"], r["dz_lo"], r["dz_hi"]))
        print("  paired t(%d) = %.2f, p = %.3g  |  Wilcoxon W = %.0f, p = %.3g"
              % (r["n"] - 1, r["t"], r["p_t"], r["W"], r["p_wilcoxon"]))
        print(("  cluster bootstrap 95%% CI [" + fm + ", " + fm + "]  (resampling fish)")
              % (r["boot_lo"], r["boot_hi"]))
        print("  RM-ANOVA  Day  F(1,%d) = %.2f, p = %.3g, eta2p = %.3f"
              % (r["df2"], r["F_day"], r["p_day"], r["eta_day"]))
        print("            Side F = %.2f, p = %.3f  |  Day x Side F = %.2f, p = %.3f"
              % (r["F_side"], r["p_side"], r["F_int"], r["p_int"]))
        print("  sensitivity, operculum level (n=%d): d_z = %.2f, p = %.3g"
              % (ro["n"], ro["dz"], ro["p_t"]))

    R = pd.DataFrame(rows)
    R.loc[R.tier == "secondary", "p_bh"] = bh(sec_p)

    rs, _ = stats.spearmanr(S.weber, S.logratio)
    print("\n" + "=" * 94)
    print("WEBER CONTRAST IS A TRANSFORM OF THE LOG-RATIO, NOT AN INDEPENDENT TEST")
    print("=" * 94)
    print("  Per spot, weber = exp(logratio) - 1, a strictly monotonic transform, so the")
    print("  two give identical rank orderings and the same non-parametric inference.")
    print("  Spearman r between the image-level means: %.4f" % rs)
    print("  The exact identity holds per spot, not after averaging within an image:")
    print("  a mean of exp(R) - 1 is not exp(mean R) - 1.")

    print("\n" + "=" * 94)
    print("SPOT vs BACKGROUND PROPORTIONAL CHANGE (fish level)")
    print("=" * 94)
    fc = fig_fold_change(W, figs)
    print("  spot       %.3f +/- %.3f SEM" % (fc["spot_fc"], fc["spot_sem"]))
    print("  background %.3f +/- %.3f SEM" % (fc["bg_fc"], fc["bg_sem"]))
    print("  ratio of fold changes %.3f  |  %d / %d fish above 1.0"
          % (fc["ratio"], fc["n_above_1"], fc["n"]))
    print("  paired t(%d) = %.2f, p = %.3g  |  Wilcoxon W = %.0f, p = %.3g"
          % (fc["n"] - 1, fc["t"], fc["p_t"], fc["W"], fc["p_wilcoxon"]))

    clip = clipping_diagnostics(W)
    if clip is not None:
        print("\n" + "=" * 94)
        print("SATURATION DIAGNOSTICS (quality flag, not an outcome)")
        print("=" * 94)
        head = clip.iloc[0]
        print("  saturated fraction: pre %.3f -> post %.3f, paired t = %.2f, p = %.3g"
              % (head.pre, head.post, head.t, head.p))
        for _, row in clip.iloc[1:].iterrows():
            flag = "   <-- significant" if row.p_r < .05 else ""
            print("  %-42s r = %+.3f, p = %.3f%s" % (row.quantity, row.r, row.p_r, flag))
        print("  Clipped pixels carry no information about how much brighter the surface")
        print("  actually was, so a saturated image underestimates background luminance.")

    print("\n" + "=" * 94)
    print("REFERENCE GROUP (cross-sectional: single timepoint, no Group x Time term)")
    print("=" * 94)
    ref_rows = []
    for feat, label in (("logratio", "Skin-referenced log-ratio"),
                        ("weber", "Weber contrast")):
        ref = C.groupby("fish")[feat].mean().values
        pre = W[(feat, 1)].groupby(level=0).mean().values
        post = W[(feat, 2)].groupby(level=0).mean().values
        print("\n  %s" % label)
        print("    reference   %+.4f +/- %.4f (n=%d)"
              % (ref.mean(), ref.std(ddof=1) / np.sqrt(len(ref)), len(ref)))
        for comp, nm in ((pre, "pre-stress"), (post, "post-stress")):
            t_, p_ = stats.ttest_ind(ref, comp, equal_var=False)
            _, pu = stats.mannwhitneyu(ref, comp)
            sp = np.sqrt(((len(ref) - 1) * ref.var(ddof=1)
                          + (len(comp) - 1) * comp.var(ddof=1))
                         / (len(ref) + len(comp) - 2))
            d = (ref.mean() - comp.mean()) / sp
            print("      vs %-12s diff %+.4f  Welch t = %+.2f, p = %.4f | MWU p = %.4f | d = %+.2f"
                  % (nm, ref.mean() - comp.mean(), t_, p_, pu, d))
            ref_rows.append(dict(feature=feat, comparison=nm, p_welch=p_, p_mwu=pu, d=d))
        eq = tost(ref, pre, abs(post.mean() - pre.mean()))
        print("      TOST vs pre-stress, bound = +/- %.4f (the confinement effect):"
              % eq["bound"])
        print("        90%% CI [%+.4f, %+.4f], p_TOST = %.4f -> %s"
              % (eq["ci90_lo"], eq["ci90_hi"], eq["p_tost"],
                 "EQUIVALENT" if eq["equivalent"] else "NOT equivalent"))
        ref_rows.append(dict(feature=feat, comparison="TOST vs pre-stress", **eq))
    print("\n  NOTE: reference fish were imaged once, in a separate session and at a")
    print("  different working distance. These comparisons are descriptive; see")
    print("  fig5 and two_point_reference_grid_tests.csv for all descriptors.")

    print("\n" + "=" * 94)
    print("MIXED-EFFECTS MODELS   feature ~ day * side + (1|fish)   [R2.5]")
    print("=" * 94)
    mm_rows = []
    for feat, label, _, _ in FEATURES:
        m = mixed_model(S, feat)
        if m is None:
            print("  statsmodels not installed - skipped")
            break
        co = m.params.index[1]
        print("  %-28s day coef = %+.5f, z = %.2f, p = %.3g"
              % (label, m.params[co], m.tvalues[co], m.pvalues[co]))
        mm_rows.append(dict(feature=feat, coef=m.params[co],
                            z=m.tvalues[co], p=m.pvalues[co]))

    fig_trajectories(W, R, figs)
    fig_forest(R, figs)
    fig_boxplots(W, R, figs)
    reference_fig_table = fig_two_point_reference_grid(W, C, R, figs)
    reference_fig_table.to_csv(out / "two_point_reference_grid_tests.csv", index=False)

    for obsolete in (
        "fig5_exposure_diagnostic.png", "fig6_reference_sensitivity.png",
        "fig7_within_fish_stability.png", "fig8_descriptor_instability.png",
        "fig9_reference_grid.png",
    ):
        (figs / obsolete).unlink(missing_ok=True)
    for obsolete in ("pipeline_sensitivity.csv", "descriptor_instability.csv"):
        (out / obsolete).unlink(missing_ok=True)

    R.to_csv(out / "results_table.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(out / "reference_table.csv", index=False)
    pd.DataFrame([fc]).to_csv(out / "fold_change_table.csv", index=False)
    if mm_rows:
        pd.DataFrame(mm_rows).to_csv(out / "mixed_models.csv", index=False)
    if clip is not None:
        clip.to_csv(out / "clipping_diagnostics.csv", index=False)

    (out / "_meta.json").write_text(json.dumps({
        "normalisation": "two-point affine on linear luminance (eye/ID-tag anchors); only pipeline",
        "measurement": "pixels linearised before averaging; ratios formed per spot",
        "primary_outcome": "logratio",
        "statistical_unit": "fish",
        "n_fish": int(S.fish.nunique()),
        "n_opercula": int(len(W)),
        "n_reference": int(C.fish.nunique()),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }, indent=2))

    print("\nwrote tables to %s and figures to %s" % (out, figs))


if __name__ == "__main__":
    main()