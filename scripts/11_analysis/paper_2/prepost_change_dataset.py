#!/usr/bin/env python3
"""
Pre- to post-confinement change per fish-by-side unit, with biometrics.

Builds one row per fish-by-side (42 units), holding each descriptor's
pre-stress value, post-stress value and the paired change, alongside the
fish's condition factor and the other biometrics. The unit is the fish-side
rather than the fish because both sides were imaged on both occasions, and
the paired change is only defined within a side.

The exposure proxy (blue-board luma) is carried through as a control: a
change in a descriptor is only interpretable as biology if the board did not
also change for that unit.

Sign convention: change = post - pre. For the skin-referenced log-ratio, more
negative means darker, so a POSITIVE change means the spots lightened.

Run:
    python prepost_change_dataset.py --results-dir results/paper_2

Datasets
--------
Reads   results/paper_2/data/image_level_measures.csv      (script 1)
        results/paper_2/data/image_level_with_exposure.csv (script 1)
        results/paper_2/data/spatial_operculum_geometry.csv(script 5)
        reference_tabular_data.csv                             biometrics, n = 21
Writes  results/paper_2/data/prepost_change_with_biometrics.csv
        results/paper_2/data/prepost_change_by_fish.csv
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

from biological_validation import load_biometrics

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRE, POST = "1", "2"

DESCRIPTORS = {
    "spot_count": "spot_count",
    "relative_spot_area_pct": "relative_spot_area_pct",
    "mean_rel_spot_area_pct": "mean_rel_spot_area_pct",
    "total_melanization": "integrated_melanization",
    "raw_mean_intensity": "raw_mean_intensity",
    "skin_ref_log_ratio_linear": "skin_ref_log_ratio",
    "skin_ref_ratio": "skin_ref_ratio",
    "image_skin_median": "opercular_skin_intensity",
    "board_luma": "board_luma",
}

BIOMETRICS = ["condition_factor", "fork_length_cm", "body_mass_g",
              "sex", "maturation", "fading_score"]


def image_id_from_sample(sample: str) -> int | None:
    m = re.search(r"id-(\d+)", sample)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=P.RESULTS_DIR)
    args = ap.parse_args()
    d = args.results_dir / "data"

    img = pd.read_csv(d / "image_level_measures.csv",
                      dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    geom = pd.read_csv(d / "spatial_operculum_geometry.csv", dtype={"DAY": str})
    img = img.merge(geom[["sample", "rel_spot_area_pct"]], on="sample", how="left")
    img["mean_rel_spot_area_pct"] = img["rel_spot_area_pct"] / img["spot_count"]

    expo = pd.read_csv(d / "image_level_with_exposure.csv")
    img = img.merge(expo[["sample", "board_luma"]], on="sample", how="left")

    bio, _ = load_biometrics(P.BIOMETRICS_CSV)
    bio["DAY"] = bio["DAY"].astype(str)
    img["image_id"] = img["sample"].map(image_id_from_sample)
    img = img.merge(bio, on=["image_id", "DAY"], how="left")

    img["unit"] = img["FISH_ID"].astype(str) + "-" + img["SIDE"].astype(str)

    rows = []
    for unit, g in img.groupby("unit"):
        pre = g[g["DAY"] == PRE]
        post = g[g["DAY"] == POST]
        if len(pre) != 1 or len(post) != 1:
            continue
        pre, post = pre.iloc[0], post.iloc[0]

        row = {"unit": unit, "FISH_ID": pre["FISH_ID"], "SIDE": pre["SIDE"]}
        for b in BIOMETRICS:
            row[b] = pre.get(b, np.nan)
        for col, label in DESCRIPTORS.items():
            a, z = pre.get(col, np.nan), post.get(col, np.nan)
            row[f"{label}_pre"] = a
            row[f"{label}_post"] = z
            row[f"{label}_change"] = z - a
            row[f"{label}_pct_change"] = (
                100.0 * (z - a) / a if label != "skin_ref_log_ratio"
                and pd.notna(a) and a != 0 else np.nan)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["FISH_ID", "SIDE"]).reset_index(drop=True)
    path = d / "prepost_change_with_biometrics.csv"
    out.to_csv(path, index=False)

    num = out.select_dtypes(include=[np.number]).columns
    by_fish = out.groupby("FISH_ID", as_index=False)[list(num)].mean()
    by_fish.to_csv(d / "prepost_change_by_fish.csv", index=False)

    print(f"wrote {path.name}  ({len(out)} fish-side units, {out.FISH_ID.nunique()} fish)")
    print(f"wrote prepost_change_by_fish.csv  ({len(by_fish)} fish)\n")

    print("Mean paired change (post - pre), 42 fish-side units:")
    for label in DESCRIPTORS.values():
        c = out[f"{label}_change"].dropna()
        t, p = stats.ttest_1samp(c, 0)
        print(f"  {label:26s} {c.mean():+10.4f}   dz {c.mean()/c.std():+6.2f}   p {p:.4f}")

    print("\nDoes the change relate to condition factor? (fish level, n = %d)"
          % by_fish["condition_factor"].notna().sum())
    for label in DESCRIPTORS.values():
        s = by_fish[["condition_factor", f"{label}_change"]].dropna()
        if len(s) > 3:
            r, p = stats.pearsonr(s["condition_factor"], s[f"{label}_change"])
            flag = "  <-- p < 0.05" if p < 0.05 else ""
            print(f"  {label:26s} r = {r:+.3f}, p = {p:.4f}{flag}")


if __name__ == "__main__":
    main()
