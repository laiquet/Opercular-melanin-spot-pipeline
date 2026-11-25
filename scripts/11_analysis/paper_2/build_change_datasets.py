#!/usr/bin/env python3
"""
Build the descriptor and change datasets at fish-side and fish level.

This is the single entry point for the pre/post change analysis, written so
that the provenance of every column can be audited. It supersedes
prepost_change_dataset.py, which pulled biometrics from the wrong table.

WHAT IT PRODUCES
    <out>/descriptors_long.csv        84 rows,  one per image
    <out>/change_by_fish_side.csv     42 rows,  one per fish-by-side
    <out>/change_by_fish.csv          21 rows,  one per fish
    <out>/PROVENANCE.txt              audit trail of every input and column

WHERE EVERY NUMBER COMES FROM

    reference_tabular_data.csv                  ......... RAW INPUT
        The only non-derived input. One row per fish, 21 rows. Supplies
        fork length (lengde), body mass (vekt), condition factor (K),
        sex (Kjonn), maturation (Kjonnsmodning), the visual fading score
        (fading), and the welfare scores. The two fish-number columns
        ("fiskenr dag 1", "fiskenr dag 2, post-stress") are what link a fish
        to its images: those numbers appear as the "id-NN" token in filenames.

    results/paper_2/data/image_level_measures.csv     from photometric_analysis.py
        spot_count, relative_spot_area_pct, total_melanization,
        raw_mean_intensity, skin_ref_log_ratio_linear, skin_ref_ratio,
        image_skin_median. Computed from head_images/ and operculum_masks/.

    results/paper_2/data/image_level_with_exposure.csv from photometric_analysis.py
        board_luma only, the independent exposure proxy measured on the blue
        background board in the uncropped camera frames.

    results/paper_2/data/spatial_operculum_geometry.csv from spatial_normalisation.py
        rel_spot_area_pct only, used to derive mean relative spot area.
        NOTE this uses the operculum-mask denominator, whereas
        relative_spot_area_pct uses the tissue-mask denominator. The two masks
        agree at IoU 0.969 but are not identical.

NOTHING here is recomputed from images. Re-running photometric_analysis.py or
spatial_normalisation.py changes the inputs; re-running this script alone does
not.

Run:
    python build_change_datasets.py
    python build_change_datasets.py --out-dir results/paper_2/data/oyvind_sugg
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

PCT_CHANGE_OK = {c for c in DESCRIPTORS.values()
                 if c != "skin_ref_log_ratio"}

COVARIATES = ["condition_factor", "fork_length_cm", "body_mass_g",
              "sex", "maturation", "fading_score"]

RAW_COLUMNS = {
    "lengde": "fork_length_cm",
    "vekt": "body_mass_g",
    "K": "condition_factor",
    "Kjønn": "sex",
    "Kjønnsmodning": "maturation",
    "fading": "fading_score",
}
COL_DAY1 = "fiskenr dag 1"
COL_DAY2 = "fiskenr dag 2, post-stress"


def image_id_from_sample(sample: str) -> int | None:
    """The 'id-NN' token in a filename is the day-specific fish number."""
    m = re.search(r"id-(\d+)", sample)
    return int(m.group(1)) if m else None


def load_covariates(path: Path) -> pd.DataFrame:
    """
    Long-form per-image covariates from the raw biometric table.

    One row per (image_id, DAY). A fish carries a different number on day 1
    and day 2, so each fish contributes two rows keyed by its two numbers,
    both holding the same per-fish values.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]

    rows = []
    for _, r in raw.iterrows():
        for col, day in ((COL_DAY1, PRE), (COL_DAY2, POST)):
            if pd.isna(r.get(col)):
                continue
            row = {"image_id": int(r[col]), "DAY": day}
            for src, dst in RAW_COLUMNS.items():
                v = r.get(src)
                row[dst] = (str(v).strip() if dst == "sex" and pd.notna(v)
                            else pd.to_numeric(v, errors="coerce"))
            rows.append(row)
    return pd.DataFrame(rows)


def build_long(data_dir: Path, biometrics_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """One row per image, descriptors joined to covariates. Returns the audit log."""
    log = []

    img = pd.read_csv(data_dir / "image_level_measures.csv",
                      dtype={"FISH_ID": str, "DAY": str, "SIDE": str})
    log.append(f"image_level_measures.csv          {len(img):3d} rows, "
               f"{img['FISH_ID'].nunique()} fish")

    geom = pd.read_csv(data_dir / "spatial_operculum_geometry.csv", dtype={"DAY": str})
    img = img.merge(geom[["sample", "rel_spot_area_pct"]], on="sample", how="left")
    img["mean_rel_spot_area_pct"] = img["rel_spot_area_pct"] / img["spot_count"]
    log.append(f"spatial_operculum_geometry.csv    {len(geom):3d} rows, "
               f"matched {img['rel_spot_area_pct'].notna().sum()}")

    expo = pd.read_csv(data_dir / "image_level_with_exposure.csv")
    img = img.merge(expo[["sample", "board_luma"]], on="sample", how="left")
    log.append(f"image_level_with_exposure.csv     {len(expo):3d} rows, "
               f"matched {img['board_luma'].notna().sum()}")

    cov = load_covariates(biometrics_csv)
    img["image_id"] = img["sample"].map(image_id_from_sample)
    img = img.merge(cov, on=["image_id", "DAY"], how="left")
    log.append(f"reference_tabular_data.csv        {len(cov):3d} keyed rows, "
               f"matched {img['condition_factor'].notna().sum()} of {len(img)} images")

    img["unit"] = img["FISH_ID"].astype(str) + "-" + img["SIDE"].astype(str)
    keep = (["sample", "unit", "FISH_ID", "SIDE", "DAY", "image_id"]
            + COVARIATES + list(DESCRIPTORS))
    return img[[c for c in keep if c in img.columns]], log


def build_change(long: pd.DataFrame) -> pd.DataFrame:
    """One row per fish-side: pre, post, change and relative change."""
    rows = []
    for unit, g in long.groupby("unit"):
        pre, post = g[g["DAY"] == PRE], g[g["DAY"] == POST]
        if len(pre) != 1 or len(post) != 1:
            continue
        pre, post = pre.iloc[0], post.iloc[0]

        row = {"unit": unit, "FISH_ID": pre["FISH_ID"], "SIDE": pre["SIDE"]}
        for c in COVARIATES:
            row[c] = pre.get(c, np.nan)

        for src, name in DESCRIPTORS.items():
            a, z = pre.get(src, np.nan), post.get(src, np.nan)
            row[f"{name}_pre"], row[f"{name}_post"] = a, z
            row[f"{name}_change"] = z - a
            if name in PCT_CHANGE_OK:
                row[f"{name}_pct_change"] = (
                    100.0 * (z - a) / a if pd.notna(a) and a != 0 else np.nan)

        d = row.get("skin_ref_log_ratio_change")
        row["skin_ref_log_ratio_rel_change_pct"] = (
            100.0 * (np.exp(d) - 1) if pd.notna(d) else np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["FISH_ID", "SIDE"]).reset_index(drop=True)


def write_provenance(path: Path, log: list[str], long: pd.DataFrame,
                     side: pd.DataFrame, fish: pd.DataFrame,
                     data_dir: Path, biometrics_csv: Path) -> None:
    lines = [
        "PROVENANCE  --  change datasets",
        f"generated  {datetime.now():%Y-%m-%d %H:%M}",
        f"by         scripts/build_change_datasets.py",
        "",
        "INPUTS READ",
        f"  {biometrics_csv}",
        f"  {data_dir / 'image_level_measures.csv'}",
        f"  {data_dir / 'image_level_with_exposure.csv'}",
        f"  {data_dir / 'spatial_operculum_geometry.csv'}",
        "",
        "MERGE AUDIT",
        *[f"  {line}" for line in log],
        "",
        "OUTPUTS",
        f"  descriptors_long.csv       {len(long):3d} rows  (one per image)",
        f"  change_by_fish_side.csv    {len(side):3d} rows  (one per fish-side)",
        f"  change_by_fish.csv         {len(fish):3d} rows  (one per fish)",
        "",
        "COLUMN PROVENANCE",
        "  spot_count, relative_spot_area_pct, integrated_melanization,",
        "  raw_mean_intensity, skin_ref_log_ratio, skin_ref_ratio,",
        "  opercular_skin_intensity          <- image_level_measures.csv",
        "  board_luma                        <- image_level_with_exposure.csv",
        "  mean_rel_spot_area_pct            <- rel_spot_area_pct (geometry) / spot_count",
        "  condition_factor, fork_length_cm, body_mass_g, sex, maturation,",
        "  fading_score                      <- reference_tabular_data.csv",
        "",
        "DERIVED COLUMNS",
        "  <name>_change            = post - pre",
        "  <name>_pct_change        = 100 * (post - pre) / pre",
        "                             blank for skin_ref_log_ratio: its zero point",
        "                             is arbitrary, so a percentage has no meaning",
        "  skin_ref_log_ratio_rel_change_pct = 100 * (exp(change) - 1)",
        "                             the proportional change in the spot-to-skin",
        "                             ratio, which IS interpretable",
        "",
        "CONVENTIONS",
        "  change = post - pre.",
        "  For skin_ref_log_ratio, more negative means darker, so a POSITIVE",
        "  change means the spots lightened.",
        "  change_by_fish.csv averages the two sides with equal weight, even",
        "  though the sides may carry different spot counts.",
        "  The 42 fish-side rows are NOT independent: two per fish.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=P.DATA_DIR,
                    help="where the upstream pipeline CSVs live")
    ap.add_argument("--biometrics", type=Path, default=P.BIOMETRICS_CSV)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to --data-dir")
    args = ap.parse_args()
    out = args.out_dir or args.data_dir
    out.mkdir(parents=True, exist_ok=True)

    long, log = build_long(args.data_dir, args.biometrics)
    side = build_change(long)
    num = side.select_dtypes(include=[np.number]).columns.tolist()
    fish = side.groupby("FISH_ID", as_index=False)[num].mean()
    fish["sex"] = side.groupby("FISH_ID")["sex"].first().values

    long.to_csv(out / "descriptors_long.csv", index=False)
    side.to_csv(out / "change_by_fish_side.csv", index=False)
    fish.to_csv(out / "change_by_fish.csv", index=False)
    write_provenance(out / "PROVENANCE.txt", log, long, side, fish,
                     args.data_dir, args.biometrics)

    print("MERGE AUDIT")
    for line in log:
        print("  " + line)
    print(f"\nwrote to {out}")
    for name, df in (("descriptors_long.csv", long),
                     ("change_by_fish_side.csv", side),
                     ("change_by_fish.csv", fish)):
        print(f"  {name:26s} {len(df):3d} rows x {len(df.columns):3d} cols")
    print("  PROVENANCE.txt")

    missing = [c for c in COVARIATES if side[c].isna().all()]
    print("\ncovariate check:",
          "all populated" if not missing else f"EMPTY -> {missing}")


if __name__ == "__main__":
    main()
