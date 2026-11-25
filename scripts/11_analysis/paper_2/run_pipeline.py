#!/usr/bin/env python3
"""
End-to-end descriptor extraction: images and labels in, every descriptor out.

Give it a directory of opercular images and a directory of spot polygon
labels, and it runs every stage in dependency order and writes the complete
descriptor set. Optional inputs unlock additional stages; if they are absent
those stages are skipped with a message rather than failing.

    python scripts/11_analysis/paper_2/run_pipeline.py      (all inputs default to _paths.py)

WHAT EACH STAGE OWNS
    If you need to change how a descriptor is computed, change the stage that
    owns it. This runner only sequences them.

    1  photometric_analysis.py           spot intensity, skin referencing, the
                                         annulus, exposure tracking, ICC
    2  segmentation_validation.py        test-split IoU, erosion sensitivity
                                         (needs --spots-dir and --weights-dir)
    3  grayscale_estimator_check.py      mean versus median sensitivity
    4  prior_study_reanalysis.py         pre/post confinement re-analysis
    5  spatial_normalisation.py          operculum geometry, relative units,
                                         the crop-convention audit
    6  spatial_2d_analysis.py            standardized opercular coordinates,
                                         the anterior-posterior gradient
    7  build_change_datasets.py          pre/post change at fish-side and fish level
    8  prepost_change_dataset.py         per fish-side change table with biometrics
    9  prepost_change_analysis.py        side agreement, baseline dependence
   10  change_vs_biometrics.py           covariates of individual change
   11  feature_relationships.py          descriptor correlations, decomposition
   12  identity_vs_state.py              repeatability, re-identification
   13  biological_validation.py          biometric associations
   14  integrated_melanization_figure.py integrated melanization figure
   15  fish_spot_pipeline.py             legacy single-file pipeline, written to
                                         <output-dir>/legacy_fish_spot_pipeline/

    Use --only to run a subset, e.g. --only change, or --no-legacy to leave
    the legacy pipeline out.

INPUTS
    REQUIRED
      --images-dir     opercular images, one per fish-side-session. Filenames
                       must carry the "id-NN" token and the fish/day/side
                       fields the pipeline parses.
      --labels-dir     YOLO-format polygon labels for spots, one .txt per
                       image, same stem as the image.

    OPTIONAL, each unlocking further stages
      --masks-dir      operculum masks. Without these, stages 5-13 and every
                       relative-area or spatial descriptor are skipped.
      --biometrics     reference_tabular_data.csv. Without it, stages 8-10
                       and 13 are skipped and stage 7 loses its covariates.
      --spots-dir      annotated spot dataset (spots/test/...) and
      --weights-dir    trained checkpoints. Both are needed for stage 2.
      --originals-dir  uncropped camera frames retaining the blue board and
                       EXIF. Without them the exposure proxy is unavailable.
                       Defaults to the images directory.

OUTPUT
    Everything lands under --output-dir (default results/paper_2/), with
    CSVs in data/ and figures in figures/. A PIPELINE_LOG.txt records which
    stages ran, which were skipped and why, and how long each took.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _paths as P

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


class Stage:
    """One pipeline step, with the inputs it needs and the files it writes."""

    def __init__(self, name: str, script: str, args: list[str],
                 needs: list[str], produces: list[str], note: str = ""):
        self.name = name
        self.script = script
        self.args = args
        self.needs = needs
        self.produces = produces
        self.note = note


def build_stages(a: argparse.Namespace) -> list[Stage]:
    """The stage list, in dependency order."""
    img = ["--images-dir", str(a.images_dir), "--labels-dir", str(a.labels_dir)]
    out = ["--output-dir", str(a.output_dir)]
    res = ["--results-dir", str(a.output_dir)]
    masks = ["--masks-dir", str(a.masks_dir)] if a.masks_dir else []
    bio = ["--biometrics", str(a.biometrics)] if a.biometrics else []
    orig = ["--originals-dir", str(a.originals_dir)] if a.originals_dir else []

    data = a.output_dir / "data"
    figs = a.output_dir / "figures"
    change_io = ["--data-dir", str(data), "--fig-dir", str(figs)]

    stages = [
        Stage("Photometric descriptors", "photometric_analysis.py",
              img + orig + out
              + ["--annulus-gap", str(a.annulus_gap),
                 "--annulus-width", str(a.annulus_width)]
              + (["--skip-figures"] if a.skip_figures else []),
              needs=[],
              produces=["spot_level_skin_referenced.csv", "image_level_measures.csv",
                        "image_level_with_exposure.csv"],
              note="spot intensity, skin referencing, exposure, ICC"),

        Stage("Segmentation validation", "segmentation_validation.py",
              img + out
              + ["--spots-dir", str(a.spots_dir), "--weights-dir", str(a.weights_dir),
                 "--device", a.device],
              needs=["spots_dir", "weights_dir"],
              produces=["iou_by_spot_size.csv", "erosion_spotwise.csv"],
              note="test-split IoU and erosion sensitivity"),

        Stage("Grayscale estimator check", "grayscale_estimator_check.py", res,
              needs=[],
              produces=["grayscale_mean_vs_median.json",
                        "grayscale_statistic_downstream.csv"],
              note="mean versus median sensitivity"),

        Stage("Prior study re-analysis", "prior_study_reanalysis.py", res,
              needs=[],
              produces=["prepost_confinement_reanalysis.csv",
                        "prepost_exposure_adjusted.csv"],
              note="pre/post confinement with the exposure-robust descriptor"),

        Stage("Spatial normalisation", "spatial_normalisation.py",
              img + masks + bio + out,
              needs=["masks_dir"],
              produces=["spatial_operculum_geometry.csv", "spatial_crop_convention.json"],
              note="operculum geometry and relative units"),

        Stage("2D coordinate system", "spatial_2d_analysis.py",
              img + masks + out,
              needs=["masks_dir"],
              produces=["spatial2d_spotwise.csv", "spatial2d_models.csv"],
              note="standardized opercular frame"),

        Stage("Change datasets", "build_change_datasets.py",
              ["--data-dir", str(a.output_dir / "data")]
              + (["--biometrics", str(a.biometrics)] if a.biometrics else [])
              + ["--out-dir", str(a.change_dir or (a.output_dir / "data"))],
              needs=["masks_dir"],
              produces=["descriptors_long.csv", "change_by_fish_side.csv",
                        "change_by_fish.csv"],
              note="pre/post change at fish-side and fish level"),

        Stage("Pre/post change table", "prepost_change_dataset.py", res,
              needs=["masks_dir", "biometrics"],
              produces=["prepost_change_with_biometrics.csv",
                        "prepost_change_by_fish.csv"],
              note="per fish-side change with biometrics"),

        Stage("Pre/post change analysis", "prepost_change_analysis.py", change_io,
              needs=["masks_dir", "biometrics"],
              produces=["change_side_agreement.csv", "change_baseline_dependence.csv"],
              note="side agreement and baseline dependence"),

        Stage("Change versus biometrics", "change_vs_biometrics.py", change_io,
              needs=["masks_dir", "biometrics"],
              produces=["change_vs_biometrics.csv"],
              note="covariates of individual change"),

        Stage("Descriptor relationships", "feature_relationships.py", res,
              needs=["masks_dir"],
              produces=["features_corr_image.csv", "features_vif_image.csv"],
              note="correlations, decomposition, collinearity"),

        Stage("Identity versus state", "identity_vs_state.py", res,
              needs=["masks_dir"],
              produces=["identity_stability_table.csv",
                        "identity_reidentification.csv"],
              note="repeatability and re-identification"),

        Stage("Biological associations", "biological_validation.py",
              res + bio,
              needs=["masks_dir", "biometrics"],
              produces=["bio_primary_associations.csv"],
              note="associations with body size, condition, sex, maturation"),

        Stage("Integrated melanization figure", "integrated_melanization_figure.py",
              res,
              needs=[],
              produces=["fig_integrated_melanization.png",
                        "integrated_melanization_summary.json"],
              note="integrated melanization figure"),
    ]

    if not a.no_legacy:
        stages.append(
            Stage("Legacy fish spot pipeline", "fish_spot_pipeline.py",
                  img + ["--output-dir", str(a.output_dir / "legacy_fish_spot_pipeline")]
                  + (["--skip-distance-plots", "--skip-summary-plots"]
                     if a.skip_figures else []),
                  needs=[],
                  produces=["legacy_fish_spot_pipeline/spotwise_data.csv"],
                  note="original single-file pipeline"))

    return stages


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=P.IMAGES_DIR,
                    help="REQUIRED in practice: opercular images")
    ap.add_argument("--labels-dir", type=Path, default=P.LABELS_DIR,
                    help="REQUIRED in practice: spot polygon labels")
    ap.add_argument("--masks-dir", type=Path, default=P.OPERCULUM_MASKS_DIR,
                    help="operculum masks; omit to skip the spatial stages")
    ap.add_argument("--biometrics", type=Path, default=P.BIOMETRICS_CSV,
                    help="reference_tabular_data.csv; omit to skip covariates")
    ap.add_argument("--originals-dir", type=Path, default=P.ORIGINALS_DIR,
                    help="uncropped frames for the board exposure proxy")
    ap.add_argument("--output-dir", type=Path, default=P.RESULTS_DIR)
    ap.add_argument("--change-dir", type=Path, default=None,
                    help="where the change datasets go; defaults inside output-dir")
    ap.add_argument("--spots-dir", type=Path, default=P.SPOTS_DATASET_DIR,
                    help="annotated spot dataset with test/ split; "
                         "omit to skip segmentation validation")
    ap.add_argument("--weights-dir", type=Path, default=P.WEIGHTS_DIR,
                    help="trained YOLO and SAM checkpoints; "
                         "omit to skip segmentation validation")
    ap.add_argument("--device", default="cuda",
                    help="device for segmentation validation (falls back to cpu)")
    ap.add_argument("--no-legacy", action="store_true",
                    help="do not run the legacy fish_spot_pipeline.py")
    ap.add_argument("--annulus-gap", type=float, default=4.0)
    ap.add_argument("--annulus-width", type=float, default=14.0)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only stages whose name contains one of these strings")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    a = ap.parse_args()

    for label, path in (("--images-dir", a.images_dir), ("--labels-dir", a.labels_dir)):
        if not path or not Path(path).is_dir():
            ap.error(f"{label} is required and must be a directory: {path}")

    n_img = len([p for p in a.images_dir.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    n_lab = len(list(a.labels_dir.glob("*.txt")))
    if n_img == 0:
        ap.error(f"no images found in {a.images_dir}")
    if n_lab == 0:
        ap.error(f"no .txt label files found in {a.labels_dir}")

    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_files = sorted(f for f in a.images_dir.iterdir() if f.suffix.lower() in exts)
    img_sizes = {Image.open(f).size for f in img_files[:8]}

    if a.masks_dir and Path(a.masks_dir).is_dir():
        mask_files = sorted(f for f in Path(a.masks_dir).iterdir()
                            if f.suffix.lower() in exts)
        if mask_files:
            mask_sizes = {Image.open(f).size for f in mask_files[:8]}
            if img_sizes.isdisjoint(mask_sizes):
                ap.error(
                    f"image size {img_sizes} does not match mask size "
                    f"{mask_sizes}. --images-dir must be the cropped opercular "
                    f"images the masks were drawn on; full head frames belong "
                    f"in --originals-dir, where they are used only for the "
                    f"exposure proxy and EXIF.")

    label_stems = {f.stem for f in a.labels_dir.glob("*.txt")}
    missing_labels = [f.name for f in img_files if f.stem not in label_stems]
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} image(s) have no label file, "
              f"e.g. {missing_labels[0]}")

    available = {
        "masks_dir": bool(a.masks_dir and Path(a.masks_dir).is_dir()),
        "biometrics": bool(a.biometrics and Path(a.biometrics).is_file()),
        "originals_dir": bool(a.originals_dir and Path(a.originals_dir).is_dir()),
        "spots_dir": bool(a.spots_dir and (Path(a.spots_dir) / "test").is_dir()),
        "weights_dir": bool(a.weights_dir and Path(a.weights_dir).is_dir()),
    }
    for k in ("masks_dir", "biometrics", "originals_dir", "spots_dir", "weights_dir"):
        if not available[k]:
            setattr(a, k, None)

    (a.output_dir / "data").mkdir(parents=True, exist_ok=True)
    (a.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PIPELINE")
    print("=" * 72)
    print(f"  images       {a.images_dir}   ({n_img} files)")
    print(f"  labels       {a.labels_dir}   ({n_lab} files)")
    for key, label in (("masks_dir", "masks       "),
                       ("biometrics", "biometrics  "),
                       ("originals_dir", "originals   "),
                       ("spots_dir", "spots data  "),
                       ("weights_dir", "weights     ")):
        val = getattr(a, key)
        print(f"  {label} {val if val else '-- absent, dependent stages will be skipped'}")
    print(f"  output       {a.output_dir}")
    print(f"  annulus      gap {a.annulus_gap} px, width {a.annulus_width} px")
    print()

    stages = build_stages(a)
    if a.only:
        stages = [s for s in stages
                  if any(k.lower() in s.name.lower() for k in a.only)]

    log = [f"PIPELINE LOG   {datetime.now():%Y-%m-%d %H:%M}",
           f"images     {a.images_dir}  ({n_img} files)",
           f"labels     {a.labels_dir}  ({n_lab} files)",
           f"masks      {a.masks_dir}",
           f"biometrics {a.biometrics}",
           f"originals  {a.originals_dir}",
           f"annulus    gap {a.annulus_gap}, width {a.annulus_width}",
           ""]

    failed = []
    for i, st in enumerate(stages, start=1):
        missing = [n for n in st.needs if not available.get(n, False)]
        head = f"[{i}/{len(stages)}] {st.name}"
        if missing:
            msg = f"SKIPPED, needs {', '.join('--' + m.replace('_', '-') for m in missing)}"
            print(f"{head:44s} {msg}")
            log.append(f"SKIP  {st.name}: {msg}")
            continue

        cmd = [_sys.executable, str(SCRIPTS / st.script), *st.args]
        if a.dry_run:
            print(f"{head:44s} DRY RUN")
            print("        " + " ".join(cmd))
            log.append(f"DRY   {st.name}: {' '.join(cmd)}")
            continue

        print(f"{head:44s} running ... ", end="", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(P.REPO))
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"ok  ({dt:.1f}s)")
            log.append(f"OK    {st.name}  {dt:.1f}s  -> {', '.join(st.produces)}")
        else:
            print(f"FAILED  ({dt:.1f}s)")
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            for line in tail:
                print("        " + line)
            log.append(f"FAIL  {st.name}\n" + "\n".join("        " + t for t in tail))
            failed.append(st.name)

    print()
    if failed:
        print(f"{len(failed)} stage(s) failed: {', '.join(failed)}")
    else:
        print("all requested stages completed")

    data = a.output_dir / "data"
    figs = a.output_dir / "figures"
    print(f"  {len(list(data.glob('*.csv')))} CSVs and "
          f"{len(list(data.glob('*.json')))} JSON in {data}")
    print(f"  {len(list(figs.glob('*.png')))} figures in {figs}")

    log_path = a.output_dir / "PIPELINE_LOG.txt"
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"  log written to {log_path}")
    _sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
