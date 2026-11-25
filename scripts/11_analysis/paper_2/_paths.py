"""
Canonical locations of every dataset this analysis reads or writes.

All analysis scripts import their default paths from here, so the pipeline can
be run from any working directory and there is a single place to look up where
a given dataset lives. Paths are resolved relative to the project root
(``all_data/``), which is three levels above this file
(``all_data/scripts/11_analysis/paper_2/_paths.py``).

Layout (consolidated project)
-----------------------------
    all_data/
      data/reference_tabular_data.csv               biometrics, 21 fish    [INPUT]
      data/stressed/images/operculum_images_roboflow/  84 measurement crops [INPUT]
      data/stressed/labels/spots_roboflow/          84 spot label files     [INPUT]
      data/stressed/images/operculum_masks/         84 operculum masks      [INPUT]
      data/stressed/images/head_images_exif/        84 original frames+EXIF [INPUT]
      datasets/spots_paper2/{train,valid,test}/     annotated spot dataset  [INPUT]
      model_weights/                                trained checkpoints     [INPUT]
      results/paper_2/
        data/                                       all generated tables    [OUTPUT]
        figures/                                    all generated figures   [OUTPUT]
      scripts/11_analysis/paper_2/                  this analysis code

Original camera frames
----------------------
``head_images_exif/`` holds the uncropped 4000 x 3000 camera frames for all 84
measurement images, retaining the blue background board and the EXIF record that
the cropping removes. These are the source for the independent exposure proxy
and the acquisition-settings audit (report Sections 2 and 7). (The copies in
``data/stressed/images/head_images/`` show the same frames re-encoded WITHOUT
EXIF, so they cannot replace these.)

``ORIGINALS_DIR`` resolves to ``head_images/`` when present, and otherwise falls
back to an external ``1. Original`` directory if one can be found; that fallback
covers only 81 frames and omits fish 22 and fish 30.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO / "data"
STRESSED = DATA_ROOT / "stressed"


IMAGES_DIR = STRESSED / "images" / "operculum_images_roboflow"

LABELS_DIR = STRESSED / "labels" / "spots_roboflow"

OPERCULUM_MASKS_DIR = STRESSED / "images" / "operculum_masks"

SPOTS_DATASET_DIR = REPO / "datasets" / "spots_paper2"
SPOTS_TEST_IMAGES = SPOTS_DATASET_DIR / "test" / "images"
SPOTS_TEST_LABELS = SPOTS_DATASET_DIR / "test" / "labels"

WEIGHTS_DIR = REPO / "model_weights"
YOLOV8_SPOTS = WEIGHTS_DIR / "yolov8_spots.pt"
YOLOV11_SPOTS = WEIGHTS_DIR / "yolov11_spots.pt"
YOLOV8_OPERCULUM = WEIGHTS_DIR / "yolov8_operculum.pt"
SAM_CHECKPOINT = WEIGHTS_DIR / "sam_checkpoint.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

BIOMETRICS_CSV = DATA_ROOT / "reference_tabular_data.csv"

HEAD_IMAGES_DIR = STRESSED / "images" / "head_images_exif"


RESULTS_DIR = REPO / "results" / "paper_2"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

REPORT_MD = RESULTS_DIR / "PHOTOMETRIC_REVISION.md"
REPORT_DOCX = RESULTS_DIR / "PHOTOMETRIC_REVISION.docx"


def autodetect_originals(start: Path | None = None) -> Path | None:
    """
    Locate the original whole-head camera frames.

    `head_images_exif/` ships with this project and covers all 84 measurement
    images, so it is preferred. The external `1. Original` directory is kept as
    a fallback for machines that have it; it holds only 81 files and is missing
    the frames for fish 22 and fish 30. Returns None if neither is present, in
    which case the affected analyses skip the board/EXIF stage rather than fail.
    """
    if HEAD_IMAGES_DIR.is_dir():
        return HEAD_IMAGES_DIR
    start = (start or REPO).resolve()
    for candidate_root in [start, *start.parents]:
        candidate = candidate_root / "1. Original"
        if candidate.is_dir():
            return candidate
    return None


ORIGINALS_DIR = autodetect_originals()


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
