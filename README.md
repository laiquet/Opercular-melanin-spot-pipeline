# Opercular melanin spot pipeline (all_data_scripts)

Everything for the operculum / spot pipeline and the paper 2 and paper 3:
code in `scripts/`, inputs and intermediate data in `data/`, training sets in `datasets/`,
checkpoints in `model_weights/` and analysis outputs in `results/`.

```
all_data/
├── scripts/                              pipeline, in run order
│   ├── 01_model_training/                YOLOv8 operculum + spots, SAM-2.1 fine-tuning
│   ├── 02_model_inference/               YOLOv8 operculum + spots, SAM-2 spot segmentation
│   ├── 03_extracting_operculum_rois/
│   ├── 04_image_registration/
│   ├── 05_photometric_normalization/     eye/tag notebook + two-point (linear-light) script
│   ├── 06_spots_matching/
│   ├── 07_sam_keypoints/
│   ├── 08_annotate_spots/
│   ├── 09_annotate_matched_spots/
│   ├── 10_graph_formation/
│   │   (feature quantification: external tool, see "Running the pipeline")
│   ├── 11_analysis/
│   │   ├── paper_2/                      photometric / spatial / biometric analysis (run_pipeline.py)
│   │   └── paper_3/                      two-point fading analysis
│   └── third_party/sam2/                 the single SAM-2 clone (inference + fine-tuning)
├── data/
│   ├── fish_ids.csv                      fish -> day-1 / day-2 image ids (stressed + misc)
│   ├── reference_tabular_data.csv        biometrics + welfare scores, 21 stressed fish
│   └── control/ | stressed/ | misc/      same structure for every group (see below)
├── datasets/                             training sets
│   ├── operculum/                        YOLO, Roboflow
│   ├── spots/                            YOLO, Roboflow
│   ├── sam2_spots/                       SAM-2 format (jpg + json) of the spots set
│   └── spots_paper2/                     YOLO, Roboflow atlantic_salmon_spots v16 (paper 2 validation)
├── model_weights/                        yolov8_operculum, yolov8_spots, yolov11_spots, yolo26_spots, sam_checkpoint
└── results/
    ├── paper_2/                          data/, figures/, legacy_fish_spot_pipeline/
    └── paper_3/                          tables, _meta.json, figs/
```

## Fish groups

| group | fish | images | file name pattern |
|---|---|---|---|
| `control` | 1-6, 8, 9 | 16 | `fish_1_C_L_h` |
| `stressed` | 10-30 | 84 | `fish_10_S_day1_id-51_L_h` |
| `misc` | 31-76 | 184 | `fish_31_S_day1_id-11_L_h` |

Each group folder `data/<group>/` has the same layout. Folders only exist where that step has been run for that group.

```
images/
  head_images/                        whole-head camera frames (input)
  operculum_images/                   1080x1080 opercular ROIs (step 03)
  registered/, registered_plots/      day-1 aligned to day-2 (step 04; stressed, misc)
  normalized_operculums/              eye/tag normalised ROIs (step 05 notebook)
  grayscale_operculum/                un-normalised grayscale ROIs (step 05 notebook)
  normalized_operculums_two_point/    linear-light two-point normalised ROIs + reference_values.csv (step 05 script)
  sam_data_plots/                     SAM-2 masks, polygons, boxes, plots (step 02, SAM)
  spot_annotations*/, operculum_annotations/, spots_annotations_model/   overlays
  matched/                            numbered matched spots (step 09)
  directed_graphs/                    k-NN spot graphs (step 10)
  stressed only, paper 2 inputs:
    head_images_exif/                 original whole-head frames with EXIF (board exposure + EXIF audit)
    operculum_images_roboflow/        the 84 measurement crops (Roboflow-named)
    operculum_masks/                  binary operculum masks for those crops
labels/
  operculum/  spots/  tags_eyes/      YOLO polygon labels (tags_eyes: class 0 = eye, else tag)
  spots_roboflow/                     (stressed only) spot labels matching operculum_images_roboflow/
keypoints_bboxes/                     keypoints.pkl, bboxes.pkl: SAM prompts (step 07)
quantification/                       global_data.csv, spotwise_data.csv (eye/tag normalised; feature quantification)
quantification_two_point/             global_data.csv, spotwise_data.csv (two-point normalised; control, stressed; feature quantification)
```

## Running the pipeline

The notebooks find the project root on their own, so each one runs from its own folder. Each pipeline notebook (03-10, plus the SAM-2 inference notebook) begins with:

```python
GROUP = "stressed"   # fish group to process: "control", "stressed" or "misc"
```

Change `GROUP` and re-run to process another group.

| step | what it reads -> writes (inside `data/<GROUP>/`) |
|---|---|
| 01 training | `datasets/*` -> `runs/` next to the notebook; the SAM fine-tune copies its result to `model_weights/` |
| 02 inference | `model_weights/*` + `images/head_images` or `images/operculum_images` -> YOLO `runs/`; SAM -> `images/sam_data_plots/` |
| 03 ROIs | `images/head_images` + `labels/operculum` -> `images/operculum_images_extracted/` |
| 04 registration | `fish_ids.csv` + `images/operculum_images` -> `images/registered/`, `images/registered_plots/` |
| 05 normalisation | notebook: `images/head_images` + `labels/tags_eyes` + `images/operculum_images` -> `images/normalized_operculums/`, `images/grayscale_operculum/` |
| | script: `python scripts/05_photometric_normalization/photometric_normalization_two_point.py --group stressed` -> `images/normalized_operculums_two_point/` |
| 06 matching | `images/normalized_operculums` + `labels/spots` -> **rewrites the day-2 files in `labels/spots/` in place** |
| 07 SAM keypoints | `labels/spots` -> `keypoints_bboxes/` |
| 08 annotate | `images/operculum_images` + `labels/spots` -> `images/spot_annotations/` |
| 09 annotate matched | `images/normalized_operculums` + `labels/spots` -> `images/matched/` |
| 10 graphs | `images/normalized_operculums` + `labels/spots` (+ optional `spot_distances.csv`) -> `images/directed_graphs/` when the distance table exists, otherwise `images/directed_graphs_from_labels/` |
| feature quantification | [multimodal-labeling-tool](https://github.com/laiquet/multimodal-labeling-tool) (separate repository): normalised operculum images + spot labels -> image-level `global_data.csv` and spot-level `spotwise_data.csv` in `quantification/` or `quantification_two_point/`. Run it after steps 01-10 and before the analysis (step 11). |
| 11 paper 2 | `python scripts/11_analysis/paper_2/run_pipeline.py`: all paths come from `paper_2/_paths.py` -> `results/paper_2/` |
| 11 paper 3 | `python scripts/11_analysis/paper_3/opercular_analysis_complete_updated.py` -> `results/paper_3/` |
